from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import re

import h5py
import numpy as np

from octavian.halo_reader.ahf import read_ahf_halos
from octavian.halo_reader.hbt import gather_subsnap_files, read_subhalos


_PID_DATASET = 'ParticleIDs'
_STAR_PARTICLE_GROUP = 'PartType4'
_SOURCE_COLUMNS = {
    'ahf': 'AHF_haloID',
    'hbt': 'HBT_trackID',
}


@dataclass(frozen=True)
class _PathSpec:
    path: Path
    snap_index: int | None = None


@dataclass
class _GroupSnapshot:
    path: Path
    group_key: str
    ids: np.ndarray
    source_ids: np.ndarray
    masses: np.ndarray
    redshift: float | None


@dataclass
class _CSRIndex:
    keys: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    values: np.ndarray

    def values_for(self, key: int, limit: int | None = None) -> np.ndarray:
        if key < 0 or len(self.keys) == 0:
            return np.empty(0, dtype=np.int64)
        pos = np.searchsorted(self.keys, key)
        if pos >= len(self.keys) or self.keys[pos] != key:
            return np.empty(0, dtype=np.int64)
        start = int(self.offsets[pos])
        length = int(self.lengths[pos])
        if limit is not None:
            length = min(length, limit)
        return self.values[start:start + length]

    def rows_for_keys(self, keys) -> np.ndarray:
        if len(self.keys) == 0:
            return np.empty(0, dtype=np.int64)
        rows = []
        seen = set()
        for key in keys:
            values = self.values_for(int(key))
            for value in values:
                value = int(value)
                if value not in seen:
                    rows.append(value)
                    seen.add(value)
        if not rows:
            return np.empty(0, dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)


class _BaseHaloLinks:
    def candidates_for_source_ids(self, source_ids, limit: int | None) -> list[np.ndarray]:
        raise NotImplementedError


class _AHFLinks(_BaseHaloLinks):
    def __init__(self, descendant_spec: _PathSpec, progenitor_spec: _PathSpec, progenitor_halos: _GroupSnapshot):
        descendant_files = _locate_ahf_files(descendant_spec)
        progenitor_files = _locate_ahf_files(progenitor_spec)
        self._primary_desc, self._primary_prog = _parse_ahf_mtree_idx(descendant_files.get('mtree_idx'))
        self._mtree = _parse_ahf_mtree(descendant_files.get('mtree'))
        masses = _read_ahf_masses(progenitor_files.get('halos'))
        if not masses:
            masses = _mass_lookup(progenitor_halos.source_ids, progenitor_halos.masses)
        self._progenitor_masses = masses

    def candidates_for_source_ids(self, source_ids, limit: int | None) -> list[np.ndarray]:
        source_ids = np.asarray(source_ids, dtype=np.int64)
        if limit == 1:
            primary = _lookup_sorted_values(
                source_ids,
                self._primary_desc,
                self._primary_prog,
                fill=-1,
            )
            out = []
            for source_id, primary_id in zip(source_ids, primary):
                candidates = []
                if primary_id >= 0:
                    candidates.append(int(primary_id))
                candidates.extend(
                    int(value)
                    for value in self._mtree.values_for(int(source_id))
                    if int(value) != int(primary_id)
                )
                out.append(np.asarray(candidates, dtype=np.int64))
            return out

        out = []
        for source_id in source_ids:
            values = self._mtree.values_for(int(source_id))
            values = _sort_sources_by_mass(values, self._progenitor_masses)
            if limit is not None:
                values = values[:limit]
            out.append(np.asarray(values, dtype=np.int64))
        return out


class _HBTLinks(_BaseHaloLinks):
    def __init__(self, progenitor_spec: _PathSpec, progenitor_halos: _GroupSnapshot):
        properties = _read_hbt_properties(progenitor_spec)
        self._previous_sources = _unique_sources(progenitor_halos.source_ids)
        self._progenitor_masses = _mass_lookup(progenitor_halos.source_ids, progenitor_halos.masses)
        self._by_descendant = _hbt_progenitors_by_descendant(properties)

    def candidates_for_source_ids(self, source_ids, limit: int | None) -> list[np.ndarray]:
        source_ids = np.asarray(source_ids, dtype=np.int64)
        out = []
        stable = _isin_sorted(source_ids, self._previous_sources)
        if limit == 1:
            for source_id, has_stable in zip(source_ids, stable):
                candidates = []
                if has_stable and source_id >= 0:
                    candidates.append(int(source_id))
                candidates.extend(
                    int(value)
                    for value in self._by_descendant.values_for(int(source_id))
                    if int(value) != int(source_id)
                )
                out.append(np.asarray(candidates, dtype=np.int64))
            return out

        for source_id, has_stable in zip(source_ids, stable):
            candidates = [int(value) for value in self._by_descendant.values_for(int(source_id))]
            if has_stable and source_id >= 0:
                candidates.append(int(source_id))
            values = _sort_sources_by_mass(candidates, self._progenitor_masses)
            if limit is not None:
                values = values[:limit]
            out.append(np.asarray(values, dtype=np.int64))
        return out


def progen(
    octavian_files,
    halo_catalogs,
    *,
    group_type: str = 'all',
    progenitors=1,
    halo_source: str | None = None,
    snap_indices=None,
    snapshot_files=None,
    min_in_common: float = 0.1,
    recompute: bool = True,
    save: bool = True,
):
    """Link adjacent Octavian files through AHF/HBT progenitor information.

    Files must be ordered newest to oldest. The return value is a list of dictionaries,
    one per adjacent pair, keyed by ``progen_halos`` and/or ``progen_galaxies``.
    """
    files = _as_path_list(octavian_files)
    if len(files) < 2:
        raise ValueError('progen requires at least two Octavian files')

    requested_groups = _normalise_group_type(group_type)
    limit = _normalise_progenitors(progenitors)
    source = _normalise_halo_source(halo_source or _infer_halo_source(files[0]))

    catalog_resolver = _PathResolver(halo_catalogs, files, snap_indices, required=True)
    snapshot_resolver = _PathResolver(snapshot_files, files, snap_indices, required=False)

    results = []
    for index in range(len(files) - 1):
        results.append(
            _link_pair(
                current_file=files[index],
                progenitor_file=files[index + 1],
                current_catalog=catalog_resolver.spec(index),
                progenitor_catalog=catalog_resolver.spec(index + 1),
                current_snapshot=snapshot_resolver.spec(index),
                progenitor_snapshot=snapshot_resolver.spec(index + 1),
                halo_source=source,
                group_types=requested_groups,
                limit=limit,
                min_in_common=min_in_common,
                recompute=recompute,
                save=save,
            )
        )
    return results


def check_if_progen_is_present(octavian_file, dataset_name: str) -> bool:
    with h5py.File(octavian_file, 'r') as handle:
        if 'tree_data' not in handle:
            return False
        tree = handle['tree_data']
        return dataset_name in tree or f'{dataset_name}_indices' in tree


def get_progen_redshift(octavian_file, dataset_name: str):
    with h5py.File(octavian_file, 'r') as handle:
        if 'tree_data' not in handle:
            return -1
        return handle['tree_data'].attrs.get(f'z_{dataset_name}', -1)


def read_progens(octavian_file, dataset_name: str):
    with h5py.File(octavian_file, 'r') as handle:
        tree = handle['tree_data']
        if dataset_name in tree:
            return tree[dataset_name][:]
        return _read_csr_result(tree, dataset_name)


def wipe_progen_info(octavian_file, dataset_name: str | None = None) -> None:
    with h5py.File(octavian_file, 'r+') as handle:
        if 'tree_data' not in handle:
            return
        tree = handle['tree_data']
        for name in list(tree.keys()):
            if dataset_name is None:
                if 'progen' in name or 'descend' in name:
                    del tree[name]
            elif name == dataset_name or name.startswith(f'{dataset_name}_'):
                del tree[name]


def _link_pair(
    *,
    current_file: Path,
    progenitor_file: Path,
    current_catalog: _PathSpec,
    progenitor_catalog: _PathSpec,
    current_snapshot: _PathSpec | None,
    progenitor_snapshot: _PathSpec | None,
    halo_source: str,
    group_types: tuple[str, ...],
    limit: int | None,
    min_in_common: float,
    recompute: bool,
    save: bool,
) -> dict[str, object]:
    source_column = _SOURCE_COLUMNS[halo_source]
    current_halos = _read_group_snapshot(current_file, 'halo_data', source_column, required=True)
    progenitor_halos = _read_group_snapshot(progenitor_file, 'halo_data', source_column, required=True)
    halo_links = _build_halo_links(halo_source, current_catalog, progenitor_catalog, progenitor_halos)

    results: dict[str, object] = {}
    if 'halos' in group_types:
        dataset_name = 'progen_halos'
        if not recompute and check_if_progen_is_present(current_file, dataset_name):
            results[dataset_name] = read_progens(current_file, dataset_name)
        else:
            halo_candidates = halo_links.candidates_for_source_ids(current_halos.source_ids, limit)
            halo_result = _map_source_candidates_to_octavian_ids(halo_candidates, progenitor_halos, limit)
            if save:
                _write_progen_result(current_file, dataset_name, halo_result, progenitor_halos.redshift)
            results[dataset_name] = halo_result

    if 'galaxies' in group_types and _has_group(current_file, 'galaxy_data') and _has_group(progenitor_file, 'galaxy_data'):
        dataset_name = 'progen_galaxies'
        if not recompute and check_if_progen_is_present(current_file, dataset_name):
            results[dataset_name] = read_progens(current_file, dataset_name)
        else:
            current_galaxies = _read_group_snapshot(current_file, 'galaxy_data', source_column, required=True)
            progenitor_galaxies = _read_group_snapshot(progenitor_file, 'galaxy_data', source_column, required=True)
            galaxy_result = _link_galaxies(
                current_galaxies,
                progenitor_galaxies,
                halo_links,
                limit,
                current_snapshot,
                progenitor_snapshot,
                min_in_common,
            )
            if save:
                _write_progen_result(current_file, dataset_name, galaxy_result, progenitor_galaxies.redshift)
            results[dataset_name] = galaxy_result

    return results


def _build_halo_links(
    halo_source: str,
    current_catalog: _PathSpec,
    progenitor_catalog: _PathSpec,
    progenitor_halos: _GroupSnapshot,
) -> _BaseHaloLinks:
    if halo_source == 'ahf':
        return _AHFLinks(current_catalog, progenitor_catalog, progenitor_halos)
    if halo_source == 'hbt':
        return _HBTLinks(progenitor_catalog, progenitor_halos)
    raise ValueError(f'Unsupported halo_source: {halo_source}')


def _link_galaxies(
    current_galaxies: _GroupSnapshot,
    progenitor_galaxies: _GroupSnapshot,
    halo_links: _BaseHaloLinks,
    limit: int | None,
    current_snapshot: _PathSpec | None,
    progenitor_snapshot: _PathSpec | None,
    min_in_common: float,
):
    current_stars = _read_galaxy_star_particle_ids(current_galaxies.path, current_snapshot)
    progenitor_stars = _read_galaxy_star_particle_ids(progenitor_galaxies.path, progenitor_snapshot)
    if current_stars is None or progenitor_stars is None:
        return _empty_result(len(current_galaxies.ids), limit)

    host_limit = None if limit is None else limit
    host_candidates = halo_links.candidates_for_source_ids(current_galaxies.source_ids, host_limit)
    progenitor_rows_by_source = _rows_by_source(progenitor_galaxies.source_ids)

    if limit is None:
        out = np.empty(len(current_galaxies.ids), dtype=object)
    elif limit == 1:
        out = np.full(len(current_galaxies.ids), -1, dtype=np.int64)
    else:
        out = np.full((len(current_galaxies.ids), limit), -1, dtype=np.int64)

    for current_row, candidate_hosts in enumerate(host_candidates):
        current_particle_ids = current_stars[current_row]
        if len(current_particle_ids) == 0 or len(candidate_hosts) == 0:
            if limit is None:
                out[current_row] = []
            continue

        candidate_rows = progenitor_rows_by_source.rows_for_keys(candidate_hosts)
        ranked = []
        for candidate_row in candidate_rows:
            candidate_particle_ids = progenitor_stars[int(candidate_row)]
            if len(candidate_particle_ids) == 0:
                continue
            shared = np.intersect1d(
                current_particle_ids,
                candidate_particle_ids,
                assume_unique=True,
            ).size
            if shared == 0:
                continue
            match_fraction = shared / len(candidate_particle_ids)
            if match_fraction < min_in_common:
                continue
            ranked.append((
                shared,
                match_fraction,
                float(progenitor_galaxies.masses[int(candidate_row)]),
                int(progenitor_galaxies.ids[int(candidate_row)]),
            ))

        ranked.sort(reverse=True)
        matched_ids = [entry[3] for entry in ranked]
        if limit is None:
            out[current_row] = matched_ids
        elif limit == 1:
            if matched_ids:
                out[current_row] = matched_ids[0]
        else:
            n = min(limit, len(matched_ids))
            if n:
                out[current_row, :n] = matched_ids[:n]

    return out


def _read_galaxy_star_particle_ids(octavian_file: Path, snapshot_spec: _PathSpec | None):
    with h5py.File(octavian_file, 'r') as handle:
        if 'galaxy_data' not in handle:
            return None
        galaxy_group = handle['galaxy_data']
        for dataset_name in ('slist_pids', 'slist_pid', 'slist_particle_ids'):
            sequences = _read_csr_sequences(galaxy_group, dataset_name)
            if sequences is not None:
                return [_sorted_unique(values) for values in sequences]

        star_rows = _read_csr_sequences(galaxy_group, 'slist')
        if star_rows is None:
            return None

    if snapshot_spec is None:
        return None

    particle_ids = _read_star_particle_ids(snapshot_spec.path)
    sequences = []
    for rows in star_rows:
        rows = np.asarray(rows, dtype=np.int64)
        valid = (rows >= 0) & (rows < len(particle_ids))
        sequences.append(_sorted_unique(particle_ids[rows[valid]]))
    return sequences


def _read_star_particle_ids(snapshot_file: Path) -> np.ndarray:
    with h5py.File(snapshot_file, 'r') as handle:
        if _STAR_PARTICLE_GROUP not in handle or _PID_DATASET not in handle[_STAR_PARTICLE_GROUP]:
            raise KeyError(f'{snapshot_file} does not contain {_STAR_PARTICLE_GROUP}/{_PID_DATASET}')
        return handle[_STAR_PARTICLE_GROUP][_PID_DATASET][:].astype(np.int64, copy=False)


def _read_group_snapshot(path: Path, group_key: str, source_column: str, *, required: bool) -> _GroupSnapshot:
    with h5py.File(path, 'r') as handle:
        if group_key not in handle:
            if required:
                raise KeyError(f'{path} does not contain {group_key}')
            return _empty_group_snapshot(path, group_key)
        group = handle[group_key]
        if source_column not in group:
            raise KeyError(f'{path}:{group_key} does not contain {source_column}')
        source_ids = group[source_column][:].astype(np.int64, copy=False)
        ids = _read_group_ids(group, group_key, len(source_ids))
        masses = _read_group_masses(group, group_key, len(source_ids))
        redshift = _read_redshift(handle)
    return _GroupSnapshot(path, group_key, ids, source_ids, masses, redshift)


def _empty_group_snapshot(path: Path, group_key: str) -> _GroupSnapshot:
    return _GroupSnapshot(
        path=path,
        group_key=group_key,
        ids=np.empty(0, dtype=np.int64),
        source_ids=np.empty(0, dtype=np.int64),
        masses=np.empty(0, dtype=float),
        redshift=None,
    )


def _read_group_ids(group, group_key: str, length: int) -> np.ndarray:
    candidates = ('HaloID', 'groupID') if group_key == 'halo_data' else ('GalID', 'groupID')
    for name in candidates:
        if name in group:
            return group[name][:].astype(np.int64, copy=False)
    return np.arange(length, dtype=np.int64)


def _read_group_masses(group, group_key: str, length: int) -> np.ndarray:
    if group_key == 'galaxy_data':
        candidates = ('dicts/masses.stellar', 'dicts/masses.total', 'mass')
    else:
        candidates = ('dicts/masses.total', 'mass')
    for name in candidates:
        if name in group:
            return group[name][:].astype(float, copy=False)
    return np.ones(length, dtype=float)


def _read_redshift(handle) -> float | None:
    for attrs in (handle.attrs, handle['Header'].attrs if 'Header' in handle else None):
        if attrs is None:
            continue
        for key in ('Redshift', 'redshift', 'z'):
            if key in attrs:
                return float(np.asarray(attrs[key]))
    return None


def _read_csr_sequences(group, dataset_name: str):
    indices_name = f'{dataset_name}_indices'
    lengths_name = f'{dataset_name}_lengths'
    if indices_name not in group or lengths_name not in group:
        return None
    indices = group[indices_name][:]
    lengths = group[lengths_name][:].astype(np.int64, copy=False)
    offsets_name = f'{dataset_name}_offsets'
    if offsets_name in group:
        offsets = group[offsets_name][:].astype(np.int64, copy=False)
    else:
        offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    return [indices[offset:offset + length].astype(np.int64, copy=False) for offset, length in zip(offsets, lengths)]


def _write_progen_result(path: Path, dataset_name: str, data, redshift: float | None) -> None:
    with h5py.File(path, 'r+') as handle:
        tree = handle.require_group('tree_data')
        _delete_progen_result(tree, dataset_name)
        if isinstance(data, np.ndarray) and data.dtype != object:
            tree.create_dataset(dataset_name, data=data)
        else:
            _write_csr_result(tree, dataset_name, data)
        if redshift is not None:
            tree.attrs[f'z_{dataset_name}'] = redshift


def _delete_progen_result(tree, dataset_name: str) -> None:
    for name in (dataset_name, f'{dataset_name}_indices', f'{dataset_name}_offsets', f'{dataset_name}_lengths'):
        if name in tree:
            del tree[name]


def _write_csr_result(tree, dataset_name: str, data) -> None:
    sequences = [np.asarray(values, dtype=np.int64) for values in data]
    lengths = np.asarray([len(values) for values in sequences], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    if int(lengths.sum()) == 0:
        indices = np.empty(0, dtype=np.int64)
    else:
        indices = np.concatenate(sequences).astype(np.int64, copy=False)
    tree.create_dataset(f'{dataset_name}_indices', data=indices)
    tree.create_dataset(f'{dataset_name}_offsets', data=offsets)
    tree.create_dataset(f'{dataset_name}_lengths', data=lengths)


def _read_csr_result(tree, dataset_name: str):
    indices = tree[f'{dataset_name}_indices'][:]
    lengths = tree[f'{dataset_name}_lengths'][:].astype(np.int64, copy=False)
    if f'{dataset_name}_offsets' in tree:
        offsets = tree[f'{dataset_name}_offsets'][:].astype(np.int64, copy=False)
    else:
        offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    out = np.empty(len(lengths), dtype=object)
    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        out[index] = indices[offset:offset + length].astype(np.int64, copy=False)
    return out


def _map_source_candidates_to_octavian_ids(
    candidates: list[np.ndarray],
    progenitor_snapshot: _GroupSnapshot,
    limit: int | None,
):
    lookup_sources, lookup_ids = _source_id_lookup(
        progenitor_snapshot.source_ids,
        progenitor_snapshot.ids,
        progenitor_snapshot.masses,
    )

    if limit is None:
        out = np.empty(len(candidates), dtype=object)
        for index, values in enumerate(candidates):
            out[index] = _unique_valid(_lookup_many(values, lookup_sources, lookup_ids))
        return out

    if limit == 1:
        out = np.full(len(candidates), -1, dtype=np.int64)
        for index, values in enumerate(candidates):
            mapped = _unique_valid(_lookup_many(values, lookup_sources, lookup_ids))
            if len(mapped):
                out[index] = mapped[0]
        return out

    out = np.full((len(candidates), limit), -1, dtype=np.int64)
    for index, values in enumerate(candidates):
        mapped = _unique_valid(_lookup_many(values, lookup_sources, lookup_ids))
        n = min(limit, len(mapped))
        if n:
            out[index, :n] = mapped[:n]
    return out


def _source_id_lookup(source_ids, octavian_ids, masses):
    source_ids = np.asarray(source_ids, dtype=np.int64)
    octavian_ids = np.asarray(octavian_ids, dtype=np.int64)
    masses = np.asarray(masses, dtype=float)
    valid = source_ids >= 0
    if not np.any(valid):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.lexsort((-masses[valid], source_ids[valid]))
    sorted_sources = source_ids[valid][order]
    sorted_ids = octavian_ids[valid][order]
    keep = np.concatenate(([True], sorted_sources[1:] != sorted_sources[:-1]))
    return sorted_sources[keep], sorted_ids[keep]


def _lookup_many(values, sorted_keys, sorted_values, fill=-1) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return _lookup_sorted_values(values, sorted_keys, sorted_values, fill=fill)


def _lookup_sorted_values(keys, sorted_keys, sorted_values, *, fill=-1) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    out = np.full(keys.shape, fill, dtype=np.int64)
    if len(keys) == 0 or len(sorted_keys) == 0:
        return out
    valid = keys >= 0
    if not np.any(valid):
        return out
    positions = np.searchsorted(sorted_keys, keys[valid])
    in_bounds = positions < len(sorted_keys)
    matched = np.zeros(len(positions), dtype=bool)
    matched[in_bounds] = sorted_keys[positions[in_bounds]] == keys[valid][in_bounds]
    valid_rows = np.flatnonzero(valid)
    out[valid_rows[matched]] = sorted_values[positions[matched]]
    return out


def _unique_valid(values) -> list[int]:
    out = []
    seen = set()
    for value in values:
        value = int(value)
        if value < 0 or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _unique_sources(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    values = values[values >= 0]
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(values)


def _isin_sorted(values, sorted_values) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros(values.shape, dtype=bool)
    if len(values) == 0 or len(sorted_values) == 0:
        return out
    valid = values >= 0
    positions = np.searchsorted(sorted_values, values[valid])
    in_bounds = positions < len(sorted_values)
    matched = np.zeros(len(positions), dtype=bool)
    matched[in_bounds] = sorted_values[positions[in_bounds]] == values[valid][in_bounds]
    out[np.flatnonzero(valid)[matched]] = True
    return out


def _empty_result(length: int, limit: int | None):
    if limit is None:
        out = np.empty(length, dtype=object)
        for index in range(length):
            out[index] = []
        return out
    if limit == 1:
        return np.full(length, -1, dtype=np.int64)
    return np.full((length, limit), -1, dtype=np.int64)


def _rows_by_source(source_ids) -> _CSRIndex:
    source_ids = np.asarray(source_ids, dtype=np.int64)
    rows = np.arange(len(source_ids), dtype=np.int64)
    valid = source_ids >= 0
    if not np.any(valid):
        return _empty_csr_index()
    return _build_grouped_index(source_ids[valid], rows[valid])


def _hbt_progenitors_by_descendant(properties) -> _CSRIndex:
    if 'TrackId' not in properties or 'DescendantTrackId' not in properties:
        return _empty_csr_index()
    track_ids = properties['TrackId'].to_numpy(dtype=np.int64)
    descendant_ids = properties['DescendantTrackId'].to_numpy(dtype=np.int64)
    if 'Mbound' in properties:
        weights = properties['Mbound'].to_numpy(dtype=float)
    elif 'Nbound' in properties:
        weights = properties['Nbound'].to_numpy(dtype=float)
    else:
        weights = np.ones(len(track_ids), dtype=float)
    valid = (track_ids >= 0) & (descendant_ids >= 0)
    if not np.any(valid):
        return _empty_csr_index()
    return _build_grouped_index(descendant_ids[valid], track_ids[valid], weights[valid])


def _build_grouped_index(keys, values, weights=None) -> _CSRIndex:
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    if len(keys) == 0:
        return _empty_csr_index()
    if weights is None:
        order = np.argsort(keys, kind='mergesort')
    else:
        order = np.lexsort((-np.asarray(weights, dtype=float), keys))
    sorted_keys = keys[order]
    sorted_values = values[order]
    starts = np.concatenate(([0], np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1))
    group_keys = sorted_keys[starts]
    lengths = np.diff(np.concatenate((starts, [len(sorted_keys)]))).astype(np.int64)
    offsets = starts.astype(np.int64)
    return _CSRIndex(group_keys, offsets, lengths, sorted_values)


def _empty_csr_index() -> _CSRIndex:
    return _CSRIndex(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )


def _parse_ahf_mtree_idx(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None or not path.is_file():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    descendants = []
    progenitors = []
    with open(path, 'r') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            descendants.append(int(parts[0]))
            progenitors.append(int(parts[1]))
    if not descendants:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    descendants = np.asarray(descendants, dtype=np.int64)
    progenitors = np.asarray(progenitors, dtype=np.int64)
    order = np.argsort(descendants)
    return descendants[order], progenitors[order]


def _parse_ahf_mtree(path: Path | None) -> _CSRIndex:
    if path is None or not path.is_file():
        return _empty_csr_index()
    keys = []
    values = []
    with open(path, 'r') as handle:
        handle.readline()
        while True:
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            descendant = int(parts[0])
            n_progenitors = int(parts[1])
            for _ in range(n_progenitors):
                entry = handle.readline()
                if not entry:
                    break
                entry_parts = entry.split()
                if entry_parts:
                    keys.append(descendant)
                    values.append(int(entry_parts[0]))
    if not keys:
        return _empty_csr_index()
    return _build_grouped_index(np.asarray(keys, dtype=np.int64), np.asarray(values, dtype=np.int64))


def _sort_sources_by_mass(source_ids, masses: Mapping[int, float]) -> list[int]:
    source_ids = [int(value) for value in source_ids if int(value) >= 0]
    source_ids = list(dict.fromkeys(source_ids))
    return sorted(source_ids, key=lambda value: masses.get(value, -np.inf), reverse=True)


def _mass_lookup(source_ids, masses) -> dict[int, float]:
    out = {}
    for source_id, mass in zip(source_ids, masses):
        source_id = int(source_id)
        if source_id < 0:
            continue
        out[source_id] = max(float(mass), out.get(source_id, -np.inf))
    return out


def _read_ahf_masses(path: Path | None) -> dict[int, float]:
    if path is None or not path.is_file():
        return {}
    halos = read_ahf_halos(path)
    if 'ID' not in halos or 'Mhalo' not in halos:
        return {}
    return {
        int(source_id): float(mass)
        for source_id, mass in zip(halos['ID'].to_numpy(), halos['Mhalo'].to_numpy())
    }


def _read_hbt_properties(spec: _PathSpec):
    files = gather_subsnap_files(spec.path, spec.snap_index)
    return read_subhalos(files)


def _locate_ahf_files(spec: _PathSpec) -> dict[str, Path | None]:
    path = spec.path
    if path.is_dir():
        return {
            'mtree_idx': _glob_ahf_file(path, spec.snap_index, 'mtree_idx'),
            'mtree': _glob_ahf_file(path, spec.snap_index, 'mtree'),
            'halos': _glob_ahf_file(path, spec.snap_index, 'halos'),
        }
    return {
        'mtree_idx': _replace_ahf_kind(path, 'mtree_idx'),
        'mtree': _replace_ahf_kind(path, 'mtree'),
        'halos': _replace_ahf_kind(path, 'halos'),
    }


def _glob_ahf_file(root: Path, snap_index: int | None, kind: str) -> Path | None:
    suffixes = (f'AHF_{kind}', f'AHF_{kind}.gz')
    patterns = []
    if snap_index is None:
        patterns = [f'*.{suffix}' for suffix in suffixes] + [f'*{suffix}' for suffix in suffixes]
    else:
        token = f'{snap_index:03d}'
        for suffix in suffixes:
            patterns.extend((
                f'*snap_{token}*.{suffix}',
                f'*snap_{token}*{suffix}',
                f'*_{token}*.{suffix}',
                f'*_{token}*{suffix}',
            ))
    matches = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    matches = sorted(set(matches))
    if len(matches) > 1:
        raise FileNotFoundError(f'Multiple AHF {kind} files matched in {root} for snapshot {snap_index}')
    return matches[0] if matches else None


def _replace_ahf_kind(path: Path, kind: str) -> Path:
    suffix = '.gz' if path.name.endswith('.gz') else ''
    stem = path.name[:-3] if suffix else path.name
    endings = ('AHF_particles', 'AHF_halos', 'AHF_mtree_idx', 'AHF_mtree')
    for ending in endings:
        if stem.endswith(ending):
            return path.with_name(stem[:-len(ending)] + f'AHF_{kind}' + suffix)
    return path.with_name(f'{stem}.AHF_{kind}{suffix}')


class _PathResolver:
    def __init__(self, paths, files: list[Path], snap_indices, *, required: bool):
        self.paths = paths
        self.files = files
        self.required = required
        if snap_indices is None:
            self.snap_indices = [_infer_snap_index(path) for path in files]
        else:
            if isinstance(snap_indices, int):
                snap_indices = [snap_indices]
            self.snap_indices = [None if value is None else int(value) for value in snap_indices]
            if len(self.snap_indices) != len(files):
                raise ValueError('snap_indices must have the same length as octavian_files')

    def spec(self, index: int) -> _PathSpec | None:
        if self.paths is None:
            if self.required:
                raise ValueError('A path argument is required')
            return None
        snap_index = self.snap_indices[index]
        paths = self.paths
        if isinstance(paths, Mapping):
            file_key = str(self.files[index])
            candidates = (file_key, self.files[index].name, snap_index, str(snap_index))
            for key in candidates:
                if key in paths:
                    return _PathSpec(Path(paths[key]), snap_index)
            if self.required:
                raise KeyError(f'No path entry for {self.files[index]}')
            return None
        if isinstance(paths, (str, Path)):
            return _PathSpec(Path(paths), snap_index)
        if isinstance(paths, Sequence):
            if len(paths) != len(self.files):
                raise ValueError('path sequences must have the same length as octavian_files')
            return _PathSpec(Path(paths[index]), snap_index)
        raise TypeError('paths must be a path, mapping, sequence, or None')


def _as_path_list(values) -> list[Path]:
    if isinstance(values, (str, Path)):
        return [Path(values)]
    return [Path(value) for value in values]


def _normalise_group_type(group_type: str) -> tuple[str, ...]:
    group_type = str(group_type).lower()
    if group_type in ('halo', 'halos', 'subhalo', 'subhalos'):
        return ('halos',)
    if group_type in ('galaxy', 'galaxies'):
        return ('galaxies',)
    if group_type == 'all':
        return ('halos', 'galaxies')
    raise ValueError("group_type must be 'halos', 'galaxies', or 'all'")


def _normalise_progenitors(progenitors) -> int | None:
    if isinstance(progenitors, str):
        if progenitors.lower() != 'all':
            raise ValueError("progenitors must be a positive integer or 'all'")
        return None
    value = int(progenitors)
    if value < 1:
        raise ValueError('progenitors must be a positive integer')
    return value


def _normalise_halo_source(halo_source: str) -> str:
    halo_source = str(halo_source).lower()
    if halo_source not in _SOURCE_COLUMNS:
        raise ValueError("halo_source must be 'ahf' or 'hbt'")
    return halo_source


def _infer_halo_source(path: Path) -> str:
    with h5py.File(path, 'r') as handle:
        for group_key in ('halo_data', 'galaxy_data'):
            if group_key not in handle:
                continue
            group = handle[group_key]
            if 'AHF_haloID' in group:
                return 'ahf'
            if 'HBT_trackID' in group:
                return 'hbt'
    raise ValueError(f'Unable to infer halo_source from {path}')


def _infer_snap_index(path: Path) -> int | None:
    for pattern in (
        r'snap[_-]?[A-Za-z0-9]*[_-](\d{1,3})(?:\D|$)',
        r'(?:^|[_-])s(\d{3})(?:\D|$)',
        r'(?:^|[_-])(\d{3})(?:\D|$)',
    ):
        match = re.search(pattern, path.name)
        if match is not None:
            return int(match.group(1))
    return None


def _has_group(path: Path, group_key: str) -> bool:
    with h5py.File(path, 'r') as handle:
        return group_key in handle


def _sorted_unique(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return values
    return np.unique(values)

