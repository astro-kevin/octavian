"""

HBT+ halo finder integration with Octavian; assumes we are analysing a HBT+ output.

Source paper: https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract
Github: https://github.com/Kambrian/HBTplus/wiki/Outputs
Code is based on the architecture outlined therein.

"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_manager import DataManager

import h5py
import numpy as np
import pandas as pd

from octavian.halo_reader.halo_utils import HaloMembership, HaloReader, HaloTree, PTYPE_ENCODE

_SUBSNAP_RE = re.compile(r'^SubSnap_(?P<snap>\d+)(?:\.(?P<file>\d+))?\.hdf5$')
_HBT_SCALAR_FIELDS = (
    'TrackId',
    'Nbound',
    'Mbound',
    'HostHaloId',
    'Rank',
    'Depth',
    'BoundM200Crit',
    'MostBoundParticleId',
    'SinkTrackId',
    'DescendantTrackId',
    'NestedParentTrackId',
)


def _normalise_snap_index(snap_index) -> int | None:
    if snap_index is None or snap_index == '':
        return None
    return int(snap_index)


def _snap_tokens(snap_index: int | None) -> tuple[str, ...]:
    if snap_index is None:
        return ()
    tokens = [str(snap_index), f'{snap_index:03d}']
    return tuple(dict.fromkeys(tokens))


def _infer_snap_index_from_path(path: Path) -> int | None:
    match = _SUBSNAP_RE.match(path.name)
    if match is not None:
        return int(match.group('snap'))
    if path.name.isdigit():
        return int(path.name)
    return None


def _subsnap_sort_key(path: Path) -> tuple[int, int, str]:
    match = _SUBSNAP_RE.match(path.name)
    if match is None:
        return (0, 0, path.name)
    file_index = match.group('file')
    return (int(match.group('snap')), -1 if file_index is None else int(file_index), path.name)


def _candidate_dirs(subhalo_path: Path, snap_index: int | None) -> list[Path]:
    if subhalo_path.is_file():
        return [subhalo_path.parent]

    dirs = [subhalo_path]
    for token in _snap_tokens(snap_index):
        child = subhalo_path / token
        if child not in dirs:
            dirs.append(child)
    return dirs


def gather_subsnap_files(subhalo_path, snap_index=None) -> list[Path]:
    """
    Locate HBT+ SubSnap files for a snapshot.

    Supports a direct file path or a snapshot directory (for example ``050``).
    Passing a simulation HBT root is only supported when ``snap_index`` is given. HBT can write
    either one ``SubSnap_050.hdf5`` file or split files such as
    ``SubSnap_050.0.hdf5`` ... ``SubSnap_050.31.hdf5``.
    """
    subhalo_path = Path(subhalo_path)
    snap_index = _normalise_snap_index(snap_index)

    if subhalo_path.is_file():
        inferred = _infer_snap_index_from_path(subhalo_path)
        if snap_index is not None and inferred is not None and snap_index != inferred:
            raise FileNotFoundError(
                f'SubSnap file {subhalo_path} is for snapshot {inferred}, not {snap_index}'
            )
        return [subhalo_path]

    if snap_index is None:
        snap_index = _infer_snap_index_from_path(subhalo_path)

    matches: list[Path] = []
    searched: list[str] = []
    for directory in _candidate_dirs(subhalo_path, snap_index):
        if not directory.exists():
            searched.append(str(directory))
            continue

        if snap_index is None:
            candidates = list(directory.glob('SubSnap_*.hdf5'))
        else:
            candidates = []
            for token in _snap_tokens(snap_index):
                candidates.extend(directory.glob(f'SubSnap_{token}.hdf5'))
                candidates.extend(directory.glob(f'SubSnap_{token}.*.hdf5'))

        searched.append(str(directory))
        matches.extend(path for path in candidates if path.is_file())

    matches = sorted(set(matches), key=_subsnap_sort_key)
    if not matches:
        snap_text = 'any snapshot' if snap_index is None else f'snapshot {snap_index}'
        raise FileNotFoundError(
            f'No HBT SubSnap files found for {snap_text} under {subhalo_path}. '
            f'Searched: {", ".join(searched)}'
        )

    return matches


def gather_subsnap_file(subhalo_path, snap_index=None) -> Path:
    """
    Backwards-compatible helper for callers that expect a single HBT file.
    """
    files = gather_subsnap_files(subhalo_path, snap_index)
    if len(files) != 1:
        raise FileNotFoundError(
            f'Expected one SubSnap file but found {len(files)}. '
            'Use gather_subsnap_files/read_subhalos/read_particles for split HBT output.'
        )
    return files[0]


def _as_path_list(filepaths) -> list[Path]:
    if isinstance(filepaths, (str, Path)):
        return [Path(filepaths)]
    if isinstance(filepaths, Iterable):
        return [Path(path) for path in filepaths]
    raise TypeError(f'Expected path or iterable of paths, got {type(filepaths)!r}')


def read_subhalos(filepaths) -> pd.DataFrame:
    """
    Read scalar HBT+ subhalo properties from one or more HDF5 files.
    """
    frames = []
    for filepath in _as_path_list(filepaths):
        with h5py.File(filepath, 'r') as f:
            dataset = f['Subhalos']
            names = [name for name in _HBT_SCALAR_FIELDS if name in dataset.dtype.names]
            subhalos = dataset.fields(names)[:]

        columns = {}
        for name in names:
            values = subhalos[name]
            if values.ndim == 1:
                columns[name] = values
        frames.append(pd.DataFrame(columns))

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True)


def read_particles(filepaths) -> tuple[np.ndarray, np.ndarray]:
    """
    Read particle memberships from SubSnap files.
    HBT+ stores variable-length particle ID arrays per subhalo.
    Returns flat aligned arrays (halo_indices, particle_ids).
    Key thing: HBT+ does not store particle types.
    """
    all_hids = []
    all_pids = []
    halo_offset = 0

    for filepath in _as_path_list(filepaths):
        with h5py.File(filepath, 'r') as f:
            particles = f['SubhaloParticles']

            # h5py returns an object array of numpy arrays for VLEN datasets.
            subhalo_particles = particles[:]

        lengths = np.fromiter((len(p) for p in subhalo_particles), dtype=np.int64, count=len(subhalo_particles))
        total = int(lengths.sum())
        if total:
            local_hids = np.repeat(np.arange(len(lengths), dtype=np.int64) + halo_offset, lengths)
            nonempty = [p for p in subhalo_particles if len(p)]
            all_hids.append(local_hids)
            all_pids.append(np.concatenate(nonempty).astype(np.int64, copy=False))
        halo_offset += len(lengths)

    if not all_pids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(all_hids), np.concatenate(all_pids)


def build_parent_ids(properties) -> np.ndarray:
    """
    Construct parent-child relationships from HBT+.

    Prefer HBT+'s NestedParentTrackId when present. It gives the immediate
    parent in the nested subhalo tree and is the closest match to Octavian's
    HaloTree model. Older HBT+ outputs or first-level satellites can still be
    resolved through the FoF HostHaloId/Rank central mapping.

    Orphan halos and central halos are assigned a parent -1.
    """
    track_ids = properties['TrackId'].to_numpy().astype(np.int64)
    parent_ids = np.full(len(track_ids), -1, dtype=np.int64)

    if 'NestedParentTrackId' in properties:
        nested_parent_ids = properties['NestedParentTrackId'].to_numpy().astype(np.int64)
        nested_mask = nested_parent_ids != -1
        parent_ids[nested_mask] = nested_parent_ids[nested_mask]

    if 'HostHaloId' not in properties or 'Rank' not in properties:
        return parent_ids

    host_ids = properties['HostHaloId'].to_numpy().astype(np.int64)
    ranks = properties['Rank'].to_numpy().astype(np.int32)

    central_mask = (ranks == 0) & (host_ids != -1)
    central_fof_ids = host_ids[central_mask]
    central_track_ids = track_ids[central_mask]

    if len(central_fof_ids) == 0:
        return parent_ids

    central_order = np.argsort(central_fof_ids)
    central_fof_ids = central_fof_ids[central_order]
    central_track_ids = central_track_ids[central_order]

    satellite_rows = np.flatnonzero((ranks > 0) & (host_ids != -1) & (parent_ids == -1))
    if len(satellite_rows):
        parent_positions = np.searchsorted(central_fof_ids, host_ids[satellite_rows])
        in_bounds = parent_positions < len(central_fof_ids)
        matched = np.zeros(len(satellite_rows), dtype=bool)
        matched[in_bounds] = central_fof_ids[parent_positions[in_bounds]] == host_ids[satellite_rows[in_bounds]]
        parent_ids[satellite_rows[matched]] = central_track_ids[parent_positions[matched]]

    parent_ids[parent_ids == track_ids] = -1
    return parent_ids


def label_ptypes(data_manager: DataManager, member_hids: np.ndarray,
                member_pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HBT+ stores particle IDs but not their types.
    We therefore check the original snapshot to label the ptypes.
    Particles not found in any Octavian ptype are dropped.
    """
    if len(member_pids) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int8),
        )

    config = data_manager.config

    # sort membership pids for searchsorted
    order = np.argsort(member_pids)
    sorted_pids = member_pids[order]
    out_ptypes = np.full(len(member_pids), -1, dtype=np.int8)

    for ptype in config['ptypes']:
        ptype_code = PTYPE_ENCODE.get(ptype)
        if ptype_code is None:
            continue

        data_manager.load_property('pid', ptype)
        snap_pids = data_manager.data[ptype]['pid'].to_numpy(dtype=np.int64)

        # find which membership particles match this ptype
        positions = np.searchsorted(sorted_pids, snap_pids)
        positions = np.clip(positions, 0, len(sorted_pids) - 1)
        matched = sorted_pids[positions] == snap_pids

        out_ptypes[order[positions[matched]]] = ptype_code

    # drop particles not used in Octavian
    valid = out_ptypes != -1
    return member_hids[valid], member_pids[valid], out_ptypes[valid]


def load_hbt(data_manager: DataManager, subhalo_path: str, snap_index: int | None = None, mode='field'):
    """
    Load HBT+ output into Octavian.
    data_manager needs to be loaded on the original snapshot for cross-referencing ptypes.
    This also has the option to just do field halos or capture substructure.
    """
    if subhalo_path is None:
        raise KeyError('hbt_subhalo_path must be set to a specific HBT snapshot directory when halo_source is hbt')

    data_manager.config['halo_source'] = 'hbt'
    data_manager.config['halo_mode'] = mode

    print('Reading HBT+ subhalos...')
    t1 = perf_counter()
    filepaths = gather_subsnap_files(subhalo_path, snap_index)
    print(f"  {len(filepaths)} SubSnap file(s) found")
    properties = read_subhalos(filepaths)
    t2 = perf_counter()
    print(f"{len(properties)} subhalos found")
    print(f"Finished in {(t2 - t1):.3f} seconds.")

    track_ids = properties['TrackId'].to_numpy().astype(np.int64)
    parent_ids = build_parent_ids(properties)

    t1 = perf_counter()
    print("Reading HBT+ particles...")
    member_hids, member_pids = read_particles(filepaths)
    t2 = perf_counter()
    print(f"  {len(member_pids)} particle entries read")
    print(f"Finished in {(t2 - t1):.3f} seconds.")

    # member_hids are subhalo row indices; map to TrackIds.
    member_hids = track_ids[member_hids]

    t1 = perf_counter()
    print(f"Cross-referencing particle types from snapshot...")
    member_hids, member_pids, member_ptypes = label_ptypes(
        data_manager, member_hids, member_pids
    )
    t2 = perf_counter()
    print(f"Finished in {(t2 - t1):.3f} seconds.")
    print(f"{len(member_pids)} particles matched to Octavian types")

    # remap TrackIds to the HaloReader-friendly 0, 1, 2 etc. format
    # HBT+ already uses -1 for field halos
    print(f"Extracting halo structure and membership...")
    t1 = perf_counter()
    reader = HaloReader(data_manager)
    track_ids, parent_ids, member_hids = reader.remap_ids(track_ids, parent_ids, member_hids)
    valid_members = member_hids >= 0
    if not np.all(valid_members):
        member_hids = member_hids[valid_members]
        member_pids = member_pids[valid_members]
        member_ptypes = member_ptypes[valid_members]

    tree = HaloTree(track_ids, parent_ids, properties)
    membership = HaloMembership(tree, member_hids, member_pids, member_ptypes, exclusive=True)
    t2 = perf_counter()
    print(f"Finished in {(t2 - t1):.3f} seconds.")

    data_manager.halo_tree = tree
    data_manager.halo_membership = membership

    reader.assign(membership, mode)

    # diagnostics
    n_orphans = (properties['Nbound'].to_numpy() <= 1).sum()
    n_centrals = (properties['Rank'].to_numpy() == 0).sum()
    print(f'  {n_centrals} centrals, {len(properties) - n_centrals} satellites, {n_orphans} orphans')
