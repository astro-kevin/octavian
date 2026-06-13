import h5py
import numpy as np
from time import perf_counter
from yaml import safe_load

from octavian.halo_reader import (
  build_snapshot_membership_arrays,
  membership_array_exclusive_ids,
  membership_top_id_counts,
  prune_halo_tree,
  write_staged_halo_tree,
)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_id_filter(f: h5py.File, ptypes: list[str], nsplit: int) -> list[list[int]]:
  ids = []
  for ptype in ptypes:
    ids_ptype = f[ptype]['HaloID'][:]
    ids.append(ids_ptype[ids_ptype != 0])

  ids = np.sort(np.concatenate(ids))
  unique_ids, counts = np.unique(ids, return_counts=True)
  cumulative_counts = np.cumsum(counts)
  total = len(ids)

  split_ids = [0]
  split_fractions = np.linspace(0., 1., nsplit + 1)
  split_fractions = split_fractions[1:]

  for fraction in split_fractions:
    fraction_count = total * fraction
    split_ids.append(unique_ids[(np.abs(cumulative_counts - fraction_count)).argmin()])

  id_filter = list(zip(split_ids[:-1], split_ids[1:]))

  return id_filter

def _rank_dtype(nsplit: int):
  if nsplit <= np.iinfo(np.int8).max + 1:
    return np.int8
  if nsplit <= np.iinfo(np.int16).max + 1:
    return np.int16
  return np.int32



def _dense_rank_ids(top_ids: np.ndarray, rank_lookup: np.ndarray, dtype) -> np.ndarray:
  rank_ids = np.full(len(top_ids), -1, dtype=dtype)
  if len(rank_lookup) == 0 or len(top_ids) == 0:
    return rank_ids

  valid = (top_ids >= 0) & (top_ids < len(rank_lookup))
  if np.any(valid):
    rank_ids[valid] = rank_lookup[top_ids[valid]]
  return rank_ids


def _rows_by_rank(rank_ids: np.ndarray, nsplit: int) -> list[np.ndarray]:
  return [np.flatnonzero(rank_ids == rank).astype(np.uint32, copy=False) for rank in range(nsplit)]


def _ranked_row_selection(rows_by_rank: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  counts = np.fromiter((len(rows) for rows in rows_by_rank), dtype=np.int64, count=len(rows_by_rank))
  boundaries = np.empty(len(rows_by_rank) + 1, dtype=np.int64)
  boundaries[0] = 0
  np.cumsum(counts, out=boundaries[1:])

  if boundaries[-1] == 0:
    dtype = rows_by_rank[0].dtype if rows_by_rank else np.uint32
    return np.empty(0, dtype=dtype), boundaries

  return np.concatenate(rows_by_rank), boundaries


def _write_dataset(group, name: str, values) -> None:
  if name in group:
    del group[name]
  group.create_dataset(name, data=values)


def _source_dataset_offset(dataset) -> int:
  try:
    offset = dataset.id.get_offset()
  except Exception:
    offset = None
  if offset is None or offset < 0:
    return np.iinfo(np.int64).max
  return int(offset)


_STAGING_REQUIRED_PROPS = {
  'gas': ('pid', 'pos', 'vel', 'mass', 'potential', 'rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature', 'dustmass'),
  'dm': ('pid', 'pos', 'vel', 'mass', 'potential'),
  'star': ('pid', 'pos', 'vel', 'mass', 'potential', 'age', 'metallicity'),
  'bh': ('pid', 'pos', 'vel', 'bhmass', 'potential', 'bhmdot'),
}


def _logical_ptype_for_group(config: dict, ptype_name: str) -> str | None:
  for logical_ptype, configured_name in config.get('ptype_names', {}).items():
    if configured_name == ptype_name:
      return logical_ptype
  return None


def _required_source_dataset_names(config: dict, ptype_name: str) -> set[str]:
  logical_ptype = _logical_ptype_for_group(config, ptype_name)
  if logical_ptype is None:
    return set()

  prop_aliases = config.get('prop_aliases', {})
  names = set()
  for prop in _STAGING_REQUIRED_PROPS.get(logical_ptype, ()):
    name = prop_aliases.get(prop)
    if name is not None:
      names.add(name)

  for prop in config.get('staging_properties', {}).get('all', ()):
    name = prop_aliases.get(prop, prop)
    if name is not None:
      names.add(name)
  for prop in config.get('staging_properties', {}).get(logical_ptype, ()):
    name = prop_aliases.get(prop, prop)
    if name is not None:
      names.add(name)

  return names


def _source_dataset_names(group, config: dict, ptype_name: str) -> list[str]:
  excluded = {'HaloID', 'HaloID_array', 'particle_index'}
  required = _required_source_dataset_names(config, ptype_name)
  names = [name for name in group.keys() if name not in excluded and name in required]
  return sorted(names, key=lambda name: (_source_dataset_offset(group[name]), name))


def _metallicity_dataset_name(config: dict):
  return config.get('prop_aliases', {}).get('metallicity')


def _include_metallicities(config: dict) -> bool:
  return bool(config.get('include_metallicities', False))


def _rank_ordered_source_dataset_names(config: dict) -> set[str]:
  props = config.get('staging_rank_order_properties', ('metallicity',))
  if props is None:
    return set()
  if isinstance(props, str):
    props = (props,)

  prop_aliases = config.get('prop_aliases', {})
  metallicity_name = _metallicity_dataset_name(config)
  names = set()
  for prop in props:
    name = prop_aliases.get(prop, prop)
    if name is None:
      continue
    if name == metallicity_name and not _include_metallicities(config):
      continue
    names.add(name)
  return names


def _write_ranked_values(groups: list, name: str, data, rows_by_rank: list[np.ndarray], ranked_rows=None) -> float:
  t = perf_counter()
  if ranked_rows is None:
    for rank, rows in enumerate(rows_by_rank):
      _write_dataset(groups[rank], name, data[rows])
    return perf_counter() - t

  ordered_rows, boundaries = ranked_rows
  selected = data[ordered_rows]
  for rank in range(len(rows_by_rank)):
    _write_dataset(groups[rank], name, selected[boundaries[rank]:boundaries[rank + 1]])
  return perf_counter() - t


def _ranked_rows_for_dataset(name: str, rank_ordered_names: set[str], rows_by_rank: list[np.ndarray], ranked_rows_cache: list) -> tuple[np.ndarray, np.ndarray] | None:
  if name not in rank_ordered_names:
    return None
  if ranked_rows_cache[0] is None:
    ranked_rows_cache[0] = _ranked_row_selection(rows_by_rank)
  return ranked_rows_cache[0]


def _rank_order_gap_rows(config: dict) -> int:
  gap_rows = int(config.get('staging_rank_order_gap_rows', 4096))
  if gap_rows < 0:
    raise ValueError('staging_rank_order_gap_rows must be non-negative')
  return gap_rows


def _rank_order_max_read_fraction(config: dict) -> float:
  fraction = float(config.get('staging_rank_order_max_read_fraction', 0.9))
  if fraction <= 0 or fraction > 1:
    raise ValueError('staging_rank_order_max_read_fraction must be in the range (0, 1]')
  return fraction


def _destination_index_dtype(counts: np.ndarray):
  max_count = int(counts.max()) if len(counts) else 0
  return np.uint32 if max_count <= np.iinfo(np.uint32).max else np.int64


def _write_empty_ranked_datasets(groups: list, name: str, dataset, counts: np.ndarray) -> tuple[float, float, str]:
  output_shape = tuple(dataset.shape[1:])
  t = perf_counter()
  for rank, count in enumerate(counts):
    values = np.empty((int(count),) + output_shape, dtype=dataset.dtype)
    _write_dataset(groups[rank], name, values)
  return 0.0, perf_counter() - t, 'hdf5_intervals'


def _write_ranked_hdf5_full(groups: list, name: str, dataset, rows_by_rank: list[np.ndarray], t0: float) -> tuple[float, float, str]:
  data = dataset[:]
  read_time = perf_counter() - t0
  ranked_rows = _ranked_row_selection(rows_by_rank)
  write_time = _write_ranked_values(groups, name, data, rows_by_rank, ranked_rows)
  del data
  return read_time, write_time, 'hdf5_ranked_full'


def _write_ranked_hdf5_intervals(groups: list, name: str, dataset, rows_by_rank: list[np.ndarray], config: dict) -> tuple[float, float, str]:
  """Copy selected HDF5 rows into rank outputs without always reading the full source dataset.

  Rows are first sorted by source offset so nearby selections can be read as contiguous
  intervals. If those intervals cover most of the dataset, this falls back to a single full
  read because that is cheaper than many small HDF5 slices.
  """
  t = perf_counter()
  counts = np.fromiter((len(rows) for rows in rows_by_rank), dtype=np.int64, count=len(rows_by_rank))
  if int(counts.sum()) == 0:
    return _write_empty_ranked_datasets(groups, name, dataset, counts)

  # Sort all requested source rows once, then split at large gaps. Each resulting interval is
  # small enough to avoid loading unrelated particles while still keeping HDF5 reads sequential.
  all_rows = np.concatenate(rows_by_rank)
  order = np.argsort(all_rows, kind='stable')
  sorted_rows = all_rows[order]
  gaps = np.diff(sorted_rows.astype(np.int64, copy=False))
  starts = np.concatenate(([0], np.flatnonzero(gaps > _rank_order_gap_rows(config)) + 1))
  ends = np.concatenate((starts[1:], [len(sorted_rows)]))
  read_rows = np.sum(sorted_rows[ends - 1].astype(np.int64) - sorted_rows[starts].astype(np.int64) + 1)

  if float(read_rows) / len(dataset) >= _rank_order_max_read_fraction(config):
    del all_rows, order, sorted_rows, gaps, starts, ends
    return _write_ranked_hdf5_full(groups, name, dataset, rows_by_rank, t)

  rank_dtype = _rank_dtype(len(rows_by_rank))
  dest_dtype = _destination_index_dtype(counts)
  all_rank = np.concatenate([np.full(len(rows), rank, dtype=rank_dtype) for rank, rows in enumerate(rows_by_rank)])
  all_dest = np.concatenate([np.arange(int(count), dtype=dest_dtype) for count in counts])
  sorted_rank = all_rank[order]
  sorted_dest = all_dest[order]
  del all_rows, all_rank, all_dest, order, gaps

  output_shape = tuple(dataset.shape[1:])
  outputs = [np.empty((int(count),) + output_shape, dtype=dataset.dtype) for count in counts]

  for i0, i1 in zip(starts, ends):
    start = int(sorted_rows[i0])
    stop = int(sorted_rows[i1 - 1]) + 1
    block = dataset[start:stop]
    local_rows = sorted_rows[i0:i1].astype(np.int64, copy=False) - start
    values = block[local_rows]
    ranks = sorted_rank[i0:i1]
    destinations = sorted_dest[i0:i1]
    for rank, output in enumerate(outputs):
      mask = ranks == rank
      if np.any(mask):
        output[destinations[mask]] = values[mask]

  read_time = perf_counter() - t

  t = perf_counter()
  for rank, output in enumerate(outputs):
    _write_dataset(groups[rank], name, output)
  write_time = perf_counter() - t
  return read_time, write_time, 'hdf5_intervals'


def _read_staging_source_dataset(dataset, name: str, config: dict):
  if name == _metallicity_dataset_name(config) and not _include_metallicities(config) and getattr(dataset, 'ndim', 1) > 1:
    return dataset[:, 0:1]
  return dataset[:]


def _is_scalar_membership(membership) -> bool:
  return isinstance(membership, np.ndarray) and membership.ndim == 1


def _halo_catalog_is_empty(tree) -> bool:
  return tree is not None and len(getattr(tree, 'halo_ids', ())) == 0


def _membership_top_id_counts_for_weights(membership, ancestor_arrays) -> tuple[np.ndarray, np.ndarray]:
  if _is_scalar_membership(membership):
    if ancestor_arrays is None:
      raise ValueError('ancestor_arrays is required for scalar halo memberships')
    encoded = membership[membership > 0]
    if len(encoded) == 0:
      return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    exclusive = encoded.astype(np.int64, copy=False) - 1
    values = ancestor_arrays[exclusive, 0].astype(np.int64, copy=False)
    values = values[values >= 0]
    if len(values) == 0:
      return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.unique(values, return_counts=True)
  return membership_top_id_counts(membership)


def _scalar_rows_by_rank(membership: np.ndarray, ancestor_arrays: np.ndarray, rank_lookup: np.ndarray, nsplit: int):
  rows = np.flatnonzero(membership > 0).astype(np.uint32, copy=False)
  if len(rows) == 0:
    return [np.empty(0, dtype=np.uint32) for _ in range(nsplit)], [np.empty(0, dtype=np.int64) for _ in range(nsplit)]

  exclusive = membership[rows].astype(np.int64, copy=False) - 1
  top_ids = ancestor_arrays[exclusive, 0].astype(np.int64, copy=False)
  valid = (top_ids >= 0) & (top_ids < len(rank_lookup))
  if not np.any(valid):
    return [np.empty(0, dtype=np.uint32) for _ in range(nsplit)], [np.empty(0, dtype=np.int64) for _ in range(nsplit)]

  rows = rows[valid]
  exclusive = exclusive[valid]
  ranks = rank_lookup[top_ids[valid]]
  assigned = ranks >= 0
  rows = rows[assigned]
  exclusive = exclusive[assigned]
  ranks = ranks[assigned]

  rows_by_rank = []
  exclusive_by_rank = []
  for rank in range(nsplit):
    mask = ranks == rank
    rows_by_rank.append(rows[mask].astype(np.uint32, copy=False))
    exclusive_by_rank.append(exclusive[mask].astype(np.int64, copy=False))
  return rows_by_rank, exclusive_by_rank


def _stage_ptype_scalar(
  f: h5py.File,
  ptype: str,
  groups: list,
  membership: np.ndarray,
  ancestor_arrays: np.ndarray,
  rank_lookup: np.ndarray,
  nsplit: int,
  mode: str,
  rank_halo_ids: list[set[int]],
  cached_datasets: dict[str, dict[str, np.ndarray]],
  source_reads: set[tuple[str, str]],
  config: dict,
) -> None:
  """Stage one particle type from scalar encoded halo memberships.

  Scalar memberships store ``halo_id + 1`` for the deepest matched source halo. The
  accompanying ``ancestor_arrays`` are used to recover the top-level halo for rank routing
  and to write full ancestry rows when subhalo mode requests ``HaloID_array``.
  """
  n_particles = len(membership)
  source_names = _source_dataset_names(f[ptype], config, ptype)
  ptype_cache = cached_datasets.get(ptype, {})
  for name in source_names:
    if len(f[ptype][name]) != n_particles:
      raise ValueError(f'{ptype}/{name} has {len(f[ptype][name])} rows but expected {n_particles}')

  rows_by_rank, exclusive_by_rank = _scalar_rows_by_rank(membership, ancestor_arrays, rank_lookup, nsplit)
  rank_ordered_names = _rank_ordered_source_dataset_names(config)
  ranked_rows_cache = [None]
  selected = sum(len(rows) for rows in rows_by_rank)
  print(f'{ptype} staging selected={selected} datasets={len(source_names)}', flush=True)

  for rank, exclusive in enumerate(exclusive_by_rank):
    if len(exclusive) == 0:
      continue
    rank_halo_ids[rank].update(int(value) for value in np.unique(exclusive) if value >= 0)

  timings = {}

  # Some readers already had to load source datasets while building memberships. Reuse those
  # arrays first so the original snapshot is not read twice.
  cached_names = [name for name in source_names if name in ptype_cache]
  hdf5_names = [name for name in source_names if name not in ptype_cache]

  for name in cached_names:
    t = perf_counter()
    data = ptype_cache.pop(name)
    read_time = perf_counter() - t
    ranked_rows = _ranked_rows_for_dataset(name, rank_ordered_names, rows_by_rank, ranked_rows_cache)
    write_time = _write_ranked_values(groups, name, data, rows_by_rank, ranked_rows)
    timings[name] = read_time + write_time
    print(f'{ptype} {name}: read={read_time:.1f}s write={write_time:.1f}s source=cached', flush=True)
    del data

  for name in hdf5_names:
    key = (ptype, name)
    if key in source_reads:
      raise RuntimeError(f'{ptype}/{name} would be read from the source snapshot more than once')
    source_reads.add(key)
    if name in rank_ordered_names:
      read_time, write_time, source = _write_ranked_hdf5_intervals(groups, name, f[ptype][name], rows_by_rank, config)
    else:
      t = perf_counter()
      data = _read_staging_source_dataset(f[ptype][name], name, config)
      read_time = perf_counter() - t
      write_time = _write_ranked_values(groups, name, data, rows_by_rank)
      source = 'hdf5'
      del data
    timings[name] = read_time + write_time
    print(f'{ptype} {name}: read={read_time:.1f}s write={write_time:.1f}s source={source}', flush=True)

  t = perf_counter()
  for rank, exclusive in enumerate(exclusive_by_rank):
    if mode == 'subhalo':
      halo_ids = exclusive
    else:
      halo_ids = ancestor_arrays[exclusive, 0].astype(np.int64, copy=False) if len(exclusive) else exclusive
    _write_dataset(groups[rank], 'HaloID', halo_ids)
  timings['HaloID'] = perf_counter() - t

  t = perf_counter()
  for rank, rows in enumerate(rows_by_rank):
    _write_dataset(groups[rank], 'particle_index', rows.astype(np.int64, copy=False))
  timings['particle_index'] = perf_counter() - t

  if mode == 'subhalo':
    t = perf_counter()
    for rank, exclusive in enumerate(exclusive_by_rank):
      values = ancestor_arrays[exclusive] if len(exclusive) else np.empty((0, ancestor_arrays.shape[1]), dtype=np.int32)
      _write_dataset(groups[rank], 'HaloID_array', values)
    timings['HaloID_array'] = perf_counter() - t

  for name in ['HaloID', 'particle_index'] + (['HaloID_array'] if mode == 'subhalo' else []):
    print(f'{ptype} {name}: {timings[name]:.1f}s', flush=True)


def _stage_ptype_dense(
  f: h5py.File,
  ptype: str,
  groups: list,
  halo_id_array: np.ndarray,
  rank_lookup: np.ndarray,
  rank_dtype,
  nsplit: int,
  mode: str,
  rank_halo_ids: list[set[int]],
  cached_datasets: dict[str, dict[str, np.ndarray]],
  source_reads: set[tuple[str, str]],
  config: dict,
  ancestor_arrays: np.ndarray | None = None,
) -> None:
  """Stage one particle type from dense particle x ancestry halo arrays.

  Dense rows already contain every valid ancestor from top-level halo to deepest subhalo.
  Rank assignment is therefore based on the first column, while the scalar ``HaloID`` output
  is either the top halo or the deepest valid halo depending on the requested mode.
  """
  if _is_scalar_membership(halo_id_array):
    if ancestor_arrays is None:
      raise ValueError('ancestor_arrays is required for scalar halo memberships')
    _stage_ptype_scalar(f, ptype, groups, halo_id_array, ancestor_arrays, rank_lookup, nsplit, mode, rank_halo_ids, cached_datasets, source_reads, config)
    return

  n_particles = len(halo_id_array)
  source_names = _source_dataset_names(f[ptype], config, ptype)
  ptype_cache = cached_datasets.get(ptype, {})
  for name in source_names:
    if len(f[ptype][name]) != n_particles:
      raise ValueError(f'{ptype}/{name} has {len(f[ptype][name])} rows but expected {n_particles}')

  top_ids = halo_id_array[:, 0]
  rank_ids = _dense_rank_ids(top_ids, rank_lookup, rank_dtype)
  rows_by_rank = _rows_by_rank(rank_ids, nsplit)
  rank_ordered_names = _rank_ordered_source_dataset_names(config)
  ranked_rows_cache = [None]
  selected = sum(len(rows) for rows in rows_by_rank)
  print(f'{ptype} staging selected={selected} datasets={len(source_names)}', flush=True)

  halo_ids = membership_array_exclusive_ids(halo_id_array) if mode == 'subhalo' else top_ids.astype(np.int64, copy=False)
  for rank, rows in enumerate(rows_by_rank):
    if len(rows) == 0:
      continue
    values = np.unique(halo_ids[rows])
    rank_halo_ids[rank].update(int(value) for value in values if value >= 0)

  timings = {}
  t = perf_counter()
  for rank, rows in enumerate(rows_by_rank):
    _write_dataset(groups[rank], 'HaloID', halo_ids[rows])
  timings['HaloID'] = perf_counter() - t

  t = perf_counter()
  for rank, rows in enumerate(rows_by_rank):
    _write_dataset(groups[rank], 'particle_index', rows.astype(np.int64, copy=False))
  timings['particle_index'] = perf_counter() - t

  if mode == 'subhalo':
    timings['HaloID_array'] = _write_ranked_values(groups, 'HaloID_array', halo_id_array, rows_by_rank)

  del halo_ids, top_ids, rank_ids, halo_id_array

  cached_names = [name for name in source_names if name in ptype_cache]
  hdf5_names = [name for name in source_names if name not in ptype_cache]

  for name in cached_names:
    t = perf_counter()
    data = ptype_cache.pop(name)
    read_time = perf_counter() - t
    ranked_rows = _ranked_rows_for_dataset(name, rank_ordered_names, rows_by_rank, ranked_rows_cache)
    write_time = _write_ranked_values(groups, name, data, rows_by_rank, ranked_rows)
    timings[name] = read_time + write_time
    print(f'{ptype} {name}: read={read_time:.1f}s write={write_time:.1f}s source=cached', flush=True)
    del data

  for name in hdf5_names:
    key = (ptype, name)
    if key in source_reads:
      raise RuntimeError(f'{ptype}/{name} would be read from the source snapshot more than once')
    source_reads.add(key)
    if name in rank_ordered_names:
      read_time, write_time, source = _write_ranked_hdf5_intervals(groups, name, f[ptype][name], rows_by_rank, config)
    else:
      t = perf_counter()
      data = _read_staging_source_dataset(f[ptype][name], name, config)
      read_time = perf_counter() - t
      write_time = _write_ranked_values(groups, name, data, rows_by_rank)
      source = 'hdf5'
      del data
    timings[name] = read_time + write_time
    print(f'{ptype} {name}: read={read_time:.1f}s write={write_time:.1f}s source={source}', flush=True)

  for name in ['HaloID', 'particle_index'] + (['HaloID_array'] if mode == 'subhalo' else []):
    print(f'{ptype} {name}: {timings[name]:.1f}s', flush=True)

def filter_snapshot_with_membership_arrays(f: h5py.File, outfile: str, config: dict, nsplit: int, membership_arrays: dict[str, np.ndarray], mode: str, tree=None, cached_datasets=None, ancestor_arrays=None):
  """Write split snapshots from external halo-reader memberships.

  The algorithm routes whole top-level halos to ranks. It estimates rank cost from star,
  gas, and DM membership counts, then greedily assigns the heaviest halos to the currently
  lightest rank. This keeps all particles needed for one halo calculation together while
  balancing the expensive FOF6D and group-property phases.
  """
  for i in range(nsplit):
    with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
      f.copy(f['Header'], f_out, 'Header')

  # Process the largest membership arrays first so temporary arrays can be released earlier.
  ptypes = sorted(membership_arrays, key=lambda ptype: membership_arrays[ptype].nbytes, reverse=True)
  star_weights, gas_weights, dm_weights = {}, {}, {}
  for ptype_name, weight_dict in [('PartType4', star_weights), ('PartType0', gas_weights), ('PartType1', dm_weights)]:
    if ptype_name not in membership_arrays:
      continue
    unique, counts = _membership_top_id_counts_for_weights(membership_arrays[ptype_name], ancestor_arrays)
    for hid, count in zip(unique, counts):
      weight_dict[int(hid)] = int(count)

  weights = {}
  for hid in set(star_weights) | set(gas_weights) | set(dm_weights):
    n_total = star_weights.get(hid, 0) + gas_weights.get(hid, 0) + dm_weights.get(hid, 0)
    if n_total < config['MINIMUM_DM_PER_HALO']: continue
    fof6d_cost = (star_weights.get(hid, 0))**1.2 + gas_weights.get(hid, 0)
    cgp_cost = n_total
    weights[hid] = 0.6 * fof6d_cost + 0.4 * cgp_cost

  # Greedy bin packing: largest estimated halos first, always placed on the lightest rank.
  rank_assignments = [set() for _ in range(nsplit)]
  rank_loads = [0] * nsplit
  for hid in sorted(weights, key=weights.get, reverse=True):
    lightest = np.argmin(rank_loads)
    rank_assignments[lightest].add(hid)
    rank_loads[lightest] += weights[hid]

  rank_dtype = _rank_dtype(nsplit)
  max_halo_id = max(weights) if weights else -1
  rank_lookup = np.full(max_halo_id + 1, -1, dtype=rank_dtype)
  for rank, halo_ids in enumerate(rank_assignments):
    if halo_ids:
      rank_lookup[np.fromiter(halo_ids, dtype=np.int64)] = rank

  rank_halo_ids = [set() for _ in range(nsplit)]

  cached_datasets = cached_datasets or {}
  source_reads = set()

  for ptype in ptypes:
    halo_id_array = membership_arrays.pop(ptype)

    out_files = [h5py.File(f'{outfile}_{i}.hdf5', 'a') for i in range(nsplit)]
    try:
      groups = [f_out.require_group(ptype) for f_out in out_files]
      _stage_ptype_dense(f, ptype, groups, halo_id_array, rank_lookup, rank_dtype, nsplit, mode, rank_halo_ids, cached_datasets, source_reads, config, ancestor_arrays=ancestor_arrays)
    finally:
      for f_out in out_files:
        f_out.close()

  if tree is not None:
    for i, halo_ids_in_rank in enumerate(rank_halo_ids):
      # Store only the hierarchy needed by this shard, including ancestors of selected halos.
      pruned_tree = prune_halo_tree(tree, halo_ids_in_rank)
      with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
        write_staged_halo_tree(f_out, pruned_tree)

def filter_snapshot(snapfile: str, outfile: str, configfile: str, nsplit: int=4):
  """
  Weighted snapshot filter.

  This snapshot filter is designed to be weighted towards balancing FOF6D. It does so by applying a
  power law to star/gas counts when deciding how to divide the snapshot. FOF6D can take extremely long
  and ranks can have wildly different runtimes if the snapshot is not weighted when filtered.
  """

  # these are weighting constants. cgp scales better than fof6d so ideally lean towards fof6d
  ALPHA = 0.6 # arbitrary fof6d constant
  BETA = 0.4 # arbitrary cgp constant

  with open(configfile, 'r') as f:
    config = safe_load(f)

  with h5py.File(snapfile, 'r') as f:
    halo_source = config.get('halo_source')
    halo_mode = config.get('halo_mode', 'field')
    if halo_source:
      t = perf_counter()
      print(f'Building {halo_source.upper()} HaloID arrays...', flush=True)
      build_result = build_snapshot_membership_arrays(f, config)
      tree, membership_arrays, counts = build_result
      cached_datasets = getattr(build_result, 'cached_datasets', {})
      ancestor_arrays = getattr(build_result, 'ancestor_arrays', None)
      print(f'  Built {halo_source.upper()} membership arrays: {perf_counter() - t:.1f}s', flush=True)
      if _halo_catalog_is_empty(tree):
        print(f'  {halo_source.upper()} catalog is empty; skipping snapshot.', flush=True)
        return
      if isinstance(counts, np.ndarray):
        print(f'  {halo_source.upper()} memberships written: {int(counts[:4].sum())}, conflicts resolved: {int(counts[7])}', flush=True)
      else:
        count_text = ', '.join(f'{ptype}={count}' for ptype, count in counts.items())
        print(f'  {halo_source.upper()} memberships written: {count_text}', flush=True)
      t = perf_counter()
      filter_snapshot_with_membership_arrays(f, outfile, config, nsplit, membership_arrays, halo_mode, tree=tree, cached_datasets=cached_datasets, ancestor_arrays=ancestor_arrays)
      print(f'  Wrote split snapshots: {perf_counter() - t:.1f}s', flush=True)
      return

    for i in range(nsplit):
      with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
        f.copy(f['Header'], f_out, 'Header')

    #
    # algorithm to weight split snapshot
    #

    ptypes = [group for group in list(f.keys()) if 'HaloID' in list(f[group].keys())] # from Jakub's code
    # initialise weight dictionaries
    star_weights = {}
    gas_weights = {}
    dm_weights = {}

    for ptype_name, weight_dict in [
    ('PartType4', star_weights),
    ('PartType0', gas_weights),
    ('PartType1', dm_weights),
    ]: # config is not passed so refer to them by PartType
      ptype_ids = f[ptype_name]['HaloID'][:] # access star/gas particles and their halo IDs
      ptype_ids = ptype_ids[ptype_ids != 0] # access only the stars/gas in a valid halo
      unique, counts = np.unique(ptype_ids, return_counts=True) # find the counts of that particle for a unique halo

      # find a raw weight
      for hid, count in zip(unique, counts):
          weight_dict[hid] = count

    weights = {}
    for hid in set(star_weights) | set(gas_weights) | set(dm_weights): # union operator: find halos in both sets
      n_total = star_weights.get(hid, 0) + gas_weights.get(hid, 0) + dm_weights.get(hid, 0)
      if n_total < config['MINIMUM_DM_PER_HALO']: continue # validate halos
      fof6d_cost = (star_weights.get(hid, 0))**1.2 + gas_weights.get(hid, 0)
      cgp_cost = n_total  # roughly linear in total particles per halo
      weights[hid] = ALPHA * fof6d_cost + BETA * cgp_cost

    # account for theoretical pure dark matter halo (these still need to be assigned)
    # this could maybe be removed
    # all_ids = set()
    # for ptype in ptypes:
    #     ptype_ids = f[ptype]['HaloID'][:]
    #     all_ids.update(ptype_ids[ptype_ids != 0])
    # for hid in all_ids:
    #     weights.setdefault(hid, 0)

    # simple sequential binning algorithm
    rank_assignments = [set() for _ in range(nsplit)] # initialise a set
    rank_loads = [0] * nsplit
    for hid in sorted(weights, key=weights.get, reverse=True): # sort by heaviest first
        # we go from heaviest -> lightest, adding the next halo to the bin with the smallest load
        lightest = np.argmin(rank_loads) # find which rank has the lowest load
        rank_assignments[lightest].add(hid)
        rank_loads[lightest] += weights[hid]

    # and now the actual filter
    # toss particles not in a halo
    for ptype in ptypes:
      datasets = list(f[ptype].keys())
      ids = f[ptype]['HaloID'][:]
      particle_index = np.arange(len(ids), dtype='int')
      in_halo = ids != 0 # find ids not in a halo
      ids_filtered = ids[in_halo]
      order = np.argsort(ids_filtered)
      ids_sorted = ids_filtered[order]
      datasets = datasets + ['particle_index']

      # Jakub's code masks once per dataset but we could mask once per ptype
      rank_masks = []
      for i in range(nsplit):
          halo_set = np.array(list(rank_assignments[i]))
          rank_masks.append(np.isin(ids_sorted, halo_set))

      for dataset in datasets:
        print(ptype, dataset)
        if dataset == 'particle_index':        # <-- add
            data = particle_index[in_halo][order]
        else:
          data = f[ptype][dataset][:][in_halo][order]
        for i in range(nsplit):
            with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
                f_out.require_group(ptype)
                f_out[ptype][dataset] = data[rank_masks[i]]

def filter_snapshot_unweighted(snapfile: str, outfile: str, nsplit: int=4):
  """
  Filters snapshot simply by number of particles.

  This can cause load balancing issues:

  FOF6D is more sensitive to particle type distributions because it does not care for dark matter particles.
  This means that the largest halos by total nparticles are not necessarily the most computationally expensive,
  meaning you can end up with wildly different FOF6D runtimes across ranks.

  """

  # original Jakub implemenation
  with h5py.File(snapfile, 'r') as f:
    for i in range(nsplit):
      with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
        f.copy(f['Header'], f_out, 'Header')


    ptypes = [group for group in list(f.keys()) if 'HaloID' in list(f[group].keys())]
    id_filter = get_id_filter(f, ptypes, nsplit)

    for ptype in ptypes:
      datasets = list(f[ptype].keys())

      ids = f[ptype]['HaloID'][:]
      particle_index = np.arange(len(ids), dtype='int')

      in_halo = ids != 0
      ids = ids[in_halo]

      order = np.argsort(ids)
      ids = ids[order]

      datasets = datasets + ['particle_index']

      for dataset in datasets:
        print(ptype, dataset)
        if dataset == 'particle_index':
          data = particle_index[in_halo][order]
        else:
          data = f[ptype][dataset][:][in_halo][order]

        for i, (start, end) in enumerate(id_filter):
          with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
            f_out.require_group(ptype)
            in_halos = np.logical_and(ids > start, ids <= end)
            f_out[ptype][dataset] = data[in_halos]
