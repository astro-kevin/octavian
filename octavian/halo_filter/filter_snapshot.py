import h5py
import numpy as np
from time import perf_counter
from yaml import safe_load

from octavian.halo_reader import (
  build_snapshot_membership_arrays,
  membership_depth_width,
  membership_particle_count,
  membership_rank_ids,
  membership_selected_exclusive_ids,
  membership_selected_particles_dense,
  membership_selected_top_ids,
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


def _rank_counts(rank_ids: np.ndarray, nsplit: int) -> np.ndarray:
  valid = rank_ids >= 0
  if not np.any(valid):
    return np.zeros(nsplit, dtype=np.int64)
  return np.bincount(rank_ids[valid].astype(np.int64, copy=False), minlength=nsplit)[:nsplit].astype(np.int64, copy=False)


def _chunk_rows_for_row_bytes(n_rows: int, row_bytes: int, config: dict) -> int:
  configured_rows = config.get('staging_property_chunk_rows', config.get('staging_chunk_rows'))
  if configured_rows is not None:
    chunk_rows = int(configured_rows)
    if chunk_rows < 1:
      raise ValueError('staging_property_chunk_rows must be positive')
    return min(n_rows, chunk_rows)

  target_bytes = int(config.get('staging_property_chunk_bytes', 256 * 1024**2))
  max_rows = int(config.get('staging_property_chunk_max_rows', 20_000_000))
  if target_bytes < 1:
    raise ValueError('staging_property_chunk_bytes must be positive')
  if max_rows < 1:
    raise ValueError('staging_property_chunk_max_rows must be positive')

  row_bytes = max(int(row_bytes), 1)
  return max(1, min(n_rows, max_rows, target_bytes // row_bytes))


def _chunk_rows_for_dataset(dataset, config: dict) -> int:
  row_shape = dataset.shape[1:]
  row_elements = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1
  return _chunk_rows_for_row_bytes(len(dataset), dataset.dtype.itemsize * row_elements, config)


def _create_ranked_datasets(groups, name: str, counts: np.ndarray, shape_tail: tuple, dtype) -> list:
  outputs = []
  shape_tail = tuple(int(axis) for axis in shape_tail)
  for group, count in zip(groups, counts):
    if name in group:
      del group[name]
    outputs.append(group.create_dataset(name, shape=(int(count),) + shape_tail, dtype=dtype))
  return outputs


def _check_cursors(name: str, cursors: np.ndarray, counts: np.ndarray) -> None:
  if not np.array_equal(cursors, counts):
    raise RuntimeError(f'{name} staging wrote {cursors.tolist()} rows but expected {counts.tolist()}')


def _write_selected_values_by_rank(values, value_ranks: np.ndarray, outputs: list, cursors: np.ndarray, nsplit: int) -> None:
  for rank in range(nsplit):
    mask = value_ranks == rank
    n_write = int(np.count_nonzero(mask))
    if n_write == 0:
      continue
    cursor = int(cursors[rank])
    outputs[rank][cursor:cursor + n_write] = values[mask]
    cursors[rank] += n_write


def _copy_source_dataset_by_rank(source_dataset, outputs: list, rank_ids: np.ndarray, counts: np.ndarray, config: dict, nsplit: int) -> None:
  chunk_rows = _chunk_rows_for_dataset(source_dataset, config)
  cursors = np.zeros(nsplit, dtype=np.int64)
  for start in range(0, len(source_dataset), chunk_rows):
    end = min(start + chunk_rows, len(source_dataset))
    rank_chunk = rank_ids[start:end]
    if not np.any(rank_chunk >= 0):
      continue

    data_chunk = source_dataset[start:end]
    for rank in range(nsplit):
      mask = rank_chunk == rank
      n_write = int(np.count_nonzero(mask))
      if n_write == 0:
        continue
      cursor = int(cursors[rank])
      outputs[rank][cursor:cursor + n_write] = data_chunk[mask]
      cursors[rank] += n_write

  _check_cursors(source_dataset.name, cursors, counts)


def _write_particle_index_by_rank(outputs: list, rank_ids: np.ndarray, counts: np.ndarray, config: dict, nsplit: int) -> None:
  chunk_rows = _chunk_rows_for_row_bytes(len(rank_ids), np.dtype(np.int64).itemsize, config)
  cursors = np.zeros(nsplit, dtype=np.int64)
  for start in range(0, len(rank_ids), chunk_rows):
    end = min(start + chunk_rows, len(rank_ids))
    rank_chunk = rank_ids[start:end]
    assigned = rank_chunk >= 0
    if not np.any(assigned):
      continue
    values = (np.flatnonzero(assigned).astype(np.int64, copy=False) + start)
    _write_selected_values_by_rank(values, rank_chunk[assigned], outputs, cursors, nsplit)

  _check_cursors('particle_index', cursors, counts)


def _write_halo_ids_by_rank(outputs: list, rank_ids: np.ndarray, counts: np.ndarray, halo_id_array, mode: str, config: dict, nsplit: int, rank_halo_ids: list[set[int]]) -> None:
  chunk_rows = _chunk_rows_for_row_bytes(len(rank_ids), np.dtype(np.int64).itemsize, config)
  cursors = np.zeros(nsplit, dtype=np.int64)
  selector = membership_selected_exclusive_ids if mode == 'subhalo' else membership_selected_top_ids

  for start in range(0, len(rank_ids), chunk_rows):
    end = min(start + chunk_rows, len(rank_ids))
    rank_chunk = rank_ids[start:end]
    assigned = rank_chunk >= 0
    if not np.any(assigned):
      continue

    rows = np.flatnonzero(assigned).astype(np.int64, copy=False) + start
    values = selector(halo_id_array, rows)
    value_ranks = rank_chunk[assigned]
    _write_selected_values_by_rank(values, value_ranks, outputs, cursors, nsplit)

    for rank in range(nsplit):
      rank_values = values[value_ranks == rank]
      if len(rank_values):
        rank_halo_ids[rank].update(int(value) for value in np.unique(rank_values) if value >= 0)

  _check_cursors('HaloID', cursors, counts)


def _write_halo_id_array_by_rank(outputs: list, rank_ids: np.ndarray, counts: np.ndarray, halo_id_array, config: dict, nsplit: int) -> None:
  row_bytes = membership_depth_width(halo_id_array) * np.dtype(np.int32).itemsize
  chunk_rows = _chunk_rows_for_row_bytes(len(rank_ids), row_bytes, config)
  cursors = np.zeros(nsplit, dtype=np.int64)

  for start in range(0, len(rank_ids), chunk_rows):
    end = min(start + chunk_rows, len(rank_ids))
    rank_chunk = rank_ids[start:end]
    assigned = rank_chunk >= 0
    if not np.any(assigned):
      continue

    rows = np.flatnonzero(assigned).astype(np.int64, copy=False) + start
    values = membership_selected_particles_dense(halo_id_array, rows)
    _write_selected_values_by_rank(values, rank_chunk[assigned], outputs, cursors, nsplit)

  _check_cursors('HaloID_array', cursors, counts)


def filter_snapshot_with_membership_arrays(f: h5py.File, outfile: str, config: dict, nsplit: int, membership_arrays: dict[str, np.ndarray], mode: str, tree=None):
  for i in range(nsplit):
    with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
      f.copy(f['Header'], f_out, 'Header')

  ptypes = list(membership_arrays)
  star_weights, gas_weights, dm_weights = {}, {}, {}
  for ptype_name, weight_dict in [('PartType4', star_weights), ('PartType0', gas_weights), ('PartType1', dm_weights)]:
    if ptype_name not in membership_arrays:
      continue
    unique, counts = membership_top_id_counts(membership_arrays[ptype_name])
    for hid, count in zip(unique, counts):
      weight_dict[int(hid)] = int(count)

  weights = {}
  for hid in set(star_weights) | set(gas_weights) | set(dm_weights):
    n_total = star_weights.get(hid, 0) + gas_weights.get(hid, 0) + dm_weights.get(hid, 0)
    if n_total < config['MINIMUM_DM_PER_HALO']: continue
    fof6d_cost = (star_weights.get(hid, 0))**1.2 + gas_weights.get(hid, 0)
    cgp_cost = n_total
    weights[hid] = 0.6 * fof6d_cost + 0.4 * cgp_cost

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

  for ptype in ptypes:
    halo_id_array = membership_arrays[ptype]
    n_particles = membership_particle_count(halo_id_array)
    rank_ids = membership_rank_ids(halo_id_array, rank_lookup, dtype=rank_dtype)
    counts = _rank_counts(rank_ids, nsplit)
    datasets = [dataset for dataset in f[ptype].keys() if dataset not in ('HaloID', 'HaloID_array', 'particle_index')]
    datasets += ['HaloID', 'particle_index']
    if mode == 'subhalo':
      datasets.append('HaloID_array')

    out_files = [h5py.File(f'{outfile}_{i}.hdf5', 'a') for i in range(nsplit)]
    try:
      groups = [f_out.require_group(ptype) for f_out in out_files]
      for dataset in datasets:
        print(ptype, dataset, flush=True)
        if dataset == 'HaloID':
          outputs = _create_ranked_datasets(groups, dataset, counts, (), np.int64)
          _write_halo_ids_by_rank(outputs, rank_ids, counts, halo_id_array, mode, config, nsplit, rank_halo_ids)
        elif dataset == 'HaloID_array':
          outputs = _create_ranked_datasets(groups, dataset, counts, (membership_depth_width(halo_id_array),), np.int32)
          _write_halo_id_array_by_rank(outputs, rank_ids, counts, halo_id_array, config, nsplit)
        elif dataset == 'particle_index':
          outputs = _create_ranked_datasets(groups, dataset, counts, (), np.int64)
          _write_particle_index_by_rank(outputs, rank_ids, counts, config, nsplit)
        else:
          source_dataset = f[ptype][dataset]
          if len(source_dataset) != n_particles:
            raise ValueError(f'{ptype}/{dataset} has {len(source_dataset)} rows but expected {n_particles}')
          outputs = _create_ranked_datasets(groups, dataset, counts, source_dataset.shape[1:], source_dataset.dtype)
          _copy_source_dataset_by_rank(source_dataset, outputs, rank_ids, counts, config, nsplit)
    finally:
      for f_out in out_files:
        f_out.close()

    del rank_ids

  if tree is not None:
    for i, halo_ids_in_rank in enumerate(rank_halo_ids):
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
      tree, membership_arrays, counts = build_snapshot_membership_arrays(f, config)
      print(f'  Built {halo_source.upper()} membership arrays: {perf_counter() - t:.1f}s', flush=True)
      if isinstance(counts, np.ndarray):
        print(f'  {halo_source.upper()} memberships written: {int(counts[:4].sum())}, conflicts resolved: {int(counts[7])}', flush=True)
      else:
        count_text = ', '.join(f'{ptype}={count}' for ptype, count in counts.items())
        print(f'  {halo_source.upper()} memberships written: {count_text}', flush=True)
      t = perf_counter()
      filter_snapshot_with_membership_arrays(f, outfile, config, nsplit, membership_arrays, halo_mode, tree=tree)
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
