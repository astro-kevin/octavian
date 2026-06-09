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



def _output_chunk_shape(shape_tail: tuple, dtype, input_chunk_rows: int, config: dict) -> tuple:
  shape_tail = tuple(int(axis) for axis in shape_tail)
  row_elements = int(np.prod(shape_tail, dtype=np.int64)) if shape_tail else 1
  row_bytes = max(np.dtype(dtype).itemsize * row_elements, 1)
  target_bytes = int(config.get('staging_output_chunk_bytes', 16 * 1024**2))
  rows = max(1, min(int(input_chunk_rows), max(1, target_bytes // row_bytes)))
  return (rows,) + shape_tail


def _create_extendible_dataset(group, name: str, shape_tail: tuple, dtype, input_chunk_rows: int, config: dict):
  shape_tail = tuple(int(axis) for axis in shape_tail)
  if name in group:
    del group[name]
  return group.create_dataset(
    name,
    shape=(0,) + shape_tail,
    maxshape=(None,) + shape_tail,
    chunks=_output_chunk_shape(shape_tail, dtype, input_chunk_rows, config),
    dtype=dtype,
  )


def _create_extendible_outputs(groups, name: str, shape_tail: tuple, dtype, input_chunk_rows: int, config: dict) -> list:
  return [_create_extendible_dataset(group, name, shape_tail, dtype, input_chunk_rows, config) for group in groups]


def _append_values(dataset, values) -> None:
  n_write = len(values)
  if n_write == 0:
    return
  old_size = len(dataset)
  new_size = old_size + n_write
  dataset.resize((new_size,) + dataset.shape[1:])
  dataset[old_size:new_size] = values


def _append_chunk_values_by_rank(outputs: list, values, rows_by_rank: list[np.ndarray]) -> None:
  for rank, rows in enumerate(rows_by_rank):
    if len(rows) == 0:
      continue
    _append_values(outputs[rank], values[rows])


def _append_selected_values_by_rank(outputs: list, values, selected_positions_by_rank: list[np.ndarray]) -> None:
  for rank, positions in enumerate(selected_positions_by_rank):
    if len(positions) == 0:
      continue
    _append_values(outputs[rank], values[positions])


def _source_dataset_row_bytes(dataset) -> int:
  row_shape = dataset.shape[1:]
  row_elements = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1
  return dataset.dtype.itemsize * row_elements


def _staging_chunk_rows(n_particles: int, source_datasets: list, halo_id_array, mode: str, config: dict) -> int:
  row_bytes = np.dtype(np.int64).itemsize  # particle_index
  row_bytes += np.dtype(np.int64).itemsize  # HaloID
  if mode == 'subhalo':
    row_bytes += membership_depth_width(halo_id_array) * np.dtype(np.int32).itemsize
  for dataset in source_datasets:
    row_bytes += _source_dataset_row_bytes(dataset)
  return _chunk_rows_for_row_bytes(n_particles, row_bytes, config)


def _chunk_rank_rows(rank_ids: np.ndarray, start: int, end: int, nsplit: int):
  rank_chunk = rank_ids[start:end]
  selected_rows = np.flatnonzero(rank_chunk >= 0).astype(np.int32, copy=False)
  if len(selected_rows) == 0:
    empty = [np.empty(0, dtype=np.int32) for _ in range(nsplit)]
    return selected_rows, empty, empty

  selected_ranks = rank_chunk[selected_rows]
  selected_positions_by_rank = [
    np.flatnonzero(selected_ranks == rank).astype(np.int32, copy=False)
    for rank in range(nsplit)
  ]
  rows_by_rank = [selected_rows[positions] for positions in selected_positions_by_rank]
  return selected_rows, rows_by_rank, selected_positions_by_rank


def _stage_ptype_by_row_chunks(
  f: h5py.File,
  ptype: str,
  groups: list,
  halo_id_array,
  rank_ids: np.ndarray,
  config: dict,
  nsplit: int,
  mode: str,
  rank_halo_ids: list[set[int]],
) -> None:
  n_particles = membership_particle_count(halo_id_array)
  source_names = [dataset for dataset in f[ptype].keys() if dataset not in ('HaloID', 'HaloID_array', 'particle_index')]
  source_specs = []
  for name in source_names:
    dataset = f[ptype][name]
    if len(dataset) != n_particles:
      raise ValueError(f'{ptype}/{name} has {len(dataset)} rows but expected {n_particles}')
    source_specs.append((name, dataset))

  source_datasets = [dataset for _, dataset in source_specs]
  chunk_rows = _staging_chunk_rows(n_particles, source_datasets, halo_id_array, mode, config)
  print(f'{ptype} staging chunk_rows={chunk_rows} datasets={len(source_specs)}', flush=True)

  source_outputs = {
    name: _create_extendible_outputs(groups, name, dataset.shape[1:], dataset.dtype, chunk_rows, config)
    for name, dataset in source_specs
  }
  halo_outputs = _create_extendible_outputs(groups, 'HaloID', (), np.int64, chunk_rows, config)
  particle_index_outputs = _create_extendible_outputs(groups, 'particle_index', (), np.int64, chunk_rows, config)
  halo_array_outputs = None
  if mode == 'subhalo':
    halo_array_outputs = _create_extendible_outputs(groups, 'HaloID_array', (membership_depth_width(halo_id_array),), np.int32, chunk_rows, config)

  halo_selector = membership_selected_exclusive_ids if mode == 'subhalo' else membership_selected_top_ids
  timings = {name: 0.0 for name, _ in source_specs}
  timings['HaloID'] = 0.0
  timings['particle_index'] = 0.0
  if mode == 'subhalo':
    timings['HaloID_array'] = 0.0
  routing_time = 0.0

  for start in range(0, n_particles, chunk_rows):
    end = min(start + chunk_rows, n_particles)
    t = perf_counter()
    selected_rows, rows_by_rank, selected_positions_by_rank = _chunk_rank_rows(rank_ids, start, end, nsplit)
    routing_time += perf_counter() - t
    if len(selected_rows) == 0:
      continue

    global_rows = selected_rows.astype(np.int64, copy=False) + start

    t = perf_counter()
    halo_values = halo_selector(halo_id_array, global_rows)
    _append_selected_values_by_rank(halo_outputs, halo_values, selected_positions_by_rank)
    for rank, positions in enumerate(selected_positions_by_rank):
      if len(positions):
        rank_values = halo_values[positions]
        rank_halo_ids[rank].update(int(value) for value in np.unique(rank_values) if value >= 0)
    timings['HaloID'] += perf_counter() - t

    if halo_array_outputs is not None:
      t = perf_counter()
      halo_array_values = membership_selected_particles_dense(halo_id_array, global_rows)
      _append_selected_values_by_rank(halo_array_outputs, halo_array_values, selected_positions_by_rank)
      timings['HaloID_array'] += perf_counter() - t

    t = perf_counter()
    for rank, rows in enumerate(rows_by_rank):
      if len(rows) == 0:
        continue
      _append_values(particle_index_outputs[rank], rows.astype(np.int64, copy=False) + start)
    timings['particle_index'] += perf_counter() - t

    for name, dataset in source_specs:
      t = perf_counter()
      data_chunk = dataset[start:end]
      _append_chunk_values_by_rank(source_outputs[name], data_chunk, rows_by_rank)
      timings[name] += perf_counter() - t

  print(f'{ptype} routing: {routing_time:.1f}s', flush=True)
  for name in list(source_outputs) + ['HaloID', 'particle_index'] + (['HaloID_array'] if mode == 'subhalo' else []):
    print(f'{ptype} {name}: {timings[name]:.1f}s', flush=True)

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
    rank_ids = membership_rank_ids(halo_id_array, rank_lookup, dtype=rank_dtype)

    out_files = [h5py.File(f'{outfile}_{i}.hdf5', 'a') for i in range(nsplit)]
    try:
      groups = [f_out.require_group(ptype) for f_out in out_files]
      _stage_ptype_by_row_chunks(f, ptype, groups, halo_id_array, rank_ids, config, nsplit, mode, rank_halo_ids)
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
