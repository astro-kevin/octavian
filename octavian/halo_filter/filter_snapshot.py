import h5py
import numpy as np
from time import perf_counter
from yaml import safe_load

from octavian.halo_reader import (
  build_snapshot_membership_arrays,
  membership_array_exclusive_ids,
  membership_particle_count,
  membership_selected_particles_dense,
  membership_top_ids,
  prune_halo_tree,
  update_rank_halo_ids_from_membership,
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

def filter_snapshot_with_membership_arrays(f: h5py.File, outfile: str, config: dict, nsplit: int, membership_arrays: dict[str, np.ndarray], mode: str, tree=None):
  for i in range(nsplit):
    with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
      f.copy(f['Header'], f_out, 'Header')

  ptypes = list(membership_arrays)
  star_weights, gas_weights, dm_weights = {}, {}, {}
  for ptype_name, weight_dict in [('PartType4', star_weights), ('PartType0', gas_weights), ('PartType1', dm_weights)]:
    if ptype_name not in membership_arrays:
      continue
    top_ids = membership_top_ids(membership_arrays[ptype_name])
    unique, counts = np.unique(top_ids[top_ids >= 0], return_counts=True)
    for hid, count in zip(unique, counts):
      weight_dict[hid] = count

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

  max_halo_id = max(weights) if weights else -1
  rank_lookup = np.full(max_halo_id + 1, -1, dtype=np.int16)
  for rank, halo_ids in enumerate(rank_assignments):
    if halo_ids:
      rank_lookup[np.fromiter(halo_ids, dtype=np.int64)] = rank

  rank_halo_ids = [set() for _ in range(nsplit)]

  for ptype in ptypes:
    halo_id_array = membership_arrays[ptype]
    ids = membership_top_ids(halo_id_array)
    halo_ids = membership_array_exclusive_ids(halo_id_array) if mode == 'subhalo' else ids
    particle_index = np.arange(membership_particle_count(halo_id_array), dtype='int')
    assigned = np.zeros(len(ids), dtype=bool)
    in_lookup = (ids >= 0) & (ids < len(rank_lookup))
    assigned[in_lookup] = rank_lookup[ids[in_lookup]] >= 0
    rank_ids = rank_lookup[ids[assigned]]
    assigned_indices = np.flatnonzero(assigned)
    datasets = [dataset for dataset in f[ptype].keys() if dataset not in ('HaloID', 'HaloID_array', 'particle_index')]
    datasets += ['HaloID', 'particle_index']
    if mode == 'subhalo':
      datasets.append('HaloID_array')
    rank_masks = [rank_ids == i for i in range(nsplit)]

    rank_for_particle = np.full(len(ids), -1, dtype=np.int16)
    rank_for_particle[assigned] = rank_ids
    if mode == 'subhalo':
      update_rank_halo_ids_from_membership(rank_halo_ids, halo_id_array, rank_for_particle)
    else:
      staged_halo_ids = halo_ids[assigned]
      for i, rank_mask in enumerate(rank_masks):
        if not np.any(rank_mask):
          continue
        values = np.unique(staged_halo_ids[rank_mask])
        rank_halo_ids[i].update(int(value) for value in values if value >= 0)

    out_files = [h5py.File(f'{outfile}_{i}.hdf5', 'a') for i in range(nsplit)]
    try:
      for dataset in datasets:
        print(ptype, dataset, flush=True)
        if dataset == 'HaloID':
          data = halo_ids[assigned]
          for i, f_out in enumerate(out_files):
            f_out.require_group(ptype)
            f_out[ptype][dataset] = data[rank_masks[i]]
        elif dataset == 'HaloID_array':
          for i, f_out in enumerate(out_files):
            f_out.require_group(ptype)
            particle_rows = assigned_indices[rank_masks[i]]
            f_out[ptype][dataset] = membership_selected_particles_dense(halo_id_array, particle_rows)
        elif dataset == 'particle_index':
          data = particle_index[assigned]
          for i, f_out in enumerate(out_files):
            f_out.require_group(ptype)
            f_out[ptype][dataset] = data[rank_masks[i]]
        else:
          data = f[ptype][dataset][:][assigned]
          for i, f_out in enumerate(out_files):
            f_out.require_group(ptype)
            f_out[ptype][dataset] = data[rank_masks[i]]
    finally:
      for f_out in out_files:
        f_out.close()

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
