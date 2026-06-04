from __future__ import annotations
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import numpy as np
import pandas as pd
import unyt
from joblib import Parallel, delayed

from time import perf_counter

from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def _global_dm_particle_count(data_manager: 'DataManager') -> int:
  counts = data_manager.simulation.get('num_part_total')
  if counts is None or len(counts) <= 1:
    return 0
  return int(counts[1])


def _assign_no_galaxies(data_manager: 'DataManager', config: dict) -> None:
  for ptype in config['ptypes']:
    data_manager.data[ptype]['GalID'] = pd.Series(
      np.full(len(data_manager.data[ptype]), -1, dtype=np.int64),
      index=data_manager.data[ptype].index,
      dtype='category',
    )
  config['groups'] = [group for group in config.get('groups', ['halos']) if group != 'galaxies'] or ['halos']


# get mis for fof6d
def get_mean_interparticle_separation(data_manager: 'DataManager') -> None:
  h = data_manager.simulation['h']
  Om = data_manager.simulation['O0']
  boxsize = data_manager.simulation['boxsize']

  dmmass = data_manager.mdm_total
  ndm = data_manager.ndm

  gmass = data_manager.mgas_total
  smass = data_manager.mstar_total
  bhmass = data_manager.mbh_total

  bmass = gmass + smass + bhmass
  total_mass = bmass + dmmass
  data_manager.Ob = bmass / total_mass * Om if total_mass > 0 else np.nan

  ndm_global = _global_dm_particle_count(data_manager)
  if ndm_global > 0:
    efres = ndm_global ** (1. / 3.)
    data_manager.mis = boxsize / h / efres
    data_manager.efres = int(round(efres))
    return

  if ndm <= 0 or dmmass <= 0:
    raise ValueError('Cannot determine FOF6D mean interparticle separation without global or local DM particles')

  t = data_manager.simulation['time']
  a = data_manager.simulation['a']

  GRAV = unyt.G.to('cm**3/(g*s**2)').d
  UL = (1. * unyt.kpc).to('cm').d
  UM = data_manager.create_unit_quantity('mass').to('g').d
  UT = t/a

  G = GRAV / UL**3 * UM * UT**2
  Hubble = 3.2407789e-18 * UT
  rhodm = (Om - data_manager.Ob) * 3.0 * Hubble**2 / (8.0 * np.pi * G) / h

  mis = ((dmmass / ndm / rhodm)**(1./3.))/h
  data_manager.mis = mis
  data_manager.efres = int(boxsize/h/mis)


# initial assignment of galaxy ids through sorting in x,y,z directions
def fof_sort_halo(pos, vel, ptype, original_idx, minstars, fof_LL):
    n = len(pos)
    gal_ids = np.zeros(n, dtype=np.int32)

    for dim in range(3):
        order = np.lexsort((pos[:, dim], gal_ids))
        pos = pos[order]
        vel = vel[order]
        ptype = ptype[order]
        original_idx = original_idx[order]
        gal_ids = gal_ids[order]

        diffs = np.diff(pos[:, dim], prepend=pos[0, dim])
        gal_ids += np.cumsum(diffs > fof_LL).astype(np.int32)

    # filter small groups
    unique, counts = np.unique(gal_ids, return_counts=True)
    valid = np.isin(gal_ids, unique[counts >= minstars])

    return pos[valid], vel[valid], ptype[valid], original_idx[valid], gal_ids[valid]

#
# helper functions
#

# kernel table for fof6d velocity criterion distance weights
def create_kernel_table(fof_LL,ntab=1000):
    kerneltab = np.zeros(ntab+1)
    hinv = 1./fof_LL
    for i in range(ntab):
        r = 1. * i / ntab
        q = 2 * r * hinv
        if q > 2: kerneltab[i] = 0.0
        elif q > 1: kerneltab[i] = 0.25 * (2 - q)**3
        else: kerneltab[i] = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return kerneltab

# kernel table lookup
def kernel(r_over_h,kerneltab):
    ntab = len(kerneltab) - 1
    rtab = ntab * r_over_h + 0.5
    itab = rtab.astype(int)
    return kerneltab[itab]

#
# fof6d functions
#

# fof6d function to apply on groups
def run_fof6d_in_halo(
    pos, vel, ptype, original_idx,
    kernel_table, minstars, fof_LL, vel_LL=None
):
  
  timings = {'n_particles': len(pos)}
  if len(pos) < minstars:
      return [], timings

  t0 = perf_counter()

  # stage 1: directional group find
  pos, vel, ptype, original_idx, gal_ids = fof_sort_halo(
      pos, vel, ptype, original_idx, minstars, fof_LL
  )
  timings['sort'] = perf_counter() - t0

  if len(pos) == 0:
    return [], timings
  
  # split into groups
  group_order = np.argsort(gal_ids, kind='stable')
  sorted_gal_ids = gal_ids[group_order]
  splits = np.flatnonzero(np.diff(sorted_gal_ids)) + 1
  group_indices = np.split(group_order, splits)

  # skip stage 2 if vel_LL not defined, all members of a group form a galaxy
  if vel_LL is None:
      galaxies = []
      for indices in group_indices:
          g_ptype = ptype[indices]
          g_idx = original_idx[indices]
          galaxy = []
          for pt in np.unique(g_ptype):
              mask = g_ptype == pt
              galaxy.append((pt, g_idx[mask]))
          galaxies.append(galaxy)
      return galaxies, timings
  

  #
  # stage 2: fof6d
  #
  # updated vectorised loop implementation (JP)
  #

  # diagnostics
  t0 = perf_counter() # reset the clock
  t_neighbors_tot = 0
  t_weights_tot = 0
  t_merge_tot = 0

  new_groups = []
  for indices in group_indices:
    g_pos = pos[indices]
    g_vel = vel[indices]
    g_ptype = ptype[indices]
    g_idx = original_idx[indices]
    n = len(indices)
    # NOTE: everything cast to KDTree with scipy functionality
    t1 = perf_counter()
    tree = KDTree(g_pos)
    sdm = tree.sparse_distance_matrix(tree, fof_LL, output_type='coo_matrix')
    t2 = perf_counter()
    t_neighbors_tot += t2 - t1 # profiling: nearest neighbour speeds

    # NOTE: new algorithm using a sparse matrix implementation to avoid expensive python loops and pass code into scipy's C
    # vectorised COO (COOrdinate) construction (scipy recommended)
    # meet the definition of a sparse matrix
    # row[i], col[i] = value[i]
    rows = sdm.row
    cols = sdm.col
    dists = sdm.data

    # vectorised kernel weights (adapted from Jakub)
    q = dists / fof_LL
    w = kernel(q, kernel_table)  # already works on arrays

    # vectorised velocity differences
    vel_diff = np.linalg.norm(g_vel[cols] - g_vel[rows], axis=1)

    # vectorised sigma per particle
    weighted_dv_sq = w * vel_diff**2 # same as Jakub (I renamed variables for readability)
    sigmas = np.sqrt(np.bincount(rows, weights=weighted_dv_sq, minlength=n)) 
    t3 = perf_counter()
    t_weights_tot += t3 - t2 # profiling: weighting 

    # vectorised velocity criterion
    valid = vel_diff <= (vel_LL * sigmas[rows])

    # build sparse matrix from valid edges only
    # REVIEW: csr matrices
    # https://stackoverflow.com/questions/11016256/connected-components-in-a-graph-with-100-million-nodes (the syntax has changed slightly with new scipy versions)
    adj = csr_matrix((np.ones(valid.sum()), (rows[valid], cols[valid])), shape=(n, n)) # np.ones matrix; boolean mask with rows, cols
    n_components, labels = connected_components(adj, directed=False) # directed=False means we only care about connections (preserves original logic)

    t4 = perf_counter()
    t_merge_tot += t4 - t3 # profiling: merging (big python loop no more)


    # split by label — numpy instead of python loop
    label_order = np.argsort(labels)
    sorted_labels = labels[label_order]
    label_splits = np.flatnonzero(np.diff(sorted_labels)) + 1
    component_groups = np.split(label_order, label_splits)

    for component in component_groups:
      if len(component) < minstars:
          continue
      c_ptype = g_ptype[component]
      if np.sum(c_ptype == 'star') >= minstars:
          new_groups.append((g_ptype[component], g_idx[component]))

  # profiling: timings
  timings['neighbors'] = t_neighbors_tot
  timings['weights'] = t_weights_tot
  timings['merge'] = t_merge_tot

  # unavoidable python loop
  galaxies = []
  for g_ptype, g_idx in new_groups:
      galaxy = []
      for pt in np.unique(g_ptype):
          mask = g_ptype == pt
          galaxy.append((pt, g_idx[mask]))
      galaxies.append(galaxy)

  return galaxies, timings

# vectorised version of caesar fof6d
def run_fof6d(data_manager: DataManager, nproc: int = 1) -> None:
  data_manager.logger.info(f'Running FOF6D...')
  t1 = perf_counter()

  config = data_manager.config

  t2 = perf_counter()
  for ptype in config['ptypes']:
    if ptype == 'bh':
      data_manager.load_property('bhmass', ptype)
    else:
      data_manager.load_property('mass', ptype)
  t3 = perf_counter()

  if 'star' not in config['ptypes'] or len(data_manager.data['star']) == 0:
    _assign_no_galaxies(data_manager, config)
    data_manager.logger.info('Skipping FOF6D; no star particles available.')
    return

  data_manager.mdm_total = 0. if 'dm' not in config['ptypes'] else np.sum(data_manager.data['dm']['mass'])
  data_manager.ndm = 0 if 'dm' not in config['ptypes'] else len(data_manager.data['dm'])

  data_manager.mgas_total = 0. if 'gas' not in config['ptypes'] else np.sum(data_manager.data['gas']['mass'])
  data_manager.mstar_total = 0. if 'star' not in config['ptypes'] else np.sum(data_manager.data['star']['mass'])
  data_manager.mbh_total = 0. if 'bh' not in config['ptypes'] else np.sum(data_manager.data['bh']['mass'])

  get_mean_interparticle_separation(data_manager)

  b = 0.02
  fof_LL = data_manager.mis * b
  vel_LL = 1.

  t4 = perf_counter()
  for ptype in config['ptypes']:
    data_manager.load_property('vel', ptype)

  # check dense
  for prop in ['rho', 'temperature', 'sfr']:
    data_manager.load_property(prop, 'gas')
  t5 = perf_counter()

  data_manager.data['gas']['temperature'] = 0.
  data_manager.data['gas']['dense_gas'] = (data_manager.data['gas']['rho'] > config['nHlim']) & ((data_manager.data['gas']['temperature'] < config['Tlim']) | (data_manager.data['gas']['sfr'] > 0))
  
  # combine dfs, reduce the gas df to common columns
  fof_columns = ['HaloID', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ptype']
  fof_filter = lambda halo: len(halo) >= config['MINIMUM_STARS_PER_GALAXY']
  fof_halos = data_manager.data['star'].groupby('HaloID', observed=True).filter(fof_filter)
  fof_haloids = np.unique(fof_halos['HaloID']) 

  if 'bh' in config['ptypes']:
    fof_halos = pd.concat([data_manager.data['gas'].loc[data_manager.data['gas']['dense_gas'], fof_columns], data_manager.data['star'][fof_columns], data_manager.data['bh'][fof_columns]]).query('HaloID in @fof_haloids')
  else:
    fof_halos = pd.concat([data_manager.data['gas'].loc[data_manager.data['gas']['dense_gas'], fof_columns], data_manager.data['star'][fof_columns]]).query('HaloID in @fof_haloids')

  fof_halos['GalID'] = 0
  kernel_table = create_kernel_table(fof_LL)
  grouped = fof_halos.groupby(by='HaloID', observed=True)

  work_items = []
  halo_ids = []
  for halo_id, halo_df in grouped:
      work_items.append((
          halo_df[['x', 'y', 'z']].to_numpy(),
          halo_df[['vx', 'vy', 'vz']].to_numpy(),
          halo_df['ptype'].to_numpy(),
          halo_df.index.to_numpy(),
      ))
      halo_ids.append(halo_id)

  # sort largest first — memory-aware scheduling
  order = sorted(range(len(work_items)), key=lambda i: len(work_items[i][0]), reverse=True)
  work_items = [work_items[i] for i in order]
  halo_ids = [halo_ids[i] for i in order]

  results = Parallel(n_jobs=nproc, pre_dispatch='2*n_jobs', batch_size=1)(
      delayed(run_fof6d_in_halo)(
          pos, vel, ptype, idx,
          kernel_table, config['MINIMUM_STARS_PER_GALAXY'], fof_LL, vel_LL
      )
      for pos, vel, ptype, idx in work_items
  )

  # unpack results (same logic as before, just reordered)
  galaxies = [g for gals, _ in results for g in gals if len(gals) != 0]

  for ptype in config['ptypes']:
    data_manager.data[ptype]['GalID'] = -1
  
  for i, galaxy in enumerate(galaxies):
    for ptype, ptype_indexes in galaxy:
      data_manager.data[ptype].loc[ptype_indexes, 'GalID'] = i

  for ptype in config['ptypes']:
    data_manager.data[ptype]['GalID'] = data_manager.data[ptype]['GalID'].astype('category')

  if np.all(data_manager.data['star']['GalID'] == -1): config['groups'] = ['halos']

  t6 = perf_counter()
  data_manager.logger.info(f'Running FOF6D done in {t6-t1:.2f} seconds. (Load reduced: {t6-t5 + t4-t3 + t2-t1:.2f} seconds)')

  all_timings = [t for _, t in results]
  if all_timings:
    timings_df = pd.DataFrame(all_timings)
    timing_columns = ['sort', 'neighbors', 'weights', 'merge']
    for column in timing_columns:
      if column not in timings_df:
        timings_df[column] = 0.
    data_manager.logger.info(f'\n=== FOF6D Timing Summary (rank {data_manager.rank}) ===')
    data_manager.logger.info(f'Halos processed: {len(timings_df)}')
    data_manager.logger.info(f"Total particles: {timings_df['n_particles'].sum()}")
    data_manager.logger.info(timings_df[timing_columns].sum().to_string())
    data_manager.logger.info(f'\nTop 5 halos by total time:')
    timings_df['total'] = timings_df[timing_columns].sum(axis=1)
    data_manager.logger.info(timings_df.nlargest(5, 'total').to_string())
    data_manager.logger.info('=' * 40)
