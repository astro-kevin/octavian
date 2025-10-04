from __future__ import annotations
import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
import octavian.constants as c
from joblib import Parallel, delayed
from tqdm.auto import tqdm

try:
  import polars as pl  # type: ignore
  HAS_POLARS = True
except Exception:  # pragma: no cover - optional dependency
  pl = None  # type: ignore
  HAS_POLARS = False

from octavian.ahf import tqdm_joblib

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

# get mis for fof6d
def get_mean_interparticle_separation(data_manager: DataManager) -> None:
  t = data_manager.simulation['time']
  a = data_manager.simulation['a']
  h = data_manager.simulation['h']
  Om = data_manager.simulation['O0']
  boxsize = data_manager.simulation['boxsize']

  GRAV = unyt.G.to('cm**3/(g*s**2)').d
  UL = (1. * unyt.kpc).to('cm').d
  UM = data_manager.create_unit_quantity('mass').to('g').d
  UT = t/a

  G = GRAV / UL**3 * UM * UT**2
  Hubble = 3.2407789e-18 * UT

  dmmass = data_manager.mdm_total
  ndm = data_manager.ndm

  gmass = data_manager.mgas_total
  smass = data_manager.mstar_total
  bhmass = data_manager.mbh_total

  bmass = gmass + smass + bhmass

  Ob = bmass / (bmass + dmmass) * Om
  rhodm = (Om - Ob) * 3.0 * Hubble**2 / (8.0 * np.pi * G) / h

  mis = ((dmmass / ndm / rhodm)**(1./3.))/h
  efres = int(boxsize/h/mis)

  data_manager.mis = mis.d
  data_manager.efres = efres
  data_manager.Ob = Ob


# initial assignment of galaxy ids through sorting in x,y,z directions
def fof_sort_halo(halo: pd.DataFrame, minstars: int, fof_LL: float) -> pd.DataFrame:
  for direction in ['x', 'y', 'z']:
    halo = halo.sort_values(by=['GalID', direction])
    halo['distance'] = np.diff(halo[direction], prepend=halo[direction].iloc[0])
    halo['GalID'] += np.cumsum(halo['distance'] > fof_LL)

  halo = halo.groupby(by='GalID').filter(lambda group: len(group) >= minstars)

  return halo


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


# fof6d function to apply on groups
def run_fof6d_in_halo(halo: pd.DataFrame, kernel_table: np.ndarray, minstars: int, fof_LL: float, vel_LL: Optional[float] = None) -> list[list[tuple[int, pd.Index]]]:
  if len(halo) < minstars:
    return []

  # stage 1: directional group find
  halo = fof_sort_halo(halo, minstars, fof_LL)
  groups = [halo.loc[halo['GalID'] == id] for id in halo['GalID'].unique()]

  if len(groups) == 0:
    return []

  # skip stage 2 if vel_LL not defined, all members of a group form a galaxy
  if vel_LL is None:
    galaxies = [[(i, group_ptype.index) for i, group_ptype in group.groupby(by='ptype')] for group in groups]
    return galaxies

  # stage 2: fof6d
  new_groups = []
  for group in groups:
    pos = group[['x', 'y', 'z']].to_numpy()
    neighbors = NearestNeighbors(radius=fof_LL)
    neighbors.fit(pos)
    neighborDistances_lists, index_lists = neighbors.radius_neighbors(pos)

    qlists = neighborDistances_lists/fof_LL
    weights = [kernel(qlist, kernel_table) for qlist in qlists]

    vel = group[['vx', 'vy', 'vz']].to_numpy()
    dvs = [np.linalg.norm(vel[index_list] - vel[i], axis=1) for i, index_list in enumerate(index_lists)]

    sigmas = [np.sqrt(np.sum(weights_i*dvs_i**2)) for weights_i, dvs_i in zip(weights, dvs)]

    # this is a graph with defined directional connections from each node (including to self)
    # galaxies = groups formed by disjoint subsets of all points, with at least a one-directional path
    valid_neighbor_index_lists = [set(index_list_i[dvs_i <= (vel_LL*sigma)]) for index_list_i, dvs_i, sigma in zip(index_lists, dvs, sigmas)]

    valid_neighbor_index_lists = [each for each in valid_neighbor_index_lists if len(each) > 1]
    if len(valid_neighbor_index_lists) == 0: continue

    group_galaxies_indexes = [valid_neighbor_index_lists[0]]
    while len(valid_neighbor_index_lists) != 0:
      current_indexes = valid_neighbor_index_lists.pop(0)
      if len(current_indexes) == 0: continue

      merge_with = -1
      merged = False
      for i, galaxy in enumerate(group_galaxies_indexes):
        if current_indexes.isdisjoint(galaxy):
          continue
        elif merge_with == -1:
          galaxy |= current_indexes
          merge_with = i
          merged = True
        else:
          current_indexes |= group_galaxies_indexes.pop(i)
          group_galaxies_indexes[merge_with] |= galaxy

      if merged == False: group_galaxies_indexes.append(current_indexes)
    
    for galaxy in group_galaxies_indexes:
      if len(galaxy) < minstars:
        continue
      
      ordered_indexes = np.sort(list(galaxy))
      galaxy = group.iloc[ordered_indexes]
      if len(galaxy.loc[galaxy['ptype'] == 4]) >= minstars:
        new_groups.append(galaxy)

  galaxies = [[(i, group_ptype.index) for i, group_ptype in group.groupby(by='ptype')] for group in new_groups]
  return galaxies


# vectorised version of caesar fof6d
def run_fof6d(data_manager: DataManager, nproc: int = 1) -> None:
  get_mean_interparticle_separation(data_manager)

  b = 0.02
  fof_LL = data_manager.mis * b
  vel_LL = 1.

  use_polars = getattr(data_manager, 'use_polars', False)

  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('vel', ptype)

  # check dense
  for prop in ['rho', 'temperature', 'sfr']:
    data_manager.load_property(prop, 'gas')

  data_manager['gas']['temperature'] = 0.
  data_manager['gas']['dense_gas'] = (data_manager['gas']['rho'] > c.nHlim) & ((data_manager['gas']['temperature'] < c.Tlim) | (data_manager['gas']['sfr'] > 0))
  if use_polars:
    data_manager._invalidate_polars('gas')

  fof_columns = ['HaloID', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ptype']
  fof_columns_polars = ['pid'] + fof_columns
  fof_filter = lambda halo: len(halo) >= c.MINIMUM_STARS_PER_GALAXY

  if use_polars:
    def _ensure_polars_table(table):
      if HAS_POLARS and isinstance(table, pl.DataFrame):
        return table
      if HAS_POLARS and isinstance(table, pd.DataFrame):
        if 'pid' not in table.columns:
          table = table.reset_index().rename(columns={'index': 'pid'})
        return pl.from_pandas(table)
      raise TypeError('Polars backend requested but table is not a Polars DataFrame and conversion is unavailable.')

    star_table = _ensure_polars_table(data_manager.get_polars_table('star'))
    gas_table = _ensure_polars_table(data_manager.get_polars_table('gas'))
    bh_table = _ensure_polars_table(data_manager.get_polars_table('bh'))

    star_counts = star_table.groupby('HaloID').agg(pl.count().alias('count'))
    valid_haloids = (
      star_counts
      .filter(pl.col('count') >= c.MINIMUM_STARS_PER_GALAXY)
      .select('HaloID')
      .to_series()
      .to_numpy()
    )

    if valid_haloids.size == 0:
      grouped = []
    else:
      dense_gas = gas_table.filter(pl.col('dense_gas')).select(fof_columns_polars)
      star_subset = star_table.select(fof_columns_polars)
      bh_subset = bh_table.select(fof_columns_polars)
      fof_halos_pl = pl.concat([dense_gas, star_subset, bh_subset], how='vertical_relaxed')
      fof_halos_pl = fof_halos_pl.filter(pl.col('HaloID').is_in(valid_haloids))

      if fof_halos_pl.is_empty():
        grouped = []
      else:
        fof_halos_pl = fof_halos_pl.with_columns(pl.lit(0).alias('GalID'))
        halo_partitions = fof_halos_pl.partition_by('HaloID', maintain_order=True, as_dict=False)
        grouped = []
        for part in halo_partitions:
          halo_id_value = int(part['HaloID'][0]) if part.height > 0 else None
          grouped.append((halo_id_value, part.to_pandas().set_index('pid')))
  else:
    fof_halos = data_manager['star'].groupby('HaloID').filter(fof_filter)
    fof_haloids = np.unique(fof_halos['HaloID'])
    fof_halos = pd.concat([
      data_manager['gas'].loc[data_manager['gas']['dense_gas'], fof_columns],
      data_manager['star'][fof_columns],
      data_manager['bh'][fof_columns]
    ]).query('HaloID in @fof_haloids')

    fof_halos['GalID'] = 0
    grouped = list(fof_halos.groupby(by='HaloID'))

  kernel_table = create_kernel_table(fof_LL)

  backend = 'loky'
  if len(grouped) == 0:
    galaxies = []
  else:
    with tqdm_joblib(tqdm(total=len(grouped), desc='FoF6D halos', unit='halo', leave=False)):
      galaxies = Parallel(n_jobs=nproc, backend=backend)(
        delayed(run_fof6d_in_halo)(halo_df,
                                   kernel_table,
                                   c.MINIMUM_STARS_PER_GALAXY,
                                   fof_LL,
                                   vel_LL)
        for _, halo_df in grouped
      )
  galaxies = [galaxy for galaxy_list in galaxies for galaxy in galaxy_list if len(galaxy_list) != 0]

  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager[ptype]['GalID'] = -1
    if use_polars:
      data_manager._invalidate_polars(ptype)

  for i, galaxy in enumerate(galaxies):
    for ptype_id, ptype_indexes in galaxy:
      data_manager[c.ptype_names[ptype_id]].loc[ptype_indexes, 'GalID'] = i

  data_manager.galaxies = pd.DataFrame(index=np.arange(len(galaxies)))
