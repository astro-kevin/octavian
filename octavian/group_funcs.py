from __future__ import annotations
import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
import octavian.constants as c
from functools import partial
from astropy import constants as const
from tqdm.auto import tqdm

try:
  import polars as pl  # type: ignore
  HAS_POLARS = True
except Exception:  # pragma: no cover - optional dependency
  pl = None  # type: ignore
  HAS_POLARS = False

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager


# helper function to broadcast group properties to particle data, based on haloid
def broadcast_properties(data: pd.DataFrame, groupID: str, collection_data: pd.DataFrame, properties: list[str] | str) -> np.ndarray:
  return data[[groupID]].merge(collection_data[properties], left_on=groupID, right_index=True)[properties].to_numpy()


def calculateGroupProperties_common(data_manager: DataManager, data: pd.DataFrame, collection: str, group_name: str) -> None:
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)

  # nparticles
  collection_data[f'n{group_name}'] = halos_grouped.size()

  # group masses
  if group_name == 'bh':
    collection_data[f'mass_{group_name}'] = halos_grouped['mass'].max()
  else:
    collection_data[f'mass_{group_name}'] = halos_grouped['mass'].sum()

  # minpotpos, mipotvel
  if collection == 'halos' and group_name == 'total':
    minimum_potential_index = halos_grouped['potential'].idxmin()
    collection_data[['minpot_x', 'minpot_y', 'minpot_z', 'minpot_vx', 'minpot_vy', 'minpot_vz']] = data.loc[minimum_potential_index, ['HaloID', 'x', 'y', 'z', 'vx', 'vy', 'vz']].set_index('HaloID')

  # centre of mass, com velocity
  for column in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
    data['temp'] = data[column] * data['mass']
    collection_data[f'{column}_{group_name}'] = halos_grouped['temp'].sum()/collection_data[f'mass_{group_name}']

  data.drop(columns='temp', inplace=True)

  # galaxy aperture masses

  # velocity dispersion
  group_velocity_columns = [f'vx_{group_name}', f'vy_{group_name}', f'vz_{group_name}']
  data[['rel_vx', 'rel_vy', 'rel_vz']] = data[['vx', 'vy', 'vz']] - broadcast_properties(data, groupID, collection_data, group_velocity_columns)

  data[f'velocity_dispersion'] = np.sum(data[['rel_vx', 'rel_vy', 'rel_vz']]**2, axis=1)
  collection_data[f'velocity_dispersion_{group_name}'] = np.sqrt(halos_grouped['velocity_dispersion'].sum() / collection_data[f'n{group_name}'])

  data.drop(columns=['rel_vx', 'rel_vy', 'rel_vz', 'velocity_dispersion'], inplace=True)

  # radius
  group_position_columns = ['minpot_x', 'minpot_y', 'minpot_z'] if collection == 'halos' else [f'x_{group_name}', f'y_{group_name}', f'z_{group_name}']
                                                                                               
  data[['rel_x', 'rel_y', 'rel_z']] = data[['x', 'y', 'z']] - broadcast_properties(data, groupID, collection_data, group_position_columns)
  data.drop(columns=['x', 'y', 'z'], inplace=True)
  data['radius'] = np.linalg.norm(data[['rel_x', 'rel_y', 'rel_z']], axis=1)

  # angular momentum
  group_velocity_columns = ['minpot_vx', 'minpot_vy', 'minpot_vz'] if collection == 'halos' else [f'vx_{group_name}', f'vy_{group_name}', f'vz_{group_name}']
  data[['rel_vx', 'rel_vy', 'rel_vz']] = data[['vx', 'vy', 'vz']] - broadcast_properties(data, groupID, collection_data, group_velocity_columns)
  data['ktot'] = 0.5*data['mass']*(data['rel_vx']**2 + data['rel_vy']**2 + data['rel_vz']**2)
  data.drop(columns=['vx', 'vy', 'vz'], inplace=True)

  data[['rel_px', 'rel_py', 'rel_pz']] = data[['rel_vx', 'rel_vy', 'rel_vz']].multiply(data['mass'], axis='index')
  data.drop(columns=['rel_vx', 'rel_vy', 'rel_vz'], inplace=True)

  data[['Lx', 'Ly', 'Lz']] = np.cross(data[['rel_x', 'rel_y', 'rel_z']], data[['rel_px', 'rel_py', 'rel_pz']])
  for direction in ['x', 'y', 'z']:
    collection_data[f'L{direction}_{group_name}'] = halos_grouped[f'L{direction}'].sum()
  data.drop(columns=['rel_px', 'rel_py', 'rel_pz'], inplace=True)

  data[['Lx_group', 'Ly_group', 'Lz_group']] = broadcast_properties(data, groupID, collection_data, [f'Lx_{group_name}', f'Ly_{group_name}', f'Lz_{group_name}'])
  data['L_dot_L_group'] = (data[f'Lx'] * data['Lx_group'] + data[f'Ly'] * data['Ly_group'] + data[f'Lz'] * data['Lz_group'])
  data.drop(columns=['Lx', 'Ly', 'Lz'], inplace=True)

  collection_data[f'L_{group_name}'] = np.linalg.norm(collection_data[[f'Lx_{group_name}', f'Ly_{group_name}', f'Lz_{group_name}']], axis=1)
  collection_data[f'ALPHA_{group_name}'] = np.arctan2(collection_data[f'Ly_{group_name}'], collection_data[f'Lz_{group_name}'])
  collection_data[f'BETA_{group_name}'] = np.arcsin(collection_data[f'Lx_{group_name}'] / collection_data[f'L_{group_name}'])

  collection_data[f'BoverT_{group_name}'] = 2 * data.loc[data['L_dot_L_group'] < 0].groupby(by='HaloID')['mass'].sum() / collection_data[f'mass_{group_name}']

  # better name?
  data['rz'] = np.sqrt((data['rel_y'] * data['Lz_group'] - data['rel_z'] * data['Ly_group'])**2 + (data['rel_z'] * data['Lx_group'] - data['rel_x'] * data['Lz_group'])**2 + (data['rel_x'] * data['Ly_group'] - data['rel_y'] * data['Lx_group'])**2)
  data.drop(columns=['rel_x', 'rel_y', 'rel_z', 'Lx_group', 'Ly_group', 'Lz_group'], inplace=True)
  
  data['krot'] = 0.5*(data['L_dot_L_group']/data['rz'])**2/data['mass']

  ordered_rotation_grouped = data.loc[data['rz'] > 0, ['HaloID', 'krot', 'ktot']].groupby(by='HaloID')
  collection_data[f'kappa_rot_{group_name}'] = ordered_rotation_grouped['krot'].sum() / ordered_rotation_grouped['ktot'].sum()

  data.drop(columns=['L_dot_L_group', 'rz', 'krot', 'ktot'], inplace=True)

  angular_quantities = [f'velocity_dispersion_{group_name}', f'Lx_{group_name}', f'Ly_{group_name}', f'Lz_{group_name}', f'BoverT_{group_name}', f'kappa_rot_{group_name}']
  collection_data.loc[collection_data[f'n{group_name}'] < 3, angular_quantities] = 0.

  # radial quantities
  data.sort_values(by='radius', inplace=True)
  halos_grouped = data.groupby(by='HaloID')

  data['cumulative_mass'] = halos_grouped['mass'].cumsum()
  data['cumulative_mass_fraction'] = data['cumulative_mass'] / broadcast_properties(data, groupID, collection_data, f'mass_{group_name}')

  for quantile, col_name in zip([0.2, 0.5, 0.8], ['r20', 'half_mass', 'r80']):
    data.loc[data['cumulative_mass_fraction'] < quantile, 'cumulative_mass_fraction'] = np.nan
    minimum_cummass_index = halos_grouped['cumulative_mass_fraction'].idxmin(skipna=True)
    collection_data[f'radius_{group_name}_{col_name}'] = data.loc[minimum_cummass_index, [groupID, 'radius']].set_index(groupID)

  # virial quantities -> around minpotpos
  if collection == 'halos' and group_name == 'total':
    collection_data['r200'] = data_manager.simulation['r200_factor'] * (collection_data[f'mass_{group_name}'])**(1/3)
    collection_data['circular_velocity'] = np.sqrt(unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d * collection_data[f'mass_{group_name}'] / collection_data['r200'])
    collection_data['temperature'] = 3.6e5 * (collection_data['circular_velocity'] / 100.0)**2
    collection_data['spin_param'] = collection_data[f'L_{group_name}'] / (np.sqrt(2) * collection_data[f'mass_{group_name}'] * collection_data['circular_velocity'] * collection_data['r200'])

    volume_factor = 4./3.*np.pi
    rhocrit = (data_manager.simulation['rhocrit'] * data_manager.create_unit_quantity('rhocrit')).to('Msun / (kpc*a)**3').d
    data['overdensity'] = data['cumulative_mass'] / (volume_factor * data['radius']**3) / rhocrit

    for factor in [200, 500, 2500]:
      data.loc[data['overdensity'] < factor, ['radius', 'cumulative_mass']] = np.nan
      collection_data[f'radius_{factor}_c'] = halos_grouped['radius'].last()
      collection_data[f'mass_{factor}_c'] = halos_grouped['cumulative_mass'].last()


def calculateGroupProperties_gas(data_manager: DataManager, data: pd.DataFrame, collection: str) -> None:
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)
  
  data['fHI'] = data['nh'] / data['mass']

  # gas massses
  not_conserving_mass = data['fHI'] + data['fH2'] > 1.
  data.loc[not_conserving_mass, 'fHI'] = 1. - data.loc[not_conserving_mass, 'fH2']

  data['mass_HI'] = c.XH * data['fHI'] * data['mass']
  data['mass_H2'] = c.XH * data['fH2'] * data['mass']

  # remember HI, H2 for aperture masses
  if collection == 'halos':
    data_manager['gas']['mass_HI'] = data['mass_HI']
    data_manager['gas']['mass_H2'] = data['mass_H2']

  collection_data['mass_HI'] = halos_grouped['mass_HI'].sum()
  collection_data['mass_H2'] = halos_grouped['mass_H2'].sum()

  data.drop(columns=['nh', 'fHI', 'fH2', 'mass_HI', 'mass_H2'], inplace=True)
  
  # sfr
  collection_data['sfr'] = halos_grouped['sfr'].sum()

  # metallicity
  data['metallicity_mass_weighted'] = data['metallicity'] * data['mass']
  data['metallicity_sfr_weighted'] = data['metallicity'] * data['sfr']

  collection_data['metallicity_mass_weighted'] = halos_grouped['metallicity_mass_weighted'].sum() / collection_data['mass_gas']
  collection_data['metallicity_sfr_weighted'] = halos_grouped['metallicity_sfr_weighted'].sum() / collection_data['sfr']

  # cgm mass, temperatures
  data['temp_mass_weighted'] = data['temperature'] * data['mass']
  data['temp_metal_weighted'] = data['temperature'] * data['mass'] * data['metallicity']

  cgm_mask = data['rho'] < c.nHlim
  halos_cgm_grouped = data.loc[cgm_mask, ['HaloID', 'mass', 'temp_mass_weighted', 'temp_metal_weighted']].groupby(by='HaloID')
  collection_data['mass_cgm'] = halos_cgm_grouped['mass'].sum()

  collection_data['temp_mass_weighted'] = halos_grouped['temp_mass_weighted'].sum() / collection_data['mass_gas']
  collection_data['temp_mass_weighted_cgm'] = halos_cgm_grouped['temp_mass_weighted'].sum()

  collection_data['temp_metal_weighted_cgm'] = halos_cgm_grouped['temp_metal_weighted'].sum() / collection_data['temp_mass_weighted_cgm']
  collection_data['temp_mass_weighted_cgm'] /= collection_data['mass_cgm']

  data.drop(columns=['temp_mass_weighted'], inplace=True)

  # metallicities
  halos_cgm_grouped = data.loc[cgm_mask, ['HaloID', 'mass', 'metallicity_mass_weighted', 'metallicity_sfr_weighted', 'temp_metal_weighted']].groupby(by='HaloID')

  collection_data['metallicity_mass_weighted_cgm'] = halos_cgm_grouped['metallicity_mass_weighted'].sum()
  collection_data['metallicity_temp_weighted_cgm'] = halos_cgm_grouped['temp_metal_weighted'].sum() / collection_data['metallicity_mass_weighted_cgm']
  collection_data['metallicity_mass_weighted_cgm'] /= collection_data['mass_cgm']


def calculateGroupProperties_star(data_manager: DataManager, data: pd.DataFrame, collection: str) -> None:
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)

  # metallicity
  data['metallicity_stellar'] = data['metallicity'] * data['mass']

  collection_data['metallicity_stellar'] = halos_grouped['metallicity_stellar'].sum()
  # age
  data['age_mass_weighted'] = data['age'] * data['mass']
  data['age_metal_weighted'] = data['age'] * data['mass'] * data['metallicity']

  collection_data['age_mass_weighted'] = halos_grouped['age_mass_weighted'].sum() / collection_data['mass_star']
  collection_data['age_metal_weighted'] = halos_grouped['age_metal_weighted'].sum() / collection_data['metallicity_stellar']

  collection_data['metallicity_stellar'] /= halos_grouped['mass'].sum()


def calculateGroupProperties_bh(data_manager: DataManager, data: pd.DataFrame, collection: str) -> None:
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)

  max_mass_index = halos_grouped['mass'].idxmax()
  data = data.loc[max_mass_index].set_index(groupID)

  # bhmdot
  collection_data['bhmdot'] = data['bhmdot'].copy()

  # bh_fedd
  FRAD = 0.1  # assume 10% radiative efficiency
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value
  collection_data['bh_fedd'] = data['bhmdot'] / (edd_factor * data['mass'])


def calculate_aperture_masses(halo: pd.DataFrame, aperture: float, galaxy_positions: np.ndarray) -> pd.DataFrame:
  galaxy_ids = halo['GalID'].unique()
  galaxy_ids = galaxy_ids[galaxy_ids != -1]
  if len(galaxy_ids) == 0: return pd.DataFrame()

  data = []
  for GalID in galaxy_ids:
    relative_positions = halo[['x', 'y', 'z']] - galaxy_positions[GalID]
    halo['radius'] = np.linalg.norm(relative_positions, axis=1)

    masses = halo.loc[halo['radius'] < aperture].groupby(by='ptype')['mass'].sum()
    masses.name = GalID
    data.append(masses)

  return pd.DataFrame(data=data)


def calculate_local_densities(data_manager: DataManager) -> None:
  collections = ['halos', 'galaxies']

  for collection in collections:
    collection_data = data_manager[collection]
    pos = collection_data[['x_total', 'y_total', 'z_total']].to_numpy()
    mass = collection_data['mass_total'].to_numpy()

    neighbors = NearestNeighbors()
    neighbors.fit(pos)

    for radius in [300., 1000., 3000.]:
      volume = 4./3. * np.pi * radius**3

      df = pd.DataFrame({
        'indexes': neighbors.radius_neighbors(pos, radius=radius, return_distance=False)
      })

      df = df.explode('indexes').dropna()

      df['mass'] = mass[df['indexes'].astype('int')]
      grouped = df.groupby(level=0)

      collection_data[f'local_mass_density_{int(radius)}'] = grouped['mass'].sum() / volume
      collection_data[f'local_number_density_{int(radius)}'] = grouped.size() / volume


def _assign_polars_result(collection_data: pd.DataFrame, key: str, result_df) -> None:
  if result_df.is_empty():
    return
  result_pd = result_df.to_pandas()
  result_pd.set_index(key, inplace=True)
  aligned = result_pd.reindex(collection_data.index)
  for col in aligned.columns:
    collection_data[col] = aligned[col]


def _safe_divide(numerator, denominator, default: float = 0.0):
  return pl.when(denominator != 0).then(numerator / denominator).otherwise(default)


def _polars_quantile_radius(df, groupID: str, quantile: float, label: str):
  if df.is_empty():
    return pl.DataFrame({groupID: [], label: []})
  filtered = df.filter(pl.col('cumulative_mass_fraction') >= quantile)
  if filtered.is_empty():
    return pl.DataFrame({groupID: [], label: []})
  return filtered.groupby(groupID).agg(pl.col('radius').min().alias(label))


def _polars_common(df, groupID: str, group_name: str, use_minpot: bool, sim: dict):
  if df.is_empty():
    return pl.DataFrame({groupID: []})

  columns = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
  required_cols = [col for col in columns if col not in df.columns]
  if required_cols:
    raise KeyError(f"Missing columns in Polars DataFrame: {required_cols}")

  mass_sum_expr = pl.col('mass').sum().over(groupID)
  df = df.with_columns([
    mass_sum_expr.alias('_mass_sum'),
    (pl.col('mass') * pl.col('x')).sum().over(groupID).alias('_sum_mass_x'),
    (pl.col('mass') * pl.col('y')).sum().over(groupID).alias('_sum_mass_y'),
    (pl.col('mass') * pl.col('z')).sum().over(groupID).alias('_sum_mass_z'),
    (pl.col('mass') * pl.col('vx')).sum().over(groupID).alias('_sum_mass_vx'),
    (pl.col('mass') * pl.col('vy')).sum().over(groupID).alias('_sum_mass_vy'),
    (pl.col('mass') * pl.col('vz')).sum().over(groupID).alias('_sum_mass_vz')
  ])

  if use_minpot:
    df = df.with_columns([
      pl.col('x').sort_by(pl.col('potential')).first().over(groupID).alias('minpot_x'),
      pl.col('y').sort_by(pl.col('potential')).first().over(groupID).alias('minpot_y'),
      pl.col('z').sort_by(pl.col('potential')).first().over(groupID).alias('minpot_z'),
      pl.col('vx').sort_by(pl.col('potential')).first().over(groupID).alias('minpot_vx'),
      pl.col('vy').sort_by(pl.col('potential')).first().over(groupID).alias('minpot_vy'),
      pl.col('vz').sort_by(pl.col('potential')).first().over(groupID).alias('minpot_vz')
    ])

  df = df.with_columns([
    _safe_divide(pl.col('_sum_mass_x'), pl.col('_mass_sum')).alias('_x_mean'),
    _safe_divide(pl.col('_sum_mass_y'), pl.col('_mass_sum')).alias('_y_mean'),
    _safe_divide(pl.col('_sum_mass_z'), pl.col('_mass_sum')).alias('_z_mean'),
    _safe_divide(pl.col('_sum_mass_vx'), pl.col('_mass_sum')).alias('_vx_mean'),
    _safe_divide(pl.col('_sum_mass_vy'), pl.col('_mass_sum')).alias('_vy_mean'),
    _safe_divide(pl.col('_sum_mass_vz'), pl.col('_mass_sum')).alias('_vz_mean')
  ])

  if use_minpot:
    df = df.with_columns([
      pl.col('minpot_x').alias('_center_x'),
      pl.col('minpot_y').alias('_center_y'),
      pl.col('minpot_z').alias('_center_z'),
      pl.col('minpot_vx').alias('_center_vx'),
      pl.col('minpot_vy').alias('_center_vy'),
      pl.col('minpot_vz').alias('_center_vz')
    ])
  else:
    df = df.with_columns([
      pl.col('_x_mean').alias('_center_x'),
      pl.col('_y_mean').alias('_center_y'),
      pl.col('_z_mean').alias('_center_z'),
      pl.col('_vx_mean').alias('_center_vx'),
      pl.col('_vy_mean').alias('_center_vy'),
      pl.col('_vz_mean').alias('_center_vz')
    ])

  df = df.with_columns([
    (pl.col('x') - pl.col('_center_x')).alias('_rel_x'),
    (pl.col('y') - pl.col('_center_y')).alias('_rel_y'),
    (pl.col('z') - pl.col('_center_z')).alias('_rel_z'),
    (pl.col('vx') - pl.col('_center_vx')).alias('_rel_vx'),
    (pl.col('vy') - pl.col('_center_vy')).alias('_rel_vy'),
    (pl.col('vz') - pl.col('_center_vz')).alias('_rel_vz')
  ])

  df = df.with_columns([
    pl.sqrt(pl.col('_rel_x')**2 + pl.col('_rel_y')**2 + pl.col('_rel_z')**2).alias('radius'),
    (pl.col('_rel_vx')**2 + pl.col('_rel_vy')**2 + pl.col('_rel_vz')**2).alias('_vel_sq'),
    (0.5 * (pl.col('_rel_vx')**2 + pl.col('_rel_vy')**2 + pl.col('_rel_vz')**2) * pl.col('mass')).alias('_ktot'),
    (pl.col('_rel_vx') * pl.col('mass')).alias('_rel_px'),
    (pl.col('_rel_vy') * pl.col('mass')).alias('_rel_py'),
    (pl.col('_rel_vz') * pl.col('mass')).alias('_rel_pz')
  ])

  df = df.with_columns([
    (pl.col('_rel_y') * pl.col('_rel_pz') - pl.col('_rel_z') * pl.col('_rel_py')).alias('_Lx'),
    (pl.col('_rel_z') * pl.col('_rel_px') - pl.col('_rel_x') * pl.col('_rel_pz')).alias('_Ly'),
    (pl.col('_rel_x') * pl.col('_rel_py') - pl.col('_rel_y') * pl.col('_rel_px')).alias('_Lz')
  ])

  df = df.with_columns([
    pl.col('_Lx').sum().over(groupID).alias('_Lx_group'),
    pl.col('_Ly').sum().over(groupID).alias('_Ly_group'),
    pl.col('_Lz').sum().over(groupID).alias('_Lz_group')
  ])

  df = df.with_columns([
    pl.sqrt(pl.col('_Lx_group')**2 + pl.col('_Ly_group')**2 + pl.col('_Lz_group')**2).alias('_L_group_mag'),
    (pl.col('_Lx') * pl.col('_Lx_group') + pl.col('_Ly') * pl.col('_Ly_group') + pl.col('_Lz') * pl.col('_Lz_group')).alias('_L_dot'),
    pl.sqrt(
      (pl.col('_rel_y') * pl.col('_Lz_group') - pl.col('_rel_z') * pl.col('_Ly_group'))**2 +
      (pl.col('_rel_z') * pl.col('_Lx_group') - pl.col('_rel_x') * pl.col('_Lz_group'))**2 +
      (pl.col('_rel_x') * pl.col('_Ly_group') - pl.col('_rel_y') * pl.col('_Lx_group'))**2
    ).alias('_rz')
  ])

  df = df.with_columns([
    pl.when(pl.col('_rz') > 0).then(0.5 * (pl.col('_L_dot') / pl.col('_rz'))**2 / pl.col('mass')).otherwise(0.0).alias('_krot')
  ])

  df = df.sort([groupID, 'radius']).with_columns([
    pl.col('mass').cumsum().over(groupID).alias('_cumulative_mass'),
    _safe_divide(pl.col('_cumulative_mass'), pl.col('_mass_sum')).alias('cumulative_mass_fraction')
  ])

  agg = df.groupby(groupID).agg([
    pl.count().alias(f'n{group_name}'),
    pl.col('_mass_sum').first().alias('mass_sum'),
    pl.col('_x_mean').first().alias(f'x_{group_name}'),
    pl.col('_y_mean').first().alias(f'y_{group_name}'),
    pl.col('_z_mean').first().alias(f'z_{group_name}'),
    pl.col('_vx_mean').first().alias(f'vx_{group_name}'),
    pl.col('_vy_mean').first().alias(f'vy_{group_name}'),
    pl.col('_vz_mean').first().alias(f'vz_{group_name}'),
    pl.col('_Lx_group').first().alias(f'Lx_{group_name}'),
    pl.col('_Ly_group').first().alias(f'Ly_{group_name}'),
    pl.col('_Lz_group').first().alias(f'Lz_{group_name}'),
    pl.col('_L_group_mag').first().alias(f'L_{group_name}'),
    pl.col('_vel_sq').sum().alias('_sum_vel_sq'),
    pl.col('_krot').sum().alias('_sum_krot'),
    pl.col('_ktot').sum().alias('_sum_ktot'),
    pl.col('mass').filter(pl.col('_L_dot') < 0).sum().alias('_bt_mass')
  ])

  if use_minpot:
    agg_minpot = df.groupby(groupID).agg([
      pl.col('minpot_x').first().alias('minpot_x'),
      pl.col('minpot_y').first().alias('minpot_y'),
      pl.col('minpot_z').first().alias('minpot_z'),
      pl.col('minpot_vx').first().alias('minpot_vx'),
      pl.col('minpot_vy').first().alias('minpot_vy'),
      pl.col('minpot_vz').first().alias('minpot_vz')
    ])
    agg = agg.join(agg_minpot, on=groupID, how='left')

  radius_r20 = _polars_quantile_radius(df, groupID, 0.2, f'radius_{group_name}_r20')
  radius_half = _polars_quantile_radius(df, groupID, 0.5, f'radius_{group_name}_half_mass')
  radius_r80 = _polars_quantile_radius(df, groupID, 0.8, f'radius_{group_name}_r80')

  for radius_df in (radius_r20, radius_half, radius_r80):
    agg = agg.join(radius_df, on=groupID, how='left')

  agg = agg.with_columns([
    pl.col('mass_sum').alias(f'mass_{group_name}')
  ])

  agg = agg.with_columns([
    pl.sqrt(_safe_divide(pl.col('_sum_vel_sq'), pl.col(f'n{group_name}'))).alias(f'velocity_dispersion_{group_name}'),
    _safe_divide(2 * pl.col('_bt_mass'), pl.col(f'mass_{group_name}')).alias(f'BoverT_{group_name}'),
    _safe_divide(pl.col('_sum_krot'), pl.col('_sum_ktot')).alias(f'kappa_rot_{group_name}'),
    pl.when(pl.col(f'L_{group_name}') != 0)
      .then(pl.arctan2(pl.col(f'Ly_{group_name}'), pl.col(f'Lz_{group_name}')))
      .otherwise(0.0)
      .alias(f'ALPHA_{group_name}'),
    pl.when(pl.col(f'L_{group_name}') != 0)
      .then(pl.arcsin(_safe_divide(pl.col(f'Lx_{group_name}'), pl.col(f'L_{group_name}'))))
      .otherwise(0.0)
      .alias(f'BETA_{group_name}')
  ])

  if use_minpot and group_name == 'total':
    r200_factor = sim.get('r200_factor', 0.0)
    G_factor = unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d
    agg = agg.with_columns([
      (r200_factor * pl.col(f'mass_{group_name}')**(1/3)).alias('r200'),
      pl.sqrt(G_factor * pl.col(f'mass_{group_name}') / _safe_divide(pl.col('r200'), pl.lit(1.0))).alias('circular_velocity')
    ])
    agg = agg.with_columns([
      3.6e5 * (_safe_divide(pl.col('circular_velocity'), pl.lit(100.0))**2).alias('temperature'),
      _safe_divide(pl.col(f'L_{group_name}'), (np.sqrt(2) * pl.col(f'mass_{group_name}') * pl.col('circular_velocity') * pl.col('r200'))).alias('spin_param')
    ])

  agg = agg.drop(['mass_sum', '_sum_vel_sq', '_sum_krot', '_sum_ktot', '_bt_mass'])

  return agg


def _polars_filter_galaxies(df: pl.DataFrame) -> pl.DataFrame:
  if 'GalID' not in df.columns:
    return df
  return df.filter(pl.col('GalID') != -1)


def _apply_angular_threshold(collection_data: pd.DataFrame, group_name: str) -> None:
  angular_cols = [
    f'velocity_dispersion_{group_name}',
    f'Lx_{group_name}', f'Ly_{group_name}', f'Lz_{group_name}',
    f'L_{group_name}', f'BoverT_{group_name}', f'kappa_rot_{group_name}',
    f'ALPHA_{group_name}', f'BETA_{group_name}'
  ]
  existing = [col for col in angular_cols if col in collection_data]
  if not existing:
    return
  mask = collection_data[f'n{group_name}'] < 3
  collection_data.loc[mask, existing] = 0.


def _prepare_polars_gas_table(gas_df, data_manager: DataManager):
  if gas_df.is_empty():
    data_manager['gas']['mass_HI'] = 0.
    data_manager['gas']['mass_H2'] = 0.
    return gas_df

  df = gas_df.with_columns([
    _safe_divide(pl.col('nh'), pl.col('mass')).alias('fHI')
  ])

  df = df.with_columns([
    pl.when(pl.col('fHI') + pl.col('fH2') > 1.0)
      .then(1.0 - pl.col('fH2'))
      .otherwise(pl.col('fHI'))
      .alias('fHI')
  ])

  df = df.with_columns([
    (c.XH * pl.col('fHI') * pl.col('mass')).alias('mass_HI'),
    (c.XH * pl.col('fH2') * pl.col('mass')).alias('mass_H2'),
    (pl.col('metallicity') * pl.col('mass')).alias('_metallicity_mass_weighted'),
    (pl.col('metallicity') * pl.col('sfr')).alias('_metallicity_sfr_weighted'),
    (pl.col('temperature') * pl.col('mass')).alias('_temp_mass_weighted'),
    (pl.col('temperature') * pl.col('mass') * pl.col('metallicity')).alias('_temp_metal_weighted')
  ])

  gas_updates = df.select(['mass_HI', 'mass_H2']).to_pandas()
  pandas_gas = data_manager['gas'].copy()
  pandas_gas['mass_HI'] = gas_updates['mass_HI'].to_numpy()
  pandas_gas['mass_H2'] = gas_updates['mass_H2'].to_numpy()
  data_manager['gas'] = pandas_gas

  return df


def _polars_gas_aggregate(df, groupID: str) -> pl.DataFrame:
  if df.is_empty():
    return pl.DataFrame({groupID: []})

  agg = df.groupby(groupID).agg([
    pl.col('mass_HI').sum().alias('mass_HI'),
    pl.col('mass_H2').sum().alias('mass_H2'),
    pl.col('sfr').sum().alias('sfr'),
    pl.col('_metallicity_mass_weighted').sum().alias('_sum_met_mass'),
    pl.col('_metallicity_sfr_weighted').sum().alias('_sum_met_sfr'),
    pl.col('_temp_mass_weighted').sum().alias('_sum_temp_mass'),
    pl.col('_temp_metal_weighted').sum().alias('_sum_temp_metal'),
    pl.col('mass').sum().alias('mass_gas')
  ])

  cgm = df.filter(pl.col('rho') < c.nHlim)
  if not cgm.is_empty():
    cgm_agg = cgm.groupby(groupID).agg([
      pl.col('mass').sum().alias('mass_cgm'),
      pl.col('_temp_mass_weighted').sum().alias('_sum_temp_mass_cgm'),
      pl.col('_temp_metal_weighted').sum().alias('_sum_temp_metal_cgm'),
      pl.col('_metallicity_mass_weighted').sum().alias('_sum_met_mass_cgm'),
      pl.col('_metallicity_sfr_weighted').sum().alias('_sum_met_sfr_cgm')
    ])
    agg = agg.join(cgm_agg, on=groupID, how='left')
  else:
    agg = agg.with_columns([
      pl.lit(0.0).alias('mass_cgm'),
      pl.lit(0.0).alias('_sum_temp_mass_cgm'),
      pl.lit(0.0).alias('_sum_temp_metal_cgm'),
      pl.lit(0.0).alias('_sum_met_mass_cgm'),
      pl.lit(0.0).alias('_sum_met_sfr_cgm')
    ])

  agg = agg.with_columns([
    pl.col('mass_cgm').fill_null(0.0),
    pl.col('_sum_temp_mass_cgm').fill_null(0.0),
    pl.col('_sum_temp_metal_cgm').fill_null(0.0),
    pl.col('_sum_met_mass_cgm').fill_null(0.0),
    pl.col('_sum_met_sfr_cgm').fill_null(0.0)
  ])

  agg = agg.with_columns([
    _safe_divide(pl.col('_sum_met_mass'), pl.col('mass_gas')).alias('metallicity_mass_weighted'),
    _safe_divide(pl.col('_sum_met_sfr'), pl.col('sfr')).alias('metallicity_sfr_weighted'),
    _safe_divide(pl.col('_sum_temp_mass'), pl.col('mass_gas')).alias('temp_mass_weighted'),
    _safe_divide(pl.col('_sum_temp_mass_cgm'), pl.col('mass_cgm')).alias('temp_mass_weighted_cgm'),
    _safe_divide(pl.col('_sum_temp_metal_cgm'), pl.col('mass_cgm')).alias('temp_metal_weighted_cgm'),
    _safe_divide(pl.col('_sum_met_mass_cgm'), pl.col('mass_cgm')).alias('metallicity_mass_weighted_cgm'),
    _safe_divide(pl.col('_sum_met_sfr_cgm'), pl.col('mass_cgm')).alias('metallicity_temp_weighted_cgm')
  ])

  agg = agg.drop([
    '_sum_met_mass', '_sum_met_sfr', '_sum_temp_mass', '_sum_temp_metal',
    '_sum_temp_mass_cgm', '_sum_temp_metal_cgm', '_sum_met_mass_cgm', '_sum_met_sfr_cgm'
  ])

  return agg


def _polars_star(df, groupID: str) -> pl.DataFrame:
  if df.is_empty():
    return pl.DataFrame({groupID: []})

  df = df.with_columns([
    (pl.col('metallicity') * pl.col('mass')).alias('_sum_met_mass'),
    (pl.col('age') * pl.col('mass')).alias('_sum_age_mass'),
    (pl.col('age') * pl.col('mass') * pl.col('metallicity')).alias('_sum_age_metal')
  ])

  agg = df.groupby(groupID).agg([
    pl.col('_sum_met_mass').sum().alias('_sum_met_mass'),
    pl.col('_sum_age_mass').sum().alias('_sum_age_mass'),
    pl.col('_sum_age_metal').sum().alias('_sum_age_metal'),
    pl.col('mass').sum().alias('mass_star')
  ])

  agg = agg.with_columns([
    _safe_divide(pl.col('_sum_met_mass'), pl.col('mass_star')).alias('metallicity_stellar'),
    _safe_divide(pl.col('_sum_age_mass'), pl.col('mass_star')).alias('age_mass_weighted'),
    _safe_divide(pl.col('_sum_age_metal'), pl.col('_sum_met_mass')).alias('age_metal_weighted')
  ])

  agg = agg.drop(['_sum_met_mass', '_sum_age_mass', '_sum_age_metal'])

  return agg


def _polars_bh(df, groupID: str) -> pl.DataFrame:
  if df.is_empty():
    return pl.DataFrame({groupID: []})

  df_sorted = df.sort([groupID, 'mass'])
  agg = df_sorted.groupby(groupID).agg([
    pl.col('bhmdot').last().alias('bhmdot'),
    pl.col('mass').last().alias('_bh_mass')
  ])

  FRAD = 0.1
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value
  agg = agg.with_columns([
    _safe_divide(pl.col('bhmdot'), edd_factor * pl.col('_bh_mass')).alias('bh_fedd')
  ])

  agg = agg.drop(['_bh_mass'])
  return agg


def calculate_group_properties_polars(data_manager: DataManager, include_global: bool = True) -> None:
  if not HAS_POLARS:
    raise RuntimeError('Polars is not available but Polars execution was requested.')

  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pot', ptype)

  collections = ['halos', 'galaxies']
  labels = _build_progress_labels(collections)
  labels_iter = iter(labels)
  progress = tqdm(total=len(labels), desc='Computing group properties', unit='task', leave=False)

  group_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potential']

  polars_tables = {}
  for ptype in c.ptypes.keys():
    table = data_manager.get_polars_table(ptype, include_index=False)
    polars_tables[ptype] = table.select(group_props_columns)

  simulation = data_manager.simulation

  halos_df = data_manager['halos'].copy()
  galaxies_df = data_manager['galaxies'].copy()

  dr = pl.concat([polars_tables[ptype] for ptype in c.ptypes.keys()], how='vertical_relaxed')
  all_particles = dr

  halos_common = _polars_common(all_particles, 'HaloID', 'total', True, simulation)
  _assign_polars_result(halos_df, 'HaloID', halos_common)
  _apply_angular_threshold(halos_df, 'total')
  _progress_step(progress, _next_label(labels_iter))

  galaxies_particles = _polars_filter_galaxies(all_particles)
  galaxies_common = _polars_common(galaxies_particles, 'GalID', 'total', False, simulation)
  _assign_polars_result(galaxies_df, 'GalID', galaxies_common)
  _apply_angular_threshold(galaxies_df, 'total')
  _progress_step(progress, _next_label(labels_iter))

  dm_particles = polars_tables['dm']
  halos_dm = _polars_common(dm_particles, 'HaloID', 'dm', False, simulation)
  _assign_polars_result(halos_df, 'HaloID', halos_dm)
  _apply_angular_threshold(halos_df, 'dm')
  _progress_step(progress, _next_label(labels_iter))

  galaxies_dm = _polars_common(_polars_filter_galaxies(dm_particles), 'GalID', 'dm', False, simulation)
  _assign_polars_result(galaxies_df, 'GalID', galaxies_dm)
  _apply_angular_threshold(galaxies_df, 'dm')
  _progress_step(progress, _next_label(labels_iter))

  baryon_particles = pl.concat([polars_tables[ptype] for ptype in ['gas', 'star', 'bh']], how='vertical_relaxed')
  halos_baryon = _polars_common(baryon_particles, 'HaloID', 'baryon', False, simulation)
  _assign_polars_result(halos_df, 'HaloID', halos_baryon)
  _apply_angular_threshold(halos_df, 'baryon')
  _progress_step(progress, _next_label(labels_iter))

  galaxies_baryon = _polars_common(_polars_filter_galaxies(baryon_particles), 'GalID', 'baryon', False, simulation)
  _assign_polars_result(galaxies_df, 'GalID', galaxies_baryon)
  _apply_angular_threshold(galaxies_df, 'baryon')
  _progress_step(progress, _next_label(labels_iter))

  gas_particles = polars_tables['gas']
  halos_gas_common = _polars_common(gas_particles, 'HaloID', 'gas', False, simulation)
  _assign_polars_result(halos_df, 'HaloID', halos_gas_common)
  _apply_angular_threshold(halos_df, 'gas')
  _progress_step(progress, _next_label(labels_iter))

  galaxies_gas_common = _polars_common(_polars_filter_galaxies(gas_particles), 'GalID', 'gas', False, simulation)
  _assign_polars_result(galaxies_df, 'GalID', galaxies_gas_common)
  _apply_angular_threshold(galaxies_df, 'gas')
  _progress_step(progress, _next_label(labels_iter))

  polars_gas_full = _prepare_polars_gas_table(polars_tables['gas'], data_manager)
  gas_cols = ['HaloID', 'GalID', 'mass', 'rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature', 'mass_HI', 'mass_H2', '_metallicity_mass_weighted', '_metallicity_sfr_weighted', '_temp_mass_weighted', '_temp_metal_weighted']
  if not polars_gas_full.is_empty():
    gas_halos = _polars_gas_aggregate(polars_gas_full.select(gas_cols), 'HaloID')
    _assign_polars_result(halos_df, 'HaloID', gas_halos)
    _progress_step(progress, _next_label(labels_iter))

    gas_galaxies = _polars_gas_aggregate(_polars_filter_galaxies(polars_gas_full.select(gas_cols)), 'GalID')
    _assign_polars_result(galaxies_df, 'GalID', gas_galaxies)
    _progress_step(progress, _next_label(labels_iter))
  else:
    _progress_step(progress, _next_label(labels_iter))
    _progress_step(progress, _next_label(labels_iter))

  star_particles = polars_tables['star']
  star_halos = _polars_common(star_particles, 'HaloID', 'star', False, simulation)
  _assign_polars_result(halos_df, 'HaloID', star_halos)
  _apply_angular_threshold(halos_df, 'star')
  _progress_step(progress, _next_label(labels_iter))

  star_galaxies = _polars_common(_polars_filter_galaxies(star_particles), 'GalID', 'star', False, simulation)
  _assign_polars_result(galaxies_df, 'GalID', star_galaxies)
  _apply_angular_threshold(galaxies_df, 'star')
  _progress_step(progress, _next_label(labels_iter))

  star_props = _polars_star(polars_tables['star'], 'HaloID')
  _assign_polars_result(halos_df, 'HaloID', star_props)
  star_props_gal = _polars_star(_polars_filter_galaxies(polars_tables['star']), 'GalID')
  _assign_polars_result(galaxies_df, 'GalID', star_props_gal)
  _progress_step(progress, _next_label(labels_iter))

  bh_particles = polars_tables['bh']
  bh_halos_common = _polars_common(bh_particles, 'HaloID', 'bh', False, simulation)
  _assign_polars_result(halos_df, 'HaloID', bh_halos_common)
  _apply_angular_threshold(halos_df, 'bh')
  _progress_step(progress, _next_label(labels_iter))

  bh_galaxies_common = _polars_common(_polars_filter_galaxies(bh_particles), 'GalID', 'bh', False, simulation)
  _assign_polars_result(galaxies_df, 'GalID', bh_galaxies_common)
  _apply_angular_threshold(galaxies_df, 'bh')
  _progress_step(progress, _next_label(labels_iter))

  bh_props = _polars_bh(polars_tables['bh'], 'HaloID')
  _assign_polars_result(halos_df, 'HaloID', bh_props)
  bh_props_gal = _polars_bh(_polars_filter_galaxies(polars_tables['bh']), 'GalID')
  _assign_polars_result(galaxies_df, 'GalID', bh_props_gal)
  _progress_step(progress, _next_label(labels_iter))

  # Drop velocity/potential columns from particle tables to mirror pandas workflow
  for ptype in ['dm', 'gas', 'star', 'bh']:
    drop_cols = [col for col in ['vx', 'vy', 'vz', 'potential'] if col in data_manager[ptype]]
    if drop_cols:
      data_manager[ptype] = data_manager[ptype].drop(columns=drop_cols)

  if include_global:
    aperture_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x',  'y', 'z']
    data = pd.concat([data_manager[ptype][aperture_props_columns] for ptype in c.ptypes.keys()])

    aperture_HI_columns = ['HaloID', 'GalID', 'ptype', 'mass_HI', 'x',  'y', 'z']
    HI_gas = data_manager['gas'][aperture_HI_columns].copy()
    HI_gas.rename(columns={'mass_HI': 'mass'}, inplace=True)
    HI_gas['ptype'] = 10

    aperture_H2_columns = ['HaloID', 'GalID', 'ptype', 'mass_H2', 'x',  'y', 'z']
    H2_gas = data_manager['gas'][aperture_H2_columns].copy()
    H2_gas.rename(columns={'mass_H2': 'mass'}, inplace=True)
    H2_gas['ptype'] = 11

    data = pd.concat([data, HI_gas, H2_gas], ignore_index=True)

    aperture = 30.
    galaxy_positions = galaxies_df[['x_total', 'y_total', 'z_total']].to_numpy()

    process_halo = partial(calculate_aperture_masses, aperture=aperture, galaxy_positions=galaxy_positions)
    aperture_masses = data.groupby(by='HaloID').apply(process_halo, include_groups = False).reset_index(names=['HaloID', 'GalID'])
    aperture_masses.set_index('GalID', inplace=True)

    galaxies_df['mass_gas_30kpc'] = aperture_masses[0]
    galaxies_df['mass_dm_30kpc'] = aperture_masses[1]
    galaxies_df['mass_star_30kpc'] = aperture_masses[4]
    galaxies_df['mass_bh_30kpc'] = aperture_masses[5]
    galaxies_df['mass_HI_30kpc'] = aperture_masses[10]
    galaxies_df['mass_H2_30kpc'] = aperture_masses[11]
    galaxies_df['mass_total_30kpc'] = galaxies_df[['mass_gas_30kpc', 'mass_dm_30kpc', 'mass_star_30kpc', 'mass_bh_30kpc']].sum(axis=1)

    _progress_step(progress, _next_label(labels_iter))
    calculate_local_densities(data_manager)
    _progress_step(progress, _next_label(labels_iter))
  else:
    _progress_step(progress, _next_label(labels_iter))
    _progress_step(progress, _next_label(labels_iter))

  data_manager['halos'] = halos_df.copy()
  data_manager['galaxies'] = galaxies_df.copy()
  progress.close()


def calculate_group_properties(data_manager: DataManager, use_polars: bool = False, include_global: bool = True) -> None:
  if use_polars and HAS_POLARS:
    calculate_group_properties_polars(data_manager, include_global=include_global)
    return
  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pot', ptype)

  collections = ['halos', 'galaxies']

  labels = _build_progress_labels(collections)
  labels_iter = iter(labels)
  progress = tqdm(total=len(labels), desc='Computing group properties', unit='task', leave=False)

  group_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potential']
  columns_to_drop = ['vx', 'vy', 'vz', 'potential']

  # total
  for collection in collections:
    data = pd.concat([data_manager[ptype][group_props_columns] for ptype in c.ptypes.keys()], ignore_index=True)
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'total')
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))

  data = None

  #dm
  for collection in collections:
    data = data_manager['dm'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'dm')
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))
  
  data = None
  data_manager['dm'].drop(columns=columns_to_drop, inplace=True)

  # baryon
  for collection in collections:
    data = pd.concat([data_manager[ptype][group_props_columns] for ptype in ['gas', 'star', 'bh']], ignore_index=True)
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'baryon')
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))
  
  data = None

  # gas
  for collection in collections:
    data = data_manager['gas'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'gas')
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))
  
  data = None
  data_manager['gas'].drop(columns=columns_to_drop, inplace=True)

  for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
    data_manager.load_property(property, 'gas')

  gas_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']
  for collection in collections:
    data = data_manager['gas'][gas_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_gas(data_manager, data, collection)
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))

  # star
  for collection in collections:
    data = data_manager['star'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'star')
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))

  data = None
  data_manager['star'].drop(columns=columns_to_drop, inplace=True)

  for property in ['age', 'metallicity']:
    data_manager.load_property(property, 'star')

  star_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'age', 'metallicity']
  for collection in collections:
    data = data_manager['star'][star_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_star(data_manager, data, collection)
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))

  # bh
  for collection in collections:
    data = data_manager['bh'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'bh')
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))
  
  data = None
  data_manager['bh'].drop(columns=columns_to_drop, inplace=True)

  for property in ['bhmdot']:
    data_manager.load_property(property, 'bh')
  
  star_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'bhmdot']
  for collection in collections:
    data = data_manager['bh'][star_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_bh(data_manager, data, collection)
    data_manager[collection] = data_manager[collection].copy()
    _progress_step(progress, _next_label(labels_iter))

  if include_global:
    aperture_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x',  'y', 'z']
    data = pd.concat([data_manager[ptype][aperture_props_columns] for ptype in c.ptypes.keys()])

    aperture_HI_columns = ['HaloID', 'GalID', 'ptype', 'mass_HI', 'x',  'y', 'z']
    HI_gas = data_manager['gas'][aperture_HI_columns].copy()
    HI_gas.rename(columns={'mass_HI': 'mass'}, inplace=True)
    HI_gas['ptype'] = 10

    aperture_H2_columns = ['HaloID', 'GalID', 'ptype', 'mass_H2', 'x',  'y', 'z']
    H2_gas = data_manager['gas'][aperture_H2_columns].copy()
    H2_gas.rename(columns={'mass_H2': 'mass'}, inplace=True)
    H2_gas['ptype'] = 11

    data = pd.concat([data, HI_gas, H2_gas], ignore_index=True)

    aperture = 30.
    galaxy_positions = data_manager['galaxies'][['x_total', 'y_total', 'z_total']].to_numpy()

    process_halo = partial(calculate_aperture_masses, aperture=aperture, galaxy_positions=galaxy_positions)
    aperture_masses = data.groupby(by='HaloID').apply(process_halo, include_groups = False).reset_index(names=['HaloID', 'GalID'])
    aperture_masses.set_index('GalID', inplace=True)

    data_manager['galaxies']['mass_gas_30kpc'] = aperture_masses[0]
    data_manager['galaxies']['mass_dm_30kpc'] = aperture_masses[1]
    data_manager['galaxies']['mass_star_30kpc'] = aperture_masses[4]
    data_manager['galaxies']['mass_bh_30kpc'] = aperture_masses[5]
    data_manager['galaxies']['mass_HI_30kpc'] = aperture_masses[10]
    data_manager['galaxies']['mass_H2_30kpc'] = aperture_masses[11]
    data_manager['galaxies']['mass_total_30kpc'] = data_manager['galaxies'][['mass_gas_30kpc', 'mass_dm_30kpc', 'mass_star_30kpc', 'mass_bh_30kpc']].sum(axis=1)

    _progress_step(progress, _next_label(labels_iter))
    calculate_local_densities(data_manager)
    _progress_step(progress, _next_label(labels_iter))
  else:
    _progress_step(progress, _next_label(labels_iter))
    _progress_step(progress, _next_label(labels_iter))
  progress.close()
