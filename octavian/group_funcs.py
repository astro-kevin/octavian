from __future__ import annotations

import numpy as np
import unyt
from sklearn.neighbors import NearestNeighbors
import octavian.constants as c
from functools import partial
from astropy import constants as const

from octavian.backend import pd, USING_MODIN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager


# helper function to broadcast group properties to particle data, based on haloid
def broadcast_properties(data: pd.DataFrame, groupID: str, collection_data: pd.DataFrame, properties: list[str] | str) -> np.ndarray:
  identifiers = data[groupID].to_numpy()
  if isinstance(properties, list):
    values = collection_data.loc[identifiers, properties]
    return values.to_numpy()
  mapped = collection_data.loc[identifiers, properties]
  return mapped.to_numpy()


def calculateGroupProperties_common(data_manager: DataManager, data: pd.DataFrame, collection: str, group_name: str) -> None:
  if data.empty:
    return
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  group_labels = data[groupID]
  grouped = data.groupby(by=groupID)

  counts = grouped.size()
  if not counts.empty:
    collection_data.loc[counts.index, f"n{group_name}"] = counts

  if group_name == 'bh':
    mass_series = grouped['mass'].max()
  else:
    mass_series = grouped['mass'].sum()
  if not mass_series.empty:
    collection_data.loc[mass_series.index, f"mass_{group_name}"] = mass_series

  if collection == 'halos' and group_name == 'total':
    min_rows = data.sort_values(by=[groupID, 'potential']).drop_duplicates(subset=groupID, keep='first')
    if not min_rows.empty:
      min_values = min_rows.set_index(groupID)[['x', 'y', 'z', 'vx', 'vy', 'vz']]
      collection_data.loc[min_values.index, ['minpot_x', 'minpot_y', 'minpot_z', 'minpot_vx', 'minpot_vy', 'minpot_vz']] = min_values

  def _assign_weighted(column: str) -> None:
    weighted = (data[column] * data['mass']).groupby(group_labels).sum()
    if weighted.empty:
      return
    idx = weighted.index
    numer = weighted.to_numpy()
    denom = mass_series.loc[idx].to_numpy()
    values = np.zeros_like(numer, dtype=np.float64)
    mask = denom != 0
    values[mask] = numer[mask] / denom[mask]
    collection_data.loc[idx, f"{column}_{group_name}"] = pd.Series(values, index=idx)

  for column in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
    _assign_weighted(column)

  if counts.empty:
    return

  velocity_df = data[['vx', 'vy', 'vz']]
  mean_velocity = broadcast_properties(data, groupID, collection_data, [f'vx_{group_name}', f'vy_{group_name}', f'vz_{group_name}'])
  mean_velocity_df = pd.DataFrame(mean_velocity, columns=['vx', 'vy', 'vz'], index=data.index)
  rel_velocity_df = velocity_df - mean_velocity_df
  rel_velocity_sq = (rel_velocity_df.pow(2)).sum(axis=1)
  dispersion_df = pd.DataFrame({'group': group_labels, 'value': rel_velocity_sq})
  dispersion_sum = dispersion_df.groupby('group')['value'].sum()
  if not dispersion_sum.empty:
    count_vals = counts.loc[dispersion_sum.index].to_numpy()
    disp_vals = np.zeros_like(count_vals, dtype=np.float64)
    mask = count_vals != 0
    disp_vals[mask] = np.sqrt(dispersion_sum.to_numpy()[mask] / count_vals[mask])
    collection_data.loc[dispersion_sum.index, f"velocity_dispersion_{group_name}"] = pd.Series(disp_vals, index=dispersion_sum.index)

  center_columns = ['minpot_x', 'minpot_y', 'minpot_z'] if collection == 'halos' and group_name == 'total' else [f'x_{group_name}', f'y_{group_name}', f'z_{group_name}']
  centers = broadcast_properties(data, groupID, collection_data, center_columns)
  centers_df = pd.DataFrame(centers, columns=['x', 'y', 'z'], index=data.index)
  rel_pos_df = data[['x', 'y', 'z']] - centers_df

  mass_series_row = data['mass']
  mass_array = mass_series_row.to_numpy()
  rel_momentum_df = rel_velocity_df.mul(mass_series_row, axis=0)

  angular_df = pd.DataFrame(
    {
      'Lx': rel_pos_df.iloc[:, 1] * rel_momentum_df['vz'] - rel_pos_df.iloc[:, 2] * rel_momentum_df['vy'],
      'Ly': rel_pos_df.iloc[:, 2] * rel_momentum_df['vx'] - rel_pos_df.iloc[:, 0] * rel_momentum_df['vz'],
      'Lz': rel_pos_df.iloc[:, 0] * rel_momentum_df['vy'] - rel_pos_df.iloc[:, 1] * rel_momentum_df['vx'],
    },
    index=data.index,
  )
  angular_df['group'] = group_labels
  angular_grouped = angular_df.groupby('group')[['Lx', 'Ly', 'Lz']].sum()
  if not angular_grouped.empty:
    for label in ['x', 'y', 'z']:
      series = angular_grouped[f'L{label}']
      collection_data.loc[series.index, f"L{label}_{group_name}"] = series

    L_mag_series = (angular_grouped.pow(2).sum(axis=1)) ** 0.5
    collection_data.loc[L_mag_series.index, f"L_{group_name}"] = L_mag_series

    Ly_vals = angular_grouped['Ly'].to_numpy()
    Lz_vals = angular_grouped['Lz'].to_numpy()
    alpha_vals = np.arctan2(Ly_vals, Lz_vals)
    collection_data.loc[angular_grouped.index, f"ALPHA_{group_name}"] = pd.Series(alpha_vals, index=angular_grouped.index)

    Lx_vals = angular_grouped['Lx'].to_numpy()
    L_mag_vals = L_mag_series.to_numpy()
    beta_vals = np.zeros_like(Lx_vals, dtype=np.float64)
    non_zero = L_mag_vals != 0
    beta_vals[non_zero] = np.arcsin(Lx_vals[non_zero] / L_mag_vals[non_zero])
    collection_data.loc[angular_grouped.index, f"BETA_{group_name}"] = pd.Series(beta_vals, index=angular_grouped.index)

  L_group = broadcast_properties(data, groupID, collection_data, [f'Lx_{group_name}', f'Ly_{group_name}', f'Lz_{group_name}'])
  L_group_df = pd.DataFrame(L_group, columns=['Lx_grp', 'Ly_grp', 'Lz_grp'], index=data.index)
  L_dot = (angular_df[['Lx', 'Ly', 'Lz']].to_numpy() * L_group_df.to_numpy()).sum(axis=1)

  negative_array = np.where(L_dot < 0, mass_array, 0.0)
  negative_df = pd.DataFrame({'group': group_labels, 'value': negative_array})
  negative_sum = negative_df.groupby('group')['value'].sum()
  if not negative_sum.empty:
    denom = mass_series.loc[negative_sum.index].to_numpy()
    bovert_vals = np.zeros_like(denom, dtype=np.float64)
    mask = denom != 0
    bovert_vals[mask] = 2.0 * negative_sum.to_numpy()[mask] / denom[mask]
    collection_data.loc[negative_sum.index, f"BoverT_{group_name}"] = pd.Series(bovert_vals, index=negative_sum.index)

  kinetic_array = 0.5 * mass_array * rel_velocity_sq.to_numpy()
  kinetic_df = pd.DataFrame({'group': group_labels, 'value': kinetic_array})
  kinetic_sum = kinetic_df.groupby('group')['value'].sum()

  cross_df = pd.DataFrame(
    {
      'cx': rel_pos_df.iloc[:, 1] * L_group_df['Lz_grp'] - rel_pos_df.iloc[:, 2] * L_group_df['Ly_grp'],
      'cy': rel_pos_df.iloc[:, 2] * L_group_df['Lx_grp'] - rel_pos_df.iloc[:, 0] * L_group_df['Lz_grp'],
      'cz': rel_pos_df.iloc[:, 0] * L_group_df['Ly_grp'] - rel_pos_df.iloc[:, 1] * L_group_df['Lx_grp'],
    },
    index=data.index,
  )
  rz = np.sqrt((cross_df.pow(2)).sum(axis=1).to_numpy())
  mass_array = data['mass'].to_numpy()
  L_dot_array = L_dot
  krot_vals = np.zeros_like(rz, dtype=np.float64)
  valid = (rz > 0) & (mass_array > 0)
  krot_vals[valid] = 0.5 * ((L_dot_array[valid] / rz[valid]) ** 2) / mass_array[valid]
  krot_df = pd.DataFrame({'group': group_labels, 'value': krot_vals})
  krot_sum = krot_df.groupby('group')['value'].sum()
  if not krot_sum.empty:
    denom = kinetic_sum.loc[krot_sum.index].to_numpy()
    kappa_vals = np.zeros_like(denom, dtype=np.float64)
    mask = denom != 0
    kappa_vals[mask] = krot_sum.to_numpy()[mask] / denom[mask]
    collection_data.loc[krot_sum.index, f"kappa_rot_{group_name}"] = pd.Series(kappa_vals, index=krot_sum.index)

  angular_cols = [
    f"velocity_dispersion_{group_name}",
    f"Lx_{group_name}", f"Ly_{group_name}", f"Lz_{group_name}",
    f"L_{group_name}", f"BoverT_{group_name}", f"kappa_rot_{group_name}",
    f"ALPHA_{group_name}", f"BETA_{group_name}",
  ]
  existing = [col for col in angular_cols if col in collection_data]
  if existing:
    mask_small = collection_data[f"n{group_name}"] < 3
    collection_data.loc[mask_small, existing] = 0.0


def calculateGroupProperties_gas(data_manager: DataManager, data: pd.DataFrame, collection: str) -> None:
  if data.empty:
    return
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)

  def _assign(series, column):
    nonlocal collection_data
    if hasattr(series, "empty"):
      is_empty = series.empty
    else:
      is_empty = len(series) == 0  # type: ignore[arg-type]
    if is_empty:
      return
    missing = series.index.difference(collection_data.index)
    if not missing.empty:
      collection_data = collection_data.reindex(collection_data.index.union(missing))
      data_manager[collection] = collection_data
    collection_data.loc[series.index, column] = series

  def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series | None:
    if hasattr(numerator, "empty") and numerator.empty:
      return None
    numer_vals = numerator.to_numpy(dtype=np.float64, copy=False)
    denom_vals = denominator.to_numpy(dtype=np.float64, copy=True)
    result = np.zeros_like(numer_vals, dtype=np.float64)
    mask = (denom_vals != 0) & ~np.isnan(denom_vals)
    result[mask] = numer_vals[mask] / denom_vals[mask]
    return pd.Series(result, index=numerator.index)

  data['fHI'] = data['nh'] / data['mass']

  not_conserving_mass = data['fHI'] + data['fH2'] > 1.0
  data.loc[not_conserving_mass, 'fHI'] = 1.0 - data.loc[not_conserving_mass, 'fH2']

  data['mass_HI'] = c.XH * data['fHI'] * data['mass']
  data['mass_H2'] = c.XH * data['fH2'] * data['mass']

  if collection == 'halos':
    data_manager['gas']['mass_HI'] = data['mass_HI']
    data_manager['gas']['mass_H2'] = data['mass_H2']

  mass_hi = halos_grouped['mass_HI'].sum()
  mass_h2 = halos_grouped['mass_H2'].sum()
  _assign(mass_hi, 'mass_HI')
  _assign(mass_h2, 'mass_H2')

  data.drop(columns=['nh', 'fHI', 'fH2', 'mass_HI', 'mass_H2'], inplace=True)

  sfr = halos_grouped['sfr'].sum()
  _assign(sfr, 'sfr')

  data['metallicity_mass_weighted'] = data['metallicity'] * data['mass']
  data['metallicity_sfr_weighted'] = data['metallicity'] * data['sfr']

  metallicity_mass_sum = halos_grouped['metallicity_mass_weighted'].sum()
  metallicity_sfr_sum = halos_grouped['metallicity_sfr_weighted'].sum()

  if not metallicity_mass_sum.empty:
    gas_mass = collection_data.loc[metallicity_mass_sum.index, 'mass_gas']
    ratio = _safe_ratio(metallicity_mass_sum, gas_mass)
    if ratio is not None:
      _assign(ratio, 'metallicity_mass_weighted')

  if not metallicity_sfr_sum.empty:
    sfr_denom = collection_data.loc[metallicity_sfr_sum.index, 'sfr']
    ratio = _safe_ratio(metallicity_sfr_sum, sfr_denom)
    if ratio is not None:
      _assign(ratio, 'metallicity_sfr_weighted')

  data['temp_mass_weighted'] = data['temperature'] * data['mass']
  data['temp_metal_weighted'] = data['temperature'] * data['mass'] * data['metallicity']

  cgm_mask = data['rho'] < c.nHlim
  halos_cgm_grouped = data.loc[cgm_mask, ['HaloID', 'mass', 'temp_mass_weighted', 'temp_metal_weighted']].groupby(by='HaloID')
  mass_cgm = halos_cgm_grouped['mass'].sum()
  _assign(mass_cgm, 'mass_cgm')

  temp_mass_sum = halos_grouped['temp_mass_weighted'].sum()
  if not temp_mass_sum.empty:
    gas_mass = collection_data.loc[temp_mass_sum.index, 'mass_gas']
    ratio = _safe_ratio(temp_mass_sum, gas_mass)
    if ratio is not None:
      _assign(ratio, 'temp_mass_weighted')

  temp_mass_cgm_sum = halos_cgm_grouped['temp_mass_weighted'].sum()
  if not temp_mass_cgm_sum.empty:
    mass_cgm_subset = collection_data.loc[temp_mass_cgm_sum.index, 'mass_cgm']
    avg_temp_cgm = _safe_ratio(temp_mass_cgm_sum, mass_cgm_subset)
    if avg_temp_cgm is not None:
      _assign(avg_temp_cgm, 'temp_mass_weighted_cgm')

    temp_metal_cgm_sum = halos_cgm_grouped['temp_metal_weighted'].sum().reindex(temp_mass_cgm_sum.index, fill_value=0.0)
    ratio = _safe_ratio(temp_metal_cgm_sum, temp_mass_cgm_sum)
    if ratio is not None:
      _assign(ratio, 'temp_metal_weighted_cgm')

  data.drop(columns=['temp_mass_weighted'], inplace=True)

  halos_cgm_grouped = data.loc[cgm_mask, ['HaloID', 'mass', 'metallicity_mass_weighted', 'metallicity_sfr_weighted', 'temp_metal_weighted']].groupby(by='HaloID')

  metallicity_mass_cgm_sum = halos_cgm_grouped['metallicity_mass_weighted'].sum()
  if not metallicity_mass_cgm_sum.empty:
    mass_cgm_subset = collection_data.loc[metallicity_mass_cgm_sum.index, 'mass_cgm']
    metallicity_mass_cgm = _safe_ratio(metallicity_mass_cgm_sum, mass_cgm_subset)
    if metallicity_mass_cgm is not None:
      _assign(metallicity_mass_cgm, 'metallicity_mass_weighted_cgm')

    temp_metal_cgm_sum = halos_cgm_grouped['temp_metal_weighted'].sum().reindex(metallicity_mass_cgm_sum.index, fill_value=0.0)
    metallicity_temp_weighted = _safe_ratio(temp_metal_cgm_sum, metallicity_mass_cgm_sum)
    if metallicity_temp_weighted is not None:
      _assign(metallicity_temp_weighted, 'metallicity_temp_weighted_cgm')


def calculateGroupProperties_star(data_manager: DataManager, data: pd.DataFrame, collection: str) -> None:
  if data.empty:
    return
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)

  data['metallicity_stellar'] = data['metallicity'] * data['mass']

  collection_data['metallicity_stellar'] = halos_grouped['metallicity_stellar'].sum()

  data['age_mass_weighted'] = data['age'] * data['mass']
  data['age_metal_weighted'] = data['age'] * data['mass'] * data['metallicity']

  collection_data['age_mass_weighted'] = halos_grouped['age_mass_weighted'].sum() / collection_data['mass_star']
  collection_data['age_metal_weighted'] = halos_grouped['age_metal_weighted'].sum() / collection_data['metallicity_stellar']

  collection_data['metallicity_stellar'] /= halos_grouped['mass'].sum()


def calculateGroupProperties_bh(data_manager: DataManager, data: pd.DataFrame, collection: str) -> None:
  if data.empty:
    return
  collection_data = data_manager[collection]
  groupID = 'HaloID' if collection == 'halos' else 'GalID'
  halos_grouped = data.groupby(by=groupID)

  max_mass = halos_grouped['mass'].transform('max')
  mask_max = data['mass'] == max_mass
  if hasattr(mask_max, "to_numpy"):
    mask_max = mask_max.to_numpy(dtype=bool, copy=False)
  selected = data.loc[mask_max]
  if not selected.empty:
    selected = selected.drop_duplicates(subset=groupID, keep='first')
    data = selected.set_index(groupID)
  else:
    data = data.groupby(groupID).head(1).set_index(groupID)

  collection_data['bhmdot'] = data['bhmdot'].copy()

  FRAD = 0.1  # assume 10% radiative efficiency
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value
  collection_data['bh_fedd'] = data['bhmdot'] / (edd_factor * data['mass'])


def calculate_aperture_masses(halo: pd.DataFrame, aperture: float, galaxy_positions: np.ndarray) -> pd.DataFrame:
  galaxy_ids = halo['GalID'].unique()
  galaxy_ids = galaxy_ids[galaxy_ids != -1]
  if len(galaxy_ids) == 0:
    return pd.DataFrame()

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

    for radius in [300.0, 1000.0, 3000.0]:
      volume = 4.0 / 3.0 * np.pi * radius**3

      df = pd.DataFrame({'indexes': neighbors.radius_neighbors(pos, radius=radius, return_distance=False)})
      df = df.explode('indexes').dropna()

      df['mass'] = mass[df['indexes'].astype('int')]
      grouped = df.groupby(level=0)

      collection_data[f'local_mass_density_{int(radius)}'] = grouped['mass'].sum() / volume
      collection_data[f'local_number_density_{int(radius)}'] = grouped.size() / volume


def calculate_group_properties(data_manager: DataManager, include_global: bool = True) -> None:
  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pot', ptype)

  collections = ['halos', 'galaxies']

  group_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potential']
  columns_to_drop = ['vx', 'vy', 'vz', 'potential']

  for collection in collections:
    data = pd.concat([data_manager[ptype][group_props_columns] for ptype in c.ptypes.keys()], ignore_index=True)
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'total')

  for collection in collections:
    data = data_manager['dm'][group_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'dm')
  data_manager['dm'].drop(columns=columns_to_drop, inplace=True)

  for collection in collections:
    data = pd.concat([data_manager[ptype][group_props_columns] for ptype in ['gas', 'star', 'bh']], ignore_index=True)
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'baryon')

  for collection in collections:
    data = data_manager['gas'][group_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'gas')
  data_manager['gas'].drop(columns=columns_to_drop, inplace=True)

  for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
    data_manager.load_property(property, 'gas')

  gas_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']
  for collection in collections:
    data = data_manager['gas'][gas_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_gas(data_manager, data, collection)

  for collection in collections:
    data = data_manager['star'][group_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'star')
  data_manager['star'].drop(columns=columns_to_drop, inplace=True)

  for property in ['age', 'metallicity']:
    data_manager.load_property(property, 'star')

  star_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'age', 'metallicity']
  for collection in collections:
    data = data_manager['star'][star_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_star(data_manager, data, collection)

  for collection in collections:
    data = data_manager['bh'][group_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'bh')
  data_manager['bh'].drop(columns=columns_to_drop, inplace=True)

  for property in ['bhmdot']:
    data_manager.load_property(property, 'bh')

  bh_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'bhmdot']
  for collection in collections:
    data = data_manager['bh'][bh_props_columns]
    if collection == 'galaxies':
      data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_bh(data_manager, data, collection)

  if include_global:
    aperture_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z']
    data = pd.concat([data_manager[ptype][aperture_props_columns] for ptype in c.ptypes.keys()])

    aperture_HI_columns = ['HaloID', 'GalID', 'ptype', 'mass_HI', 'x', 'y', 'z']
    HI_gas = data_manager['gas'][aperture_HI_columns].copy()
    HI_gas.rename(columns={'mass_HI': 'mass'}, inplace=True)
    HI_gas['ptype'] = 10

    aperture_H2_columns = ['HaloID', 'GalID', 'ptype', 'mass_H2', 'x', 'y', 'z']
    H2_gas = data_manager['gas'][aperture_H2_columns].copy()
    H2_gas.rename(columns={'mass_H2': 'mass'}, inplace=True)
    H2_gas['ptype'] = 11

    data = pd.concat([data, HI_gas, H2_gas], ignore_index=True)

    aperture = 30.0
    galaxy_positions = data_manager['galaxies'][['x_total', 'y_total', 'z_total']].to_numpy()

    process_halo = partial(calculate_aperture_masses, aperture=aperture, galaxy_positions=galaxy_positions)
    aperture_masses = data.groupby(by='HaloID').apply(process_halo, include_groups=False).reset_index(names=['HaloID', 'GalID'])
    aperture_masses.set_index('GalID', inplace=True)

    data_manager['galaxies']['mass_gas_30kpc'] = aperture_masses[0]
    data_manager['galaxies']['mass_dm_30kpc'] = aperture_masses[1]
    data_manager['galaxies']['mass_star_30kpc'] = aperture_masses[4]
    data_manager['galaxies']['mass_bh_30kpc'] = aperture_masses[5]
    data_manager['galaxies']['mass_HI_30kpc'] = aperture_masses[10]
    data_manager['galaxies']['mass_H2_30kpc'] = aperture_masses[11]
    data_manager['galaxies']['mass_total_30kpc'] = data_manager['galaxies'][['mass_gas_30kpc', 'mass_dm_30kpc', 'mass_star_30kpc', 'mass_bh_30kpc']].sum(axis=1)

    calculate_local_densities(data_manager)
