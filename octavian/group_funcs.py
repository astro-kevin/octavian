from __future__ import annotations
import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
import constants as c
from functools import partial
from astropy import constants as const

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from data_manager import DataManager


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


def calculate_group_properties(data_manager: DataManager) -> None:
  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pot', ptype)

  collections = ['halos', 'galaxies']

  group_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potential']
  columns_to_drop = ['vx', 'vy', 'vz', 'potential']

  # total
  for collection in collections:
    data = pd.concat([data_manager[ptype][group_props_columns] for ptype in c.ptypes.keys()], ignore_index=True)
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'total')
    data_manager[collection] = data_manager[collection].copy()

  data = None

  #dm
  for collection in collections:
    data = data_manager['dm'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'dm')
    data_manager[collection] = data_manager[collection].copy()
  
  data = None
  data_manager['dm'].drop(columns=columns_to_drop, inplace=True)

  # baryon
  for collection in collections:
    data = pd.concat([data_manager[ptype][group_props_columns] for ptype in ['gas', 'star', 'bh']], ignore_index=True)
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'baryon')
    data_manager[collection] = data_manager[collection].copy()
  
  data = None

  # gas
  for collection in collections:
    data = data_manager['gas'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'gas')
    data_manager[collection] = data_manager[collection].copy()
  
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

  # star
  for collection in collections:
    data = data_manager['star'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'star')
    data_manager[collection] = data_manager[collection].copy()

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

  # bh
  for collection in collections:
    data = data_manager['bh'][group_props_columns].copy()
    if collection == 'galaxies': data = data.loc[data['GalID'] != -1]
    calculateGroupProperties_common(data_manager, data, collection, 'bh')
    data_manager[collection] = data_manager[collection].copy()
  
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

  # aperture
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

  
  calculate_local_densities(data_manager)
