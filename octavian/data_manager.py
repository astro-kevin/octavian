import numpy as np
import pandas as pd
import h5py
import unyt
from astropy.cosmology import FlatLambdaCDM
from sympy import sympify
import octavian.constants as c

class DataManager:
  def __init__(self, snapfile: str, fraction: list[float] = [0, 1]):
    self.snapfile = snapfile
    self.start_fraction = fraction[0]
    self.end_fraction = fraction[1]

    self.load_simulation_constants()

    self.initialise_dataframes_with_haloids()
    self.validate_halos()

    self.load_masses()

    # initialise group dfs, halos with pids
    self.halos = pd.DataFrame(index=self.haloIDs)
    self.load_halo_pids()

  # programmatic access to attrs based on https://peps.python.org/pep-0363/
  def __getitem__(self, name):
     return getattr(self, name)
  
  def __setitem__(self, name, value):
    return setattr(self, name, value)
  
  def __delitem__(self, name):
    return delattr(self, name)
  
  def __contains__(self, name):
     return hasattr(self, name)
  
  def load_simulation_constants(self) -> None:
    self.simulation = {}
    with h5py.File(self.snapfile) as f:
      header = f['Header'].attrs

      self.simulation['boxsize'] = header['BoxSize']
      self.simulation['O0'] = header['Omega0']
      self.simulation['Ol'] = header['OmegaLambda']
      self.simulation['Ok'] = 0 # header['Omegak']
      self.simulation['h'] = header['HubbleParam']
      self.simulation['redshift'] = header['Redshift']
      self.simulation['a'] = header['Time']

    self.simulation['G'] = unyt.G.to('cm**3/(g * s**2)')

    registry = unyt.UnitRegistry()
    registry.add('h', self.simulation['h'], sympify(1))
    registry.add('a', self.simulation['a'], sympify(1))

    self.units = unyt.UnitSystem('gadget', 'kpc', 'Msun', '1.e9 * yr', registry=registry)
    self.units['velocity'] = 'km/s'

    self.cosmology = FlatLambdaCDM(H0=100*self.simulation['h'], Om0=self.simulation['O0'])
    self.simulation['time_gyr'] = self.cosmology.age(self.simulation['redshift']).value
    self.simulation['time'] = (self.simulation['time_gyr'] * unyt.Gyr).to('s').d

    self.simulation['Hz'] = 100 * self.simulation['h'] * np.sqrt(self.simulation['Ol'] + self.simulation['O0'] * self.simulation['a']**-3) / (1 * unyt.kpc).to('m').d / (1 * unyt.s)
    self.simulation['rhocrit'] = (3. * self.simulation['Hz']**2 / (8. * np.pi * self.simulation['G'])).to('Msun / kpc**3').d

    self.simulation['E_z'] = np.sqrt(self.simulation['Ol'] + self.simulation['Ok'] * self.simulation['a']**-2 + self.simulation['O0'] * self.simulation['a']**-3)
    self.simulation['Om_z'] = self.simulation['O0'] * self.simulation['a']**-3 / self.simulation['E_z']**2

    self.simulation['r200_factor'] = (200 * 4./3. * np.pi*self.simulation['Om_z'] * self.simulation['rhocrit'] * self.simulation['a']**3)**(-1./3.)

  def create_unit_quantity(self, prop: str) -> unyt.unyt_quantity:
    return unyt.unyt_quantity(1., c.code_units[prop], registry=self.units.registry)

  def initialise_dataframes_with_haloids(self) -> None:
    with h5py.File(self.snapfile) as f:
      for ptype, name in c.ptypes.items():
        self[ptype] = pd.DataFrame()
        self[ptype]['HaloID'] = f[name]['HaloID'][:]

        self[ptype] = self[ptype].loc[self[ptype]['HaloID'] != 0]
        self[ptype]['ptype'] = c.ptype_ids[ptype]

        self[ptype].sort_values(by='HaloID', inplace=True)
        self[ptype].rename_axis(index='pid', inplace=True)

  def validate_halos(self) -> None:
    halos_combined = pd.concat([self[ptype][['HaloID']] for ptype in c.ptypes.keys()]).groupby(by='HaloID').filter(lambda halo: len(halo) >= c.MINIMUM_DM_PER_HALO).sort_values(by='HaloID')
    valid_halos = halos_combined['HaloID'].unique()

    if self.start_fraction != 0:
      start_at = halos_combined.iloc[int(len(halos_combined) * self.start_fraction)]['HaloID']
      valid_halos = valid_halos[valid_halos > start_at]
    if self.end_fraction != 1:
      end_at = halos_combined.iloc[int(len(halos_combined) * self.end_fraction)]['HaloID']
      valid_halos = valid_halos[valid_halos <= end_at]

    self.haloIDs = np.sort(valid_halos)

    for ptype in c.ptypes.keys():
        self[ptype] = self[ptype].query('HaloID in @valid_halos')

  def load_halo_pids(self) -> None:
    for ptype in c.ptypes.keys():
      ptype_plists = self[ptype].groupby(by='HaloID').apply(lambda x: x.index.to_numpy())
      plist = c.ptype_lists[ptype]

      self.halos[plist] = ptype_plists

    # cursed replacement of nans with empty lists, based on https://stackoverflow.com/a/62689667
    for plist in c.ptype_lists.values():
      self.halos[plist] = self.halos[plist].fillna({i: [] for i in self.halos.index})

  def load_galaxy_pids(self) -> None:
    for ptype in c.ptypes.keys():
      if ptype == 'dm': continue

      data = self[ptype].loc[self[ptype]['GalID'] != -1].copy()
      ptype_plists = data.groupby(by='GalID').apply(lambda x: x.index.to_numpy())
      plist = c.ptype_lists[ptype]

      self.galaxies[plist] = ptype_plists

    for plist in c.ptype_lists.values():
      self.galaxies[plist] = self.galaxies[plist].fillna({i: [] for i in self.galaxies.index})

  def load_masses(self):
    with h5py.File(self.snapfile) as f:
      for ptype, id in c.ptypes.items():
        dataset = c.prop_aliases['bhmass'] if ptype == 'bh' else c.prop_aliases['mass']
        masses = f[id][dataset][:] * self.get_unit_conversion_factor('mass')
        self[ptype]['mass'] = masses[self[ptype].index]
        
        self[f'n{ptype}'] = len(masses)
        self[f'm{ptype}_total'] = np.sum(masses)

  def get_prop_name(self, prop: str) -> str:
    return c.prop_aliases[prop.strip(' _').lower()]
  
  def get_ptype_name(self, ptype: str) -> str:
    return c.ptypes[ptype.strip(' _').lower()]
  
  def get_column_name(self, prop: str) -> str | list[str]:
    return c.prop_columns[prop.strip(' _').lower()]
  
  def get_unit_conversion_factor(self, prop: str):
    try:
      data_units = c.prop_units[prop]
      code_units = c.code_units[prop]
      factor = unyt.unyt_quantity(1., data_units, registry=self.units.registry).to(code_units)
      if prop == 'rho':
        factor *= c.XH / unyt.mp.to('g').d
    except KeyError:
      factor = 1.

    return factor
  
  def load_property(self, requested_prop: str, requested_ptype: str):
    prop = self.get_prop_name(requested_prop)
    ptype = self.get_ptype_name(requested_ptype)
    column = self.get_column_name(requested_prop)

    with h5py.File(self.snapfile) as f:
      if requested_prop == 'metallicity':
        self[requested_ptype][column] = f[ptype][prop][:, 0][self[requested_ptype].index]
      elif requested_prop == 'age':
        data = f[ptype][prop][:][self[requested_ptype].index]
        self[requested_ptype][column] = self.simulation['time_gyr'] - self.cosmology.age(1/data - 1).value
      else:
        self[requested_ptype][column] = f[ptype][prop][:][self[requested_ptype].index] * self.get_unit_conversion_factor(requested_prop)