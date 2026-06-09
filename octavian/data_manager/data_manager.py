import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import unyt
from astropy.cosmology import FlatLambdaCDM
from sympy import sympify
from octavian.utils import setup_logger
from octavian.halo_reader.halo_utils import membership_top_ids
from time import perf_counter

class DataManager:
  def __init__(self, snapfile: str, log_file: str, config: dict, comm=None):
    t1 = perf_counter()
    # mpi functionality
    self.comm = comm
    self.rank = comm.Get_rank() if comm else 0 # if statement allows for serial case
    self.size = comm.Get_size() if comm else 1

    self.snapfile = snapfile
    self.logger = setup_logger(log_file)
    if comm:
      self.logger.info(f'Initialising DataManager on Rank {self.rank}...')
    else:
      self.logger.info(f'Initialising DataManager...')

    self.initialise_config(config)
    self.initialise_data()
    self.load_simulation_constants()
    self.load_halo_ids()
    self.add_ptype_columns()

    t2 = perf_counter()
    self.logger.info(f'Initialising DataManager done in {t2-t1:.2f} seconds.')


  # programmatic access to attrs based on https://peps.python.org/pep-0363/
  def __getitem__(self, name):
     return getattr(self, name)
  
  def __setitem__(self, name, value):
    return setattr(self, name, value)
  
  def __delitem__(self, name):
    return delattr(self, name)
  
  def __contains__(self, name):
     return hasattr(self, name)
  
  def initialise_config(self, config: dict) -> None:
    self.config = config

    with h5py.File(self.snapfile) as f:
      groups = list(f.keys())

    ptypes = []
    for ptype, ptype_name in self.config['ptype_names'].items():
      if ptype_name in groups:
        ptypes.append(ptype)

    self.config['ptypes'] = ptypes
    self.config['ptypes_baryon'] = [ptype for ptype in ptypes if ptype != 'dm']
    self.config['to_process'] = ptypes + ['baryon', 'total', 'local_densities', 'apertures']
    

  def initialise_data(self) -> None:
    self.data = {}
    self.halo_id_arrays = {}
    self.halo_membership_rows = {}
    
    ptypes = self.config['ptypes']
    
    for ptype in ptypes:
      self.data[ptype] = pd.DataFrame()

    

  def initialise_group_data(self) -> None:
    self.group_data = {}

    config = self.config

    ptypes = config['ptypes']
    groups = config['groups']
    groupIDs = config['groupIDs']

    for group in groups:
      ids = []
      for ptype in ptypes:
        id_column = groupIDs[group]
        if group == 'halos' and ptype in self.halo_id_arrays:
          halo_ids, _ = self.get_halo_membership_rows(ptype)
          ids.append(halo_ids)
        else:
          ids.append(self.data[ptype][id_column].unique())

      ids = np.unique(np.concatenate(ids))
      if group == 'galaxies': ids = ids[ids != -1]

      self.group_data[group] = pd.DataFrame(index=ids)
      self.group_data[group][id_column] = ids

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
      if 'NumPart_Total' in header:
        num_part_total = header['NumPart_Total'].astype(np.int64)
        if 'NumPart_Total_HighWord' in header:
          num_part_total = num_part_total + header['NumPart_Total_HighWord'].astype(np.int64) * 2**32
        self.simulation['num_part_total'] = num_part_total

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
    return unyt.unyt_quantity(1., self.config['code_units'][prop], registry=self.units.registry)
  
  def get_prop_name(self, prop: str) -> str:
    return self.config['prop_aliases'][prop]

  def get_ptype_name(self, ptype: str) -> str:
    return self.config['ptype_names'][ptype]

  def get_column_name(self, prop: str) -> str | list[str]:
    try:
      column = self.config['prop_columns'][prop]
    except KeyError:
      column = prop

    return column
  
  def load_halo_ids(self):
    with h5py.File(self.snapfile) as f:
      for ptype in self.config['ptypes']:
        ptype_name = self.get_ptype_name(ptype)
        ptype_group = f[ptype_name]
        if 'HaloID_array' in ptype_group:
          self.halo_id_arrays[ptype] = ptype_group['HaloID_array'][:].astype(np.int32, copy=False)
        if 'HaloID' in ptype_group:
          halo_ids = ptype_group['HaloID'][:]
        else:
          pid_name = self.get_prop_name('pid')
          if pid_name in ptype_group:
            n_particles = len(ptype_group[pid_name])
          else:
            first_dataset = next(iter(ptype_group.values()))
            n_particles = len(first_dataset)
          halo_ids = np.full(n_particles, -1, dtype=np.int64)
        self.data[ptype]['HaloID'] = pd.Series(halo_ids, dtype='category')

  def get_halo_ids(self, ptype: str, mode: str = 'exclusive') -> np.ndarray:
    if mode == 'exclusive':
      return self.data[ptype]['HaloID'].to_numpy()
    if mode == 'top':
      if ptype not in self.halo_id_arrays:
        raise KeyError(f'HaloID_array not loaded for {ptype}')
      return membership_top_ids(self.halo_id_arrays[ptype])
    raise ValueError(f'Unsupported HaloID mode: {mode}')

  def get_halo_membership_rows(self, ptype: str) -> tuple[np.ndarray, np.ndarray]:
    if ptype not in self.halo_id_arrays:
      return self.get_halo_ids(ptype), np.arange(len(self.data[ptype]))
    if ptype in self.halo_membership_rows:
      return self.halo_membership_rows[ptype]
    halo_id_array = self.halo_id_arrays[ptype]
    if sp.issparse(halo_id_array):
      coo = halo_id_array.tocoo(copy=False)
      rows = (coo.data.astype(np.int64, copy=False), coo.col)
    else:
      particle_rows, array_cols = np.nonzero(halo_id_array >= 0)
      rows = (halo_id_array[particle_rows, array_cols], particle_rows)
    self.halo_membership_rows[ptype] = rows
    return rows
  
  def get_unit_conversion_factor(self, prop: str) -> float:
    try:
      data_units = self.config['prop_units'][prop]
      code_units = self.config['code_units'][prop]
      factor = unyt.unyt_quantity(1., data_units, registry=self.units.registry).to(code_units).d
      if prop == 'rho':
        factor *= self.config['XH'] / unyt.mp.to('g').d
    except KeyError:
      factor = 1.

    return factor

  def add_ptype_columns(self):
    for ptype in self.config['ptypes']:
      self.data[ptype]['ptype'] = pd.Series(np.full(len(self.data[ptype]), ptype), dtype='category')

  def load_property(self, prop: str, ptype: str, optional: bool = False, default=0.):
    column = self.get_column_name(prop)
    prop_name = self.get_prop_name(prop)
    ptype_name = self.get_ptype_name(ptype)
    self.logger.info(f'Loading data: {column}')

    with h5py.File(self.snapfile) as f:
      if prop_name not in f[ptype_name]:
        if optional:
          self.data[ptype][column] = default
          return False
        raise KeyError(f'{prop_name} not found for {ptype_name}')
      if prop == 'metallicity':
        self.data[ptype][column] = f[ptype_name][prop_name][:, 0]
      elif prop == 'age':
        data = f[ptype_name][prop_name][:]
        self.data[ptype][column] = self.simulation['time_gyr'] - self.cosmology.age(1/data - 1).value
      else:
        self.data[ptype][column] = f[ptype_name][prop_name][:] * self.get_unit_conversion_factor(prop)
    return True
  
