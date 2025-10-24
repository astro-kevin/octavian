import numpy as np
import h5py
import unyt
from astropy.cosmology import FlatLambdaCDM
from sympy import sympify
from typing import Optional, Sequence, Dict, Iterable, List, Tuple

from octavian.backend import pd, USING_MODIN, activate_modin, deactivate_modin

import octavian.constants as c


class DataManager:
  def __init__(
    self,
    snapfile: str,
    fraction: Sequence[float] = (0, 1),
    mode: str = 'fof',
    include_unassigned: Optional[bool] = None,
    use_modin: Optional[bool] = None,
    particle_indices: Optional[Dict[str, Iterable[int]]] = None,
    particle_ids: Optional[Dict[str, Iterable[int]]] = None,
    map_threads: Optional[int] = None,
  ):
    self.snapfile = snapfile
    self.mode = mode.lower()
    self.start_fraction = fraction[0]
    self.end_fraction = fraction[1]

    if use_modin is None:
      self.use_modin = USING_MODIN
    else:
      self.use_modin = bool(use_modin)
      if self.use_modin and not USING_MODIN:
        raise ImportError('Modin was requested but is not available. Install modin to enable this backend.')
    if self.use_modin:
      activate_modin()
    else:
      deactivate_modin()

    self._particle_indices: Dict[str, np.ndarray] = {}
    self._map_threads = max(1, int(map_threads)) if map_threads else 1
    if particle_indices:
      for ptype, values in particle_indices.items():
        arr = np.fromiter((int(v) for v in values), dtype=np.int64)
        if arr.size:
          self._particle_indices[ptype] = np.unique(arr)
        else:
          self._particle_indices[ptype] = arr

    if particle_ids:
      mapped = self._map_particle_ids_to_indices(particle_ids)
      for ptype, indices in mapped.items():
        if ptype in self._particle_indices:
          combined = np.unique(np.concatenate([self._particle_indices[ptype], indices]))
          self._particle_indices[ptype] = combined
        else:
          self._particle_indices[ptype] = indices

    self.load_simulation_constants()

    keep_unassigned = include_unassigned if include_unassigned is not None else (self.mode == 'ahf-fast')
    self.initialise_dataframes_with_haloids(remove_unassigned=not keep_unassigned)
    self.ensure_membership_columns()

    if self.mode in ('fof', 'ahf'):
      self.validate_halos()
    else:
      self.haloIDs = np.array([], dtype=int)

    self.load_masses()

    if self.mode in ('fof', 'ahf'):
      self.halos = pd.DataFrame(index=self.haloIDs)
      self.load_halo_pids()
    else:
      self.halos = pd.DataFrame()

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

  def initialise_dataframes_with_haloids(self, remove_unassigned: bool = True) -> None:
    with h5py.File(self.snapfile) as f:
      for ptype, name in c.ptypes.items():
        frame = pd.DataFrame()
        index_subset = None
        if ptype in self._particle_indices:
          index_subset = self._particle_indices[ptype]

        haloid_dataset = f[name]['HaloID']
        if index_subset is None:
          haloids = haloid_dataset[:]
          dataset_indices = np.arange(len(haloids), dtype=np.int64)
        else:
          haloids = haloid_dataset[index_subset]
          dataset_indices = index_subset.astype(np.int64, copy=False)

        if remove_unassigned:
          selection = haloids != 0
          haloids = haloids[selection]
          dataset_indices = dataset_indices[selection]

        frame['HaloID'] = haloids
        frame['ptype'] = c.ptype_ids[ptype]
        frame.index = dataset_indices
        frame.rename_axis(index='pid', inplace=True)
        frame = frame.reset_index().sort_values(by=['HaloID', 'pid']).set_index('pid')
        setattr(self, ptype, frame)


  def _map_particle_ids_to_indices(self, particle_ids: Dict[str, Iterable[int]]) -> Dict[str, np.ndarray]:
    index_map: Dict[str, np.ndarray] = {}
    with h5py.File(self.snapfile) as f:
      for ptype, ids in particle_ids.items():
        ids_array = np.asarray(ids, dtype=np.int64)
        if ids_array.size == 0:
          index_map[ptype] = np.array([], dtype=np.int64)
          continue

        dataset = f[c.ptypes[ptype]][c.prop_aliases['pid']]
        ids_unique = np.unique(ids_array).astype(np.int64, copy=False)

        total_rows = dataset.shape[0]
        print(
          f"  Mapping {ptype}: {ids_unique.size} particle IDs across {total_rows} snapshot rows",
          flush=True,
        )

        if total_rows == 0 or ids_unique.size == 0:
          index_map[ptype] = np.array([], dtype=np.int64)
          continue

        remaining = set(int(i) for i in ids_unique.tolist())
        mapping: Dict[int, int] = {}

        # Read the HDF5 dataset in manageable chunks to avoid loading it entirely into memory.
        chunk_size = min(max(500_000, len(ids_unique) * 20), 2_000_000)

        ids_unique_array = ids_unique

        for start in range(0, total_rows, chunk_size):
          end = min(start + chunk_size, total_rows)
          chunk = dataset[start:end]
          if chunk.size == 0:
            continue

          chunk_array = np.asarray(chunk, dtype=np.int64)
          mask = np.isin(chunk_array, ids_unique_array, assume_unique=True)
          if mask.any():
            matches = chunk_array[mask]
            positions = np.nonzero(mask)[0]
            for local_pos, value in zip(positions, matches):
              ivalue = int(value)
              if ivalue in remaining and ivalue not in mapping:
                mapping[ivalue] = int(start + local_pos)
                remaining.discard(ivalue)
          if not remaining:
            break

        if remaining:
          sample = list(sorted(remaining))[:10]
          raise ValueError(f"Missing particle IDs for {ptype}: {sample} ...")

        index_map[ptype] = np.sort(np.fromiter(mapping.values(), dtype=np.int64))

    return index_map

  @staticmethod
  def _fetch_dataset(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
      shape = (0,) + dataset.shape[1:]
      return np.empty(shape, dtype=dataset.dtype)

    order = np.argsort(indices)
    sorted_idx = indices[order]
    values = dataset[sorted_idx]
    if not np.all(order == np.arange(order.size)):
      inverse = np.empty_like(order)
      inverse[order] = np.arange(order.size)
      values = values[inverse]
    return values

  def ensure_membership_columns(self) -> None:
    for ptype in c.ptypes.keys():
      if 'GalID' not in self[ptype]:
        self[ptype]['GalID'] = -1
      else:
        self[ptype]['GalID'] = self[ptype]['GalID'].astype(int)

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

    valid_set = set(valid_halos.tolist())
    for ptype in c.ptypes.keys():
      frame = self[ptype]
      if frame.empty:
        continue
      mask = frame['HaloID'].isin(valid_set)
      self[ptype] = frame.loc[mask]

  def load_halo_pids(self) -> None:
    for ptype in c.ptypes.keys():
      frame = self[ptype]
      if 'HaloID' not in frame or len(frame) == 0:
        plist = pd.Series(dtype=object)
      else:
        plist = frame.groupby(by='HaloID').apply(lambda x: x.index.to_numpy())
      self.halos[c.ptype_lists[ptype]] = plist

    for plist in c.ptype_lists.values():
      default = pd.Series([[] for _ in range(len(self.halos.index))], index=self.halos.index, dtype=object)
      if plist in self.halos:
        existing = self.halos[plist]
        default.loc[existing.index] = existing
      self.halos[plist] = default

  def load_galaxy_pids(self) -> None:
    for ptype in c.ptypes.keys():
      if ptype == 'dm': continue

      data = self[ptype].loc[self[ptype]['GalID'] != -1].copy()
      if len(data) == 0:
        plist = pd.Series(dtype=object)
      else:
        plist = data.groupby(by='GalID').apply(lambda x: x.index.to_numpy())
      self.galaxies[c.ptype_lists[ptype]] = plist

    for plist in c.ptype_lists.values():
      if plist in self.galaxies:
        self.galaxies[plist] = self.galaxies[plist].fillna({i: [] for i in self.galaxies.index})

  def load_masses(self):
    with h5py.File(self.snapfile) as f:
      for ptype, id in c.ptypes.items():
        dataset = c.prop_aliases['bhmass'] if ptype == 'bh' else c.prop_aliases['mass']
        masses_dset = f[id][dataset]
        indexer = self[ptype].index.to_numpy(dtype=np.int64, copy=False)
        factor = self.get_unit_conversion_factor('mass')

        masses = self._fetch_dataset(masses_dset, indexer) * factor

        self[ptype]['mass'] = masses

        self[f'n{ptype}'] = masses_dset.shape[0]

        total_mass = 0.0
        chunk = max(1, min(1_000_000, masses_dset.shape[0]))
        for start in range(0, masses_dset.shape[0], chunk):
          block = masses_dset[start:start + chunk]
          if block.size:
            total_mass += block.sum()
        self[f'm{ptype}_total'] = total_mass * factor

  def ensure_property(self, prop: str, ptype: str) -> None:
    column = self.get_column_name(prop)
    frame = self[ptype]
    if isinstance(column, list):
      if all(col in frame for col in column):
        return
    else:
      if column in frame:
        return
    self.load_property(prop, ptype)

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
      quantity = unyt.unyt_quantity(1., data_units, registry=self.units.registry)
      factor = quantity.to_value(code_units)
      if prop == 'rho':
        factor *= c.XH / unyt.mp.to('g').d
    except KeyError:
      factor = 1.0

    return float(factor)
  
  def load_property(self, requested_prop: str, requested_ptype: str):
    prop = self.get_prop_name(requested_prop)
    ptype = self.get_ptype_name(requested_ptype)
    ptype_key = requested_ptype.strip(' _').lower()
    column = self.get_column_name(requested_prop)

    with h5py.File(self.snapfile) as f:
      dataset = f[ptype][prop]
      indexer = self[requested_ptype].index.to_numpy(dtype=np.int64, copy=False)
      values = self._fetch_dataset(dataset, indexer)
      has_data = values.size != 0

      if isinstance(column, list):
        factor = self.get_unit_conversion_factor(requested_prop)
        if has_data:
          scaled = values * factor
          for idx, col in enumerate(column):
            self[requested_ptype][col] = scaled[:, idx]
        else:
          for col in column:
            self[requested_ptype][col] = np.array([], dtype=float)
      else:
        if requested_prop == 'metallicity':
          self[requested_ptype][column] = values[:, 0] if has_data else np.array([])
        elif requested_prop == 'age':
          if has_data:
            selection = values
            self[requested_ptype][column] = self.simulation['time_gyr'] - self.cosmology.age(1/selection - 1).value
          else:
            self[requested_ptype][column] = np.array([])
        else:
          factor = self.get_unit_conversion_factor(requested_prop)
          self[requested_ptype][column] = values * factor if has_data else np.array([])
