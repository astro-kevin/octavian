import numpy as np
import pandas as pd
import h5py
import unyt
from astropy.cosmology import FlatLambdaCDM
from sympy import sympify
from typing import Optional, Sequence, Dict, Any, Iterable, List, Tuple

try:
  import polars as pl  # type: ignore
  HAS_POLARS = True
except Exception:  # pragma: no cover - optional dependency
  pl = None  # type: ignore
  HAS_POLARS = False

import octavian.constants as c


class DataManager:
  def __init__(
    self,
    snapfile: str,
    fraction: Sequence[float] = (0, 1),
    mode: str = 'fof',
    include_unassigned: Optional[bool] = None,
    use_polars: Optional[bool] = None,
    particle_indices: Optional[Dict[str, Iterable[int]]] = None,
    particle_ids: Optional[Dict[str, Iterable[int]]] = None,
    map_threads: Optional[int] = None,
  ):
    self.snapfile = snapfile
    self.mode = mode.lower()
    self.start_fraction = fraction[0]
    self.end_fraction = fraction[1]

    if not HAS_POLARS:
      self.use_polars = False
    elif use_polars is None:
      self.use_polars = True
    else:
      self.use_polars = bool(use_polars)
    self._polars_tables: Dict[str, "pl.DataFrame"] = {}
    self._polars_units: Dict[str, Dict[str, unyt.Unit]] = {}
    self._halos_polars: Optional["pl.DataFrame"] = None
    self._galaxies_polars: Optional["pl.DataFrame"] = None
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
      if self.use_polars and HAS_POLARS:
        self._halos_polars = pl.DataFrame({'HaloID': self.halos.index.to_numpy(dtype=np.int64, copy=False)})
    else:
      self.halos = pd.DataFrame()

    if self.use_polars and HAS_POLARS and not self.halos.empty:
      halos_pd = self.halos.reset_index()
      halos_pd.rename(columns={'index': 'HaloID'}, inplace=True)
      self._halos_polars = pl.from_pandas(halos_pd)

  # programmatic access to attrs based on https://peps.python.org/pep-0363/
  def __getitem__(self, name):
     return getattr(self, name)
  
  def __setitem__(self, name, value):
    setattr(self, name, value)
    if not self.use_polars:
      return
    if isinstance(name, str) and name in c.ptypes.keys():
      self._invalidate_polars(name)
    elif name == 'halos':
      self._invalidate_collection_polars('halos')
    elif name == 'galaxies':
      self._invalidate_collection_polars('galaxies')
  
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

        if self.use_polars and HAS_POLARS:
          table = pl.DataFrame({
            'pid': frame.index.to_numpy(dtype=np.int64, copy=False),
            'HaloID': frame['HaloID'].to_numpy(dtype=np.int64, copy=False),
            'ptype': np.full(len(frame), c.ptype_ids[ptype], dtype=np.int64),
          })
          table = table.sort(['HaloID', 'pid'])
          self._polars_tables[ptype] = table
          self._polars_units[ptype] = {}

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

        print(
          f"  Mapping {ptype}: {ids_unique.size} particle IDs across {dataset.shape[0]} snapshot rows",
          flush=True,
        )

        pid_values = dataset[:]
        if pid_values.size == 0:
          index_map[ptype] = np.array([], dtype=np.int64)
          continue

        worker_count = max(1, self._map_threads)
        chunk_size = max(1, int(np.ceil(ids_unique.size / worker_count)))

        def _locate(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
          pos = np.searchsorted(pid_values, chunk)
          valid = (pos < pid_values.size) & (pid_values[pos] == chunk)
          return pos[valid], chunk[~valid]

        if worker_count == 1 or ids_unique.size < 2 * chunk_size:
          positions, missing_ids = _locate(ids_unique)
        else:
          from concurrent.futures import ThreadPoolExecutor

          positions_list: List[np.ndarray] = []
          missing_list: List[np.ndarray] = []
          with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = []
            for start in range(0, ids_unique.size, chunk_size):
              chunk = ids_unique[start:start + chunk_size]
              futures.append(pool.submit(_locate, chunk))
            for fut in futures:
              pos, missing_chunk = fut.result()
              if pos.size:
                positions_list.append(pos)
              if missing_chunk.size:
                missing_list.append(missing_chunk)
          positions = np.concatenate(positions_list) if positions_list else np.array([], dtype=np.int64)
          missing_ids = np.concatenate(missing_list) if missing_list else np.array([], dtype=np.int64)

        if missing_ids.size:
          raise ValueError(f"Missing particle IDs for {ptype}: {missing_ids[:10].tolist()} ...")

        index_map[ptype] = positions.astype(np.int64, copy=False)

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

      if self.use_polars and HAS_POLARS and ptype in self._polars_tables:
        gal_values = self[ptype]['GalID'].to_numpy(dtype=np.int64, copy=False)
        table = self._polars_tables[ptype]
        table = table.with_columns(pl.Series('GalID', gal_values))
        self._polars_tables[ptype] = table
        self._polars_units.setdefault(ptype, {})['GalID'] = None

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
      if self.use_polars and HAS_POLARS and ptype in self._polars_tables:
        table = self._polars_tables[ptype]
        table = table.filter(pl.col('HaloID').is_in(valid_halos))
        self._polars_tables[ptype] = table

  def load_halo_pids(self) -> None:
    for ptype in c.ptypes.keys():
      frame = self[ptype]
      if 'HaloID' not in frame or len(frame) == 0:
        plist = pd.Series(dtype=object)
      else:
        plist = frame.groupby(by='HaloID').apply(lambda x: x.index.to_numpy())
      self.halos[c.ptype_lists[ptype]] = plist

    for plist in c.ptype_lists.values():
      self.halos[plist] = self.halos.get(plist, pd.Series(dtype=object)).fillna({i: [] for i in self.halos.index})
    self._invalidate_collection_polars('halos')

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
    self._invalidate_collection_polars('galaxies')

  def load_masses(self):
    with h5py.File(self.snapfile) as f:
      for ptype, id in c.ptypes.items():
        dataset = c.prop_aliases['bhmass'] if ptype == 'bh' else c.prop_aliases['mass']
        masses_dset = f[id][dataset]
        indexer = self[ptype].index.to_numpy(dtype=np.int64, copy=False)
        factor = self.get_unit_conversion_factor('mass')

        masses = self._fetch_dataset(masses_dset, indexer) * factor

        self[ptype]['mass'] = masses

        if self.use_polars and HAS_POLARS and ptype in self._polars_tables:
          table = self._polars_tables[ptype]
          if len(indexer) == 0:
            table = table.with_columns(pl.Series('mass', [], dtype=pl.Float64))
          else:
            table = table.with_columns(pl.Series('mass', masses))
          self._polars_tables[ptype] = table
          self._polars_units.setdefault(ptype, {})['mass'] = self.create_unit_quantity('mass').units
        else:
          if self.use_polars:
            self._invalidate_polars(ptype)

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
    if self.use_polars:
      self._invalidate_polars(ptype)

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

    if self.use_polars and HAS_POLARS and ptype_key in self._polars_tables:
      table = self._polars_tables[ptype_key]

      def _series_for(col_name: str) -> pl.Series:
        pandas_values = self[requested_ptype][col_name].to_numpy(copy=False)
        dtype = pl.Float64 if pandas_values.dtype.kind in {'f', 'i'} else None
        return pl.Series(col_name, pandas_values, dtype=dtype)

      if isinstance(column, list):
        for col_name in column:
          table = table.with_columns(_series_for(col_name))
      else:
        table = table.with_columns(_series_for(column))

      self._polars_tables[ptype_key] = table
      units = self._polars_units.setdefault(ptype_key, {})
      if isinstance(column, list):
        for col_name in column:
          units[col_name] = self.create_unit_quantity(requested_prop).units if requested_prop in c.code_units else None
      else:
        units[column] = self.create_unit_quantity(requested_prop).units if requested_prop in c.code_units else None
    elif self.use_polars:
      self._invalidate_polars(ptype_key)

  # -- Polars helpers --------------------------------------------------

  def _invalidate_polars(self, ptype: Optional[str] = None) -> None:
    if not self.use_polars:
      return
    if ptype is None:
      self._polars_tables.clear()
      self._polars_units.clear()
    elif ptype in self._polars_tables:
      self._polars_tables.pop(ptype, None)
      self._polars_units.pop(ptype, None)

  def _invalidate_collection_polars(self, collection: str) -> None:
    if not self.use_polars or not HAS_POLARS:
      return
    if collection == 'halos':
      self._halos_polars = None
    elif collection == 'galaxies':
      self._galaxies_polars = None

  def get_collection_polars(self, collection: str) -> "pl.DataFrame":
    if not self.use_polars or not HAS_POLARS:
      raise RuntimeError('Polars collections requested but Polars mode is disabled.')
    if collection == 'halos':
      if self._halos_polars is None:
        halos_pd = self['halos'].reset_index()
        first_col = halos_pd.columns[0]
        if first_col != 'HaloID':
          halos_pd.rename(columns={first_col: 'HaloID'}, inplace=True)
        self._halos_polars = pl.from_pandas(halos_pd)
      return self._halos_polars.clone()
    if collection == 'galaxies':
      if self._galaxies_polars is None:
        galaxies_pd = self['galaxies'].reset_index()
        first_col = galaxies_pd.columns[0]
        if first_col != 'GalID':
          galaxies_pd.rename(columns={first_col: 'GalID'}, inplace=True)
        self._galaxies_polars = pl.from_pandas(galaxies_pd)
      return self._galaxies_polars.clone()
    raise KeyError(f'Unknown collection: {collection}')

  def set_collection_polars(self, collection: str, table: "pl.DataFrame") -> None:
    if not self.use_polars or not HAS_POLARS:
      raise RuntimeError('Polars collections requested but Polars mode is disabled.')
    if collection == 'halos':
      self._halos_polars = table
      halos_pd = table.to_pandas()
      if 'HaloID' in halos_pd.columns:
        halos_pd.set_index('HaloID', inplace=True)
      else:
        halos_pd.index.name = 'HaloID'
      self.halos = halos_pd
      return
    if collection == 'galaxies':
      self._galaxies_polars = table
      galaxies_pd = table.to_pandas()
      if 'GalID' in galaxies_pd.columns:
        galaxies_pd.set_index('GalID', inplace=True)
      else:
        galaxies_pd.index.name = 'GalID'
      self.galaxies = galaxies_pd
      return
    raise KeyError(f'Unknown collection: {collection}')

  def _convert_series_to_numeric(self, series: pd.Series) -> tuple[np.ndarray, Optional[unyt.Unit]]:
    if series.empty:
      return series.to_numpy(), None
    values = series.to_numpy()
    if isinstance(values, unyt.unyt_array):
      unit = values.units
      return values.to_value(unit), unit
    first_valid = None
    for val in series:
      if isinstance(val, unyt.unyt_quantity) or isinstance(val, unyt.unyt_array) or (hasattr(val, 'to_value') and hasattr(val, 'units')):
        first_valid = val
        break
    if first_valid is not None:
      unit = getattr(first_valid, 'units', None)
      if unit is None and isinstance(first_valid, unyt.unyt_array):
        unit = first_valid.units

      def _to_value(val):
        if isinstance(val, (unyt.unyt_quantity, unyt.unyt_array)):
          return val.to_value(unit)
        if hasattr(val, 'to_value'):
          try:
            return val.to_value(unit)
          except Exception:
            pass
        if val is None:
          return np.nan
        return float(val)

      numeric = np.array([_to_value(val) for val in series], dtype=np.float64)
      return numeric, unit
    return series.to_numpy(), None

  def get_polars_table(self, ptype: str, *, include_index: bool = True, mutable: bool = False) -> "pl.DataFrame":
    if not self.use_polars:
      raise RuntimeError('Polars tables requested but Polars mode is disabled.')
    if ptype not in c.ptypes.keys():
      raise KeyError(f'Unknown particle type: {ptype}')

    if ptype not in self._polars_tables:
      frame = self[ptype]
      data: Dict[str, np.ndarray] = {}
      units: Dict[str, unyt.Unit] = {}
      if include_index:
        data['pid'] = frame.index.to_numpy(dtype=np.int64, copy=False)
      for column in frame.columns:
        series = frame[column]
        numeric, unit = self._convert_series_to_numeric(series)
        data[column] = numeric
        if unit is not None:
          units[column] = unit
      self._polars_tables[ptype] = pl.DataFrame(data)
      self._polars_units[ptype] = units
    table = self._polars_tables[ptype]
    if mutable:
      return table
    clone = table.clone()
    if not include_index and 'pid' in clone.columns:
      clone = clone.drop('pid')
    return clone

  def set_ptype_from_polars(self, ptype: str, table: "pl.DataFrame") -> None:
    if not self.use_polars:
      raise RuntimeError('Polars tables requested but Polars mode is disabled.')
    if ptype not in c.ptypes.keys():
      raise KeyError(f'Unknown particle type: {ptype}')
    pdf = table.to_pandas()
    if 'pid' in pdf.columns:
      pdf.set_index('pid', inplace=True)
    units = self._polars_units.get(ptype, {})
    for column, unit in units.items():
      if column in pdf.columns:
        pdf[column] = unyt.unyt_array(pdf[column].to_numpy(), unit, registry=self.units.registry)
    object.__setattr__(self, ptype, pdf)
    self._polars_tables[ptype] = table

  def to_polars(self) -> Dict[str, "pl.DataFrame"]:
    if not self.use_polars:
      raise RuntimeError('Polars tables requested but Polars mode is disabled.')
    tables: Dict[str, "pl.DataFrame"] = {}
    for ptype in c.ptypes.keys():
      tables[ptype] = self.get_polars_table(ptype)
    return tables
