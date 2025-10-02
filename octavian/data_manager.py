import numpy as np
import pandas as pd
import h5py
import unyt
from astropy.cosmology import FlatLambdaCDM
from sympy import sympify
from typing import Optional, Sequence, Dict, Any

try:
  import polars as pl  # type: ignore
  HAS_POLARS = True
except Exception:  # pragma: no cover - optional dependency
  pl = None  # type: ignore
  HAS_POLARS = False

import octavian.constants as c


class DataManager:
  def __init__(self, snapfile: str, fraction: Sequence[float] = (0, 1), mode: str = 'fof', include_unassigned: Optional[bool] = None, use_polars: Optional[bool] = None):
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
    setattr(self, name, value)
    if self.use_polars and isinstance(name, str) and name in c.ptypes.keys():
      self._invalidate_polars(name)
  
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
        haloids = f[name]['HaloID'][:]
        if remove_unassigned:
          selection = haloids != 0
          frame['HaloID'] = haloids[selection]
          frame.index = np.arange(len(haloids))[selection]
        else:
          frame['HaloID'] = haloids
          frame.index = np.arange(len(haloids))

        frame['ptype'] = c.ptype_ids[ptype]
        frame.sort_values(by='HaloID', inplace=True)
        frame.rename_axis(index='pid', inplace=True)
        setattr(self, ptype, frame)

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

    for ptype in c.ptypes.keys():
        self[ptype] = self[ptype].query('HaloID in @valid_halos')

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
        masses = f[id][dataset][:] * self.get_unit_conversion_factor('mass')
        self[ptype]['mass'] = masses[self[ptype].index]

        if self.use_polars:
          self._invalidate_polars(ptype)
        
        self[f'n{ptype}'] = len(masses)
        self[f'm{ptype}_total'] = np.sum(masses)

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
      data = f[ptype][prop][:]
      indexer = self[requested_ptype].index
      if isinstance(column, list):
        values = data[indexer]
        for idx, col in enumerate(column):
          self[requested_ptype][col] = values[:, idx] * self.get_unit_conversion_factor(requested_prop)
      else:
        if requested_prop == 'metallicity':
          self[requested_ptype][column] = data[:, 0][indexer]
        elif requested_prop == 'age':
          selection = data[indexer]
          self[requested_ptype][column] = self.simulation['time_gyr'] - self.cosmology.age(1/selection - 1).value
        else:
          factor = self.get_unit_conversion_factor(requested_prop)
          self[requested_ptype][column] = data[indexer] * factor
    if self.use_polars:
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

  def _convert_series_to_numeric(self, series: pd.Series) -> tuple[np.ndarray, Optional[unyt.Unit]]:
    if series.empty:
      return series.to_numpy(), None
    values = series.to_numpy()
    if isinstance(values, unyt.unyt_array):
      unit = values.units
      return values.to_value(unit), unit
    first = series.iloc[0]
    if isinstance(first, unyt.unyt_quantity):
      unit = first.units
      return np.array([val.to_value(unit) for val in series], dtype=np.float64), unit
    if hasattr(first, 'to_value') and hasattr(first, 'units'):
      unit = first.units
      return np.array([val.to_value(unit) for val in series], dtype=np.float64), unit
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
    return table if mutable else table.clone()

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
