from __future__ import annotations
import numpy as np
import pandas as pd
import unyt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager


# handle periodic boundary for fof and CoM calculations, reset out of bounds positions during save
def _ensure_position_units(series: pd.Series, target_unit: unyt.unyt_unit, registry) -> unyt.unyt_array:
  values = series.to_numpy()
  if len(values) == 0:
    return unyt.unyt_array(values, target_unit, registry=registry)

  if isinstance(values, unyt.unyt_array):
    if values.units.is_dimensionless:
      return unyt.unyt_array(values.value, target_unit, registry=registry)
    return values.to(target_unit)

  if values.dtype == object:
    converted = []
    for val in values:
      if isinstance(val, unyt.unyt_array):
        converted.append(val.to_value(target_unit))
      elif isinstance(val, unyt.unyt_quantity):
        converted.append(val.to_value(target_unit))
      elif hasattr(val, 'to_value'):
        converted.append(val.to_value(target_unit))
      elif hasattr(val, 'to'):
        converted.append(val.to(target_unit).value)
      elif hasattr(val, 'units') and hasattr(val, 'value'):
        converted.append(unyt.unyt_quantity(val.value, val.units, registry=registry).to_value(target_unit))
      else:
        converted.append(float(val))
    return unyt.unyt_array(np.asarray(converted, dtype=np.float64), target_unit, registry=registry)

  return unyt.unyt_array(values.astype(np.float64, copy=False), target_unit, registry=registry)


def wrap_positions(data_manager: DataManager) -> None:
  boxsize = data_manager.simulation['boxsize']/data_manager.simulation['h']
  position_unit = unyt.unyt_quantity(1., 'kpc*a', registry=data_manager.units.registry).units

  # select halos with particles near boundary
  halos_to_wrap = {}
  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pos', ptype)
    for direction in ['x', 'y', 'z']:
      series = data_manager[ptype][direction]
      data_manager[ptype].loc[:, direction] = _ensure_position_units(series, position_unit, data_manager.units.registry)

  data = pd.concat([data_manager[ptype] for ptype in ['gas', 'dm', 'star', 'bh']])
  halos_grouped = data.groupby(by='HaloID')
  for direction in ['x', 'y', 'z']:
    check_wrap = (halos_grouped[direction].max() - halos_grouped[direction].min()) > 0.5*boxsize
    halos_to_wrap[direction] = check_wrap[check_wrap].index.unique()

  # wrap positions
  for ptype in ['gas', 'dm', 'star', 'bh']:
    halos_grouped = data_manager[ptype].groupby(by='HaloID')
    for direction in ['x', 'y', 'z']:
      in_halos_to_wrap = np.isin(data_manager[ptype]['HaloID'], halos_to_wrap[direction])
      too_high = data_manager[ptype][direction] > 0.5*boxsize

      data_manager[ptype].loc[in_halos_to_wrap & too_high, direction] -= boxsize
