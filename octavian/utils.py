from __future__ import annotations
import numpy as np
import pandas as pd
import unyt

from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

try:
  import polars as pl  # type: ignore
  HAS_POLARS = True
except Exception:  # pragma: no cover - optional dependency
  pl = None  # type: ignore
  HAS_POLARS = False


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
  half_box = 0.5 * boxsize
  registry = data_manager.units.registry
  position_unit = unyt.unyt_quantity(1., 'kpc*a', registry=registry).units

  use_polars = getattr(data_manager, 'use_polars', False)
  if use_polars and not HAS_POLARS:
    raise RuntimeError('Polars requested for wrap_positions but Polars is unavailable.')

  halos_to_wrap = {direction: set() for direction in ['x', 'y', 'z']}
  cached_positions: Dict[str, Dict[str, np.ndarray]] = {}

  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pos', ptype)
    frame = data_manager[ptype]
    if frame.empty:
      continue

    halo_ids = frame['HaloID'].to_numpy()
    if halo_ids.size == 0:
      continue

    pos_columns: Dict[str, np.ndarray] = {}
    for direction in ['x', 'y', 'z']:
      series = frame[direction]
      pos_columns[direction] = _ensure_position_units(series, position_unit, registry).to_value(position_unit)

    cached_positions[ptype] = pos_columns

    local = pd.DataFrame(pos_columns)
    local['HaloID'] = halo_ids
    grouped = local.groupby('HaloID', sort=False)

    for direction in ['x', 'y', 'z']:
      span = grouped[direction].max() - grouped[direction].min()
      to_wrap = span.index[span > half_box]
      if len(to_wrap):
        halos_to_wrap[direction].update(to_wrap.to_numpy())

  for ptype in ['gas', 'dm', 'star', 'bh']:
    frame = data_manager[ptype]
    if frame.empty:
      continue
    halo_ids = frame['HaloID'].to_numpy()
    if halo_ids.size == 0:
      continue
    positions = cached_positions.get(ptype)
    if positions is None:
      continue

    for direction in ['x', 'y', 'z']:
      if not halos_to_wrap[direction]:
        continue
      in_halos = np.isin(halo_ids, list(halos_to_wrap[direction]))
      if not in_halos.any():
        continue
      coords = positions[direction]
      mask = in_halos & (coords > half_box)
      if mask.any():
        coords = coords.copy()
        coords[mask] -= boxsize
        positions[direction] = coords

    for direction in ['x', 'y', 'z']:
      frame.loc[:, direction] = unyt.unyt_array(positions[direction], position_unit, registry=registry)

  if use_polars and HAS_POLARS:
    for ptype in ['gas', 'dm', 'star', 'bh']:
      data_manager._invalidate_polars(ptype)
