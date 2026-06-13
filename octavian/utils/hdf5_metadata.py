from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


COMPLETE_ATTR = 'octavian_complete'
FILE_TYPE_ATTR = 'octavian_file_type'
COMPLETED_AT_ATTR = 'octavian_completed_at_utc'
METADATA_GROUP = 'octavian_metadata'
SIMULATION_GROUP = 'simulation'


_HEADER_SIMULATION_ATTRS = {
  'BoxSize': 'boxsize',
  'Omega0': 'O0',
  'OmegaLambda': 'Ol',
  'HubbleParam': 'h',
  'Redshift': 'redshift',
  'Time': 'a',
}


def _metadata_group(handle: h5py.File):
  return handle.require_group(METADATA_GROUP)


def mark_incomplete(handle: h5py.File, file_type: str | None = None) -> None:
  handle.attrs[COMPLETE_ATTR] = np.bool_(False)
  if file_type is not None:
    handle.attrs[FILE_TYPE_ATTR] = file_type
  if COMPLETED_AT_ATTR in handle.attrs:
    del handle.attrs[COMPLETED_AT_ATTR]


def mark_complete(handle: h5py.File) -> None:
  handle.attrs[COMPLETE_ATTR] = np.bool_(True)
  handle.attrs[COMPLETED_AT_ATTR] = datetime.now(timezone.utc).isoformat()


def validate_complete(handle: h5py.File, path: str | Path | None = None) -> None:
  if COMPLETE_ATTR not in handle.attrs:
    return
  if bool(handle.attrs[COMPLETE_ATTR]):
    return
  label = str(path) if path is not None else handle.filename
  raise RuntimeError(f'{label} is marked as an incomplete Octavian HDF5 file')


def file_type(handle: h5py.File) -> str | None:
  value = handle.attrs.get(FILE_TYPE_ATTR)
  if isinstance(value, bytes):
    return value.decode()
  if value is None:
    return None
  return str(value)


def _serialisable_value(value: Any):
  if hasattr(value, 'd'):
    value = value.d

  arr = np.asarray(value)
  if arr.dtype.kind == 'O':
    return None
  if arr.shape == ():
    scalar = arr.item()
    if isinstance(scalar, (str, bytes, int, float, complex, bool, np.generic)):
      return scalar
    return None
  if arr.dtype.kind in 'biufcS':
    return arr
  return None


def write_simulation_metadata(handle: h5py.File, simulation: dict[str, Any]) -> None:
  group = _metadata_group(handle).require_group(SIMULATION_GROUP)
  for key, value in simulation.items():
    serialisable = _serialisable_value(value)
    if serialisable is not None:
      group.attrs[key] = serialisable


def write_header_simulation_metadata(handle: h5py.File, header_group) -> None:
  simulation = {}
  for key, value in header_group.attrs.items():
    simulation[f'header_{key}'] = value
    canonical = _HEADER_SIMULATION_ATTRS.get(key)
    if canonical is not None:
      simulation[canonical] = value
  write_simulation_metadata(handle, simulation)


def read_simulation_metadata(handle: h5py.File) -> dict[str, Any]:
  path = f'{METADATA_GROUP}/{SIMULATION_GROUP}'
  if path not in handle:
    return {}
  return dict(handle[path].attrs.items())


def copy_simulation_metadata(source: h5py.File, destination: h5py.File) -> bool:
  simulation = read_simulation_metadata(source)
  if not simulation:
    return False
  write_simulation_metadata(destination, simulation)
  return True
