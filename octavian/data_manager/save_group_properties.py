from __future__ import annotations
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import h5py
import os
import numpy as np
from time import perf_counter
from octavian.utils.dataset_columns import resolve_dataset_columns, resolve_list_dataset_columns
from octavian.utils.hdf5_metadata import mark_complete, mark_incomplete, write_simulation_metadata

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _column_for_group(column, group_name: str):
  if isinstance(column, Mapping):
    return column.get(group_name)
  return column


def _has_columns(columns, available_columns) -> bool:
  if columns is None:
    return False
  return bool(np.all(np.isin(columns, available_columns)))


def _write_sequence_dataset(hdf5_group, dataset_name: str, values) -> None:
  sequences = []
  for value in values:
    if value is None:
      sequences.append(np.empty(0, dtype=np.int64))
    else:
      sequences.append(np.asarray(value, dtype=np.int64))

  lengths = np.asarray([len(value) for value in sequences], dtype=np.int32)
  offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
  if len(sequences) == 0 or lengths.sum() == 0:
    indices = np.empty(0, dtype=np.int64)
  else:
    indices = np.concatenate(sequences).astype(np.int64, copy=False)

  hdf5_group.create_dataset(f'{dataset_name}_indices', data=indices)
  hdf5_group.create_dataset(f'{dataset_name}_offsets', data=offsets)
  hdf5_group.create_dataset(f'{dataset_name}_lengths', data=lengths)


def _dataset_exists(hdf5_group, dataset_name: str) -> bool:
  try:
    return isinstance(hdf5_group[dataset_name], h5py.Dataset)
  except KeyError:
    return False


def _write_dataset_if_missing(hdf5_group, dataset_name: str, values) -> None:
  if not _dataset_exists(hdf5_group, dataset_name):
    hdf5_group.create_dataset(dataset_name, data=values)


def _empty_dataset_values(column, length: int):
  if isinstance(column, (list, tuple)):
    return np.empty((length, len(column)), dtype=float)
  return np.empty(length, dtype=float)


def _group_id_values(data_manager: DataManager, group_name: str):
  group_data = data_manager.group_data[group_name]
  id_column = data_manager.config['groupIDs'][group_name]
  if id_column in group_data:
    return group_data[id_column].to_numpy()
  return group_data.index.to_numpy(dtype=np.int64)


def _column_or_default(group_data, column: str, default=0.0):
  if column in group_data:
    return group_data[column].to_numpy()
  return np.full(len(group_data), default, dtype=float)


def _write_rank_completion_schema(data_manager: DataManager, halo_data, galaxy_data) -> None:
  """Write the minimal rank-catalogue schema even when a staged config is stale."""
  halos = data_manager.group_data['halos']
  _write_dataset_if_missing(halo_data, 'groupID', _group_id_values(data_manager, 'halos'))
  _write_dataset_if_missing(halo_data, 'dicts/masses.total', _column_or_default(halos, 'mass_total'))
  _write_dataset_if_missing(
      halo_data,
      'dicts/velocity_dispersions.total',
      _column_or_default(halos, 'velocity_dispersion_total'),
  )
  _write_dataset_if_missing(
      halo_data,
      'dicts/virial_quantities.temperature',
      _column_or_default(halos, 'temperature'),
  )

  if galaxy_data is not None and 'galaxies' in data_manager.group_data:
    galaxies = data_manager.group_data['galaxies']
    _write_dataset_if_missing(galaxy_data, 'groupID', _group_id_values(data_manager, 'galaxies'))
    _write_dataset_if_missing(galaxy_data, 'dicts/masses.total', _column_or_default(galaxies, 'mass_total'))


def save_group_properties(data_manager: DataManager, filename: str) -> None:
  data_manager.logger.info('Saving datasets...')
  t1 = perf_counter()

  config = data_manager.config

  if os.path.exists(filename):
    os.remove(filename)

  with h5py.File(filename, 'w') as f:
    mark_incomplete(f, 'rank_output')
    write_simulation_metadata(f, data_manager.simulation)

    halo_data = f.create_group('halo_data')
    halo_columns = data_manager.group_data['halos'].columns

    if 'galaxies' in config['groups']:
      galaxy_data = f.create_group('galaxy_data')
      galaxy_columns = data_manager.group_data['galaxies'].columns
    else:
      galaxy_columns = []

    # write particle lists in flat CSR format
    ptype_lists = ['glist', 'slist', 'dmlist', 'bhlist']
    for group_name, hdf5_group in [('halos', halo_data), ('galaxies', galaxy_data if 'galaxies' in config['groups'] else None)]:
      if hdf5_group is None:
        continue
      for ptype_list in ptype_lists:
        if ptype_list not in data_manager.particle_lists[group_name]:
          continue
        pl = data_manager.particle_lists[group_name][ptype_list]
        hdf5_group.create_dataset(f'{ptype_list}_indices', data=pl['indices'])
        hdf5_group.create_dataset(f'{ptype_list}_offsets', data=pl['offsets'])
        hdf5_group.create_dataset(f'{ptype_list}_lengths', data=pl['lengths'])

    # write all other datasets
    for dataset_name, column in resolve_dataset_columns(config).items():
      if dataset_name in ptype_lists:
        continue

      halo_column = _column_for_group(column, 'halos')
      if _has_columns(halo_column, halo_columns):
        halo_data.create_dataset(dataset_name, data=data_manager.group_data['halos'][halo_column].to_numpy())
      elif halo_column is not None and len(data_manager.group_data['halos']) == 0:
        halo_data.create_dataset(dataset_name, data=_empty_dataset_values(halo_column, 0))

      galaxy_column = _column_for_group(column, 'galaxies')
      if 'galaxies' in config['groups'] and _has_columns(galaxy_column, galaxy_columns):
        galaxy_data.create_dataset(dataset_name, data=data_manager.group_data['galaxies'][galaxy_column].to_numpy())
      elif 'galaxies' in config['groups'] and galaxy_column is not None and len(data_manager.group_data['galaxies']) == 0:
        galaxy_data.create_dataset(dataset_name, data=_empty_dataset_values(galaxy_column, 0))

    for dataset_name, column in resolve_list_dataset_columns(config).items():
      halo_column = _column_for_group(column, 'halos')
      if isinstance(halo_column, str) and halo_column in halo_columns:
        _write_sequence_dataset(halo_data, dataset_name, data_manager.group_data['halos'][halo_column].to_numpy(dtype=object))

      galaxy_column = _column_for_group(column, 'galaxies')
      if 'galaxies' in config['groups'] and isinstance(galaxy_column, str) and galaxy_column in galaxy_columns:
        _write_sequence_dataset(galaxy_data, dataset_name, data_manager.group_data['galaxies'][galaxy_column].to_numpy(dtype=object))

    _write_rank_completion_schema(data_manager, halo_data, galaxy_data if 'galaxies' in config['groups'] else None)

    mark_complete(f)

  t2 = perf_counter()
  data_manager.logger.info(f'Saving datasets done in {t2-t1:.2f} seconds.')
