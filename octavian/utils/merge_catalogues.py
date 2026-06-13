import h5py
import numpy as np
import warnings
from collections.abc import Mapping
from yaml import safe_load
from octavian.halo_reader import halo_source_metadata_schema
from octavian.utils.dataset_columns import resolve_dataset_columns, resolve_list_dataset_columns
from octavian.utils.hdf5_metadata import (
  mark_complete,
  mark_incomplete,
  read_simulation_metadata,
  validate_complete,
  write_simulation_metadata,
)
from octavian.utils.local_densities import calculate_local_density_arrays

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _column_for_group(column, group_name: str):
  if isinstance(column, Mapping):
    return column.get(group_name)
  return column


def _empty_values(dataset: str, length: int):
  if dataset in {'pos', 'vel', 'minpotpos', 'minpotvel'} or '_L' in dataset:
    return np.full((length, 3), np.nan)
  return np.full(length, np.nan)


def _replace_dataset(group, dataset: str, values) -> None:
  parent_path, _, name = dataset.rpartition('/')
  parent = group.require_group(parent_path) if parent_path else group
  if name in parent:
    del parent[name]
  parent.create_dataset(name, data=values)


def _dataset_for_column(config, group_name: str, wanted):
  for dataset, column in resolve_dataset_columns(config).items():
    group_column = _column_for_group(column, group_name)
    if isinstance(wanted, (list, tuple)):
      if isinstance(group_column, (list, tuple)) and list(group_column) == list(wanted):
        return dataset
    elif group_column == wanted:
      return dataset
  return None


def _local_density_output_datasets(config, group_name: str) -> dict[str, str]:
  columns = {
    'local_mass_density_300',
    'local_mass_density_1000',
    'local_mass_density_3000',
    'local_number_density_300',
    'local_number_density_1000',
    'local_number_density_3000',
  }
  outputs = {}
  for dataset, column in resolve_dataset_columns(config).items():
    group_column = _column_for_group(column, group_name)
    if isinstance(group_column, str) and group_column in columns:
      outputs[group_column] = dataset
  return outputs


def _read_first_simulation_metadata(files: list[str]) -> dict:
  for file in files:
    with h5py.File(file, 'r') as f:
      validate_complete(f, file)
      simulation = read_simulation_metadata(f)
      if simulation:
        return simulation
  return {}


def _simulation_boxsize(simulation: dict):
  for key in ('boxsize', 'BoxSize', 'header_BoxSize'):
    if key in simulation:
      return np.asarray(simulation[key]).reshape(-1)[0]
  raise ValueError('Cannot calculate global local densities without simulation boxsize metadata')


def _write_empty_local_density_datasets(out_group, outputs: dict[str, str]) -> None:
  for dataset in outputs.values():
    _replace_dataset(out_group, dataset, np.empty(0, dtype=float))


def _write_global_local_densities(out_group, config, group_name: str, simulation: dict) -> None:
  outputs = _local_density_output_datasets(config, group_name)
  if not outputs:
    return

  pos_dataset = _dataset_for_column(config, group_name, ['x_total', 'y_total', 'z_total'])
  mass_dataset = _dataset_for_column(config, group_name, 'mass_total')
  if pos_dataset is None or mass_dataset is None:
    raise ValueError(f'Cannot calculate {group_name} local densities without position and total mass output columns')

  if pos_dataset not in out_group or mass_dataset not in out_group:
    id_dataset = 'GalID' if group_name == 'galaxies' else 'HaloID'
    if id_dataset in out_group and len(out_group[id_dataset]) == 0:
      _write_empty_local_density_datasets(out_group, outputs)
      return
    raise ValueError(f'Cannot calculate {group_name} local densities; merged position or mass datasets are missing')

  densities = calculate_local_density_arrays(
    out_group[pos_dataset][:],
    out_group[mass_dataset][:],
    _simulation_boxsize(simulation),
  )
  for column, dataset in outputs.items():
    _replace_dataset(out_group, dataset, densities[column])


def _columns_for_group(columns, group_name: str):
  if not isinstance(columns, Mapping):
    return set()
  values = columns.get(group_name, [])
  if values is None:
    return set()
  if isinstance(values, str):
    return {values}
  return set(values)


def _halo_index_columns(config):
  metadata = halo_source_metadata_schema(config) or {}
  columns = metadata.get('halo_index_columns')
  if isinstance(columns, Mapping):
    return {
      'halos': _columns_for_group(columns, 'halos'),
      'galaxies': _columns_for_group(columns, 'galaxies'),
    }

  common = {'caesar_parent_halo_index', 'caesar_top_halo_index'}
  resolved = {'halos': set(common), 'galaxies': set(common)}
  host_index_column = metadata.get('host_index_column')
  if host_index_column is not None:
    resolved['galaxies'].add(host_index_column)
  return resolved


def _remap_halo_source_ids(values, sorted_halo_source_ids, halo_source_order, halo_inverse_order):
  values = np.asarray(values)
  remapped = np.full(values.shape, -1, dtype=np.int64)
  valid = values >= 0
  if not np.any(valid):
    return remapped

  positions = np.searchsorted(sorted_halo_source_ids, values[valid])
  in_bounds = positions < len(sorted_halo_source_ids)
  matched = np.zeros(len(positions), dtype=bool)
  matched[in_bounds] = sorted_halo_source_ids[positions[in_bounds]] == values[valid][in_bounds]
  if np.any(matched):
    valid_rows = np.flatnonzero(valid)
    remapped[valid_rows[matched]] = halo_inverse_order[halo_source_order[positions[matched]]]
  return remapped


def _write_merged_csr_dataset(out_group, dataset: str, files: list[str], group_key: str, order, file_lengths):
  """Merge variable-length CSR datasets while applying the final row ordering.

  Each shard stores rows as flat indices plus per-row lengths. The merge reconstructs
  row slices, reorders those slices with the scalar dataset order, then serializes the
  reordered slices back to CSR form."""
  length_key = 'halos' if group_key == 'halo_data' else 'galaxies'
  all_indices = []
  all_lengths = []
  index_dtype = None
  length_dtype = None
  seen = False

  for file in files:
    n_rows = file_lengths[length_key][file]
    with h5py.File(file, 'r') as f_in:
      validate_complete(f_in, file)
      if group_key not in f_in:
        all_indices.append(np.empty(0, dtype=np.int64))
        all_lengths.append(np.zeros(n_rows, dtype=np.int32))
        continue

      group = f_in[group_key]
      indices_name = f'{dataset}_indices'
      lengths_name = f'{dataset}_lengths'
      if indices_name not in group or lengths_name not in group:
        all_indices.append(np.empty(0, dtype=np.int64))
        all_lengths.append(np.zeros(n_rows, dtype=np.int32))
        continue

      indices = group[indices_name][:]
      lengths = group[lengths_name][:]
      if len(lengths) != n_rows:
        raise ValueError(
          f'{file}:{group_key}/{dataset} has {len(lengths)} lengths for {n_rows} rows'
        )

      seen = True
      index_dtype = indices.dtype if index_dtype is None else np.result_type(index_dtype, indices.dtype)
      length_dtype = lengths.dtype if length_dtype is None else np.result_type(length_dtype, lengths.dtype)
      all_indices.append(indices)
      all_lengths.append(lengths)

  if not seen:
    return

  # Build row offsets in pre-merge order, then pull those row slices in final order.
  old_lengths = np.concatenate(all_lengths).astype(np.int64, copy=False)
  old_offsets = np.concatenate([[0], np.cumsum(old_lengths[:-1])]).astype(np.int64)
  merged_lengths = old_lengths[order].astype(length_dtype, copy=False)
  merged_offsets = np.concatenate([[0], np.cumsum(merged_lengths[:-1])]).astype(np.int64)

  all_flat = np.concatenate([indices.astype(index_dtype, copy=False) for indices in all_indices])
  pieces = [
    all_flat[old_offsets[i]:old_offsets[i] + old_lengths[i]]
    for i in order
    if old_lengths[i] > 0
  ]
  if pieces:
    reordered = np.concatenate(pieces).astype(index_dtype, copy=False)
  else:
    reordered = np.empty(0, dtype=index_dtype)

  out_group.create_dataset(f'{dataset}_indices', data=reordered)
  out_group.create_dataset(f'{dataset}_offsets', data=merged_offsets)
  out_group.create_dataset(f'{dataset}_lengths', data=merged_lengths)


def merge_catalogues(files: list[str], outfile: str, configfile: str) -> None:
  with open(configfile, 'r') as f:
    config = safe_load(f)

  simulation = _read_first_simulation_metadata(files)

  galaxy_parent_halo = []
  halo_source_ids = []
  halo_masses = []
  galaxy_masses = []
  file_lengths = {'halos': {}, 'galaxies': {}}

  for file in files:
    with h5py.File(file, 'r') as f:
      validate_complete(f, file)
      file_lengths['halos'][file] = len(f['halo_data']['dicts/masses.total'])
      try:
        file_lengths['galaxies'][file] = len(f['galaxy_data']['dicts/masses.total'])
      except:
        file_lengths['galaxies'][file] = 0

      halo_masses.append(f['halo_data']['dicts/masses.total'][:])
      halo_source_ids.append(f['halo_data']['groupID'][:])
      try:
        file_galaxy_masses = f['galaxy_data']['dicts/masses.stellar'][:]
        if len(file_galaxy_masses) != 0:
          galaxy_parent_halo.append(f['galaxy_data']['parent_halo_index'][:])
          galaxy_masses.append(file_galaxy_masses)
      except: pass
  print(file_lengths)
  galaxy_offsets = {}
  galaxy_offset = 0
  for file in files:
    galaxy_offsets[file] = galaxy_offset
    galaxy_offset += file_lengths['galaxies'][file]

  halo_masses = np.concatenate(halo_masses)
  halo_source_ids = np.concatenate(halo_source_ids)
  if galaxy_masses:
    galaxy_masses = np.concatenate(galaxy_masses)
    galaxy_parent_halo = np.concatenate(galaxy_parent_halo)
  else:
    galaxy_masses = np.empty(0, dtype=float)
    galaxy_parent_halo = np.empty(0, dtype=np.int64)

  halo_order = np.argsort(halo_masses)
  galaxy_order = np.argsort(galaxy_masses)
  halo_inverse_order = np.empty(len(halo_order), dtype=np.int64)
  halo_inverse_order[halo_order] = np.arange(len(halo_order), dtype=np.int64)
  galaxy_inverse_order = np.empty(len(galaxy_order), dtype=np.int64)
  galaxy_inverse_order[galaxy_order] = np.arange(len(galaxy_order), dtype=np.int64)
  halo_source_order = np.argsort(halo_source_ids)
  sorted_halo_source_ids = halo_source_ids[halo_source_order]
  if np.any(sorted_halo_source_ids[1:] == sorted_halo_source_ids[:-1]):
    raise ValueError('Cannot merge catalogues with duplicate halo groupID values')
  parent_positions = np.searchsorted(sorted_halo_source_ids, galaxy_parent_halo)
  in_bounds = parent_positions < len(sorted_halo_source_ids)
  matched = np.zeros(len(galaxy_parent_halo), dtype=bool)
  matched[in_bounds] = sorted_halo_source_ids[parent_positions[in_bounds]] == galaxy_parent_halo[in_bounds]
  if not np.all(matched):
    missing = np.unique(galaxy_parent_halo[~matched])
    raise ValueError(f'Cannot merge catalogues; missing parent halo groupIDs: {missing[:10]}')
  galaxy_parent_halo = halo_inverse_order[halo_source_order[parent_positions]][galaxy_order]

  halo_index_columns = _halo_index_columns(config)

  with h5py.File(outfile, 'w') as f_out:
    mark_incomplete(f_out, 'merged_catalogue')
    if simulation:
      write_simulation_metadata(f_out, simulation)

    halo_group = f_out.create_group('halo_data')
    galaxy_group = f_out.create_group('galaxy_data')

    for dataset, column in resolve_dataset_columns(config).items():
      print(dataset)
      if dataset in ('groupID', 'parent_halo_index'):
        continue
      if dataset in ['glist', 'slist', 'dmlist', 'bhlist']:
        continue # old code guard

      halo_data = []
      galaxy_data = []
      halo_seen = False
      galaxy_seen = False
      include_halos = _column_for_group(column, 'halos') is not None
      include_galaxies = _column_for_group(column, 'galaxies') is not None

      for file in files:
        with h5py.File(file, 'r') as f:
          validate_complete(f, file)
          if include_halos:
            try:
              values = f['halo_data'][dataset][:]
              if dataset == 'central_galaxy':
                valid = values >= 0
                remapped = np.full(values.shape, -1, dtype=np.int64)
                if np.any(valid):
                  remapped[valid] = galaxy_inverse_order[galaxy_offsets[file] + values[valid].astype(np.int64)]
                values = remapped
              elif dataset in halo_index_columns['halos']:
                values = _remap_halo_source_ids(values, sorted_halo_source_ids, halo_source_order, halo_inverse_order)
              halo_data.append(values)
              halo_seen = True
            except KeyError:
              halo_data.append(_empty_values(dataset, file_lengths['halos'][file]))

          if include_galaxies and file_lengths['galaxies'][file] != 0:
            try:
              values = f['galaxy_data'][dataset][:]
              if dataset in halo_index_columns['galaxies']:
                values = _remap_halo_source_ids(values, sorted_halo_source_ids, halo_source_order, halo_inverse_order)
              galaxy_data.append(values)
              galaxy_seen = True
            except KeyError:
              galaxy_data.append(_empty_values(dataset, file_lengths['galaxies'][file]))

      if halo_seen:
        halo_group[dataset] = np.concatenate(halo_data)[halo_order]
      if galaxy_seen:
        galaxy_group[dataset] = np.concatenate(galaxy_data)[galaxy_order]

    halo_ids = np.arange(np.sum(list(file_lengths['halos'].values())), dtype=np.int64)
    halo_group['HaloID'] = halo_ids

    galaxy_ids = np.arange(np.sum(list(file_lengths['galaxies'].values())), dtype=np.int64)
    galaxy_group['GalID'] = galaxy_ids
    galaxy_group['parent_halo_index'] = galaxy_parent_halo

    halo_galaxies = [np.empty(0, dtype=np.int64) for _id in halo_ids]
    for parent_halo_id in np.unique(galaxy_parent_halo):
      halo_galaxies[parent_halo_id] = galaxy_ids[galaxy_parent_halo == parent_halo_id]

    lengths = np.asarray([len(id_list) for id_list in halo_galaxies], dtype=np.int64)
    offsets = np.cumsum(lengths) - lengths
    if lengths.sum() == 0:
      serialised_galaxy_ids = np.empty(0, dtype=np.int64)
    else:
      serialised_galaxy_ids = np.concatenate(halo_galaxies).astype(np.int64, copy=False)

    halo_group['galaxy_index_list'] = serialised_galaxy_ids
    halo_group['galaxy_index_list_offsets'] = offsets
    halo_group['galaxy_index_list_lengths'] = lengths

    ptype_lists = ['glist', 'slist', 'dmlist', 'bhlist']
    for ptype_list in ptype_lists:
      for group_key, out_group, order in [
        ('halo_data', halo_group, halo_order),
        ('galaxy_data', galaxy_group, galaxy_order),
      ]:
        _write_merged_csr_dataset(out_group, ptype_list, files, group_key, order, file_lengths)

    for dataset, column in resolve_list_dataset_columns(config).items():
      for group_name, group_key, out_group, order in [
        ('halos', 'halo_data', halo_group, halo_order),
        ('galaxies', 'galaxy_data', galaxy_group, galaxy_order),
      ]:
        if _column_for_group(column, group_name) is None:
          continue
        _write_merged_csr_dataset(out_group, dataset, files, group_key, order, file_lengths)

    _write_global_local_densities(halo_group, config, 'halos', simulation)
    _write_global_local_densities(galaxy_group, config, 'galaxies', simulation)
    mark_complete(f_out)
