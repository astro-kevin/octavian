from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
from functools import partial
from astropy import constants as const
from time import perf_counter

from scipy.spatial import KDTree

from octavian.group_properties_calc.group_computations import (
    accumulate_membership_array_common_first,
    accumulate_membership_array_common_second,
    accumulate_membership_array_rotation,
    compute_angular_momentum,
    compute_aperture_component_properties,
    compute_central_galaxy_flags,
    compute_galaxy_hydrogen_assignment,
    compute_gas_scalar_sums,
    compute_membership_array_bh_max,
    compute_membership_array_gas_scalar_sums,
    compute_membership_array_star_scalar_sums,
    compute_rotation_quantities,
    compute_radial_quantiles_and_rmax,
    compute_star_scalar_sums,
    compute_virial_quantities,
    flatten_membership_array_radius_mass,
)

from octavian.halo_reader import halo_source_metadata_schema
from octavian.group_properties_calc.group_helpers import (
    sum_per_group,
    count_per_group,
    max_value_per_group,
    max_idx_per_group,
    min_idx_per_group,
    broadcast_to_particles,
    sort_by_group,
    weighted_mean_per_group,
    extract_particle_arrays,
)

# Suppress pandas fragmented frame performance warnings (superfluous)  https://stackoverflow.com/a/76306267
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def _safe_divide(numerator, denominator, fill=np.nan):
  numerator = np.asarray(numerator, dtype=float)
  denominator = np.asarray(denominator, dtype=float)
  result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), fill, dtype=float)
  return np.divide(numerator, denominator, out=result, where=denominator != 0)

def _group_ids_and_rows(data_manager: DataManager, group_name: str, ptype: str, df, groupID_key: str):
  if group_name == 'halos' and ptype in data_manager.halo_id_arrays:
    return data_manager.get_halo_membership_rows(ptype)
  return df[groupID_key].to_numpy(), slice(None)


def _group_index_map(group_data) -> np.ndarray:
  group_ids = group_data.index.to_numpy(dtype=np.int64)
  valid = group_ids >= 0
  if not np.any(valid):
    return np.empty(0, dtype=np.int64)
  group_map = np.full(int(group_ids[valid].max()) + 1, -1, dtype=np.int64)
  group_map[group_ids[valid]] = np.flatnonzero(valid).astype(np.int64)
  return group_map


def _assign_empty_common_particle_properties(group_data, particle_type: str, n_groups: int) -> None:
  group_data[f'n{particle_type}'] = np.zeros(n_groups, dtype=np.int64)
  group_data[f'mass_{particle_type}'] = np.zeros(n_groups, dtype=float)


def _uses_halo_id_arrays(data_manager: DataManager, group_name: str, ptypes: list[str]) -> bool:
  if group_name != 'halos' or not data_manager.halo_id_arrays:
    return False
  missing = [ptype for ptype in ptypes if ptype not in data_manager.halo_id_arrays]
  if missing:
    raise KeyError(f'HaloID_array missing for halo ptypes: {missing}')
  return True


def _load_property_if_missing(data_manager: DataManager, prop: str, ptype: str) -> None:
  columns = data_manager.get_column_name(prop)
  if isinstance(columns, str):
    columns = [columns]
  if all(column in data_manager.data[ptype] for column in columns):
    return
  data_manager.load_property(prop, ptype)


def _load_halo_common_particle_inputs(data_manager: DataManager) -> None:
  for ptype in data_manager.config['ptypes']:
    _load_property_if_missing(data_manager, 'pos', ptype)
    _load_property_if_missing(data_manager, 'vel', ptype)
    _load_property_if_missing(data_manager, 'bhmass' if ptype == 'bh' else 'mass', ptype)
    _load_property_if_missing(data_manager, 'potential', ptype)


def _source_lineage_for_ids(tree, source_halo_ids, output_halo_ids, original_id_column):
  source_halo_ids = np.asarray(source_halo_ids, dtype=np.int64)
  output_halo_ids = np.asarray(output_halo_ids, dtype=np.int64)
  n = len(source_halo_ids)

  source_id = np.full(n, -1, dtype=np.int64)
  source_parent_id = np.full(n, -1, dtype=np.int64)
  source_top_id = np.full(n, -1, dtype=np.int64)
  source_depth = np.full(n, -1, dtype=np.int64)
  caesar_parent_halo_index = np.full(n, -1, dtype=np.int64)
  caesar_top_halo_index = np.full(n, -1, dtype=np.int64)

  if len(tree.halo_ids) == 0:
    return (
        source_id,
        source_parent_id,
        source_top_id,
        source_depth,
        caesar_parent_halo_index,
        caesar_top_halo_index,
    )

  max_halo_id = len(tree._id_to_idx)
  valid = (source_halo_ids >= 0) & (source_halo_ids < max_halo_id)
  valid[valid] = tree._id_to_idx[source_halo_ids[valid]] != -1
  valid_ids = source_halo_ids[valid]
  tree_rows = tree._id_to_idx[valid_ids]

  original_ids = tree.halo_ids.copy()
  if tree.properties is not None and original_id_column in tree.properties:
    original_ids = tree.properties[original_id_column].to_numpy(dtype=np.int64)

  original_by_halo_id = np.full(max_halo_id, -1, dtype=np.int64)
  original_by_halo_id[tree.halo_ids] = original_ids

  output_by_halo_id = np.full(max_halo_id, -1, dtype=np.int64)
  present_output = (output_halo_ids >= 0) & (output_halo_ids < max_halo_id)
  output_by_halo_id[output_halo_ids[present_output]] = output_halo_ids[present_output]

  parent_ids = np.full(n, -1, dtype=np.int64)
  parent_ids[valid] = tree.parent_ids[tree_rows]
  top_ids = np.full(n, -1, dtype=np.int64)
  top_ids[valid] = tree.field_map[valid_ids]

  source_id[valid] = original_by_halo_id[valid_ids]
  parent_valid = valid & (parent_ids >= 0) & (parent_ids < max_halo_id)
  source_parent_id[parent_valid] = original_by_halo_id[parent_ids[parent_valid]]
  top_valid = valid & (top_ids >= 0) & (top_ids < max_halo_id)
  source_top_id[top_valid] = original_by_halo_id[top_ids[top_valid]]
  source_depth[valid] = tree.depths[tree_rows]

  caesar_parent_halo_index[parent_valid] = output_by_halo_id[parent_ids[parent_valid]]
  caesar_top_halo_index[top_valid] = output_by_halo_id[top_ids[top_valid]]

  return (
      source_id,
      source_parent_id,
      source_top_id,
      source_depth,
      caesar_parent_halo_index,
      caesar_top_halo_index,
  )


def _source_ancestor_halo_ids(tree, source_halo_ids, original_id_column):
  source_halo_ids = np.asarray(source_halo_ids, dtype=np.int64)
  if len(tree.halo_ids) == 0:
    return [np.empty(0, dtype=np.int64) for _ in source_halo_ids]

  max_halo_id = len(tree._id_to_idx)
  original_ids = tree.halo_ids.copy()
  if tree.properties is not None and original_id_column in tree.properties:
    original_ids = tree.properties[original_id_column].to_numpy(dtype=np.int64)

  original_by_halo_id = np.full(max_halo_id, -1, dtype=np.int64)
  original_by_halo_id[tree.halo_ids] = original_ids

  ancestor_halo_ids = []
  for source_halo_id in source_halo_ids:
    if source_halo_id < 0 or source_halo_id >= max_halo_id:
      ancestor_halo_ids.append(np.empty(0, dtype=np.int64))
      continue

    row = tree._id_to_idx[source_halo_id]
    if row == -1:
      ancestor_halo_ids.append(np.empty(0, dtype=np.int64))
      continue

    ancestors = []
    seen = set()
    parent = tree.parent_ids[row]
    while parent != -1 and parent not in seen and 0 <= parent < max_halo_id:
      parent_row = tree._id_to_idx[parent]
      if parent_row == -1:
        break
      ancestors.append(original_by_halo_id[parent])
      seen.add(parent)
      parent = tree.parent_ids[parent_row]

    ancestor_halo_ids.append(np.asarray(ancestors, dtype=np.int64))

  return ancestor_halo_ids


def _assign_halo_source_properties(data_manager: DataManager) -> None:
  schema = halo_source_metadata_schema(data_manager.config)
  if schema is None:
    return
  if 'halos' not in data_manager.group_data or not hasattr(data_manager, 'halo_tree'):
    return

  original_id_column = schema['original_id_column']
  halo_id_column = schema['halo_id_column']
  parent_id_column = schema['parent_id_column']
  top_id_column = schema['top_id_column']
  depth_column = schema['depth_column']
  host_index_column = schema['host_index_column']
  ancestor_column = schema['ancestor_column']

  tree = data_manager.halo_tree
  halo_data = data_manager.group_data['halos']
  halo_ids = halo_data.index.to_numpy(dtype=np.int64)
  (
    source_halo_id,
    source_parent_halo_id,
    source_top_halo_id,
    source_depth,
    caesar_parent_halo_index,
    caesar_top_halo_index,
  ) = _source_lineage_for_ids(tree, halo_ids, halo_ids, original_id_column)

  halo_data[halo_id_column] = source_halo_id
  halo_data[parent_id_column] = source_parent_halo_id
  halo_data[top_id_column] = source_top_halo_id
  halo_data[depth_column] = source_depth
  halo_data['caesar_parent_halo_index'] = caesar_parent_halo_index
  halo_data['caesar_top_halo_index'] = caesar_top_halo_index
  halo_data[ancestor_column] = pd.Series(
      _source_ancestor_halo_ids(tree, halo_ids, original_id_column), index=halo_data.index, dtype=object
  )
  halo_data['child'] = caesar_parent_halo_index != -1

  galaxy_data = data_manager.group_data.get('galaxies')
  if galaxy_data is None or 'parent_halo_index' not in galaxy_data:
    return

  galaxy_source_halo_ids = galaxy_data['parent_halo_index'].to_numpy(dtype=np.int64)
  (
    source_halo_id,
    source_parent_halo_id,
    source_top_halo_id,
    source_depth,
    caesar_parent_halo_index,
    caesar_top_halo_index,
  ) = _source_lineage_for_ids(tree, galaxy_source_halo_ids, halo_ids, original_id_column)

  galaxy_data[halo_id_column] = source_halo_id
  galaxy_data[parent_id_column] = source_parent_halo_id
  galaxy_data[top_id_column] = source_top_halo_id
  galaxy_data[depth_column] = source_depth
  galaxy_data[host_index_column] = caesar_top_halo_index
  galaxy_data['caesar_parent_halo_index'] = caesar_parent_halo_index
  galaxy_data['caesar_top_halo_index'] = caesar_top_halo_index
  galaxy_data[ancestor_column] = pd.Series(
      _source_ancestor_halo_ids(tree, galaxy_source_halo_ids, original_id_column), index=galaxy_data.index, dtype=object
  )


def _assign_central_galaxies(data_manager: DataManager) -> None:
  halo_data = data_manager.group_data.get('halos')
  if halo_data is None:
    return
  if 'galaxies' not in data_manager.group_data:
    halo_data['central_galaxy'] = -1
    return

  galaxy_data = data_manager.group_data['galaxies']
  if len(galaxy_data) == 0 or 'parent_halo_index' not in galaxy_data or 'mass_star' not in galaxy_data:
    halo_data['central_galaxy'] = -1
    galaxy_data['central'] = False
    return

  parent_key = '_central_host_halo_index' if '_central_host_halo_index' in galaxy_data else 'parent_halo_index'
  parent_halo = galaxy_data[parent_key].to_numpy(dtype=np.int64)
  stellar_mass = galaxy_data['mass_star'].to_numpy(dtype=float)
  galaxy_ids = galaxy_data.index.to_numpy(dtype=np.int64)
  n_halos = len(halo_data)

  halo_positions = halo_data.index.get_indexer(parent_halo)
  central, central_by_halo = compute_central_galaxy_flags(
      halo_positions.astype(np.int64), stellar_mass, galaxy_ids, n_halos
  )

  halo_data['central_galaxy'] = central_by_halo
  galaxy_data['central'] = central


def _common_halo_array_properties(data_manager: DataManager, particle_type: str, ptypes: list[str]) -> None:
  """Compute common halo properties when particles have top-to-deepest ancestry rows.

  The scalar groupby path cannot be used here because one particle can contribute to
  several halo rows along its ancestry chain. This path walks every valid ancestry
  column for every particle and accumulates each output halo independently."""
  group_data = data_manager.group_data['halos']
  n_groups = len(group_data)
  group_map = _group_index_map(group_data)
  mass_mode = 1 if particle_type == 'bh' else 0
  do_minpot = particle_type == 'total'

  counts = np.zeros(n_groups, dtype=np.int64)
  group_mass = np.full(n_groups, -np.inf if mass_mode else 0., dtype=float)
  pos_mass_sum = np.zeros((n_groups, 3))
  vel_mass_sum = np.zeros((n_groups, 3))
  min_potential = np.full(n_groups, np.inf)
  minpot_position = np.full((n_groups, 3), np.nan)
  minpot_velocity = np.full((n_groups, 3), np.nan)

  # First pass: counts, masses, centers of mass, and minimum-potential anchors.
  for ptype in ptypes:
    df = data_manager.data[ptype]
    accumulate_membership_array_common_first(
        data_manager.halo_id_arrays[ptype],
        group_map,
        df[['x', 'y', 'z']].to_numpy(),
        df[['vx', 'vy', 'vz']].to_numpy(),
        df['mass'].to_numpy(),
        df['potential'].to_numpy(),
        counts,
        group_mass,
        pos_mass_sum,
        vel_mass_sum,
        min_potential,
        minpot_position,
        minpot_velocity,
        mass_mode,
        do_minpot,
    )

  if counts.sum() == 0:
    return
  if mass_mode:
    group_mass[~np.isfinite(group_mass)] = 0.

  group_data[f'n{particle_type}'] = counts
  group_data[f'mass_{particle_type}'] = group_mass

  if do_minpot:
    valid = np.isfinite(min_potential)
    minpot_position[~valid] = np.nan
    minpot_velocity[~valid] = np.nan
    for i, d in enumerate(['x', 'y', 'z']):
      group_data[f'minpot_{d}'] = minpot_position[:, i]
      group_data[f'minpot_v{d}'] = minpot_velocity[:, i]

  com_positions = _safe_divide(pos_mass_sum, group_mass[:, None])
  com_velocities = _safe_divide(vel_mass_sum, group_mass[:, None])
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'{d}_{particle_type}'] = com_positions[:, i]
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'v{d}_{particle_type}'] = com_velocities[:, i]

  if particle_type == 'total':
    ref_positions = minpot_position
    ref_velocities = minpot_velocity
  else:
    ref_positions = group_data[['minpot_x', 'minpot_y', 'minpot_z']].to_numpy()
    ref_velocities = group_data[['minpot_vx', 'minpot_vy', 'minpot_vz']].to_numpy()

  # Second pass: quantities measured relative to the selected reference position.
  disp_sums = np.zeros(n_groups)
  L = np.zeros((n_groups, 3))
  for ptype in ptypes:
    df = data_manager.data[ptype]
    accumulate_membership_array_common_second(
        data_manager.halo_id_arrays[ptype],
        group_map,
        df[['x', 'y', 'z']].to_numpy(),
        df[['vx', 'vy', 'vz']].to_numpy(),
        df['mass'].to_numpy(),
        ref_positions,
        ref_velocities,
        com_velocities,
        disp_sums,
        L,
    )

  group_data[f'velocity_dispersion_{particle_type}'] = np.sqrt(_safe_divide(disp_sums, counts))
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'L{d}_{particle_type}'] = L[:, i]

  L_mag = np.linalg.norm(L, axis=1)
  group_data[f'L_{particle_type}'] = L_mag
  group_data[f'ALPHA_{particle_type}'] = np.arctan2(L[:, 1], L[:, 2])
  group_data[f'BETA_{particle_type}'] = np.arcsin(_safe_divide(L[:, 0], L_mag))

  counter_mass = np.zeros(n_groups)
  krot = np.zeros(n_groups)
  ktot = np.zeros(n_groups)
  for ptype in ptypes:
    df = data_manager.data[ptype]
    accumulate_membership_array_rotation(
        data_manager.halo_id_arrays[ptype],
        group_map,
        df[['x', 'y', 'z']].to_numpy(),
        df[['vx', 'vy', 'vz']].to_numpy(),
        df['mass'].to_numpy(),
        ref_positions,
        ref_velocities,
        L,
        counter_mass,
        krot,
        ktot,
    )

  group_data[f'BoverT_{particle_type}'] = _safe_divide(2 * counter_mass, group_mass)
  group_data[f'kappa_rot_{particle_type}'] = _safe_divide(krot, ktot)

  angular_cols = [
    f'velocity_dispersion_{particle_type}',
    f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}',
    f'BoverT_{particle_type}', f'kappa_rot_{particle_type}',
    ]
  small = counts < 3
  for col in angular_cols:
    vals = group_data[col].to_numpy().copy()
    vals[small] = 0.
    group_data[col] = vals

  # Radial quantiles need a flattened membership list because particles can appear
  # once per ancestor halo.
  radial_group_idx, radial_radii, radial_masses = [], [], []
  for ptype in ptypes:
    df = data_manager.data[ptype]
    group_idx, radii, masses = flatten_membership_array_radius_mass(
        data_manager.halo_id_arrays[ptype],
        group_map,
        df[['x', 'y', 'z']].to_numpy(),
        df['mass'].to_numpy(),
        ref_positions,
    )
    radial_group_idx.append(group_idx)
    radial_radii.append(radii)
    radial_masses.append(masses)

  group_idx = np.concatenate(radial_group_idx)
  radii = np.concatenate(radial_radii)
  masses = np.concatenate(radial_masses)
  quantiles = np.array([0.2, 0.5, 0.8])
  radial_results, radius_rmax = compute_radial_quantiles_and_rmax(
      radii, masses, group_idx, n_groups, quantiles
  )

  for q, col_name in enumerate(['r20', 'half_mass', 'r80']):
    group_data[f'radius_{particle_type}_{col_name}'] = radial_results[:, q]
  group_data[f'radius_{particle_type}_rmax'] = radius_rmax

  if particle_type == 'total':
    group_data['r200'] = data_manager.simulation['r200_factor'] * group_mass**(1/3)
    G_factor = unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d
    group_data['circular_velocity'] = np.sqrt(G_factor * group_mass / group_data['r200'])
    group_data['temperature'] = 3.6e5 * (group_data['circular_velocity'] / 100.0)**2
    group_data['spin_param'] = _safe_divide(
        L_mag,
        np.sqrt(2) * group_mass * group_data['circular_velocity'].to_numpy() * group_data['r200'].to_numpy()
    )

    rhocrit = (
        data_manager.simulation['rhocrit'] *
        data_manager.create_unit_quantity('rhocrit')
    ).to('Msun / (kpc*a)**3').d
    factors = np.array([200., 500., 2500.])
    virial_r, virial_m = compute_virial_quantities(radii, masses, group_idx, n_groups, rhocrit, factors)
    for f, factor in enumerate([200, 500, 2500]):
      group_data[f'radius_{factor}_c'] = virial_r[:, f]
      group_data[f'mass_{factor}_c'] = virial_m[:, f]


def common_group_properties(data_manager: DataManager, group_name: str, particle_type: str) -> None:
  """
  Computes properties common to all particle types.
  """
  config = data_manager.config
  groupID_key = config['groupIDs'][group_name]
  group_data = data_manager.group_data[group_name]
  n_groups = len(group_data)
  _assign_empty_common_particle_properties(group_data, particle_type, n_groups)

  # -
  # step 1: extract arrays from datamanager so the entire operation can be vectorised
  # -

  if particle_type == 'total':
      ptypes = config['ptypes']
  elif particle_type == 'baryon':
      ptypes = config['ptypes_baryon']
  else:
      ptypes = [particle_type]

  if _uses_halo_id_arrays(data_manager, group_name, ptypes):
    _common_halo_array_properties(data_manager, particle_type, ptypes)
    return

  use_central_host_ids = group_name == 'galaxies' and particle_type == 'total' and bool(data_manager.halo_id_arrays)
  if use_central_host_ids:
    missing = [ptype for ptype in ptypes if ptype not in data_manager.halo_id_arrays]
    if missing:
      raise KeyError(f'HaloID_array missing for galaxy central-host ptypes: {missing}')

  # ids: groups, halos, galaxies
  group_ids_list, halo_ids_list, central_host_ids_list, gal_ids_list = [], [], [], []
  # physical quantities: mass, potential
  masses_list, potentials_list = [], []
  # kinematics: positions, velocities
  positions_list, velocities_list = [], []

  # cast the required data to numpy.
  for ptype in ptypes:
    df = data_manager.data[ptype]
    group_ids, rows = _group_ids_and_rows(data_manager, group_name, ptype, df, groupID_key)
    group_ids_list.append(group_ids)
    halo_ids_list.append(df['HaloID'].to_numpy()[rows])
    if use_central_host_ids:
      central_host_ids_list.append(data_manager.get_halo_ids(ptype, mode='top')[rows])
    gal_ids_list.append(df['GalID'].to_numpy()[rows])
    masses_list.append(df['mass'].to_numpy()[rows])
    potentials_list.append(df['potential'].to_numpy()[rows])
    positions_list.append(df[['x', 'y', 'z']].to_numpy()[rows])
    velocities_list.append(df[['vx', 'vy', 'vz']].to_numpy()[rows])

  # flat arrays
  # group_ids = np.concatenate([data_manager.data[ptype][groupID_key] for ptype in ptypes])
  # halo_ids = np.concatenate([data_manager.data[ptype]['HaloID'] for ptype in ptypes])
  # gal_ids = np.concatenate([data_manager.data[ptype]['GalID'] for ptype in ptypes])
  # masses = np.concatenate([data_manager.data[ptype]['mass'] for ptype in ptypes])
  # potentials = np.concatenate([data_manager.data[ptype]['potential'] for ptype in ptypes])
  # potentials = np.vstack([data_manager.data[ptype][['x', 'y', 'z']] for ptype in ptypes])
  # velocities = np.vstack([data_manager.data[ptype][['vx', 'vy', 'vz']] for ptype in ptypes])

  group_ids = np.concatenate(group_ids_list)
  halo_ids = np.concatenate(halo_ids_list)
  central_host_ids = np.concatenate(central_host_ids_list) if central_host_ids_list else None
  gal_ids = np.concatenate(gal_ids_list)
  masses = np.concatenate(masses_list)
  potentials = np.concatenate(potentials_list)
  # vstack is equivalent to concatenate here: but these are 3D so this makes the format more explicit
  positions = np.vstack(positions_list)
  velocities = np.vstack(velocities_list)

  # filter particles not assigned to a specific galaxy
  if group_name == 'galaxies':
    keep = gal_ids != -1
    group_ids = group_ids[keep]
    halo_ids = halo_ids[keep]
    if central_host_ids is not None:
      central_host_ids = central_host_ids[keep]
    masses = masses[keep]
    potentials = potentials[keep]
    positions = positions[keep]
    velocities = velocities[keep]

  # if a group is empty
  if len(masses) == 0:
    return

  # -
  # step 2: group indexing
  # -

  group_idx = group_data.index.get_indexer(group_ids) # build a list of particles where the value of each particle is its group ID

  # parent halo assignment
  if group_name == 'galaxies' and particle_type == 'total':
      parent = np.full(n_groups, -1, dtype=np.int64)
      central_host = np.full(n_groups, -1, dtype=np.int64) if central_host_ids is not None else None
      for i in range(len(group_ids)):
          g = group_idx[i]
          if parent[g] == -1:
              parent[g] = halo_ids[i]
          if central_host is not None and central_host[g] == -1:
              central_host[g] = central_host_ids[i]
      group_data['parent_halo_index'] = parent
      if central_host is not None:
          group_data['_central_host_halo_index'] = central_host

  # -
  # step 3: nparticles
  # -

  counts = count_per_group(group_idx, n_groups) # number of particles per group
  group_data[f'n{particle_type}'] = counts

  # -
  # step 4: masses
  # -

  if particle_type == 'bh':
    group_mass = max_value_per_group(masses, group_idx, n_groups) # REVIEW: why?
    group_mass[~np.isfinite(group_mass)] = 0.
  else:
    group_mass = sum_per_group(masses, group_idx, n_groups)

  group_data[f'mass_{particle_type}'] = group_mass

  # -
  # step 5: minimum potential for halos
  # -

  if group_name == 'halos' and particle_type == 'total':

    minimum_potential_idx = min_idx_per_group(potentials, group_idx, n_groups)
    valid = minimum_potential_idx >= 0 # mask all else away

    minimum_potential_position = np.full((n_groups, 3), np.nan)
    minimum_potential_velocity = np.full((n_groups, 3), np.nan)
    minimum_potential_position[valid] = positions[minimum_potential_idx[valid]]
    minimum_potential_velocity[valid] = velocities[minimum_potential_idx[valid]]

    for i, d in enumerate(['x', 'y', 'z']):
      group_data[f'minpot_{d}'] = minimum_potential_position[:, i]
      group_data[f'minpot_v{d}'] = minimum_potential_velocity[:, i]

  # -
  # stage 6: centre-of-mass
  # -

  com_positions = np.zeros((n_groups, 3))
  com_velocities = np.zeros((n_groups, 3))

  for d in range(3):
    com_positions[:, d] = _safe_divide(sum_per_group(positions[:, d] * masses, group_idx, n_groups), group_mass)
    com_velocities[:, d] = _safe_divide(sum_per_group(velocities[:, d] * masses, group_idx, n_groups), group_mass)

  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'{d}_{particle_type}'] = com_positions[:, i]
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'v{d}_{particle_type}'] = com_velocities[:, i]

  # -
  # stage 7: relative quantities
  # -

  # halos: potential well, galaxies: centre-of-mass
  if group_name == 'halos':
    if particle_type == 'total':
        ref_positions = minimum_potential_position
        ref_velocities = minimum_potential_velocity
    else:
        ref_positions = group_data[['minpot_x', 'minpot_y', 'minpot_z']].to_numpy()
        ref_velocities = group_data[['minpot_vx', 'minpot_vy', 'minpot_vz']].to_numpy()
  else:
      ref_positions = com_positions
      ref_velocities = com_velocities

  positions_rel = positions - broadcast_to_particles(ref_positions, group_idx)
  velocities_rel_com = velocities - broadcast_to_particles(com_velocities, group_idx)
  velocities_rel_ref = velocities - broadcast_to_particles(ref_velocities, group_idx)

  radii = np.linalg.norm(positions_rel, axis=1)

  # -
  # step 8: velocity dispersion
  # -

  disp_sums = sum_per_group(np.sum(velocities_rel_com**2, axis=1), group_idx, n_groups)
  vel_disps = np.sqrt(_safe_divide(disp_sums, counts))
  group_data[f'velocity_dispersion_{particle_type}'] = vel_disps

  # -
  # step 9: angular momentum and rotation
  # -

  L, ktot = compute_angular_momentum(positions_rel, velocities_rel_ref, masses, group_idx, n_groups)

  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'L{d}_{particle_type}'] = L[:, i]

  L_mag = np.linalg.norm(L, axis=1)
  group_data[f'L_{particle_type}'] = L_mag
  group_data[f'ALPHA_{particle_type}'] = np.arctan2(L[:, 1], L[:, 2])
  group_data[f'BETA_{particle_type}'] = np.arcsin(_safe_divide(L[:, 0], L_mag))

  counter_mass, krot, ktot = compute_rotation_quantities(
      positions_rel, velocities_rel_ref, masses, group_idx, L, n_groups
  )

  group_data[f'BoverT_{particle_type}'] = _safe_divide(2 * counter_mass, group_mass)
  group_data[f'kappa_rot_{particle_type}'] = _safe_divide(krot, ktot)

  angular_cols = [
    f'velocity_dispersion_{particle_type}',
    f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}',
    f'BoverT_{particle_type}', f'kappa_rot_{particle_type}',
    ]

  # for small groups: set quantites = 0 as they are not meaningful (as done in original code)
  small = counts < 3
  for col in angular_cols:
      vals = group_data[col].to_numpy().copy()
      vals[small] = 0.
      group_data[col] = vals

  # -
  # step 10: radial quantities
  # -

  # from previous code
  quantiles = np.array([0.2, 0.5, 0.8])
  quantile_names = ['r20', 'half_mass', 'r80']

  radial_results, radius_rmax = compute_radial_quantiles_and_rmax(
      radii, masses, group_idx, n_groups, quantiles
  )

  for q, col_name in enumerate(quantile_names):
      group_data[f'radius_{particle_type}_{col_name}'] = radial_results[:, q]
  group_data[f'radius_{particle_type}_rmax'] = radius_rmax

  # -
  # step 11: virial quantities for halos
  # -

  if group_name == 'halos' and particle_type == 'total':
    # from previous code
    group_data['r200'] = data_manager.simulation['r200_factor'] * group_mass**(1/3)
    G_factor = unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d
    group_data['circular_velocity'] = np.sqrt(G_factor * group_mass / group_data['r200'])
    group_data['temperature'] = 3.6e5 * (group_data['circular_velocity'] / 100.0)**2
    group_data['spin_param'] = _safe_divide(
        L_mag,
        np.sqrt(2) * group_mass * group_data['circular_velocity'].to_numpy() * group_data['r200'].to_numpy()
    )

    rhocrit = (
        data_manager.simulation['rhocrit'] *
        data_manager.create_unit_quantity('rhocrit')
    ).to('Msun / (kpc*a)**3').d

    factors = np.array([200., 500., 2500.])

    virial_r, virial_m = compute_virial_quantities(radii, masses, group_idx, n_groups, rhocrit, factors)

    for f, factor in enumerate([200, 500, 2500]):
        group_data[f'radius_{factor}_c'] = virial_r[:, f]
        group_data[f'mass_{factor}_c'] = virial_m[:, f]

def gas_group_properties(data_manager: DataManager, group_name: str) -> None:

  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  # load relevant quantities
  df = data_manager.data['gas']

  if _uses_halo_id_arrays(data_manager, group_name, ['gas']):
    n_groups = len(group_data)
    group_map = _group_index_map(group_data)
    dust_masses = df['dustmass'].to_numpy() if 'dustmass' in df else np.zeros(len(df))
    (
      _group_mass_sum,
      gas_HI,
      gas_H2,
      gas_HI_ism,
      gas_H2_ism,
      dust_mass,
      ndust,
      sfr_sum,
      metal_mass,
      metal_sfr,
      temp_mass,
      cgm_mass_sum,
      cgm_temp_mass,
      cgm_temp_metal,
      cgm_metal_mass,
    ) = compute_membership_array_gas_scalar_sums(
        data_manager.halo_id_arrays['gas'],
        group_map,
        df['mass'].to_numpy(),
        df['metallicity'].to_numpy(),
        df['sfr'].to_numpy(),
        df['temperature'].to_numpy(),
        df['rho'].to_numpy(),
        df['mass_HI'].to_numpy(),
        df['mass_H2'].to_numpy(),
        df['mass_HI_ism'].to_numpy(),
        df['mass_H2_ism'].to_numpy(),
        dust_masses,
        n_groups,
        config['nHlim'],
    )

    group_data['mass_HI'] = gas_HI
    group_data['mass_H2'] = gas_H2
    group_data['mass_dust'] = dust_mass
    group_data['ndust'] = ndust
    group_data['mass_HI_ism'] = gas_HI_ism
    group_data['mass_H2_ism'] = gas_H2_ism
    group_data['sfr'] = sfr_sum

    group_mass = group_data[f'mass_gas'].to_numpy()
    group_sfr = group_data['sfr'].to_numpy()
    group_data['metallicity_mass_weighted'] = _safe_divide(metal_mass, group_mass, fill=0.0)
    group_data['metallicity_sfr_weighted'] = _safe_divide(metal_sfr, group_sfr, fill=0.0)
    group_data['mass_cgm'] = cgm_mass_sum
    cgm_mass = group_data['mass_cgm'].to_numpy()
    group_data['temp_mass_weighted'] = _safe_divide(temp_mass, group_mass, fill=0.0)
    group_data['temp_mass_weighted_cgm'] = _safe_divide(cgm_temp_mass, cgm_mass, fill=0.0)
    group_data['temp_metal_weighted_cgm'] = _safe_divide(cgm_temp_metal, cgm_metal_mass, fill=0.0)
    group_data['metallicity_mass_weighted_cgm'] = _safe_divide(cgm_metal_mass, cgm_mass, fill=0.0)
    group_data['metallicity_temp_weighted_cgm'] = _safe_divide(cgm_temp_metal, cgm_temp_mass, fill=0.0)
    return

  group_ids, rows = _group_ids_and_rows(data_manager, group_name, 'gas', df, groupID_key)
  masses = df['mass'].to_numpy()[rows]
  metallicities = df['metallicity'].to_numpy()[rows]
  sfrs = df['sfr'].to_numpy()[rows]
  temperatures = df['temperature'].to_numpy()[rows]
  rhos = df['rho'].to_numpy()[rows]
  mass_HI = df['mass_HI'].to_numpy()[rows]
  mass_H2 = df['mass_H2'].to_numpy()[rows]
  mass_HI_ism = df['mass_HI_ism'].to_numpy()[rows]
  mass_H2_ism = df['mass_H2_ism'].to_numpy()[rows]
  dust_masses = df['dustmass'].to_numpy()[rows] if 'dustmass' in df else np.zeros(len(group_ids))

  if group_name == 'galaxies':
    keep = df['GalID'].to_numpy() != -1
    group_ids = group_ids[keep]
    masses = masses[keep]
    metallicities = metallicities[keep]
    sfrs = sfrs[keep]
    temperatures = temperatures[keep]
    rhos = rhos[keep]
    mass_HI = mass_HI[keep]
    mass_H2 = mass_H2[keep]
    mass_HI_ism = mass_HI_ism[keep]
    mass_H2_ism = mass_H2_ism[keep]
    dust_masses = dust_masses[keep]

  # guard against empty group
  if len(masses) == 0:
    return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)
  order, unique_groups, start, end = sort_by_group(group_idx)
  (
    _group_mass_sum,
    gas_HI,
    gas_H2,
    gas_HI_ism,
    gas_H2_ism,
    dust_mass,
    ndust,
    sfr_sum,
    metal_mass,
    metal_sfr,
    temp_mass,
    cgm_mass_sum,
    cgm_temp_mass,
    cgm_temp_metal,
    cgm_metal_mass,
  ) = compute_gas_scalar_sums(
      unique_groups.astype(np.int64),
      start.astype(np.int64),
      end.astype(np.int64),
      masses[order],
      metallicities[order],
      sfrs[order],
      temperatures[order],
      rhos[order],
      mass_HI[order],
      mass_H2[order],
      mass_HI_ism[order],
      mass_H2_ism[order],
      dust_masses[order],
      n_groups,
      config['nHlim'],
  )

  # HI, H2 masses
  group_data['mass_HI'] = gas_HI
  group_data['mass_H2'] = gas_H2
  group_data['mass_dust'] = dust_mass
  group_data['ndust'] = ndust

  group_data['mass_HI_ism'] = gas_HI_ism
  group_data['mass_H2_ism'] = gas_H2_ism

  # SFR
  group_data['sfr'] = sfr_sum

  # metallicity
  group_mass = group_data[f'mass_gas'].to_numpy()
  group_sfr = group_data['sfr'].to_numpy()

  group_data['metallicity_mass_weighted'] = _safe_divide(metal_mass, group_mass, fill=0.0)
  group_data['metallicity_sfr_weighted'] = _safe_divide(metal_sfr, group_sfr, fill=0.0)

  # CGM quantities
  group_data['mass_cgm'] = cgm_mass_sum
  cgm_mass = group_data['mass_cgm'].to_numpy()

  # temperatures
  group_data['temp_mass_weighted'] = _safe_divide(temp_mass, group_mass, fill=0.0)

  group_data['temp_mass_weighted_cgm'] = _safe_divide(cgm_temp_mass, cgm_mass, fill=0.0)
  group_data['temp_metal_weighted_cgm'] = _safe_divide(cgm_temp_metal, cgm_metal_mass, fill=0.0)

  # CGM metallicity
  group_data['metallicity_mass_weighted_cgm'] = _safe_divide(cgm_metal_mass, cgm_mass, fill=0.0)
  group_data['metallicity_temp_weighted_cgm'] = _safe_divide(cgm_temp_metal, cgm_temp_mass, fill=0.0)

def star_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['star']

  if _uses_halo_id_arrays(data_manager, group_name, ['star']):
    n_groups = len(group_data)
    total_mass, metal_mass, age_mass, age_metal, young_mass = compute_membership_array_star_scalar_sums(
        data_manager.halo_id_arrays['star'],
        _group_index_map(group_data),
        df['mass'].to_numpy(),
        df['metallicity'].to_numpy(),
        df['age'].to_numpy(),
        n_groups,
    )
    group_data['metallicity_stellar'] = _safe_divide(metal_mass, total_mass, fill=0.0)
    group_mass_star = group_data['mass_star'].to_numpy()
    group_data['age_mass_weighted'] = _safe_divide(age_mass, group_mass_star, fill=0.0)
    group_data['age_metal_weighted'] = _safe_divide(age_metal, metal_mass, fill=0.0)
    group_data['sfr_100'] = young_mass / 100.e6
    return

  group_ids, rows = _group_ids_and_rows(data_manager, group_name, 'star', df, groupID_key)
  masses = df['mass'].to_numpy()[rows]
  metallicities = df['metallicity'].to_numpy()[rows]
  ages = df['age'].to_numpy()[rows]

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      metallicities = metallicities[keep]
      ages = ages[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)
  order, unique_groups, start, end = sort_by_group(group_idx)
  total_mass, metal_mass, age_mass, age_metal, young_mass = compute_star_scalar_sums(
      unique_groups.astype(np.int64),
      start.astype(np.int64),
      end.astype(np.int64),
      masses[order],
      metallicities[order],
      ages[order],
      n_groups,
  )

  # metallicity
  group_data['metallicity_stellar'] = _safe_divide(metal_mass, total_mass, fill=0.0)

  # age
  group_mass_star = group_data['mass_star'].to_numpy()
  group_data['age_mass_weighted'] = _safe_divide(age_mass, group_mass_star, fill=0.0)
  group_data['age_metal_weighted'] = _safe_divide(age_metal, metal_mass, fill=0.0)
  group_data['sfr_100'] = young_mass / 100.e6

def bh_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['bh']

  if _uses_halo_id_arrays(data_manager, group_name, ['bh']):
    n_groups = len(group_data)
    masses = df['mass'].to_numpy()
    max_mass, bhmdot = compute_membership_array_bh_max(
        data_manager.halo_id_arrays['bh'],
        _group_index_map(group_data),
        masses,
        df['bhmdot'].to_numpy(),
        n_groups,
    )
    valid = np.isfinite(max_mass) & (max_mass > 0.0) & np.isfinite(bhmdot)
    bhmdot_out = np.zeros(n_groups)
    bhmass_out = np.zeros(n_groups)
    bhmdot_out[valid] = bhmdot[valid]
    bhmass_out[valid] = max_mass[valid]
    group_data['bhmdot'] = bhmdot_out

    FRAD = 0.1
    edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value
    group_data['bh_fedd'] = _safe_divide(bhmdot_out, edd_factor * bhmass_out, fill=0.0)
    return

  group_ids, rows = _group_ids_and_rows(data_manager, group_name, 'bh', df, groupID_key)
  masses = df['mass'].to_numpy()[rows]
  bhmdots = df['bhmdot'].to_numpy()[rows]

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      bhmdots = bhmdots[keep]

  if len(masses) == 0:
      group_data['bhmdot'] = np.zeros(len(group_data))
      group_data['bh_fedd'] = np.zeros(len(group_data))
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # find most massive BH per group
  max_idx = max_idx_per_group(masses, group_idx, n_groups)
  valid = max_idx >= 0

  # bhmdot of most massive BH
  bhmdot = np.zeros(n_groups)
  bhmdot[valid] = bhmdots[max_idx[valid]]
  group_data['bhmdot'] = bhmdot

  # Eddington fraction
  FRAD = 0.1
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value

  bh_mass = np.zeros(n_groups)
  bh_mass[valid] = masses[max_idx[valid]]
  group_data['bh_fedd'] = _safe_divide(bhmdot, edd_factor * bh_mass, fill=0.0)

def calculate_local_densities(data_manager: DataManager) -> None:

  config = data_manager.config
  groups = config['groups']

  for group in groups:
    group_data = data_manager.group_data[group]

    if len(group_data) == 0:
      print(f"No group data!")
      continue

    pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    mass = group_data['mass_total'].to_numpy()

    # REVIEW: moving a FOF6D optimisation into this function
    # previously the pandas .explode() calls were memory-intensive
    boxsize = data_manager.simulation['boxsize']
    pos = np.where(pos > boxsize, pos - boxsize, pos)
    pos = np.where(pos < 0, pos + boxsize, pos)
    tree = KDTree(pos, boxsize=boxsize)
    for radius in [300., 1000., 3000.]:
      volume = 4./3. * np.pi * radius**3
      index_lists = tree.query_ball_point(pos, radius, workers=-1) # workers=-1 means all processors are used (from documentation)
      mass_sums = np.array([mass[il].sum() for il in index_lists])
      counts = np.array([len(il) for il in index_lists])

      group_data[f'local_mass_density_{int(radius)}'] = mass_sums / volume
      group_data[f'local_number_density_{int(radius)}'] = counts / volume


def _flatten_neighbor_lists(neighbor_lists):
    offsets = np.zeros(len(neighbor_lists) + 1, dtype=np.int64)
    for i, neighbours in enumerate(neighbor_lists):
        offsets[i + 1] = offsets[i] + len(neighbours)

    indices = np.empty(offsets[-1], dtype=np.int64)
    for i, neighbours in enumerate(neighbor_lists):
        start = offsets[i]
        end = offsets[i + 1]
        if end > start:
            indices[start:end] = neighbours

    return offsets, indices



def assign_galaxy_hydrogen_masses(data_manager: DataManager) -> None:
  """Assign halo-level HI and H2 gas to galaxies in the same top-level host.

  Gas particles are grouped by host halo, then each gas particle is assigned to the
  best galaxy candidate in that host using periodic distance and galaxy mass weighting."""
  if 'galaxies' not in data_manager.group_data or 'gas' not in data_manager.data:
    return

  group_data = data_manager.group_data['galaxies']
  gas = data_manager.data['gas']
  if len(group_data) == 0 or len(gas) == 0:
    return

  if 'gas' in data_manager.halo_id_arrays:
    gas_halos = data_manager.get_halo_ids('gas', mode='top')
  else:
    gas_halos = gas['HaloID'].to_numpy()

  if '_central_host_halo_index' in group_data:
    galaxy_halos = group_data['_central_host_halo_index'].to_numpy()
  else:
    galaxy_halos = group_data['parent_halo_index'].to_numpy()

  valid_gas = gas_halos >= 0
  valid_galaxies = galaxy_halos >= 0
  if not np.any(valid_gas) or not np.any(valid_galaxies):
    group_data['mass_HI'] = np.zeros(len(group_data))
    group_data['mass_H2'] = np.zeros(len(group_data))
    return

  # Sort gas and galaxies by host halo so the kernel can sweep matching halo blocks.
  gas_order, gas_unique_halos, gas_start, gas_end = sort_by_group(gas_halos[valid_gas])
  galaxy_order, galaxy_unique_halos, galaxy_start, galaxy_end = sort_by_group(galaxy_halos[valid_galaxies])
  galaxy_rows = np.flatnonzero(valid_galaxies)[galaxy_order].astype(np.int64)

  galaxy_HI, galaxy_H2 = compute_galaxy_hydrogen_assignment(
      gas_unique_halos.astype(np.int64),
      gas_start.astype(np.int64),
      gas_end.astype(np.int64),
      galaxy_unique_halos.astype(np.int64),
      galaxy_start.astype(np.int64),
      galaxy_end.astype(np.int64),
      gas[['x', 'y', 'z']].to_numpy()[valid_gas][gas_order],
      gas['mass_HI'].to_numpy()[valid_gas][gas_order],
      gas['mass_H2'].to_numpy()[valid_gas][gas_order],
      group_data[['x_total', 'y_total', 'z_total']].to_numpy()[valid_galaxies][galaxy_order],
      group_data['mass_total'].to_numpy()[valid_galaxies][galaxy_order],
      galaxy_rows,
      len(group_data),
      data_manager.simulation['boxsize'],
  )
  group_data['mass_HI'] = galaxy_HI
  group_data['mass_H2'] = galaxy_H2

def calculate_aperture_masses(data_manager, config):
    """Compute 30 kpc aperture masses and velocity dispersions around galaxies.

    Particles and galaxies are grouped by parent halo so each KDTree only covers a
    halo-local particle set. The include matrix maps raw particle components onto
    outputs such as gas, HI, H2, dust, total, and baryon."""

    group_data = data_manager.group_data['galaxies']
    n_galaxies = len(group_data)
    galaxy_pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    parent_halo = group_data['parent_halo_index'].to_numpy()
    aperture = 30. # as defined previously

    # use helper function
    include_hydrogen = 'gas' in config['ptypes'] and {'mass_HI', 'mass_H2'}.issubset(data_manager.data['gas'].columns)
    include_dust = 'gas' in config['ptypes'] and 'dustmass' in data_manager.data['gas'].columns
    all_pos, all_mass, all_codes, all_halos, ptype_names, all_vel = extract_particle_arrays(
        data_manager, config, include_hydrogen=include_hydrogen, include_dust=include_dust, include_velocities=True
    )
    # may want to check the helper function include_hydrogen part, thought it might be useful in future
    n_ptypes = len(ptype_names)
    # include_matrix[out, code] records which particle components contribute to each
    # output component, including derived total and baryon apertures.
    output_names = ptype_names + ['total', 'baryon']
    output_index = {name: i for i, name in enumerate(output_names)}
    include_matrix = np.zeros((len(output_names), n_ptypes), dtype=np.bool_)
    for i in range(n_ptypes):
        include_matrix[i, i] = True
    for i, name in enumerate(ptype_names):
        if name in config['ptypes']:
            include_matrix[output_index['total'], i] = True
        if name in config['ptypes_baryon']:
            include_matrix[output_index['baryon'], i] = True

    # pre-sort particles by halo
    order, unique_halos, h_start, h_end = sort_by_group(all_halos)
    all_pos = all_pos[order]
    all_mass = all_mass[order]
    all_codes = all_codes[order]
    all_vel = all_vel[order]

    # pre-sort galaxies by parent halo
    gal_order, halos_with_galaxies, gal_start, gal_end = sort_by_group(parent_halo)

    result = np.zeros((n_galaxies, len(output_names)))
    velocity_result = np.zeros((n_galaxies, len(output_names)))
    boxsize = data_manager.simulation['boxsize']

    # Work halo-by-halo so neighbor searches stay small and respect host boundaries.
    for h in range(len(unique_halos)):
        halo_id = unique_halos[h]
        halo_pos = all_pos[h_start[h]:h_end[h]]
        halo_mass = all_mass[h_start[h]:h_end[h]]
        halo_codes = all_codes[h_start[h]:h_end[h]]
        halo_vel = all_vel[h_start[h]:h_end[h]]

        # guard
        if len(halo_pos) == 0:
            continue

        gh_idx = np.searchsorted(halos_with_galaxies, halo_id) # find where the hid sits in halos_with_galaxies
        if gh_idx >= len(halos_with_galaxies) or halos_with_galaxies[gh_idx] != halo_id:
            continue # skip halos with no galaxies
        gal_indices = gal_order[gal_start[gh_idx]:gal_end[gh_idx]]

        # guard
        if len(gal_indices) == 0:
            continue

        # build KDTree (explained in FOF6D code)
        # Wrap into the periodic box before using a periodic KDTree, so galaxies near
        # boundaries can see particles across the box edge.
        halo_pos_wrapped = np.mod(halo_pos, boxsize)
        galaxy_pos_wrapped = np.mod(galaxy_pos[gal_indices], boxsize)
        tree = KDTree(halo_pos_wrapped, boxsize=boxsize)
        try:
            neighbor_lists = tree.query_ball_point(galaxy_pos_wrapped, aperture, workers=-1)
        except TypeError:
            neighbor_lists = tree.query_ball_point(galaxy_pos_wrapped, aperture)

        neighbor_offsets, neighbor_indices = _flatten_neighbor_lists(neighbor_lists)
        masses_local, sigmas_local = compute_aperture_component_properties(
            neighbor_offsets,
            neighbor_indices,
            halo_mass,
            halo_codes.astype(np.int64),
            halo_vel,
            include_matrix,
        )
        result[gal_indices, :] = masses_local
        velocity_result[gal_indices, :] = sigmas_local

    for i, name in enumerate(ptype_names):
        group_data[f'mass_{name}_30kpc'] = result[:, i]
        group_data[f'velocity_dispersion_{name}_30kpc'] = velocity_result[:, i]

    group_data['mass_total_30kpc'] = result[:, output_index['total']]
    group_data['velocity_dispersion_total_30kpc'] = velocity_result[:, output_index['total']]
    group_data['velocity_dispersion_baryon_30kpc'] = velocity_result[:, output_index['baryon']]

def calculate_group_properties(data_manager: DataManager) -> None:
  data_manager.logger.info('Calculating group properties...')
  t1 = perf_counter()

  # admin
  config = data_manager.config

  groups = config['groups']

  t2 = perf_counter()
  if 'halos' in groups:
    _load_halo_common_particle_inputs(data_manager)
  else:
    for ptype in config['ptypes']:
      data_manager.load_property('potential', ptype)
  t3 = perf_counter()
  t4 = t5 = t6 = t7 = t8 = t9 = t3

  to_process = config['to_process']
  aperture_velocities_needed = 'apertures' in to_process and 'galaxies' in config['groups']

  # unnecessary columns
  columns_to_drop = ['potential'] if aperture_velocities_needed else ['vx', 'vy', 'vz', 'potential']

  # order to iterate over
  ptype_order = ['total', 'dm', 'baryon', 'gas', 'star', 'bh']
  # drop the necessary columns afterwards
  drop_after = {'dm', 'gas', 'star', 'bh'}

  # common group properties
  phase_start = perf_counter()
  for ptype in ptype_order:
      if ptype not in to_process:
          continue
      for group in groups:
          common_group_properties(data_manager, group, ptype)
      if ptype in drop_after:
          data_manager.data[ptype].drop(columns=columns_to_drop, inplace=True, errors='ignore')
  data_manager.logger.info(f'Common group properties done in {perf_counter() - phase_start:.2f} seconds.')

  _assign_central_galaxies(data_manager)

  # gas properties
  if 'gas' in to_process:
    phase_start = perf_counter()
    t4 = perf_counter()
    for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
      data_manager.load_property(property, 'gas')
    data_manager.load_property('dustmass', 'gas', optional=True, default=0.)
    t5 = perf_counter()

    # gas masses
    data = data_manager.data['gas']
    mass = data['mass'].to_numpy()
    rho = data['rho'].to_numpy()
    fHI_ism = data['nh'].to_numpy().copy()
    fH2_ism = data['fH2'].to_numpy().copy()
    fHI = fHI_ism.copy()
    fH2 = fH2_ism.copy()
    fH2[rho < config['nHlim']] = 0.
    not_conserving_mass = (fHI + fH2) > 1.
    fHI[not_conserving_mass] = 1. - fH2[not_conserving_mass]

    data['fHI'] = fHI
    data['mass_HI'] = config['XH'] * fHI * mass
    data['mass_H2'] = config['XH'] * fH2 * mass
    data['mass_HI_ism'] = config['XH'] * fHI_ism * mass
    data['mass_H2_ism'] = config['XH'] * fH2_ism * mass

    for group in groups:
      gas_group_properties(data_manager, group)
    if 'galaxies' in groups:
      assign_galaxy_hydrogen_masses(data_manager)
    data_manager.logger.info(f'Gas group properties done in {perf_counter() - phase_start:.2f} seconds.')

  # star properties
  if 'star' in to_process:
    phase_start = perf_counter()
    t6 = perf_counter()
    for property in ['age', 'metallicity']:
      data_manager.load_property(property, 'star')
    t7 = perf_counter()

    for group in groups:
      star_group_properties(data_manager, group)
    data_manager.logger.info(f'Star group properties done in {perf_counter() - phase_start:.2f} seconds.')

  # bh properties
  if 'bh' in to_process:
    phase_start = perf_counter()
    t8 = perf_counter()
    for property in ['bhmdot']:
      data_manager.load_property(property, 'bh')
    t9 = perf_counter()

    for group in groups:
      bh_group_properties(data_manager, group)
    data_manager.logger.info(f'BH group properties done in {perf_counter() - phase_start:.2f} seconds.')

  # apertures
  if 'apertures' in to_process and 'galaxies' in config['groups']:
    phase_start = perf_counter()
    calculate_aperture_masses(data_manager, config)
    data_manager.logger.info(f'Aperture properties done in {perf_counter() - phase_start:.2f} seconds.')

  if aperture_velocities_needed:
    for ptype in drop_after:
      if ptype in data_manager.data:
        data_manager.data[ptype].drop(columns=['vx', 'vy', 'vz'], inplace=True, errors='ignore')

  # densities
  if 'local_densities' in to_process:
    phase_start = perf_counter()
    calculate_local_densities(data_manager)
    data_manager.logger.info(f'Local density properties done in {perf_counter() - phase_start:.2f} seconds.')

  phase_start = perf_counter()
  _assign_halo_source_properties(data_manager)
  data_manager.logger.info(f'Halo source metadata done in {perf_counter() - phase_start:.2f} seconds.')

  t10 = perf_counter()
  data_manager.logger.info(f'Calculating group properties done in {t10-t1:.2f} seconds. (Load reduced: {t10-t9 + t8-t7 + t6-t5 + t4-t3 + t2-t1:.2f} seconds.)')
