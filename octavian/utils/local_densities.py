from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


LOCAL_DENSITY_RADII = (300.0, 1000.0, 3000.0)


def calculate_local_density_arrays(positions, masses, boxsize, radii=LOCAL_DENSITY_RADII, workers=1):
  positions = np.asarray(positions)
  masses = np.asarray(masses)
  boxsize = float(np.asarray(boxsize).reshape(-1)[0])

  results = {}
  for radius in radii:
    radius_int = int(radius)
    results[f'local_mass_density_{radius_int}'] = np.empty(len(positions), dtype=float)
    results[f'local_number_density_{radius_int}'] = np.empty(len(positions), dtype=float)

  if len(positions) == 0:
    return results

  positions = np.mod(positions, boxsize)
  tree = KDTree(positions, boxsize=boxsize)

  for radius in radii:
    radius_int = int(radius)
    volume = 4. / 3. * np.pi * radius**3
    try:
      index_lists = tree.query_ball_point(positions, radius, workers=workers)
    except TypeError:
      index_lists = tree.query_ball_point(positions, radius)
    results[f'local_mass_density_{radius_int}'] = np.asarray(
      [masses[indices].sum() for indices in index_lists],
      dtype=float,
    ) / volume
    results[f'local_number_density_{radius_int}'] = np.asarray(
      [len(indices) for indices in index_lists],
      dtype=float,
    ) / volume

  return results
