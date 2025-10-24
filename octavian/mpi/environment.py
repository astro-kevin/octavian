"""
Helpers for computing environment-dependent properties on the driver.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from octavian.backend import pd


def _compute_local_densities(
  frame: pd.DataFrame,
  radii: Sequence[float],
) -> pd.DataFrame:
  if frame.empty:
    for radius in radii:
      frame[f'local_mass_density_{int(radius)}'] = 0.0
      frame[f'local_number_density_{int(radius)}'] = 0.0
    return frame

  required = {'x_total', 'y_total', 'z_total', 'mass_total'}
  missing = required.difference(frame.columns)
  if missing:
    raise KeyError(f"Missing columns required for environment metrics: {sorted(missing)}")

  positions = frame[['x_total', 'y_total', 'z_total']].to_numpy(dtype=np.float64, copy=False)
  mass = frame['mass_total'].to_numpy(dtype=np.float64, copy=False)

  neighbors = NearestNeighbors()
  neighbors.fit(positions)

  for radius in radii:
    radius = float(radius)
    volume = 4.0 / 3.0 * np.pi * radius**3
    indices = neighbors.radius_neighbors(positions, radius=radius, return_distance=False)
    total_mass = np.fromiter((mass[idx].sum() for idx in indices), dtype=np.float64, count=positions.shape[0])
    counts = np.fromiter((len(idx) for idx in indices), dtype=np.int64, count=positions.shape[0])
    frame[f'local_mass_density_{int(radius)}'] = total_mass / volume
    frame[f'local_number_density_{int(radius)}'] = counts / volume

  return frame


def default_environment_callback(
  halos: pd.DataFrame,
  galaxies: pd.DataFrame,
  metadata: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
  """
  Lightweight placeholder that derives simple environment metadata.

  Runs after all Ray tasks finish, so it executes on the driver/home node.
  Users with richer requirements can supply their own callback to
  ``run_with_ray``; this default keeps a predictable baseline so downstream
  code can expect a minimal ``environment`` entry in the metadata.
  """
  radii: Tuple[float, float, float] = (300.0, 1000.0, 3000.0)

  halos_env = halos.copy()
  galaxies_env = galaxies.copy()

  def _needs_density(frame: pd.DataFrame) -> bool:
    return any(
      (f'local_mass_density_{int(radius)}' not in frame)
      or (f'local_number_density_{int(radius)}' not in frame)
      for radius in radii
    )

  if _needs_density(halos_env):
    halos_env = _compute_local_densities(halos_env, radii)
  if _needs_density(galaxies_env):
    galaxies_env = _compute_local_densities(galaxies_env, radii)

  env_meta: Dict[str, Any] = {
    "radii_kpc": list(radii),
    "halos_processed": int(len(halos_env)),
    "galaxies_processed": int(len(galaxies_env)),
  }
  return halos_env, galaxies_env, env_meta
