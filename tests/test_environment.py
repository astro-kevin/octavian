from __future__ import annotations

import numpy as np
import pandas as pd

from octavian.mpi.environment import default_environment_callback


def test_environment_callback_computes_local_densities() -> None:
  halos = pd.DataFrame(
    {
      "HaloID": [1, 2],
      "x_total": [0.0, 100.0],
      "y_total": [0.0, 0.0],
      "z_total": [0.0, 0.0],
      "mass_total": [10.0, 20.0],
    }
  )
  galaxies = pd.DataFrame(
    {
      "GalID": [10, 11],
      "x_total": [0.0, 50.0],
      "y_total": [0.0, 0.0],
      "z_total": [0.0, 0.0],
      "mass_total": [5.0, 15.0],
    }
  )

  halos_env, galaxies_env, meta = default_environment_callback(halos, galaxies, {})

  for radius in (300, 1000, 3000):
    mass_col = f"local_mass_density_{radius}"
    number_col = f"local_number_density_{radius}"
    assert mass_col in halos_env
    assert number_col in halos_env
    assert mass_col in galaxies_env
    assert number_col in galaxies_env
    assert np.isfinite(halos_env[mass_col]).all()
    assert np.isfinite(galaxies_env[mass_col]).all()

  assert meta["halos_processed"] == 2
  assert meta["galaxies_processed"] == 2
