from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from octavian.group_properties_calc.calculate_group_properties import calculate_aperture_masses


def _particle_frame(rows):
    return pd.DataFrame(rows, columns=[
        'HaloID', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mass',
    ])


def test_aperture_masses_include_derived_gas_components_without_duplicate_particles():
    gas = _particle_frame([
        [0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0],
        [0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 20.0],
    ])
    gas['mass_HI'] = [1.0, 2.0]
    gas['mass_H2'] = [0.5, 0.0]
    gas['dustmass'] = [0.1, 0.2]
    gas['rho'] = [1.0, 0.01]

    star = _particle_frame([
        [0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0],
    ])

    galaxies = pd.DataFrame({
        'x_total': [0.0, 100.0],
        'y_total': [0.0, 0.0],
        'z_total': [0.0, 0.0],
        'parent_halo_index': [0, 0],
    })

    data_manager = SimpleNamespace(
        data={'gas': gas, 'star': star},
        group_data={'galaxies': galaxies},
        simulation={'boxsize': 1000.0},
    )
    config = {
        'ptypes': ['gas', 'star'],
        'ptypes_baryon': ['gas', 'star'],
        'nHlim': 0.13,
    }

    calculate_aperture_masses(data_manager, config)

    assert galaxies['mass_gas_30kpc'].to_numpy().tolist() == pytest.approx([10.0, 20.0])
    assert galaxies['mass_star_30kpc'].to_numpy().tolist() == pytest.approx([5.0, 0.0])
    assert galaxies['mass_HI_30kpc'].to_numpy().tolist() == pytest.approx([1.0, 2.0])
    assert galaxies['mass_H2_30kpc'].to_numpy().tolist() == pytest.approx([0.5, 0.0])
    assert galaxies['mass_dust_30kpc'].to_numpy().tolist() == pytest.approx([0.1, 0.0])
    assert galaxies['mass_total_30kpc'].to_numpy().tolist() == pytest.approx([15.0, 20.0])
