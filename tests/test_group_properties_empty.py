from types import SimpleNamespace

import numpy as np
import pandas as pd

from octavian.group_properties_calc.calculate_group_properties import (
    common_group_properties,
    star_group_properties,
)


def _empty_star_dataframe():
    return pd.DataFrame({
        'x': np.array([], dtype=float),
        'y': np.array([], dtype=float),
        'z': np.array([], dtype=float),
        'vx': np.array([], dtype=float),
        'vy': np.array([], dtype=float),
        'vz': np.array([], dtype=float),
        'mass': np.array([], dtype=float),
        'potential': np.array([], dtype=float),
        'age': np.array([], dtype=float),
        'metallicity': np.array([], dtype=float),
        'HaloID': np.array([], dtype=np.int64),
        'GalID': np.array([], dtype=np.int64),
    })


def test_empty_star_halo_membership_initializes_mass_star():
    halo_data = pd.DataFrame(index=np.asarray([0, 1], dtype=np.int64))
    data_manager = SimpleNamespace(
        config={'groupIDs': {'halos': 'HaloID'}},
        group_data={'halos': halo_data},
        data={'star': _empty_star_dataframe()},
        halo_id_arrays={'star': np.empty((0, 2), dtype=np.int64)},
    )

    common_group_properties(data_manager, 'halos', 'star')

    assert halo_data['nstar'].to_numpy().tolist() == [0, 0]
    assert halo_data['mass_star'].to_numpy().tolist() == [0.0, 0.0]

    star_group_properties(data_manager, 'halos')

    assert halo_data['age_mass_weighted'].to_numpy().tolist() == [0.0, 0.0]
    assert halo_data['sfr_100'].to_numpy().tolist() == [0.0, 0.0]


def test_empty_star_regular_path_initializes_mass_star():
    halo_data = pd.DataFrame(index=np.asarray([0, 1], dtype=np.int64))
    data_manager = SimpleNamespace(
        config={'groupIDs': {'halos': 'HaloID'}},
        group_data={'halos': halo_data},
        data={'star': _empty_star_dataframe()},
        halo_id_arrays={},
    )

    common_group_properties(data_manager, 'halos', 'star')

    assert halo_data['nstar'].to_numpy().tolist() == [0, 0]
    assert halo_data['mass_star'].to_numpy().tolist() == [0.0, 0.0]
