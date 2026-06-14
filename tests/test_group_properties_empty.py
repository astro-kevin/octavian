from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from octavian.data_manager.save_group_properties import save_group_properties
from octavian.group_properties_calc.calculate_group_properties import (
    common_group_properties,
    star_group_properties,
)
from octavian.utils.hdf5_metadata import COMPLETE_ATTR


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


def test_empty_total_halo_membership_initializes_completion_markers():
    halo_data = pd.DataFrame(index=np.asarray([0, 1], dtype=np.int64))
    data_manager = SimpleNamespace(
        config={
            'groupIDs': {'halos': 'HaloID'},
            'ptypes': ['star'],
            'ptypes_baryon': ['star'],
        },
        group_data={'halos': halo_data},
        data={'star': _empty_star_dataframe()},
        halo_id_arrays={'star': np.empty((0, 2), dtype=np.int64)},
    )

    common_group_properties(data_manager, 'halos', 'total')

    assert halo_data['ntotal'].to_numpy().tolist() == [0, 0]
    assert halo_data['mass_total'].to_numpy().tolist() == [0.0, 0.0]
    assert halo_data['velocity_dispersion_total'].to_numpy().tolist() == [0.0, 0.0]
    assert halo_data['temperature'].to_numpy().tolist() == [0.0, 0.0]


def test_save_group_properties_writes_empty_completion_schema(tmp_path):
    outfile = tmp_path / 'empty_rank.hdf5'
    halos = pd.DataFrame({'HaloID': np.asarray([], dtype=np.int64)})
    galaxies = pd.DataFrame({'GalID': np.asarray([], dtype=np.int64)})
    data_manager = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        simulation={'boxsize': 1000.0},
        config={
            'groups': ['halos', 'galaxies'],
            'groupIDs': {'halos': 'HaloID', 'galaxies': 'GalID'},
            'dataset_columns': {
                'groupID': {'halos': 'HaloID', 'galaxies': 'GalID'},
                'pos': {'halos': ['x_total', 'y_total', 'z_total']},
                'dicts/masses.total': {'halos': 'mass_total'},
                'dicts/virial_quantities.temperature': {'halos': 'temperature'},
                'dicts/velocity_dispersions.total': {'halos': 'velocity_dispersion_total'},
            },
            'list_dataset_columns': {},
        },
        group_data={'halos': halos, 'galaxies': galaxies},
        particle_lists={'halos': {}, 'galaxies': {}},
    )

    save_group_properties(data_manager, str(outfile))

    with h5py.File(outfile, 'r') as handle:
        assert bool(handle.attrs[COMPLETE_ATTR]) is True
        halos_out = handle['halo_data']
        assert halos_out['groupID'].shape == (0,)
        assert halos_out['pos'].shape == (0, 3)
        assert halos_out['dicts/masses.total'].shape == (0,)
        assert halos_out['dicts/virial_quantities.temperature'].shape == (0,)
        assert halos_out['dicts/velocity_dispersions.total'].shape == (0,)


def test_save_group_properties_writes_completion_schema_when_config_omits_markers(tmp_path):
    outfile = tmp_path / 'stale_config_rank.hdf5'
    halos = pd.DataFrame(
        {
            'HaloID': np.asarray([10, 20], dtype=np.int64),
            'mass_total': np.asarray([1.0, 2.0]),
            'velocity_dispersion_total': np.asarray([3.0, 4.0]),
            'temperature': np.asarray([5.0, 6.0]),
        },
        index=np.asarray([10, 20], dtype=np.int64),
    )
    galaxies = pd.DataFrame(
        {
            'GalID': np.asarray([], dtype=np.int64),
            'mass_total': np.asarray([], dtype=float),
        },
        index=np.asarray([], dtype=np.int64),
    )
    data_manager = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        simulation={'boxsize': 1000.0},
        config={
            'groups': ['halos', 'galaxies'],
            'groupIDs': {'halos': 'HaloID', 'galaxies': 'GalID'},
            'dataset_columns': {
                'groupID': {'halos': 'HaloID', 'galaxies': 'GalID'},
            },
            'list_dataset_columns': {},
        },
        group_data={'halos': halos, 'galaxies': galaxies},
        particle_lists={'halos': {}, 'galaxies': {}},
    )

    save_group_properties(data_manager, str(outfile))

    with h5py.File(outfile, 'r') as handle:
        halos_out = handle['halo_data']
        assert halos_out['dicts/masses.total'][:].tolist() == [1.0, 2.0]
        assert halos_out['dicts/velocity_dispersions.total'][:].tolist() == [3.0, 4.0]
        assert halos_out['dicts/virial_quantities.temperature'][:].tolist() == [5.0, 6.0]
        assert handle['galaxy_data']['dicts/masses.total'].shape == (0,)
