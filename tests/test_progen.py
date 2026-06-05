import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from octavian.progen import progen


HBT_DTYPE = np.dtype([
    ('TrackId', '<i8'),
    ('Nbound', '<i8'),
    ('Mbound', '<f4'),
    ('DescendantTrackId', '<i8'),
])


def _write_csr(group, name, sequences):
    lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    if lengths.sum():
        indices = np.concatenate([np.asarray(seq, dtype=np.int64) for seq in sequences])
    else:
        indices = np.empty(0, dtype=np.int64)
    group.create_dataset(f'{name}_indices', data=indices)
    group.create_dataset(f'{name}_offsets', data=offsets)
    group.create_dataset(f'{name}_lengths', data=lengths)


def _write_octavian(path, halo_rows, galaxy_rows=None, source='hbt', redshift=0.0):
    source_column = 'HBT_trackID' if source == 'hbt' else 'AHF_haloID'
    with h5py.File(path, 'w') as handle:
        header = handle.create_group('Header')
        header.attrs['Redshift'] = redshift

        halos = handle.create_group('halo_data')
        halos.create_dataset('HaloID', data=np.asarray([row['id'] for row in halo_rows], dtype=np.int64))
        halos.create_dataset('groupID', data=np.asarray([row['id'] for row in halo_rows], dtype=np.int64))
        halos.create_dataset(source_column, data=np.asarray([row['source'] for row in halo_rows], dtype=np.int64))
        halo_dicts = halos.create_group('dicts')
        halo_dicts.create_dataset('masses.total', data=np.asarray([row.get('mass', 1.0) for row in halo_rows], dtype=float))

        if galaxy_rows is None:
            return

        galaxies = handle.create_group('galaxy_data')
        galaxies.create_dataset('GalID', data=np.asarray([row['id'] for row in galaxy_rows], dtype=np.int64))
        galaxies.create_dataset('groupID', data=np.asarray([row['id'] for row in galaxy_rows], dtype=np.int64))
        galaxies.create_dataset(source_column, data=np.asarray([row['source'] for row in galaxy_rows], dtype=np.int64))
        galaxy_dicts = galaxies.create_group('dicts')
        galaxy_dicts.create_dataset('masses.stellar', data=np.asarray([row.get('mass', 1.0) for row in galaxy_rows], dtype=float))
        galaxy_dicts.create_dataset('masses.total', data=np.asarray([row.get('mass', 1.0) for row in galaxy_rows], dtype=float))
        _write_csr(galaxies, 'slist', [row.get('slist', []) for row in galaxy_rows])


def _write_snapshot(path, star_particle_ids):
    with h5py.File(path, 'w') as handle:
        stars = handle.create_group('PartType4')
        stars.create_dataset('ParticleIDs', data=np.asarray(star_particle_ids, dtype=np.int64))


def _write_hbt(root, snap, rows):
    directory = root / f'{snap:03d}'
    directory.mkdir(parents=True)
    data = np.zeros(len(rows), dtype=HBT_DTYPE)
    data['DescendantTrackId'] = -1
    for index, row in enumerate(rows):
        for key, value in row.items():
            data[key][index] = value
    with h5py.File(directory / f'SubSnap_{snap:03d}.0.hdf5', 'w') as handle:
        handle.create_dataset('Subhalos', data=data)


def _write_ahf_halos(path, rows):
    with open(path, 'w') as handle:
        handle.write('#ID(1) Mhalo(4)\n')
        for source_id, mass in rows:
            handle.write(f'{source_id} {mass}\n')


def _write_ahf_mtree_idx(path, rows):
    with open(path, 'w') as handle:
        handle.write('# HaloID(1) HaloID(2), MergerTree version: 1.3\n')
        for descendant, progenitor in rows:
            handle.write(f'{descendant} {progenitor}\n')


def _write_ahf_mtree(path, mapping):
    with open(path, 'w') as handle:
        handle.write(f'{len(mapping)}\n')
        for descendant, progenitors in mapping.items():
            handle.write(f'{descendant} {len(progenitors)}\n')
            for progenitor in progenitors:
                handle.write(f'{progenitor}\n')


def test_ahf_main_progenitor_maps_to_previous_octavian_halo_ids(tmp_path):
    current = tmp_path / 'oct_151.hdf5'
    previous = tmp_path / 'oct_150.hdf5'
    current_ahf = tmp_path / 'snap_151.AHF_halos'
    previous_ahf = tmp_path / 'snap_150.AHF_halos'
    _write_octavian(current, [{'id': 0, 'source': 101}, {'id': 1, 'source': 102}], source='ahf')
    _write_octavian(previous, [
        {'id': 10, 'source': 201},
        {'id': 20, 'source': 202},
        {'id': 30, 'source': 203},
    ], source='ahf')
    _write_ahf_halos(current_ahf, [(101, 1.0), (102, 1.0)])
    _write_ahf_halos(previous_ahf, [(201, 1.0), (202, 2.0), (203, 3.0)])
    _write_ahf_mtree_idx(tmp_path / 'snap_151.AHF_mtree_idx', [(101, 202), (102, 999)])
    _write_ahf_mtree(tmp_path / 'snap_151.AHF_mtree', {102: [203]})

    result = progen(
        [current, previous],
        [current_ahf, previous_ahf],
        group_type='halos',
        progenitors=1,
        halo_source='ahf',
        save=False,
    )

    assert result[0]['progen_halos'].tolist() == [20, 30]


def test_ahf_multiple_progenitors_are_sorted_by_previous_mhalo(tmp_path):
    current = tmp_path / 'oct_151.hdf5'
    previous = tmp_path / 'oct_150.hdf5'
    current_ahf = tmp_path / 'snap_151.AHF_halos'
    previous_ahf = tmp_path / 'snap_150.AHF_halos'
    _write_octavian(current, [{'id': 0, 'source': 101}], source='ahf')
    _write_octavian(previous, [{'id': 10, 'source': 201}, {'id': 20, 'source': 202}], source='ahf')
    _write_ahf_halos(current_ahf, [(101, 1.0)])
    _write_ahf_halos(previous_ahf, [(201, 5.0), (202, 10.0)])
    _write_ahf_mtree(tmp_path / 'snap_151.AHF_mtree', {101: [201, 202]})

    result = progen(
        [current, previous],
        [current_ahf, previous_ahf],
        group_type='halos',
        progenitors=2,
        halo_source='ahf',
        save=False,
    )

    assert result[0]['progen_halos'].tolist() == [[20, 10]]


def test_hbt_main_progenitor_prefers_stable_trackid_then_descendant_fallback(tmp_path):
    current = tmp_path / 'oct_151.hdf5'
    previous = tmp_path / 'oct_150.hdf5'
    hbt_root = tmp_path / 'hbt'
    _write_octavian(current, [
        {'id': 0, 'source': 100},
        {'id': 1, 'source': 200},
        {'id': 2, 'source': 300},
    ])
    _write_octavian(previous, [
        {'id': 7, 'source': 100},
        {'id': 8, 'source': 201},
        {'id': 9, 'source': 202},
    ])
    _write_hbt(hbt_root, 150, [
        {'TrackId': 100, 'Mbound': 1.0, 'Nbound': 10, 'DescendantTrackId': -1},
        {'TrackId': 201, 'Mbound': 4.0, 'Nbound': 40, 'DescendantTrackId': 200},
        {'TrackId': 202, 'Mbound': 8.0, 'Nbound': 80, 'DescendantTrackId': 200},
    ])

    result = progen(
        [current, previous],
        hbt_root,
        group_type='halos',
        progenitors=1,
        halo_source='hbt',
        snap_indices=[151, 150],
        save=False,
    )

    assert result[0]['progen_halos'].tolist() == [7, 9, -1]


def test_hbt_multiple_progenitors_include_stable_trackid_and_sort_by_mass(tmp_path):
    current = tmp_path / 'oct_151.hdf5'
    previous = tmp_path / 'oct_150.hdf5'
    hbt_root = tmp_path / 'hbt'
    _write_octavian(current, [{'id': 0, 'source': 200}])
    _write_octavian(previous, [
        {'id': 7, 'source': 200, 'mass': 6.0},
        {'id': 8, 'source': 201, 'mass': 4.0},
        {'id': 9, 'source': 202, 'mass': 8.0},
    ])
    _write_hbt(hbt_root, 150, [
        {'TrackId': 200, 'Mbound': 6.0, 'Nbound': 60, 'DescendantTrackId': -1},
        {'TrackId': 201, 'Mbound': 4.0, 'Nbound': 40, 'DescendantTrackId': 200},
        {'TrackId': 202, 'Mbound': 8.0, 'Nbound': 80, 'DescendantTrackId': 200},
    ])

    result = progen(
        [current, previous],
        hbt_root,
        group_type='halos',
        progenitors=3,
        halo_source='hbt',
        snap_indices=[151, 150],
        save=False,
    )

    assert result[0]['progen_halos'].tolist() == [[9, 7, 8]]


def test_group_type_all_writes_halos_and_star_overlap_galaxies(tmp_path):
    current = tmp_path / 'oct_151.hdf5'
    previous = tmp_path / 'oct_150.hdf5'
    current_snapshot = tmp_path / 'snap_151.hdf5'
    previous_snapshot = tmp_path / 'snap_150.hdf5'
    hbt_root = tmp_path / 'hbt'
    _write_octavian(
        current,
        [{'id': 0, 'source': 10}],
        galaxy_rows=[{'id': 1, 'source': 10, 'mass': 1.0, 'slist': [0, 1, 2]}],
    )
    _write_octavian(
        previous,
        [{'id': 20, 'source': 20}, {'id': 30, 'source': 30}],
        galaxy_rows=[
            {'id': 5, 'source': 20, 'mass': 10.0, 'slist': [0, 1]},
            {'id': 6, 'source': 30, 'mass': 1.0, 'slist': [2, 3]},
            {'id': 7, 'source': 999, 'mass': 100.0, 'slist': [4, 5, 6]},
        ],
    )
    _write_snapshot(current_snapshot, [101, 102, 103])
    _write_snapshot(previous_snapshot, [101, 999, 102, 103, 101, 102, 103])
    _write_hbt(hbt_root, 150, [
        {'TrackId': 20, 'Mbound': 50.0, 'Nbound': 50, 'DescendantTrackId': 10},
        {'TrackId': 30, 'Mbound': 10.0, 'Nbound': 10, 'DescendantTrackId': 10},
    ])

    result = progen(
        [current, previous],
        hbt_root,
        group_type='all',
        progenitors=2,
        halo_source='hbt',
        snap_indices=[151, 150],
        snapshot_files=[current_snapshot, previous_snapshot],
        save=True,
    )

    assert result[0]['progen_halos'].tolist() == [[20, 30]]
    assert result[0]['progen_galaxies'].tolist() == [[6, 5]]
    with h5py.File(current, 'r') as handle:
        assert handle['tree_data']['progen_halos'][:].tolist() == [[20, 30]]
        assert handle['tree_data']['progen_galaxies'][:].tolist() == [[6, 5]]


def test_all_progenitors_write_csr_datasets(tmp_path):
    current = tmp_path / 'oct_151.hdf5'
    previous = tmp_path / 'oct_150.hdf5'
    hbt_root = tmp_path / 'hbt'
    _write_octavian(current, [{'id': 0, 'source': 200}])
    _write_octavian(previous, [
        {'id': 7, 'source': 200, 'mass': 6.0},
        {'id': 8, 'source': 201, 'mass': 4.0},
        {'id': 9, 'source': 202, 'mass': 8.0},
    ])
    _write_hbt(hbt_root, 150, [
        {'TrackId': 200, 'Mbound': 6.0, 'Nbound': 60, 'DescendantTrackId': -1},
        {'TrackId': 201, 'Mbound': 4.0, 'Nbound': 40, 'DescendantTrackId': 200},
        {'TrackId': 202, 'Mbound': 8.0, 'Nbound': 80, 'DescendantTrackId': 200},
    ])

    progen(
        [current, previous],
        hbt_root,
        group_type='halos',
        progenitors='all',
        halo_source='hbt',
        snap_indices=[151, 150],
        save=True,
    )

    with h5py.File(current, 'r') as handle:
        tree = handle['tree_data']
        assert tree['progen_halos_indices'][:].tolist() == [9, 7, 8]
        assert tree['progen_halos_offsets'][:].tolist() == [0]
        assert tree['progen_halos_lengths'][:].tolist() == [3]


def test_hbt_m25n256_151_to_150_readonly_integration():
    if os.environ.get('OCTAVIAN_RUN_LOCAL_INTEGRATION') != '1':
        pytest.skip('set OCTAVIAN_RUN_LOCAL_INTEGRATION=1 to run local production-data integration test')

    current = Path('/disk04/kevin/ML/Octavian/HBT-m25n256/octavian_hbt_m25n256_151_subhalo.hdf5')
    previous = Path('/disk04/kevin/ML/Octavian/HBT-m25n256/octavian_hbt_m25n256_150_subhalo.hdf5')
    hbt_root = Path('/disk04/kevin/HBT-m25n256/')
    for path in (current, previous, hbt_root):
        if not path.exists():
            pytest.skip(f'{path} not available')

    result = progen(
        [current, previous],
        hbt_root,
        group_type='all',
        progenitors=1,
        halo_source='hbt',
        snap_indices=[151, 150],
        save=False,
    )[0]

    with h5py.File(current, 'r') as current_handle, h5py.File(previous, 'r') as previous_handle:
        current_halo_count = len(current_handle['halo_data']['HBT_trackID'])
        previous_halo_ids = previous_handle['halo_data']['HaloID'][:]

    halo_progenitors = result['progen_halos']
    valid = halo_progenitors >= 0
    assert len(halo_progenitors) == current_halo_count
    assert np.isin(halo_progenitors[valid], previous_halo_ids).all()
    assert valid.any()

    if 'progen_galaxies' in result and np.any(result['progen_galaxies'] >= 0):
        with h5py.File(previous, 'r') as previous_handle:
            previous_galaxy_ids = previous_handle['galaxy_data']['GalID'][:]
        valid_galaxies = result['progen_galaxies'] >= 0
        assert np.isin(result['progen_galaxies'][valid_galaxies], previous_galaxy_ids).all()
