import h5py
import numpy as np
from yaml import safe_dump

from octavian.utils.merge_catalogues import _halo_index_columns, merge_catalogues


def test_halo_index_columns_are_reader_metadata_driven():
    ahf_columns = _halo_index_columns({'halo_source': 'ahf'})
    hbt_columns = _halo_index_columns({'halo_source': 'hbt'})

    assert ahf_columns['halos'] == {'caesar_parent_halo_index', 'caesar_top_halo_index'}
    assert hbt_columns['halos'] == {'caesar_parent_halo_index', 'caesar_top_halo_index'}
    assert ahf_columns['galaxies'] == {
        'caesar_parent_halo_index',
        'caesar_top_halo_index',
        '_ahf_host_halo_index',
    }
    assert hbt_columns['galaxies'] == {
        'caesar_parent_halo_index',
        'caesar_top_halo_index',
        '_hbt_host_halo_index',
    }


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


def _write_rank(path, halo_rows, galaxy_rows):
    with h5py.File(path, 'w') as f:
        halos = f.create_group('halo_data')
        galaxies = f.create_group('galaxy_data')
        halos.create_group('dicts')
        galaxies.create_group('dicts')

        for key in [
            'groupID',
            'HBT_trackID',
            'HBT_parent_trackID',
            'HBT_top_trackID',
            'HBT_depth',
            'caesar_parent_halo_index',
            'caesar_top_halo_index',
            'central_galaxy',
        ]:
            halos.create_dataset(key, data=np.asarray([row[key] for row in halo_rows]))
        halos.create_dataset('dicts/masses.total', data=np.asarray([row['mass'] for row in halo_rows], dtype=float))
        _write_csr(halos, 'HBT_ancestor_trackIDs', [row['ancestors'] for row in halo_rows])
        _write_csr(halos, 'glist', [row['glist'] for row in halo_rows])

        for key in [
            'groupID',
            'parent_halo_index',
            'HBT_trackID',
            'HBT_parent_trackID',
            'HBT_top_trackID',
            'HBT_depth',
            'caesar_parent_halo_index',
            'caesar_top_halo_index',
            '_hbt_host_halo_index',
        ]:
            galaxies.create_dataset(key, data=np.asarray([row[key] for row in galaxy_rows]))
        galaxies.create_dataset('dicts/masses.total', data=np.asarray([row['mass'] for row in galaxy_rows], dtype=float))
        galaxies.create_dataset('dicts/masses.stellar', data=np.asarray([row['mass'] for row in galaxy_rows], dtype=float))
        _write_csr(galaxies, 'HBT_ancestor_trackIDs', [row['ancestors'] for row in galaxy_rows])
        _write_csr(galaxies, 'glist', [row['glist'] for row in galaxy_rows])


def test_merge_catalogues_remaps_hbt_indices_and_merges_list_columns(tmp_path):
    first = tmp_path / 'catalogue_0.hdf5'
    second = tmp_path / 'catalogue_1.hdf5'
    outfile = tmp_path / 'merged.hdf5'
    config = tmp_path / 'config.yaml'

    _write_rank(
        first,
        halo_rows=[
            {
                'groupID': 10, 'mass': 100.0, 'HBT_trackID': 10, 'HBT_parent_trackID': -1,
                'HBT_top_trackID': 10, 'HBT_depth': 0, 'caesar_parent_halo_index': -1,
                'caesar_top_halo_index': 10, 'central_galaxy': 0, 'ancestors': [], 'glist': [101, 102],
            },
            {
                'groupID': 20, 'mass': 10.0, 'HBT_trackID': 20, 'HBT_parent_trackID': 10,
                'HBT_top_trackID': 10, 'HBT_depth': 1, 'caesar_parent_halo_index': 10,
                'caesar_top_halo_index': 10, 'central_galaxy': -1, 'ancestors': [10], 'glist': [],
            },
        ],
        galaxy_rows=[
            {
                'groupID': 0, 'mass': 5.0, 'parent_halo_index': 10, 'HBT_trackID': 10,
                'HBT_parent_trackID': -1, 'HBT_top_trackID': 10, 'HBT_depth': 0,
                'caesar_parent_halo_index': -1, 'caesar_top_halo_index': 10, '_hbt_host_halo_index': 10,
                'ancestors': [], 'glist': [11],
            },
            {
                'groupID': 1, 'mass': 50.0, 'parent_halo_index': 20, 'HBT_trackID': 20,
                'HBT_parent_trackID': 10, 'HBT_top_trackID': 10, 'HBT_depth': 1,
                'caesar_parent_halo_index': 10, 'caesar_top_halo_index': 10, '_hbt_host_halo_index': 10,
                'ancestors': [10], 'glist': [21, 22],
            },
        ],
    )
    _write_rank(
        second,
        halo_rows=[
            {
                'groupID': 30, 'mass': 50.0, 'HBT_trackID': 30, 'HBT_parent_trackID': -1,
                'HBT_top_trackID': 30, 'HBT_depth': 0, 'caesar_parent_halo_index': -1,
                'caesar_top_halo_index': 30, 'central_galaxy': 0, 'ancestors': [], 'glist': [301],
            },
        ],
        galaxy_rows=[
            {
                'groupID': 0, 'mass': 20.0, 'parent_halo_index': 30, 'HBT_trackID': 30,
                'HBT_parent_trackID': -1, 'HBT_top_trackID': 30, 'HBT_depth': 0,
                'caesar_parent_halo_index': -1, 'caesar_top_halo_index': 30, '_hbt_host_halo_index': 30,
                'ancestors': [], 'glist': [31],
            },
        ],
    )

    config.write_text(safe_dump({
        'halo_source': 'hbt',
        'dataset_columns': {
            'groupID': {'halos': 'groupID', 'galaxies': 'groupID'},
            'parent_halo_index': 'parent_halo_index',
            'central_galaxy': {'halos': 'central_galaxy'},
        },
        'dataset_columns_by_halo_source': {
            'hbt': {
                'HBT_trackID': {'halos': 'HBT_trackID', 'galaxies': 'HBT_trackID'},
                'HBT_parent_trackID': {'halos': 'HBT_parent_trackID', 'galaxies': 'HBT_parent_trackID'},
                'HBT_top_trackID': {'halos': 'HBT_top_trackID', 'galaxies': 'HBT_top_trackID'},
                'HBT_depth': {'halos': 'HBT_depth', 'galaxies': 'HBT_depth'},
                'caesar_parent_halo_index': {'halos': 'caesar_parent_halo_index', 'galaxies': 'caesar_parent_halo_index'},
                'caesar_top_halo_index': {'halos': 'caesar_top_halo_index', 'galaxies': 'caesar_top_halo_index'},
                '_hbt_host_halo_index': {'galaxies': '_hbt_host_halo_index'},
            },
        },
        'list_dataset_columns_by_halo_source': {
            'hbt': {
                'HBT_ancestor_trackIDs': {'halos': 'HBT_ancestor_trackIDs', 'galaxies': 'HBT_ancestor_trackIDs'},
            },
        },
    }))

    merge_catalogues([str(first), str(second)], str(outfile), str(config))

    with h5py.File(outfile, 'r') as f:
        halos = f['halo_data']
        galaxies = f['galaxy_data']

        assert halos['HBT_trackID'][:].tolist() == [20, 30, 10]
        assert halos['caesar_parent_halo_index'][:].tolist() == [2, -1, -1]
        assert halos['caesar_top_halo_index'][:].tolist() == [2, 1, 2]
        assert halos['central_galaxy'][:].tolist() == [-1, 1, 0]

        assert galaxies['parent_halo_index'][:].tolist() == [2, 1, 0]
        assert galaxies['caesar_parent_halo_index'][:].tolist() == [-1, -1, 2]
        assert galaxies['caesar_top_halo_index'][:].tolist() == [2, 1, 2]
        assert galaxies['_hbt_host_halo_index'][:].tolist() == [2, 1, 2]

        assert halos['HBT_ancestor_trackIDs_lengths'][:].tolist() == [1, 0, 0]
        assert halos['HBT_ancestor_trackIDs_indices'][:].tolist() == [10]
        assert galaxies['HBT_ancestor_trackIDs_lengths'][:].tolist() == [0, 0, 1]
        assert galaxies['HBT_ancestor_trackIDs_indices'][:].tolist() == [10]

        assert halos['glist_lengths'][:].tolist() == [0, 1, 2]
        assert halos['glist_indices'][:].tolist() == [301, 101, 102]
        assert galaxies['glist_lengths'][:].tolist() == [1, 1, 2]
        assert galaxies['glist_indices'][:].tolist() == [11, 31, 21, 22]

        assert halos['galaxy_index_list'].dtype.kind in 'iu'
        assert halos['galaxy_index_list'][:].tolist() == [2, 1, 0]
        assert halos['galaxy_index_list_lengths'][:].tolist() == [1, 1, 1]
