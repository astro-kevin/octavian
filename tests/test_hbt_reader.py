from types import SimpleNamespace

import h5py
import pytest
import numpy as np
from yaml import safe_dump

from octavian.halo_filter import filter_snapshot
from octavian.halo_reader import load_halo_tree, membership_selected_particles_dense
from octavian.halo_reader.hbt import (
    build_hbt_snapshot_membership_arrays,
    build_parent_ids,
    gather_subsnap_files,
    load_hbt,
    read_particles,
    read_subhalos,
)
from octavian.utils.hdf5_metadata import COMPLETE_ATTR, FILE_TYPE_ATTR, read_simulation_metadata


HBT_DTYPE = np.dtype([
    ('TrackId', '<i8'),
    ('Nbound', '<i8'),
    ('Mbound', '<f4'),
    ('HostHaloId', '<i8'),
    ('Rank', '<i8'),
    ('Depth', '<i4'),
    ('NestedParentTrackId', '<i8'),
    ('NboundType', '<i8', (6,)),
])


def _write_subsnap(path, rows, particles):
    data = np.zeros(len(rows), dtype=HBT_DTYPE)
    for i, row in enumerate(rows):
        for key, value in row.items():
            data[key][i] = value

    with h5py.File(path, 'w') as f:
        f.create_dataset('Subhalos', data=data)
        dataset = f.create_dataset(
            'SubhaloParticles',
            shape=(len(particles),),
            dtype=h5py.vlen_dtype(np.dtype('uint64')),
        )
        for i, values in enumerate(particles):
            dataset[i] = np.asarray(values, dtype=np.uint64)


def test_gather_subsnap_files_accepts_specific_snapshot_directory(tmp_path):
    snap_dir = tmp_path / '050'
    snap_dir.mkdir()
    for file_index in (10, 0, 2):
        _write_subsnap(
            snap_dir / f'SubSnap_050.{file_index}.hdf5',
            [{'TrackId': file_index, 'HostHaloId': -1, 'Rank': 0, 'NestedParentTrackId': -1}],
            [[]],
        )

    files = gather_subsnap_files(snap_dir)

    assert [path.name for path in files] == [
        'SubSnap_050.0.hdf5',
        'SubSnap_050.2.hdf5',
        'SubSnap_050.10.hdf5',
    ]


def test_read_subhalos_and_particles_concatenate_split_files(tmp_path):
    first = tmp_path / 'SubSnap_050.0.hdf5'
    second = tmp_path / 'SubSnap_050.1.hdf5'
    _write_subsnap(
        first,
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': -1},
        ],
        [[11, 12], []],
    )
    _write_subsnap(
        second,
        [{'TrackId': 102, 'HostHaloId': 7, 'Rank': 2, 'NestedParentTrackId': 101}],
        [[21]],
    )

    files = [first, second]
    properties = read_subhalos(files)
    member_hids, member_pids = read_particles(files)

    assert properties['TrackId'].to_list() == [100, 101, 102]
    assert member_hids.tolist() == [0, 0, 2]
    assert member_pids.tolist() == [11, 12, 21]


def test_build_parent_ids_uses_nested_parent_then_fof_central_fallback(tmp_path):
    path = tmp_path / 'SubSnap_050.0.hdf5'
    _write_subsnap(
        path,
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': -1},
            {'TrackId': 102, 'HostHaloId': 7, 'Rank': 2, 'NestedParentTrackId': 101},
        ],
        [[], [], []],
    )

    properties = read_subhalos(path)

    assert build_parent_ids(properties).tolist() == [-1, 100, 101]


def _write_snapshot(path, particle_ids=None):
    if particle_ids is None:
        particle_ids = [11, 12, 99]
    particle_ids = np.asarray(particle_ids, dtype=np.int64)
    with h5py.File(path, 'w') as f:
        header = f.create_group('Header')
        header.attrs['BoxSize'] = 1000.0
        header.attrs['Omega0'] = 0.3
        header.attrs['OmegaLambda'] = 0.7
        header.attrs['HubbleParam'] = 0.68
        header.attrs['Redshift'] = 0.0
        header.attrs['Time'] = 1.0
        ptype = f.create_group('PartType1')
        ptype.create_dataset('ParticleIDs', data=particle_ids)
        ptype.create_dataset('Masses', data=np.ones(len(particle_ids), dtype=np.float64))


def _write_config(path, hbt_path, mode):
    path.write_text(safe_dump({
        'ptype_names': {'dm': 'PartType1'},
        'prop_aliases': {'pid': 'ParticleIDs'},
        'halo_source': 'hbt',
        'halo_mode': mode,
        'hbt_subhalo_path': str(hbt_path),
        'MINIMUM_DM_PER_HALO': 1,
    }))


def test_hbt_membership_arrays_are_scalar_with_reconstructable_ancestors(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    _write_snapshot(snapshot)
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
        ],
        [[11], [12]],
    )
    config = {
        'ptype_names': {'dm': 'PartType1'},
        'prop_aliases': {'pid': 'ParticleIDs'},
    }

    with h5py.File(snapshot, 'r') as f:
        result = build_hbt_snapshot_membership_arrays(f, config, hbt_dir)
        _, membership_arrays, _ = result

    halo_id_array = membership_arrays['PartType1']
    assert isinstance(halo_id_array, np.ndarray)
    assert halo_id_array.shape == (3,)
    assert halo_id_array.dtype == np.int32
    assert halo_id_array.tolist() == [1, 2, 0]
    assert membership_selected_particles_dense(halo_id_array, [0, 1, 2], result.ancestor_arrays).tolist() == [[0, -1], [0, 1], [-1, -1]]


def test_hbt_membership_matching_scans_snapshot_ptypes(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    with h5py.File(snapshot, 'w') as f:
        f.create_group('Header')
        gas = f.create_group('PartType0')
        gas.create_dataset('ParticleIDs', data=np.asarray([31, 11, 41], dtype=np.int64))
        gas.create_dataset('Masses', data=np.ones(3, dtype=np.float64))
        dm = f.create_group('PartType1')
        dm.create_dataset('ParticleIDs', data=np.asarray([12, 99, 21], dtype=np.int64))
        dm.create_dataset('Masses', data=np.ones(3, dtype=np.float64))
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
        ],
        [[12, 11], [21, 31]],
    )
    config = {
        'ptype_names': {'gas': 'PartType0', 'dm': 'PartType1'},
        'prop_aliases': {'pid': 'ParticleIDs'},
    }

    with h5py.File(snapshot, 'r') as f:
        result = build_hbt_snapshot_membership_arrays(f, config, hbt_dir)
        _, membership_arrays, counts = result

    assert counts == {'gas': 2, 'dm': 2}
    assert membership_selected_particles_dense(membership_arrays['PartType0'], [0, 1, 2], result.ancestor_arrays).tolist() == [
        [0, 1],
        [0, -1],
        [-1, -1],
    ]
    assert membership_selected_particles_dense(membership_arrays['PartType1'], [0, 1, 2], result.ancestor_arrays).tolist() == [
        [0, -1],
        [-1, -1],
        [0, 1],
    ]


def test_hbt_membership_stream_chooses_deepest_duplicate_particle(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    _write_snapshot(snapshot, particle_ids=[11, 12])
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
        ],
        [[11, 12], [12]],
    )
    config = {
        'ptype_names': {'dm': 'PartType1'},
        'prop_aliases': {'pid': 'ParticleIDs'},
        'hbt_subhalo_chunk_size': 1,
    }

    with h5py.File(snapshot, 'r') as f:
        result = build_hbt_snapshot_membership_arrays(f, config, hbt_dir)
        _, membership_arrays, counts = result

    assert counts == {'dm': 2}
    assert membership_selected_particles_dense(membership_arrays['PartType1'], [0, 1], result.ancestor_arrays).tolist() == [
        [0, -1],
        [0, 1],
    ]


def test_load_hbt_subhalo_uses_membership_array_builder(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    _write_snapshot(snapshot)
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
        ],
        [[11], [12]],
    )
    data_manager = SimpleNamespace(
        snapfile=str(snapshot),
        config={
            'ptype_names': {'dm': 'PartType1'},
            'ptypes': ['dm'],
            'prop_aliases': {'pid': 'ParticleIDs'},
        },
        data={'dm': {}},
        halo_id_arrays={},
    )
    data_manager.get_ptype_name = lambda ptype: data_manager.config['ptype_names'][ptype]

    load_hbt(data_manager, hbt_dir, mode='subhalo')

    assert data_manager.halo_tree.halo_ids.tolist() == [0, 1]
    assert data_manager.data['dm']['HaloID'].astype(int).tolist() == [0, 1, -1]
    assert membership_selected_particles_dense(data_manager.halo_id_arrays['dm'], [0, 1, 2]).tolist() == [
        [0, -1],
        [0, 1],
        [-1, -1],
    ]



def test_filter_snapshot_skips_empty_hbt_catalog(tmp_path, capsys):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    config = tmp_path / 'config.yaml'
    outfile = tmp_path / 'split'
    _write_snapshot(snapshot, particle_ids=[11, 12, 99])
    _write_subsnap(hbt_dir / 'SubSnap_050.0.hdf5', [], [])
    _write_config(config, hbt_dir, 'subhalo')

    filter_snapshot(str(snapshot), str(outfile), str(config), nsplit=2)

    captured = capsys.readouterr().out
    assert 'HBT catalog empty; skipping particle ID lookup and particle stream.' in captured
    assert 'HBT catalog is empty; skipping snapshot.' in captured
    assert 'Particle ID location lookup' not in captured
    assert not (tmp_path / 'split_0.hdf5').exists()
    assert not (tmp_path / 'split_1.hdf5').exists()


def test_filter_snapshot_uses_hbt_reader_for_field_mode(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    config = tmp_path / 'config.yaml'
    outfile = tmp_path / 'split'
    _write_snapshot(snapshot)
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
        ],
        [[11], [12]],
    )
    _write_config(config, hbt_dir, 'field')

    filter_snapshot(str(snapshot), str(outfile), str(config), nsplit=1)

    with h5py.File(f'{outfile}_0.hdf5', 'r') as f:
        assert f['PartType1']['HaloID'][:].tolist() == [0, 0]
        assert 'HaloID_array' not in f['PartType1']
        assert f['PartType1']['particle_index'][:].tolist() == [0, 1]


def test_filter_snapshot_uses_hbt_reader_for_subhalo_mode(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    config = tmp_path / 'config.yaml'
    outfile = tmp_path / 'split'
    _write_snapshot(snapshot)
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
        ],
        [[11], [12]],
    )
    _write_config(config, hbt_dir, 'subhalo')

    filter_snapshot(str(snapshot), str(outfile), str(config), nsplit=1)

    with h5py.File(f'{outfile}_0.hdf5', 'r') as f:
        assert bool(f.attrs[COMPLETE_ATTR]) is True
        assert f.attrs[FILE_TYPE_ATTR] == 'staging_split'
        assert read_simulation_metadata(f)['boxsize'] == 1000.0
        assert f['PartType1']['HaloID'][:].tolist() == [0, 1]
        assert f['PartType1']['HaloID_array'][:].tolist() == [[0, -1], [0, 1]]
        assert f['PartType1']['particle_index'][:].tolist() == [0, 1]


def test_filter_snapshot_writes_dense_ranked_properties(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    config = tmp_path / 'config.yaml'
    outfile = tmp_path / 'split'

    particle_ids = np.asarray([11, 21, 12, 99, 22, 13], dtype=np.int64)
    with h5py.File(snapshot, 'w') as f:
        f.create_group('Header')
        ptype = f.create_group('PartType1')
        ptype.create_dataset('ParticleIDs', data=particle_ids)
        ptype.create_dataset('Masses', data=np.arange(len(particle_ids), dtype=np.float64))
        ptype.create_dataset('Coordinates', data=np.arange(len(particle_ids) * 3, dtype=np.float32).reshape(len(particle_ids), 3))

    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 200, 'HostHaloId': 8, 'Rank': 0, 'NestedParentTrackId': -1},
        ],
        [[11, 12, 13], [21, 22]],
    )
    config.write_text(safe_dump({
        'ptype_names': {'dm': 'PartType1'},
        'prop_aliases': {'pid': 'ParticleIDs', 'mass': 'Masses', 'pos': 'Coordinates'},
        'halo_source': 'hbt',
        'halo_mode': 'subhalo',
        'hbt_subhalo_path': str(hbt_dir),
        'MINIMUM_DM_PER_HALO': 1,
    }))

    filter_snapshot(str(snapshot), str(outfile), str(config), nsplit=2)

    with h5py.File(f'{outfile}_0.hdf5', 'r') as f:
        assert f['PartType1']['ParticleIDs'][:].tolist() == [11, 12, 13]
        assert f['PartType1']['Masses'][:].tolist() == [0.0, 2.0, 5.0]
        assert f['PartType1']['Coordinates'][:].tolist() == [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [15.0, 16.0, 17.0]]
        assert f['PartType1']['HaloID'][:].tolist() == [0, 0, 0]
        assert f['PartType1']['HaloID_array'][:].tolist() == [[0], [0], [0]]
        assert f['PartType1']['particle_index'][:].tolist() == [0, 2, 5]

    with h5py.File(f'{outfile}_1.hdf5', 'r') as f:
        assert f['PartType1']['ParticleIDs'][:].tolist() == [21, 22]
        assert f['PartType1']['Masses'][:].tolist() == [1.0, 4.0]
        assert f['PartType1']['Coordinates'][:].tolist() == [[3.0, 4.0, 5.0], [12.0, 13.0, 14.0]]
        assert f['PartType1']['HaloID'][:].tolist() == [1, 1]
        assert f['PartType1']['HaloID_array'][:].tolist() == [[1], [1]]
        assert f['PartType1']['particle_index'][:].tolist() == [1, 4]


@pytest.mark.parametrize('include_metallicities', [None, False, True])
def test_filter_snapshot_metallicity_columns_follow_include_option(tmp_path, include_metallicities):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    config = tmp_path / 'config.yaml'
    outfile = tmp_path / 'split'

    particle_ids = np.asarray([11, 21, 12, 99, 22, 13], dtype=np.int64)
    metallicity = np.arange(len(particle_ids) * 3, dtype=np.float32).reshape(len(particle_ids), 3)
    with h5py.File(snapshot, 'w') as f:
        f.create_group('Header')
        ptype = f.create_group('PartType0')
        ptype.create_dataset('ParticleIDs', data=particle_ids)
        ptype.create_dataset('Metallicity', data=metallicity)

    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 200, 'HostHaloId': 8, 'Rank': 0, 'NestedParentTrackId': -1},
        ],
        [[11, 12, 13], [21, 22]],
    )
    config_values = {
        'ptype_names': {'gas': 'PartType0'},
        'prop_aliases': {'pid': 'ParticleIDs', 'metallicity': 'Metallicity'},
        'halo_source': 'hbt',
        'halo_mode': 'subhalo',
        'hbt_subhalo_path': str(hbt_dir),
        'MINIMUM_DM_PER_HALO': 1,
    }
    if include_metallicities is not None:
        config_values['include_metallicities'] = include_metallicities
    config.write_text(safe_dump(config_values))

    filter_snapshot(str(snapshot), str(outfile), str(config), nsplit=2)

    full_metallicity = include_metallicities is not False
    expected_rank0 = metallicity[[0, 2, 5]] if full_metallicity else metallicity[[0, 2, 5], 0:1]
    expected_rank1 = metallicity[[1, 4]] if full_metallicity else metallicity[[1, 4], 0:1]

    with h5py.File(f'{outfile}_0.hdf5', 'r') as f:
        data = f['PartType0']['Metallicity'][:]
        assert data.shape == expected_rank0.shape
        assert data.tolist() == expected_rank0.tolist()

    with h5py.File(f'{outfile}_1.hdf5', 'r') as f:
        data = f['PartType0']['Metallicity'][:]
        assert data.shape == expected_rank1.shape
        assert data.tolist() == expected_rank1.tolist()


def test_filter_snapshot_writes_pruned_staged_tree_per_top_halo(tmp_path):
    snapshot = tmp_path / 'snap.hdf5'
    hbt_dir = tmp_path / '050'
    hbt_dir.mkdir()
    config = tmp_path / 'config.yaml'
    outfile = tmp_path / 'split'
    _write_snapshot(snapshot, particle_ids=[11, 12, 21, 99])
    _write_subsnap(
        hbt_dir / 'SubSnap_050.0.hdf5',
        [
            {'TrackId': 100, 'HostHaloId': 7, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 101, 'HostHaloId': 7, 'Rank': 1, 'NestedParentTrackId': 100},
            {'TrackId': 200, 'HostHaloId': 8, 'Rank': 0, 'NestedParentTrackId': -1},
            {'TrackId': 300, 'HostHaloId': 9, 'Rank': 0, 'NestedParentTrackId': -1},
        ],
        [[11], [12], [21], []],
    )
    _write_config(config, hbt_dir, 'subhalo')

    filter_snapshot(str(snapshot), str(outfile), str(config), nsplit=2)

    with h5py.File(f'{outfile}_0.hdf5', 'r') as f:
        tree = f['OctavianHaloTree']
        assert tree['halo_ids'][:].tolist() == [0, 1]
        assert tree['parent_ids'][:].tolist() == [-1, 0]
        assert tree['properties']['TrackId'][:].tolist() == [100, 101]
        assert f['PartType1']['HaloID_array'][:].tolist() == [[0, -1], [0, 1]]

    with h5py.File(f'{outfile}_1.hdf5', 'r') as f:
        tree = f['OctavianHaloTree']
        assert tree['halo_ids'][:].tolist() == [2]
        assert tree['parent_ids'][:].tolist() == [-1]
        assert tree['properties']['TrackId'][:].tolist() == [200]
        assert f['PartType1']['HaloID_array'][:].tolist() == [[2, -1]]

    data_manager = SimpleNamespace(
        snapfile=f'{outfile}_0.hdf5',
        config={
            'halo_source': 'hbt',
            'halo_mode': 'subhalo',
            'hbt_subhalo_path': str(tmp_path / 'missing_hbt_catalog'),
        },
    )
    staged_tree = load_halo_tree(data_manager, mode='subhalo')

    assert staged_tree.halo_ids.tolist() == [0, 1]
    assert staged_tree.parent_ids.tolist() == [-1, 0]
    assert staged_tree.properties['TrackId'].to_list() == [100, 101]


def test_load_halo_tree_requires_staged_tree(tmp_path):
    shard = tmp_path / 'split.hdf5'
    with h5py.File(shard, 'w') as f:
        f.create_group('Header')

    data_manager = SimpleNamespace(
        snapfile=str(shard),
        config={
            'halo_source': 'hbt',
            'halo_mode': 'subhalo',
            'hbt_subhalo_path': str(tmp_path / 'missing_hbt_catalog'),
        },
    )

    with pytest.raises(RuntimeError, match='Staged halo tree missing'):
        load_halo_tree(data_manager, mode='subhalo')
