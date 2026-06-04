from types import SimpleNamespace

import h5py
import pytest
import numpy as np
from yaml import safe_dump

from octavian.halo_filter import filter_snapshot
from octavian.halo_reader import load_halo_tree
from octavian.halo_reader.hbt import (
    build_parent_ids,
    gather_subsnap_files,
    read_particles,
    read_subhalos,
)


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
        f.create_group('Header')
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
        assert f['PartType1']['HaloID'][:].tolist() == [0, 1]
        assert f['PartType1']['HaloID_array'][:].tolist() == [[0, -1], [0, 1]]
        assert f['PartType1']['particle_index'][:].tolist() == [0, 1]


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
