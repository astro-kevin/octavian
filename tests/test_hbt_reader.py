import h5py
import numpy as np

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
