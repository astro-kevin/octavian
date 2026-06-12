"""

AHF Halo Finder integration to Octavian; assumes we are analysing a snapshot which AHF has already run on.

Source paper: https://arxiv.org/pdf/0904.3662
Github: https://github.com/weiguangcui/AHF
Code is based on the architecture outlined therein.

"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_manager import DataManager

import ctypes # for accelerated reading
import gzip
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd

from octavian.halo_reader.halo_utils import (
    HaloBuildResult,
    HaloMembership,
    HaloReader,
    HaloTree,
    PTYPE_ENCODE,
    membership_array_exclusive_ids,
)

# AHF assigns ptype codes, we change these to ptype names for Octavian compatibility
_PTYPE_MAP = {0: 0, # gas
              1: 1, # dm
              4: 2, # star
              5: 3} # bh

# AHF compresses its output (which is what gzip is for)
def _open_catalog(path: Path):
    """
    Open an AHF catalogue file.
    """
    if path.suffix == '.gz':
        return gzip.open(path, 'rt')
    return open(path, 'r')

def read_ahf_halos(path: Path) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Parse an AHF file.
    """
    # headers are #-commented
    with _open_catalog(path) as f:
        header_line = f.readline().strip().lstrip('#').split() # strip the #

    clean_headers = [h.split('(')[0] for h in header_line] # AHF adds numbering so strip this too

    # whitespace-delimited rows
    # NOTE: only pandas can handle this (at present March 2026)
    dtypes = {key: np.int64 for key in ('ID', 'hostHalo') if key in clean_headers}
    df = pd.read_csv(path, comment='#', sep=r'\s+', names=clean_headers, dtype=dtypes)

    return df # convert to polars

# previously defined as two functions: now convolved
def read_ahf_particles(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse AHF _particles file into flat numpy arrays (only Octavian particles).
    The format of these is slightly fiddly because the format alternates.
    Data structure of AHF catalog:

    Number of Particles | Halo ID 
    Particle ID | Particle Type
    Particle ID | Particle Type
    etc. with alternating lengths under the halo header, making reading the output finicky.
    """
    # AHF uses gzip
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        raw_lines = f.readlines()

    n = len(raw_lines) # start with all lines
    out_hids = np.empty(n, dtype=np.int64)
    out_pids = np.empty(n, dtype=np.int64)
    out_ptypes = np.empty(n, dtype=np.int8) # small numbers, use smallest footprint data structure

    # this is a loop-based state machine
    idx = 0
    remaining = 0
    current_hid = 0

    for line in raw_lines:
        parts = line.split()

        # this avoids MPI separators in the file (potentially redundant)
        if len(parts) != 2:
            continue

        # if we are at a header (nparticles | hid)
        if remaining == 0:
            try:
                remaining = int(parts[0])
                current_hid = int(parts[1])
            except ValueError:
                remaining = 0
            continue

        # ptype
        ptype_code = int(parts[1])
        remaining -= 1 # then register this as completed, move on

        # the alternating structure makes this finicky because if it were columnar we could mask these
        if ptype_code not in _PTYPE_MAP:
            continue # dict lookup is famously fast: but unavoidable overhead

        # write to numpy
        out_hids[idx] = current_hid
        out_pids[idx] = int(parts[0])
        out_ptypes[idx] = _PTYPE_MAP[ptype_code]
        idx += 1

    return out_hids[:idx].copy(), out_pids[:idx].copy(), out_ptypes[:idx].copy()

def read_ahf_particles_c(filepath, n_estimate=None):
    """
    C-accelerated AHF particle parser.
    Requires ahf_parser.so to be compiled in the same directory.
    """
    import ctypes
    
    so_path = Path(__file__).parent / 'ahf_parser.so'
    if not so_path.exists():
        raise FileNotFoundError(f'Compiled parser not found at {so_path}. Compile with: gcc -O2 -shared -fPIC -o ahf_parser.so ahf_parser.c')
    
    lib = ctypes.CDLL(str(so_path))
    lib.parse_ahf_particles.restype = ctypes.c_long
    
    filepath = Path(filepath)
    if n_estimate is None:
        n_estimate = filepath.stat().st_size // 8
    
    out_hids = np.empty(n_estimate, dtype=np.int64)
    out_pids = np.empty(n_estimate, dtype=np.int64)
    out_ptypes = np.empty(n_estimate, dtype=np.int8)
    valid_ptypes = np.array([0, 1, 4, 5], dtype=np.int32)
    
    n = lib.parse_ahf_particles(
        str(filepath).encode(),
        out_hids.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        out_pids.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        out_ptypes.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
        valid_ptypes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(valid_ptypes)
    )
    
    if n < 0:
        raise IOError(f'Failed to open {filepath}')
    
    # C parser returns raw AHF ptype codes (0, 1, 4, 5)
    # map to Octavian encoding (0, 1, 2, 3)
    ptype_remap = np.full(6, -1, dtype=np.int8)  # max AHF code is 5
    for ahf_code, octavian_code in _PTYPE_MAP.items():
        ptype_remap[ahf_code] = octavian_code
    
    result_ptypes = ptype_remap[out_ptypes[:n]]
    
    return out_hids[:n].copy(), out_pids[:n].copy(), result_ptypes.copy()

def _remap_ahf_ids(halo_ids, parent_ids, member_hids):
    unique_raw = np.unique(halo_ids)
    halo_ids = np.searchsorted(unique_raw, halo_ids)
    valid_parents = parent_ids != -1
    parent_pos = np.searchsorted(unique_raw, parent_ids[valid_parents])
    parent_matched = unique_raw[np.clip(parent_pos, 0, len(unique_raw) - 1)] == parent_ids[valid_parents]
    new_parent_ids = np.full_like(parent_ids, -1)
    new_parent_ids[valid_parents] = np.where(parent_matched, parent_pos, -1)
    member_pos = np.searchsorted(unique_raw, member_hids)
    matched = unique_raw[np.clip(member_pos, 0, len(unique_raw) - 1)] == member_hids
    return halo_ids, new_parent_ids, np.where(matched, member_pos, -1)

def read_ahf_membership(particles_path, halos_path=None):
    particles_path = Path(particles_path)
    if halos_path is None:
        halos_path = particles_path.with_name(particles_path.name.replace('particles', 'halos'))
    else:
        halos_path = Path(halos_path)

    properties = read_ahf_halos(halos_path)
    member_hids, member_pids, member_ptypes = read_ahf_particles_c(particles_path)
    halo_ids = properties['ID'].to_numpy().astype(np.int64)
    parent_ids = properties['hostHalo'].to_numpy().astype(np.int64)
    parent_ids[parent_ids == 0] = -1
    halo_ids, parent_ids, member_hids = _remap_ahf_ids(halo_ids, parent_ids, member_hids)
    return HaloTree(halo_ids, parent_ids, properties), member_hids, member_pids, member_ptypes

def _membership_array_exclusive_ids(halo_id_array) -> np.ndarray:
    return membership_array_exclusive_ids(halo_id_array)

def _build_halo_ancestor_arrays(tree: HaloTree, width: int) -> np.ndarray:
    arrays = np.full((len(tree._id_to_idx), width), -1, dtype=np.int32)
    for halo_id in tree.halo_ids:
        current = int(halo_id)
        while current != -1:
            row = tree._id_to_idx[current]
            if row == -1:
                break
            arrays[int(halo_id), int(tree.depths[row])] = current
            current = int(tree.parent_ids[row])
    return arrays

def read_ahf_tree(halos_path):
    properties = read_ahf_halos(Path(halos_path))
    raw_halo_ids = properties['ID'].to_numpy(dtype=np.int64)
    parent_ids = properties['hostHalo'].to_numpy(dtype=np.int64)
    parent_ids[parent_ids == 0] = -1
    halo_ids, parent_ids, _ = _remap_ahf_ids(raw_halo_ids, parent_ids, np.empty(0, dtype=np.int64))
    return HaloTree(halo_ids, parent_ids, properties), np.sort(raw_halo_ids)

def _load_ahf_parser():
    so_path = Path(__file__).parent / 'ahf_parser.so'
    if not so_path.exists():
        raise FileNotFoundError(f'Compiled parser not found at {so_path}. Compile with: gcc -O2 -shared -fPIC -o ahf_parser.so ahf_parser.c')
    return ctypes.CDLL(str(so_path))

def _pid_lookup_chunk_size(config: dict) -> int:
    chunk_size = int(config.get('pid_lookup_chunk_size', 20_000_000))
    if chunk_size < 1:
        raise ValueError('pid_lookup_chunk_size must be positive')
    return chunk_size


def _build_particle_location_lookup(snapshot, config, pid_dataset):
    max_row = np.iinfo(np.uint32).max - 1
    entries = []
    max_pid = 0
    chunk_size = _pid_lookup_chunk_size(config)

    for ptype, ptype_name in config['ptype_names'].items():
        if ptype not in PTYPE_ENCODE:
            continue
        if ptype_name not in snapshot or pid_dataset not in snapshot[ptype_name]:
            continue
        slot = PTYPE_ENCODE[ptype]
        dataset = snapshot[ptype_name][pid_dataset]
        n_particles = len(dataset)
        if n_particles > max_row:
            raise ValueError(f'{ptype_name} has too many particles for uint32 row lookup')
        entries.append((slot, dataset))
        for start in range(0, n_particles, chunk_size):
            pids = dataset[start:start + chunk_size]
            if len(pids):
                max_pid = max(max_pid, int(pids.max()))

    row_lookup = np.zeros(max_pid + 1, dtype=np.uint32)
    slot_lookup = np.zeros(max_pid + 1, dtype=np.int8)
    for slot, dataset in entries:
        n_particles = len(dataset)
        for start in range(0, n_particles, chunk_size):
            end = min(start + chunk_size, n_particles)
            pids = dataset[start:end]
            row_lookup[pids] = np.arange(start + 1, end + 1, dtype=np.uint32)
            slot_lookup[pids] = slot + 1

    return max_pid, row_lookup, slot_lookup, {}

def _allocate_membership_arrays(snapshot, config, pid_dataset):
    membership_arrays = {}
    by_slot = [np.empty(0, dtype=np.int32) for _ in range(4)]
    for ptype, ptype_name in config['ptype_names'].items():
        if ptype_name not in snapshot:
            continue
        membership = np.zeros(len(snapshot[ptype_name][pid_dataset]), dtype=np.int32)
        membership_arrays[ptype_name] = membership
        by_slot[PTYPE_ENCODE[ptype]] = membership
    return membership_arrays, by_slot

def build_ahf_snapshot_membership_arrays(snapshot, config, particles_path, halos_path=None):
    particles_path = Path(particles_path)
    if halos_path is None:
        halos_path = particles_path.with_name(particles_path.name.replace('particles', 'halos'))
    t = perf_counter()
    tree, raw_halo_ids = read_ahf_tree(halos_path)
    print(f'  AHF halo tree: {perf_counter() - t:.1f}s', flush=True)
    pid_dataset = config.get('prop_aliases', {}).get('pid', 'ParticleIDs')
    width = int(tree.depths.max()) + 1 if len(tree.depths) else 1
    ancestor_arrays = _build_halo_ancestor_arrays(tree, width)

    if len(tree.halo_ids) == 0:
        membership_arrays, _ = _allocate_membership_arrays(snapshot, config, pid_dataset)
        counts = np.zeros(8, dtype=np.uint64)
        print('  AHF catalog empty; skipping particle ID lookup and particle stream.', flush=True)
        return HaloBuildResult(tree, membership_arrays, counts, {}, ancestor_arrays=ancestor_arrays)

    t = perf_counter()
    max_pid, row_lookup, slot_lookup, cached_datasets = _build_particle_location_lookup(snapshot, config, pid_dataset)
    print(f'  Particle ID lookups: {perf_counter() - t:.1f}s', flush=True)
    t = perf_counter()
    membership_arrays, arrays_by_slot = _allocate_membership_arrays(snapshot, config, pid_dataset)
    print(f'  Scalar membership allocation: {perf_counter() - t:.1f}s', flush=True)

    depth_by_halo = np.full(len(tree._id_to_idx), -1, dtype=np.int32)
    if len(tree.halo_ids):
        depth_by_halo[tree.halo_ids] = tree.depths.astype(np.int32, copy=False)

    lib = _load_ahf_parser()
    lib.fill_ahf_scalar_membership_arrays.restype = ctypes.c_long
    lib.fill_ahf_scalar_membership_arrays.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    counts = np.zeros(8, dtype=np.uint64)
    t = perf_counter()
    written = lib.fill_ahf_scalar_membership_arrays(
        str(particles_path).encode(),
        raw_halo_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ancestor_arrays.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        len(raw_halo_ids),
        row_lookup.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        slot_lookup.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        max_pid,
        width,
        depth_by_halo.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        arrays_by_slot[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        arrays_by_slot[1].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        arrays_by_slot[2].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        arrays_by_slot[3].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    if written < 0:
        raise IOError(f'Failed to open {particles_path}')
    print(f'  AHF particle stream: {perf_counter() - t:.1f}s', flush=True)

    return HaloBuildResult(tree, membership_arrays, counts, cached_datasets, ancestor_arrays=ancestor_arrays)


def _paths_from_config(config: dict) -> tuple[Path, Path | None]:
    particles_path = Path(config['ahf_particles_path'])
    halos_path = config.get('ahf_halos_path') or None
    if halos_path is not None:
        halos_path = Path(halos_path)
    return particles_path, halos_path


def metadata_schema() -> dict[str, object]:
    return {
        'original_id_column': 'ID',
        'halo_id_column': 'AHF_haloID',
        'parent_id_column': 'AHF_parent_haloID',
        'top_id_column': 'AHF_top_haloID',
        'depth_column': 'AHF_depth',
        'host_index_column': '_ahf_host_halo_index',
        'ancestor_column': 'AHF_ancestor_haloIDs',
        'halo_index_columns': {
            'halos': ['caesar_parent_halo_index', 'caesar_top_halo_index'],
            'galaxies': ['caesar_parent_halo_index', 'caesar_top_halo_index', '_ahf_host_halo_index'],
        },
    }


def read_tree(config: dict) -> HaloTree:
    particles_path, halos_path = _paths_from_config(config)
    if halos_path is None:
        halos_path = particles_path.with_name(particles_path.name.replace('particles', 'halos'))
    tree, _ = read_ahf_tree(halos_path)
    return tree


def build_snapshot_membership_arrays(snapshot, config: dict):
    particles_path, halos_path = _paths_from_config(config)
    return build_ahf_snapshot_membership_arrays(snapshot, config, particles_path, halos_path)


def load(data_manager, mode='field'):
    particles_path, halos_path = _paths_from_config(data_manager.config)
    return load_ahf(data_manager, particles_path, halos_path, mode=mode)

def load_ahf(data_manager, particles_path, halos_path=None, mode='field'):

    data_manager.config['halo_source'] = 'ahf'
    data_manager.config['halo_mode'] = mode

    particles_path = Path(particles_path)
    
    if halos_path is None:
        halos_path = particles_path.with_name(
            particles_path.name.replace('particles', 'halos')
        )
    else:
        halos_path = Path(halos_path)

    print(f"Loading AHF data...")
    t0 = perf_counter()

    if mode == 'subhalo':
        with h5py.File(data_manager.snapfile, 'r') as f:
            build_result = build_ahf_snapshot_membership_arrays(f, data_manager.config, particles_path, halos_path)
            tree, membership_arrays, counts = build_result
        ancestor_arrays = build_result.ancestor_arrays
        data_manager.halo_tree = tree
        for ptype in data_manager.config['ptypes']:
            ptype_name = data_manager.get_ptype_name(ptype)
            membership = membership_arrays[ptype_name]
            data_manager.halo_id_arrays[ptype] = (
                ancestor_arrays[np.maximum(membership.astype(np.int64, copy=False) - 1, 0)]
                if ancestor_arrays is not None else membership
            )
            missing = membership == 0
            if ancestor_arrays is not None and np.any(missing):
                data_manager.halo_id_arrays[ptype][missing] = -1
            data_manager.data[ptype]['HaloID'] = pd.Series(_membership_array_exclusive_ids(membership), dtype='category')
        print(f"  Membership array assign: {perf_counter() - t0:.3f}s")
        print(f"  AHF memberships written: {int(counts[:4].sum())}, conflicts resolved: {int(counts[7])}")
    else:
        t1 = perf_counter()
        tree, member_hids, member_pids, member_ptypes = read_ahf_membership(particles_path, halos_path)
        print(f"Finished in {(perf_counter() - t1):.3f} seconds.")
    
        print(f"Extracting halo structure and membership...")
        t0 = perf_counter()
        print(f"  HaloTree: {perf_counter() - t0:.3f}s")
        data_manager.halo_tree = tree
        reader = HaloReader(data_manager)
        t1 = perf_counter()
        membership = HaloMembership(tree, member_hids, member_pids, member_ptypes, exclusive=False)
        data_manager.halo_membership = membership
        print(f"  HaloMembership: {perf_counter() - t1:.3f}s")

        t1 = perf_counter()
        reader.assign(membership, mode)
        print(f"  Assign: {perf_counter() - t1:.3f}s")
    print(f"Finished in {(perf_counter() - t0):.3f} seconds.")
