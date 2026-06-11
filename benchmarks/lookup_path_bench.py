from __future__ import annotations

import argparse
import gc
import resource
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from yaml import safe_load

from octavian.halo_reader.ahf import read_ahf_tree
from octavian.halo_reader.halo_utils import PTYPE_ENCODE


def _maxrss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _snapshot_ptypes(config: dict) -> list[tuple[str, str, int]]:
    ptypes = []
    for logical_ptype, ptype_name in config["ptype_names"].items():
        if logical_ptype not in PTYPE_ENCODE:
            continue
        ptypes.append((logical_ptype, ptype_name, PTYPE_ENCODE[logical_ptype]))
    return ptypes


def _scan_max_particle_id(snapshot, config, pid_dataset, chunk_size):
    max_pid = 0
    for _, ptype_name, _ in _snapshot_ptypes(config):
        if ptype_name not in snapshot:
            continue
        dataset = snapshot[ptype_name][pid_dataset]
        for start in range(0, len(dataset), chunk_size):
            pids = dataset[start : start + chunk_size]
            if len(pids):
                max_pid = max(max_pid, int(pids.max()))
    return max_pid


def _build_old_ptype_lookups(snapshot, config, pid_dataset, max_pid, chunk_size):
    sentinel = np.iinfo(np.uint32).max
    lookups = [np.full(max_pid + 1, sentinel, dtype=np.uint32) for _ in range(4)]
    for _, ptype_name, slot in _snapshot_ptypes(config):
        if ptype_name not in snapshot:
            continue
        dataset = snapshot[ptype_name][pid_dataset]
        if len(dataset) >= sentinel:
            raise ValueError(f"{ptype_name} has too many particles for uint32 row lookup")
        for start in range(0, len(dataset), chunk_size):
            end = min(start + chunk_size, len(dataset))
            pids = dataset[start:end]
            lookups[slot][pids] = np.arange(start, end, dtype=np.uint32)
    return lookups


def _build_current_global_lookup(snapshot, config, pid_dataset):
    sentinel = np.iinfo(np.uint32).max
    pids_by_slot = []
    cached_datasets = {}
    max_pid = 0

    for _, ptype_name, slot in _snapshot_ptypes(config):
        if ptype_name not in snapshot or pid_dataset not in snapshot[ptype_name]:
            continue
        dataset = snapshot[ptype_name][pid_dataset]
        if len(dataset) >= sentinel:
            raise ValueError(f"{ptype_name} has too many particles for uint32 row lookup")
        pids = dataset[:]
        cached_datasets.setdefault(ptype_name, {})[pid_dataset] = pids
        if len(pids):
            max_pid = max(max_pid, int(pids.max()))
        pids_by_slot.append((slot, pids))

    row_lookup = np.full(max_pid + 1, sentinel, dtype=np.uint32)
    slot_lookup = np.full(max_pid + 1, -1, dtype=np.int8)
    for slot, pids in pids_by_slot:
        row_lookup[pids] = np.arange(len(pids), dtype=np.uint32)
        slot_lookup[pids] = slot
    return max_pid, row_lookup, slot_lookup, cached_datasets


def _allocate_membership_arrays(snapshot, config, pid_dataset, width):
    arrays = {}
    for _, ptype_name, _ in _snapshot_ptypes(config):
        if ptype_name not in snapshot:
            continue
        t = perf_counter()
        arrays[ptype_name] = np.full(
            (len(snapshot[ptype_name][pid_dataset]), width),
            -1,
            dtype=np.int32,
        )
        print(
            f"allocate_{ptype_name}={perf_counter() - t:.3f}s "
            f"bytes={arrays[ptype_name].nbytes} maxrss={_maxrss_gib():.3f}GiB",
            flush=True,
        )
    return arrays


def _load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return safe_load(handle)


def _tree_info(config: dict) -> tuple[int, int]:
    halos_path = config.get("ahf_halos_path")
    if not halos_path:
        particles_path = Path(config["ahf_particles_path"])
        halos_path = particles_path.with_name(particles_path.name.replace("particles", "halos"))
    t = perf_counter()
    tree, raw_halo_ids = read_ahf_tree(halos_path)
    width = int(tree.depths.max()) + 1
    print(f"tree={perf_counter() - t:.3f}s halos={len(raw_halo_ids)} width={width}", flush=True)
    return len(raw_halo_ids), width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("old-lookup", "current-lookup", "old-prep", "current-prep", "tree"), required=True)
    parser.add_argument("--chunk-size", type=int, default=20_000_000)
    args = parser.parse_args()

    config = _load_config(args.config)
    pid_dataset = config.get("prop_aliases", {}).get("pid", "ParticleIDs")
    total_start = perf_counter()

    if args.mode == "tree":
        _tree_info(config)
        print(f"total={perf_counter() - total_start:.3f}s maxrss={_maxrss_gib():.3f}GiB")
        return

    _, width = _tree_info(config)
    with h5py.File(args.snapshot, "r") as snapshot:
        if args.mode in ("old-lookup", "old-prep"):
            t = perf_counter()
            max_pid = _scan_max_particle_id(snapshot, config, pid_dataset, args.chunk_size)
            print(f"old_scan_max_pid={perf_counter() - t:.3f}s max_pid={max_pid} maxrss={_maxrss_gib():.3f}GiB", flush=True)
            t = perf_counter()
            lookup = _build_old_ptype_lookups(snapshot, config, pid_dataset, max_pid, args.chunk_size)
            print(f"old_build_ptype_lookups={perf_counter() - t:.3f}s maxrss={_maxrss_gib():.3f}GiB", flush=True)
            if args.mode == "old-prep":
                _allocate_membership_arrays(snapshot, config, pid_dataset, width)
            del lookup
        else:
            t = perf_counter()
            max_pid, row_lookup, slot_lookup, cached_datasets = _build_current_global_lookup(snapshot, config, pid_dataset)
            print(f"current_build_global_lookup={perf_counter() - t:.3f}s max_pid={max_pid} maxrss={_maxrss_gib():.3f}GiB", flush=True)
            if args.mode == "current-prep":
                _allocate_membership_arrays(snapshot, config, pid_dataset, width)
            del row_lookup, slot_lookup, cached_datasets

    gc.collect()
    print(f"total={perf_counter() - total_start:.3f}s maxrss={_maxrss_gib():.3f}GiB", flush=True)


if __name__ == "__main__":
    main()
