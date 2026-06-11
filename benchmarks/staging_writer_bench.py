from __future__ import annotations

import argparse
import gc
import resource
import shutil
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np


def _maxrss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _counts(total: int, nsplit: int) -> np.ndarray:
    counts = np.full(nsplit, total // nsplit, dtype=np.int64)
    counts[: total % nsplit] += 1
    return counts


def make_rows_by_rank(n_particles: int, n_selected: int, nsplit: int) -> list[np.ndarray]:
    """Deterministic sparse, sorted rows spanning most of the source dataset."""
    counts = _counts(n_selected, nsplit)
    step = max(1, (n_particles - nsplit) // int(counts.max()))
    rows_by_rank = []
    for rank, count in enumerate(counts):
        rows = np.arange(count, dtype=np.uint64) * step + rank
        if rows[-1] >= n_particles:
            raise ValueError('row generator exceeded source particle count')
        rows_by_rank.append(rows.astype(np.uint32, copy=False))
    return rows_by_rank


def prepare_output(outdir: Path, nsplit: int) -> list[h5py.File]:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    return [h5py.File(outdir / f'split_{rank}.hdf5', 'w') for rank in range(nsplit)]


def run_current(dataset, rows_by_rank: list[np.ndarray], outdir: Path) -> dict[str, float]:
    timings = {}
    t = perf_counter()
    data = dataset[:]
    timings['read'] = perf_counter() - t

    files = prepare_output(outdir, len(rows_by_rank))
    try:
        timings['gather'] = 0.0
        timings['write'] = 0.0
        for rank, rows in enumerate(rows_by_rank):
            t = perf_counter()
            values = data[rows]
            timings['gather'] += perf_counter() - t
            t = perf_counter()
            files[rank].create_dataset('values', data=values)
            timings['write'] += perf_counter() - t
            del values
            gc.collect()
    finally:
        for handle in files:
            handle.close()

    return timings


def run_ordered(dataset, rows_by_rank: list[np.ndarray], outdir: Path) -> dict[str, float]:
    timings = {}
    counts = np.asarray([len(rows) for rows in rows_by_rank], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    t = perf_counter()
    ordered_rows = np.concatenate(rows_by_rank)
    timings['order'] = perf_counter() - t

    t = perf_counter()
    data = dataset[:]
    timings['read'] = perf_counter() - t

    t = perf_counter()
    selected = data[ordered_rows]
    timings['gather'] = perf_counter() - t
    del data, ordered_rows
    gc.collect()

    files = prepare_output(outdir, len(rows_by_rank))
    try:
        timings['write'] = 0.0
        for rank in range(len(rows_by_rank)):
            start = offsets[rank]
            end = offsets[rank + 1]
            t = perf_counter()
            files[rank].create_dataset('values', data=selected[start:end])
            timings['write'] += perf_counter() - t
    finally:
        for handle in files:
            handle.close()

    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--dataset', default='PartType1/Coordinates')
    parser.add_argument('--n-selected', type=int, default=401_643_725)
    parser.add_argument('--nsplit', type=int, default=8)
    parser.add_argument('--mode', choices=('current', 'ordered'), required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    total_start = perf_counter()
    with h5py.File(args.snapshot, 'r') as handle:
        dataset = handle[args.dataset]
        n_particles = len(dataset)

        t = perf_counter()
        rows_by_rank = make_rows_by_rank(n_particles, args.n_selected, args.nsplit)
        row_time = perf_counter() - t

        if args.mode == 'current':
            timings = run_current(dataset, rows_by_rank, Path(args.outdir))
        else:
            timings = run_ordered(dataset, rows_by_rank, Path(args.outdir))

    total = perf_counter() - total_start
    print(f'mode={args.mode}')
    print(f'dataset={args.dataset}')
    print(f'n_particles={n_particles}')
    print(f'n_selected={args.n_selected}')
    print(f'nsplit={args.nsplit}')
    print(f'row_build={row_time:.3f}s')
    for key, value in timings.items():
        print(f'{key}={value:.3f}s')
    print(f'total={total:.3f}s')
    print(f'maxrss={_maxrss_gib():.3f}GiB')


if __name__ == '__main__':
    main()
