from __future__ import annotations

import math

import numpy as np
import pytest

from octavian.ahf import AHFCatalog
from octavian.mpi import (
  HaloScheduler,
  HaloWorkload,
  NodeResources,
  build_workloads_from_catalog,
  collect_top_level_hosts,
  detect_cluster_resources,
)


def _workload(halo_id: int, *, dm: int = 0, gas: int = 0, star: int = 0, bh: int = 0) -> HaloWorkload:
  return HaloWorkload(
    halo_id=halo_id,
    particle_counts={
      "dm": dm,
      "gas": gas,
      "star": star,
      "bh": bh,
    },
  )


def test_scheduler_scales_core_counts_with_halo_size() -> None:
  workloads = [
    _workload(1, dm=12_000, gas=4_000, star=2_000),
    _workload(2, dm=30_000, gas=10_000, star=5_000),
    _workload(3, dm=60_000, gas=20_000, star=10_000),
  ]

  scheduler = HaloScheduler(
    total_memory_gb=32.0,
    max_cores=32,
    reserve_memory_gb=4.0,
    min_cores_per_halo=2,
  )

  plan = scheduler.build_plan(workloads)
  allocations = scheduler.flatten(plan)
  allocated = {alloc.halo_id: alloc for alloc in allocations}

  assert allocated[3].assigned_cores > allocated[2].assigned_cores >= allocated[1].assigned_cores
  assert allocated[1].assigned_cores >= 2


def test_scheduler_respects_memory_and_core_limits() -> None:
  workloads = [
    _workload(10, dm=15_000, gas=5_000, star=2_500),
    _workload(11, dm=7_000, gas=2_000, star=1_000),
    _workload(12, dm=4_000, gas=1_500, star=800),
  ]

  scheduler = HaloScheduler(
    total_memory_gb=6.0,
    max_cores=16,
    reserve_memory_gb=1.0,
    min_cores_per_halo=1,
  )

  plan = scheduler.build_plan(workloads)
  assert len(plan) == 2

  limit = scheduler.memory_limit_gb
  if plan[0].allocations and plan[0].allocations[0].forced:
    assert plan[0].total_memory_gb >= limit
  else:
    assert plan[0].total_memory_gb <= limit + 1e-6
  assert plan[0].total_cores <= 16
  assert plan[1].total_memory_gb <= limit + 1e-6
  assert plan[1].total_cores <= 16

  batch_ids = {alloc.halo_id: alloc.batch_id for alloc in scheduler.flatten(plan)}
  assert batch_ids[10] != batch_ids[11]
  assert batch_ids[11] == batch_ids[12]


def test_scheduler_marks_overflowing_halo() -> None:
  workloads = [
    _workload(42, dm=18_000, gas=6_000, star=3_000),
  ]

  scheduler = HaloScheduler(
    total_memory_gb=4.0,
    max_cores=8,
    reserve_memory_gb=1.0,
    min_cores_per_halo=1,
  )

  plan = scheduler.build_plan(workloads)
  assert len(plan) == 1
  allocation = scheduler.flatten(plan)[0]
  assert allocation.forced is True
  assert allocation.fits_memory is False
  assert allocation.needs_reschedule is True
  assert math.isclose(plan[0].total_memory_gb, allocation.estimated_memory_gb, rel_tol=1e-6)


def test_build_workloads_from_catalog_counts_particles() -> None:
  particles = {
    100: {
      "dm": np.arange(5, dtype=np.int64),
      "gas": np.arange(3, dtype=np.int64),
      "star": np.arange(2, dtype=np.int64),
      "bh": np.array([], dtype=np.int64),
    },
    101: {
      "dm": np.arange(4, dtype=np.int64),
      "gas": np.arange(6, dtype=np.int64),
      "star": np.arange(0, dtype=np.int64),
      "bh": np.array([1], dtype=np.int64),
    },
  }
  catalog = AHFCatalog(
    particles=particles,
    star_owner={},
    parent_of={100: 0, 101: 100},
    children_of={0: [100], 100: [101], 101: []},
  )

  workloads = build_workloads_from_catalog(catalog, [100])
  assert len(workloads) == 1
  workload = workloads[0]
  assert workload.halo_id == 100
  assert workload.particle_counts["dm"] == 9
  assert workload.particle_counts["gas"] == 9
  assert workload.particle_counts["star"] == 2
  assert workload.particle_counts["bh"] == 1


def test_collect_top_level_hosts(tmp_path) -> None:
  particles_path = tmp_path / "sample.AHF_particles"
  halos_path = tmp_path / "sample.AHF_halos"
  particles_path.write_text("")  # file presence only
  halos_path.write_text(
    "\n".join(
      [
        "100 0",
        "101 100",
        "200 0",
        "201 200",
      ]
    )
  )

  hosts = collect_top_level_hosts(particles_path)
  assert hosts == [100, 200]

  hosts_filtered = collect_top_level_hosts(particles_path, host_filter=[200])
  assert hosts_filtered == [200]


def test_detect_cluster_resources_returns_local_snapshot() -> None:
  nodes = detect_cluster_resources()
  assert nodes, "Expected at least one node to be detected"
  node = nodes[0]
  assert isinstance(node, NodeResources)
  assert node.total_cores >= 1
  assert node.total_memory_gb > 0


def test_detect_cluster_resources_respects_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('SLURM_NODELIST', 'node[001-002]')
  monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '4(x2)')
  monkeypatch.setenv('SLURM_MEM_PER_NODE', '64000')

  nodes = detect_cluster_resources()
  hosts = {node.host for node in nodes}
  assert hosts == {'node001', 'node002'}
  for node in nodes:
    assert node.total_cores == 4
    assert pytest.approx(node.total_memory_gb, rel=1e-3) == 62.5
