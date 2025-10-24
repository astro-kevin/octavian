"""
MPI-oriented scheduling helpers for Octavian.

The top-level objects exported here let callers reason about halo sizes,
estimate resource requirements, and build execution plans that respect
per-node memory and CPU core limits.
"""

from __future__ import annotations

from .scheduler import (
  HaloAllocation,
  HaloBatch,
  HaloMemoryModel,
  HaloScheduler,
  HaloWorkload,
  NodeResources,
  build_workloads_from_catalog,
  collect_top_level_hosts,
  detect_node_resources,
  detect_cluster_resources,
)
from .environment import default_environment_callback
from .ray_dispatcher import (
  DistributedPlan,
  RayDispatchResult,
  run_with_ray,
)

__all__ = [
  "HaloAllocation",
  "HaloBatch",
  "HaloMemoryModel",
  "HaloScheduler",
  "HaloWorkload",
  "NodeResources",
  "build_workloads_from_catalog",
  "collect_top_level_hosts",
  "detect_node_resources",
  "detect_cluster_resources",
  "default_environment_callback",
  "DistributedPlan",
  "RayDispatchResult",
  "run_with_ray",
]
