"""
Ray-based orchestration for Octavian halo workloads.

The dispatcher bridges the HaloScheduler plan with Ray's dynamic resource
allocator: workloads are converted into Ray tasks whose CPU and memory
requirements mirror the scheduler's allocation, letting heterogeneous clusters
place halos on the most appropriate nodes without materialising the full
snapshot in a single process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

try:  # pragma: no cover - optional dependency
  import ray
except Exception:  # pragma: no cover - fallback when Ray missing
  ray = None  # type: ignore

from octavian.backend import pd
import octavian.constants as c
from octavian.ahf import load_catalog
from octavian.core import CoreInputs, CoreResult, Mode, run_core
from octavian.mpi.scheduler import (
  HaloAllocation,
  HaloScheduler,
  HaloWorkload,
  build_workloads_from_catalog,
  collect_top_level_hosts,
)
from octavian.mpi.environment import default_environment_callback

# Ray expects memory limits in bytes.
_GBYTES = 1024 ** 3


@dataclass
class DistributedPlan:
  """Plan and execution metadata returned by the Ray dispatcher."""

  allocations: List[HaloAllocation]
  workloads: List[HaloWorkload]
  catalog_hosts: List[int]


@dataclass
class RayDispatchResult:
  """Aggregate result object produced by :func:`run_with_ray`."""

  core_result: CoreResult
  plan: DistributedPlan


def _initialise_ray(address: Optional[str], init_kwargs: Mapping[str, Any]) -> None:
  if ray is None:  # pragma: no cover - dependency guard
    raise ImportError("Ray is required for distributed execution; install ray to enable this path.")
  if ray.is_initialized():  # pragma: no cover - skip double init
    return
  ray.init(address=address, **dict(init_kwargs))


def _build_core_inputs(
  snapshot_path: Path,
  ahf_particles: Path,
  host_id: int,
  *,
  mode: Mode,
  n_threads: int,
  min_stars: int,
  use_modin: Optional[bool],
) -> CoreInputs:
  return CoreInputs(
    snapshot_path=snapshot_path,
    ahf_particles=ahf_particles,
    host_ids=[int(host_id)],
    mode=mode,
    use_modin=use_modin,
    n_threads=n_threads,
    min_stars=min_stars,
  )


def _byte_limit_from_gb(gb: float) -> Optional[float]:
  if gb <= 0:
    return None
  return max(gb * _GBYTES, 512 * 1024 * 1024)  # at least 512 MiB


def _prepare_remote_options(
  allocation: HaloAllocation,
  extra_options: Mapping[str, Any],
) -> Dict[str, Any]:
  options: Dict[str, Any] = {"num_cpus": max(1, int(allocation.assigned_cores))}
  memory_bytes = _byte_limit_from_gb(allocation.estimated_memory_gb)
  if allocation.fits_memory and memory_bytes is not None:
    options["memory"] = memory_bytes
  options.update(extra_options)
  return options


if ray is not None:  # pragma: no cover - skip when Ray missing

  @ray.remote
  def _run_core_remote(config_kwargs: Dict[str, Any]) -> CoreResult:
    config = _build_core_inputs(**config_kwargs)
    return run_core(config)


def _aggregate_results(results: Sequence[CoreResult]) -> CoreResult:
  host_ids: List[int] = []
  halo_frames: List[pd.DataFrame] = []
  galaxy_frames: List[pd.DataFrame] = []
  metadata: Dict[str, Any] = {"partials": []}

  for result in results:
    host_ids.extend(result.host_ids)
    if not result.halos.empty:
      halo_frames.append(result.halos)
    if not result.galaxies.empty:
      galaxy_frames.append(result.galaxies)
    metadata["partials"].append(result.metadata)

  halos = pd.concat(halo_frames, ignore_index=True) if halo_frames else pd.DataFrame()
  galaxies = pd.concat(galaxy_frames, ignore_index=True) if galaxy_frames else pd.DataFrame()
  metadata["host_ids"] = sorted(set(host_ids))

  return CoreResult(host_ids=host_ids, halos=halos, galaxies=galaxies, metadata=metadata)


EnvironmentCallback = Callable[
  [pd.DataFrame, pd.DataFrame, Mapping[str, Any]],
  Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]],
]


def run_with_ray(
  snapshot_path: Path | str,
  ahf_particles: Path | str,
  host_ids: Optional[Sequence[int]] = None,
  *,
  scheduler: HaloScheduler,
  mode: Mode = "ahf-fast",
  min_stars: int = c.MINIMUM_STARS_PER_GALAXY,
  use_modin: Optional[bool] = None,
  ahf_halos: Optional[Path | str] = None,
  ray_address: Optional[str] = None,
  ray_init_kwargs: Optional[Mapping[str, Any]] = None,
  ray_remote_options: Optional[Mapping[str, Any]] = None,
  environment_callback: Optional[EnvironmentCallback] = default_environment_callback,
) -> RayDispatchResult:
  """
  Execute the Octavian core across the requested hosts using Ray.

  Parameters
  ----------
  snapshot_path / ahf_particles:
      Paths passed through to :class:`CoreInputs` for each halo task.
  host_ids:
      Optional list of top-level host IDs. When omitted, the dispatcher reads
      the ``.AHF_halos`` file to enumerate all top-level hosts automatically.
  scheduler:
      Configured :class:`HaloScheduler` describing per-node resource limits.
  mode:
      Octavian execution mode (``"ahf"`` or ``"ahf-fast"`` today).
  ahf_halos:
      Optional explicit path to the ``.AHF_halos`` file when it does not follow
      the default naming convention.
  environment_callback:
      Optional callable invoked after the per-host results are aggregated,
      giving callers a place to compute environment-dependent properties. It
      receives the concatenated halo and galaxy tables along with an initial
      metadata dictionary and must return updated frames plus metadata.
  """
  snapshot_path = Path(snapshot_path)
  ahf_particles = Path(ahf_particles)
  halos_path = Path(ahf_halos) if ahf_halos is not None else None

  host_list: List[int]
  if host_ids:
    host_list = [int(h) for h in host_ids]
  else:
    host_list = collect_top_level_hosts(ahf_particles, halos_path)
    if not host_list:
      raise ValueError("No top-level hosts found in the AHF catalogue.")

  init_kwargs = dict(ray_init_kwargs or {})
  remote_options = dict(ray_remote_options or {})

  _initialise_ray(ray_address, init_kwargs)

  catalog = load_catalog(str(ahf_particles), halos_path=str(halos_path) if halos_path else None, host_filter=host_list)
  workloads = build_workloads_from_catalog(catalog, host_list)

  plan_batches = scheduler.build_plan(workloads)
  allocations = scheduler.flatten(plan_batches)
  allocation_map = {alloc.halo_id: alloc for alloc in allocations}

  missing = [hid for hid in host_list if hid not in allocation_map]
  if missing:
    raise RuntimeError(f"Scheduler failed to allocate hosts: {missing}")

  tasks = []
  forced_hosts: List[int] = []
  for workload in workloads:
    allocation = allocation_map[workload.halo_id]
    if allocation.forced:
      forced_hosts.append(workload.halo_id)

    config_kwargs = {
      "snapshot_path": snapshot_path,
      "ahf_particles": ahf_particles,
      "host_id": workload.halo_id,
      "mode": mode,
      "n_threads": max(1, allocation.assigned_cores),
      "min_stars": min_stars,
      "use_modin": use_modin,
    }

    options = _prepare_remote_options(allocation, remote_options)
    if ray is None:  # pragma: no cover - guard for mypy
      raise ImportError("Ray is required for distributed execution.")
    task = _run_core_remote.options(**options).remote(config_kwargs)  # type: ignore[attr-defined]
    tasks.append(task)

  if forced_hosts:
    warnings.warn(
      f"The following hosts exceeded the configured limits and were forced into dedicated tasks: {forced_hosts}",
      RuntimeWarning,
      stacklevel=2,
    )

  results = ray.get(tasks) if ray is not None else []  # type: ignore[assignment]
  core_result = _aggregate_results(results)

  if environment_callback is not None:
    halos, galaxies, env_meta = environment_callback(core_result.halos, core_result.galaxies, core_result.metadata)
    core_result = CoreResult(
      host_ids=core_result.host_ids,
      halos=halos,
      galaxies=galaxies,
      metadata={**core_result.metadata, "environment": env_meta},
    )

  plan = DistributedPlan(
    allocations=allocations,
    workloads=workloads,
    catalog_hosts=host_list,
  )

  return RayDispatchResult(core_result=core_result, plan=plan)
