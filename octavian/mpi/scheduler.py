"""
Halo scheduling utilities for Octavian's MPI layer.

The scheduler estimates how much memory an AHF halo will require once loaded
into the ``DataManager`` and chooses a CPU core allocation that scales with the
halo size.  It then greedily packs halos into sequential batches that respect
per-node memory and core limits, ensuring the largest halos get priority while
surfacing any cases that exceed the configured capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from octavian.ahf import read_ahf_hierarchy

try:  # pragma: no cover - optional dependency
  import psutil
except Exception:  # pragma: no cover
  psutil = None

try:  # pragma: no cover - optional dependency
  import ray
except Exception:  # pragma: no cover
  ray = None  # type: ignore

if TYPE_CHECKING:
  from octavian.ahf import AHFCatalog

# Guard against floating-point edge cases when comparing memory usage.
_EPS = 1e-9


@dataclass(frozen=True)
class HaloWorkload:
  """
  Minimal workload descriptor for a halo.

  Parameters
  ----------
  halo_id:
      Unique identifier of the halo or host to schedule.
  particle_counts:
      Mapping of particle type → particle count.  The scheduler uses these
      counts to estimate memory consumption and relative halo size.  Any
      missing particle type defaults to zero.
  bytes_hint:
      Optional raw byte count hint when callers have a more accurate estimate
      of the memory footprint.  When supplied, the scheduler will prefer the
      hint if it is larger than the model-based estimate.
  metadata:
      Arbitrary dictionary passed through to help downstream consumers keep
      contextual information (e.g. parent host ID, depth, etc.).
  """

  halo_id: int
  particle_counts: Mapping[str, int]
  bytes_hint: Optional[int] = None
  metadata: Mapping[str, object] = field(default_factory=dict)

  def __post_init__(self) -> None:
    normalised_counts: Dict[str, int] = {}
    for key, value in self.particle_counts.items():
      normalised_counts[str(key)] = int(value)
    object.__setattr__(self, "particle_counts", normalised_counts)
    if self.bytes_hint is not None:
      object.__setattr__(self, "bytes_hint", int(self.bytes_hint))
    if not isinstance(self.metadata, Mapping):
      raise TypeError("metadata must be a mapping")

  @property
  def total_particles(self) -> int:
    return sum(self.particle_counts.values())

  @property
  def work_units(self) -> float:
    total = max(1, self.total_particles)
    return float(total) * math.log2(max(total, 2))


class HaloMemoryModel:
  """
  Estimate a halo's memory footprint from its particle counts.

  The defaults are deliberately conservative so the scheduler errs on the side
  of allocating a slightly larger batch.  Callers can override the per-particle
  coefficients or the per-halo overhead to suit their environment.

  ``bytes_per_particle`` defaults to ~180 kB, which matches the peak RSS we saw
  when running ``test.py``/``test-FAST.py`` (~0.5 GB for ~2.7k particles) with a
  20% buffer and a small 0.05 GB per-halo overhead.
  """

  def __init__(
    self,
    *,
    bytes_per_particle: float = 180_000.0,
    per_halo_overhead_gb: float = 0.05,
    buffer_fraction: float = 0.20,
  ) -> None:
    self._bytes_per_particle = max(0.0, float(bytes_per_particle))
    self._per_halo_overhead_gb = max(0.0, float(per_halo_overhead_gb))
    self._buffer_fraction = max(0.0, float(buffer_fraction))

  def estimate_gb(self, workload: HaloWorkload) -> float:
    total_bytes = self._bytes_per_particle * max(0, workload.total_particles)
    if workload.bytes_hint is not None:
      total_bytes = max(total_bytes, float(workload.bytes_hint))
    total_bytes *= 1.0 + self._buffer_fraction
    total_gb = total_bytes / 1e9
    return self._per_halo_overhead_gb + total_gb


@dataclass(frozen=True)
class HaloAllocation:
  """Allocation decision for a single halo inside a batch."""

  halo_id: int
  batch_id: int
  estimated_memory_gb: float
  assigned_cores: int
  total_particles: int
  fits_memory: bool
  fits_cores: bool
  forced: bool = False

  @property
  def needs_reschedule(self) -> bool:
    """Return True when the allocation exceeds the configured limits."""
    return not (self.fits_memory and self.fits_cores)


@dataclass
class HaloBatch:
  """Collection of halo allocations that share a node or MPI rank."""

  batch_id: int
  memory_limit_gb: float
  core_limit: int
  allocations: List[HaloAllocation] = field(default_factory=list)
  total_memory_gb: float = 0.0
  total_cores: int = 0

  def try_add(
    self,
    workload: HaloWorkload,
    *,
    estimated_memory_gb: float,
    assigned_cores: int,
  ) -> Optional[HaloAllocation]:
    """Attempt to add ``workload`` into this batch."""
    projected_memory = self.total_memory_gb + estimated_memory_gb
    projected_cores = self.total_cores + assigned_cores

    if self.memory_limit_gb > 0:
      fits_memory = (projected_memory - self.memory_limit_gb) <= _EPS
    else:
      fits_memory = True

    if self.core_limit > 0:
      fits_cores = projected_cores <= self.core_limit
    else:
      fits_cores = True

    forced = False
    if not (fits_memory and fits_cores):
      if not self.allocations:
        # Always accept the first halo so large hosts can still be scheduled,
        # but mark them for follow-up so the caller can react.
        forced = True
      else:
        return None

    allocation = HaloAllocation(
      halo_id=workload.halo_id,
      batch_id=self.batch_id,
      estimated_memory_gb=estimated_memory_gb,
      assigned_cores=assigned_cores,
      total_particles=workload.total_particles,
      fits_memory=fits_memory,
      fits_cores=fits_cores,
      forced=forced,
    )

    self.allocations.append(allocation)
    self.total_memory_gb = projected_memory
    self.total_cores = projected_cores
    return allocation


@dataclass(frozen=True)
class NodeResources:
  node_id: str
  host: str
  total_cores: float
  available_cores: float
  total_memory_gb: float
  available_memory_gb: float


def _detect_local_resources() -> Tuple[int, float]:
  cores = os.cpu_count() or 1

  memory_bytes: Optional[int] = None
  if psutil is not None:
    try:
      memory_bytes = int(psutil.virtual_memory().total)
    except Exception:
      memory_bytes = None
  if memory_bytes is None and hasattr(os, "sysconf"):
    try:
      pages = os.sysconf("SC_PHYS_PAGES")
      page_size = os.sysconf("SC_PAGE_SIZE")
      memory_bytes = int(pages * page_size)
    except Exception:
      memory_bytes = None
  if memory_bytes is None:
    raise RuntimeError("Unable to detect system memory; please specify total_memory_gb explicitly.")

  return cores, memory_bytes / 1e9


def _split_nodelist_entries(nodelist: str) -> List[str]:
  parts: List[str] = []
  depth = 0
  current = []
  for char in nodelist:
    if char == ',' and depth == 0:
      if current:
        parts.append(''.join(current))
        current = []
      continue
    if char == '[':
      depth += 1
    elif char == ']':
      depth = max(depth - 1, 0)
    current.append(char)
  if current:
    parts.append(''.join(current))
  return [part.strip() for part in parts if part.strip()]


def _expand_bracket_expression(expr: str) -> List[str]:
  if not expr:
    return ['']
  start = expr.find('[')
  if start == -1:
    return [expr]
  end = expr.find(']', start)
  if end == -1:
    return [expr]
  prefix = expr[:start]
  suffix = expr[end + 1:]
  body = expr[start + 1:end]

  expansions: List[str] = []
  for token in body.split(','):
    token = token.strip()
    if not token:
      continue
    sequence: List[str]
    if '-' in token:
      start_token, end_token = token.split('-', 1)
      width = len(start_token)
      try:
        start_int = int(start_token)
        end_int = int(end_token)
      except ValueError:
        sequence = [token]
      else:
        step = 1 if end_int >= start_int else -1
        rng = range(start_int, end_int + step, step)
        sequence = [f"{value:0{width}d}" for value in rng]
    else:
      sequence = [token]

    tails = _expand_bracket_expression(suffix)
    for value in sequence:
      if tails:
        for tail in tails:
          expansions.append(prefix + value + tail)
      else:
        expansions.append(prefix + value)

  return expansions or [expr]


def _expand_slurm_nodelist(nodelist: str) -> List[str]:
  try:
    output = subprocess.check_output(['scontrol', 'show', 'hostnames', nodelist], text=True)
  except Exception:
    output = ''
  hosts = [line.strip() for line in output.splitlines() if line.strip()]
  if hosts:
    return hosts

  entries = _split_nodelist_entries(nodelist)
  hosts = []
  for entry in entries:
    hosts.extend(_expand_bracket_expression(entry))
  if not hosts and nodelist:
    hosts = [nodelist]
  return hosts


_CPU_SPEC_PATTERN = re.compile(r'^(?P<count>\d+)(?:\(x(?P<rep>\d+)\))?$')

_MEM_SPEC_PATTERN = re.compile(r'^(?P<value>\d+(?:\.\d+)?)(?P<unit>[KMGTP]?)$', re.IGNORECASE)

_MEM_UNIT_TO_GB = {
  '': 1.0 / 1024.0,  # default is MB
  'K': 1.0 / (1024.0 * 1024.0),
  'M': 1.0 / 1024.0,
  'G': 1.0,
  'T': 1024.0,
  'P': 1024.0 * 1024.0,
}


def _parse_slurm_cpu_list(spec: str, expected: int) -> List[int]:
  values: List[int] = []
  for chunk in spec.split(','):
    chunk = chunk.strip()
    if not chunk:
      continue
    match = _CPU_SPEC_PATTERN.match(chunk)
    if not match:
      continue
    count = int(match.group('count'))
    reps = int(match.group('rep') or 1)
    values.extend([count] * reps)

  if not values:
    return []
  if len(values) == 1 and expected > 1:
    values = values * expected
  if len(values) < expected:
    last = values[-1]
    values.extend([last] * (expected - len(values)))
  elif len(values) > expected:
    values = values[:expected]
  return values


def _parse_memory_to_gb(value: str) -> float:
  match = _MEM_SPEC_PATTERN.match(value.strip())
  if not match:
    return 0.0
  number = float(match.group('value'))
  unit = match.group('unit').upper()
  factor = _MEM_UNIT_TO_GB.get(unit, 0.0)
  return number * factor


def _collect_slurm_resources() -> List[NodeResources]:
  nodelist = os.environ.get('SLURM_NODELIST')
  if not nodelist:
    return []

  hosts = _expand_slurm_nodelist(nodelist)
  if not hosts:
    return []

  cpu_spec = os.environ.get('SLURM_JOB_CPUS_PER_NODE', '')
  cpu_counts = _parse_slurm_cpu_list(cpu_spec, len(hosts)) if cpu_spec else []
  if not cpu_counts:
    default_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', '1') or 1)
    cpu_counts = [default_cpus] * len(hosts)

  mem_list: List[float]
  mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
  mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
  if mem_per_node:
    mem_gb = _parse_memory_to_gb(mem_per_node)
    if mem_gb > 0.0:
      mem_list = [mem_gb] * len(hosts)
    else:
      mem_list = []
  elif mem_per_cpu:
    per_cpu_gb = _parse_memory_to_gb(mem_per_cpu)
    if per_cpu_gb > 0.0:
      mem_list = [per_cpu_gb * cpus for cpus in cpu_counts]
    else:
      mem_list = []
  else:
    _, local_mem = _detect_local_resources()
    mem_list = [local_mem] * len(hosts)

  if not mem_list:
    _, local_mem = _detect_local_resources()
    mem_list = [local_mem] * len(hosts)

  resources: List[NodeResources] = []
  for host, cores, mem_gb in zip(hosts, cpu_counts, mem_list):
    resources.append(
      NodeResources(
        node_id=host,
        host=host,
        total_cores=float(cores),
        available_cores=float(cores),
        total_memory_gb=mem_gb,
        available_memory_gb=mem_gb,
      )
    )

  return resources


class HaloScheduler:
  """
  Build resource-aware execution plans for halo processing.

  Parameters
  ----------
  total_memory_gb:
      Total memory available to each batch (typically the node capacity).
  max_cores:
      Maximum number of CPU cores usable concurrently.
  reserve_memory_gb:
      Memory to keep in reserve for MPI buffers, filesystem caches, etc.
      Reducing this number allows denser packing at the cost of less headroom.
  min_cores_per_halo:
      Floor applied to every halo allocation.
  max_cores_per_halo:
      Optional ceiling for per-halo core assignments.
  memory_model:
      Custom :class:`HaloMemoryModel` to use for size estimation.  If omitted,
      the default conservative model is applied.
  work_units_per_core:
      Number of ``n log n`` work units a single CPU core can handle. The
      scheduler divides each halo's work units by this value and rounds up to
      pick the CPU request. Defaults to ``25_000`` (roughly 2–3k particles per
      core based on profiling).
  work_unit_baseline:
      (removed) – CPU scaling now relies on ``work_units_per_core``.
  """

def detect_node_resources() -> Tuple[int, float]:
  """
  Inspect the current node to determine the available cores and memory.

  Returns
  -------
  (cores, memory_gb)
      Tuple containing logical CPUs and total physical memory in gigabytes.
  """
  cores, memory_gb = _detect_local_resources()
  return cores, memory_gb


def detect_cluster_resources(ray_address: Optional[str] = None) -> List[NodeResources]:
  """
  Inspect the resources available to the current job.

  Priority order:
    1. If SLURM environment variables are present, return the nodes allocated to
       the job with their CPU/memory requests.
    2. Otherwise, if Ray is initialised (locally or via ``ray_address``), use the
       cluster view that Ray exposes, filtered by the SLURM allocation when both
       are available.
    3. Fallback to the local node snapshot.

  Memory values are expressed in gigabytes.
  """
  slurm_nodes = _collect_slurm_resources()
  slurm_hosts = {node.host for node in slurm_nodes} if slurm_nodes else None

  ray_nodes: List[NodeResources] = []
  if ray is not None:
    if not ray.is_initialized() and ray_address is not None:
      ray.init(address=ray_address)  # type: ignore[call-arg]

    if ray.is_initialized():
      for node in ray.nodes():
        if not node.get("Alive", False):
          continue
        host = str(node.get("NodeManagerAddress", ""))
        if slurm_hosts and host not in slurm_hosts:
          continue
        resources = node.get("Resources", {})
        available = node.get("AvailableResources", resources)

        total_cpu = float(resources.get("CPU", 0.0))
        available_cpu = float(available.get("CPU", total_cpu))

        total_mem_bytes = float(resources.get("memory", 0.0))
        available_mem_bytes = float(available.get("memory", total_mem_bytes))

        total_mem_gb = total_mem_bytes / 1e9
        available_mem_gb = available_mem_bytes / 1e9

        ray_nodes.append(
          NodeResources(
            node_id=str(node.get("NodeID", host)),
            host=host,
            total_cores=total_cpu,
            available_cores=available_cpu,
            total_memory_gb=total_mem_gb,
            available_memory_gb=available_mem_gb,
          )
        )

  if ray_nodes:
    return ray_nodes
  if slurm_nodes:
    return slurm_nodes

  cores, memory_gb = _detect_local_resources()
  hostname = socket.gethostname()
  return [
    NodeResources(
      node_id="local",
      host=hostname,
      total_cores=float(cores),
      available_cores=float(cores),
      total_memory_gb=memory_gb,
      available_memory_gb=memory_gb,
    )
  ]


class HaloScheduler:
  """
  Build resource-aware execution plans for halo processing.

  Parameters
  ----------
  total_memory_gb:
      Total memory available to each batch (typically the node capacity).
      When omitted, the value is detected from the host system.
  max_cores:
      Maximum number of CPU cores usable concurrently. Defaults to the number
      of logical CPUs detected on the node.
  reserve_memory_gb:
      Memory to keep in reserve for MPI buffers, filesystem caches, etc.
      Reducing this number allows denser packing at the cost of less headroom.
  min_cores_per_halo:
      Floor applied to every halo allocation.
  max_cores_per_halo:
      Optional ceiling for per-halo core assignments.
  core_scaling_exponent:
      Controls how aggressively core counts are scaled with halo size.
      Values < 1 favour medium halos, while values > 1 concentrate cores on
      the very largest halos.
  memory_model:
      Custom :class:`HaloMemoryModel` to use for size estimation.  If omitted,
      the default conservative model is applied.
  """

  def __init__(
    self,
    total_memory_gb: Optional[float] = None,
    max_cores: Optional[int] = None,
    *,
    reserve_memory_gb: float = 4.0,
    min_cores_per_halo: int = 1,
    max_cores_per_halo: Optional[int] = None,
    memory_model: Optional[HaloMemoryModel] = None,
  work_units_per_core: float = 25_000.0,
  ) -> None:
    detected_cores: Optional[int] = None
    detected_memory: Optional[float] = None
    if total_memory_gb is None or max_cores is None:
      detected_cores, detected_memory = detect_node_resources()
      if total_memory_gb is None:
        total_memory_gb = detected_memory
      if max_cores is None:
        max_cores = detected_cores

    if max_cores is None or total_memory_gb is None:
      raise ValueError("Both total_memory_gb and max_cores must be provided or detectable.")

    if max_cores < 1:
      raise ValueError("max_cores must be at least 1")
    if min_cores_per_halo < 1:
      raise ValueError("min_cores_per_halo must be at least 1")
    if max_cores_per_halo is not None and max_cores_per_halo < min_cores_per_halo:
      raise ValueError("max_cores_per_halo must be >= min_cores_per_halo")
    if work_units_per_core <= 0:
      raise ValueError("work_units_per_core must be positive")

    self._total_memory_gb = float(total_memory_gb)
    self._reserve_memory_gb = max(0.0, float(reserve_memory_gb))
    effective = self._total_memory_gb - self._reserve_memory_gb
    if effective <= 0:
      effective = max(self._total_memory_gb * 0.25, 1.0)
    self._memory_limit_gb = effective

    self._max_cores = int(max_cores)
    self._min_cores = int(min_cores_per_halo)
    self._max_cores_per_halo = int(max_cores_per_halo) if max_cores_per_halo is not None else None
    self._memory_model = memory_model or HaloMemoryModel()
    self._work_units_per_core = float(work_units_per_core)

  @property
  def memory_limit_gb(self) -> float:
    """Return the memory limit applied to each batch."""
    return self._memory_limit_gb

  def _assign_cores(self, work_units: float) -> int:
    cores = work_units / self._work_units_per_core
    suggested = max(self._min_cores, int(math.ceil(cores)))
    if self._max_cores_per_halo is not None:
      suggested = min(suggested, self._max_cores_per_halo)
    return min(suggested, self._max_cores)

  def build_plan(self, workloads: Sequence[HaloWorkload]) -> List[HaloBatch]:
    """
    Construct a set of sequential batches covering all ``workloads``.

    The resulting plan packs halos greedily (largest first) while respecting
    memory and core limits.  Any halo that does not fit within the limits is
    still scheduled but marked via ``HaloAllocation.needs_reschedule`` so
    higher-level orchestration can decide how to handle it (e.g. split across
    multiple ranks or run in isolation).
    """
    halo_list = list(workloads)
    if not halo_list:
      return []

    estimates: List[Dict[str, object]] = []
    for workload in halo_list:
      estimated_memory = self._memory_model.estimate_gb(workload)
      work_units = workload.work_units
      estimates.append(
        {
          "workload": workload,
          "memory_gb": estimated_memory,
          "work_units": work_units,
        }
      )

    for item in estimates:
      work_units = float(item["work_units"])  # type: ignore[arg-type]
      cores = self._assign_cores(work_units)
      item["cores"] = cores

    estimates.sort(key=lambda item: float(item["memory_gb"]), reverse=True)

    batches: List[HaloBatch] = []
    current_batch: Optional[HaloBatch] = None

    for item in estimates:
      workload = item["workload"]  # type: ignore[assignment]
      estimated_memory_gb = float(item["memory_gb"])  # type: ignore[arg-type]
      assigned_cores = int(item["cores"])  # type: ignore[arg-type]

      if current_batch is None:
        current_batch = HaloBatch(
          batch_id=len(batches),
          memory_limit_gb=self._memory_limit_gb,
          core_limit=self._max_cores,
        )
        batches.append(current_batch)

      allocation = current_batch.try_add(
        workload,
        estimated_memory_gb=estimated_memory_gb,
        assigned_cores=assigned_cores,
      )

      if allocation is None:
        current_batch = HaloBatch(
          batch_id=len(batches),
          memory_limit_gb=self._memory_limit_gb,
          core_limit=self._max_cores,
        )
        batches.append(current_batch)
        allocation = current_batch.try_add(
          workload,
          estimated_memory_gb=estimated_memory_gb,
          assigned_cores=assigned_cores,
        )
        if allocation is None:  # pragma: no cover - defensive (should not happen)
          raise RuntimeError("Failed to schedule halo despite empty batch.")

    return batches

  def flatten(self, plan: Iterable[HaloBatch]) -> List[HaloAllocation]:
    """Return a flattened list of allocations for convenience."""
    allocations: List[HaloAllocation] = []
    for batch in plan:
      allocations.extend(batch.allocations)
    return allocations


def build_workloads_from_catalog(
  catalog: "AHFCatalog",
  hosts: Iterable[int],
) -> List[HaloWorkload]:
  """
  Create :class:`HaloWorkload` entries for the requested top-level hosts.

  The catalog already contains per-node particle memberships, so we walk each
  host subtree and accumulate the particle counts without touching the Gadget
  snapshot.  This keeps memory usage bounded by the catalog slices we actually
  need to process.
  """
  workloads: List[HaloWorkload] = []
  for host in hosts:
    hid = int(host)
    totals: Dict[str, int] = {"gas": 0, "dm": 0, "star": 0, "bh": 0}
    bytes_hint = 0
    for node in catalog.iter_subtree(hid):
      pdata = catalog.particles.get(int(node))
      if not pdata:
        continue
      for ptype in totals.keys():
        arr = pdata.get(ptype)
        if arr is None:
          continue
        if isinstance(arr, np.ndarray):
          totals[ptype] += int(arr.size)
          bytes_hint += int(arr.nbytes)
        else:
          size = len(arr)
          totals[ptype] += int(size)
          # Assume 64-bit integers for non-numpy containers.
          bytes_hint += int(size * 8)
    workloads.append(
      HaloWorkload(
        halo_id=hid,
        particle_counts=totals,
        bytes_hint=bytes_hint or None,
        metadata={"top_host_id": hid},
      )
    )
  return workloads


def collect_top_level_hosts(
  particles_path: str | Path,
  halos_path: Optional[str | Path] = None,
  host_filter: Optional[Iterable[int]] = None,
) -> List[int]:
  """
  Read the AHF hierarchy and return the top-level hosts.

  When ``host_filter`` is provided, the returned list is restricted to that
  subset (intersection with the top-level hosts).  This only touches the
  ``.AHF_halos`` file, keeping snapshot IO out of the planning phase.
  """
  particles_file = Path(particles_path)
  halos_file = Path(halos_path) if halos_path is not None else None
  parent_of, _ = read_ahf_hierarchy(particles_file, halos_file)

  top_hosts = [hid for hid, parent in parent_of.items() if parent in (0, None)]
  if host_filter is None:
    return sorted(top_hosts)

  allowed = {int(h) for h in host_filter}
  return sorted(h for h in top_hosts if h in allowed)
