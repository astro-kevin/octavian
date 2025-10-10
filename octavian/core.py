"""Core orchestration helpers for forthcoming MPI workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

import pandas as pd

import octavian.constants as c
from octavian.ahf import load_catalog, apply_ahf_matching
from octavian.data_manager import DataManager
from octavian.utils import wrap_positions
from octavian.fof6d import run_fof6d
from octavian.group_funcs import calculate_group_properties

Mode = Literal["fof", "ahf", "ahf-fast"]


@dataclass(frozen=True)
class CoreInputs:
  """Configuration passed to the core execution routine."""

  snapshot_path: Path
  ahf_particles: Path
  host_ids: Tuple[int, ...]
  mode: Mode = "ahf-fast"
  use_polars: bool = True
  n_threads: int = 1
  min_stars: int = c.MINIMUM_STARS_PER_GALAXY

  def __init__(
    self,
    snapshot_path: Path | str,
    ahf_particles: Path | str,
    host_ids: Iterable[int],
    *,
    mode: Mode = "ahf-fast",
    use_polars: bool = True,
    n_threads: int = 1,
    min_stars: int = c.MINIMUM_STARS_PER_GALAXY,
  ) -> None:
    object.__setattr__(self, "snapshot_path", Path(snapshot_path))
    object.__setattr__(self, "ahf_particles", Path(ahf_particles))
    object.__setattr__(self, "host_ids", tuple(int(h) for h in host_ids))
    object.__setattr__(self, "mode", mode)
    object.__setattr__(self, "use_polars", bool(use_polars))
    object.__setattr__(self, "n_threads", int(n_threads))
    object.__setattr__(self, "min_stars", int(min_stars))


@dataclass
class CoreResult:
  """Result packet returned by :func:`run_core`."""

  host_ids: List[int]
  halos: pd.DataFrame
  galaxies: pd.DataFrame
  metadata: Dict[str, Any] = field(default_factory=dict)


def run_core(config: CoreInputs) -> CoreResult:  # pragma: no cover - placeholder
  """Execute identification for the requested hosts.

  The implementation will reuse the existing DataManager, FoF, and AHF helpers
  to build halo and galaxy tables for ``config.host_ids``.
  """
  if not config.host_ids:
    raise ValueError("At least one host ID must be supplied to the core")

  if config.mode != "ahf":
    raise NotImplementedError("Core execution currently supports AHF mode only")

  catalog = load_catalog(str(config.ahf_particles))

  particle_ids: Dict[str, List[int]] = {ptype: [] for ptype in c.ptypes.keys()}
  seen: Dict[str, set[int]] = {ptype: set() for ptype in c.ptypes.keys()}
  for host_id in config.host_ids:
    for node in catalog.iter_subtree(int(host_id)):
      particles = catalog.particles.get(int(node))
      if particles is None:
        continue
      for ptype in c.ptypes.keys():
        entries = particles.get(ptype)
        if entries is None or entries.size == 0:
          continue
        bucket = seen[ptype]
        for pid in entries.tolist():
          if pid not in bucket:
            bucket.add(pid)
            particle_ids[ptype].append(pid)

  manager = DataManager(
    str(config.snapshot_path),
    mode=config.mode,
    use_polars=config.use_polars,
    particle_ids=particle_ids,
    include_unassigned=True,
  )

  wrap_positions(manager)
  run_fof6d(manager, nproc=config.n_threads)

  halo_map, missing = apply_ahf_matching(manager, catalog, n_jobs=config.n_threads)

  calculate_group_properties(
    manager,
    use_polars=getattr(manager, "use_polars", False),
    include_global=False,
  )

  metadata: Dict[str, Any] = {
    "host_ids": list(config.host_ids),
    "matched_halos": len(halo_map),
    "missing_particles": dict(missing),
    "n_galaxies": len(manager.galaxies),
  }

  return CoreResult(
    host_ids=list(config.host_ids),
    halos=manager.halos.copy(),
    galaxies=manager.galaxies.copy(),
    metadata=metadata,
  )
