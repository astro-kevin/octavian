"""Core orchestration helpers for forthcoming MPI workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from octavian.backend import pd
import numpy as np

import octavian.constants as c
from octavian.ahf import load_catalog, apply_ahf_matching, build_galaxies_from_fast
from octavian.data_manager import DataManager
from octavian.utils import wrap_positions
from octavian.fof6d import run_fof6d
from octavian.group_funcs import calculate_group_properties
from octavian.backend import backend_info

Mode = Literal["fof", "ahf", "ahf-fast"]


@dataclass(frozen=True)
class CoreInputs:
  """Configuration passed to the core execution routine."""

  snapshot_path: Path
  ahf_particles: Path
  host_ids: Tuple[int, ...]
  mode: Mode = "ahf-fast"
  use_modin: bool = field(default_factory=lambda: backend_info()[1])
  n_threads: int = 1
  min_stars: int = c.MINIMUM_STARS_PER_GALAXY

  def __init__(
    self,
    snapshot_path: Path | str,
    ahf_particles: Path | str,
    host_ids: Iterable[int],
    *,
    mode: Mode = "ahf-fast",
    use_modin: Optional[bool] = None,
    n_threads: int = 1,
    min_stars: int = c.MINIMUM_STARS_PER_GALAXY,
  ) -> None:
    object.__setattr__(self, "snapshot_path", Path(snapshot_path))
    object.__setattr__(self, "ahf_particles", Path(ahf_particles))
    object.__setattr__(self, "host_ids", tuple(int(h) for h in host_ids))
    object.__setattr__(self, "mode", mode)
    if use_modin is None:
      resolved_use_modin = backend_info()[1]
    else:
      resolved_use_modin = bool(use_modin)
    object.__setattr__(self, "use_modin", resolved_use_modin)
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

  if config.mode not in {"ahf", "ahf-fast"}:
    raise NotImplementedError("Core execution currently supports AHF and AHF-FAST modes")

  catalog = load_catalog(str(config.ahf_particles), host_filter=config.host_ids)

  pid_chunks: Dict[str, List[np.ndarray]] = {ptype: [] for ptype in c.ptypes.keys()}
  print(f"Processing {len(config.host_ids)} top-level host(s)...", flush=True)
  for host_id in config.host_ids:
    for node in catalog.iter_subtree(int(host_id)):
      particles = catalog.particles.get(int(node))
      if particles is None:
        continue
      for ptype in c.ptypes.keys():
        entries = particles.get(ptype)
        if entries is None or entries.size == 0:
          continue
        pid_chunks[ptype].append(entries.astype(np.int64, copy=False))

  particle_ids: Dict[str, np.ndarray] = {}
  for ptype, arrays in pid_chunks.items():
    if not arrays:
      particle_ids[ptype] = np.array([], dtype=np.int64)
      continue
    combined = np.unique(np.concatenate(arrays))
    particle_ids[ptype] = combined
    print(f"  {ptype}: {combined.size} unique particles", flush=True)

  manager = DataManager(
    str(config.snapshot_path),
    mode=config.mode,
    use_modin=config.use_modin,
    particle_ids=particle_ids,
    include_unassigned=True,
    map_threads=config.n_threads,
  )

  halo_map = {}
  missing = {}

  if config.mode == "ahf":
    print("Stage 1/4: Wrapping positions...", flush=True)
    wrap_positions(manager)

    print("Stage 2/4: Running FoF6D...", flush=True)
    run_fof6d(manager, nproc=config.n_threads)

    print("Stage 3/4: Matching AHF halos...", flush=True)
    halo_map, missing = apply_ahf_matching(manager, catalog, n_jobs=config.n_threads)

    print("Stage 4/4: Computing group properties...", flush=True)
    calculate_group_properties(manager, include_global=True)

    metadata: Dict[str, Any] = {
      "host_ids": list(config.host_ids),
      "matched_halos": len(halo_map),
      "missing_particles": dict(missing),
      "n_galaxies": len(manager.galaxies),
    }
  else:
    print("Stage 1/3: Building galaxies from AHF-FAST catalogue...", flush=True)
    missing = build_galaxies_from_fast(
      manager,
      catalog,
      min_stars=config.min_stars,
      n_jobs=config.n_threads,
      hosts=config.host_ids,
    )
    print(f"  Missing particle counts: {missing}", flush=True)

    for ptype in c.ptypes.keys():
      manager.load_property('vel', ptype)

    print("Stage 2/3: Wrapping positions...", flush=True)
    wrap_positions(manager)

    print("Stage 3/3: Computing group properties...", flush=True)
    calculate_group_properties(manager, include_global=True)

    halo_map = {int(hid): 1 for hid in getattr(manager, "haloIDs", np.array([], dtype=int))}
    metadata = {
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
