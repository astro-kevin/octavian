"""Utilities for integrating AHF halo catalogues with Octavian."""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

import octavian.constants as c

if TYPE_CHECKING:
  from octavian.data_manager import DataManager

# Particle type mapping used in AHF catalogue output
_PTYPE_NAME = {
  0: "gas",
  1: "dm",
  4: "star",
  5: "bh",
}

_BARYON_TYPES = ("gas", "star", "bh")


def _open_catalog(path: Path):
  if path.suffix == ".gz":
    return gzip.open(path, "rt")
  return open(path, "r")


def _normalise_memberships(memberships: Dict[int, Dict[str, Set[int]]]) -> None:
  for hid in memberships:
    pdata = memberships[hid]
    for name in ("gas", "dm", "star", "bh"):
      pdata.setdefault(name, set())


def read_ahf_particles(path: Path) -> Tuple[Dict[int, Dict[str, Set[int]]], Dict[int, Set[int]]]:
  """Parse an ``AHF_particles`` catalogue."""
  memberships: Dict[int, Dict[str, Set[int]]] = {}
  star_owner: Dict[int, Set[int]] = {}

  with _open_catalog(path) as fh:
    current_hid: Optional[int] = None
    remaining = 0
    for raw in fh:
      line = raw.strip()
      if not line:
        continue
      parts = line.split()
      if remaining == 0:
        if len(parts) != 2:
          continue
        try:
          remaining = int(parts[0])
          current_hid = int(parts[1])
        except ValueError:
          current_hid = None
          remaining = 0
          continue
        memberships.setdefault(current_hid, {})
        continue

      if current_hid is None or len(parts) != 2:
        remaining -= 1
        continue

      try:
        pid = int(parts[0])
        ptype = int(parts[1])
      except ValueError:
        remaining -= 1
        continue

      name = _PTYPE_NAME.get(ptype)
      if name is None:
        remaining -= 1
        continue

      pdata = memberships.setdefault(current_hid, {})
      pdata.setdefault(name, set()).add(pid)
      if name == "star":
        star_owner.setdefault(pid, set()).add(current_hid)

      remaining -= 1
      if remaining == 0:
        current_hid = None

  _normalise_memberships(memberships)
  return memberships, star_owner


def read_ahf_hierarchy(particles_path: Path, halos_path: Optional[Path] = None) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
  if halos_path is None:
    name = particles_path.name
    if name.endswith("particles.gz"):
      halos_path = particles_path.with_name(name.replace("particles.gz", "halos"))
    else:
      halos_path = particles_path.with_name(name.replace("particles", "halos"))

  if not halos_path.exists():
    raise FileNotFoundError(f"AHF halos file not found: {halos_path}")

  parent_of: Dict[int, int] = {}
  children_of: Dict[int, List[int]] = {}

  opener = gzip.open if halos_path.suffix == ".gz" else open

  with opener(halos_path, "rt") as fh:
    for raw in fh:
      line = raw.strip()
      if not line:
        continue
      parts = line.split()
      if len(parts) < 2:
        continue
      try:
        hid = int(parts[0])
        host = int(parts[1])
      except ValueError:
        continue
      parent_of[hid] = host
      children_of.setdefault(host, []).append(hid)
      children_of.setdefault(hid, [])

  return parent_of, children_of


def compute_depths(parent_of: Mapping[int, int]) -> Dict[int, int]:
  depths: Dict[int, int] = {}

  def depth(hid: int) -> int:
    if hid in depths:
      return depths[hid]
    host = parent_of.get(hid, 0)
    if host in (0, None):
      depths[hid] = 0
    else:
      depths[hid] = depth(host) + 1
    return depths[hid]

  for hid in parent_of.keys():
    depth(hid)
  return depths


def top_host(hid: int, parent_of: Mapping[int, int]) -> int:
  cur = hid
  while True:
    host = parent_of.get(cur, 0)
    if host in (0, None):
      return cur
    cur = host


@dataclass
class AHFCatalog:
  particles: Dict[int, Dict[str, Set[int]]]
  star_owner: Dict[int, Set[int]]
  parent_of: Dict[int, int]
  children_of: Dict[int, List[int]]
  depths: Dict[int, int] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.depths:
      self.depths = compute_depths(self.parent_of)

  @property
  def top_level_halos(self) -> List[int]:
    return [hid for hid, parent in self.parent_of.items() if parent in (0, None)]

  def iter_subtree(self, root: int) -> Iterator[int]:
    stack = [root]
    seen: Set[int] = set()
    while stack:
      hid = stack.pop()
      if hid in seen:
        continue
      seen.add(hid)
      yield hid
      for child in self.children_of.get(hid, []):
        stack.append(child)

  def match_galaxies_to_halos(self, galaxy_star_sets: Mapping[int, Set[int]]) -> Dict[int, int]:
    matches: Dict[int, int] = {}
    for gid, stars in galaxy_star_sets.items():
      counts: Dict[int, int] = {}
      for pid in stars:
        for hid in self.star_owner.get(int(pid), ()):  # tolerate numpy ints
          counts[hid] = counts.get(hid, 0) + 1
      if not counts:
        continue
      best_hid = max(counts.items(), key=lambda item: (self.depths.get(item[0], 0), item[1], item[0]))[0]
      matches[gid] = best_hid
    return matches

  def compute_exclusive_baryons(self, selected_halos: Iterable[int]) -> Dict[int, Dict[str, Set[int]]]:
    selected = set(selected_halos)
    exclusives: Dict[int, Dict[str, Set[int]]] = {}

    def compute(hid: int) -> Dict[str, Set[int]]:
      if hid in exclusives:
        return exclusives[hid]
      pdata = self.particles.get(hid)
      if pdata is None:
        exclusives[hid] = {name: set() for name in _BARYON_TYPES}
        return exclusives[hid]
      result = {name: set(pdata[name]) for name in _BARYON_TYPES}
      for child in self.children_of.get(hid, []):
        if child not in selected:
          continue
        child_sets = compute(child)
        for name in _BARYON_TYPES:
          if child_sets[name]:
            result[name].difference_update(child_sets[name])
      exclusives[hid] = result
      return result

    for hid in selected:
      compute(hid)
    return exclusives

  def gather_dm_sets(self, halos: Iterable[int]) -> Dict[int, Set[int]]:
    dm_map: Dict[int, Set[int]] = {}
    for hid in halos:
      pdata = self.particles.get(hid)
      dm_map[hid] = set() if pdata is None else set(pdata.get("dm", set()))
    return dm_map

  def build_fast_payloads(self, min_stars: int) -> List[Tuple[int, int, Dict[str, Set[int]], Set[int]]]:
    payloads: List[Tuple[int, int, Dict[str, Set[int]], Set[int]]] = []
    carry_gas: Dict[int, Set[int]] = {}
    carry_star: Dict[int, Set[int]] = {}
    carry_bh: Dict[int, Set[int]] = {}

    for host in sorted(self.top_level_halos, key=lambda h: self.depths.get(h, 0)):
      nodes = sorted(self.iter_subtree(host), key=lambda h: self.depths.get(h, 0), reverse=True)
      for node in nodes:
        pdata = self.particles.get(node)
        gas = set(pdata.get("gas", set())) if pdata else set()
        star = set(pdata.get("star", set())) if pdata else set()
        bh = set(pdata.get("bh", set())) if pdata else set()

        gas |= carry_gas.pop(node, set())
        star |= carry_star.pop(node, set())
        bh |= carry_bh.pop(node, set())

        if len(star) >= min_stars:
          dm = set(pdata.get("dm", set())) if pdata else set()
          payloads.append((node, host, {"gas": gas, "star": star, "bh": bh}, dm))
        else:
          parent = self.parent_of.get(node, 0)
          if parent not in (0, None):
            carry_gas.setdefault(parent, set()).update(gas)
            carry_star.setdefault(parent, set()).update(star)
            carry_bh.setdefault(parent, set()).update(bh)

    return payloads


def load_catalog(particles_path: str, halos_path: Optional[str] = None) -> AHFCatalog:
  particles_file = Path(particles_path)
  halos_file = Path(halos_path) if halos_path is not None else None
  memberships, star_owner = read_ahf_particles(particles_file)
  parent_of, children_of = read_ahf_hierarchy(particles_file, halos_file)
  return AHFCatalog(
    particles=memberships,
    star_owner=star_owner,
    parent_of=parent_of,
    children_of=children_of,
  )


def _ensure_pid_columns(manager: 'DataManager', ptypes: Iterable[str]) -> None:
  for ptype in ptypes:
    try:
      manager.ensure_property('pid', ptype)
    except Exception:
      # Some simulations may not contain all particle types
      continue


def _build_pid_maps(manager: 'DataManager', ptypes: Iterable[str]) -> Dict[str, Dict[int, int]]:
  pid_maps: Dict[str, Dict[int, int]] = {}
  for ptype in ptypes:
    frame = manager[ptype]
    if 'pid' not in frame:
      pid_maps[ptype] = {}
      continue
    pid_values = frame['pid'].to_numpy()
    indices = frame.index.to_numpy()
    pid_maps[ptype] = {int(pid): int(idx) for pid, idx in zip(pid_values, indices)}
  return pid_maps


def apply_ahf_matching(manager: 'DataManager', catalog: AHFCatalog) -> Tuple[Dict[int, int], Dict[str, int]]:
  _ensure_pid_columns(manager, c.ptypes.keys())
  stars = manager['star']
  if 'pid' not in stars:
    raise ValueError('Star particle IDs are required for AHF matching.')

  galaxies = stars.loc[stars['GalID'] != -1]
  galaxy_star_sets: Dict[int, Set[int]] = {}
  for gid, subset in galaxies.groupby('GalID'):
    galaxy_star_sets[int(gid)] = set(int(pid) for pid in subset['pid'].to_numpy())

  if not galaxy_star_sets:
    raise ValueError('No FoF galaxies available to match against AHF halos.')

  matches = catalog.match_galaxies_to_halos(galaxy_star_sets)
  if not matches:
    raise ValueError('Unable to match any galaxies to AHF halos.')

  selected_halos = sorted({hid for hid in matches.values() if hid is not None and hid >= 0})
  if not selected_halos:
    raise ValueError('No valid AHF halos matched to FoF galaxies.')

  exclusives = catalog.compute_exclusive_baryons(selected_halos)
  dm_sets = catalog.gather_dm_sets(selected_halos)

  pid_maps = _build_pid_maps(manager, c.ptypes.keys())
  for ptype in c.ptypes.keys():
    manager[ptype]['GalID'] = -1

  halo_to_gid = {hid: idx for idx, hid in enumerate(selected_halos)}
  manager.galaxies = pd.DataFrame(index=np.arange(len(selected_halos)))
  manager.galaxies['AHF_halo_id'] = selected_halos
  manager.galaxies['AHF_host_id'] = [top_host(hid, catalog.parent_of) for hid in selected_halos]

  missing_counts = {'gas': 0, 'star': 0, 'bh': 0, 'dm': 0}

  for hid in selected_halos:
    gid = halo_to_gid[hid]
    baryons = exclusives.get(hid, {})
    for ptype in ('gas', 'star', 'bh'):
      pid_set = baryons.get(ptype, set()) or set()
      if not pid_set:
        continue
      indexes = [pid_maps[ptype].get(int(pid)) for pid in pid_set]
      indexes = [idx for idx in indexes if idx is not None]
      missing_counts[ptype] += len(pid_set) - len(indexes)
      if indexes:
        manager[ptype].loc[indexes, 'GalID'] = gid

    dm_set = dm_sets.get(hid, set())
    if dm_set:
      indexes = [pid_maps['dm'].get(int(pid)) for pid in dm_set]
      indexes = [idx for idx in indexes if idx is not None]
      missing_counts['dm'] += len(dm_set) - len(indexes)
      if indexes:
        manager['dm'].loc[indexes, 'GalID'] = gid

  manager.load_galaxy_pids()
  return halo_to_gid, missing_counts


def build_galaxies_from_fast(manager: 'DataManager', catalog: AHFCatalog, min_stars: int = c.MINIMUM_STARS_PER_GALAXY) -> Dict[str, int]:
  _ensure_pid_columns(manager, c.ptypes.keys())

  for ptype in c.ptypes.keys():
    frame = manager[ptype]
    frame['GalID'] = -1
    frame['HaloID'] = -1

  payloads = catalog.build_fast_payloads(min_stars)
  if not payloads:
    raise ValueError('AHF-FAST produced no halos meeting the star threshold.')

  host_ids = sorted({host for _, host, _, _ in payloads if host not in (0, None)})
  manager.haloIDs = np.array(host_ids, dtype=int)
  manager.halos = pd.DataFrame(index=manager.haloIDs)
  manager.galaxies = pd.DataFrame(index=np.arange(len(payloads)))

  pid_maps = _build_pid_maps(manager, c.ptypes.keys())
  missing_counts = {'gas': 0, 'star': 0, 'bh': 0, 'dm': 0}

  for gid, (node_id, host_id, baryons, dm_set) in enumerate(payloads):
    if host_id in (0, None):
      host_id = top_host(node_id, catalog.parent_of)
    manager.galaxies.loc[gid, 'AHF_halo_id'] = node_id
    manager.galaxies.loc[gid, 'AHF_host_id'] = host_id

    for ptype in ('gas', 'star', 'bh'):
      pid_set = baryons.get(ptype, set()) or set()
      if not pid_set:
        continue
      indexes = [pid_maps[ptype].get(int(pid)) for pid in pid_set]
      indexes = [idx for idx in indexes if idx is not None]
      missing_counts[ptype] += len(pid_set) - len(indexes)
      if indexes:
        manager[ptype].loc[indexes, 'GalID'] = gid
        manager[ptype].loc[indexes, 'HaloID'] = host_id

    if dm_set:
      indexes = [pid_maps['dm'].get(int(pid)) for pid in dm_set]
      indexes = [idx for idx in indexes if idx is not None]
      missing_counts['dm'] += len(dm_set) - len(indexes)
      if indexes:
        manager['dm'].loc[indexes, 'GalID'] = gid
        manager['dm'].loc[indexes, 'HaloID'] = host_id

  for ptype in c.ptypes.keys():
    frame = manager[ptype]
    assigned_mask = frame['HaloID'] != -1
    manager[ptype] = frame.loc[assigned_mask].copy() if assigned_mask.any() else frame.loc[assigned_mask]

  if len(manager.haloIDs) > 0:
    manager.load_halo_pids()
  manager.load_galaxy_pids()
  return missing_counts
