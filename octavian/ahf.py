"""Utilities for integrating AHF halo catalogues with Octavian."""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

import octavian.constants as c
from contextlib import contextmanager
from joblib import Parallel, delayed
from joblib import parallel as joblib_parallel

try:
  from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
  def tqdm(iterable, *args, **kwargs):  # type: ignore
    return iterable

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


@contextmanager
def tqdm_joblib(tqdm_object):
  class TqdmBatchCompletionCallback(joblib_parallel.BatchCompletionCallBack):
    def __call__(self, *args, **kwargs):
      tqdm_object.update(n=self.batch_size)
      return super().__call__(*args, **kwargs)

  old_callback = joblib_parallel.BatchCompletionCallBack
  joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
  try:
    yield tqdm_object
  finally:
    joblib_parallel.BatchCompletionCallBack = old_callback
    tqdm_object.close()


def _open_catalog(path: Path):
  if path.suffix == ".gz":
    return gzip.open(path, "rt")
  return open(path, "r")


def _to_sorted_array(values) -> np.ndarray:
  if isinstance(values, np.ndarray):
    arr = values.astype(np.int64, copy=True)
  else:
    arr = np.array(list(values), dtype=np.int64)
  if arr.size == 0:
    return np.empty(0, dtype=np.int64)
  return np.unique(arr)


def _normalise_memberships(memberships: Dict[int, Dict[str, Set[int]]]) -> None:
  for hid, pdata in memberships.items():
    for name in ("gas", "dm", "star", "bh"):
      pdata[name] = _to_sorted_array(pdata.get(name, set()))


def _empty_array() -> np.ndarray:
  return np.empty(0, dtype=np.int64)


def _union_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  if a.size == 0 and b.size == 0:
    return _empty_array()
  if a.size == 0:
    return b.copy()
  if b.size == 0:
    return a.copy()
  return np.union1d(a, b)


def _prepare_index_lookup(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
  if 'pid' not in frame or frame.empty:
    return _empty_array(), _empty_array()
  pid_values = frame['pid'].to_numpy(dtype=np.int64, copy=False)
  indices = frame.index.to_numpy(dtype=np.int64, copy=False)
  order = np.argsort(pid_values)
  return pid_values[order], indices[order]


def _map_pid_array(pid_array: np.ndarray, values: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, int]:
  if pid_array.size == 0:
    return _empty_array(), 0
  if values.size == 0:
    return _empty_array(), pid_array.size
  positions = np.searchsorted(values, pid_array)
  within = positions < values.size
  matches = np.zeros(pid_array.size, dtype=bool)
  if within.any():
    pos_within = positions[within]
    pid_within = pid_array[within]
    matches_within = values[pos_within] == pid_within
    matches[within] = matches_within
  if matches.any():
    matched_indices = indices[positions[matches]]
  else:
    matched_indices = _empty_array()
  missing = pid_array.size - matched_indices.size
  return matched_indices, missing


def read_ahf_particles(path: Path) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, np.ndarray]]:
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
  for pid, owners in list(star_owner.items()):
    star_owner[pid] = _to_sorted_array(owners)

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
  particles: Dict[int, Dict[str, np.ndarray]]
  star_owner: Dict[int, np.ndarray]
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

  def match_galaxies_to_halos(self, galaxy_star_sets: Mapping[int, np.ndarray], n_jobs: int = 1) -> Dict[int, int]:
    def _match(item: Tuple[int, np.ndarray]):
      gid, stars = item
      owner_arrays = [self.star_owner.get(int(pid)) for pid in stars if int(pid) in self.star_owner]
      owner_arrays = [arr for arr in owner_arrays if arr is not None and arr.size]
      if not owner_arrays:
        return gid, None
      all_halos = np.concatenate(owner_arrays)
      unique_halos, counts = np.unique(all_halos, return_counts=True)
      if unique_halos.size == 0:
        return gid, None
      depths = np.array([self.depths.get(int(h), 0) for h in unique_halos], dtype=np.int64)
      order = np.lexsort((unique_halos, -counts, -depths))
      best_idx = order[0]
      return gid, int(unique_halos[best_idx])

    items = list(galaxy_star_sets.items())
    matches: Dict[int, int] = {}
    if n_jobs == 1 or len(items) <= 1:
      for item in tqdm(items, total=len(items), desc='Matching galaxies to AHF halos', unit='gal', leave=False):
        gid, best = _match(item)
        if best is not None:
          matches[gid] = best
    else:
      with tqdm_joblib(tqdm(total=len(items), desc='Matching galaxies to AHF halos', unit='gal', leave=False)):
        results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_match)(item) for item in items)
      for gid, best in results:
        if best is not None:
          matches[gid] = best
    return matches

  def _compute_exclusives_for_nodes(self, nodes: Set[int], selected: Set[int]) -> Dict[int, Dict[str, np.ndarray]]:
    exclusives_local: Dict[int, Dict[str, np.ndarray]] = {}

    def compute(hid: int) -> Dict[str, np.ndarray]:
      if hid in exclusives_local:
        return exclusives_local[hid]
      pdata = self.particles.get(hid)
      if pdata is None:
        exclusives_local[hid] = {name: np.empty(0, dtype=np.int64) for name in _BARYON_TYPES}
        return exclusives_local[hid]
      result = {name: pdata[name].copy() for name in _BARYON_TYPES}
      for child in self.children_of.get(hid, []):
        if child not in selected:
          continue
        child_sets = compute(child)
        for name in _BARYON_TYPES:
          if child_sets[name].size:
            result[name] = np.setdiff1d(result[name], child_sets[name], assume_unique=True)
      exclusives_local[hid] = result
      return result

    for hid in sorted(nodes, key=lambda h: self.depths.get(h, 0)):
      compute(hid)

    return exclusives_local

  def compute_exclusive_baryons(self, selected_halos: Iterable[int], n_jobs: int = 1) -> Dict[int, Dict[str, np.ndarray]]:
    selected = set(selected_halos)
    host_to_nodes: Dict[int, Set[int]] = {}
    for hid in selected:
      host = top_host(hid, self.parent_of)
      host_to_nodes.setdefault(host, set()).add(hid)

    exclusives: Dict[int, Dict[str, Set[int]]] = {}
    if n_jobs == 1 or len(host_to_nodes) <= 1:
      for host, nodes in tqdm(host_to_nodes.items(), total=len(host_to_nodes), desc='Computing exclusive baryons', unit='host', leave=False):
        exclusives.update(self._compute_exclusives_for_nodes(nodes, selected))
    else:
      host_items = list(host_to_nodes.items())
      with tqdm_joblib(tqdm(total=len(host_items), desc='Computing exclusive baryons', unit='host', leave=False)):
        results = Parallel(n_jobs=n_jobs, backend='threading')(
          delayed(self._compute_exclusives_for_nodes)(nodes, selected)
          for _, nodes in host_items
        )
      for result in results:
        exclusives.update(result)

    return exclusives

  def gather_dm_sets(self, halos: Iterable[int]) -> Dict[int, np.ndarray]:
    dm_map: Dict[int, np.ndarray] = {}
    for hid in halos:
      pdata = self.particles.get(hid)
      if pdata is None:
        dm_map[hid] = _empty_array()
      else:
        dm_map[hid] = pdata["dm"].copy()
    return dm_map

  def build_fast_payloads(self, min_stars: int, n_jobs: int = 1) -> List[Tuple[int, int, Dict[str, np.ndarray], np.ndarray]]:
    def process_host(host: int) -> List[Tuple[int, int, Dict[str, np.ndarray], np.ndarray]]:
      local_payloads: List[Tuple[int, int, Dict[str, np.ndarray], np.ndarray]] = []
      carry_gas: Dict[int, np.ndarray] = {}
      carry_star: Dict[int, np.ndarray] = {}
      carry_bh: Dict[int, np.ndarray] = {}

      nodes = sorted(self.iter_subtree(host), key=lambda h: self.depths.get(h, 0), reverse=True)
      for node in nodes:
        pdata = self.particles.get(node)
        gas = pdata["gas"].copy() if pdata is not None else _empty_array()
        star = pdata["star"].copy() if pdata is not None else _empty_array()
        bh = pdata["bh"].copy() if pdata is not None else _empty_array()

        if node in carry_gas:
          gas = _union_arrays(gas, carry_gas.pop(node))
        if node in carry_star:
          star = _union_arrays(star, carry_star.pop(node))
        if node in carry_bh:
          bh = _union_arrays(bh, carry_bh.pop(node))

        if len(star) >= min_stars:
          dm = pdata["dm"].copy() if pdata is not None else _empty_array()
          local_payloads.append((node, host, {"gas": gas, "star": star, "bh": bh}, dm))
        else:
          parent = self.parent_of.get(node, 0)
          if parent not in (0, None):
            carry_gas[parent] = _union_arrays(carry_gas.get(parent, _empty_array()), gas)
            carry_star[parent] = _union_arrays(carry_star.get(parent, _empty_array()), star)
            carry_bh[parent] = _union_arrays(carry_bh.get(parent, _empty_array()), bh)

      return local_payloads

    hosts = sorted(self.top_level_halos, key=lambda h: self.depths.get(h, 0))
    if n_jobs == 1 or len(hosts) <= 1:
      payloads: List[Tuple[int, int, Dict[str, np.ndarray], np.ndarray]] = []
      for host in tqdm(hosts, total=len(hosts), desc='Processing AHF hosts', unit='host', leave=False):
        payloads.extend(process_host(host))
      return payloads

    with tqdm_joblib(tqdm(total=len(hosts), desc='Processing AHF hosts', unit='host', leave=False)):
      results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_host)(host) for host in hosts)

    payloads: List[Tuple[int, int, Dict[str, np.ndarray], np.ndarray]] = []
    for chunk in results:
      payloads.extend(chunk)
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


def apply_ahf_matching(manager: 'DataManager', catalog: AHFCatalog, n_jobs: int = 1) -> Tuple[Dict[int, int], Dict[str, int]]:
  _ensure_pid_columns(manager, c.ptypes.keys())
  stars = manager['star']
  if 'pid' not in stars:
    raise ValueError('Star particle IDs are required for AHF matching.')

  galaxies = stars.loc[stars['GalID'] != -1]
  galaxy_star_sets: Dict[int, np.ndarray] = {}
  galaxy_groups = galaxies.groupby('GalID')
  total_groups = getattr(galaxy_groups, 'ngroups', None)
  for gid, subset in tqdm(galaxy_groups, total=total_groups, desc='Collecting galaxy members', unit='gal', leave=False):
    galaxy_star_sets[int(gid)] = np.unique(subset['pid'].to_numpy(dtype=np.int64))

  if not galaxy_star_sets:
    raise ValueError('No FoF galaxies available to match against AHF halos.')

  from collections import defaultdict, Counter

  exclusives = catalog.compute_exclusive_baryons(set(catalog.particles.keys()), n_jobs=n_jobs)
  exclusive_star_map: Dict[int, int] = {}
  for hid, data in exclusives.items():
    stars = data.get('star')
    if stars is None or stars.size == 0:
      continue
    for pid in stars:
      exclusive_star_map[int(pid)] = int(hid)

  num_original_galaxies = len(manager.galaxies)
  galaxy_to_halo = np.full(num_original_galaxies, -1, dtype=np.int64)
  halo_to_galaxy_indices: Dict[int, List[int]] = defaultdict(list)
  matches: Dict[int, int] = {}

  match_iterable = galaxy_star_sets.items()
  if len(galaxy_star_sets) > 1:
    match_iterable = tqdm(
      galaxy_star_sets.items(),
      total=len(galaxy_star_sets),
      desc='Matching galaxies to AHF halos',
      unit='gal',
      leave=False
    )

  for gid, stars in match_iterable:
    counts: Counter[int] = Counter()
    for pid in stars:
      hid = exclusive_star_map.get(int(pid))
      if hid is not None:
        counts[hid] += 1

    if not counts:
      continue

    best_hid, _ = max(
      counts.items(),
      key=lambda item: (catalog.depths.get(int(item[0]), 0), item[1], -int(item[0]))
    )
    matches[gid] = int(best_hid)
    if 0 <= gid < num_original_galaxies:
      galaxy_to_halo[gid] = int(best_hid)
      halo_to_galaxy_indices[int(best_hid)].append(gid)

  unique_halos = sorted(halo_to_galaxy_indices.keys())

  if 'AHF_halo_id' not in manager.galaxies:
    manager.galaxies['AHF_halo_id'] = -1
  if 'AHF_host_id' not in manager.galaxies:
    manager.galaxies['AHF_host_id'] = -1
  if 'AHF_matched' not in manager.galaxies:
    manager.galaxies['AHF_matched'] = False

  ahf_halo_ids = np.full(num_original_galaxies, -1, dtype=np.int64)
  ahf_host_ids = np.full(num_original_galaxies, -1, dtype=np.int64)

  for gid, hid in enumerate(galaxy_to_halo.tolist()):
    if hid < 0:
      continue
    ahf_halo_ids[gid] = hid
    ahf_host_ids[gid] = top_host(hid, catalog.parent_of)

  manager.galaxies.loc[:, 'AHF_halo_id'] = ahf_halo_ids[:len(manager.galaxies)]
  manager.galaxies.loc[:, 'AHF_host_id'] = ahf_host_ids[:len(manager.galaxies)]
  manager.galaxies.loc[:, 'AHF_matched'] = manager.galaxies['AHF_halo_id'] >= 0

  missing_counts = {'gas': 0, 'star': 0, 'bh': 0, 'dm': 0}
  halo_map = {hid: len(halo_to_galaxy_indices.get(hid, [])) for hid in unique_halos}
  return halo_map, missing_counts


def build_galaxies_from_fast(manager: 'DataManager', catalog: AHFCatalog, min_stars: int = c.MINIMUM_STARS_PER_GALAXY, n_jobs: int = 1) -> Dict[str, int]:
  _ensure_pid_columns(manager, c.ptypes.keys())

  for ptype in c.ptypes.keys():
    frame = manager[ptype]
    frame['GalID'] = -1
    frame['HaloID'] = -1

  payloads = catalog.build_fast_payloads(min_stars, n_jobs=n_jobs)
  if not payloads:
    raise ValueError('AHF-FAST produced no halos meeting the star threshold.')

  host_ids = sorted({host for _, host, _, _ in payloads if host not in (0, None)})
  manager.haloIDs = np.array(host_ids, dtype=int)
  manager.halos = pd.DataFrame(index=manager.haloIDs)
  manager.galaxies = pd.DataFrame(index=np.arange(len(payloads)))

  index_lookup = {ptype: _prepare_index_lookup(manager[ptype]) for ptype in c.ptypes.keys()}
  missing_counts = {'gas': 0, 'star': 0, 'bh': 0, 'dm': 0}

  for gid, (node_id, host_id, baryons, dm_set) in tqdm(enumerate(payloads), total=len(payloads), desc='Assigning AHF-FAST galaxies', unit='gal', leave=False):
    if host_id in (0, None):
      host_id = top_host(node_id, catalog.parent_of)
    manager.galaxies.loc[gid, 'AHF_halo_id'] = node_id
    manager.galaxies.loc[gid, 'AHF_host_id'] = host_id

    for ptype in ('gas', 'star', 'bh'):
      pid_array = baryons.get(ptype, _empty_array())
      if pid_array.size == 0:
        continue
      lookup_values, lookup_indices = index_lookup[ptype]
      valid, missing = _map_pid_array(pid_array, lookup_values, lookup_indices)
      missing_counts[ptype] += missing
      if missing:
        missing_pid_sample = np.setdiff1d(pid_array, lookup_values, assume_unique=False)[:10]
        raise RuntimeError(
          f"AHF-FAST assignment failed: {missing} {ptype} particles missing for galaxy {gid} "
          f"(node={node_id}, host={host_id}). Sample PIDs: {missing_pid_sample.tolist()}"
        )
      if valid.size:
        manager[ptype].loc[valid, 'GalID'] = gid
        manager[ptype].loc[valid, 'HaloID'] = host_id

    if dm_set.size:
      dm_values, dm_indices = index_lookup['dm']
      valid_dm, missing_dm = _map_pid_array(dm_set, dm_values, dm_indices)
      missing_counts['dm'] += missing_dm
      if missing_dm:
        missing_dm_sample = np.setdiff1d(dm_set, dm_values, assume_unique=False)[:10]
        raise RuntimeError(
          f"AHF-FAST assignment failed: {missing_dm} dm particles missing for galaxy {gid} "
          f"(node={node_id}, host={host_id}). Sample PIDs: {missing_dm_sample.tolist()}"
        )
      if valid_dm.size:
        manager['dm'].loc[valid_dm, 'GalID'] = gid
        manager['dm'].loc[valid_dm, 'HaloID'] = host_id

  for ptype in c.ptypes.keys():
    frame = manager[ptype]
    assigned_mask = frame['HaloID'] != -1
    manager[ptype] = frame.loc[assigned_mask].copy() if assigned_mask.any() else frame.loc[assigned_mask]

  if len(manager.haloIDs) > 0:
    manager.load_halo_pids()
  manager.load_galaxy_pids()
  return missing_counts
