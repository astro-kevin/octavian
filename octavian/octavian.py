from os import PathLike
from time import perf_counter
from typing import Optional

from octavian.constants import code_units
from octavian.data_manager import DataManager
from octavian.group_funcs import calculate_group_properties
from octavian.saver import Saver
from octavian.utils import wrap_positions
from octavian.fof6d import run_fof6d
from octavian.ahf import load_catalog, apply_ahf_matching, build_galaxies_from_fast


class OCTAVIAN:
  def __init__(self, dataset: PathLike, units: dict = {}, nproc: int = 1, mode: str = 'fof', ahf_particles: Optional[PathLike] = None, ahf_halos: Optional[PathLike] = None, use_polars: Optional[bool] = None, *args, **kwargs):
    self._args = args
    self.dataset = dataset

    self.units = code_units | units
    self.nproc = nproc

    self.mode = mode.lower()
    if self.mode not in {'fof', 'ahf', 'ahf-fast'}:
      raise ValueError("mode must be one of 'fof', 'ahf', or 'ahf-fast'")

    self.ahf_particles = ahf_particles
    self.ahf_halos = ahf_halos
    if self.mode in {'ahf', 'ahf-fast'} and self.ahf_particles is None:
      raise ValueError('AHF modes require the path to an AHF_particles catalogue.')

    from octavian.group_funcs import HAS_POLARS

    if not HAS_POLARS:
      self.use_polars = False
    elif use_polars is None:
      self.use_polars = True
    else:
      self.use_polars = bool(use_polars)

  def _log_step(self, current: int, total: int, message: str) -> None:
    print(f"[{current}/{total}] {message}", flush=True)

  def _log_duration(self, start_time: float) -> None:
    print(f"    Done in {perf_counter() - start_time:.2f} seconds.", flush=True)

  def run(self, file: PathLike, *args, **kwargs):
    if self.mode == 'fof':
      total_steps = 5
    elif self.mode == 'ahf':
      total_steps = 7
    else:
      total_steps = 6

    step = 1
    if self.use_polars:
      print('Polars backend enabled for pipeline.', flush=True)
    else:
      print('Using pandas backend for pipeline.', flush=True)

    self._log_step(step, total_steps, 'Initialising data manager...')
    t1 = perf_counter()
    data_manager = DataManager(self.dataset, mode=self.mode, use_polars=self.use_polars)
    self._log_duration(t1)
    step += 1

    catalog = None
    if self.mode in {'ahf', 'ahf-fast'}:
      self._log_step(step, total_steps, 'Loading AHF catalogue...')
      t1 = perf_counter()
      catalog = load_catalog(str(self.ahf_particles), str(self.ahf_halos) if self.ahf_halos else None)
      self._log_duration(t1)
      step += 1

    if self.mode == 'fof':
      step = self._run_fof(data_manager, step, total_steps)
    elif self.mode == 'ahf':
      step = self._run_ahf(data_manager, catalog, step, total_steps, self.nproc)
    else:
      step = self._run_ahf_fast(data_manager, catalog, step, total_steps, self.nproc)

    self._log_step(step, total_steps, 'Calculating group properties...')
    t1 = perf_counter()
    calculate_group_properties(data_manager, use_polars=self.use_polars)
    self._log_duration(t1)
    step += 1

    self._log_step(step, total_steps, 'Saving datasets...')
    t1 = perf_counter()
    if 'halos' in data_manager:
      data_manager['halos'].fillna(0., inplace=True)
    if 'galaxies' in data_manager:
      data_manager['galaxies'].fillna(0., inplace=True)
    saver = Saver(file)
    saver.save_data(data_manager)
    self._log_duration(t1)

  def member_search(self, file: PathLike, *args, **kwargs):
    print('member_search() is deprecated; use run() instead.', flush=True)
    self.run(file, *args, **kwargs)

  def _run_fof(self, data_manager: DataManager, step: int, total_steps: int) -> int:
    self._log_step(step, total_steps, 'Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    self._log_duration(t1)
    step += 1

    self._log_step(step, total_steps, 'Running FOF6D...')
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=self.nproc)
    self._log_duration(t1)
    step += 1
    return step

  def _run_ahf(self, data_manager: DataManager, catalog, step: int, total_steps: int, n_jobs: int) -> int:
    self._log_step(step, total_steps, 'Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    self._log_duration(t1)
    step += 1

    self._log_step(step, total_steps, 'Running FOF6D...')
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=self.nproc)
    self._log_duration(t1)
    step += 1

    self._log_step(step, total_steps, 'Matching FoF galaxies to AHF halos...')
    t1 = perf_counter()
    halo_map, missing = apply_ahf_matching(data_manager, catalog, n_jobs=n_jobs)
    self._log_duration(t1)
    matched = len(halo_map)
    merged_halos = sum(1 for count in halo_map.values() if count > 1)
    merged_galaxies = sum((count - 1) for count in halo_map.values() if count > 1)
    total_galaxies = sum(halo_map.values())
    print(
      f"    Matched {matched} AHF halos covering {total_galaxies} FoF galaxies "
      f"(missing particles: {missing}; merged {merged_galaxies} FoF galaxies across {merged_halos} halos).",
      flush=True
    )
    step += 1
    return step

  def _run_ahf_fast(self, data_manager: DataManager, catalog, step: int, total_steps: int, n_jobs: int) -> int:
    self._log_step(step, total_steps, 'Building halos/galaxies from AHF-FAST catalogue...')
    t1 = perf_counter()
    missing = build_galaxies_from_fast(data_manager, catalog, n_jobs=n_jobs)
    self._log_duration(t1)
    print(f"    Assigned particles with missing counts: {missing}", flush=True)
    step += 1

    self._log_step(step, total_steps, 'Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    self._log_duration(t1)
    step += 1
    return step
