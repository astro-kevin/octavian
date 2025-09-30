from os import PathLike
from time import perf_counter
from typing import Optional

from tqdm.auto import tqdm

from octavian.constants import code_units
from octavian.data_manager import DataManager
from octavian.group_funcs import calculate_group_properties
from octavian.saver import Saver
from octavian.utils import wrap_positions
from octavian.fof6d import run_fof6d
from octavian.ahf import load_catalog, apply_ahf_matching, build_galaxies_from_fast


class OCTAVIAN:
  def __init__(self, dataset: PathLike, units: dict = {}, nproc: int = 1, mode: str = 'fof', ahf_particles: Optional[PathLike] = None, ahf_halos: Optional[PathLike] = None, *args, **kwargs):
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

  def _log_step(self, current: int, total: int, message: str, stage_progress: Optional[tqdm] = None) -> None:
    if stage_progress is not None:
      stage_progress.set_description(f"[{current}/{total}] {message}")
      stage_progress.refresh()
    print(f"[{current}/{total}] {message}", flush=True)

  def _log_duration(self, start_time: float, stage_progress: Optional[tqdm] = None) -> None:
    elapsed = perf_counter() - start_time
    if stage_progress is not None:
      stage_progress.set_postfix_str(f"{elapsed:.2f}s")
    print(f"    Done in {elapsed:.2f} seconds.", flush=True)

  def run(self, file: PathLike, *args, **kwargs):
    if self.mode == 'fof':
      total_steps = 5
    elif self.mode == 'ahf':
      total_steps = 7
    else:
      total_steps = 6

    step = 1
    with tqdm(total=total_steps, desc='Octavian Pipeline', unit='stage', leave=True) as stage_progress:
      self._log_step(step, total_steps, 'Initialising data manager...', stage_progress)
      t1 = perf_counter()
      data_manager = DataManager(self.dataset, mode=self.mode)
      self._log_duration(t1, stage_progress)
      stage_progress.update(1)
      step += 1

      catalog = None
      if self.mode in {'ahf', 'ahf-fast'}:
        self._log_step(step, total_steps, 'Loading AHF catalogue...', stage_progress)
        t1 = perf_counter()
        catalog = load_catalog(str(self.ahf_particles), str(self.ahf_halos) if self.ahf_halos else None)
        self._log_duration(t1, stage_progress)
        stage_progress.update(1)
        step += 1

      if self.mode == 'fof':
        step = self._run_fof(data_manager, step, total_steps, stage_progress)
      elif self.mode == 'ahf':
        step = self._run_ahf(data_manager, catalog, step, total_steps, self.nproc, stage_progress)
      else:
        step = self._run_ahf_fast(data_manager, catalog, step, total_steps, self.nproc, stage_progress)

      self._log_step(step, total_steps, 'Calculating group properties...', stage_progress)
      t1 = perf_counter()
      calculate_group_properties(data_manager)
      self._log_duration(t1, stage_progress)
      stage_progress.update(1)
      step += 1

      self._log_step(step, total_steps, 'Saving datasets...', stage_progress)
      t1 = perf_counter()
      if 'halos' in data_manager:
        data_manager['halos'].fillna(0., inplace=True)
      if 'galaxies' in data_manager:
        data_manager['galaxies'].fillna(0., inplace=True)
      saver = Saver(file)
      saver.save_data(data_manager)
      self._log_duration(t1, stage_progress)
      stage_progress.update(1)

  def member_search(self, file: PathLike, *args, **kwargs):
    print('member_search() is deprecated; use run() instead.', flush=True)
    self.run(file, *args, **kwargs)

  def _run_fof(self, data_manager: DataManager, step: int, total_steps: int, stage_progress: Optional[tqdm]) -> int:
    self._log_step(step, total_steps, 'Wrapping positions...', stage_progress)
    t1 = perf_counter()
    wrap_positions(data_manager)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    step += 1

    self._log_step(step, total_steps, 'Running FOF6D...', stage_progress)
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=self.nproc)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    step += 1
    return step

  def _run_ahf(self, data_manager: DataManager, catalog, step: int, total_steps: int, n_jobs: int, stage_progress: Optional[tqdm]) -> int:
    self._log_step(step, total_steps, 'Wrapping positions...', stage_progress)
    t1 = perf_counter()
    wrap_positions(data_manager)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    step += 1

    self._log_step(step, total_steps, 'Running FOF6D...', stage_progress)
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=self.nproc)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    step += 1

    self._log_step(step, total_steps, 'Matching FoF galaxies to AHF halos...', stage_progress)
    t1 = perf_counter()
    halo_map, missing = apply_ahf_matching(data_manager, catalog, n_jobs=n_jobs)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    matched = len(halo_map)
    print(f"    Matched {matched} AHF halos (missing particles: {missing}).", flush=True)
    step += 1
    return step

  def _run_ahf_fast(self, data_manager: DataManager, catalog, step: int, total_steps: int, n_jobs: int, stage_progress: Optional[tqdm]) -> int:
    self._log_step(step, total_steps, 'Building halos/galaxies from AHF-FAST catalogue...', stage_progress)
    t1 = perf_counter()
    missing = build_galaxies_from_fast(data_manager, catalog, n_jobs=n_jobs)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    print(f"    Assigned particles with missing counts: {missing}", flush=True)
    step += 1

    self._log_step(step, total_steps, 'Wrapping positions...', stage_progress)
    t1 = perf_counter()
    wrap_positions(data_manager)
    self._log_duration(t1, stage_progress)
    if stage_progress is not None:
      stage_progress.update(1)
    step += 1
    return step
