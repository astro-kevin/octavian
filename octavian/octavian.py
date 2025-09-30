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

  def member_search(self, file: PathLike, *args, **kwargs):
    print('Initialising Data Manager...')
    t1 = perf_counter()
    data_manager = DataManager(self.dataset, mode=self.mode)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    if self.mode == 'fof':
      self._run_fof(data_manager)
    elif self.mode == 'ahf':
      catalog = load_catalog(str(self.ahf_particles), str(self.ahf_halos) if self.ahf_halos else None)
      self._run_ahf(data_manager, catalog)
    else:  # AHF-FAST
      catalog = load_catalog(str(self.ahf_particles), str(self.ahf_halos) if self.ahf_halos else None)
      self._run_ahf_fast(data_manager, catalog)

    print('Calculating group properties...')
    t1 = perf_counter()
    calculate_group_properties(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    print('Saving datasets...')
    t1 = perf_counter()
    if 'halos' in data_manager:
      data_manager['halos'].fillna(0., inplace=True)
    if 'galaxies' in data_manager:
      data_manager['galaxies'].fillna(0., inplace=True)
    saver = Saver(file)
    saver.save_data(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

  def _run_fof(self, data_manager: DataManager) -> None:
    print('Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    print('Running FOF6D...')
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=self.nproc)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

  def _run_ahf(self, data_manager: DataManager, catalog) -> None:
    print('Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    print('Running FOF6D...')
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=self.nproc)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    print('Matching FoF galaxies to AHF halos...')
    t1 = perf_counter()
    halo_map, missing = apply_ahf_matching(data_manager, catalog)
    t2 = perf_counter()
    matched = len(halo_map)
    print(f'Matched {matched} AHF halos in {t2-t1:.2f} seconds (missing particles: {missing}).')

  def _run_ahf_fast(self, data_manager: DataManager, catalog) -> None:
    print('Building halos/galaxies from AHF-FAST catalogue...')
    t1 = perf_counter()
    missing = build_galaxies_from_fast(data_manager, catalog)
    t2 = perf_counter()
    print(f'Assigned AHF-FAST payloads in {t2-t1:.2f} seconds (missing particles: {missing}).')

    print('Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')
