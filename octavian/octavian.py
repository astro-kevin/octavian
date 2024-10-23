import numpy as np
import pandas as pd
from data_manager import DataManager
from saver import Saver
from constants import code_units
from os import PathLike
from time import perf_counter

from utils import wrap_positions
from fof6d import run_fof6d
from group_funcs import calculate_group_properties

class OCTAVIAN:
  def __init__(self, dataset: PathLike, units: dict = {}, nproc: int = 1, *args, **kwargs):
    self._args = args
    self.dataset = dataset

    self.units = code_units | units
    self.nproc = nproc

  def member_search(self, file: PathLike, *args, **kwargs):
    print('Initialising Data Manager...')
    t1 = perf_counter()
    data_manager = DataManager(self.dataset)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')
    
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

    print('Calculating group properties...')
    t1 = perf_counter()
    calculate_group_properties(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    print('Saving datasets...')
    t1 = perf_counter()
    data_manager['halos'].fillna(0., inplace=True)
    data_manager['galaxies'].fillna(0., inplace=True)
    saver = Saver(file)
    saver.save_data(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')