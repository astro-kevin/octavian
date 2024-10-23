from __future__ import annotations
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from data_manager import DataManager


# handle periodic boundary for fof and CoM calculations, reset out of bounds positions during save
def wrap_positions(data_manager: DataManager) -> None:
  boxsize = data_manager.simulation['boxsize']/data_manager.simulation['h']

  # select halos with particles near boundary
  halos_to_wrap = {}
  for ptype in ['gas', 'dm', 'star', 'bh']:
    data_manager.load_property('pos', ptype)

  data = pd.concat([data_manager[ptype] for ptype in ['gas', 'dm', 'star', 'bh']])
  halos_grouped = data.groupby(by='HaloID')
  for direction in ['x', 'y', 'z']:
    check_wrap = (halos_grouped[direction].max() - halos_grouped[direction].min()) > 0.5*boxsize
    halos_to_wrap[direction] = check_wrap[check_wrap].index.unique()

  # wrap positions
  for ptype in ['gas', 'dm', 'star', 'bh']:
    halos_grouped = data_manager[ptype].groupby(by='HaloID')
    for direction in ['x', 'y', 'z']:
      in_halos_to_wrap = np.isin(data_manager[ptype]['HaloID'], halos_to_wrap[direction])
      too_high = data_manager[ptype][direction] > 0.5*boxsize

      data_manager[ptype].loc[in_halos_to_wrap & too_high, direction] -= boxsize
