from time import perf_counter

from octavian.data_manager import DataManager, save_group_properties
from octavian.utils import wrap_positions
from octavian.halo_finder import run_fof6d
from octavian.halo_reader import load_halo_source, load_halo_tree
from octavian.group_properties_calc import calculate_group_properties, get_particle_lists

from yaml import safe_load

def run(snapshot: str, outfile: str, configfile: str, logfile: str | None = None, comm=None):
  with open(configfile, 'r') as f:
    config = safe_load(f)

  config['Tlim'] = float(config['Tlim'])
  if logfile is None:
    logfile = f'{outfile}.log'

  data_manager = DataManager(snapshot, logfile, config, comm=comm)

  halo_source = config.get('halo_source')
  halo_mode = config.get('halo_mode', 'field')
  staged_subhalo_membership = halo_mode == 'subhalo' and bool(data_manager.halo_id_arrays)
  if halo_source and staged_subhalo_membership:
    load_halo_tree(data_manager, mode=halo_mode)
  elif halo_source:
    load_halo_source(data_manager, mode=halo_mode)

  wrap_positions(data_manager)
  
  run_fof6d(data_manager, nproc=config.get('nproc', 1))

  data_manager.initialise_group_data()
  calculate_group_properties(data_manager)
  
  get_particle_lists(data_manager)
  
  save_group_properties(data_manager, outfile)
  
