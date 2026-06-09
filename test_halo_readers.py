# test_ahf_reader.py
import os
from pathlib import Path
from time import perf_counter

import pytest
from yaml import safe_load

from octavian.data_manager import DataManager
from octavian.halo_reader.ahf import load_ahf

if os.environ.get('OCTAVIAN_RUN_LOCAL_INTEGRATION') != '1':
    pytest.skip(
        'local AHF reader integration test; set OCTAVIAN_RUN_LOCAL_INTEGRATION=1 to run',
        allow_module_level=True,
    )

snapshot = Path('/disk04/rad/sim/m100n1024/s50/snap_m100n1024_151.hdf5')
particles_path = Path('/home/jpduminy/octavian-snapshots/Simba_M200_snap_151.z0.000.AHF_particles')
halos_path = Path('/home/jpduminy/octavian-snapshots/Simba_M200_snap_151.z0.000.AHF_halos')
missing = [str(path) for path in (snapshot, particles_path, halos_path) if not path.exists()]
if missing:
    pytest.skip(
        f'local AHF reader integration inputs are unavailable: {missing}',
        allow_module_level=True,
    )

print(f"Beginning analysis.")

with open('config.yaml', 'r') as f:
    config = safe_load(f)
config['Tlim'] = float(config['Tlim'])
config['prop_aliases']['pid'] = 'ParticleIDs'

dm = DataManager(str(snapshot), 'test_halo_readers.log', config)

t1 = perf_counter()
load_ahf(dm, str(particles_path), halos_path=str(halos_path), mode='field')
t2 = perf_counter()
time_taken = (t2 - t1) / 60 # mins
print(f"AHF Data loaded. Total time: {time_taken:.3f} minutes.")

for ptype in dm.config['ptypes']:
    assigned = (dm.data[ptype]['HaloID'] != -1).sum()
    total = len(dm.data[ptype])
    fraction = (assigned / total) * 100
    print(f"Particle type breakdown:")
    print(f"    {ptype}: {assigned}/{total} particles assigned ({fraction:.2f} %)")

print(f'\nField halos: {len(dm.halo_tree._field_halos)}')
print(f'Total halos: {len(dm.halo_tree.halo_ids)}')
print(f'Max depth: {dm.halo_tree.depths.max()}')

print(f"\n Test: sample hids (star, first 20):")
print(dm.data['star']['HaloID'].head(20))
