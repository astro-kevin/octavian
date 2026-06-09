import os
from pathlib import Path

import pytest

if os.environ.get('OCTAVIAN_RUN_LOCAL_INTEGRATION') != '1':
    pytest.skip(
        'local full-pipeline integration test; set OCTAVIAN_RUN_LOCAL_INTEGRATION=1 to run',
        allow_module_level=True,
    )

from mpi4py import MPI
import octavian

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nsplit = comm.Get_size()

path_to_snapshot = Path('/disk04/rad/sim/m100n1024/s50/snap_m100n1024_151.hdf5')
path_to_filtered_output = Path('/home/jpduminy/octavian-analysis/split-snapshot')
path_to_filtered_analysis = Path('/home/jpduminy/octavian-analysis/finished-snapshot')
path_to_output = Path('/home/jpduminy/octavian-analysis')
path_to_config = Path('/home/jpduminy/octavian_scripts/octavian-1/config.yaml')
missing = [str(path) for path in (path_to_snapshot, path_to_output, path_to_config) if not path.exists()]
if missing:
    pytest.skip(
        f'local full-pipeline integration inputs are unavailable: {missing}',
        allow_module_level=True,
    )

if rank == 0:
    import caesar
    print(f"caesar imported successfully.")
    import pygadgetreader
    print(f"pygadgetreader imported successfully.")
    pytest.importorskip('fsps')
    print(f"fsps imported successfully.")
    import octavian
    print(f"octavian imported successfully.")
    print(f"Packages imported successfully!")

    # remove files from previous runs
    for i in range(nsplit):
        old_file = f"{path_to_filtered_output}_{i}.hdf5"
        if os.path.exists(old_file):
            os.remove(old_file)

    print(f"Attempting to filter snapshot...")
    octavian.filter_snapshot(str(path_to_snapshot), str(path_to_filtered_output), nsplit=nsplit)
    print("Filtering complete.")

comm.Barrier()

octavian.mpirun(str(path_to_filtered_output), str(path_to_filtered_analysis), str(path_to_config))

if rank == 0:
    print(f"Success.")
