from types import SimpleNamespace

import numpy as np
import pandas as pd

from octavian.halo_finder.fof6d import get_mean_interparticle_separation, run_fof6d


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


class _NoStarDataManager:
    def __init__(self):
        self.config = {
            'ptypes': ['gas', 'dm'],
            'groups': ['halos', 'galaxies'],
        }
        self.data = {
            'gas': pd.DataFrame(index=np.arange(2)),
            'dm': pd.DataFrame(index=np.arange(3)),
        }
        self.logger = _Logger()
        self.loaded = []

    def load_property(self, prop, ptype):
        self.loaded.append((prop, ptype))
        if prop in ('mass', 'bhmass'):
            self.data[ptype]['mass'] = np.ones(len(self.data[ptype]), dtype=float)


def test_mean_interparticle_separation_uses_global_header_count_for_empty_dm_shard():
    data_manager = SimpleNamespace(
        simulation={
            'h': 0.68,
            'O0': 0.3,
            'boxsize': 25000.0,
            'num_part_total': np.asarray([16777216, 16777216, 0, 0, 0, 0], dtype=np.int64),
        },
        mdm_total=0.0,
        ndm=0,
        mgas_total=0.0,
        mstar_total=0.0,
        mbh_total=0.0,
    )

    get_mean_interparticle_separation(data_manager)

    assert data_manager.efres == 256
    assert np.isclose(data_manager.mis, 25000.0 / 0.68 / 256)


def test_run_fof6d_skips_no_star_shards():
    data_manager = _NoStarDataManager()

    run_fof6d(data_manager)

    assert data_manager.config['groups'] == ['halos']
    assert data_manager.data['gas']['GalID'].cat.categories.tolist() == [-1]
    assert data_manager.data['dm']['GalID'].cat.categories.tolist() == [-1]
    assert data_manager.data['gas']['GalID'].to_numpy().tolist() == [-1, -1]
    assert data_manager.data['dm']['GalID'].to_numpy().tolist() == [-1, -1, -1]
