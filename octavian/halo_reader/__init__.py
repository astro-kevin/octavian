from __future__ import annotations

from importlib import import_module
from types import ModuleType

import h5py

from octavian.halo_reader.halo_utils import (
    membership_array_exclusive_ids,
    prune_halo_tree,
    read_staged_halo_tree,
    write_staged_halo_tree,
)


def _halo_source(config_or_source) -> str:
    if isinstance(config_or_source, str):
        source = config_or_source
    else:
        source = config_or_source.get('halo_source')
    if source is None:
        raise KeyError('halo_source must be set to use an external halo finder')
    return str(source).lower()


def get_reader(config_or_source) -> ModuleType:
    source = _halo_source(config_or_source)
    try:
        return import_module(f'octavian.halo_reader.{source}')
    except ModuleNotFoundError as exc:
        if exc.name == f'octavian.halo_reader.{source}':
            raise ValueError(f'Unsupported halo_source: {source}') from exc
        raise


def halo_source_metadata_schema(config: dict) -> dict | None:
    source = config.get('halo_source')
    if source is None:
        return None
    reader = get_reader(source)
    metadata_schema = getattr(reader, 'metadata_schema', None)
    return None if metadata_schema is None else metadata_schema()


def build_snapshot_membership_arrays(snapshot, config: dict):
    """Build source-agnostic per-particle halo ancestry arrays."""
    return get_reader(config).build_snapshot_membership_arrays(snapshot, config)


def load_halo_source(data_manager, mode='field'):
    """Load external halo finder assignments through the configured reader."""
    return get_reader(data_manager.config).load(data_manager, mode=mode)


def load_halo_tree(data_manager, mode='field'):
    """Load only the halo hierarchy for staged snapshots that already contain IDs."""
    source = _halo_source(data_manager.config)
    with h5py.File(data_manager.snapfile, 'r') as handle:
        tree = read_staged_halo_tree(handle)
    if tree is None:
        raise RuntimeError(
            'Staged halo tree missing from split snapshot; rerun filter_snapshot with a current Octavian version.'
        )
    data_manager.config['halo_source'] = source
    data_manager.config['halo_mode'] = mode
    data_manager.halo_tree = tree
    return tree
