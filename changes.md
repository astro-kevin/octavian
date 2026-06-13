# Octavian package changes since the last non-astro-kevin commit

Comparison base: `8371b6ede9f0b7e26200a01545172d804456fb6b` (`jszpila314`, 2026-04-30, `Load correct bhmass dataset`)

Compared range: `8371b6ede9f0b7e26200a01545172d804456fb6b..HEAD -- octavian`

Current package head at comparison time: `470367641ecb8e1e3969d7744d8d23cc01ed1e33` (`astro-kevin`, 2026-06-12, `Add optional metallicity staging controls`)

Scope: only files under `octavian/`. The package diff changes 18 files, with 5,183 insertions and 338 deletions.

## Changed files

| File | Diff size | Main difference | Justification |
| --- | ---: | --- | --- |
| `octavian/__init__.py` | +1/-0 | Exports `progen`. | Makes the new progenitor-linking API reachable from the package top level. |
| `octavian/run.py` | +13/-2 | Loads configured external halo sources before FOF6D and makes `logfile` optional. | Required so AHF/HBT assignments can replace or seed internal halo finding; the default logfile keeps old callers from needing to pass a path explicitly. |
| `octavian/data_manager/data_manager.py` | +61/-4 | Adds `halo_id_arrays`, cached halo membership rows, optional property loading, `HaloID_array` reading, top-level halo ID access, and high-word particle-count support. | Required for staged subhalo membership, external halo trees, missing optional datasets, and simulations with particle counts beyond 32-bit headers. |
| `octavian/data_manager/save_group_properties.py` | +55/-9 | Resolves output columns through metadata helpers, supports per-group column mappings, writes variable-length list datasets, and writes particle-list datasets without compression. | Required for halo-source-specific outputs such as ancestor lists. Removing compression favors faster intermediate writes; file-size impact should be checked if compressed output is a priority. |
| `octavian/group_properties_calc/calculate_group_properties.py` | +830/-91 | Reworks common, gas, star, BH, aperture, central-galaxy, and halo-source metadata calculations. | Required for CAESAR-compatible outputs and for correct aggregation when one particle can belong to a halo ancestry chain rather than exactly one halo. |
| `octavian/group_properties_calc/get_particle_lists.py` | +8/-2 | Uses `DataManager.get_halo_membership_rows()` for halo particle lists when ancestry arrays are present. | Required so halo particle lists include all relevant ancestor/subhalo memberships instead of only an exclusive `HaloID`. |
| `octavian/group_properties_calc/group_computations.py` | +664/-1 | Adds numba kernels for scalar sums, central flags, apertures, hydrogen assignment, radial `rmax`, and membership-array accumulations. | Required to keep the new group-property paths fast enough for large particle counts. |
| `octavian/group_properties_calc/group_helpers.py` | +24/-3 | Extends particle-array extraction to include dust masses and velocities. | Required by 30 kpc aperture masses and velocity-dispersion outputs, including optional dust/HI/H2 components. |
| `octavian/halo_filter/filter_snapshot.py` | +566/-0 | Adds external halo-source staging, rank routing, dataset selection, optional metallicity staging, scalar/dense membership handling, and staged halo-tree writing. | Required to split snapshots using AHF/HBT memberships without first writing global `HaloID` fields or loading unnecessary datasets. |
| `octavian/halo_finder/fof6d.py` | +63/-28 | Uses global DM counts when available, skips galaxy finding when no stars exist, tolerates missing DM, uses configured `nproc`, and guards timing summaries. | Required for more robust runs on staged or reduced snapshots; replacing the hard-coded `12` with `nproc` restores caller control. |
| `octavian/halo_reader/__init__.py` | +76/-3 | Replaces direct imports with reader dispatch, shared membership-helper exports, staged tree loading, and metadata-schema access. | Required to treat `ahf` and `hbt` as configurable halo sources behind one interface. |
| `octavian/halo_reader/ahf.py` | +272/-33 | Adds AHF tree reading, config-path helpers, metadata schema, streaming snapshot membership arrays, subhalo mode assignment, and typed halo parsing. | Required for scalable AHF staging and source-specific output metadata. |
| `octavian/halo_reader/ahf_parser.c` | +279/-1 | Adds C routines to stream AHF particle files into dense/scalar membership arrays with deterministic conflict handling. | Required to avoid Python-level parsing and sorting for very large AHF particle files. |
| `octavian/halo_reader/halo_utils.py` | +407/-19 | Adds `HaloBuildResult`, shared ID remapping, dense/sparse membership utilities, staged halo-tree read/write/prune helpers, and sparse-aware membership selection. | Required common infrastructure for both AHF and HBT staging paths. |
| `octavian/halo_reader/hbt.py` | +654/-89 | Adds split `SubSnap` discovery, multi-file readers, nested-parent hierarchy support, streaming HBT membership arrays, config wrappers, metadata schema, and subhalo mode assignment. | Required for HBT outputs that are split across files and for nested HBT substructure/progenitor workflows. |
| `octavian/progen.py` | +940/-0 | Adds progenitor linking for AHF and HBT halo/galaxy catalogues. | Required to produce `progen_halos` and `progen_galaxies` links from halo-finder merger data and star-particle overlap. |
| `octavian/utils/dataset_columns.py` | +55/-0 | Adds helpers to resolve normal and list output columns, including conditional entries by halo source and halo mode. | Required so AHF/HBT and field/subhalo modes can enable different catalogue datasets without duplicating save/merge logic. |
| `octavian/utils/merge_catalogues.py` | +215/-53 | Makes catalogue merging metadata-driven, remaps halo indices after mass sorting, handles central-galaxy IDs, supports list CSR datasets, and handles empty galaxy shards. | Required so new source metadata and variable-length ancestor/progenitor lists remain valid after rank outputs are merged. |

## Functional differences and justifications

### External halo-source dispatch

- `octavian/halo_reader/__init__.py` now exposes `get_reader()`, `build_snapshot_membership_arrays()`, `load_halo_source()`, `load_halo_tree()`, and `halo_source_metadata_schema()`.
  - Justification: AHF and HBT now share the same config-driven entry points. This prevents `run.py`, `filter_snapshot.py`, and merge/save code from hard-coding reader-specific imports.
- `octavian/run.py` checks `config["halo_source"]` and `config["halo_mode"]` before `run_fof6d()`.
  - Justification: External halo catalogues must be loaded before group finding and property calculation. In staged subhalo mode, only the pruned halo tree needs to be loaded because particle memberships are already in the split snapshot.
- `octavian/__init__.py` exports `progen`.
  - Justification: This is a public API addition for the new progenitor-linking subsystem.

### DataManager and membership model

- `DataManager` now keeps `halo_id_arrays` and `halo_membership_rows`.
  - Justification: A single categorical `HaloID` is insufficient for subhalo mode, where a particle may contribute to a top halo and one or more descendants.
- `DataManager.load_data()` reads `HaloID_array` when present and synthesizes `HaloID = -1` if a staged particle group has no scalar `HaloID`.
  - Justification: Staged snapshots can carry ancestry arrays or no assigned halo for some particles. The fallback keeps downstream group operations well-defined.
- `DataManager.get_halo_ids(ptype, mode="exclusive"|"top")` provides either deepest/exclusive IDs or top-level IDs.
  - Justification: Halo and galaxy calculations need both views: subhalo outputs need exclusive IDs, while central-host and field-halo calculations need top-level IDs.
- `DataManager.load_property(..., optional=True, default=...)` was added.
  - Justification: Optional datasets such as dust masses should not fail runs when absent.
- `NumPart_Total_HighWord` is included when computing simulation particle totals.
  - Justification: Large simulations can exceed 32-bit `NumPart_Total`; FOF6D mean-spacing estimates need the full count.

### Snapshot staging

- `filter_snapshot()` now branches to external halo-source staging when `halo_source` is configured.
  - Justification: AHF/HBT can provide halo memberships directly; rerunning FOF6D or expecting a pre-existing global `HaloID` dataset is unnecessary.
- `filter_snapshot_with_membership_arrays()` balances split snapshots by top-level halo membership counts and writes per-rank `HaloID`, `particle_index`, and optional `HaloID_array`.
  - Justification: Rank routing keeps all particles for the same top-level halo together while balancing the expensive FOF6D/group-property workload.
- Staging writes a pruned `OctavianHaloTree` into each output shard.
  - Justification: Later stages can recover source hierarchy and metadata without rereading full halo catalogues.
- Staging now has configurable dataset selection through `_STAGING_REQUIRED_PROPS`, `staging_properties`, `staging_rank_order_properties`, `include_metallicities`, and interval-based HDF5 reads.
  - Justification: Large snapshots cannot always afford full dataset reads. Optional metallicity staging reduces I/O when only the first metallicity column is needed.
- Scalar and dense membership arrays are both supported.
  - Justification: AHF/HBT staging can represent memberships compactly as scalar encoded deepest halo IDs plus ancestry arrays, while staged subhalo snapshots may carry dense ancestry arrays.

### Halo-reader shared infrastructure

- `HaloBuildResult` packages the halo tree, membership arrays, counts, cached datasets, and optional ancestry arrays.
  - Justification: Staging needs more than the old flat membership triples, and readers need a common return type.
- Membership helper functions now handle dense arrays, scalar encoded arrays, and scipy sparse arrays.
  - Justification: This allows reader-specific memory layouts without changing downstream staging and calculation code.
- `write_staged_halo_tree()`, `read_staged_halo_tree()`, and `prune_halo_tree()` were added.
  - Justification: Sharded staged snapshots need a small self-contained halo tree, not the full original external catalogue.
- ID remapping was moved into a shared `remap_halo_ids()` helper and reused by AHF/HBT.
  - Justification: The same compact `0..N-1` internal ID convention is needed across readers.

### AHF integration

- `read_ahf_halos()` now uses explicit integer dtypes for `ID` and `hostHalo`.
  - Justification: Prevents large AHF IDs from being coerced incorrectly and makes remapping deterministic.
- AHF now has `read_ahf_tree()`, `read_ahf_membership()`, `_paths_from_config()`, `metadata_schema()`, `read_tree()`, `build_snapshot_membership_arrays()`, and `load()`.
  - Justification: These provide the same reader contract as HBT and expose source-specific metadata columns such as `AHF_haloID`, `AHF_parent_haloID`, and `AHF_ancestor_haloIDs`.
- `build_ahf_snapshot_membership_arrays()` streams AHF particle data through the compiled parser into per-particle memberships.
  - Justification: This avoids building huge Python arrays and sorting them for every staged run.
- `load_ahf(..., mode="subhalo")` now populates `DataManager.halo_id_arrays` and scalar `HaloID` values from membership arrays.
  - Justification: Subhalo mode needs ancestry-aware aggregation while keeping older scalar consumers functional.
- `ahf_parser.c` now includes membership-array fill functions and conflict resolution that prefers deeper membership, then lower compact halo ID on ties.
  - Justification: AHF can contain duplicate or overlapping particle assignments. Deterministic conflict handling makes repeated runs reproducible.

### HBT integration

- HBT file discovery now accepts direct files, snapshot directories, simulation roots with `hbt_snap_index`, and split files such as `SubSnap_050.0.hdf5`.
  - Justification: HBT outputs are not always one file per snapshot.
- `read_subhalos()` reads a limited set of scalar fields and can concatenate multiple SubSnap files.
  - Justification: Avoids loading unnecessary structured fields and supports split outputs.
- `build_parent_ids()` now prefers `NestedParentTrackId` and falls back to `HostHaloId`/`Rank`.
  - Justification: `NestedParentTrackId` better preserves immediate nested subhalo hierarchy when available.
- `build_hbt_snapshot_membership_arrays()` streams HBT particle memberships in chunks through particle-ID lookup tables.
  - Justification: HBT particle lists can be large and lack particle types, so chunked PID cross-referencing is required for memory control.
- `load_hbt(..., mode="subhalo")` now populates ancestry arrays in the `DataManager`.
  - Justification: Keeps subhalo-mode calculations consistent with AHF and staged snapshots.
- HBT now exposes `metadata_schema()`, `read_tree()`, `build_snapshot_membership_arrays()`, and `load()`.
  - Justification: Provides the same config-driven interface and output metadata naming as AHF.

### FOF6D behavior

- FOF6D uses global dark-matter particle counts from snapshot headers when available.
  - Justification: Mean interparticle spacing should use global resolution rather than only the particles present in a filtered shard.
- FOF6D skips galaxy finding when there are no star particles and marks `GalID = -1`.
  - Justification: Star-free inputs cannot form galaxies under the current algorithm; skipping avoids crashes and removes `galaxies` from config groups.
- The joblib worker count now uses the caller's `nproc` instead of a hard-coded `12`.
  - Justification: Honors the configured resource allocation.
- Timing-summary output is guarded for empty timing data.
  - Justification: Avoids failures on no-op or star-free runs.

### Group-property calculations

- Common group properties now support halo ancestry arrays through `_common_halo_array_properties()`.
  - Justification: Halo totals, centers, velocity dispersions, angular momentum, radial quantiles, and virial masses must account for all memberships in subhalo mode.
- Calculations initialize `n*` and `mass_*` columns to zero before early returns.
  - Justification: Empty groups should still have stable output schemas.
- Division now uses `_safe_divide()` in many places.
  - Justification: Empty or zero-mass groups should produce controlled fill values rather than warnings, infinities, or NaNs where zeros are expected.
- Halo-source metadata is assigned after property calculation.
  - Justification: Outputs can preserve source IDs, parent IDs, top IDs, depths, CAESAR index mappings, and ancestor lists for AHF/HBT catalogues.
- Central galaxies are assigned by the most massive stellar galaxy in each host halo.
  - Justification: `central_galaxy` and `central` are required CAESAR-compatible relationships; in subhalo mode the host is the top-level halo.
- Gas calculations now distinguish `mass_HI`/`mass_H2` from `mass_HI_ism`/`mass_H2_ism`, gate H2 by `nHlim`, include optional dust mass/counts, and compute CGM sums through numba kernels.
  - Justification: Matches the intended CAESAR hydrogen filters while preserving an ISM-style mass view and improving performance.
- Galaxy HI/H2 masses are assigned by host halo and nearest/mass-weighted candidate galaxy.
  - Justification: Gas is associated to galaxies after host assignment instead of assuming direct gas `GalID` is always sufficient.
- Star calculations add `sfr_100` from young stellar mass.
  - Justification: Provides a CAESAR-style recent star-formation proxy.
- BH calculations now output zeros for groups without black holes and support membership-array aggregation.
  - Justification: Avoids NaN-only empty outputs and keeps subhalo-mode BH accretion properties defined.
- Aperture calculations now include optional HI, H2, dust, total, and baryon components and add velocity-dispersion outputs for 30 kpc apertures.
  - Justification: Expands CAESAR-compatible aperture properties and keeps velocities loaded until aperture calculation finishes.
- Aperture KDTree queries now use periodic wrapping and fallback when `workers` is unavailable.
  - Justification: Corrects edge behavior in periodic boxes and keeps compatibility with older scipy versions.

### Low-level group computation kernels

- `group_computations.py` adds `compute_radial_quantiles_and_rmax()`.
  - Justification: `radius_*_rmax` is now emitted alongside `r20`, half-mass, and `r80`.
- New gas/star/BH scalar kernels replace repeated Python/pandas reductions.
  - Justification: Reduces overhead for repeated group sums on large arrays.
- New membership-array kernels accumulate common properties across all halo ancestry columns.
  - Justification: Vectorized groupby logic for one scalar `HaloID` cannot represent subhalo ancestry.
- New aperture and galaxy-hydrogen kernels were added.
  - Justification: These paths perform many local reductions and benefit from numba loops.

### Output schema and merging

- `dataset_columns.py` resolves `dataset_columns`, `dataset_columns_by_halo_source`, `dataset_columns_by_halo_mode`, and equivalent list-dataset entries.
  - Justification: AHF/HBT and field/subhalo modes need different output columns without scattering conditionals through save/merge code.
- `save_group_properties.py` supports dataset mappings that differ for `halos` and `galaxies`.
  - Justification: Some metadata columns are only meaningful for one group type.
- `save_group_properties.py` writes object/list columns as CSR-style `*_indices`, `*_offsets`, and `*_lengths`.
  - Justification: Ancestor IDs and progenitor IDs are variable-length per row and cannot be stored as simple dense arrays when `progenitors="all"`.
- `merge_catalogues.py` now validates unique halo `groupID` values and remaps parent/central/halo-source index columns after mass sorting.
  - Justification: Shard-local indices are invalid after concatenation and sorting; remapping preserves referential integrity.
- `merge_catalogues.py` now handles missing per-shard datasets with typed empty fills and supports shards with no galaxies.
  - Justification: External halo staging and no-star runs can produce partial schemas or empty galaxy groups.
- Particle-list CSR merging is now generic and reused for configured list datasets.
  - Justification: The same reorder-by-output-order logic is needed for `glist`/`slist` and new variable-length metadata datasets.

### Progenitor linking

- `octavian/progen.py` adds `progen()`, `check_if_progen_is_present()`, `get_progen_redshift()`, `read_progens()`, and `wipe_progen_info()`.
  - Justification: Provides a package-level workflow for linking adjacent Octavian catalogues and managing stored progenitor data.
- AHF progenitor candidates are read from `AHF_mtree_idx` and `AHF_mtree`, with mass-based ordering for multiple candidates.
  - Justification: Uses native AHF merger-tree products and ranks candidates reproducibly.
- HBT progenitor candidates are read from `DescendantTrackId`, with stable same-track IDs included when present.
  - Justification: HBT tracks can persist across snapshots, so the same `TrackId` should remain a preferred candidate when available.
- Halo progenitors are mapped from source IDs back to Octavian output IDs.
  - Justification: Stored results need to reference merged Octavian catalogue IDs, not raw halo-finder IDs.
- Galaxy progenitors use host-halo candidates plus star-particle overlap, with configurable `min_in_common`.
  - Justification: Galaxy identity is better tracked by shared stars within plausible host-halo progenitors than by halo IDs alone.
- Results are written under `tree_data` as dense arrays for one/fixed-N progenitors or CSR datasets for all progenitors.
  - Justification: Supports both fixed-width and variable-width progenitor outputs.

## Notes

- The package retains the older scalar `HaloID` paths where possible. New ancestry-aware paths activate when staged `HaloID_array` data or external halo-source membership arrays are present.
- The largest behavioral changes are in external halo-source support, subhalo membership handling, CAESAR-compatible output expansion, and catalogue merge remapping.
- No files under `octavian/` were deleted. Two new files were added: `octavian/progen.py` and `octavian/utils/dataset_columns.py`.
