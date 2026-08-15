# Changelog

All notable changes to this project are documented in this file.



## [Unreleased]

## [0.10.0] - 2026-08-15

### Fixed

- **Integer / RangeIndex gene names** with `balance_method="none"` wrote
  `n_highly_variable_final=k` but left `var["highly_variable"]` all False.
  Ranking helpers now stringify names, matching the append path.
- **`flavor="seurat"` / `"cell_ranger"` + `batch_key`**: mixed-sign
  `dispersions_norm` could invert scanpy's nbatches-primary merge order.
- **Log flavors no longer re-log** a staged non-integer matrix (already-log
  `.X` without `layers["counts"]`, or `seurat_v3` fallback to seurat).
- **Non-integer leftover `layers["counts"]` + integer `.X`** no longer feeds
  the log layer into HVG; `.X` is staged on an internal layer.
- **`store_raw` snapshots** refresh when counts change. On-disk files with the
  same path are rewritten unless shape, names, and fingerprint still match.
- **`store_raw=True` + `inplace=False`** no longer writes a snapshot on a
  discarded copy (warns and skips).
- **`restore_raw_counts(full_genes=True)`** does not let a same-size snapshot
  hide a larger `adata.raw`. Missing on-disk files fall back to the counts
  layer instead of raising. `prefer_snapshot` is honored on the full-gene path.
- Failed HVG calls now drop a scanpy `uns["hvg"]` they created.
- **`pearson_residuals`** forwards `batch_key`.
- Auto-n: `TRUE_SHORT_ND_MIN` is 11 so nd=10 cannot be both "few cores" and
  true-SHORT. Unanimous buffered-SHORT (k=1000) no longer overrides an
  aggregate mid/long list on large n. Failed structure seeds no longer inject
  fake `n_density_pops` into the median.
- `n_leiden` for the v7 ratio uses size-filtered clusters, not Leiden dust.
- Missing-label tokens (`NA`, `n/a`, `unknown`) no longer inflate `n_types`
  into fine mode.
- Fine mode floors every k&lt;2000, not only the SHORT rungs.
- Density merge no longer divides by a zero-height peak. User `uns["umap"]`
  is copied before the 3D UMAP used for population count.

### Documentation

- Cite the bioRxiv preprint (doi:10.64898/2026.08.08.743679) in the README,
  docs, and `CITATION.cff`.
- Recut the published docs and tutorials against the current API: drop leftover
  allocation / product wording, match current `uns` keys, and shorten the
  notebooks to a scanpy-style analysis path. Re-executed both tutorials on
  0.9.0 (stored outputs had been 0.5.0 / 0.7.0).

## [0.9.0] - 2026-08-12

### Fixed

- **seurat_v3 loess guard** now also checks `n_obs` (1-cell matrices SIGSEGV).
  Structure auto no longer calls `seurat_v3` on unsafe shapes.
- **`label_key` type count** drops missing / empty / `"nan"` labels, matching
  `diagnose_from_labels` (missing values no longer flip fine mode or true-SHORT).
- **`restore_raw_counts`** aligns `adata.raw` by gene name after HVG subset;
  `full_genes=True` uses `.raw` as the full universe when no snapshot exists.
- Failed HVG no longer leaves a newly created `layers["counts"]`.
- Missing `marker_genes` emit `UserWarning`. `inplace=False, subset=True`
  warns that subset is ignored.

### Removed

- Dead leftover from cluster-aware HVG: Leiden resolution bisection
  (`resolution_from_density_field`, `resolution_for_n_clusters`), PC-elbow
  diagnosis, no-op `_discard_raw_snapshot`, and `scfair_hvg_clusters` writes.
- Deprecated top-level HVG option kwargs. Secondary knobs only via
  `options=HVGOptions(...)`. Removed names still raise `TypeError`.

### Documentation

- Tighten published docs and source comments: American English, drop leftover
  hybrid/fairness wording, and remove claims that do not match the code
  (`uns` timings; default `raw_snapshot` discard).
- Quickstart and FAQ rewritten around copy-paste recipes (`HVGOptions`,
  restore from `.raw`, `inplace=False` + `subset=True`).

## [0.8.0] - 2026-08-08

### Changed

- **Default remains `n_top_genes="auto"`.** Structure fall-through no longer
  hard-codes 2000: when no SHORT/LONG/MID rule fires, base k scales as
  `round(n_density_pops × 150)` (`nd_budget`). Classical 2000 only when `nd`
  is missing; low-confidence floors still apply. Pass a fixed int for locked
  protocols.
- **`append` documented as same-rank list buffer** (`top-(k+m)`), not
  population-aware reallocation. Density population count
  (`estimate_n_populations`) documented as a first-class feature (scope
  ~≤20 well-separated populations).
- **Public `restore_raw_counts`** for `store_raw=True` snapshots.
- **`pp.__all__`** is the documented surface only.
- **Removed dead auto-n strategies** (elbow/knee/cumfrac/… dispatcher);
  `auto_n_method` option removed.
- Dropped `py.typed` (no mypy CI / type guarantee). Gitignore root
  `PKG-INFO` / `setup.cfg` egg_info noise.
- Structure embedding prep uses a slim AnnData; stable gene ranking;
  `n_top_genes=True` rejected; `mode=None` → `"auto"`;
  diagnose_from_labels reports `n_labels_dropped`.

### Fixed

- **Counts fingerprint no longer treats dense vs sparse as different content.**
  Format-only changes (e.g. ``layers['counts'] = X.copy()`` then densify
  ``.X``) no longer bypass the user's counts layer via a false mismatch.
- **Backed AnnData** raises a clear ``NotImplementedError`` (use
  ``to_memory()``) instead of a numpy ``isfinite`` TypeError.
- **Integer-counts check** full-scan threshold raised to 1e8 nnz; above that,
  strided sampling always pins first/mid/last entries (docstring no longer
  overclaims full coverage when subsampled).
- **`batch_key` selection matches scanpy.** After the per-batch HVG merge,
  ranking now follows scanpy's criteria (`highly_variable_rank` /
  `highly_variable_nbatches` / `dispersions_norm` as appropriate), not the
  cross-batch mean of ``variances_norm``. Previously up to ~36% of selected
  genes could differ, including genes with ``nbatches==0``.
- **No fake `layers['counts']` from log `.X`.** When there is no integer counts
  layer and `.X` is non-integer, HVG stages data on internal `_scfair_counts`
  (popped after the call) instead of permanently writing log values as
  `layers['counts']`. Emits `UserWarning`.
- **`store_raw=True` snapshots survive later default calls.** A subsequent
  `store_raw=False` HVG no longer deletes `uns['scfair']['raw_snapshot']`.
- **Structure auto LONG branch** caps `hi` at `min(k_max, ⌊0.5 · n_genes⌋)` so
  small gene pools no longer return `k = n_vars` via `long_shallow_few_cores`.
- **`estimate_n_populations` without neighbors** uses `warnings.warn` (visible
  by default) and documents the `sc.pp.neighbors` precondition.
- Auto progress threshold lowered to `n_obs >= 1000`; docs note auto cost
  (~10–100× vs fixed `n_top_genes`).
- `HVGOptions.merged` applies only non-`None` overrides; options/legacy mix
  error text matches “do not mix” policy; counts validation can record multiple
  warning codes; log-flavor materialization uses a minimal AnnData (lower peak
  memory); ruff `extend-exclude` for `docs/`.

## [0.7.0] - 2026-08-04

### Removed

- Cluster-aware `balance_method` values **`hybrid`**, **`score`**, and
  **`reweight`** (and aliases). Product surface is **`append`** (default) and
  **`none`** only.
- Related knobs: `blend_global`, `neighbor_contrast`, `combine`,
  `cluster_pool` / `cluster_genes`, `allocation_method`,
  `spec_on_legitimate_units`, intermediate `resolution` /
  `min_cluster_size` on the HVG call, and ensemble auto strategies
  (`elbow` / `knee` / `cumfrac` / … as `n_top_genes` or `auto_n_method`).
  Product auto is **structure only**.

### Changed

- HVG implementation and tests slimmed to the product path (global ranking +
  append / none + structure auto + MT/ribo filters).
- Structure auto keeps pair-stability helpers used by density rules; if
  structure still fails, fall back to base `2000` with an explicit
  `auto_message` (not a silent “from data structure” line).
- **Default append budget (tight density rule):** floor **200** (never
  reduced). When `n_top_genes="auto"` and `append_budget` is left `None`,
  budget may rise with structure `n_density_pops`:
  `m = max(200, min(300, 200 + max(0, n_need − 12) × 12))`. Fixed-k calls
  without a user budget stay at 200. Override with
  `options=HVGOptions(append_budget=N)`.
- **Default `n_top_genes="auto"`** — structure-aware base list size, then
  default `append`. Pass `n_top_genes=2000` for a fixed budget; use
  `n_top_genes=2000, balance_method="none"` to match a single global scanpy
  HVG pass.
- Structure auto: when density confidence is low, prefer a classical base of
  **2000** over short lists (except multi-core short geometry with enough
  labeled types, which may keep a shorter base).
- Write an inline `uns["scfair"]["raw_snapshot"]` only when
  `options=HVGOptions(store_raw=True)` (or `"ondisk"`); later default calls do
  not delete a snapshot the user kept.
- Document `batch_key` on `HVGOptions` (forwarded to scanpy on the global HVG
  pass only).
- **Default `filter_mito=False` and `filter_ribo=False`** on `HVGOptions`:
  keep the global HVG ranking as-is. Opt in with
  `options=HVGOptions(filter_mito=True, filter_ribo=True)` to drop MT /
  ribosomal structural-protein symbols and refill (markers kept;
  auto-detects human/mouse naming via `gene_nomenclature`).
- Progress is on by default for auto once `n_obs >= 1000`, and for large
  fixed-k runs. After auto, stderr and
  `uns["scfair"]["hvg"]["auto_message"]` carry one plain-language line for the
  chosen base size.
- Optional faster auto pass: `options=HVGOptions(structure_n_seeds=1)` (product
  default remains multi-seed).

### Documentation

- User guide, README, quickstart, and FAQ describe structure-aware auto-`n`
  and an optional same-ranking list buffer (not population-aware
  reallocation).
- Quickstart and README show a full downstream scanpy path without subsetting
  the gene matrix.

## [0.6.0] - 2026-08-02

### Fixed

- Structure auto: large datasets with few density cores and low density
  confidence floor short base lists to 2000 after the soft size buffer.
- Structure auto: with labels (`n_types ≥ 5`) and multi-core short geometry,
  keep base k=500 (skip the 500→1000 soft buffer). Boards with fewer labeled
  types still use the soft buffer.
- `n_top_genes="auto"` passes `label_key` / `n_types` into structure estimation
  when provided via `HVGOptions`.

### Changed

- Progress and diagnosis tips stay short and user-facing (no internal feature
  dumps on stderr). At most two tips per call.
- Default `append_budget` is **200** (absolute, not scaled with base). Override
  with `HVGOptions(append_budget=…)`.

### Documentation

- Product tables for auto base size and append vs HVG@2000 (see docs).

## [0.5.0] - 2026-08-02

First public release prepared for PyPI and Read the Docs.

### Added

- `scfair.pp.highly_variable_genes` — fair / balanced HVG selection with a short
  product surface and optional research knobs via `HVGOptions`.
  - Default `balance_method="append"`: freeze the global top-`k` list, then
    append the next `append_budget` genes from the same ranking (default 200).
  - `balance_method="none"` reproduces a single global scanpy-style pass.
  - Opt-in cluster-aware methods: `hybrid`, `score`, `reweight`.
  - `n_top_genes="auto"` (structure-aware) and product `mode` presets
    (`compact` / `balanced` / `fine`, or `mode="auto"`).
- `scfair.pp.HVGOptions` — secondary controls (append budget, clustering pool,
  neighbor contrast, raw-count snapshot policy, and related fields).
- `scfair.pp.diagnose_from_labels` — pre-call imbalance tips from known labels
  (does not run HVG).
- `scfair.pp.estimate_n_populations` — label-free estimate of how many
  populations a density field supports (advisory).
- Downstream helpers: `recommend_cluster_resolution`,
  `resolve_cluster_resolution`, `resolve_hvg_mode`, and structure-based auto-`k`
  utilities under `scfair.pp`.
- Sphinx documentation (installation, quickstart, user guide, API, FAQ,
  changelog) and a PBMC 10k tutorial comparing standard HVG with scFair.
- CI matrix for Python 3.10–3.12, a min-versions job for declared dependency
  floors, and a manual PyPI publish workflow (Trusted Publishing).

### Changed

- Default path is append-on-global-rank (no intermediate clustering). Final
  list size is `n_top_genes + append_budget`, capped at `n_vars`.
- Intermediate clustering (when used) scales genes before PCA by default
  (`HVGOptions.scale_clustering=True`).
- Structure auto-`k` applies a soft buffer and floors short lists when density
  confidence is low, so under-powered short lists are less common on large
  multi-type datasets.
- Non-selected genes use `NaN` in `var["highly_variable_rank"]` (scanpy
  seurat_v3 convention).

### Fixed

- Count-layer handling no longer overwrites a user `layers["counts"]` when it
  disagrees with integer `.X`; staging uses an internal layer and is cleaned up
  after the call.
- Repeated HVG calls on the same object do not leak scanpy score columns or
  redirect later default calls to a previous `layer=`.
- Explicit `append_budget` (including `0`) is never replaced by mode defaults.
- seurat_v3 loess path falls back on tiny gene sets before a hard crash.
- Empty or all-zero counts raise a clear error before PCA.
- Ribosomal filter no longer matches kinase names such as `RPS6KA*`.
- Failed HVG runs roll back partial `var` HVG columns and record
  `uns["scfair"]["hvg_failed"]`.

### Documentation

- README oriented for PyPI / GitHub (install, first run, status).
- User-facing docs in American English; research notes stay out of the
  published documentation tree.

## [0.3.0] - 2026-08-01

Internal pre-release: API hygiene, structure auto-`k` anti-SHORT floor, scaled
intermediate PCA, and reliability fixes for empty counts and filter rules.

## [0.2.0] - 2026-08-01

Internal pre-release: product default moved to append, `HVGOptions` introduced,
and the experimental `cap` allocation path removed.

## [0.1.0] - 2026-07-29

Initial private package scaffold and core HVG entry point.
