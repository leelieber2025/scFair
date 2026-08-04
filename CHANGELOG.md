# Changelog

All notable changes to this project are documented in this file.



## [Unreleased]

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
- Do not leave an inline `uns["scfair"]["raw_snapshot"]` matrix after HVG
  unless `options=HVGOptions(store_raw=True)` (or `"ondisk"`).
- Document `batch_key` on `HVGOptions` (forwarded to scanpy on the global HVG
  pass only).
- **Default `filter_mito=False` and `filter_ribo=False`** on `HVGOptions`:
  keep the global HVG ranking as-is. Opt in with
  `options=HVGOptions(filter_mito=True, filter_ribo=True)` to drop MT /
  ribosomal structural-protein symbols and refill (markers kept;
  auto-detects human/mouse naming via `gene_nomenclature`).
- Progress is on by default for auto on medium-size data (≥3k cells) as well as
  large fixed-k runs. After auto, stderr and
  `uns["scfair"]["hvg"]["auto_message"]` carry one plain-language line for the
  chosen base size.
- Optional faster auto pass: `options=HVGOptions(structure_n_seeds=1)` (product
  default remains multi-seed).

### Documentation

- User guide, README, quickstart, and FAQ emphasize two product goals: choosing
  a sensible gene budget (`n`), and fairer allocation of HVG slots for smaller
  populations. Auto is described as a safe default, not a claim of global
  optimality.
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
