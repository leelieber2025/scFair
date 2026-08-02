# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/). While the package
is in the `0.x` series, defaults may still change in a minor release when new
evidence warrants it.

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
- Structure auto-`k` applies a soft buffer and an anti-SHORT floor when density
  confidence is low, so under-powered short lists are less common on large,
  multi-type PBMCs.
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
- User-facing docs in American English; research notes and paper tables stay
  local and are not part of the published documentation tree.

## [0.3.0] - 2026-08-01

Internal pre-release: API hygiene, structure auto-`k` anti-SHORT floor, scaled
intermediate PCA, and reliability fixes for empty counts and filter rules.

## [0.2.0] - 2026-08-01

Internal pre-release: product default moved to append, `HVGOptions` introduced,
and the experimental `cap` allocation path removed.

## [0.1.0] - 2026-07-29

Initial private package scaffold and core HVG entry point.
