# Changelog

All notable changes to scFair are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/), with the caveat that while the
package is `0.x`, a default value may change in a minor release when new
evidence warrants it.

## [Unreleased]

### Added

- **`n_top_genes="auto"` with `auto_n_method="structure"`** (now the
  default): estimates how many genes to select from the data's own
  structure — density-based population counting and cluster valley
  geometry — instead of a fixed number or a variance-curve heuristic.
  Multi-seed by default in the product path for run-to-run stability.
  Fixed integers (e.g. `n_top_genes=2000`) and `auto_n_method="ensemble"`
  remain fully supported.
- **`resolution="auto"`** (now the default): the intermediate clustering's
  resolution is derived from a 3D density-field estimate of population
  count instead of a fixed number. Falls back to a fixed resolution when
  no usable population count can be produced (too few cells, no usable
  neighbor graph). `uns["scfair"]["hvg"]["resolution_used"]` records what
  actually ran.
- **`scfair.pp.estimate_n_populations(adata)`** — estimate how many
  populations the data's density field supports, without running HVG
  selection. Advisory only; does not modify `adata`.
- **Suitability diagnostics** (`diagnose=True` by default): writes
  `adata.uns["scfair"]["hvg"]["diagnosis"]` with intermediate-cluster
  imbalance, human-readable tips, and a recommendation. Purely advisory —
  gene selection is unaffected by whether this runs. Pre-call planning
  from known labels: `scfair.pp.diagnose_from_labels(...)`.
- **The intermediate clustering is now reported**, in
  `adata.uns["scfair"]["hvg"]["clustering"]`: the PCA/neighbor settings
  actually used (after internal clamping), resolution, cluster sizes,
  clusters dropped below `min_cluster_size`, and variance explained per
  PC. Diagnostic only.
- **`allocation_method="starved_topup"`** (experimental, off by default):
  targeted post-hoc top-up for clusters whose own-gene share falls below
  a budget-adaptive threshold, without a uniform tax on every call.
- **`progress`** parameter (default: auto-enabled on large calls): prints
  stage progress to stderr, including step-by-step status for the slower
  multi-seed structure estimation and the intermediate PCA/neighbor/Leiden
  pass, so a long call isn't mistaken for a hang. Diagnostic tips print
  alongside progress output when both are enabled.
- Experimental knobs for the intermediate clustering: `consensus_resolutions`
  (average specificity over a small resolution ladder instead of trusting
  one), `cluster_genes` / `cluster_pool` (decouple the clustering gene
  space from `n_top_genes`, for iterated selection).
- `n_marker_genes_already_selected` in `adata.uns["scfair"]["hvg"]`: how
  many of the supplied `marker_genes` the algorithm would have chosen
  unaided.

### Changed

- Default `n_top_genes` is now `"auto"` (structure-based); default
  `resolution` is now `"auto"` (density-field-derived). Both previous
  fixed defaults (`2000` and `0.5`) remain fully supported.
- `allocation_method` / `cap_allocation` default to off (`"none"` /
  `False`). Post-hoc equal-share reallocation is opt-in.
- `blend_global` is documented as a re-ranking-strength knob, not a
  fairness lever: lowering it below the default `0.95` does not reduce
  gene-starvation of small clusters and can make it worse. Use
  `allocation_method` for targeted fairness instead.
- The diagnosis tip that downstream clustering resolution matters more
  than the choice of gene-selection method no longer quotes an unstable
  numeric factor — direction only.
- `requires-python` is now `>=3.10` (previously declared as `>=3.9` but
  never tested at that floor).

### Deprecated

- **`allocation_method="cap"` / `cap_allocation=True`** — emits a
  `DeprecationWarning`. Prefer `allocation_method="starved_topup"` or the
  default `"none"`. Scheduled for removal.

### Removed

- **`marker_mode="boost"`** and the `marker_boost` parameter, before the
  first tagged release — the scoring bonus made every marker gene a hard
  include rather than a soft preference. Use `marker_mode="force"`
  instead; `marker_mode="boost"` now raises `ValueError`.

### Fixed

- **Dependency floors corrected and CI-verified**: `scanpy>=1.10` (the
  default intermediate clustering calls `sc.tl.leiden(flavor="igraph")`,
  unavailable before 1.10), `scikit-misc>=0.3` (required by the default
  `flavor="seurat_v3"`), `anndata>=0.10`. `numpy`, `pandas` and `scipy`
  floors raised to tested versions. A CI job installs the exact floors
  and runs the test suite against them.
- **`igraph` is now a declared dependency** — the default
  `balance_method="hybrid"` requires it and previously raised
  `ImportError` on a clean install.
- Multi-seed structure gene-budget estimation is now used consistently
  in the default `n_top_genes="auto"` path, improving run-to-run
  stability of the selected gene count.

### Known limitations

- The cluster-balanced re-ranking layer has little effect on short gene
  lists (roughly below `n_top_genes=500`) — at the default
  `blend_global=0.95`, few genes are displaced from the plain-variance
  ranking at that list length.
- `n_top_max` is not merely a safety bound: it also affects the gene set
  `"auto"` resolves to, not just an upper clip.

## [0.1.0] — 2026-07-29

First tagged release (alpha).

### Added

- `scfair.pp.highly_variable_genes` — the single public entry point, with
  `balance_method` in `{"hybrid", "score", "reweight", "none"}`.
- Automatic `n_top_genes="auto"` gene-budget selection (ensemble method).
- `neighbor_contrast` — scores specificity against the nearest cluster
  rather than cluster-vs-rest, improving sensitivity for rare populations
  adjacent to a common one.
- `marker_genes` with `marker_mode` in `{"force", "boost", "none"}`;
  `marker_extra=True` by default so supplied markers don't displace
  algorithm-selected genes.
- `global_score` — supply an external global variability ranking as the
  anchor, so another method's shortlist can sit underneath the balanced
  layer.
- `combine="best_rank"` — rank-merge instead of score-blend.
- Internal raw-counts snapshot/restore, so the function is safe to call
  once on an AnnData whose `.X` will be normalized afterward.
- Selection metadata under `adata.uns["scfair"]["hvg"]`.

### Changed

- Default intermediate-clustering `resolution` lowered from `1.0` to
  `0.5` — broader intermediate clusters gave steadier specificity
  estimates across the evaluation panel.
- A `n_top_genes>=3000` call now logs an advisory: scFair's measured
  advantage over plain variance-based HVG selection tends to vanish at
  that list length.

### Notes on claims

Against orthogonal (protein / sorting) ground-truth labels rather than
labels derived from a similar HVG pipeline, the measured margin over
plain `seurat_v3` HVG at matched gene-budget is modest but positive
across a labeled evaluation panel. Earlier, larger internal figures were
single-seed and measured against labels produced by pipelines using a
similar HVG step, which overstates the margin; those figures have been
retired.
