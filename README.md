# scFair

Cluster-aware highly variable gene (HVG) selection for single-cell RNA-seq.

Global HVG ranks genes by genome-wide variance, so a top-2000 list can be
dominated by large populations. scFair keeps a **global anchor** and re-ranks
inside that pool using **cluster-vs-rest specificity**, so smaller populations
still contribute identity genes.

One function, a scanpy-shaped API. Intermediate choices are recorded in
`adata.uns["scfair"]`.

## Install

```bash
pip install -e ".[dev]"   # or: pip install -e .
```

Python ≥ 3.10 · `scanpy` · `anndata` · `scikit-learn`.

## Quick start

```python
import scfair as scf

# Raw integer counts in .X, or layers["counts"] (pass layer= if needed).
# Default: hybrid + structure-aware auto k (v7) + resolution="auto"
scf.pp.highly_variable_genes(adata)

adata = adata[:, adata.var["highly_variable"]].copy()
```

Fixed classical size (still fully supported):

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000)
```

## How it works

1. **Global HVG** (`flavor="seurat_v3"` by default) — same family as scanpy.
2. **Intermediate clustering** — PCA → neighbours → Leiden on the selection
   pool (or `cluster_pool` / `cluster_genes` if set).
3. **Specificity** — one-sided logFC vs rest per intermediate cluster, size-
   weighted so large clusters do not monopolise the score.
4. **Hybrid re-rank** (default) — inside the global top-`2 × k` pool:

   `S = α · global + (1 − α) · specificity`,  α = `blend_global` (default **0.95**).

With `n_top_genes="auto"` (default), scFair first estimates **structure-aware
k** (density valleys + multi-seed features), then runs hybrid **once** at that
k. Pass an integer k to skip structure and use a fixed shortlist size.

## Defaults that matter

| Parameter | Default | Notes |
|---|---|---|
| `n_top_genes` | `"auto"` | Structure **v7** picks k (often 500 / 1500 / 2000 / 3000+). Use `2000` for a fixed protocol. |
| `auto_n_method` | `"structure"` | Only when `n_top_genes="auto"`. Use `"ensemble"` for the older ~2000 anchor. |
| `balance_method` | `"hybrid"` | `"score"` / `"reweight"` / `"none"` (plain scanpy-like) also available. |
| `blend_global` | `0.95` | Keep most of the ranking global. **Do not lower this to “fix fairness”** — that worsens equal-share deprivation; fairness research uses post-hoc `allocation_method` (default off). |
| `resolution` | `"auto"` | Density-field target cluster count for the intermediate Leiden. Pass a float (e.g. `0.5`) if types are contiguous sub-states. |
| `min_cluster_size` | `30` | Smaller intermediate clusters are dropped from scoring. |
| `diagnose` | `True` | Advisory tips in `uns` only; does not change genes. |
| `allocation_method` | `"none"` | Optional post-hybrid reallocation (`"starved_topup"` experimental; `"cap"` deprecated). |

### `balance_method`

| Value | When |
|---|---|
| **`hybrid`** (default) | Scanpy-like subspace with rare-type protection. |
| `score` | Specificity-driven ranking (stronger, larger trade-offs). |
| `reweight` | Cell-resampled global HVG. |
| `none` | Single global HVG pass (control / scanpy-like). |

## Choosing `n_top_genes`

```python
# Default product path
scf.pp.highly_variable_genes(adata)

# Fixed size
scf.pp.highly_variable_genes(adata, n_top_genes=2000)

# Old ensemble auto (~2000 anchor)
scf.pp.highly_variable_genes(adata, n_top_genes="auto", auto_n_method="ensemble")
```

| Strategy | Behaviour |
|---|---|
| **`structure`** (default auto) | Label-free features (valleys, density pops, Leiden, coarse pair-stability). Multi-seed (**3**, full graph redraw per seed) then hybrid at chosen k. v7 fine-atlas band floors seurat-like large multi-core sets near 2000. |
| `ensemble` | Shape + cumfrac + depth-aware anchor; often near 2000. |
| int | Fixed shortlist; hybrid re-ranks global top-`2k`. |

```python
h = adata.uns["scfair"]["hvg"]
h["n_top_genes_used"]   # k that was used
h["auto_n"]             # strategy, structure features, k_source, …
```

**Tips**

- Prefer **fixed `n_top_genes=2000`** in papers or when you need a reproducible protocol without structure.
- Structure may choose **k ≥ 3000** on Duo-like residual structure; hybrid logs a warning there by design.
## `resolution="auto"`

Intermediate granularity is taken from a 3D density field on the graph already
built for clustering, then Leiden is tuned toward that population count.

```python
d = adata.uns["scfair"]["hvg"]["clustering"]
d["resolution"]           # value that ran
d["resolution_source"]    # "density_field" or "fallback"
```

Override when types are **contiguous** sub-states (density valleys miss them):

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000, resolution=0.5)
```

Falls back to `0.5` when the density field is unusable.

## Optional knobs

### Rare sibling types (`neighbor_contrast`)

Cluster-vs-**rest** misses the boundary between two adjacent populations. For a
rare subset next to a common sibling:

```python
scf.pp.highly_variable_genes(
    adata,
    n_top_genes=1000,
    neighbor_contrast=1.0,
    resolution=1.0,   # needed so siblings are split in the intermediate graph
)
```

Off by default. Prefer shorter lists (~1000) for this use case.

### Force markers

```python
scf.pp.highly_variable_genes(adata, marker_genes=panel, marker_mode="force")
# Check first: marker_mode="none" still records how many would have been selected
```

Forcing large classic panels can hurt subspace quality; check
`n_marker_genes_already_selected` in `uns` before injecting.

### Post-hoc allocation (research, default off)

```python
# Experimental: top-up units starved of equal-share own-genes (adaptive budget)
scf.pp.highly_variable_genes(
    adata, n_top_genes=2000, allocation_method="starved_topup"
)
```

Default remains `"none"`. `"cap"` is deprecated.

## Outputs

```python
adata.var["highly_variable"]
adata.var["highly_variable_rank"]
adata.var["scfair_score"]
adata.obs["scfair_hvg_clusters"]     # intermediate partition (balanced methods)
adata.uns["scfair"]["hvg"]           # full metadata for the call
```

### Intermediate clustering diagnostics

```python
d = adata.uns["scfair"]["hvg"]["clustering"]
d["n_pcs_used"], d["n_neighbors_used"]
d["resolution"], d["n_genes_clustered"]
d["n_clusters_total"], d["n_clusters_kept"]
d["clusters_dropped"]    # below min_cluster_size → no specificity score
d["n_passes"]            # physical intermediate builds (often 1 after reuse / structure-fast)
```

`clusters_dropped` is the list to watch: those communities never vote in
specificity. This partition is a **diagnostic**, not a substitute for your
downstream clustering.

### Diagnosis (advisory)

```python
diag = adata.uns["scfair"]["hvg"]["diagnosis"]
diag["tips"], diag["recommendation"]
```

With known labels, plan before a full call:

```python
scf.pp.diagnose_from_labels(adata.obs["cell_type"])
```

## Utilities

```python
sc.pp.neighbors(adata)
est = scf.pp.estimate_n_populations(adata)   # same density field as resolution="auto"
est.n_populations, est.reason
```

Does not overwrite your existing UMAP unless you opt in to label copy.

## Runtime & progress

The intermediate PCA → neighbours → Leiden is the slow step. Stage messages go
to stderr on large data (or always with `progress=True`):

```python
scf.pp.highly_variable_genes(adata, progress=True)
# progress=False to silence
```

Rough order of magnitude (single mid-size PBMC-scale run): plain global HVG
seconds; hybrid tens of seconds; structure auto adds multi-seed feature
extraction plus one hybrid pass at the chosen k.

## Docs

See [`CHANGELOG.md`](CHANGELOG.md) for user-facing changes and current defaults.

scFair’s `balance_method="hybrid"` is **not** mixHVG’s hybrid: mixHVG mixes
method rankings; scFair blends global variability with cluster specificity
inside one pool.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
