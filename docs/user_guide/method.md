# How selection works

scFair addresses two choices in HVG selection. Flavors and diagnostics are
secondary.

## Two problems

### Problem A — How many genes (`n`)?

Pipelines need a gene budget. There is no single correct length a priori.
Copying 2000 from a tutorial is simple and can be wrong for a given dataset
(too short erases structure; too long adds noise and cost).

**Default:** `n_top_genes="auto"` estimates a base size `k` from multi-seed
density structure. Auto is a data-informed default, not a proof of the best
`n` on every dataset. When density confidence is low, it prefers classical
**2000** over a short list. Pass a fixed int for papers and locked protocols.
Auto is multi-seed and slower than a fixed `k`.

### Problem B — Hard cutoff on a global ranking

Global HVG ranking measures variability across all cells. When a few cell types
dominate, the top of the list is filled by genes that separate those large
groups. Markers for smaller populations often sit just below a fixed cutoff,
because bulk variation fills the slots first.

**Default:** `append` keeps the global top-`k` and adds a short extension of
near-miss genes from the **same** ranking (ranks `k+1 … k+m`). Nothing is
pushed out of the base. The selected set equals **`top-(k+m)`** (same genes as
`balance_method="none"` with `n_top_genes=k+m`).

Pass **`balance_method="none"`** (or `options=HVGOptions(append_budget=0)`)
for pure top-`k`.

## Default path: auto `n` + append

1. Rank all genes with a standard global method (`flavor="seurat_v3"` by default).
2. **Problem A:** choose base size **`k`** with structure-aware auto (default).
   Low density confidence floors soft short lists to classical **2000**
   (except labeled true-SHORT paths). Pass `n_top_genes=2000` to skip auto.
3. Keep the top `k` genes.
4. **Problem B:** append the next `append_budget` genes from the same ranking
   (ranks `k+1 … k+m`). Default floor **200**; when `n_top_genes="auto"`,
   budget may rise with structure `n_density_pops` as
   `m = max(200, min(300, 200 + max(0, n_need − 12) × 12))`. Explicit
   `HVGOptions(append_budget=N)` always wins.

Properties:

- No intermediate clustering for gene selection.
- The base top-`k` is kept in full; the extra genes are a tail.
- Final size is `k + append_budget` (capped at `n_vars`) when append is on.
- Auto is multi-seed and slower than a fixed `k`.

GiniClust and CellSIUS score rare-subtype genes (Gini, cluster-wise tests).
Default `append` only adds near-miss genes from the global HVG list. For a
known missing type, force-include markers (`marker_mode="force"`).

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)  # auto + append
# scf.pp.highly_variable_genes(adata, n_top_genes=2000)  # fixed base
```

If the data has technical batches and you want scanpy's per-batch HVG merge on
that global ranking, pass `options=HVGOptions(batch_key="...")`. See
{doc}`parameters` and the FAQ entry on multi-batch data.

## Match scanpy: fixed `k` + `balance_method="none"`

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
```

This is a single global HVG pass with no extension. Use it for comparisons
against existing protocols.

## Choosing `n_top_genes`

| Value | Behavior |
|-------|----------|
| `"auto"` (**default**) | Structure-aware base size `k`, then your `balance_method` (default: append) |
| int (e.g. **2000**) | Fixed base size; best for papers and fixed protocols |
| `"structure"` | Same size estimator as `"auto"`, exposed as an explicit name |

With the default `balance_method="append"`, auto only chooses **how many**
genes. It does not re-rank the list. The gene set is a frozen global top-`k`
plus a small extension.

### What auto does under the hood

Auto builds a short multi-seed view of cell **density** in a low-dimensional
embedding (after neighbors / PCA). It is not a second gene-variance formula.
Roughly:

- several clear dense cores → a longer or shorter list by rule, often near
  the classical 2000 band
- low density confidence → prefer classical **2000** rather than a short list
- labeled multi-core short geometry can still keep a short base

Bounds: `n_top_min` / `n_top_max` (default 500–5000). Cost: several graph
builds, so large objects take longer than `n_top_genes=2000`.

After a call, `adata.uns["scfair"]["hvg"]["auto_n"]` records what happened:

| Field | Meaning |
|-------|---------|
| `strategy` | Always `"structure"` (auto is structure-only) |
| `n_top_selected` | The base `k` it picked |
| `structure` | Density features (population count, valley depth, stability) |
| `rule_branch` | Internal tag for which rule fired; useful in bug reports |

**Limitation:** on small matrices the estimator can hit `n_top_max` and
select almost every gene (see {doc}`../faq`). Prefer a fixed
`n_top_genes=2000` for locked paper protocols.

## Product modes

`mode` adjusts the default base size / append budget for broad regimes
(`compact`, `balanced`, `fine`, or `auto`). With a fixed integer
`n_top_genes`, mode does not rewrite the gene list; it mainly matters when you
use `"auto"` or leave size decisions to the defaults. See {doc}`parameters`.

## Markers

You can force known markers into the final set with `marker_genes` and
`marker_mode`. With the default `marker_extra=True`, forced markers are added
on top of the selected list, so the final size can exceed `n_top_genes`.
