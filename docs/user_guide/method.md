# How selection works

## The problem

Global HVG ranking measures variability across all cells. When a few cell types
dominate the sample, the top of the list is filled by genes that separate those
large groups. Markers for smaller populations often sit just below a fixed
cutoff such as 2000 genes. They are not uninformative — they simply lose the
vote count to bulk variation.

## Default: `balance_method="append"`

1. Rank all genes with a standard global method (`flavor="seurat_v3"` by default).
2. Freeze the top `n_top_genes` genes. This is the familiar HVG backbone.
3. Append the next `append_budget` genes from the **same** ranking
   (ranks `k+1 … k+m`).

Properties that matter in practice:

- No intermediate clustering and no re-ranking inside a pool.
- Nothing is removed from the base top-`k`.
- Final size is `n_top_genes + append_budget` (capped at `n_vars`).
- Runtime is on the same order as a single `scanpy.pp.highly_variable_genes` call.

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)  # append is the default
```

## Reproduce scanpy: `balance_method="none"`

```python
scf.pp.highly_variable_genes(adata, balance_method="none")
```

This is a single global HVG pass with no extension. Use it for drop-in
comparisons against existing protocols.

## Opt-in cluster-aware methods

These paths build intermediate clusters and re-score genes. They can help on
hard tasks but cost a PCA → neighbors → Leiden pass and may reshuffle the base
list. Prefer them for research comparisons, not as the first default.

| Method | Idea |
|--------|------|
| `"hybrid"` | Blend global variability with per-cluster specificity inside a candidate pool |
| `"score"` | Rank mainly by cluster-vs-rest specificity |
| `"reweight"` | Resample cells to rebalance cluster mass, then one global HVG pass |

```python
scf.pp.highly_variable_genes(
    adata,
    balance_method="hybrid",
    blend_global=0.95,
)
```

`resolution` and `min_cluster_size` only affect methods that cluster. They are
ignored on the default `"append"` path.

## Choosing `n_top_genes`

| Value | Behavior |
|-------|----------|
| int (default **2000**) | Fixed base size; classical community choice |
| `"auto"` | Label-free structure tools pick `k` from the data, then apply the same base + append step |

Fixed `k` is better for reproducible protocols. `"auto"` is better for
exploration and will record why it chose a value under
`adata.uns["scfair"]["hvg"]["auto_n"]`.

## Product modes

`mode` adjusts the default base size / append budget for broad regimes
(`compact`, `balanced`, `fine`, or `auto`). With a fixed integer
`n_top_genes`, mode does not rewrite the gene list; it mainly matters when you
use `"auto"` or leave size decisions to the product defaults. See
{doc}`parameters`.

## Markers

You can force known markers into the final set with `marker_genes` and
`marker_mode`. Read the function docstring before relying on this: injecting
markers is not free even when they are added outside the base count.
