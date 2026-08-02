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
| `"structure"` | Same estimator as `"auto"`, used directly without the extra re-selection pass |

Fixed `k` is better for reproducible protocols. `"auto"` / `"structure"` are
better for a first look at a new dataset when you do not want to guess `k`
by hand.

### What `"auto"` / `"structure"` actually do

Both read the **density** of cells in a low-dimensional embedding (after
`sc.pp.neighbors`), rather than looking at gene variance directly. The idea:
a dataset with several well-separated cell populations shows up as several
dense cores with valleys between them; a dataset dominated by one or two
large populations shows shallow or few valleys. More, deeper valleys are
read as evidence of more distinct populations worth keeping separate
markers for, and the estimator picks a larger gene list; fewer or shallower
valleys give a shorter list. The result is clipped to `n_top_min` /
`n_top_max` (default 500–5000).

`"auto"` additionally re-selects genes within the global top `2×k` using
the same specificity-aware step as `balance_method="hybrid"`, which is why
it costs noticeably more than a fixed `k` — expect a call to run for tens
of seconds instead of a couple, since it repeats the PCA → neighbours step
internally. `"structure"` only asks "how many genes", then hands `k`
straight to your chosen `balance_method`, so it is cheaper.

After a call, `adata.uns["scfair"]["hvg"]["auto_n"]` records what happened:

| Field | Meaning |
|-------|---------|
| `strategy` | Which estimator ran (`"structure"` unless you changed `auto_n_method`) |
| `n_top_selected` | The `k` it picked |
| `structure` | The density features it measured (population count, valley depth, stability) |
| `rule_branch` | An internal tag naming which rule fired — useful when reporting an issue, not meant to be interpreted on its own |

**Known limitation:** on small datasets the estimator can hit `n_top_max`
and select almost every gene (see {doc}`../faq`). Prefer a fixed `n_top_genes`
whenever you need to re-run the same protocol later, e.g. for a paper.

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
