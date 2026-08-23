# How selection works

Two decisions sit in every HVG step: the list length, and what happens
at a hard cutoff. scFair sets both. Flavors and diagnostics are
secondary. The 18-dataset panel is in the
[preprint](https://www.biorxiv.org/content/10.64898/2026.08.08.743679v1)
(Li, James & Li, 2026).

## List size

There is no universally correct `n`. A fixed value of 2,000 is common and
often reasonable, but it is not data-adaptive.

`n_top_genes="auto"` (the default) estimates a base size `k` from a
short multi-seed view of cell density after PCA / neighbors. It is not
a second variance formula. Low density confidence floors a short list
to 2000. Multi-core short geometry can keep a short base. Bounds are
`n_top_min` / `n_top_max` (500–5000). The estimator builds several graphs,
so it is slower on large datasets than a fixed `n_top_genes=2000` call.

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Structure estimator, then your `balance_method` |
| int (e.g. `2000`) | Fixed base. Use this in a paper methods section. |
| `"structure"` | Same estimator, named explicitly |

`adata.uns["scfair"]["hvg"]["auto_n"]` records the run:

| Field | Meaning |
|-------|---------|
| `strategy` | Always `"structure"` |
| `n_top_selected` | Base `k` |
| `structure` | Density features (population count, valley depth, stability) |
| `rule_branch` | Which rule fired; useful in a bug report |

On a small gene panel the estimator can hit `n_top_max` and keep almost
every gene. Pin `n_top_genes=2000` for a locked protocol. See {doc}`../faq`.

## Cutoff

Global HVG ranking scores variation across all cells. Large populations
fill the top of the list. Markers for smaller types often sit just
below a fixed cut.

`balance_method="append"` keeps the global top-`k` and adds ranks
`k+1 … k+m` from the **same** list. Nothing is pushed out of the base.
With no gene filters or forced markers, and with enough genes available, the
selected set equals `top-(k+m)`: the same genes as `balance_method="none"`
with `n_top_genes=k+m`.

Default `m` is 200. On `n_top_genes="auto"`, `m` may rise with the
density-core count `n_density_pops`:

```text
m = max(200, min(300, 200 + max(0, n_density_pops − 12) × 12))
```

An explicit `HVGOptions(append_budget=N)` is never overridden.
`balance_method="none"` (or `append_budget=0`) is pure top-`k`.

The append step does not perform cluster-specific gene scoring. The default
auto-size estimator does run an internal Leiden clustering on a temporary
representation, but it does not require or alter the user's cluster labels.
GiniClust and CellSIUS score rare-subtype genes; append only lengthens the
global HVG list. Force known markers with `marker_genes` if a type is still
missing.

## Default path

1. Rank all genes (`flavor="seurat_v3"`).
2. Choose base `k` (auto, or the integer you passed).
3. Keep the top `k`.
4. If append is on, add the next `m` genes from that ranking.

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)
scf.pp.highly_variable_genes(adata, n_top_genes=2000)
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
```

For scanpy's per-batch HVG merge on that ranking, pass
`options=HVGOptions(batch_key="...")`. See {doc}`parameters`.

## `mode`

`mode` (`auto`, `compact`, `balanced`, `fine`) steers auto-`k` floors
and the default append budget. A fixed integer `n_top_genes` ignores
`mode` for the gene list (you get a warning). It matters when size is
left to auto.

## Markers

`marker_genes` with the default `marker_mode="force"` puts those genes
in the final set. `HVGOptions.marker_extra=True` (default) adds them
on top of the selected list, so the size can exceed `n_top_genes`.
