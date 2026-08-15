# Tutorials

Notebooks are pre-executed. The HTML on Read the Docs already has the
tables and figures.

| Notebook | What it does |
|----------|----------------|
| {doc}`pbmc3k_first_analysis` | Public PBMC 3k. Load, QC, default scFair, Leiden / UMAP. No labels. |
| {doc}`pbmc10k_hvg_compare` | Labeled PBMC 10k. Exact top-2000 vs top-2000 + append, same Leiden recipe. |

{doc}`pbmc3k_first_analysis` downloads its own data.

{doc}`pbmc10k_hvg_compare` reads this repository-relative file:

```text
examples/data/pbmc_10k_v3_labeled.h5ad
```

That object is not on PyPI. Public 10x PBMC 10k counts are at
[10x Genomics](https://www.10xgenomics.com/datasets). Any AnnData with
raw counts and a cell-type column can follow the same comparison.
Runtime with data on disk: a few minutes for two HVG → Leiden → UMAP
passes on ~10k cells.

```bash
pip install scfair
jupyter lab docs/tutorials/
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PBMC 3k
:link: pbmc3k_first_analysis
:link-type: doc

QC → default HVG → PCA / Leiden / UMAP, no gold labels.
+++
Auto-downloaded public dataset
:::

:::{grid-item-card} PBMC 10k
:link: pbmc10k_hvg_compare
:link-type: doc

Majority-vote clustering error, top-2000 vs top-2000 + append.
+++
Needs the labeled h5ad under `examples/data/`
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

pbmc3k_first_analysis
pbmc10k_hvg_compare
```
