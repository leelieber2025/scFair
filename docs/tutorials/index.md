# Tutorials

Notebooks are pre-executed: the HTML on Read the Docs already includes tables
and figures, so you can read results online without downloading large `.h5ad`
files or re-running cells.

## Pick a notebook

| If you want… | Open |
|--------------|------|
| Gold-label clustering errors: standard HVG vs scFair (PBMC 10k) | {doc}`pbmc10k_hvg_compare` |

**If you are new:** read {doc}`../quickstart`, then
{doc}`pbmc10k_hvg_compare`.

## Run locally

```bash
pip install scfair
# or from a clone:
# pip install -e ".[dev]"
jupyter lab docs/tutorials/
```

Put the data file where the notebook expects it (repository layout):

```text
examples/data/pbmc_10k_v3_labeled.h5ad
```

That labeled object is not on PyPI. Public 10x PBMC 10k count matrices are
available from [10x Genomics](https://www.10xgenomics.com/datasets). Any AnnData
with raw counts and a cell-type column can follow the same comparison pattern.

Rough runtime with data on disk: a few minutes for two full HVG → Leiden → UMAP
pipelines on ~10k cells.

---

## Notebook cards

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PBMC 10k — misclustered cells vs gold labels
:link: pbmc10k_hvg_compare
:link-type: doc

Same recipe after HVG: count cells whose Leiden cluster majority label is wrong.
+++
Error counts · per-type table · UMAP of misclustered cells
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

pbmc10k_hvg_compare
```
