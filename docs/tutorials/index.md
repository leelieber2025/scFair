# Tutorials

Notebooks are pre-executed: the HTML on Read the Docs already includes tables
and figures, so you can read results online without downloading data or
re-running cells. {doc}`pbmc3k_first_analysis` embeds only the output that
matters for following along (one UMAP figure, a few short tables/printouts) —
it does not aim for the same output density as {doc}`pbmc10k_hvg_compare`.

## Pick a notebook

| If you want… | Open |
|--------------|------|
| A real workflow: load → QC → scFair → Leiden/UMAP, no labels needed | {doc}`pbmc3k_first_analysis` |
| Gold-label clustering errors: standard HVG vs scFair (PBMC 10k) | {doc}`pbmc10k_hvg_compare` |

**If you are new:** read {doc}`../quickstart`, then
{doc}`pbmc3k_first_analysis`. {doc}`pbmc10k_hvg_compare` is a method
comparison, not a usage guide — read it once you want to see the evidence
behind the default.

## Run locally

```bash
pip install scfair
# or from a clone:
# pip install -e ".[dev]"
jupyter lab docs/tutorials/
```

{doc}`pbmc3k_first_analysis` downloads its own (small, public) data on first
run — nothing to place by hand.

{doc}`pbmc10k_hvg_compare` needs a labeled file at this repository-relative
path:

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

:::{grid-item-card} PBMC 3k — a first real analysis
:link: pbmc3k_first_analysis
:link-type: doc

Load counts → QC → scFair HVG → PCA/Leiden/UMAP, no gold labels, plus a short
guide to when to change the defaults.
+++
Auto-downloaded public dataset; nothing to place by hand
:::

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

pbmc3k_first_analysis
pbmc10k_hvg_compare
```
