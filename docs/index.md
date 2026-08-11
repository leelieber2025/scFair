# scFair Documentation

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![PyPI downloads](https://img.shields.io/pepy/dt/scfair.svg)](https://pepy.tech/project/scfair)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/scfair.svg)](https://anaconda.org/bioconda/scfair)
[![Conda downloads](https://img.shields.io/conda/dn/bioconda/scfair.svg)](https://anaconda.org/bioconda/scfair)
[![Python versions](https://img.shields.io/pypi/pyversions/scfair.svg)](https://pypi.org/project/scfair/)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/scFair/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21761251.svg)](https://doi.org/10.5281/zenodo.21761251)

## What scFair does

scFair is HVG selection for two problems that show up in almost every
scRNA-seq pipeline — easy to get wrong if you only call
`scanpy.pp.highly_variable_genes` with a copied `n_top_genes=2000`.

### 1. How many genes (`n`)?

**Nobody can predict a reasonable list length before seeing the data.** Most
pipelines copy `2000` from a tutorial — simple, and easy to get wrong silently
(too short erases structure; too long adds noise and cost).

**`n_top_genes="auto"` is the default for that reason.** It estimates a base
size from multi-seed density structure so you are less likely to ship a bad
`n` by habit. The extra compute is intentional: **avoiding a wrong `n` is
worth more than a sub-second HVG pass.** We do **not** claim auto is always
optimal — only that it is a safer default than an unexamined fixed length.
Pass a fixed int when a paper or locked protocol already requires one.

### 2. Unfair HVG allocation at the cutoff

Global ranking measures variability across **all** cells. Large populations
fill the top of the list; markers for smaller types often land **just below**
a fixed top-`k` cutoff. They are not uninformative — they lose the vote count
to bulk variation. That is unfair **allocation of slots** in a fixed-length
list.

**What scFair does:** keep a standard global ranking as the backbone, then by
default (`balance_method="append"`) **freeze** the base top-`k` and **append**
a short same-rank tail of near-miss genes so they are not discarded. The
selected set equals **`top-(k+m)`**. This is a **conservative response to
cutoff unfairness** — not cluster-conditional reallocation (per-cluster
quota / hybrid methods were removed). Use `balance_method="none"` for exact
top-`k`.

### 3. How many populations (optional)

{func}`scfair.pp.estimate_n_populations` reads a 3D density field after
`sc.pp.neighbors`. Advisory only (does not change genes); works best on
well-separated data up to ~20 populations.

### Default HVG in one line

```text
auto n (default)  →  global top-k  →  same-rank append (cutoff buffer)
```

## Where to go

| Goal | Page |
|------|------|
| Install and run a first analysis | {doc}`installation` → {doc}`quickstart` |
| Understand the default and the options | {doc}`user_guide/index` |
| A full real workflow on real (unlabeled) data | {doc}`tutorials/pbmc3k_first_analysis` |
| Gold-label clustering errors vs standard HVG (PBMC 10k) | {doc}`tutorials/pbmc10k_hvg_compare` |
| Function signatures | {doc}`api/index` |
| Common issues | {doc}`faq` |

### A sensible path

1. Install: `pip install scfair` (or `conda install -c conda-forge -c bioconda scfair`)
2. Follow {doc}`quickstart`
3. Work through the {doc}`tutorials/pbmc3k_first_analysis` notebook
4. Use {doc}`faq` when a call looks off

### Default call

```python
import scfair as scf

# raw counts in .X or layers["counts"]; default: auto k + append
scf.pp.highly_variable_genes(adata)
# marks HVGs in adata.var; do not subset the matrix (scanpy PCA uses the mask)
```

## Author

**Zhao Li (李钊)**  
Email: [leelieber@gmail.com](mailto:leelieber@gmail.com)

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
user_guide/index
tutorials/index
api/index
faq
changelog
license
```
