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

scFair is HVG selection for two choices that appear in almost every scRNA-seq
pipeline. Both are easy to get wrong if you only call
`scanpy.pp.highly_variable_genes` with a fixed `n_top_genes=2000`.

### 1. How many genes (`n`)?

There is no universal correct list length. Pipelines often copy `2000` from a
tutorial. Too short can erase structure; too long adds noise and cost.

**`n_top_genes="auto"` is the default.** It estimates a base size from
multi-seed density structure. Auto is not guaranteed optimal on every dataset;
it is a data-informed alternative to an unexamined fixed length. Pass a fixed
int when a paper or locked protocol already requires one. Auto needs extra
graph builds and is slower than a fixed `k`.

### 2. Hard cutoff on a global ranking

Global ranking measures variability across all cells. Large populations fill
the top of the list; markers for smaller types often land just below a fixed
top-`k` cutoff.

**Default (`balance_method="append"`):** keep a standard global ranking as the
backbone, freeze the base top-`k`, and append a short same-rank tail of
near-miss genes. The selected set equals **`top-(k+m)`**. This is not
cluster-conditional reallocation (per-cluster quota methods were removed). Use
`balance_method="none"` for exact top-`k`.

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
