# scFair Documentation

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![Python versions](https://img.shields.io/pypi/pyversions/scfair.svg)](https://pypi.org/project/scfair/)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/scFair/blob/main/LICENSE)

## What scFair does

scFair is HVG selection with two problems in mind — both common in real
analysis, both easy to get wrong if you only call
`scanpy.pp.highly_variable_genes` with a copied `n_top_genes=2000`.

### 1. How many genes (`n`)?

Most pipelines pick a fixed length (often 2000) by habit. That number is
rarely tuned to the dataset: too short can under-represent structure; too long
adds noise and cost. **`n_top_genes="auto"`** (the default) estimates a base
size from cell density structure.

We do **not** claim auto is always the best `n` for every experiment. Its job
is to be a **data-informed default and a safety net** — so users who are not
ready to sweep `k` still get a reasoned list size instead of an arbitrary
cutoff. When density confidence is low, auto prefers classical 2000 over a
risky short list. Pass `n_top_genes=2000` (or any int) for papers and fixed
protocols.

### 2. Unfair HVG allocation

Global HVG ranking is driven by the largest populations. Markers for smaller
or rare types often land just below the cutoff even when they matter for later
clustering and annotation. That is an **allocation** problem, not a missing
variance formula.

scFair keeps a standard global ranking as the backbone. By default
(`balance_method="append"`) it **freezes** the base top-`k` and **appends** a
small extension of near-miss genes from the **same** ranking. Nothing is
pushed *out* of the base list. Use `balance_method="none"` for a scanpy-like
exact top-`k` with no extension.

### Default in one line

```text
auto n  →  global top-k  →  append near-miss genes
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

1. Install: `pip install scfair`
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
