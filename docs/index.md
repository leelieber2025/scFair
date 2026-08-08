# scFair Documentation

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![Python versions](https://img.shields.io/pypi/pyversions/scfair.svg)](https://pypi.org/project/scfair/)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/scFair/blob/main/LICENSE)

## What scFair does

scFair is HVG selection focused on **list size**, with a small optional buffer
and diagnostics. Easy to get wrong if you only call
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

### 2. How many populations (no resolution sweep)?

{func}`scfair.pp.estimate_n_populations` reads a 3D density field after
`sc.pp.neighbors`. It is advisory (does not change genes) and works best on
well-separated data up to ~20 populations.

### 3. Optional list buffer (`append`)

Default `balance_method="append"` extends the **same** global ranking by
`append_budget` genes → **`top-(k+m)`**, not population-aware reallocation.
Use `balance_method="none"` for exact top-`k`.

### Default HVG in one line

```text
auto n (default)  →  global top-k  →  optional same-ranking buffer
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
