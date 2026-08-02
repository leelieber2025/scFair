# scFair Documentation

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![Python versions](https://img.shields.io/pypi/pyversions/scfair.svg)](https://pypi.org/project/scfair/)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/scFair/blob/main/LICENSE)

## What scFair does

Highly variable gene (HVG) selection ranks genes by how much they vary across
cells. On a dataset with one or two dominant cell types, that ranking is driven
almost entirely by what separates the large populations. Genes that mark a
rare population can miss the cutoff even though they are exactly what you need
later for clustering or annotation.

scFair keeps a standard global ranking as the backbone and, by default, appends
a small extension of near-miss genes from the same ranking. The default path is
a strict superset of a plain global HVG list: nothing is pushed *out* of the
base top-`k`.

## Where to go

| Goal | Page |
|------|------|
| Install and run a first analysis | {doc}`installation` → {doc}`quickstart` |
| Understand the default and the options | {doc}`user_guide/index` |
| Gold-label clustering errors vs standard HVG (PBMC 10k) | {doc}`tutorials/index` |
| Function signatures | {doc}`api/index` |
| Common issues | {doc}`faq` |

### A sensible path

1. Install: `pip install scfair`
2. Follow {doc}`quickstart`
3. Work through the {doc}`tutorials/pbmc10k_hvg_compare` notebook
4. Use {doc}`faq` when a call looks off

### Default call

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)  # raw counts in .X or layers["counts"]
adata = adata[:, adata.var["highly_variable"]].copy()
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
