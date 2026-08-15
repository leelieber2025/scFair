# scFair Documentation

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![PyPI downloads](https://img.shields.io/pepy/dt/scfair.svg)](https://pepy.tech/project/scfair)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/scfair.svg)](https://anaconda.org/bioconda/scfair)
[![Conda downloads](https://img.shields.io/conda/dn/bioconda/scfair.svg)](https://anaconda.org/bioconda/scfair)
[![Python versions](https://img.shields.io/pypi/pyversions/scfair.svg)](https://pypi.org/project/scfair/)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/scFair/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21761251.svg)](https://doi.org/10.5281/zenodo.21761251)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.08.08.743679v1)

scFair selects highly variable genes for scRNA-seq. The default call
estimates a base list size from multi-seed density structure, then keeps
that global top-`k` and appends a short same-rank tail.

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)  # writes adata.var["highly_variable"]
```

Raw integer counts belong in `.X` or `layers["counts"]`. Leave the gene
axis intact; `sc.tl.pca` reads the HVG mask.

| If you need | Pass |
|-------------|------|
| A protocol-fixed length | `n_top_genes=2000` |
| Exact top-`k` (scanpy-sized) | `n_top_genes=2000, balance_method="none"` |
| A label-free population count | {func}`scfair.pp.estimate_n_populations` after `sc.pp.neighbors` |

Auto builds extra neighbor graphs, so it is slower than a fixed `k`.
Read {doc}`user_guide/method` for the estimator and the append rule.

| Page | Contents |
|------|----------|
| {doc}`installation` · {doc}`quickstart` | Install and a first call |
| {doc}`tutorials/pbmc3k_first_analysis` | PBMC 3k: QC → HVG → Leiden / UMAP |
| {doc}`tutorials/pbmc10k_hvg_compare` | PBMC 10k: top-2000 vs top-2000 + append |
| {doc}`user_guide/index` | Method, parameters, outputs |
| {doc}`api/index` | Signatures |
| {doc}`faq` | Troubleshooting |
| {doc}`citation` | Preprint and software DOI |

## Citation

Li, James & Li, 2026.
[bioRxiv 10.64898/2026.08.08.743679](https://doi.org/10.64898/2026.08.08.743679).
BibTeX: {doc}`citation`.

## Author

**Zhao Li (李钊)**
[leelieber@gmail.com](mailto:leelieber@gmail.com)

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
user_guide/index
tutorials/index
api/index
faq
citation
changelog
license
```
