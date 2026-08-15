# scFair

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![PyPI downloads](https://img.shields.io/pepy/dt/scfair.svg)](https://pepy.tech/project/scfair)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/scfair.svg)](https://anaconda.org/bioconda/scfair)
[![Conda downloads](https://img.shields.io/conda/dn/bioconda/scfair.svg)](https://anaconda.org/bioconda/scfair)
[![Python versions](https://img.shields.io/pypi/pyversions/scfair.svg)](https://pypi.org/project/scfair/)
[![Documentation Status](https://readthedocs.org/projects/scfair/badge/?version=latest)](https://scfair.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21761251.svg)](https://doi.org/10.5281/zenodo.21761251)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.08.08.743679v1)

**scFair** selects highly variable genes (HVGs) for single-cell RNA-seq. It
handles two choices most pipelines already make, plus an optional population
count:

| Choice | Common default | What scFair does |
|--------|----------------|------------------|
| **How many genes (`n`)?** | Fixed `2000` | **`n_top_genes="auto"`** (default) estimates a base size from multi-seed density structure. Pass an int when a protocol already fixes `n`. Auto is slower than a fixed `k`. |
| **Hard top-`k` cutoff** | Rank globally and keep top-`k` | Default **`append`** keeps the global top-`k` and adds the next `append_budget` genes from the same ranking (`top-(k+m)`). Use `balance_method="none"` for exact top-`k`. |

Optional: **`estimate_n_populations`** counts populations from a 3D density
field after `sc.pp.neighbors`. It works best on well-separated data with up to
about 20 populations.

Docs: [Read the Docs](https://scfair.readthedocs.io/en/latest/).

## What you need

- Python 3.10+ (tested 3.10–3.12)
- An AnnData object with raw integer counts in `.X` or `layers["counts"]`

## Install

```bash
pip install scfair
# or: conda install -c conda-forge -c bioconda scfair
```

Details: [installation guide](https://scfair.readthedocs.io/en/latest/installation.html).

## First run

```python
import scanpy as sc
import scfair as scf

# raw integer counts in .X or layers["counts"]
scf.pp.highly_variable_genes(adata)
# annotates adata.var["highly_variable"]; keep the full gene matrix
# (scanpy PCA uses that mask by default — no manual subset step)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
```

Default path: structure auto-`n` + same-rank **append**. Auto runs extra
multi-seed graph builds and is slower than a fixed `k`. Read
`adata.uns["scfair"]["hvg"]["auto_message"]` for a one-line summary of the
chosen base size. Fixed length for a locked protocol:

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000)
```

Pure top-`k` (no append tail):

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
```

Population count after neighbors:

```python
sc.pp.neighbors(adata)
est = scf.pp.estimate_n_populations(adata)
print(est.n_populations, est.confidence)
```

## Status

**0.10.0 (Beta).** Import as `import scfair as scf` and use names in
`scfair.__all__` and `scf.pp`. See the
[API reference](https://scfair.readthedocs.io/en/latest/api/index.html).

## Next steps

1. [Installation](https://scfair.readthedocs.io/en/latest/installation.html)
2. [Quickstart](https://scfair.readthedocs.io/en/latest/quickstart.html)
3. [Tutorials](https://scfair.readthedocs.io/en/latest/tutorials/index.html)
4. [FAQ](https://scfair.readthedocs.io/en/latest/faq.html) if something looks off

## Citation

If you use scFair in published work, please cite the preprint:

> Zhao Li, Aaron James, Shengxuan Li.  
> scFair: Geometry-Aware Gene Budgets and Same-Rank Extension for Highly Variable Gene Selection.  
> *bioRxiv* (2026). https://doi.org/10.64898/2026.08.08.743679

```bibtex
@article{li2026scfair,
  title   = {scFair: Geometry-Aware Gene Budgets and Same-Rank Extension for Highly Variable Gene Selection},
  author  = {Li, Zhao and James, Aaron and Li, Shengxuan},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.08.743679},
  url     = {https://www.biorxiv.org/content/10.64898/2026.08.08.743679v1},
}
```

For the software record, cite the [Zenodo DOI](https://doi.org/10.5281/zenodo.21761251).
Pin the package version used in the analysis, for example `scfair==0.10.0`.
See `CITATION.cff`.

## License

Software: [Apache License 2.0](LICENSE).

## Author

**Zhao Li (李钊)**  
Email: [leelieber@gmail.com](mailto:leelieber@gmail.com)
