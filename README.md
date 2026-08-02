# scFair

[![PyPI version](https://img.shields.io/pypi/v/scfair.svg)](https://pypi.org/project/scfair/)
[![Documentation Status](https://readthedocs.org/projects/scfair/badge/?version=latest)](https://scfair.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/scFair/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**scFair** selects highly variable genes (HVGs) for single-cell RNA-seq while
giving small cell populations a fairer chance at the gene list.

Standard HVG ranking is driven by the largest populations. Markers that matter
for rare types can fall just below the cutoff even when they are exactly what
you need later for clustering or annotation. scFair keeps a familiar global
ranking as the backbone and, by default, appends a small extension from the
same ranking so near-miss genes are not discarded.

Docs: [Read the Docs](https://scfair.readthedocs.io/en/latest/).

## What you need

- Python 3.10+ (tested 3.10–3.12)
- An AnnData object with raw integer counts in `.X` or `layers["counts"]`

## Install

```bash
pip install scfair
```

## First run

```python
import scfair as scf

# raw integer counts in .X or layers["counts"]
scf.pp.highly_variable_genes(adata)

adata = adata[:, adata.var["highly_variable"]].copy()
```

That is the default path: one global HVG pass, then a small append of the next
genes in the same ranking. Runtime is on the same order as
`scanpy.pp.highly_variable_genes`.

To let the data suggest how many genes to keep:

```python
scf.pp.highly_variable_genes(adata, n_top_genes="auto")
```

To match scanpy with no extension:

```python
scf.pp.highly_variable_genes(adata, balance_method="none")
```

## Status

**0.5.0 (Beta).** Import as `import scfair as scf` and use names in
`scfair.__all__` and `scf.pp`. See the
[API reference](https://scfair.readthedocs.io/en/latest/api/index.html).

## Next steps

1. [Installation](https://scfair.readthedocs.io/en/latest/installation.html)
2. [Quickstart](https://scfair.readthedocs.io/en/latest/quickstart.html)
3. [Tutorial: PBMC 10k misclustered cells vs standard HVG](https://scfair.readthedocs.io/en/latest/tutorials/index.html)
4. [FAQ](https://scfair.readthedocs.io/en/latest/faq.html) if something looks off

## License

Software: [Apache License 2.0](LICENSE).

## Author

**Zhao Li (李钊)**  
Email: [leelieber@gmail.com](mailto:leelieber@gmail.com)
