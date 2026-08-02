# Installation

## Requirements

- Python 3.10+ (CI covers 3.10–3.12)
- A normal scientific stack (NumPy, SciPy, pandas, AnnData, scanpy) — installed
  as dependencies

Runtime floors that matter in practice:

| Dependency | Why it is required |
|------------|--------------------|
| `scanpy>=1.10` | Intermediate Leiden uses `flavor="igraph"` (not available in 1.9) |
| `scikit-misc>=0.3` | Default `flavor="seurat_v3"` needs `skmisc.loess` |
| `igraph>=0.10` | Backend for `sc.tl.leiden(flavor="igraph")` |
| `scikit-learn>=1.2` | Direct imports for diagnostics and structure tools |

## Install

**PyPI (recommended):**

```bash
pip install scfair
```

## Check the install

```python
import scfair as scf
print(scf.__version__)
```

## After install

| Step | Page |
|------|------|
| First analysis | {doc}`quickstart` |
| Options and outputs | {doc}`user_guide/index` |
| PBMC 10k comparison | {doc}`tutorials/index` |
| Troubleshooting | {doc}`faq` |

## Development install

```bash
git clone https://github.com/leelieber2025/scFair.git
cd scFair
pip install -e ".[dev]"
# optional: docs build tools
pip install -e ".[docs]"
```

## Versioning

The single source of truth is `src/scfair/_version.py`. Runtime
`scfair.__version__`, packaging metadata, and the docs release string all read
it. For a release: bump `__version__`, update `CHANGELOG.md`, then
`python -m build` or the GitHub **Publish to PyPI** workflow.

## Logging

```python
import logging
logging.getLogger("scfair").setLevel(logging.INFO)
```
