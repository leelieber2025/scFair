# Installation

Python 3.10–3.12. NumPy, SciPy, pandas, AnnData, and scanpy come in as
dependencies.

| Package | Why the floor matters |
|---------|------------------------|
| `scanpy>=1.10` | Intermediate Leiden uses `flavor="igraph"` |
| `scikit-misc>=0.3` | Default `flavor="seurat_v3"` needs `skmisc.loess` |
| `igraph>=0.10` | Backend for that Leiden call |
| `scikit-learn>=1.2` | Diagnostics and the structure estimator |

```bash
pip install scfair
# or
conda install -c conda-forge -c bioconda scfair
```

```python
import scfair as scf
print(scf.__version__)
```

The version string is `src/scfair/_version.py` (currently **0.10.0**).

```bash
git clone https://github.com/leelieber2025/scFair.git
cd scFair
pip install -e ".[dev]"
pip install -e ".[docs]"   # Sphinx build, optional
```

```python
import logging
logging.getLogger("scfair").setLevel(logging.INFO)
```

First call: {doc}`quickstart`.
