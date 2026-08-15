# User Guide

```python
import scfair as scf
from scfair.pp import HVGOptions

scf.pp.highly_variable_genes(adata)
# scf.pp.highly_variable_genes(adata, n_top_genes=2000)
# scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
# scf.pp.highly_variable_genes(adata, options=HVGOptions(batch_key="batch"))
```

Raw integer counts in `.X` or `layers["counts"]`. Keep the full gene
matrix; `sc.tl.pca` uses `var["highly_variable"]`.

| Topic | Page |
|-------|------|
| Why auto + append, and what the estimator does | {doc}`method` |
| Arguments and `HVGOptions` | {doc}`parameters` |
| Columns on `var` / keys in `uns` | {doc}`outputs` |

| Default | Meaning |
|---------|---------|
| `n_top_genes="auto"` | Base size `k` from density structure. Pass `2000` to skip that step. |
| `balance_method="append"` | Global top-`k` plus a same-rank tail. Set equals `top-(k+m)`. |
| `append_budget` | Floor 200. On auto, may rise to 300 with the density-core count. |
| Scanpy-sized list | `n_top_genes=2000, balance_method="none"` |

```{toctree}
:maxdepth: 2

method
parameters
outputs
```
