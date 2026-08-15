# Quickstart

```bash
pip install scfair
```

```python
import scanpy as sc
import scfair as scf

scf.pp.highly_variable_genes(adata)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
```

`adata` must hold **raw integer counts** in `.X` or `layers["counts"]`.
Pass `layer=` if they live somewhere else. Do not subset to HVGs;
`sc.tl.pca` uses `var["highly_variable"]`.

```python
print(int(adata.var["highly_variable"].sum()), "genes")
print(adata.uns["scfair"]["hvg"].get("auto_message"))
```

Default path: `flavor="seurat_v3"` ranking, `n_top_genes="auto"` for the
base size `k`, then about 200 genes from the same ranking. Final size is
`k + append_budget`. Auto is slower than a fixed `k`.

```python
# protocol already fixes the length (still appends)
scf.pp.highly_variable_genes(adata, n_top_genes=2000)

# exact top-2000
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")

# batch-aware ranking, or a shorter tail
from scfair.pp import HVGOptions
scf.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    options=HVGOptions(batch_key="batch", append_budget=100),
)
```

## Written to `adata`

| Location | Meaning |
|----------|---------|
| `var["highly_variable"]` | Final mask |
| `var["highly_variable_rank"]` | Rank among selected genes; NaN otherwise |
| `var["scfair_score"]` | Score used for that ranking |
| `uns["scfair"]["hvg"]` | Resolved options, base `k`, diagnosis |
| `uns["scfair"]["hvg"]["auto_message"]` | One-line reason for the auto `k` |

## Optional helpers

These do not change the gene list.

```python
scf.pp.diagnose_from_labels(adata.obs["cell_type"])

sc.pp.neighbors(adata)
est = scf.pp.estimate_n_populations(adata)
est.n_populations, est.confidence
```

More knobs: {doc}`user_guide/parameters`. Worked examples: {doc}`tutorials/index`.
