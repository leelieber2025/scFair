# Quickstart

Install the package, run one HVG selection, and continue a normal scanpy
pipeline.

## 1. Install

```bash
pip install scfair
```

```python
import scfair as scf
print(scf.__version__)
```

You need an AnnData object with **raw integer counts** in `.X` or
`layers["counts"]`. If counts live in another layer, pass `layer="..."`.

## 2. Select HVGs

```python
import scanpy as sc
import scfair as scf

scf.pp.highly_variable_genes(adata)

print(int(adata.var["highly_variable"].sum()), "genes selected")
print(adata.uns["scfair"]["hvg"].get("auto_message"))
```

Keep the full gene matrix. Scanpy PCA uses `var["highly_variable"]` as a mask.

Default behavior:

1. Rank genes with `flavor="seurat_v3"`.
2. Choose a base size `k` from the data (`n_top_genes="auto"`).
3. Add a short same-rank tail (`append_budget`, usually 200 genes).

Final size is about `k + 200`. Auto is slower than a fixed `k` because it
builds extra neighbor graphs.

### Common alternatives

```python
# locked paper / protocol (still appends ~200 genes)
scf.pp.highly_variable_genes(adata, n_top_genes=2000)

# exact scanpy top-2000 (no tail)
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")

# extra knobs (batch, store_raw, append size, …)
from scfair.pp import HVGOptions
scf.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    options=HVGOptions(batch_key="batch", append_budget=100),
)
```

## 3. Downstream

```python
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
```

## 4. What was written

| Where | Meaning |
|-------|---------|
| `adata.var["highly_variable"]` | Final gene mask |
| `adata.var["highly_variable_rank"]` | Rank among selected genes; NaN otherwise |
| `adata.var["scfair_score"]` | Score used for ranking |
| `adata.uns["scfair"]["hvg"]` | Options, chosen `k`, diagnosis tips |
| `adata.uns["scfair"]["hvg"]["auto_message"]` | One-line summary when auto ran |

## 5. Optional helpers

```python
# labels you already have
scf.pp.diagnose_from_labels(adata.obs["cell_type"])

# after sc.pp.neighbors(adata)
est = scf.pp.estimate_n_populations(adata)
print(est.n_populations, est.confidence)
```

These do not change the HVG list.

## Next

| Step | Page |
|------|------|
| Parameters and recipes | {doc}`user_guide/parameters` |
| How selection works | {doc}`user_guide/method` |
| PBMC notebooks | {doc}`tutorials/index` |
| FAQ | {doc}`faq` |
