# Quickstart

Install the package, run one HVG selection, and read the main outputs.

## 1. Install

```bash
pip install scfair
```

```python
import scfair as scf
print(scf.__version__)
```

### What data do you need?

An AnnData object with **raw integer counts** in `.X` or `layers["counts"]`.
If counts live elsewhere, pass `layer="..."`.

```python
# example: keep a dedicated counts layer for later steps
# adata.layers["counts"] = adata.X.copy()
```

## 2. Run the default path

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
print(int(adata.var["highly_variable"].sum()), "genes selected")
```

By default this is:

1. One global HVG ranking (`flavor="seurat_v3"`)
2. Freeze the top `n_top_genes` (default **2000**)
3. Append the next `append_budget` genes from the **same** ranking (default **200**)

Final list size is `n_top_genes + append_budget`, capped at the number of genes
you have. There is no intermediate clustering on this path.

### Match scanpy exactly

```python
scf.pp.highly_variable_genes(adata, balance_method="none")
```

### Let the data suggest `k`

```python
scf.pp.highly_variable_genes(adata, n_top_genes="auto")
h = adata.uns["scfair"]["hvg"]
print(h["n_top_genes_used"])
print(h.get("auto_n", {}).get("rule_branch"))
```

`"auto"` costs extra graph builds and is better for exploration than for a
locked paper protocol. Prefer a fixed `n_top_genes` when you need bit-for-bit
reproducible pipelines.

## 3. What to look at

```python
adata.var["highly_variable"]       # boolean mask
adata.var["highly_variable_rank"]  # rank among selected; NaN if not selected
adata.var["scfair_score"]          # ranking score used for selection
h = adata.uns["scfair"]["hvg"]     # options used, k, timings, diagnosis
```

| Field | Meaning |
|-------|---------|
| `var["highly_variable"]` | Final gene mask |
| `var["highly_variable_rank"]` | Finite ranks for selected genes only |
| `uns["scfair"]["hvg"]` | Full call record (method, k, diagnosis tips) |

Diagnosis (`diagnose=True`, default) writes advisory tips. It never changes
which genes are selected.

## 4. Optional planning helpers

```python
# Known cell-type labels: imbalance tips before you choose a method
scf.pp.diagnose_from_labels(adata.obs["cell_type"])

# Label-free: how many populations the density field supports
# (expects neighbors already computed, or computes them as needed)
est = scf.pp.estimate_n_populations(adata)
print(est.n_populations, est.confidence)
```

Both helpers are advisory and do not run HVG selection.

## Next

| Step | Page |
|------|------|
| Defaults and parameters | {doc}`user_guide/index` |
| PBMC 10k: misclustered cells vs standard HVG | {doc}`tutorials/pbmc10k_hvg_compare` |
| Full signatures | {doc}`api/index` |
| Common questions | {doc}`faq` |
