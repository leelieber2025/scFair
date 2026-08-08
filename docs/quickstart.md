# Quickstart

Install the package, run one HVG selection, and read the main outputs.

scFair is aimed at problems you already face in every scRNA-seq workflow:

1. **How many HVGs (`n`)?** — nobody knows a safe length a priori. Default
   `"auto"` estimates a base size from the data so you do not ship a silent
   wrong `n` by copying `2000`. The extra compute is intentional. Use a fixed
   int only when a paper or protocol is already locked.
2. **List buffer near the cutoff** — default `"append"` extends the **same**
   global ranking by a short tail (`top-(k+m)`), not population-aware
   reallocation.

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

print(int(adata.var["highly_variable"].sum()), "genes selected")
# Keep the full matrix. Downstream scanpy PCA uses
# mask_var="highly_variable" by default when that column is present.
```

By default this is:

1. One global HVG ranking (`flavor="seurat_v3"`)
2. Choose a base size **`k` from the data** (`n_top_genes="auto"`) — often
   near 2000; when the density signal is weak, scFair prefers 2000 over a
   short list
3. Append the next `append_budget` genes from the **same** ranking.
   Default is a **tight density** rule: floor **200**, and when auto sees
   many density cores may rise up to **300**
   (`m = max(200, min(300, 200 + max(0, n_need − 12) × 12))`). Fixed-`k`
   calls stay at 200 unless you set `options=HVGOptions(append_budget=…)`.

Final list size is `k + append_budget`, capped at `n_vars`. The default append
path does not re-rank genes with intermediate clustering. Auto needs extra
graph builds and is slower than a fixed `k` — that is the price of not guessing
`n`.

See what auto picked:

```python
h = adata.uns["scfair"]["hvg"]
print(h.get("auto_message"))          # one plain-language line
print(h.get("n_top_genes_used"))      # base k
print(h.get("append_budget"))         # resolved m (often 200–300)
print(h.get("auto_n", {}).get("rule_branch"))  # optional diagnostics
print(h.get("auto_n", {}).get("append_budget_info"))  # density rule detail
```

### Fixed size (locked paper / protocol only)

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000)
```

### Match scanpy gene set (no append)

```python
# same flavor + n_top_genes + batch_key (if any) → same selected genes as scanpy
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
# multi-batch:
# scf.pp.highly_variable_genes(
#     adata, n_top_genes=2000, balance_method="none",
#     options=scf.pp.HVGOptions(batch_key="batch"),
# )
```

## 3. Downstream (keep the full matrix)

Do **not** subset to HVGs. Keep all genes; scanpy PCA uses
`mask_var="highly_variable"` when that column is present.

```python
import scanpy as sc

# after scf.pp.highly_variable_genes(adata) …
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)                       # uses highly_variable by default
sc.pp.neighbors(adata)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
# sc.tl.umap(adata)                    # optional
```

## 4. What to look at

```python
adata.var["highly_variable"]       # boolean mask
adata.var["highly_variable_rank"]  # rank among selected; NaN if not selected
adata.var["scfair_score"]          # ranking score used for selection
h = adata.uns["scfair"]["hvg"]     # options used, k, timings, diagnosis
print(h.get("auto_message"))       # plain summary when auto ran
```

| Field | Meaning |
|-------|---------|
| `var["highly_variable"]` | Final gene mask |
| `var["highly_variable_rank"]` | Finite ranks for selected genes only |
| `uns["scfair"]["hvg"]` | Full call record (method, k, diagnosis tips) |
| `uns["scfair"]["hvg"]["auto_message"]` | One-line plain English when `n_top_genes="auto"` |

Diagnosis (`diagnose=True`, default) writes advisory tips. It never changes
which genes are selected.

## 5. Optional planning helpers

```python
# Known cell-type labels: imbalance tips before you choose a method
scf.pp.diagnose_from_labels(adata.obs["cell_type"])

# Label-free: how many populations the density field supports
# needs sc.pp.neighbors(adata) run first (or pass embedding=...);
# without a graph it returns n_populations=None, confidence="none"
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

