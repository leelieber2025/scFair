# Parameters

This page summarizes the main arguments of
{func}`scfair.pp.highly_variable_genes`. Full signatures live in the
{doc}`../api/index`.

## Core arguments

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_top_genes` | `2000` | Base list size, or `"auto"` |
| `flavor` | `"seurat_v3"` | Global ranking method; same family as scanpy |
| `layer` | `None` | Counts layer; default prepares / uses `layers["counts"]` |
| `balance_method` | `"append"` | `"none"` for scanpy-like; `"hybrid"` / `"score"` / `"reweight"` are opt-in |
| `mode` | `"auto"` | Product size preset for auto / default budgets |
| `blend_global` | `0.95` | Used by `hybrid` |
| `marker_genes` / `marker_mode` | `None` | Optional forced markers |
| `diagnose` | `True` | Advisory tips in `uns`; never changes the gene list |
| `strict` | `False` | Raise instead of falling back when a dependency is missing |
| `random_state` | `0` | Reproducibility for clustering paths |
| `inplace` | `True` | Write results into `adata` |
| `subset` | `False` | If `True`, return a subset AnnData (and keep a recoverable gene universe) |
| `options` | `None` | An {class}`scfair.pp.HVGOptions` instance |

`resolution` and `min_cluster_size` apply only when you opt into a clustering
method (`hybrid` / `score` / related paths).

## Research knobs (`HVGOptions`)

Less common controls live on {class}`scfair.pp.HVGOptions` so the everyday call
stays short:

```python
from scfair.pp import HVGOptions
import scfair as scf

scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(append_budget=500),
)
```

Pass an `HVGOptions` instance, not a bare dict. Fields that used to be
top-level kwargs still work with a deprecation warning; prefer `options=`.

Useful fields:

| Field | Role |
|-------|------|
| `append_budget` | Genes appended after the frozen base (`None` → product default, usually 200). Explicit `0` or `N` is never overwritten by mode. |
| `n_top_min` / `n_top_max` | Bounds for automatic `n_top_genes` |
| `auto_n_method` | Strategy when `n_top_genes="auto"` (default `"structure"`) |
| `scale_clustering` | Scale genes before intermediate PCA (default `True`) |
| `cluster_pool` | Size of the gene pool used for intermediate clustering |
| `neighbor_contrast` | Extra weight for neighbor-contrast scoring (research) |
| `store_raw` | Optional second full-matrix snapshot in `uns` (default `False`) |
| `label_key` | Optional obs key for type-count detection in auto mode |

## Typical recipes

**Everyday preprocessing**

```python
scf.pp.highly_variable_genes(adata)
```

**Paper protocol that must match scanpy size exactly**

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
```

**Exploration when you do not want to pick `k`**

```python
scf.pp.highly_variable_genes(adata, n_top_genes="auto")
```

**Larger extension without re-ranking**

```python
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(append_budget=500),
)
```
