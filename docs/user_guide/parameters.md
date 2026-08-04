# Parameters

This page summarizes the main arguments of
{func}`scfair.pp.highly_variable_genes`. Full signatures live in the
{doc}`../api/index`.

## Core arguments

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_top_genes` | `"auto"` | Structure-aware base size (default); pass an int (e.g. `2000`) for a fixed list |
| `flavor` | `"seurat_v3"` | Global ranking method; same family as scanpy |
| `layer` | `None` | Counts layer; default prepares / uses `layers["counts"]` |
| `balance_method` | `"append"` | `"append"` (base + tail) or `"none"` (scanpy-like exact size) |
| `mode` | `"auto"` | Product size preset for auto / default budgets |
| `marker_genes` / `marker_mode` | `None` | Optional forced markers |
| `diagnose` | `True` | Advisory tips in `uns`; never changes the gene list |
| `strict` | `False` | Raise instead of falling back when a dependency is missing |
| `random_state` | `0` | Reproducibility for structure auto |
| `inplace` | `True` | Write results into `adata` |
| `subset` | `False` | If `True`, subset genes after selection |
| `options` | `None` | An {class}`scfair.pp.HVGOptions` instance |

## Secondary knobs (`HVGOptions`)

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
| `append_budget` | Genes appended after the frozen base. `None` → **floor 200**; with `n_top_genes="auto"`, may rise as `max(200, min(300, 200 + max(0, n_density_pops − 12) × 12))`. Explicit `0`/`N` never overridden. |
| `n_top_min` / `n_top_max` | Bounds for automatic `n_top_genes` |
| `auto_n_method` | Product auto is structure-only (default `"structure"`) |
| `structure_n_seeds` | Multi-seed count for structure auto (`None` → product default 3; use `1` for a faster pass) |
| `store_raw` | Opt-in: keep a full raw-count sidecar in `uns['scfair']['raw_snapshot']` after HVG (default `False`). `True` = inline; `"ondisk"` needs `snapshot_path` |
| `label_key` | Optional obs key for type-count detection in auto mode |
| `batch_key` | Optional `obs` column for scanpy-style per-batch HVG merge on the **global** ranking (default `None`). See below. |
| `filter_mito` / `filter_ribo` | **Default `False`**: keep MT / ribo in the global ranking. Set `True` to drop them and refill (markers kept). |
| `gene_nomenclature` | `None` (default) = **auto** detect `human` / `mouse` / `mixed` / `unknown` from gene names; optional force for mito rules. |

### Multi-batch: `batch_key`

When cells come from several technical batches (or samples you treat as
batches), pass the `obs` column name via `HVGOptions`. scFair forwards it to
scanpy on the global HVG pass — the same lightweight merge as
`scanpy.pp.highly_variable_genes(..., batch_key=...)`:

```python
from scfair.pp import HVGOptions
import scfair as scf

scf.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    options=HVGOptions(batch_key="batch"),
)
```

Effects and limits:

- **Applies to** the global ranking used by `"append"` and `"none"`.
- **Does not** run batch-corrected PCA / integration. For full integration,
  correct embeddings after HVG selection.
- Resolved value is recorded in `adata.uns["scfair"]["hvg"]["batch_key"]`.

If you do not have a batch column, leave the default (`None`).

## Typical recipes

**Everyday preprocessing** (auto size + append)

```python
scf.pp.highly_variable_genes(adata)
```

**Faster fixed size** (skip auto)

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000)
```

**Paper protocol that must match scanpy size exactly**

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
```

**Larger extension without re-ranking**

```python
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(append_budget=500),
)
```

**Multi-batch data (per-batch HVG merge on the global ranking)**

```python
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(batch_key="batch"),
)
```
