# Parameters

Main arguments of {func}`scfair.pp.highly_variable_genes`. Signatures:
{doc}`../api/index`.

## Core arguments

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_top_genes` | `"auto"` | Structure-aware base size. Pass an int to lock it. |
| `flavor` | `"seurat_v3"` | Global ranking; same family as scanpy |
| `layer` | `None` | Counts layer. Default prepares / uses `layers["counts"]` |
| `balance_method` | `"append"` | `"append"` (base + tail) or `"none"` (exact size) |
| `mode` | `"auto"` | Size preset for auto / default budgets |
| `marker_genes` / `marker_mode` | `None` | Optional forced markers |
| `diagnose` | `True` | Tips in `uns`; does not change the gene list |
| `strict` | `False` | Raise instead of falling back |
| `progress` | `None` | Stage messages on stderr. `None` turns them on for auto once `n_obs >= 1000`, and for any call that is large (`n_obs >= 10_000` or `n_obs * n_vars >= 5e7`) |
| `random_state` | `0` | Structure auto |
| `inplace` | `True` | Write into `adata` |
| `subset` | `False` | Subset genes after selection |
| `options` | `None` | An {class}`scfair.pp.HVGOptions` instance |

## `HVGOptions`

Secondary knobs live here so the top-level signature stays short.
Pass an instance, not a dict. Top-level kwargs such as `append_budget=200`
are rejected.

```python
from scfair.pp import HVGOptions
import scfair as scf

scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(append_budget=500),
)
```

| Field | Role |
|-------|------|
| `append_budget` | Tail length after the frozen base. `None` → floor 200; on auto, may rise as `max(200, min(300, 200 + max(0, n_density_pops − 12) × 12))`. An explicit `0` or `N` is never overridden. |
| `n_top_min` / `n_top_max` | Bounds for automatic `n_top_genes` |
| `structure_n_seeds` | Multi-seed count (`None` → 3; `1` is faster and less stable) |
| `store_raw` | Keep a full raw-count sidecar in `uns["scfair"]["raw_snapshot"]`. Default `False`. `True` = inline; `"ondisk"` needs `snapshot_path` |
| `label_key` | Optional `obs` key for type-count detection in auto |
| `batch_key` | `obs` column for scanpy's per-batch HVG merge on the global ranking |
| `filter_mito` / `filter_ribo` | Default `False`. Set `True` to drop MT / ribosomal structural proteins and refill (markers are kept) |
| `gene_nomenclature` | `None` → infer `human` / `mouse` / `mixed` / `unknown` from gene names |

### `batch_key`

Forwarded to `scanpy.pp.highly_variable_genes`. scFair reuses that merge
order for `"append"` and `"none"`. This is not Harmony / scVI / BBKNN.

```python
scf.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    options=HVGOptions(batch_key="batch"),
)
```

Resolved value: `adata.uns["scfair"]["hvg"]["batch_key"]`.

### `marker_genes`

With markers given, `marker_mode` defaults to `"force"`:

```python
scf.pp.highly_variable_genes(adata, marker_genes=["CD3D", "CD8A"])
```

Forced markers are added on top of `n_top_genes` (`marker_extra=True`).
`marker_mode="none"` leaves the list alone and records how many
candidates were already selected. See {doc}`../faq`.

## Recipes

```python
scf.pp.highly_variable_genes(adata)

scf.pp.highly_variable_genes(adata, n_top_genes=2000)

scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")

scf.pp.highly_variable_genes(adata, options=HVGOptions(append_budget=500))

scf.pp.highly_variable_genes(adata, options=HVGOptions(batch_key="batch"))

scf.pp.highly_variable_genes(adata, marker_genes=["CD3D", "CD8A"])
```
