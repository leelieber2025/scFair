# Outputs

## Columns on `adata.var`

| Column | Meaning |
|--------|---------|
| `highly_variable` | Boolean mask of the final gene set |
| `highly_variable_rank` | Rank among selected genes; **NaN** if not selected (scanpy seurat_v3 style) |
| `scfair_score` | Score used for ranking; see the note in `uns` for the exact definition |

Global flavor-specific columns from scanpy (for example `variances_norm`) may
also be present after the global pass.

## Record in `adata.uns["scfair"]["hvg"]`

Every successful call writes a structured record, including:

- options actually used (including resolved `n_top_genes` and `append_budget`)
- method / balance path
- diagnosis tips when `diagnose=True`

```python
h = adata.uns["scfair"]["hvg"]
h["n_top_genes_used"]
h.get("diagnosis")
h.get("auto_n")  # present for the default auto path (and any auto call)
```

If a call fails after partial writes, scFair rolls back HVG columns when it can
and records `adata.uns["scfair"]["hvg_failed"]` so a caught exception does not
leave a false mask behind.

## Counts layer

For count-based flavors, scFair prepares a usable raw counts matrix (typically
`layers["counts"]`). It does not overwrite a user counts layer when that
layer disagrees with integer `.X`; staging uses an internal layer that is
removed after the call. Prefer an explicit `layer=` when you already know where
raw counts live.

## No raw-count matrix in `uns` (default)

By default the call does not write a second full counts table under
`adata.uns["scfair"]["raw_snapshot"]`. That sidecar is opt-in via
`options=HVGOptions(store_raw=True)` (or `"ondisk"`) when you need the
pre-subset gene universe after `subset=True`. Most analyses only need the HVG
mask and, if present, `layers["counts"]`.

## Planning helpers

| Function | Writes |
|----------|--------|
| {func}`scfair.pp.diagnose_from_labels` | A short imbalance report (does not modify the HVG mask) |
| {func}`scfair.pp.estimate_n_populations` | An advisory estimate in `uns` / return object |
