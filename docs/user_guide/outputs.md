# Outputs

## `adata.var`

| Column | Meaning |
|--------|---------|
| `highly_variable` | Final gene mask |
| `highly_variable_rank` | Rank among selected genes; **NaN** if not selected |
| `scfair_score` | Score used for ranking. `uns["scfair"]["hvg"]["scfair_score_note"]` names the statistic (`variances_norm` for `seurat_v3`). It is not a probability. |

Scanpy flavor columns (for example `variances_norm`) may also be present.

## `adata.uns["scfair"]["hvg"]`

Written on every successful call:

```python
h = adata.uns["scfair"]["hvg"]
h["n_top_genes_used"]          # base k
h["n_highly_variable_final"]   # |selected|, including the append tail
h.get("auto_message")
h.get("auto_n")
h.get("diagnosis")
```

If the call fails after partial writes, HVG columns are rolled back when
possible and `uns["scfair"]["hvg_failed"]` records the error.

## Counts layer

For count-based flavors, scFair prepares a usable raw counts matrix
(typically `layers["counts"]`). It does not overwrite a user counts
layer that disagrees with integer `.X`; staging uses an internal layer
that is removed after the call. Pass `layer=` when you already know
where the counts live.

No second full matrix is stored under `uns["scfair"]["raw_snapshot"]`
unless you pass `options=HVGOptions(store_raw=True)` (or `"ondisk"`).
Use that when `subset=True` and you later need the pre-subset gene
universe.

## Helpers

| Function | Effect |
|----------|--------|
| {func}`scfair.pp.diagnose_from_labels` | Returns an imbalance report. Does not write to `adata`. |
| {func}`scfair.pp.estimate_n_populations` | Returns a `GranularityEstimate` and stores it under `uns["scfair"]["granularity"]` (override with `key_added=`). |
