# FAQ / Troubleshooting

First call: {doc}`quickstart`. Method: {doc}`user_guide/method`.

## What should I call?

| Goal | Call |
|------|------|
| Default HVG | {func}`scfair.pp.highly_variable_genes` |
| Locked protocol | same, with `n_top_genes=2000` |
| Match scanpy's gene set | `n_top_genes=2000, balance_method="none"` (and the same `batch_key` if any) |
| How many populations | {func}`scfair.pp.estimate_n_populations` after `sc.pp.neighbors` |
| Imbalance tips from labels | {func}`scfair.pp.diagnose_from_labels` |

## Is `append` different from a larger `k`?

Under the default filtering and marker settings, no. When enough genes are
available, `append` with base `k` and budget `m` selects the same genes as
`balance_method="none"` with `n_top_genes=k+m`. Auto chooses `k`, then the
tail is a fixed-length extension. Gene filters, forced markers, or a small
gene matrix can change the final set. Metadata records the base and tail
separately.

## Why is my list larger than `n_top_genes`?

Append is on by default. Final size is about `base_k + append_budget`
(floor **200**, up to **300** on auto when density cores are high).
Use `balance_method="none"` or `options=HVGOptions(append_budget=0)`
for a strict size.

## Can scFair drop genes from the top `k`?

No. `"append"` freezes the base. `"none"` is exact size `k`. Forced
`marker_genes` with `marker_extra=False` can displace the lowest-ranked
selected genes; that is a marker option, not append.

## Do I need to cluster first?

No. The default auto-size estimator runs an internal Leiden clustering on a
temporary representation, but you do not need to supply cluster labels or run
clustering beforehand. A fixed `n_top_genes` skips this structure-estimation
step.

## What counts matrix do I need?

Raw **integer** counts in `.X` or a counts layer. Log-normalized
matrices are the wrong input for `flavor="seurat_v3"`. If integer `.X`
and `layers["counts"]` both exist and disagree, pass `layer=`
explicitly; scFair will not overwrite your layer.

## How do I restore counts after `store_raw=True`?

```python
from scfair.pp import HVGOptions

scf.pp.highly_variable_genes(
    adata, options=HVGOptions(store_raw=True), subset=True
)
full = scf.pp.restore_raw_counts(adata, full_genes=True)
scf.pp.restore_raw_counts(adata, inplace=True)  # current genes, into .X
```

If you already stored full counts in `adata.raw` (the usual scanpy
pattern) and did not pass `store_raw=True`,
`restore_raw_counts(..., full_genes=True)` uses that `.raw` matrix.

By default the call does **not** write `uns["scfair"]["raw_snapshot"]`.

## Multi-batch data

`batch_key` is on {class}`scfair.pp.HVGOptions`:

```python
from scfair.pp import HVGOptions

scf.pp.highly_variable_genes(adata, options=HVGOptions(batch_key="batch"))
```

That value is forwarded to scanpy on the global HVG pass. scFair reuses
the merge order (`highly_variable_nbatches` + rank / dispersion), so
`balance_method="none"` matches scanpy for the same flavor and
`batch_key`. `seurat_v3` and `seurat_v3_paper` differ only when
`batch_key` is set. This is not embedding-level integration.

## Mitochondrial and ribosomal genes

Left in the list by default (`filter_mito` / `filter_ribo` are
`False`). Turn them on to drop matching symbols and refill from the
global ranking. Forced `marker_genes` are never removed.

Naming is inferred from `var_names` (Ensembl `ENSG` / `ENSMUSG`, mito
`MT-` vs `mt-`, ribo casing). Structural proteins match `RPL*` / `RPS*`
/ `Rpl*` / `Rps*`, not kinases such as `RPS6KA*`. Inference is stored
in `uns["scfair"]["hvg"]["gene_nomenclature"]`. Other species may come
back as `unknown`.

```python
from scfair.pp import HVGOptions

scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(filter_mito=True, filter_ribo=True),
)
# force the naming rules:
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(
        filter_mito=True, filter_ribo=True, gene_nomenclature="mouse"
    ),
)
```

## Forced markers

With `marker_genes` given, `marker_mode` defaults to `"force"`.
`marker_extra=True` (default) adds them on top of the selected list.

```python
scf.pp.highly_variable_genes(adata, marker_genes=["CD3D", "CD8A"])

# report only; list unchanged
scf.pp.highly_variable_genes(
    adata, marker_genes=["CD3D", "CD8A", "MS4A1"], marker_mode="none"
)
h = adata.uns["scfair"]["hvg"]
h["n_marker_genes"], h["n_marker_genes_already_selected"]
```

`options=HVGOptions(marker_extra=False)` folds markers into the
`n_top_genes` budget and may displace the lowest-ranked selected genes.

## `seurat_v3` failed / fell back to `seurat`

The default flavor needs `scikit-misc`. Install it, or set `strict=True`
to raise. Tiny gene panels can also trigger a fallback before loess
crashes.

## `inplace=False` and `subset=True`

`inplace=False` returns a DataFrame, so `subset=True` has nowhere to
land and is ignored (a warning is emitted). Use `inplace=True, subset=True`,
or apply `var["highly_variable"]` yourself.

## `"auto"` selected almost every gene

On small matrices the structure estimator can hit the ceiling. Pass a
fixed `n_top_genes=2000`. Check `uns["scfair"]["hvg"]["auto_n"]` for
the rule branch.

## Why is the default slower than scanpy?

`n_top_genes="auto"` builds multi-seed HVG → PCA → neighbors → density
graphs. That is often 10–150× one scanpy HVG pass. Progress prints on
stderr for auto once `n_obs >= 1000`. Pass a fixed int when the length
is already locked.

## How many populations without a resolution sweep?

```python
sc.pp.neighbors(adata)
est = scf.pp.estimate_n_populations(adata)
# est.n_populations, est.confidence
```

Needs a neighbor graph, or pass `embedding=`. On well-separated data it
tracks the true count up to about **20 populations**. Many tiny groups
(30 types × tens of cells) can be under-counted. The return object
answers *how many*, not *which cell goes where*. Stored under
`uns["scfair"]["granularity"]`.

## GiniClust, CellSIUS, mixHVG

GiniClust / CellSIUS hunt rare-subpopulation markers (Gini or
per-cluster tests). Default scFair only extends the same global HVG
ranking by a short tail. If a rare type is still unresolved, pass its
markers with `marker_genes`.

mixHVG mixes *method* rankings. scFair keeps one ranking.

## Is auto always better than 2000?

No. On many datasets it lands near 2,000. On some benchmark datasets a
hand-tuned `k` still performs better by ARI. For a manuscript, pass
`n_top_genes=` and report that number.

## Version and citation

```python
import scfair as scf
print(scf.__version__)
```

Pin it, for example `scfair==0.10.0`. Cite the
[preprint](https://www.biorxiv.org/content/10.64898/2026.08.08.743679v1)
(doi:[10.64898/2026.08.08.743679](https://doi.org/10.64898/2026.08.08.743679))
and the software record
[10.5281/zenodo.21761251](https://doi.org/10.5281/zenodo.21761251).
BibTeX: {doc}`citation`.
