# FAQ / Troubleshooting

First analysis: {doc}`quickstart`. Method detail: {doc}`user_guide/method`.

## What should I call?

| Goal | Function |
|------|----------|
| Default HVG selection | {func}`scfair.pp.highly_variable_genes` (default: **auto** + append) |
| Locked paper / fixed protocol | Same function with `n_top_genes=2000` |
| Match scanpy gene set (no buffer) | `n_top_genes=2000, balance_method="none"` (+ same `batch_key` if any) |
| How many populations (no res sweep) | {func}`scfair.pp.estimate_n_populations` after `sc.pp.neighbors` |
| Pre-call imbalance tips from labels | {func}`scfair.pp.diagnose_from_labels` |

## What does scFair actually do?

**1. How many genes (`n`)?**  
There is no a priori correct list length. Default `n_top_genes="auto"`
estimates a base size from density structure. Auto is not always optimal;
override with a fixed int when the protocol is already locked.

**2. Hard cutoff on a global ranking?**  
A plain top-`k` global ranking is dominated by large populations; markers for
smaller types often miss the cut. Default `append` freezes that ranking and
adds a short **same-rank** tail of near-miss genes so the base top-`k` is never
displaced. Not a new variance model and not cluster-conditional reallocation
(those methods were removed). Use `balance_method="none"` for pure top-`k`.

See {doc}`user_guide/method` for the full story.

## Is `append` different from just taking a larger `k`?

**No, for the gene set.**  
`balance_method="append"` with base `k` and budget `m` selects the same genes
as `balance_method="none"` with `n_top_genes=k+m`. The difference is product
semantics: auto chooses `k` (problem A), then append adds a fixed-style tail
for cutoff near-misses (problem B), and metadata records base vs tail. It is
not a per-population quota algorithm.

## How do I restore counts after `store_raw=True`?

```python
scf.pp.highly_variable_genes(adata, options=scf.pp.HVGOptions(store_raw=True), subset=True)
full = scf.pp.restore_raw_counts(adata, full_genes=True)  # new object, full gene axis
# or put counts back into .X for current genes:
scf.pp.restore_raw_counts(adata, inplace=True)
```

## Do I need to cluster first?

**No.** After auto chooses a base size, `append` is one global ranking plus a
small extension — no Leiden for gene selection.

## Why is my list larger than `n_top_genes`?

Because the default appends extra genes. Final size is roughly
`base_k + append_budget` (auto base often ~2000; append floor **200**, up to
**300** when structure density cores are high). Use
`balance_method="none"` or `options=HVGOptions(append_budget=0)` if you need a
strict size.

## Can scFair push genes *out* of the top 2000?

**No.** The base top-`k` is frozen on `"append"`. Only `"none"` (exact size
`k`) and `"append"` (base + tail) are available.

## Are mitochondrial and ribosomal genes removed from HVGs?

**No, not by default.** `HVGOptions.filter_mito` and `filter_ribo` default to
`False` so the global ranking is left as-is (closer to a plain scanpy HVG
list). When you turn them on, scFair **auto-detects** human vs mouse naming
from `var_names` (Ensembl `ENSG` / `ENSMUSG`, mito `MT-` vs `mt-`, ribo
casing), then drops matching mitochondrial symbols and ribosomal structural
proteins (`RPL*` / `RPS*` / `Rpl*` / `Rps*`; not kinases like `RPS6KA*`) and
refills from the global ranking. Forced `marker_genes` are never removed.

Inference is recorded in `adata.uns["scfair"]["hvg"]["gene_nomenclature"]`
when filters run. For other species, inference may return `unknown` and you
get a short tip if nothing matched.

```python
from scfair.pp import HVGOptions

# drop MT/ribo from the final HVG set (opt-in)
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(filter_mito=True, filter_ribo=True),
)
# force naming rules if needed:
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(
        filter_mito=True,
        filter_ribo=True,
        gene_nomenclature="mouse",  # or "human"
    ),
)
```

## How do I add my own genes to the HVG list?

Pass `marker_genes=[...]`. With markers given, `marker_mode` defaults to
`"force"`: those genes are guaranteed present in the final output regardless
of where they rank in the global HVG pass.

```python
import scfair as scf

scf.pp.highly_variable_genes(adata, marker_genes=["CD3D", "CD8A"])
adata.var.loc[["CD3D", "CD8A"], "highly_variable"]   # both True
```

Check first, without forcing anything, by passing `marker_mode="none"`
explicitly — the call records how many of your candidates were *already*
selected, and the gene list is unaffected:

```python
scf.pp.highly_variable_genes(
    adata,
    marker_genes=["CD3D", "CD8A", "MS4A1"],
    marker_mode="none",
)
h = adata.uns["scfair"]["hvg"]
h["n_marker_genes"], h["n_marker_genes_already_selected"]   # e.g. 3, 1
```

## Does `marker_genes` always add on top of `n_top_genes`, or can it replace genes?

**Adds on top, by default.** `HVGOptions.marker_extra` defaults to `True`, so
forced `marker_genes` (`marker_mode="force"`) are added *in addition to* the
`n_top_genes` selection — nothing is displaced. This is what you want for the
common workflow of noticing a cell type didn't resolve well, adding its known
markers, and re-clustering. If you explicitly set
`options=HVGOptions(marker_extra=False)`, the behavior silently flips: markers
are folded *into* the `n_top_genes` budget instead, potentially displacing the
lowest-ranked algorithm-selected genes. Not a bug — just easy to miss if
you copy an example that sets it.

```python
from scfair.pp import HVGOptions

# markers fold into the n_top_genes budget instead of extending it
scf.pp.highly_variable_genes(
    adata,
    marker_genes=["CD3D", "CD8A"],
    options=HVGOptions(marker_extra=False),
)
```

## What counts matrix do I need?

Raw **integer** counts in `.X` or a counts layer. Log-normalized matrices are
not appropriate for `flavor="seurat_v3"`. If both integer `.X` and
`layers["counts"]` exist but disagree, scFair will not silently overwrite your
layer; pass `layer=` explicitly.

## Does scFair keep a raw counts matrix in `uns`?

**No, not by default.** After HVG finishes, only `layers["counts"]` (when
needed) and the usual `var` / `uns["scfair"]["hvg"]` records remain. A second
full matrix under `uns["scfair"]["raw_snapshot"]` is discarded at the end of the
call so `write_h5ad` does not triple disk use for most users.

To keep the sidecar deliberately:

```python
from scfair.pp import HVGOptions

scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(store_raw=True),  # or store_raw="ondisk" + snapshot_path=
)
```

Use this when you call `subset=True` and later need the pre-subset gene universe
from `uns`, or when another tool expects that sidecar. Most pipelines can leave
the default alone.

## Multi-batch data: how do I pass `batch_key`?

scFair does not put `batch_key` on the short public signature. Use
{class}`scfair.pp.HVGOptions` and point it at a column in `adata.obs`:

```python
from scfair.pp import HVGOptions
import scfair as scf

# adata.obs["batch"] labels technical batches / samples / donors
scf.pp.highly_variable_genes(
    adata,
    options=HVGOptions(batch_key="batch"),
)
```

That value is forwarded to `scanpy.pp.highly_variable_genes` on the global HVG
pass. Scanpy selects HVGs within each batch and merges them
(`highly_variable_nbatches` + rank / dispersion). scFair **reuses that merge
order** for the final list (including default `"append"`), so with
`balance_method="none"` the selected gene set matches scanpy for the same
flavor and `batch_key`. See the scanpy docs for flavor-specific merge rules
(`seurat_v3` vs `seurat_v3_paper` differ only when `batch_key` is set).

**What this is not**

- Not full batch correction (no Harmony / scVI / BBKNN-style integration).
- Default `"append"` uses the same batch-aware ranking for base top-`k` and the
  tail (`top-(k+m)`).

Leave `batch_key=None` (default) when there is no meaningful batch column.

## `seurat_v3` failed / fell back to `seurat`

The default flavor needs `scikit-misc`. Install it (`pip install scikit-misc`)
or set `strict=True` to raise instead of falling back. Tiny gene panels can also
trigger a safe fallback before loess crashes.

## `"auto"` selected almost every gene

On small matrices, structure-based auto-`k` can hit the ceiling. Prefer a fixed
`n_top_genes=2000` for protocols you will re-run. Check
`adata.uns["scfair"]["hvg"]["auto_n"]` for the rule branch.

## Why is the default slower than scanpy?

Because the default is **`n_top_genes="auto"`**, not a fixed `2000`.

Auto builds multi-seed HVG→PCA→neighbors→density graphs to estimate list size.
That is much slower than one scanpy HVG pass (often ~10–150×). Progress messages
print on stderr once `n_obs >= 1000` under auto.

Pass a fixed int when a paper or protocol already locks the length (append
still applies unless you set `balance_method="none"`).

## How many populations without sweeping Leiden?

```python
sc.pp.neighbors(adata)
est = scf.pp.estimate_n_populations(adata)
# est.n_populations, est.confidence
```

Requires a neighbour graph (or pass `embedding=`). On well-separated data it
tracks the true count well up to roughly **~20 populations**; with many tiny
groups (e.g. 30 types × tens of cells each) it can under-count. It answers
*how many*, not *which cell goes where*.

## How does this differ from GiniClust or CellSIUS?

**Different problem.** GiniClust / CellSIUS hunt **rare-subpopulation markers**
(Gini or per-cluster tests, usually after a coarse partition). Default scFair
**`append`** only extends the **same global HVG ranking** by a short tail
(`k+1 … k+m`). No Gini LOESS pool, no cluster-wise FDR, no co-expression
communities.

**Rare types still poorly resolved?** Pass known markers with
`marker_genes=…`, `marker_mode="force"` (additive when `marker_extra=True`,
the default) and re-cluster.

## Is scFair the same as mixHVG?

**No.** mixHVG mixes *method rankings*. scFair keeps a single global ranking
and (by default) appends near-miss genes. Older scFair releases offered a
cluster-aware `"hybrid"` re-rank; that is not in the product API and is
unrelated to mixHVG’s hybrid.

## How do I report the version?

```python
import scfair as scf
print(scf.__version__)
```

Pin the version in analysis code and manuscripts, for example `scfair==0.8.0`.

## Is auto always better than 2000?

**No.** Auto is a data-informed default. On many datasets it lands near 2000;
on some gold panels a hand-tuned `k` still wins on ARI. For manuscripts and
benchmarks, fix `n_top_genes=` explicitly and report that number.
