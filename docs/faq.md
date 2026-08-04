# FAQ / Troubleshooting

First analysis: {doc}`quickstart`. Method detail: {doc}`user_guide/method`.

## What should I call?

| Goal | Function |
|------|----------|
| Everyday HVG selection | {func}`scfair.pp.highly_variable_genes` (default: auto + append) |
| Fixed classical list | Same function with `n_top_genes=2000` |
| Match scanpy with no extension | `n_top_genes=2000, balance_method="none"` |
| Pre-call imbalance tips from labels | {func}`scfair.pp.diagnose_from_labels` |
| Label-free population count estimate | {func}`scfair.pp.estimate_n_populations` |

## What two problems does scFair actually solve?

**1. How many genes (`n`)?**  
Default `n_top_genes="auto"` suggests a base list size from the data. We do
**not** claim this is always the optimal `n`. It is a **safe default and a
prompt** so users who will not run a `k`-sweep are less likely to pick a
silent bad length. Override with any integer when you have a protocol.

**2. Unfair HVG allocation?**  
A plain top-`k` global ranking is dominated by large populations; markers for
smaller types often miss the cut. Default `append` keeps that ranking but adds
a short tail of near-miss genes so the base top-`k` is never displaced. That
is allocation fairness on top of a standard HVG backbone — not a new variance
model.

See {doc}`user_guide/method` for the full story.

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
`k`) and `"append"` (base + tail) are product methods.

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
from `uns`, or when another tool expects that sidecar. Everyday pipelines can
leave the default alone.

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

That value is forwarded to `scanpy.pp.highly_variable_genes` on the **global**
HVG pass (and on the second global pass inside `balance_method="reweight"`).
Scanpy then selects HVGs within each batch and merges them — a lightweight way
to down-weight batch-private genes. See the scanpy docs for flavor-specific
merge rules (`seurat_v3` vs `seurat_v3_paper` differ only when `batch_key` is
set).

**What this is not**

- Not full batch correction (no Harmony / scVI / BBKNN-style integration).
- Intermediate Leiden used by opt-in `"hybrid"` / `"score"` still builds a
  **mixed-batch** graph; only the global ranking uses `batch_key`.
- Default `"append"` benefits automatically: the base top-`k` and the append
  extension share the same batch-aware ranking.

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

`n_top_genes="auto"` builds a few extra neighbor graphs to estimate list size.
That is intentional and slower than one scanpy HVG pass. For interactive work
or benchmarks, pass `n_top_genes=2000` (append still applies unless you set
`balance_method="none"`).

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

Pin the version in analysis code and manuscripts, for example `scfair==0.7.0`.

## Is auto always better than 2000?

**No.** Auto is a data-informed default so you are less likely to pick a bad
length by accident. On many datasets it lands near 2000; on some gold panels
a hand-tuned `k` still wins on ARI. For manuscripts and benchmarks, fix
`n_top_genes=` explicitly and report that number.
