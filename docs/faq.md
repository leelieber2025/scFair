# FAQ / Troubleshooting

First analysis: {doc}`quickstart`. Method detail: {doc}`user_guide/method`.

## What should I call?

| Goal | Function |
|------|----------|
| Everyday HVG selection | {func}`scfair.pp.highly_variable_genes` |
| Match scanpy with no extension | Same function with `balance_method="none"` |
| Pre-call imbalance tips from labels | {func}`scfair.pp.diagnose_from_labels` |
| Label-free population count estimate | {func}`scfair.pp.estimate_n_populations` |

## Do I need to cluster first?

**No** for the default path. `balance_method="append"` is one global ranking
plus a small extension. Intermediate clustering is used only by opt-in methods
such as `"hybrid"`.

## Why is my list larger than `n_top_genes`?

Because the default appends extra genes. Final size is roughly
`n_top_genes + append_budget` (default 2000 + 200). Use
`balance_method="none"` or `options=HVGOptions(append_budget=0)` if you need a
strict size.

## Can scFair push genes *out* of the top 2000?

Not on the default `"append"` path. The base top-`k` is frozen. Cluster-aware
methods (`hybrid`, `score`, …) can reshuffle a candidate pool; that is why they
are opt-in.

## What counts matrix do I need?

Raw **integer** counts in `.X` or a counts layer. Log-normalized matrices are
not appropriate for `flavor="seurat_v3"`. If both integer `.X` and
`layers["counts"]` exist but disagree, scFair will not silently overwrite your
layer; pass `layer=` explicitly.

## `seurat_v3` failed / fell back to `seurat`

The default flavor needs `scikit-misc`. Install it (`pip install scikit-misc`)
or set `strict=True` to raise instead of falling back. Tiny gene panels can also
trigger a safe fallback before loess crashes.

## `"auto"` selected almost every gene

On small matrices, structure-based auto-`k` can hit the ceiling. Prefer a fixed
`n_top_genes` for protocols you will re-run. Check
`adata.uns["scfair"]["hvg"]["auto_n"]` for the rule branch.

## Is `hybrid` the same as mixHVG?

**No.** Other tools sometimes use “hybrid” for mixing *method rankings*.
scFair’s `"hybrid"` blends a global variability score with cluster specificity
inside one candidate pool. The shared word does not mean shared math.

## How do I report the version?

```python
import scfair as scf
print(scf.__version__)
```

Pin the version in analysis code and manuscripts, for example `scfair==0.6.0`.
