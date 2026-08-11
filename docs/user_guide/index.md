# User Guide

Read {doc}`../quickstart` first if you have not run scFair yet. This guide
covers the default method, the main parameters, and the opt-in alternatives.

## Two problems scFair targets

| # | Problem | Default answer | Scope |
|---|---------|----------------|-------|
| **A** | **How many genes (`n`)?** No universal safe length; users often copy 2000. | `n_top_genes="auto"` | Data-informed base size from density structure. Not a proof of the best `n`. Extra multi-seed cost vs fixed `k`. |
| **B** | **Hard top-`k` cutoff** Large types fill the list; small-type markers often miss the cut. | `balance_method="append"` | Freezes global top-`k`, adds same-rank near-miss tail. Set = `top-(k+m)`. Not cluster-conditional reallocation. |

Details and caveats: {doc}`method`.

## Which page do I need?

| Task | Page |
|------|------|
| Why auto + append, and how selection works | {doc}`method` |
| Parameters, modes, and `HVGOptions` | {doc}`parameters` |
| Outputs written to `var` / `uns` | {doc}`outputs` |

## Default analysis steps

1. **Prepare** — AnnData with raw integer counts in `.X` or `layers["counts"]`.
2. **Select HVGs** — `scf.pp.highly_variable_genes(adata)` (auto `n` + append).
3. **Continue** — PCA, neighbors, clustering, annotation as usual.
   Keep the full gene matrix; `sc.tl.pca` uses `var["highly_variable"]` as a
   mask by default (no `adata = adata[:, …]` subset step).

## Defaults worth remembering

| Topic | Usual practice |
|-------|----------------|
| Entry point | `scf.pp.highly_variable_genes` |
| Problem A (`n`) | `"auto"` from structure; pass `2000` for a fixed base |
| Problem B (cutoff) | `"append"`: freeze top-`k`, same-rank tail → `top-(k+m)` |
| Extension | floor 200 genes (`append_budget`); auto may raise to ≤300 via density cores |
| Match scanpy | `n_top_genes=2000, balance_method="none"` |
| Secondary knobs | `options=HVGOptions(...)` |

Stuck? {doc}`../faq`.

```{toctree}
:maxdepth: 2

method
parameters
outputs
```
