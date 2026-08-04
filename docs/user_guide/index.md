# User Guide

Read {doc}`../quickstart` first if you have not run scFair yet. This guide
covers the default method, the main parameters, and the opt-in alternatives.

## Two problems scFair targets

| # | Problem | Default answer | Honest scope |
|---|---------|----------------|--------------|
| **A** | **How many genes (`n`)?** Users copy 2000 or guess. | `n_top_genes="auto"` | A data-informed **default / prompt**, not a proof of the best `n` for every study. Avoids the worst silent mistakes. |
| **B** | **Unfair HVG allocation** Large types fill the list; rare markers miss the cut. | `balance_method="append"` | Freezes the global top-`k`, then adds near-miss genes. Does not invent a new variance formula. |

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
| Problem A (`n`) | `"auto"` from the data; pass `2000` for a fixed list |
| Problem B (allocation) | `"append"` (freeze base top-`k`, then extend) |
| Extension | floor 200 genes (`append_budget`); auto may raise to ≤300 via density cores |
| Match scanpy | `n_top_genes=2000, balance_method="none"` |
| Research knobs | `options=HVGOptions(...)` |

Stuck? {doc}`../faq`.

```{toctree}
:maxdepth: 2

method
parameters
outputs
```
