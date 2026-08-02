# User Guide

Read {doc}`../quickstart` first if you have not run scFair yet. This guide
covers the default method, the main parameters, and the opt-in alternatives.

## Which page do I need?

| Task | Page |
|------|------|
| Default append method and when to use it | {doc}`method` |
| Parameters, modes, and `HVGOptions` | {doc}`parameters` |
| Outputs written to `var` / `uns` | {doc}`outputs` |

## Default analysis steps

1. **Prepare** — AnnData with raw integer counts in `.X` or `layers["counts"]`.
2. **Select HVGs** — `scf.pp.highly_variable_genes(adata)`.
3. **Subset** — `adata[:, adata.var["highly_variable"]].copy()`.
4. **Continue** — PCA, neighbors, clustering, annotation as usual.

## Defaults worth remembering

| Topic | Usual practice |
|-------|----------------|
| Entry point | `scf.pp.highly_variable_genes` |
| Balance method | `"append"` (base top-`k` + extension) |
| Base size | `n_top_genes=2000` |
| Extension | about 200 genes (`append_budget`) |
| Scanpy-like only | `balance_method="none"` |
| Research knobs | `options=HVGOptions(...)` |

Stuck? {doc}`../faq`.

```{toctree}
:maxdepth: 2

method
parameters
outputs
```
