# How selection works

scFair is built around **list size** and a small optional buffer. Everything
else (flavors, diagnostics) is secondary. Cluster-aware reallocation methods
were removed from the product.

## What the product actually does

### 1 — How many genes (`n`)?

Pipelines almost always need a gene budget. **There is no a priori correct
length** — copying 2000 from a tutorial is simple and easy to get wrong
silently (too short erases structure; too long adds noise and cost).

**What scFair does:** default **`n_top_genes="auto"`** estimates a base size
`k` from the data’s density structure (multi-seed). That is the product
default **on purpose**: the first job is to **avoid a wrong `n`**, not to win
a wall-clock race against a fixed-int HVG call.

**What we claim:** auto is a **safer default than guessing**, not a proof of
global optimality. The extra compute is intentional and worth it when you
cannot predict a reasonable length. When density confidence is low, it still
prefers classical **2000** over a risky short list. Pass a fixed int only for
papers and locked protocols.

### 2 — List-length buffer (`append`)

Default **`balance_method="append"`** extends the **same** global ranking by
`append_budget` genes (ranks `k+1 … k+m`). The selected set is mathematically
**`top-(k+m)`** — equivalent to `balance_method="none"` with
`n_top_genes=k+m`. It is a **conservative list buffer**, not
population-aware reallocation of HVG slots.

Pass **`balance_method="none"`** (or `append_budget=0`) for pure top-`k`.

## Default path: auto `n` + list buffer

1. Rank all genes with a standard global method (`flavor="seurat_v3"` by default).
2. Choose base size **`k`** with structure-aware auto (default). Low density
   confidence floors soft short lists to classical **2000** (except labeled
   true-SHORT paths). Pass `n_top_genes=2000` to skip auto.
3. If append is on: take the next `append_budget` genes from the same ranking.
   Product default floor **200**; when `n_top_genes="auto"`, budget may rise
   with structure `n_density_pops` as
   `m = max(200, min(300, 200 + max(0, n_need − 12) × 12))`. Explicit
   `HVGOptions(append_budget=N)` always wins.

Properties that matter in practice:

- No intermediate clustering for gene selection.
- Final size is `k + append_budget` (capped at `n_vars`) when append is on.
- Auto is multi-seed and slower than a fixed `k`. That cost buys a
  data-informed list length when you would otherwise guess.

**Not GiniClust / CellSIUS.** Those methods score **rare** genes (normalized
Gini, cluster-wise tests, evidence communities). Product `append` only adds
near-miss genes from the **global** HVG list — slot fairness at a fixed
cutoff, not rare-subtype discovery. For a known missing type, force-include
markers (`marker_mode="force"`) rather than expecting auto rare compensation.

```python
import scfair as scf

scf.pp.highly_variable_genes(adata)  # auto + append
# scf.pp.highly_variable_genes(adata, n_top_genes=2000)  # fixed base
```

If the data has technical batches and you want scanpy's per-batch HVG merge on
that global ranking, pass `options=HVGOptions(batch_key="...")`. See
{doc}`parameters` and the FAQ entry on multi-batch data.

## Reproduce scanpy: fixed `k` + `balance_method="none"`

```python
scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
```

This is a single global HVG pass with no extension. Use it for drop-in
comparisons against existing protocols.

## Choosing `n_top_genes`

| Value | Behavior |
|-------|----------|
| `"auto"` (**default**) | Structure-aware base size `k`, then your `balance_method` (default: append) |
| int (e.g. **2000**) | Fixed base size; best for papers and fixed protocols |
| `"structure"` | Same size estimator as `"auto"`, exposed as an explicit name |

With the default `balance_method="append"`, auto only chooses **how many**
genes; it does **not** re-rank the list. That keeps the gene set a frozen
global top-`k` plus a small extension.

### What auto does under the hood

Auto builds a short multi-seed view of cell **density** in a low-dimensional
embedding (after neighbors / PCA), not a second gene-variance formula. Roughly:

- several clear dense cores → allow a longer or shorter list by rule, often
  near the classical 2000 band
- low density confidence → prefer classical **2000** rather than a short list
  (label-free safety)
- true multi-core short geometry **with** cell-type labels can still keep a
  short base (research / fine atlases)

Bounds: `n_top_min` / `n_top_max` (default 500–5000). Cost: several graph
builds, so large objects take longer than `n_top_genes=2000`.

After a call, `adata.uns["scfair"]["hvg"]["auto_n"]` records what happened:

| Field | Meaning |
|-------|---------|
| `strategy` | Which estimator ran (`"structure"` unless you changed `auto_n_method`) |
| `n_top_selected` | The base `k` it picked |
| `structure` | Density features (population count, valley depth, stability) |
| `rule_branch` | Internal tag for which rule fired — useful in bug reports |

**Known limitation:** on small matrices the estimator can hit `n_top_max`
and select almost every gene (see {doc}`../faq`). Prefer a fixed
`n_top_genes=2000` for locked paper protocols.

## Product modes

`mode` adjusts the default base size / append budget for broad regimes
(`compact`, `balanced`, `fine`, or `auto`). With a fixed integer
`n_top_genes`, mode does not rewrite the gene list; it mainly matters when you
use `"auto"` or leave size decisions to the product defaults. See
{doc}`parameters`.

## Markers

You can force known markers into the final set with `marker_genes` and
`marker_mode`. Read the function docstring before relying on this: injecting
markers is not free even when they are added outside the base count.
