# How selection works

scFair is built around **two** problems. Everything else (flavors, research
knobs, cluster-aware modes) is secondary.

## Two problems

### Problem A — How many genes (`n`)?

Pipelines almost always need a gene budget. The usual answer is a fixed
integer (often 2000) copied from a tutorial. That is simple, but it is also
where unfamiliar users make silent mistakes: a short list can erase fine
structure; a long list adds cost and noise with little gain.

**What scFair does:** default **`n_top_genes="auto"`** estimates a base size
from the data’s density structure (multi-seed). 

**What we claim:** auto is a **sensible default and a prompt**, not a proof of
global optimality. It reduces the chance of shipping an arbitrary bad `n` when
you have not run a `k`-sweep. When density confidence is low, it prefers
classical **2000** over a risky short list. For papers and locked protocols,
pass an integer (e.g. `n_top_genes=2000`).

### Problem B — Unfair HVG allocation

Global HVG ranking measures variability across all cells. When a few cell types
dominate, the top of the list is filled by genes that separate those large
groups. Markers for smaller populations often sit just below a fixed cutoff.
They are not uninformative — they lose the **vote count** to bulk variation.
That is unfair **allocation of slots** in a fixed-length list, not a failure of
“finding variable genes” in the abstract.

**What scFair does:** keep a standard global ranking as the backbone, then by
default **`append`** a small extension of near-miss genes from the **same**
ranking so genes that barely missed the base cut are not discarded. The base
top-`k` is **frozen** — nothing is pushed out. The only other product method is
`"none"` (scanpy-like global HVG with no extension).

## Default path: auto `n` + fairer allocation (`append`)

1. Rank all genes with a standard global method (`flavor="seurat_v3"` by default).
2. **Problem A:** choose base size **`k`** with structure-aware auto (default).
   Low density confidence floors soft short lists to classical **2000**
   (except labeled true-SHORT paths). Pass `n_top_genes=2000` to skip auto.
3. Freeze the top `k` genes (familiar HVG backbone).
4. **Problem B:** append the next `append_budget` genes from the **same**
   ranking (ranks `k+1 … k+m`). Product default is **tight density**: floor
   **200** (never reduced for short/mid base `k`); when
   `n_top_genes="auto"`, budget may rise with structure
   `n_density_pops` as
   `m = max(200, min(300, 200 + max(0, n_need − 12) × 12))`. Explicit
   `HVGOptions(append_budget=N)` always wins.

Properties that matter in practice:

- No intermediate clustering and no re-ranking inside a pool (append path).
- Nothing is removed from the base top-`k` (allocation is “base + tail”).
- Final size is `k + append_budget` (capped at `n_vars`).
- Auto is multi-seed and slower than a fixed `k`; fixed `2000` is closer to
  one `scanpy.pp.highly_variable_genes` call.

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
