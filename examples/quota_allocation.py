#!/usr/bin/env python
"""Equal per-cluster quota vs the shipped global re-rank, at a fixed k=2000.

The question
------------
Today's `hybrid` reserves **no** slots per cluster. It takes the global top
``2 x n_top`` as a candidate pool, re-ranks the *whole* pool by
``0.95 * norm(global) + 0.05 * norm(spec)`` and keeps the top ``n_top``
(`_highly_variable_genes.py:265-271`). Specificity enters only through the
cross-cluster sum ``S_g = sum_c w_c * logFC+_{g,c}`` with ``w_c ~ n_c**0.5``, so
a gene specific to one small cluster competes against genes specific to big ones
through that weight alone. This is open question 8 ("soft per-cluster gene
floor").

The arms
--------
Both arms are cut from **the same partition and the same candidate pool**, so
this measures allocation and nothing else:

- ``rank`` — the shipped default. ``scfair.pp.highly_variable_genes`` at
  ``n_top_genes=2000``, everything else at defaults.
- ``quota`` — the same intermediate partition, the same global top-4000 pool,
  but the 2000 slots are dealt out round-robin over clusters from each
  cluster's own one-sided ``logFC+`` order (`_build_cluster_gene_ranks`), a
  gene taken once, until 2000 are filled. Collisions are absorbed by carrying
  on down that cluster's list (option (a)), never by falling back to the global
  rank. The starting cluster rotates each round so no cluster wins ties by
  virtue of its label.
- ``mix70`` — the first 70% (1400) of ``rank``'s own selection in its blended
  score order, then the remaining 600 dealt by the same quota round-robin,
  **skipping genes the first 70% already took**, so every quota slot buys a
  gene the default would not have chosen. Pure ``quota`` was a coin flip
  per-population and lost on ``min_pop_f1``; this asks whether a minority
  reservation on top of the default ranking does better than either endpoint.

``rank`` / ``mix70`` / ``quota`` are the 100/0, 70/30 and 0/100 points of one
line, measured on the same partitions and seeds so the three are paired.

Not varied here: the pool (both arms draw from the global top 4000 — a
whole-transcriptome quota is a *different* experiment, since it would change
the pool and the allocation at once), and size-weighted quotas (``n_c**beta``),
which are one line away if equal allocation shows anything.

Reading it
----------
The headline is **per-population**: which labelled population gets separated
more cleanly under each arm. `quota_allocation_pops.csv` is one row per
(dataset, arm, seed, resolution, population). Pooled ARI / macro_f1 /
min_pop_f1 are background in `quota_allocation.csv`.

Scored over a resolution grid and summarised by the mean over resolution
(§5.19.4). Resumable per (dataset, seed).

Outputs (examples/results/):
  quota_allocation.csv       one row per (dataset, arm, seed, resolution)
  quota_allocation_pops.csv  per-population F1, long format
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402
from scfair.pp._highly_variable_genes import _build_cluster_gene_ranks  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "quota_mix.csv"
POPS = OUT / "quota_mix_pops.csv"

K = 2000
RANK_FRAC = 0.70  # mix70: share of the k slots kept from the default ranking
POOL_MULT = 2  # hybrid's candidate pool is the global top 2*k
SEEDS = [0, 1, 2, 3, 4]
RES_GRID = [0.3, 0.5, 0.8, 1.2]
MIN_CLUSTER_SIZE = 30  # the public default

ORDER = [
    "paul15",
    "pbmc3k_louvain",
    "pancreas_smartseq2",
    "duo4_pbmc",
    "duo8_pbmc",
    "duo4un_pbmc",
    "pbmc5k_adt29",
    "pbmc10k_adt14",
    "sln_208_mouse",
    "pbmc_seurat_v4_20k",
]


def global_pool(a, n_pool):
    """The global top-`n_pool` genes: hybrid's candidate pool, same flavor pass."""
    p = a.copy()
    sc.pp.highly_variable_genes(
        p,
        n_top_genes=min(n_pool, p.n_vars - 1),
        flavor="seurat_v3",
        layer="counts",
        span=0.3,
        inplace=True,
        subset=False,
    )
    return [str(g) for g in p.var_names[p.var["highly_variable"]]]


def quota_select(ranks, pool, k, preselected=None):
    """Round-robin the k slots over clusters, each from its own logFC+ order.

    `ranks` maps cluster -> full gene order. Restricted to `pool`, dealt one
    gene per cluster per round, duplicates skipped by advancing that cluster's
    own cursor. The round's starting cluster rotates so ties are not settled by
    cluster label.

    `preselected` seeds the output (mix70: the 70% kept from the default
    ranking). Those genes count against `k` and are excluded from the quota
    lists, so the rounds only ever deal genes the prefix did not already take.
    """
    keep = set(pool)
    lists = {c: [g for g in order if g in keep] for c, order in ranks.items()}
    lists = {c: g for c, g in lists.items() if g}
    if not lists:
        return [], {}
    cl = sorted(lists)
    cursor = {c: 0 for c in cl}
    taken: list[str] = list(preselected or [])
    seen: set[str] = set(taken)
    per_cluster = {c: 0 for c in cl}
    rnd = 0
    while len(taken) < k:
        progressed = False
        for i in range(len(cl)):
            c = cl[(rnd + i) % len(cl)]
            lst = lists[c]
            j = cursor[c]
            while j < len(lst) and lst[j] in seen:
                j += 1
            cursor[c] = j
            if j >= len(lst):
                continue
            g = lst[j]
            cursor[c] = j + 1
            taken.append(g)
            seen.add(g)
            per_cluster[c] += 1
            progressed = True
            if len(taken) >= k:
                break
        rnd += 1
        if not progressed:  # every cluster exhausted
            break
    return taken, per_cluster


def evaluate(a, genes, seed, rows, pop_rows, base):
    e = a.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    e = e[:, [g for g in genes if g in e.var_names]].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    conf = (
        e.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in e.obs.columns
        else np.ones(e.n_obs, dtype=bool)
    )
    y_true = e.obs["cell_type"].astype(str)[conf]
    prev = y_true.value_counts(normalize=True)

    for res in RES_GRID:
        sc.tl.leiden(
            e, resolution=res, key_added="L", flavor="igraph", n_iterations=2, random_state=seed
        )
        y_pred = e.obs["L"].astype(str)[conf]
        f1s = {}
        for pop in y_true.unique():
            t = (y_true == pop).to_numpy()
            best = 0.0
            for cl in y_pred.unique():
                p = (y_pred == cl).to_numpy()
                tp = float(np.sum(t & p))
                if tp:
                    pr, rc = tp / p.sum(), tp / t.sum()
                    best = max(best, 2 * pr * rc / (pr + rc))
            f1s[pop] = best
        for pop, f1 in f1s.items():
            pop_rows.append(
                {
                    "dataset": base["dataset"],
                    "arm": base["arm"],
                    "seed": base["seed"],
                    "resolution": res,
                    "population": pop,
                    "prevalence": float(prev.get(pop, np.nan)),
                    "f1": float(f1),
                }
            )
        rare = [p for p in f1s if prev.get(p, 0) < 0.02]
        rows.append(
            {
                **base,
                "resolution": res,
                "n_leiden": int(y_pred.nunique()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
                "rare_f1_mean": (float(np.mean([f1s[p] for p in rare])) if rare else np.nan),
            }
        )


def main(which=None) -> None:
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pop_rows = pd.read_csv(POPS).to_dict("records") if POPS.exists() else []
    seen_arms: dict[tuple, set] = {}
    for r in rows:
        seen_arms.setdefault((r["dataset"], r["seed"]), set()).add(r["arm"])
    done = {k for k, v in seen_arms.items() if {"rank", "mix70", "quota"} <= v}
    print(f"resuming: {len(done)} (dataset, seed) blocks done", flush=True)

    for name in which or ORDER:
        if all((name, s) in done for s in SEEDS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        try:
            a = LOADERS[name]()
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        if "cell_type" not in a.obs:
            print("  no cell_type; skipping", flush=True)
            continue
        print(
            f"  {a.n_obs} cells x {a.n_vars} genes, {a.obs['cell_type'].nunique()} labelled types",
            flush=True,
        )

        pool = global_pool(a, POOL_MULT * K)
        print(f"  candidate pool: {len(pool)} genes", flush=True)

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.time()
            try:
                sel = a.copy()
                scf.pp.highly_variable_genes(
                    sel, n_top_genes=min(K, sel.n_vars - 1), random_state=seed
                )
                # In the selector's own blended order, so the 70% prefix is the
                # top 70% of the ranking and not an arbitrary 1400 of the 2000.
                hv = sel.var["highly_variable"].to_numpy(dtype=bool)
                order = pd.to_numeric(sel.var["highly_variable_rank"], errors="coerce")[
                    hv
                ].sort_values()
                genes_rank = [str(g) for g in order.index]
                labels = sel.obs["scfair_hvg_clusters"]

                ranks = _build_cluster_gene_ranks(
                    sel,
                    cluster_labels=labels,
                    counts_layer="counts",
                    min_cluster_size=MIN_CLUSTER_SIZE,
                    logfc_space="log1p",
                )
                genes_quota, per_cluster = quota_select(ranks, pool, K)
                if not genes_quota:
                    print(f"  seed={seed}: no cluster ranks; skipping", flush=True)
                    continue
                n_keep = int(round(RANK_FRAC * K))
                genes_mix, per_cluster_mix = quota_select(
                    ranks, pool, K, preselected=genes_rank[:n_keep]
                )

                overlap = len(set(genes_rank) & set(genes_quota))
                share = sorted(per_cluster.values())
                share_mix = sorted(per_cluster_mix.values())
                print(
                    f"  seed={seed}  {len(ranks)} clusters, "
                    f"quota {len(genes_quota)} genes "
                    f"({share[0]}..{share[-1]}/cluster), "
                    f"overlap with rank arm {overlap}/{len(genes_rank)}; "
                    f"mix70 {len(genes_mix)} genes, "
                    f"{share_mix[0]}..{share_mix[-1]} quota-added/cluster",
                    flush=True,
                )

                for arm, genes in (
                    ("rank", genes_rank),
                    ("mix70", genes_mix),
                    ("quota", genes_quota),
                ):
                    base = {
                        "dataset": name,
                        "arm": arm,
                        "seed": seed,
                        "n_genes": len(genes),
                        "n_clusters": len(ranks),
                        "overlap": overlap,
                    }
                    evaluate(a, genes, seed, rows, pop_rows, base)
                del sel
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL seed={seed}: {type(e).__name__}: {e}", flush=True)
                continue
            done.add((name, seed))
            pd.DataFrame(rows).to_csv(CSV, index=False)
            pd.DataFrame(pop_rows).to_csv(POPS, index=False)
            got = {
                arm: np.mean(
                    [
                        r["ARI"]
                        for r in rows
                        if r["dataset"] == name and r["seed"] == seed and r["arm"] == arm
                    ]
                )
                for arm in ("rank", "mix70", "quota")
            }
            print(
                f"    ARI rank={got['rank']:.3f} mix70={got['mix70']:.3f} "
                f"quota={got['quota']:.3f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(f"\nwrote {CSV} ({len(rows)} rows) and {POPS} ({len(pop_rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
