#!/usr/bin/env python
"""Should `cluster_pool=5000` become the default? (§5.17 A1)

The decision
------------
Today the intermediate clustering runs on the top-`n_top_genes` mask, so `k`
silently controls two things: how many genes are kept, and what defines
specificity. `cluster_pool` decouples them. Two panels already favour it
(§5.16), but two panels is not the §5.11 bar, so this re-runs §5.13's 8-dataset
head-to-head under §5.16's sweep protocol.

Adopting it also closes `auto@k != hybrid@k` (§5.15 item 4) by construction,
which is why it is first in the queue.

Arms (3, at the current default k=2000)
---------------------------------------
hvg2000     scanpy highly_variable_genes(flavor="seurat_v3")  -- the baseline
scfair2000  current default (cluster_pool=None)               -- the incumbent
scfair2000_cp5000                                             -- the proposal

k is deliberately fixed at the current default: the question on the table is
`cluster_pool`'s default *at* the default k, not a joint (k, pool) search.

Protocol
--------
Post-§5.16, so: every arm is evaluated over a resolution grid and compared at
peak and mean, never at one resolution. The grid runs to 2.5 because §5.16.6's
grid stopped at 1.5 and `hvg` selected that edge on 9/20 seeds, censoring its
peak -- each row therefore also records whether that seed's argmax sits on a
grid boundary, so truncation is detected by the analysis instead of discovered
afterwards.

20 seeds (§5.15's floor). Loop order is dataset -> seed -> arm, so an
interrupted run is still a complete paired experiment on the seeds that
finished. Resumable: re-running skips (dataset, seed, arm) blocks already in
the CSV.

Outputs (examples/results/):
  cluster_pool_panel.csv   one row per (dataset, arm, seed, resolution)
  cluster_pool_panel.log
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

import scfair as scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from adt_gold_benchmark import load_labeled as load_adt14  # noqa: E402
from adt_multi_validation import load_cite  # noqa: E402
from p3_public_validation import (  # noqa: E402
    load_paul15,
    load_pbmc3k_labeled,
    load_scib_pancreas_one_tech,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "cluster_pool_panel.csv"

K = 2000
CLUSTER_POOL = 5000
SEEDS = list(range(20))
RES_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
ARMS = ["hvg2000", "scfair2000", "scfair2000_cp5000"]

NON_CIRCULAR = ["pbmc10k_adt14", "pbmc5k_adt29", "pbmc_seurat_v4_20k", "sln_208_mouse"]
SHADOWED = ["paul15", "pbmc3k_louvain", "pancreas_smartseq2"]

# ---------------------------------------------------------------------------
# Excluded datasets
# ---------------------------------------------------------------------------
# Criterion: median genes detected per cell < 300.
#
# Cao (sci-RNA-seq, CellBRF's Cao.h5) sits at 150 genes/cell on a median of 245
# UMI. The next-shallowest set in the panel is paul15 at 872 genes/cell -- there
# is no dataset between them, so this is not a tail of a distribution, it is an
# outlier by ~6x. Every method here runs PCA -> neighbours -> Leiden on a
# 2000-gene space in which almost every cell is zero on almost every gene, so
# the panel was averaging over a regime the rest of it does not represent.
#
# Two honesty notes that belong with any exclusion made after seeing results:
#   1. It was applied post hoc, on 2026-07-30, after Cao produced the largest
#      single effect in the cluster_pool comparison (DEVELOPMENT_LOG §5.21).
#   2. It cuts both ways. Cao was also one of scFair's *largest wins* over
#      scanpy on this panel (peak macro_f1 +0.0148, 18/20 seeds, p=0.001) and
#      the source of README's retired "+8.0%" figure. Dropping it weakens a
#      favourable headline as much as an unfavourable one.
#
# The file stays on disk and the loader still works; it is out of the panel,
# not deleted. Re-including it means re-running every arm, not filtering.
EXCLUDED = {
    "Cao": "median genes/cell = 150 (< 300 threshold); next lowest is 872",
}

LOADERS = {
    "pbmc10k_adt14": load_adt14,
    "pbmc5k_adt29": lambda: load_cite("pbmc_5k_v3"),
    "pbmc_seurat_v4_20k": lambda: load_cite("pbmc_seurat_v4_20k"),
    "sln_208_mouse": lambda: load_cite("sln_208_mouse"),
    "paul15": load_paul15,
    "pbmc3k_louvain": load_pbmc3k_labeled,
    "pancreas_smartseq2": lambda: load_scib_pancreas_one_tech("smartseq2"),
}
# smallest first: a failure or a bad assumption surfaces in minutes, not hours
ORDER = [
    "paul15",
    "pbmc3k_louvain",
    "pancreas_smartseq2",
    "pbmc5k_adt29",
    "pbmc10k_adt14",
    "sln_208_mouse",
    "pbmc_seurat_v4_20k",
]


def select(adata, arm: str, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    k = min(K, a.n_vars - 1)
    if arm == "hvg2000":
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
        meta: dict = {"cluster_pool_effective": np.nan}
    else:
        kw = dict(
            n_top_genes=k,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            random_state=seed,
        )
        if arm == "scfair2000_cp5000":
            # capped: on paul15 (3.4k genes) the pool is the whole gene set, which
            # is the intended limit of the knob, not a silent degenerate case
            kw["cluster_pool"] = min(CLUSTER_POOL, a.n_vars - 1)
        scf.pp.highly_variable_genes(a, **kw)
        d = a.uns["scfair"]["hvg"]
        meta = {
            "cluster_pool_effective": d["clustering"]["n_genes_clustered"],
            "n_clusters_total": d["clustering"]["n_clusters_total"],
            "n_clusters_kept": d["clustering"]["n_clusters_kept"],
            "n_clusters_dropped": len(d["clustering"]["clusters_dropped"]),
        }
    return a.var_names[a.var["highly_variable"]].astype(str).tolist(), meta


def evaluate(adata, genes, seed: int, rows: list, base: dict, res_grid=None) -> None:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, [g for g in genes if g in a.var_names]].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    conf = (
        a.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in a.obs.columns
        else np.ones(a.n_obs, dtype=bool)
    )
    y_true = a.obs["cell_type"].astype(str)[conf]
    prev = y_true.value_counts(normalize=True)
    smallest = prev.idxmin()

    for res in RES_GRID if res_grid is None else res_grid:
        sc.tl.leiden(
            a,
            resolution=res,
            key_added="L",
            flavor="igraph",
            n_iterations=2,
            random_state=seed,
        )
        y_pred = a.obs["L"].astype(str)[conf]
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
        rare = [p for p in f1s if prev.get(p, 0) < 0.02]
        rows.append(
            {
                **base,
                "resolution": res,
                "n_genes": a.n_vars,
                "n_leiden": int(y_pred.nunique()),
                "n_eval_cells": int(conf.sum()),
                "n_pop": len(f1s),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(smallest)]),
                "min_pop": str(smallest),
                "rare_f1_mean": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
            }
        )


def main(which=None) -> None:
    if CSV.exists():
        rows = pd.read_csv(CSV).to_dict("records")
        done = {(r["dataset"], r["seed"], r["arm"]) for r in rows}
    else:
        rows, done = [], set()
    print(f"resuming with {len(done)} blocks already done", flush=True)

    for dname in which or ORDER:
        if all((dname, s, arm) in done for s in SEEDS for arm in ARMS):
            print(f"### {dname}: already complete, skipping", flush=True)
            continue
        print(f"\n################ {dname} ################", flush=True)
        try:
            adata = LOADERS[dname]()
        except Exception as e:
            print(f"  LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

        for seed in SEEDS:
            for arm in ARMS:
                if (dname, seed, arm) in done:
                    continue
                t0 = time.time()
                try:
                    genes, meta = select(adata, arm, seed)
                    t_sel = time.time() - t0
                    evaluate(
                        adata,
                        genes,
                        seed,
                        rows,
                        base={
                            "dataset": dname,
                            "arm": arm,
                            "seed": seed,
                            "circular": dname in SHADOWED,
                            **meta,
                        },
                    )
                    print(
                        f"  {arm:18s} seed={seed:2d} k={len(genes):5d} "
                        f"pool={meta.get('cluster_pool_effective')} "
                        f"sel={t_sel:5.1f}s total={time.time() - t0:6.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                pd.DataFrame(rows).to_csv(CSV, index=False)
        print(f"===== {dname} COMPLETE =====", flush=True)

    print("CLUSTER POOL PANEL DONE", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(which=args or None)
