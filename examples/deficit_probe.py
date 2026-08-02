#!/usr/bin/env python
"""Why does scFair lose on pbmc_seurat_v4_20k? (§5.19.3)

On the largest non-circular panel (228-plex, 20k cells) the current default
loses to plain scanpy -- peak macro_f1 -0.0077 (p=0.015), mean ARI -0.0081
(p=0.0018). Together with duo8 it is one of two confirmed places where the
selector is simply worse, and it is the better-evidenced of the two.

Hypothesis
----------
Specificity is scored against the intermediate Leiden partition, so it can only
be as good as that partition. §5.18 already showed the partition is unstable
(a 2% gene change moves ~17% of it). If on this dataset the partition is a poor
match to the true cell types, every specificity score is computed against the
wrong question, and the re-ranking would be expected to hurt rather than help.

Two parts, deliberately separate:

  A. Cross-dataset. Measure ARI(intermediate partition, true labels) on all 8
     datasets and check it against each dataset's measured scFair margin. With
     n=8 this is descriptive, not inferential -- the interim cluster-count story
     in §5.19.2 died exactly here, so both Pearson and Spearman are reported and
     neither is treated as a mechanism unless the rank correlation holds.

  B. Within-dataset. Sweep k on pbmc_seurat_v4_20k. §5.15 found the ADT optimum
     is k=1000, not 2000, and this panel was run at 2000 -- so the deficit may
     be a k artifact rather than a defect. This part can settle that on its own,
     independent of A.

Outputs (examples/results/):
  deficit_probe_partition.csv   part A, per (dataset, seed)
  deficit_probe_ksweep.csv      part B, per (k, arm, seed, resolution)
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
from cluster_pool_panel import LOADERS, ORDER, evaluate, select  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
PART_CSV = OUT / "deficit_probe_partition.csv"
KSWEEP_CSV = OUT / "deficit_probe_ksweep.csv"

PART_SEEDS = list(range(5))
K_LIST = [500, 1000, 2000, 3000]
KSWEEP_SEEDS = list(range(10))
KSWEEP_RES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
TARGET = "pbmc_seurat_v4_20k"


def true_labels(adata):
    conf = (
        adata.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in adata.obs.columns
        else np.ones(adata.n_obs, dtype=bool)
    )
    return adata.obs["cell_type"].astype(str), conf


def part_a() -> None:
    """Is the intermediate partition worse where scFair loses?"""
    rows = pd.read_csv(PART_CSV).to_dict("records") if PART_CSV.exists() else []
    done = {(r["dataset"], r["seed"]) for r in rows}

    for dname in ORDER:
        if all((dname, s) in done for s in PART_SEEDS):
            print(f"### {dname}: done, skipping", flush=True)
            continue
        print(f"\n######## part A: {dname} ########", flush=True)
        adata = LOADERS[dname]()
        y_true, conf = true_labels(adata)
        n_types = y_true[conf].nunique()

        for seed in PART_SEEDS:
            if (dname, seed) in done:
                continue
            t0 = time.time()
            a = adata.copy()
            scf.pp.highly_variable_genes(
                a,
                n_top_genes=min(2000, a.n_vars - 1),
                flavor="seurat_v3",
                layer="counts",
                marker_mode="none",
                balance_method="hybrid",
                random_state=seed,
            )
            part = a.obs["scfair_hvg_clusters"].astype(str)
            d = a.uns["scfair"]["hvg"]["clustering"]
            scfair_genes = set(a.var_names[a.var["highly_variable"]].astype(str))

            b = adata.copy()
            sc.pp.highly_variable_genes(
                b,
                n_top_genes=min(2000, b.n_vars - 1),
                flavor="seurat_v3",
                layer="counts",
            )
            scanpy_genes = set(b.var_names[b.var["highly_variable"]].astype(str))

            rows.append(
                {
                    "dataset": dname,
                    "seed": seed,
                    "n_true_types": int(n_types),
                    "n_clusters_total": d["n_clusters_total"],
                    "n_clusters_kept": d["n_clusters_kept"],
                    "n_clusters_dropped": len(d["clusters_dropped"]),
                    # how well the partition that DEFINES specificity matches truth
                    "ari_partition_truth": float(adjusted_rand_score(y_true[conf], part[conf])),
                    "nmi_partition_truth": float(
                        normalized_mutual_info_score(y_true[conf], part[conf])
                    ),
                    # how much the re-ranking actually moved
                    "genes_swapped": len(scfair_genes - scanpy_genes),
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(
                f"  seed={seed} ARI(part,truth)={rows[-1]['ari_partition_truth']:.4f} "
                f"clusters={d['n_clusters_total']} vs {n_types} true types, "
                f"swapped={rows[-1]['genes_swapped']}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(PART_CSV, index=False)
    print("PART A DONE", flush=True)


def part_b() -> None:
    """Does the deficit depend on k?"""
    rows = pd.read_csv(KSWEEP_CSV).to_dict("records") if KSWEEP_CSV.exists() else []
    done = {(r["k"], r["arm"], r["seed"]) for r in rows}

    print(f"\n######## part B: {TARGET} k-sweep ########", flush=True)
    adata = LOADERS[TARGET]()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

    import cluster_pool_panel as cpp

    for k in K_LIST:
        for seed in KSWEEP_SEEDS:
            for arm in ["hvg2000", "scfair2000"]:
                if (k, arm, seed) in done:
                    continue
                t0 = time.time()
                cpp.K = k  # select() reads the module-level k
                try:
                    genes, meta = select(adata, arm, seed)
                    new: list[dict] = []
                    evaluate(
                        adata,
                        genes,
                        seed,
                        new,
                        base={"dataset": TARGET, "k": k, "arm": arm, "seed": seed},
                        res_grid=KSWEEP_RES,
                    )
                    rows.extend(new)
                    print(
                        f"  k={k:5d} {arm:12s} seed={seed:2d} "
                        f"genes={len(genes)} ({time.time() - t0:5.1f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  k={k} {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                pd.DataFrame(rows).to_csv(KSWEEP_CSV, index=False)
    cpp.K = 2000
    print("PART B DONE", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        part_a()
    if which in ("all", "b"):
        part_b()
    print("DEFICIT PROBE DONE", flush=True)
