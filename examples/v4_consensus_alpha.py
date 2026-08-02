#!/usr/bin/env python
"""Does consensus specificity help, and does alpha gate whether it can?

Two results force this design (§5.23, §5.24):

  1. The pbmc_seurat_v4_20k deficit scales strictly monotonically with how much
     specificity is injected -- Spearman rho = 1.000 over alpha in
     {0.80 ... 1.00}, and at alpha=1.00 (specificity weighted zero) the deficit
     is exactly zero. The mechanism is confirmed: hybrid sharpens a partition
     that is misaligned with the labelled sub-states.

  2. Consensus only changes the specificity term, which carries weight
     (1 - alpha). At the default alpha=0.95 that is 5%, and a 3-rung ladder
     moves 0 of 60 genes on synthetic data (score correlation 0.9995).

So evaluating consensus at alpha=0.95 alone would measure differences inside a
5% slice and return "no effect" regardless of whether the idea works. This
crosses the two: if consensus is worth anything, the gap between a ladder and a
single partition should widen as alpha falls.

Grid: alpha in {0.70, 0.90, 0.95} x {single resolution 0.5, ladder [0.3, 0.5, 1.0]}
plus a scanpy baseline. 10 seeds, k=2000, pbmc_seurat_v4_20k.

kNN purity is the primary metric: it is computed from the neighbour graph with
no clustering involved, so it isolates "did the gene set make a better space"
from any resolution artefact (§5.16).

Outputs (examples/results/):
  v4_consensus_alpha.csv        per (seed, arm)
  v4_consensus_populations.csv  per (seed, arm, population)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import scfair as scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cluster_pool_panel import LOADERS  # noqa: E402
from v4_deficit_dissect import knn_purity, per_population_f1  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "v4_consensus_alpha.csv"
POP_CSV = OUT / "v4_consensus_populations.csv"

TARGET = "pbmc_seurat_v4_20k"
K = 2000
SEEDS = list(range(10))
ALPHAS = [0.70, 0.90, 0.95]
LADDER = [0.3, 0.5, 1.0]
BASE_RES = 0.5
RES_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

ARMS = ["scanpy"] + [f"a{a:.2f}_{tag}" for a in ALPHAS for tag in ("single", "consensus")]


def select_genes(adata, arm: str, seed: int) -> list[str]:
    a = adata.copy()
    k = min(K, a.n_vars - 1)
    if arm == "scanpy":
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
    else:
        alpha = float(arm[1:5])
        consensus = LADDER if arm.endswith("consensus") else None
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            blend_global=alpha,
            resolution=BASE_RES,
            consensus_resolutions=consensus,
            random_state=seed,
            diagnose=False,
        )
    return a.var_names[a.var["highly_variable"]].astype(str).tolist()


def main() -> None:
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pops = pd.read_csv(POP_CSV).to_dict("records") if POP_CSV.exists() else []
    done = {(r["seed"], r["arm"]) for r in rows}

    adata = LOADERS[TARGET]()
    print(f"{TARGET}: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    conf = (
        adata.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in adata.obs.columns
        else np.ones(adata.n_obs, dtype=bool)
    )
    y_all = adata.obs["cell_type"].astype(str)
    y_true = y_all[conf]

    for seed in SEEDS:
        for arm in ARMS:
            if (seed, arm) in done:
                continue
            t0 = time.time()
            genes = select_genes(adata, arm, seed)
            t_sel = time.time() - t0

            a = adata.copy()
            a.X = a.layers["counts"].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            a = a[:, [g for g in genes if g in adata.var_names]].copy()
            sc.pp.scale(a, max_value=10)
            n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
            sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
            sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

            pur = knn_purity(a.obsp["connectivities"], y_all.to_numpy())
            best_f1: dict[str, float] = {}
            for res in RES_GRID:
                sc.tl.leiden(
                    a,
                    resolution=res,
                    key_added="L",
                    flavor="igraph",
                    n_iterations=2,
                    random_state=seed,
                )
                for pop, f1 in per_population_f1(y_true, a.obs["L"].astype(str)[conf]).items():
                    best_f1[pop] = max(best_f1.get(pop, 0.0), f1)

            for pop in y_true.unique():
                m = (y_all == pop).to_numpy() & conf
                pops.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "population": pop,
                        "knn_purity": float(pur[m].mean()),
                        "best_f1": float(best_f1[pop]),
                    }
                )
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "n_genes": a.n_vars,
                    "knn_purity": float(pur[conf].mean()),
                    "macro_best_f1": float(np.mean(list(best_f1.values()))),
                    "select_seconds": round(t_sel, 1),
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(
                f"  seed={seed} {arm:18s} purity={rows[-1]['knn_purity']:.4f} "
                f"macroF1={rows[-1]['macro_best_f1']:.4f} "
                f"sel={t_sel:.0f}s tot={time.time() - t0:.0f}s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(CSV, index=False)
            pd.DataFrame(pops).to_csv(POP_CSV, index=False)

    print("V4 CONSENSUS x ALPHA DONE", flush=True)


if __name__ == "__main__":
    main()
