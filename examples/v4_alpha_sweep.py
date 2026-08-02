#!/usr/bin/env python
"""Does the pbmc_seurat_v4_20k deficit scale with how much specificity is injected?

§5.22 established the deficit is a **subspace** effect: hybrid's gene set gives
worse kNN purity than plain scanpy (-0.0201, 0/10 seeds, p<1e-4), with no
clustering involved, and the damage concentrates in lymphoid populations
(p13_NK: purity -0.061, F1 -0.163, both 0/10). The proposed mechanism is that
hybrid faithfully sharpens an intermediate partition that is fine-grained but
*misaligned* with the labelled sub-states.

`blend_global` (alpha) is the single knob controlling how much specificity is
mixed into the global ranking:

    S_g = alpha * norm(global_g) + (1 - alpha) * norm(specificity_g)

If the mechanism is right, the deficit should scale with (1 - alpha) and vanish
as alpha -> 1. That turns a plausible story into a measurement, and it is a
*within-dataset* design -- the only kind that has worked on this question.

Built-in control: at **alpha = 1.0** the specificity term is weighted zero, so
the selection must reduce to the global top-k, i.e. bit-identical to scanpy.
Any deviation there is a bug in the blend, not a finding.

Note what this is not: §5.14 tried to make alpha adaptive and failed, but that
was a *cross-dataset* search for a label-free gate at n=8, where nothing below
|r|=0.71 can reach significance. Measuring alpha's response curve inside one
dataset is a different question with real power behind it.

Outputs (examples/results/):
  v4_alpha_sweep.csv        per (seed, arm) embedding + macro metrics
  v4_alpha_populations.csv  per (seed, arm, population)
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
CSV = OUT / "v4_alpha_sweep.csv"
POP_CSV = OUT / "v4_alpha_populations.csv"

TARGET = "pbmc_seurat_v4_20k"
K = 2000
SEEDS = list(range(10))
ALPHAS = [0.80, 0.90, 0.95, 0.98, 1.00]  # 0.95 is the default; 1.00 is the control
RES_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]


def select_genes(adata, arm: str, seed: int) -> list[str]:
    a = adata.copy()
    k = min(K, a.n_vars - 1)
    if arm == "scanpy":
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
    else:
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            blend_global=float(arm.split("=")[1]),
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

    arms = ["scanpy"] + [f"alpha={a:.2f}" for a in ALPHAS]

    for seed in SEEDS:
        ref_genes: list[str] | None = None
        for arm in arms:
            if (seed, arm) in done:
                continue
            t0 = time.time()
            genes = select_genes(adata, arm, seed)
            if arm == "scanpy":
                ref_genes = genes
            overlap = len(set(genes) & set(ref_genes)) if ref_genes is not None else np.nan

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
                    "overlap_with_scanpy": overlap,
                    "knn_purity": float(pur[conf].mean()),
                    "macro_best_f1": float(np.mean(list(best_f1.values()))),
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(
                f"  seed={seed} {arm:12s} overlap={overlap}/{len(genes)} "
                f"purity={rows[-1]['knn_purity']:.4f} "
                f"macroF1={rows[-1]['macro_best_f1']:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(CSV, index=False)
            pd.DataFrame(pops).to_csv(POP_CSV, index=False)

    print("V4 ALPHA SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
