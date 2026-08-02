#!/usr/bin/env python
"""Is `blend_global=0.95` the right default? 7-dataset panel.

Why this and not another specificity-side knob (§5.23)
------------------------------------------------------
alpha sets how much of the ranking the specificity machinery actually controls:

    S_g = alpha * norm(global_g) + (1 - alpha) * norm(specificity_g)

At the 0.95 default that is 5%, and §5.13 measured scFair's whole advantage at
~+0.006 -- which is what 5% buys. Every knob tested on the specificity side has
come in at +-0.005 and non-significant (`neighbor_contrast`, `cluster_pool`,
consensus). That is not three independent negative results; it is one
coefficient flattening all of them.

alpha itself moves an order of magnitude more. On pbmc_seurat_v4_20k, going
0.70 -> 0.95 closed the kNN-purity gap from -0.037 to -0.020, while
single-vs-consensus moved +-0.002 at any fixed alpha. So alpha is the one
parameter with an effect size worth a panel.

And it has never been tested properly: §5.5's sweeps were PBMC-only, at a
single clustering resolution -- the protocol §5.16 showed can invert a verdict
outright.

Design
------
Arms: scanpy baseline + alpha in {0.85, 0.90, 0.95, 0.98}. k fixed at 2000.
7 datasets x 20 seeds x 11 resolutions (0.2-2.5, the common grid from §5.21).

Primary protocol is **mean-over-resolution** (§5.19.4): peak is censored on
several datasets even at a 2.5 ceiling, and the censoring is arm-specific, so
it biases comparisons asymmetrically.

This is an adoption test, so it is built to clear the §5.11 bar: multi-dataset,
20 seeds, resolution-swept. Loop order is dataset -> seed -> arm, so an
interrupted run is still a complete paired experiment on the seeds that
finished. Resumable per (dataset, seed, arm).

Outputs (examples/results/):
  alpha_panel.csv
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
from cluster_pool_panel import LOADERS, ORDER, SHADOWED, evaluate  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "alpha_panel.csv"

K = 2000
SEEDS = list(range(20))
ALPHAS = [0.85, 0.90, 0.95, 0.98]
RES_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
ARMS = ["scanpy"] + [f"alpha{a:.2f}" for a in ALPHAS]


def select(adata, arm: str, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    k = min(K, a.n_vars - 1)
    if arm == "scanpy":
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
        meta: dict = {"alpha": np.nan}
    else:
        alpha = float(arm.replace("alpha", ""))
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            blend_global=alpha,
            random_state=seed,
            diagnose=False,
        )
        d = a.uns["scfair"]["hvg"]["clustering"]
        meta = {
            "alpha": alpha,
            "n_clusters_total": d["n_clusters_total"],
            "n_clusters_kept": d["n_clusters_kept"],
        }
    return a.var_names[a.var["highly_variable"]].astype(str).tolist(), meta


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
                        res_grid=RES_GRID,
                    )
                    print(
                        f"  {arm:10s} seed={seed:2d} k={len(genes):5d} ({time.time() - t0:5.1f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                pd.DataFrame(rows).to_csv(CSV, index=False)
        print(f"===== {dname} COMPLETE =====", flush=True)

    print("ALPHA PANEL DONE", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(which=args or None)
