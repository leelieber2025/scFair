#!/usr/bin/env python
"""adt14 resolution sweep: does the k=1000 min_pop_f1 win survive resolution?

Motivation
----------
The whole §5.9-5.15 line, and sorted_gold_panel.py, evaluate at a single
hardcoded LEIDEN_RES=0.8. On duo4_pbmc that single point inverted the verdict:
hybrid "lost" -0.16 min_pop_f1 at res=0.8 but *won* +0.02 (10/10 seeds,
p=4e-5) once each seed was allowed its own best resolution. adt14's +0.1435
headline is measured the same way and needs the same check.

Protocol
--------
Same gene sets and eval path as sorted_gold_panel.py, but the kNN graph is
built once per (k, config, seed) and Leiden is swept over RES_GRID on it.
Loop order is k -> seed -> config so that a partial run is still a complete
paired experiment on however many seeds finished.

Resumable: re-running skips (k, config, seed) blocks already in the CSV and
reuses cached gene sets.

Outputs (examples/results/):
  adt14_res_sweep.csv / .genes.json / .log
"""

from __future__ import annotations

import json
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
from sorted_gold_panel import load_dataset, per_population_f1, select_genes  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "adt14_res_sweep.csv"
GENES = OUT / "adt14_res_sweep.genes.json"

# adt14 has 14 types and lands on ~15 Leiden clusters at res=0.8, so bracket
# that rather than reusing the duo grid (which topped out at 4-8 clusters).
RES_GRID = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
K_LIST = [1000, 2000]
CONFIGS = ["hvg", "hybrid", "hybrid_cp5000"]
SEEDS = range(20)  # review.md §9 floor for the Hard tier


def main() -> None:
    cache = json.load(open(GENES)) if GENES.exists() else {}
    if CSV.exists():
        rows = pd.read_csv(CSV).to_dict("records")
        done = {(r["k"], r["config"], r["seed"]) for r in rows}
    else:
        rows, done = [], set()
    print(f"resuming with {len(done)} blocks already done", flush=True)

    a0, info = load_dataset("adt14")
    print(f"adt14: {a0.n_obs} cells x {a0.n_vars} genes", flush=True)

    for k in K_LIST:
        for seed in SEEDS:
            for cfg in CONFIGS:
                if (k, cfg, seed) in done:
                    continue
                t0 = time.time()
                key = f"{k}|{cfg}|{seed}"
                if key not in cache:
                    g, _ = select_genes(a0, cfg, n_top=k, seed=seed)
                    cache[key] = g
                    json.dump(cache, open(GENES, "w"))
                t_sel = time.time() - t0

                a = a0.copy()
                a.X = a.layers["counts"].copy()
                sc.pp.normalize_total(a, target_sum=1e4)
                sc.pp.log1p(a)
                a = a[:, [g for g in cache[key] if g in a0.var_names]].copy()
                sc.pp.scale(a, max_value=10)
                n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
                sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
                sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

                keep = (
                    a.obs[info["conf_col"]].to_numpy(dtype=bool)
                    if info["conf_col"]
                    else np.ones(a.n_obs, dtype=bool)
                )
                y_true = a.obs[info["label_col"]].astype(str)[keep]
                prev = y_true.value_counts(normalize=True)
                smallest = prev.idxmin()

                for res in RES_GRID:
                    sc.tl.leiden(
                        a,
                        resolution=res,
                        key_added="L",
                        flavor="igraph",
                        n_iterations=2,
                        random_state=seed,
                    )
                    y_pred = a.obs["L"].astype(str)[keep]
                    f1 = per_population_f1(y_true, y_pred)
                    rare = [p for p in f1 if prev.get(p, 0.0) < 0.02]
                    rows.append(
                        dict(
                            dataset="adt14",
                            k=k,
                            config=cfg,
                            seed=seed,
                            resolution=res,
                            n_genes=a.n_vars,
                            n_leiden=int(y_pred.nunique()),
                            n_eval_cells=int(keep.sum()),
                            ARI=float(adjusted_rand_score(y_true, y_pred)),
                            NMI=float(normalized_mutual_info_score(y_true, y_pred)),
                            macro_f1=float(np.mean(list(f1.values()))),
                            min_pop_f1=float(f1[str(smallest)]),
                            min_pop=str(smallest),
                            rare_f1_mean=float(np.mean([f1[p] for p in rare])) if rare else np.nan,
                        )
                    )
                pd.DataFrame(rows).to_csv(CSV, index=False)
                print(
                    f"k={k} seed={seed:2d} {cfg:14s} "
                    f"sel={t_sel:5.1f}s total={time.time() - t0:5.1f}s",
                    flush=True,
                )
        print(f"===== k={k} COMPLETE =====", flush=True)
    print("ADT14 SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
