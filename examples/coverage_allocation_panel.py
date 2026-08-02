#!/usr/bin/env python
"""Compare post-hybrid allocation policies: none / cap (current default) /
coverage (legitimate units + coverage-floor top-up).

Arms
----
- ``none``     -- ``allocation_method="none"`` (hybrid blend only)
- ``cap``      -- current default equal-share ceiling
- ``coverage`` -- experimental: stability∧DE units + coverage floor

Otherwise identical: ``n_top_genes=2000``, ``balance_method="hybrid"``,
``resolution=0.5`` (fixed, so the panel isolates allocation rather than
confounding it with ``resolution="auto"``).

Outputs (examples/results/):
  coverage_allocation_panel.csv
  coverage_allocation_panel_pops.csv

Usage
-----
  python examples/coverage_allocation_panel.py           # full 10×5 panel
  python examples/coverage_allocation_panel.py quick     # 4 datasets × 2 seeds
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

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "coverage_allocation_panel.csv"
POPS = OUT / "coverage_allocation_panel_pops.csv"

K = 2000
# Fixed resolution: isolate allocation method from density-auto.
RESOLUTION = 0.5
RES_GRID = [0.3, 0.5, 0.8, 1.2]
ARMS = ("none", "cap", "coverage")

ORDER_FULL = [
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
ORDER_QUICK = [
    "pancreas_smartseq2",
    "duo4_pbmc",
    "duo8_pbmc",
    "pbmc3k_louvain",
]


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
            e,
            resolution=res,
            key_added="L",
            flavor="igraph",
            n_iterations=2,
            random_state=seed,
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
        rows.append(
            {
                **base,
                "resolution": res,
                "n_leiden": int(y_pred.nunique()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
            }
        )


def summarise(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    # mean over resolution grid then over seeds
    g = (
        df.groupby(["dataset", "arm", "seed"], as_index=False)[["ARI", "macro_f1", "min_pop_f1"]]
        .mean()
        .groupby(["dataset", "arm"], as_index=False)[["ARI", "macro_f1", "min_pop_f1"]]
        .mean()
    )
    wide = g.pivot(index="dataset", columns="arm", values="ARI")
    for arm in ARMS:
        if arm not in wide.columns:
            wide[arm] = np.nan
    wide = wide[list(ARMS)]
    wide["d_cov_cap"] = wide["coverage"] - wide["cap"]
    wide["d_cov_none"] = wide["coverage"] - wide["none"]
    wide["d_cap_none"] = wide["cap"] - wide["none"]
    print("\n=== mean ARI (over seeds × res grid) ===", flush=True)
    print(wide.round(4).to_string(), flush=True)
    pool = g.groupby("arm")[["ARI", "macro_f1", "min_pop_f1"]].mean()
    print("\n=== pooled mean ===", flush=True)
    print(pool.round(4).to_string(), flush=True)


def run_panel(order: list[str], seeds: list[int]) -> None:
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pop_rows = pd.read_csv(POPS).to_dict("records") if POPS.exists() else []
    seen: dict[tuple, set] = {}
    for r in rows:
        if r.get("dataset") not in order:
            continue
        seen.setdefault((r["dataset"], r["seed"]), set()).add(r["arm"])
    done = {k for k, v in seen.items() if set(ARMS) <= v}
    print(f"resuming: {len(done)} (dataset, seed) blocks done", flush=True)

    for name in order:
        if all((name, s) in done for s in seeds):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            for arm in ARMS:
                ad = a.copy()
                scf.pp.highly_variable_genes(
                    ad,
                    n_top_genes=K,
                    balance_method="hybrid",
                    resolution=RESOLUTION,
                    allocation_method=arm,
                    random_state=seed,
                    diagnose=False,
                )
                genes = list(ad.var_names[ad.var["highly_variable"]])
                meta = ad.uns["scfair"]["hvg"]
                base = {
                    "dataset": name,
                    "arm": arm,
                    "seed": seed,
                    "n_genes": len(genes),
                    "allocation_method": meta.get("allocation_method"),
                    "n_clusters_used": meta.get("n_clusters_used"),
                }
                clus = meta.get("clustering") or {}
                if arm == "coverage":
                    base["n_coverage_merges"] = len(clus.get("coverage_merges") or [])
                    base["coverage_n_added"] = clus.get("coverage_n_added")
                    base["coverage_n_starved"] = clus.get("coverage_n_starved")
                elif arm == "cap":
                    base["n_cap_merges"] = len(clus.get("cap_merges") or [])
                    base["cap_n_added"] = clus.get("cap_n_added")
                    base["cap_n_over"] = clus.get("cap_n_over")
                evaluate(a, genes, seed, rows, pop_rows, base)
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
                for arm in ARMS
            }
            print(
                f"  seed={seed}  ARI none={got['none']:.3f} cap={got['cap']:.3f} "
                f"coverage={got['coverage']:.3f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        del a
        summarise([r for r in rows if r["dataset"] in order])

    print(f"\nwrote {CSV} ({len(rows)} rows) and {POPS} ({len(pop_rows)} rows)", flush=True)
    summarise([r for r in rows if r["dataset"] in order])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "quick":
        run_panel(ORDER_QUICK, seeds=[0, 1])
    elif mode == "full":
        run_panel(ORDER_FULL, seeds=[0, 1, 2, 3, 4])
    else:
        raise SystemExit(f"unknown mode {mode!r}: use 'quick' or 'full'")
