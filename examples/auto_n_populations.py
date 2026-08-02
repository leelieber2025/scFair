#!/usr/bin/env python
"""Calibrate `k` from the population count, and test equal vs size-weighted.

Three questions, one sweep
--------------------------
1. **Is there a constant?** `k = c * n_populations` — what is `c`, and does one
   value work across datasets?
2. **Equal share or more genes for bigger populations?** Compared as
   `effective = n_pop` versus `effective = sum((s_p / median_s) ** 0.5)`.
   Argument favours equal (`_auto_n.select_n_top_from_populations`), but the
   knob exists so it can be measured instead.
3. **Does either beat the shipped `auto`?** which returns ~2000 everywhere —
   over these datasets `elbow`/`knee` sit on `k_min` for 11 of 12 and the whole
   ensemble spans 1.33x, while the measured optimum spans 3x.

Design
------
Per dataset: one default scFair call to read the density field's population
count and sizes, then a k-sweep with everything else at defaults
(`balance_method="hybrid"`, `resolution="auto"`).

Each (dataset, k, seed) is scored over a resolution grid and summarised by the
**mean over resolution** (§5.19.4 — peak is censored asymmetrically and biases
arms against each other). The grid is deliberately wider at the top than the
archived sweeps: both existing calibration points are censored by their grids
({100,500,1000,2000} and {500,1000,2000,3000}, the latter still rising at its
top rung), which is exactly the error this run must not repeat.

`c` is not swept here. `c` is read off afterwards: the optimum k per dataset
divided by that dataset's population count is the estimate, and whether those
ratios agree across datasets *is* question 1.

Resumable per (dataset, k, seed).

Outputs (examples/results/):
  auto_n_populations.csv        one row per (dataset, k, seed, resolution)
  auto_n_populations_meta.csv   population count and sizes per dataset
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
CSV = OUT / "auto_n_populations.csv"
META = OUT / "auto_n_populations_meta.csv"

# Wide at BOTH ends: the archived sweeps were censored above (v4_20k still
# rising at its top rung of 3000), and the paul15 smoke run was censored below
# (k=500, the old floor, was its best point on ARI, macro_f1 and min_pop_f1
# alike). An optimum sitting on a grid boundary is not an optimum.
K_GRID = [200, 300, 500, 1000, 1500, 2000, 3000, 4000]
SEEDS = [0, 1, 2]
RES_GRID = [0.3, 0.5, 0.8, 1.2]

# labelled datasets only
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


def evaluate(a, genes, seed, rows, base):
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
    done = {(r["dataset"], r["k"], r["seed"]) for r in rows}
    metas = pd.read_csv(META).to_dict("records") if META.exists() else []
    have_meta = {m["dataset"] for m in metas}
    print(f"resuming: {len(done)} blocks done", flush=True)

    for name in which or ORDER:
        if all((name, k, s) in done for k in K_GRID for s in SEEDS):
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

        if name not in have_meta:
            probe = a.copy()
            scf.pp.highly_variable_genes(probe)
            g = probe.uns["scfair"]["hvg"]["clustering"].get("granularity") or {}
            sizes = g.get("population_sizes") or []
            eff_w = (
                float(np.sum((np.asarray(sizes, float) / np.median(sizes)) ** 0.5))
                if sizes
                else np.nan
            )
            metas.append(
                {
                    "dataset": name,
                    "n_cells": int(a.n_obs),
                    "n_labelled_types": int(a.obs["cell_type"].nunique()),
                    "n_populations": g.get("n_populations"),
                    "effective_weighted": round(eff_w, 3),
                    "population_sizes": ";".join(str(s) for s in sizes),
                }
            )
            pd.DataFrame(metas).to_csv(META, index=False)
            print(
                f"  density field: n_pop={g.get('n_populations')} "
                f"weighted={eff_w:.2f} sizes={sizes}",
                flush=True,
            )
            del probe

        for k in K_GRID:
            for seed in SEEDS:
                if (name, k, seed) in done:
                    continue
                t0 = time.time()
                try:
                    sel = a.copy()
                    scf.pp.highly_variable_genes(
                        sel, n_top_genes=min(k, sel.n_vars - 1), random_state=seed
                    )
                    genes = sel.var_names[sel.var["highly_variable"]].astype(str).tolist()
                    base = {"dataset": name, "k": k, "seed": seed, "n_genes": len(genes)}
                    evaluate(a, genes, seed, rows, base)
                    del sel
                except Exception as e:  # noqa: BLE001
                    print(f"  FAIL k={k} seed={seed}: {type(e).__name__}: {e}", flush=True)
                    continue
                done.add((name, k, seed))
                pd.DataFrame(rows).to_csv(CSV, index=False)
                last = [
                    r for r in rows if r["dataset"] == name and r["k"] == k and r["seed"] == seed
                ]
                print(
                    f"  k={k:<5} seed={seed}  "
                    f"ARI={np.mean([r['ARI'] for r in last]):.3f} "
                    f"f1={np.mean([r['macro_f1'] for r in last]):.3f}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    print(f"\nwrote {CSV} ({len(rows)} rows) and {META}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
