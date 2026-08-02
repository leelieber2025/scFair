#!/usr/bin/env python
"""End-to-end validation of several `cap_merge_threshold` values through the
real API (not the offline pair-score simulation in
`cap_merge_threshold_calibration.py`) -- does raising the threshold actually
move downstream ARI/macro_f1, not just the merge-count proxy?

Arms: nocap, cap_nomerge, and cap_merge at threshold in {0.5, 0.65, 0.85, 0.95}
-- 0.5 is shipped, 0.65/0.7 is where the offline sweep's TP started moving,
0.85 is where TP and FP both start rising together, 0.95 is deliberately
over-aggressive (catches almost everything, including real pairs) as an
upper bound on how much headroom exists on this axis at all.

3 datasets (the same ones the offline sweep used), 3 seeds, 4 resolutions.

Outputs (examples/results/):
  cap_merge_threshold_sweep_real.csv
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
CSV = OUT / "cap_merge_threshold_sweep_real.csv"

K = 2000
SEEDS = [0, 1, 2]
RES_GRID = [0.3, 0.5, 0.8, 1.2]
DATASETS = ["pancreas_smartseq2", "sln_208_mouse", "pbmc_seurat_v4_20k"]
THRESHOLDS = [0.5, 0.65, 0.85, 0.95]
ARMS = {
    "nocap": dict(cap_allocation=False),
    "cap_nomerge": dict(cap_allocation=True, cap_merge_threshold=None),
}
for t in THRESHOLDS:
    ARMS[f"cap_merge_{t:.2f}"] = dict(cap_allocation=True, cap_merge_threshold=t)


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


def main():
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    seen: dict[tuple, set] = {}
    for r in rows:
        seen.setdefault((r["dataset"], r["seed"]), set()).add(r["arm"])
    done = {k for k, v in seen.items() if set(ARMS) <= v}
    print(f"resuming: {len(done)} (dataset, seed) blocks done", flush=True)

    for name in DATASETS:
        if all((name, s) in done for s in SEEDS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.time()
            for arm, kwargs in ARMS.items():
                ad = a.copy()
                scf.pp.highly_variable_genes(
                    ad,
                    n_top_genes=K,
                    flavor="seurat_v3",
                    layer="counts",
                    marker_mode="none",
                    balance_method="hybrid",
                    random_state=seed,
                    diagnose=False,
                    **kwargs,
                )
                genes = list(ad.var_names[ad.var["highly_variable"]])
                base = {"dataset": name, "arm": arm, "seed": seed, "n_genes": len(genes)}
                evaluate(a, genes, seed, rows, base)
            done.add((name, seed))
            pd.DataFrame(rows).to_csv(CSV, index=False)
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
            ari_str = "  ".join(f"{a}={v:.3f}" for a, v in got.items())
            print(f"  seed={seed}  ARI {ari_str}  ({time.time() - t0:.0f}s)", flush=True)
        del a
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
