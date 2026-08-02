#!/usr/bin/env python
"""Is "resolution matters 13x" an artifact of split-punishing metrics?

The objection (raised 2026-07-30, and it is a good one): over-clustering is not
a real analysis failure. An analyst sets a generous resolution, annotates each
cluster by its dominant identity, and merges clusters that got the same call.
Several clusters of one cell type cost nothing. What is unrecoverable is
*under*-clustering -- two distinct types fused into one cluster.

Every metric used in §5.16-§5.25 is asymmetric the wrong way for that workflow:

  - `macro_f1` takes, per population, the single best-matching cluster. Split a
    population across three clusters and the best one has high precision but a
    third of the recall, so F1 collapses.
  - ARI penalises splitting directly.
  - NMI penalises it through the completeness term.

So the measured "resolution span" may be mostly the metric punishing something
harmless, which would make the 13x figure -- and the whole resolution-sweep
protocol -- an artifact.

The merge-tolerant metric
-------------------------
Model the actual workflow: assign each cluster the majority true label, relabel
every cell with its cluster's assignment, then score. Over-splitting is benign
(more clusters can only become purer); under-splitting still costs, because a
cluster dominated by type A donates all its type-B cells to A and they cannot
be recovered.

Note this is *not* homogeneity, which is trivially maximised by giving every
cell its own cluster. Many-to-one assignment stays bounded by the mixing that
actually occurs.

If the objection is right, the resolution span should largely vanish under this
metric while the scFair-vs-scanpy comparison stays put.

Two datasets, chosen as the strongest test: pbmc5k_adt29 has the largest
macro_f1 resolution span in the panel (0.176), pbmc10k_adt14 the third largest.
If the span survives there it survives everywhere.

Outputs (examples/results/):
  merge_tolerant_metric.csv
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
from cluster_pool_panel import LOADERS, select  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "merge_tolerant_metric.csv"

DATASETS = ["pbmc5k_adt29", "pbmc10k_adt14"]
ARMS = ["hvg2000", "scfair2000"]
SEEDS = list(range(10))
RES_GRID = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]


def best_match_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Per population, the best single cluster. Punishes splitting."""
    out = []
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in y_pred.unique():
            m = (y_pred == cl).to_numpy()
            tp = float(np.sum(t & m))
            if tp:
                pr, rc = tp / m.sum(), tp / t.sum()
                best = max(best, 2 * pr * rc / (pr + rc))
        out.append(best)
    return float(np.mean(out))


def merge_tolerant_f1(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """Assign each cluster its majority label, relabel, then score.

    Returns (macro F1 over populations, accuracy).
    """
    df = pd.DataFrame({"t": y_true.to_numpy(), "c": y_pred.to_numpy()})
    assign = df.groupby("c")["t"].agg(lambda s: s.value_counts().idxmax())
    pred = df["c"].map(assign)
    f1s = []
    for pop in df["t"].unique():
        t = (df["t"] == pop).to_numpy()
        m = (pred == pop).to_numpy()
        tp = float(np.sum(t & m))
        if tp == 0:
            f1s.append(0.0)
            continue
        pr, rc = tp / m.sum(), tp / t.sum()
        f1s.append(2 * pr * rc / (pr + rc))
    return float(np.mean(f1s)), float((pred.to_numpy() == df["t"].to_numpy()).mean())


def main() -> None:
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    done = {(r["dataset"], r["arm"], r["seed"]) for r in rows}

    for dname in DATASETS:
        adata = LOADERS[dname]()
        print(f"\n#### {dname}: {adata.n_obs} x {adata.n_vars}", flush=True)
        conf = (
            adata.obs["adt_confident"].to_numpy(dtype=bool)
            if "adt_confident" in adata.obs.columns
            else np.ones(adata.n_obs, dtype=bool)
        )
        y_true = adata.obs["cell_type"].astype(str)[conf]

        for seed in SEEDS:
            for arm in ARMS:
                if (dname, arm, seed) in done:
                    continue
                t0 = time.time()
                genes, _ = select(adata, arm, seed)
                a = adata.copy()
                a.X = a.layers["counts"].copy()
                sc.pp.normalize_total(a, target_sum=1e4)
                sc.pp.log1p(a)
                a = a[:, [g for g in genes if g in adata.var_names]].copy()
                sc.pp.scale(a, max_value=10)
                n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
                sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
                sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
                for res in RES_GRID:
                    sc.tl.leiden(
                        a,
                        resolution=res,
                        key_added="L",
                        flavor="igraph",
                        n_iterations=2,
                        random_state=seed,
                    )
                    y_pred = a.obs["L"].astype(str)[conf]
                    mt_f1, mt_acc = merge_tolerant_f1(y_true, y_pred)
                    rows.append(
                        {
                            "dataset": dname,
                            "arm": arm,
                            "seed": seed,
                            "resolution": res,
                            "n_leiden": int(y_pred.nunique()),
                            "best_match_f1": best_match_f1(y_true, y_pred),
                            "merge_tolerant_f1": mt_f1,
                            "merge_tolerant_acc": mt_acc,
                            "ARI": float(adjusted_rand_score(y_true, y_pred)),
                            "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                        }
                    )
                print(f"  {arm:12s} seed={seed:2d} ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(CSV, index=False)

    print("MERGE TOLERANT METRIC DONE", flush=True)


if __name__ == "__main__":
    main()
