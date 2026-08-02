#!/usr/bin/env python
"""Which sub-splits of the intermediate clustering are real?

The idea (raised 2026-07-30): a single global `resolution` is the wrong control.
On pbmc5k_adt29 at the default 0.5, one true cell type is already cut in two
while another stays whole until 1.0 -- so whatever value you pick, some regions
are over-split and others under-split. What a human does instead is look at the
embedding and ask, per boundary, "are these really two groups?"

That question is computable. The structure lives in the kNN graph and PC space;
the UMAP is only a rendering of it (and a stochastic, distance-distorting one,
so it should not be the input).

Avoiding the circularity
------------------------
Testing a split for differential expression on the same cells that defined it
always finds genes -- the classic double dip. This uses **split-half
reproducibility** instead:

  1. halve the cluster's cells at random into A and B;
  2. sub-cluster A into two, and take the top up-genes of each side as a
     signature;
  3. score B's cells on that signature -- built without ever seeing B;
  4. sub-cluster B independently, and ask how well A's signature separates B's
     own two groups (AUC).

Real structure: two independent halves find the same axis, AUC -> 1.
Noise: Leiden still splits (it always does), but the axis does not reproduce,
AUC -> 0.5.

Repeated over several halvings so the number is not one draw.

Ground truth is available here, so each split is also labelled by what it
actually is: a boundary between two annotated cell types, or a division inside
one type.

Output: examples/results/split_support_probe.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cluster_pool_panel import LOADERS  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
DATASET = "pbmc5k_adt29"
BASE_RES = 0.5
N_HALVINGS = 5
N_SIG_GENES = 25
MIN_CELLS = 40


def subcluster(X, idx, seed):
    """Leiden into (up to) 2 groups on the given cells, in PC space."""
    sub = sc.AnnData(X[idx])
    sc.pp.neighbors(sub, n_neighbors=min(15, len(idx) - 1), use_rep="X", random_state=seed)
    for res in (0.2, 0.4, 0.8, 1.5):
        sc.tl.leiden(
            sub, resolution=res, key_added="s", flavor="igraph", n_iterations=2, random_state=seed
        )
        if sub.obs["s"].nunique() >= 2:
            break
    lab = sub.obs["s"].astype(str).to_numpy()
    if len(set(lab)) < 2:
        return None
    # collapse to the two largest groups
    top2 = pd.Series(lab).value_counts().index[:2]
    keep = np.isin(lab, top2)
    return keep, (lab[keep] == top2[0]).astype(int)


def main() -> None:
    a = LOADERS[DATASET]()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    expr = a.copy()  # full log-normalised matrix for DE
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3", layer="counts")
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=30, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=0)
    sc.tl.leiden(
        a, resolution=BASE_RES, key_added="L", flavor="igraph", n_iterations=2, random_state=0
    )
    print(
        f"{DATASET}: {a.n_obs} cells, {a.obs.L.nunique()} clusters at res={BASE_RES}\n", flush=True
    )

    PC = a.obsm["X_pca"]
    E = np.asarray(expr.X.todense()) if hasattr(expr.X, "todense") else np.asarray(expr.X)
    truth = a.obs["cell_type"].astype(str).to_numpy()

    rows = []
    for cl in sorted(a.obs.L.unique(), key=int):
        cells = np.where(a.obs.L.to_numpy() == cl)[0]
        if len(cells) < MIN_CELLS * 2:
            continue
        vc = pd.Series(truth[cells]).value_counts(normalize=True)
        aucs = []
        for h in range(N_HALVINGS):
            rng = np.random.default_rng(h)
            perm = rng.permutation(len(cells))
            A, B = cells[perm[: len(cells) // 2]], cells[perm[len(cells) // 2 :]]
            if min(len(A), len(B)) < MIN_CELLS:
                continue
            ra, rb = subcluster(PC, A, h), subcluster(PC, B, h + 100)
            if ra is None or rb is None:
                continue
            ka, ga = ra
            kb, gb = rb
            Acells, Bcells = A[ka], B[kb]
            if min(gb.sum(), (1 - gb).sum()) < 10:
                continue
            # signature from A only
            m1 = E[Acells[ga == 1]].mean(axis=0)
            m0 = E[Acells[ga == 0]].mean(axis=0)
            d = m1 - m0
            up1, up0 = np.argsort(-d)[:N_SIG_GENES], np.argsort(d)[:N_SIG_GENES]
            score = E[Bcells][:, up1].mean(axis=1) - E[Bcells][:, up0].mean(axis=1)
            auc = roc_auc_score(gb, score)
            aucs.append(max(auc, 1 - auc))  # direction is arbitrary
        if not aucs:
            continue
        rows.append(
            {
                "cluster": cl,
                "n_cells": len(cells),
                "dominant_type": vc.index[0],
                "purity": float(vc.iloc[0]),
                "n_types_5pct": int((vc >= 0.05).sum()),
                "auc_mean": float(np.mean(aucs)),
                "auc_min": float(np.min(aucs)),
                "n_halvings": len(aucs),
            }
        )
        print(
            f"  cluster {cl:>2s}  n={len(cells):>4d}  "
            f"dominant={vc.index[0]:<12s} purity={vc.iloc[0]:.2f}  "
            f"split AUC={np.mean(aucs):.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_support_probe.csv", index=False)
    print("\nSPLIT SUPPORT PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
