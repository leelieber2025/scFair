#!/usr/bin/env python
"""Data-derived bandwidth: is the core count a plateau, or a knob?

Why this replaces the voxel version
-----------------------------------
`umap3d_sensitivity.py` killed the voxel formulation's "one knob" claim. The
count at `depth=0.5` ran 3..22 over the (BINS, SMOOTH) grid on a dataset with 8
populations, and the earlier k=8 hit was a property of the (48, 1.2) cell.

But the failure was structured, not random: BINS and SMOOTH are one physical
quantity written twice. At matched `SMOOTH/BINS` the count is tight (6..10 over
12 configs spanning BINS 32->64), and the count falls monotonically as that
ratio grows: 20.5, 14.5, 12.5, 9.0, 7.0, 6.0, 3.5.

So the defect is the bandwidth definition: a fixed fraction of the bounding-box
span is a property of the UMAP layout, not of the data. This version sets the
bandwidth from the data instead -- kNN density, where the smoothing length at
each point is the distance to its k-th neighbour. It widens in sparse regions
and narrows in dense ones automatically, and `k` is in units of cells.

The test, and it is binary
--------------------------
A count that is read off the data must be **flat over a range of bandwidth**.
The voxel version had no plateau -- the count tracked the knob monotonically,
which means the knob was choosing the answer. So: sweep `k` and look for a
plateau. No plateau, and this whole line is done, whatever the bandwidth is
derived from.

Outputs (examples/results/):
  umap3d_knn_density.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from adt_gold_benchmark import load_labeled as load_adt14  # noqa: E402
from adt_multi_validation import load_cite  # noqa: E402
from umap3d_cores import SEED, K, score  # noqa: E402

# Single implementation: the probe measures exactly the code that would ship.
# The first version of this file kept its own copy of ToMATo and the copy had a
# wrong non-merge branch (it moved the current cell's root to the taller peak
# even when the two peaks were *not* merged, so later neighbours in the same
# inner loop were compared against the wrong root). Results from it were
# discarded.
from scfair.pp._granularity import knn_density, knn_graph, merge_peaks  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "umap3d_knn_density.csv"

DIM = 3
SEEDS = [0, 1, 2]
# The first grid stopped at 200 and that was mid-descent: on pbmc5k_adt29
# (4k cells, 8 populations) the count at depth=0.5 ran 42, 28, 20, 15, 13, 12,
# 9, 9 -- still falling, and only flattening in the last two rungs, which span
# 1.67x and so do not clear the plateau bar. 200 is also 1% of the 20k dataset.
# The grid now runs to 5% of the smallest panel member; `k_frac` in the output
# is the same bandwidth as a fraction of n, because which of the two is flat is
# itself the finding:
#   plateau in absolute k        -> the parameter is a neighbour count
#   plateau at matched k/n       -> the parameter is a fraction of the cells
#   neither                      -> the line ends
# Whatever value the plateau sits at is the answer, including if it is not the
# number of labelled populations.
K_DENSITY = [10, 20, 30, 50, 80, 120, 200, 320, 500, 800, 1200]
K_GRAPH = 15  # merge graph, held fixed
DEPTHS = [0.3, 0.4, 0.5, 0.6, 0.7]

LOADERS = {
    "pbmc5k_adt29": lambda: load_cite("pbmc_5k_v3"),
    "pbmc10k_adt14": load_adt14,
    "pbmc_seurat_v4_20k": lambda: load_cite("pbmc_seurat_v4_20k"),
}


def run(dname: str, rows: list) -> None:
    print(f"\n############ {dname} ############", flush=True)
    a = LOADERS[dname]()
    a.X = a.layers["counts"].copy()
    sc.pp.highly_variable_genes(
        a, n_top_genes=min(K, a.n_vars - 1), flavor="seurat_v3", layer="counts"
    )
    genes = a.var_names[a.var["highly_variable"]].tolist()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=SEED)

    conf = (
        a.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in a.obs.columns
        else np.ones(a.n_obs, dtype=bool)
    )
    y_true = a.obs["cell_type"].astype(str)[conf]
    n_pop = int(y_true.nunique())
    print(f"  {a.n_obs} cells, {n_pop} protein-defined populations", flush=True)

    for seed in SEEDS:
        sc.tl.umap(a, n_components=DIM, random_state=seed)
        X = np.asarray(a.obsm["X_umap"], dtype=float)
        gidx = knn_graph(X, K_GRAPH)
        for kd in K_DENSITY:
            rho = knn_density(X, kd)
            line = []
            for depth in DEPTHS:
                lab = merge_peaks(rho, gidx, depth)
                s = score(y_true, lab[conf])
                rows.append(
                    {
                        "dataset": dname,
                        "n_pop": n_pop,
                        "seed": seed,
                        "n_cells": int(a.n_obs),
                        "k_density": kd,
                        "k_frac": round(kd / a.n_obs, 5),
                        "depth": depth,
                        **s,
                    }
                )
                line.append(f"{depth:.1f}:{s['n_clusters']}")
            print(f"  seed={seed} k_density={kd:>3} | k@depth " + " ".join(line), flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)


def main(which=None) -> None:
    rows: list = []
    for dname in which or ["pbmc5k_adt29"]:
        run(dname, rows)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    d = pd.DataFrame(rows)
    print("\n=== plateau check: k at depth=0.5, by bandwidth ===", flush=True)
    for ds in d.dataset.unique():
        sub = d[(d.dataset == ds) & (np.isclose(d.depth, 0.5))]
        piv = sub.pivot_table(
            index="k_density", values="n_clusters", aggfunc=["median", "min", "max"]
        )
        print(f"\n{ds} (truth {int(sub.n_pop.iloc[0])}):", flush=True)
        print(piv.to_string(), flush=True)
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
