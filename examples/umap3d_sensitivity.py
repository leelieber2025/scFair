#!/usr/bin/env python
"""Is the core count read off the data, or off the knobs?

`umap3d_cores.py` recovered exactly the 8 protein-defined populations of
pbmc5k_adt29 at `depth=0.5` -- but that run used one UMAP seed and one
voxelisation (BINS=48, SMOOTH=1.2). The method is only worth anything if the
count survives all three:

  seed    UMAP is stochastic; a count that moves with the seed is noise
  BINS    voxels per axis -- the grid the density field is built on
  SMOOTH  gaussian sigma in voxels -- how much ripple is erased before flooding

The claim being tested is "one interpretable knob (`depth`), the rest are
implementation detail". If the count at depth=0.5 wanders with BINS or SMOOTH,
that claim is false and the method has three knobs, not one.

Reported: the count at each depth for every (seed, BINS, SMOOTH), and the
spread of the count at the a-priori depth=0.5.

Outputs (examples/results/):
  umap3d_sensitivity.csv
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
from adt_multi_validation import load_cite  # noqa: E402
from umap3d_cores import (  # noqa: E402
    DEPTHS,
    SEED,
    K,
    flood,
    merge_by_depth,
    score,
    voxelise,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "umap3d_sensitivity.csv"

DATASET = "pbmc5k_adt29"
SEEDS = [0, 1, 2, 3, 4]
BINS_GRID = [32, 48, 64]
SMOOTH_GRID = [0.8, 1.2, 1.6]
DIM = 3


def main() -> None:
    a = load_cite("pbmc_5k_v3")
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
    print(f"{a.n_obs} cells, {n_pop} protein-defined populations", flush=True)

    rows: list = []
    for seed in SEEDS:
        sc.tl.umap(a, n_components=DIM, random_state=seed)
        X = np.asarray(a.obsm["X_umap"], dtype=float)
        for bins in BINS_GRID:
            for smooth in SMOOTH_GRID:
                field, ix = voxelise(X, bins, smooth)
                occupied = field > field.max() * 0.01
                lab_img, peaks, saddles = flood(field, occupied)
                cell_vox = lab_img[tuple(ix.T)]
                line = []
                for depth in DEPTHS:
                    mapping = merge_by_depth(peaks, saddles, depth)
                    lab = np.array([mapping.get(int(v), -1) for v in cell_vox])
                    s = score(y_true, lab[conf])
                    rows.append(
                        {
                            "dataset": DATASET,
                            "n_pop": n_pop,
                            "seed": seed,
                            "bins": bins,
                            "smooth": smooth,
                            "depth": depth,
                            "n_cores": len(peaks),
                            **s,
                        }
                    )
                    line.append(f"{depth:.1f}:{s['n_clusters']}")
                print(
                    f"  seed={seed} bins={bins} smooth={smooth} "
                    f"cores={len(peaks):>3} | k@depth " + " ".join(line),
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(CSV, index=False)

    d = pd.DataFrame(rows)
    at5 = d[np.isclose(d["depth"], 0.5)]
    print(
        f"\nk at depth=0.5 over {len(at5)} configs: "
        f"median={at5['n_clusters'].median():.0f} "
        f"min={at5['n_clusters'].min()} max={at5['n_clusters'].max()} "
        f"(truth {n_pop})",
        flush=True,
    )
    print(
        at5.groupby(["bins", "smooth"])["n_clusters"].agg(["median", "min", "max"]).to_string(),
        flush=True,
    )
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
