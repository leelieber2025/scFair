#!/usr/bin/env python
"""Find each population's density core in a 3D embedding, by image analysis.

The premise
-----------
Every population has a most-concentrated region -- a density core. Finding
cores by voxelising the space only works in a narrow band of dimensions:

    d=2   48^2 =  2,304 voxels, but populations overlap in projection
    d=3   48^3 = 110,592 voxels  <- 4k cells is enough to fill out the shape
    d=50  48^50 voxels           <- the regime where the field's methods live,
                                    which is why they use classifiers and
                                    permutation tests instead of geometry
                                    (sc-SHC, CHOIR)

So this is not "cluster the UMAP" (a discredited 2D practice). It is: use the
one dimensionality where a density field can actually be built, which is a
dimensionality nobody renders because nobody looks at it.

Also relevant to why 2D is not enough: segment a 2D field into connected
regions and the region-adjacency graph is necessarily planar -- at most 3n-6
of the n(n-1)/2 adjacencies. At n=19 that is 51/171 = 30%. 3D has no bound.

The algorithm (one knob)
------------------------
1. 3D UMAP from the kNN graph the package already builds.
2. Voxelise into a B^3 histogram, Gaussian-smoothed -> a 3D image.
3. Flood from the densest voxel downwards (priority flood = watershed).
   A new local maximum starts a region; a voxel touching two regions IS the
   saddle between them, so the valley depth is recorded for free.
4. Merge two regions when the valley between them is shallow *relative to the
   shorter core*: merge if (peak_lo - saddle) / peak_lo < `depth`. Dimensionless,
   so it needs no per-dataset tuning. depth=0.5 reads as "the valley must drop
   below half the shorter core's height to count as two populations".
5. Assign every cell to its voxel's region.

Cores outnumber cell types (sub-states have their own small cores), so step 4
is not optional -- without it the field shatters.

What this run answers
---------------------
1. How many regions survive, and is the count stable over `depth`? A plateau
   means the answer is being read off the data rather than off the knob.
2. Does it agree with the protein-defined populations better than Leiden swept
   over its whole resolution grid?
3. Does d=3 actually beat d=2, as the planarity argument predicts (and does the
   gap widen with cluster count)?

Non-circular labels only. No library change; this is a probe.

Outputs (examples/results/):
  umap3d_cores.csv    one row per (dataset, embedding, method, param)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import ndimage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from adt_gold_benchmark import load_labeled as load_adt14  # noqa: E402
from adt_multi_validation import load_cite  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "umap3d_cores.csv"

K = 2000
SEED = 0
RES_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
# 2D dropped after the first run: it undercounts, and the error grows with the
# population count (-2, -2, -7 at 8/9/16 populations) exactly as the planarity
# bound predicts. Nothing further to learn from it.
DIMS = [3]
BINS = 48  # voxels per axis
SMOOTH = 1.2  # gaussian sigma, in voxels
DEPTHS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

LOADERS = {
    "pbmc5k_adt29": lambda: load_cite("pbmc_5k_v3"),
    "pbmc10k_adt14": load_adt14,
    "pbmc_seurat_v4_20k": lambda: load_cite("pbmc_seurat_v4_20k"),
}


# ---------------------------------------------------------------------------
# the 3D image
# ---------------------------------------------------------------------------
def voxelise(X: np.ndarray, bins: int, smooth: float):
    """Histogram the embedding into a bins^d image; return field + cell->voxel."""
    d = X.shape[1]
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    ix = np.clip(((X - lo) / span * bins).astype(np.int64), 0, bins - 1)
    field = np.zeros((bins,) * d, dtype=np.float64)
    np.add.at(field, tuple(ix.T), 1.0)
    field = ndimage.gaussian_filter(field, sigma=smooth, mode="constant")
    return field, ix


def flood(field: np.ndarray, occupied: np.ndarray):
    """Watershed by descending flood. Returns (labels, peaks, saddles).

    labels   : int array over voxels, -1 where not occupied
    peaks    : {region_id: peak height}
    saddles  : list of (region_a, region_b, saddle_height), in flood order
    """
    shape = field.shape
    d = field.ndim
    flat = field.ravel()
    occ = occupied.ravel()
    order = np.argsort(-flat, kind="stable")
    order = order[occ[order]]

    lab = np.full(flat.size, -1, dtype=np.int64)
    parent: dict[int, int] = {}
    peaks: dict[int, float] = {}
    saddles: list[tuple[int, int, float]] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    strides = np.array(shape[1:][::-1], dtype=np.int64)
    strides = np.concatenate([np.cumprod(strides)[::-1], [1]])
    offsets = []
    for step in np.ndindex(*(3,) * d):
        delta = np.array(step, dtype=np.int64) - 1
        if not delta.any():
            continue
        offsets.append(delta)
    offsets = np.array(offsets)

    coords = np.array(np.unravel_index(order, shape)).T
    next_id = 0
    for pos, vox in zip(coords, order):
        nb = pos + offsets
        ok = ((nb >= 0) & (nb < np.array(shape))).all(axis=1)
        nb = nb[ok]
        nbf = (nb * strides).sum(axis=1)
        seen = {find(int(lab[j])) for j in nbf if lab[j] >= 0}
        if not seen:
            parent[next_id] = next_id
            peaks[next_id] = float(flat[vox])
            lab[vox] = next_id
            next_id += 1
            continue
        seen = sorted(seen, key=lambda r: -peaks[r])
        keep = seen[0]
        lab[vox] = keep
        for other in seen[1:]:
            saddles.append((keep, other, float(flat[vox])))
            parent[other] = keep  # provisional: merged at this saddle
    return lab.reshape(shape), peaks, saddles


def merge_by_depth(peaks: dict, saddles: list, depth: float) -> dict[int, int]:
    """Union regions whose separating valley is shallower than `depth`.

    Valley depth is relative to the *shorter* of the two cores, so the rule is
    scale-free: (peak_lo - saddle) / peak_lo < depth  ->  same population.
    """
    parent = {r: r for r in peaks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    height = dict(peaks)
    # Saddles arrive in descending flood order, so the first saddle joining two
    # regions is the highest point of the pass between them -- the shallowest
    # valley, and the one the rule must be applied to.
    for a, b, s in saddles:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lo, hi = (ra, rb) if height[ra] < height[rb] else (rb, ra)
        if height[lo] <= 0 or (height[lo] - s) / height[lo] < depth:
            parent[lo] = hi  # valley too shallow: one population
        # else: the pass drops far enough below the shorter core -- keep both
    return {r: find(r) for r in peaks}


# ---------------------------------------------------------------------------
def score(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    yp = pd.Series([str(v) for v in y_pred], index=y_true.index)
    prev = y_true.value_counts(normalize=True)
    f1s = {}
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in yp.unique():
            p = (yp == cl).to_numpy()
            tp = float(np.sum(t & p))
            if tp:
                pr, rc = tp / p.sum(), tp / t.sum()
                best = max(best, 2 * pr * rc / (pr + rc))
        f1s[pop] = best
    rare = [p for p in f1s if prev.get(p, 0) < 0.02]
    return {
        "n_clusters": int(yp.nunique()),
        "ARI": float(adjusted_rand_score(y_true, yp)),
        "NMI": float(normalized_mutual_info_score(y_true, yp)),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "min_pop_f1": float(f1s[str(prev.idxmin())]),
        "rare_f1_mean": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
    }


def run(dname: str, rows: list) -> None:
    print(f"\n############ {dname} ############", flush=True)
    t0 = time.time()
    adata = LOADERS[dname]()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes ({time.time() - t0:.0f}s)", flush=True)

    a = adata.copy()
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
    base = {"dataset": dname, "n_cells": int(a.n_obs), "n_pop": int(y_true.nunique())}
    print(
        f"  {base['n_pop']} protein-defined populations, {int(conf.sum())} confident cells",
        flush=True,
    )

    for res in RES_GRID:
        sc.tl.leiden(
            a, resolution=res, key_added="L", flavor="igraph", n_iterations=2, random_state=SEED
        )
        s = score(y_true, a.obs["L"].astype(str)[conf].to_numpy())
        rows.append({**base, "method": "leiden", "embedding": "pca30", "param": res, **s})
        print(
            f"  leiden res={res:<4} k={s['n_clusters']:>3} "
            f"ARI={s['ARI']:.3f} f1={s['macro_f1']:.3f}",
            flush=True,
        )

    for d in DIMS:
        t = time.time()
        sc.tl.umap(a, n_components=d, random_state=SEED)
        X = np.asarray(a.obsm["X_umap"], dtype=float)
        field, ix = voxelise(X, BINS, SMOOTH)
        occupied = field > field.max() * 0.01
        lab_img, peaks, saddles = flood(field, occupied)
        print(
            f"  [d={d}] umap {time.time() - t:.0f}s  "
            f"{occupied.sum()} occupied voxels, {len(peaks)} cores",
            flush=True,
        )

        cell_vox = lab_img[tuple(ix.T)]
        for depth in DEPTHS:
            mapping = merge_by_depth(peaks, saddles, depth)
            lab = np.array([mapping.get(int(v), -1) for v in cell_vox])
            s = score(y_true, lab[conf])
            rows.append(
                {
                    **base,
                    "method": "cores",
                    "embedding": f"umap{d}d",
                    "param": depth,
                    "n_cores": len(peaks),
                    **s,
                }
            )
            print(
                f"    depth={depth:.1f} k={s['n_clusters']:>3} "
                f"ARI={s['ARI']:.3f} f1={s['macro_f1']:.3f}",
                flush=True,
            )


def main(which=None) -> None:
    rows: list = []
    for dname in which or ["pbmc5k_adt29"]:
        run(dname, rows)
        pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
