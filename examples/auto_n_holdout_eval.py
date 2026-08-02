#!/usr/bin/env python
"""Hold-out evaluation of structure-aware auto_n (rule v4).

Datasets here must NOT have been used to set thresholds in §5.34.

Default hold-outs (local files):
  - lung_atlas: examples/data/Lung_atlas_public.h5ad  (obs['cell_type'])
  - seurat_v4_20k: subsample of pbmc_seurat_v4.h5ad  (obs['celltype.l2'])

For each dataset:
  1. structure features @ HVG2000 intermediate graph
  2. pred_k = select_n_top_from_structure(...)
  3. k-sweep ARI (hybrid, allocation none, res grid) for k grid
  4. report ARI(pred)-ARI(2000) and gap to oracle

Usage:
  python examples/auto_n_holdout_eval.py              # all registered holdouts
  python examples/auto_n_holdout_eval.py lung         # lung only
  python examples/auto_n_holdout_eval.py v4           # seurat v4 20k only
  python examples/auto_n_holdout_eval.py cite         # cbmc8k + pbmc_cite GSE100866
  python examples/auto_n_holdout_eval.py gbm          # GSE84465 sorted panning
  python examples/auto_n_holdout_eval.py new          # cite + gbm (downloaded panel)
  python examples/auto_n_holdout_eval.py cellxgene    # CellxGene PBMC author gold
  python examples/auto_n_holdout_eval.py features     # features only (no k-sweep)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

import scfair as scf  # noqa: E402
from scfair.pp._auto_n import select_n_top_from_structure  # noqa: E402
from scfair.pp._granularity import (  # noqa: E402
    DEFAULT_DEPTH,
    _embedding_3d,
    default_bandwidth,
    knn_density,
    knn_graph,
    population_count_from_embedding,
)
from scfair.pp._highly_variable_genes import (  # noqa: E402
    _nearest_cluster_map,
    _pair_bootstrap_stability,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
FEAT_CSV = OUT / "auto_n_holdout_features.csv"
SWEEP_CSV = OUT / "auto_n_holdout_ksweep.csv"
SUM_CSV = OUT / "auto_n_holdout_summary.csv"

K_GRID = [500, 1000, 1500, 2000, 3000, 4000]
RES_GRID = [0.3, 0.5, 0.8, 1.2]
SEEDS = [0, 1]
INT_RES = 0.5


def _ensure_counts(a: ad.AnnData) -> ad.AnnData:
    a = a.copy()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    return a


def load_lung():
    a = ad.read_h5ad(DATA / "Lung_atlas_public.h5ad")
    a = _ensure_counts(a)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    # drop missing
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    return a


def load_seurat_v4_20k(n=20000, seed=0):
    a = ad.read_h5ad(DATA / "pbmc_seurat_v4.h5ad")
    a = _ensure_counts(a)
    # prefer l2 if present
    lab = "celltype.l2" if "celltype.l2" in a.obs.columns else "celltype.l1"
    a.obs["cell_type"] = a.obs[lab].astype(str)
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    if a.n_obs > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(a.n_obs, size=n, replace=False)
        a = a[idx].copy()
    return a


def load_cbmc8k_cite():
    a = ad.read_h5ad(DATA / "cbmc8k_cite_holdout.h5ad")
    a = _ensure_counts(a)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    return a


def load_pbmc_cite_gse100866():
    a = ad.read_h5ad(DATA / "pbmc_cite_gse100866_holdout.h5ad")
    a = _ensure_counts(a)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    return a


def load_gbm_sd():
    a = ad.read_h5ad(DATA / "gbm_sd_gse84465_holdout.h5ad")
    a = _ensure_counts(a)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    # drop Unpanned — not a sorted gold class (mixed population)
    a = a[
        a.obs["cell_type"].notna()
        & (a.obs["cell_type"] != "nan")
        & (a.obs["cell_type"] != "Unpanned")
    ].copy()
    return a


def load_pbmc_cellxgene_gold():
    """CellxGene 10x v3 PBMC with author coarse labels (not in auto_n cal)."""
    a = ad.read_h5ad(DATA / "pbmc_10x_v3_cellxgene_gold.h5ad")
    a = _ensure_counts(a)
    lab = "author_cell_type" if "author_cell_type" in a.obs.columns else "cell_type"
    a.obs["cell_type"] = a.obs[lab].astype(str)
    a = a[
        a.obs["cell_type"].notna()
        & (a.obs["cell_type"] != "nan")
        & (a.obs["cell_type"] != "unknown")
        & (a.obs["cell_type"] != "multiplet")
    ].copy()
    return a


def load_pbmc10k_v3_labeled():
    """10x v3 PBMC with 8-type labels (not in auto_n cal; not cellxgene 5-type)."""
    a = ad.read_h5ad(DATA / "pbmc_10k_v3_labeled.h5ad")
    a = _ensure_counts(a)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    return a


def load_lung_15k(n=15000, seed=0):
    """Lung atlas subsample — formal holdout (full ~32k is slow for 2-seed k-sweep)."""
    a = load_lung()
    if a.n_obs > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(a.n_obs, size=n, replace=False)
        a = a[idx].copy()
    return a


LOADERS = {
    "lung_atlas": load_lung,
    "lung_atlas_15k": load_lung_15k,
    "seurat_v4_20k": load_seurat_v4_20k,
    "cbmc8k_cite": load_cbmc8k_cite,
    "pbmc_cite_gse100866": load_pbmc_cite_gse100866,
    "gbm_sd": load_gbm_sd,
    "pbmc_cellxgene_gold": load_pbmc_cellxgene_gold,
    "pbmc10k_v3_labeled": load_pbmc10k_v3_labeled,
}


def extract_features(adata, seed=0) -> dict:
    ad0 = adata.copy()
    scf.pp.highly_variable_genes(
        ad0,
        n_top_genes=2000,
        balance_method="none",
        random_state=seed,
        diagnose=False,
    )
    e = ad0[:, ad0.var["highly_variable"]].copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    n_pcs = min(30, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=n_pcs, random_state=seed)

    sc.tl.leiden(
        e,
        resolution=INT_RES,
        key_added="L",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    labels = e.obs["L"].astype(str)
    sizes = labels.value_counts()
    active = [c for c in sizes.index if sizes[c] >= 30]
    X_pca = np.asarray(e.obsm["X_pca"])
    scores = []
    if len(active) >= 2:
        masks = {c: (labels == c).to_numpy() for c in active}
        try:
            nn = _nearest_cluster_map(X_pca, masks)
        except Exception:
            nn = {}
        seen = set()
        for c, nbr in nn.items():
            key = (c, nbr) if c < nbr else (nbr, c)
            if key in seen:
                continue
            seen.add(key)
            scores.append(
                _pair_bootstrap_stability(
                    X_pca,
                    masks[c],
                    masks[nbr],
                    random_state=seed,
                )
            )
    mean_stab = float(np.mean(scores)) if scores else float("nan")
    min_stab = float(np.min(scores)) if scores else float("nan")

    X3 = _embedding_3d(e, n_components=3, random_state=seed, neighbors_key=None)
    bw = default_bandwidth(X3.shape[0])
    nbrs = knn_graph(X3, 15)
    rho = knn_density(X3, bw)
    # valley median via same instrument as structure_solve
    n = rho.size
    order = np.argsort(-rho, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    parent = np.full(n, -1, dtype=np.int64)
    valleys = []

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in order:
        higher = nbrs[i][rank[nbrs[i]] < rank[i]]
        if higher.size == 0:
            parent[i] = i
            continue
        ri = find(int(higher[np.argmin(rank[higher])]))
        parent[i] = ri
        for j in higher:
            rj = find(int(j))
            if rj == ri:
                continue
            lo, hi = (rj, ri) if rho[rj] < rho[ri] else (ri, rj)
            v = float((rho[lo] - rho[i]) / max(rho[lo], 1e-12))
            valleys.append(max(0.0, min(1.0, v)))
            if v < DEFAULT_DEPTH:
                parent[lo] = hi
            ri = find(i)
    arr = np.asarray(valleys) if valleys else np.array([np.nan])
    est = population_count_from_embedding(X3, depth=DEFAULT_DEPTH, bandwidth=bw)
    feat = {
        "n_obs": int(adata.n_obs),
        "n_leiden": int(labels.nunique()),
        "n_density_pops": int(est.n_populations or 0),
        "valley_median": float(np.nanmedian(arr)),
        "valley_mean": float(np.nanmean(arr)),
        "frac_shallow": float(np.mean(arr < DEFAULT_DEPTH)) if np.isfinite(arr).any() else np.nan,
        "mean_stability": mean_stab,
        "min_stability": min_stab,
    }
    common = dict(
        valley_median=feat["valley_median"],
        frac_shallow=feat["frac_shallow"],
        n_density_pops=feat["n_density_pops"],
        mean_stability=feat["mean_stability"],
        min_stability=feat["min_stability"],
        n_obs=feat.get("n_obs"),
        n_leiden=feat.get("n_leiden"),
    )
    feat["pred_k"] = int(select_n_top_from_structure(**common, version="v6"))
    feat["pred_k_v4"] = int(select_n_top_from_structure(**common, version="v4"))
    feat["pred_k_v5"] = int(select_n_top_from_structure(**common, version="v5"))
    return feat


def evaluate_genes(adata, genes, seed, res_grid=RES_GRID):
    e = adata.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    keep = [g for g in genes if g in e.var_names]
    e = e[:, keep].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
    y_true = e.obs["cell_type"].astype(str)
    aris = []
    for res in res_grid:
        sc.tl.leiden(
            e,
            resolution=res,
            key_added="L",
            flavor="igraph",
            n_iterations=2,
            random_state=seed,
        )
        aris.append(adjusted_rand_score(y_true, e.obs["L"].astype(str)))
    return float(np.mean(aris))


def run_ksweep(name: str, adata, seeds=SEEDS, k_grid=K_GRID):
    rows = []
    feat_rows = []
    for seed in seeds:
        print(f"\n### {name} seed={seed} features", flush=True)
        feat = extract_features(adata, seed=seed)
        feat["dataset"] = name
        feat["seed"] = seed
        feat_rows.append(feat)
        pred_k = feat["pred_k"]
        print(
            f"  dens={feat['n_density_pops']} valley_med={feat['valley_median']:.3f} "
            f"frac_shallow={feat['frac_shallow']:.2f} "
            f"mean_stab={feat['mean_stability']:.3f} → pred_k={pred_k}",
            flush=True,
        )
        for k in k_grid:
            t0 = time.time()
            ad = adata.copy()
            scf.pp.highly_variable_genes(
                ad,
                n_top_genes=k,
                balance_method="hybrid",
                resolution=INT_RES,
                allocation_method="none",
                random_state=seed,
                diagnose=False,
            )
            genes = list(ad.var_names[ad.var["highly_variable"]])
            ari = evaluate_genes(adata, genes, seed)
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "k": k,
                    "ARI": ari,
                    "pred_k": pred_k,
                    "n_genes": len(genes),
                }
            )
            print(f"  k={k} ARI={ari:.4f} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(SWEEP_CSV, index=False)
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def summarise(sweep: pd.DataFrame, feats: pd.DataFrame):
    # mean over seeds
    g = sweep.groupby(["dataset", "k"], as_index=False)["ARI"].mean()
    summaries = []
    for name, sub in g.groupby("dataset"):
        best_row = sub.loc[sub["ARI"].idxmax()]
        a2000 = float(sub.loc[sub.k == 2000, "ARI"].iloc[0])
        pred_k = int(feats.loc[feats.dataset == name, "pred_k"].mode().iloc[0])
        # ARI at pred_k (mean seeds already in g)
        a_pred = (
            float(sub.loc[sub.k == pred_k, "ARI"].iloc[0]) if (sub.k == pred_k).any() else np.nan
        )
        # if pred not on grid, nearest
        if not np.isfinite(a_pred):
            kk = int(sub.k.iloc[(sub.k - pred_k).abs().argmin()])
            a_pred = float(sub.loc[sub.k == kk, "ARI"].iloc[0])
            pred_k_used = kk
        else:
            pred_k_used = pred_k
        summaries.append(
            {
                "dataset": name,
                "pred_k": pred_k,
                "pred_k_used": pred_k_used,
                "best_k": int(best_row.k),
                "ARI_pred": a_pred,
                "ARI_2000": a2000,
                "ARI_best": float(best_row.ARI),
                "d_vs_2000": a_pred - a2000,
                "gap_to_oracle": float(best_row.ARI) - a_pred,
                "n_types": int(LOADERS[name]().obs["cell_type"].nunique())
                if name in LOADERS
                else np.nan,
            }
        )
    out = pd.DataFrame(summaries)
    out.to_csv(SUM_CSV, index=False)
    print("\n=== HOLDOUT SUMMARY ===", flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    if len(out):
        print(
            f"\nmean Δ vs 2000: {out['d_vs_2000'].mean():+.4f}  "
            f"oracle recovery: {out['d_vs_2000'].sum() / max(out['ARI_best'].sum() - out['ARI_2000'].sum(), 1e-9):.1%}",
            flush=True,
        )
    return out


def main(which: str):
    names = list(LOADERS) if which in ("all", "default") else [which]
    if which == "features":
        # features only
        all_f = []
        for name in LOADERS:
            print(f"features {name}", flush=True)
            a = LOADERS[name]()
            for seed in SEEDS:
                f = extract_features(a, seed)
                f["dataset"] = name
                f["seed"] = seed
                all_f.append(f)
                print(f"  seed={seed} pred_k={f['pred_k']}", flush=True)
        pd.DataFrame(all_f).to_csv(FEAT_CSV, index=False)
        print(f"wrote {FEAT_CSV}", flush=True)
        return

    if which == "lung":
        names = ["lung_atlas"]
    elif which == "v4":
        names = ["seurat_v4_20k"]
    elif which == "cite":
        names = ["cbmc8k_cite", "pbmc_cite_gse100866"]
    elif which == "gbm":
        names = ["gbm_sd"]
    elif which == "new":
        # newly downloaded holdouts only (not lung/v4)
        names = ["cbmc8k_cite", "pbmc_cite_gse100866", "gbm_sd"]
    elif which in ("cellxgene", "pbmc_cellxgene_gold"):
        names = ["pbmc_cellxgene_gold"]
    elif which in ("untouched", "fresh"):
        # not previously in holdout CSVs: lung subsample + 8-type PBMC
        names = ["lung_atlas_15k", "pbmc10k_v3_labeled"]
    elif which in LOADERS:
        names = [which]

    all_sweep, all_feat = [], []
    for name in names:
        print(f"\n######## HOLDOUT {name} ########", flush=True)
        a = LOADERS[name]()
        print(f"shape={a.shape} n_types={a.obs['cell_type'].nunique()}", flush=True)
        sw, ft = run_ksweep(name, a)
        all_sweep.append(sw)
        all_feat.append(ft)
        del a
    sweep = pd.concat(all_sweep, ignore_index=True)
    feats = pd.concat(all_feat, ignore_index=True)
    sweep.to_csv(SWEEP_CSV, index=False)
    feats.to_csv(FEAT_CSV, index=False)
    summarise(sweep, feats)
    print(f"\nwrote {SWEEP_CSV}\nwrote {FEAT_CSV}\nwrote {SUM_CSV}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(which)
