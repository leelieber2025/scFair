#!/usr/bin/env python
"""Do 3D-UMAP density-field features track / predict best_k (ARI)?

Builds the same intermediate space hybrid uses at k=2000 (seurat_v3 top-2000
→ normalize → log1p → PCA → neighbors), then:

  1. 3D UMAP density field (scfair.pp._granularity)
  2. multi-depth population counts + size imbalance
  3. Leiden@0.5 cluster count (for comparison with separability probe)
  4. optional: NN bootstrap stability (same family as auto_n_separability_probe)

Correlates features with known best_k from the expanded k-sweep table, and
fits a tiny leave-one-out linear predictor (n is small — descriptive only).

Outputs:
  examples/results/umap3d_bestk_probe.csv
  examples/results/umap3d_bestk_corr.csv

Usage:
  python examples/umap3d_bestk_probe.py
  python examples/umap3d_bestk_probe.py quick   # 4 datasets
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402
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
CSV = OUT / "umap3d_bestk_probe.csv"
CORR = OUT / "umap3d_bestk_corr.csv"

# From the expanded k-sweep discussion (mean over seeds/res); best_k_ari.
# duo8 still rising at 4000 — coded as 4000.
BEST_K = {
    "paul15": 1000,
    "pbmc3k_louvain": 500,
    "pancreas_smartseq2": 500,
    "duo4_pbmc": 3000,
    "duo8_pbmc": 4000,
    "duo4un_pbmc": 3000,
    "pbmc5k_adt29": 1500,
    "crafted_base": 500,
}
ARI_AT_2000 = {
    "paul15": 0.322,
    "pbmc3k_louvain": 0.736,
    "pancreas_smartseq2": 0.505,
    "duo4_pbmc": 0.808,
    "duo8_pbmc": 0.626,
    "duo4un_pbmc": 0.635,
    "pbmc5k_adt29": 0.612,
    "crafted_base": 0.273,
}
ARI_AT_BEST = {
    "paul15": 0.347,
    "pbmc3k_louvain": 0.791,
    "pancreas_smartseq2": 0.634,
    "duo4_pbmc": 0.837,
    "duo8_pbmc": 0.646,
    "duo4un_pbmc": 0.675,
    "pbmc5k_adt29": 0.616,
    "crafted_base": 0.294,
}
LABEL_KIND = {
    "paul15": "SHADOWED",
    "pbmc3k_louvain": "SHADOWED",
    "pancreas_smartseq2": "SHADOWED",
    "duo4_pbmc": "independent",
    "duo8_pbmc": "independent",
    "duo4un_pbmc": "independent",
    "pbmc5k_adt29": "independent",
    "crafted_base": "independent",
}

DEPTHS = (0.3, 0.5, 0.7)
SEEDS = (0, 1)


def _gini(sizes: np.ndarray) -> float:
    x = np.sort(np.asarray(sizes, dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def _valley_stats(rho: np.ndarray, nbrs: np.ndarray, depth: float) -> dict:
    """Instrument ToMATo: collect valley depths at first contact of two peaks."""
    n = rho.size
    order = np.argsort(-rho, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    parent = np.full(n, -1, dtype=np.int64)
    valleys: list[float] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    n_raw_peaks = 0
    for i in order:
        nb = nbrs[i]
        higher = nb[rank[nb] < rank[i]]
        if higher.size == 0:
            parent[i] = i
            n_raw_peaks += 1
            continue
        ri = find(int(higher[np.argmin(rank[higher])]))
        parent[i] = ri
        for j in higher:
            rj = find(int(j))
            if rj == ri:
                continue
            lo, hi = (rj, ri) if rho[rj] < rho[ri] else (ri, rj)
            # relative drop from shorter peak to the pass cell
            v = float((rho[lo] - rho[i]) / max(rho[lo], 1e-12))
            valleys.append(max(0.0, min(1.0, v)))
            if v < depth:
                parent[lo] = hi
            ri = find(i)
    labels = np.array([find(i) for i in range(n)])
    n_merged = int(pd.unique(labels).size)
    arr = np.asarray(valleys, dtype=float) if valleys else np.array([np.nan])
    return {
        "n_raw_peaks": n_raw_peaks,
        "n_merged_pops": n_merged,
        "n_merges_events": max(0, n_raw_peaks - n_merged),
        "valley_mean": float(np.nanmean(arr)),
        "valley_median": float(np.nanmedian(arr)),
        "valley_min": float(np.nanmin(arr)),
        "valley_p25": float(np.nanpercentile(arr, 25)) if np.isfinite(arr).any() else np.nan,
        "frac_shallow_valleys": float(np.mean(arr < depth)) if np.isfinite(arr).any() else np.nan,
    }


def prepare_graph(adata, seed: int = 0):
    """HVG@2000 intermediate graph (product-like)."""
    ad = adata.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=2000,
        balance_method="none",
        random_state=seed,
        diagnose=False,
    )
    e = ad[:, ad.var["highly_variable"]].copy()
    layer = "counts" if "counts" in e.layers else None
    if layer:
        e.X = e.layers[layer].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    n_pcs = min(30, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=n_pcs, random_state=seed)
    return e


def leiden_and_stability(e, seed: int = 0) -> dict:
    sc.tl.leiden(
        e,
        resolution=0.5,
        key_added="L",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    labels = e.obs["L"].astype(str)
    sizes = labels.value_counts()
    active = [c for c in sizes.index if sizes[c] >= 30]
    out = {
        "n_leiden": int(labels.nunique()),
        "n_leiden_kept": len(active),
        "leiden_max_frac": float(sizes.max() / sizes.sum()),
        "leiden_gini": _gini(sizes.to_numpy()),
    }
    if len(active) < 2:
        out.update(mean_stability=np.nan, min_stability=np.nan, n_stab_pairs=0)
        return out

    X_pca = np.asarray(e.obsm["X_pca"])
    masks = {c: (labels == c).to_numpy() for c in active}
    try:
        nn = _nearest_cluster_map(X_pca, masks)
    except Exception:
        out.update(mean_stability=np.nan, min_stability=np.nan, n_stab_pairs=0)
        return out
    seen = set()
    scores = []
    for c, nbr in nn.items():
        key = (c, nbr) if c < nbr else (nbr, c)
        if key in seen:
            continue
        seen.add(key)
        s = _pair_bootstrap_stability(
            X_pca,
            masks[c],
            masks[nbr],
            random_state=seed,
        )
        scores.append(s)
    out["n_stab_pairs"] = len(scores)
    out["mean_stability"] = float(np.mean(scores)) if scores else np.nan
    out["min_stability"] = float(np.min(scores)) if scores else np.nan
    return out


def density_features(e, seed: int = 0) -> dict:
    X3 = _embedding_3d(e, n_components=3, random_state=seed, neighbors_key=None)
    if X3 is None:
        return {"reason": "no_embedding"}
    bw = default_bandwidth(X3.shape[0])
    nbrs = knn_graph(X3, 15)
    rho = knn_density(X3, bw)
    feat: dict = {
        "reason": "ok",
        "bandwidth": bw,
        "rho_mean": float(rho.mean()),
        "rho_std": float(rho.std()),
        "rho_cv": float(rho.std() / max(rho.mean(), 1e-12)),
    }
    # default depth full estimate + size stats
    est = population_count_from_embedding(X3, depth=DEFAULT_DEPTH, bandwidth=bw)
    feat["n_density_pops"] = est.n_populations
    if est.labels is not None:
        _, counts = np.unique(est.labels, return_counts=True)
        feat["density_max_frac"] = float(counts.max() / counts.sum())
        feat["density_gini"] = _gini(counts)
        feat["density_n_small"] = int(np.sum(counts < max(30, 0.02 * X3.shape[0])))
    # multi-depth counts
    for d in DEPTHS:
        est_d = population_count_from_embedding(X3, depth=d, bandwidth=bw)
        feat[f"n_pops_depth_{d}"] = est_d.n_populations
    # sensitivity: how much count changes when merge is stricter
    if feat.get("n_pops_depth_0.3") and feat.get("n_pops_depth_0.7"):
        feat["n_pops_depth_span"] = int(feat["n_pops_depth_0.3"] - feat["n_pops_depth_0.7"])
    else:
        feat["n_pops_depth_span"] = np.nan
    # valley instrumentation at default depth
    v = _valley_stats(rho, nbrs, DEFAULT_DEPTH)
    feat.update(v)
    # raw peaks vs merged: over-split proxy in density field
    if feat.get("n_raw_peaks") and feat.get("n_merged_pops"):
        feat["peak_merge_ratio"] = float(feat["n_merged_pops"] / max(feat["n_raw_peaks"], 1))
    return feat


def probe_one(name: str, seed: int = 0) -> dict:
    a = LOADERS[name]()
    e = prepare_graph(a, seed=seed)
    row = {
        "dataset": name,
        "seed": seed,
        "label_kind": LABEL_KIND[name],
        "best_k": BEST_K[name],
        "ari_2000": ARI_AT_2000[name],
        "ari_best": ARI_AT_BEST[name],
        "delta_ari": ARI_AT_BEST[name] - ARI_AT_2000[name],
        "log_best_k": float(np.log10(BEST_K[name])),
    }
    row.update(leiden_and_stability(e, seed=seed))
    row.update(density_features(e, seed=seed))
    return row


def correlate(df: pd.DataFrame) -> pd.DataFrame:
    # mean over seeds
    num_cols = [
        c
        for c in df.columns
        if c not in ("dataset", "seed", "label_kind", "reason")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    g = df.groupby("dataset", as_index=False)[num_cols].mean()
    y = g["best_k"]
    y_log = g["log_best_k"]
    rows = []
    for c in num_cols:
        if c in ("best_k", "log_best_k", "ari_2000", "ari_best", "delta_ari"):
            continue
        x = g[c].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y.to_numpy(dtype=float))
        if mask.sum() < 4 or np.unique(x[mask]).size < 2:
            continue
        sp = stats.spearmanr(x[mask], y.to_numpy()[mask])
        sp_log = stats.spearmanr(x[mask], y_log.to_numpy()[mask])
        pr = stats.pearsonr(x[mask], y.to_numpy()[mask])
        rows.append(
            {
                "feature": c,
                "n": int(mask.sum()),
                "spearman_best_k": float(sp.correlation) if sp.correlation is not None else np.nan,
                "spearman_p": float(sp.pvalue) if sp.pvalue is not None else np.nan,
                "spearman_log_best_k": float(sp_log.correlation)
                if sp_log.correlation is not None
                else np.nan,
                "pearson_best_k": float(pr[0]),
                "pearson_p": float(pr[1]),
            }
        )
    out = pd.DataFrame(rows).sort_values("spearman_best_k", key=lambda s: s.abs(), ascending=False)
    return g, out


def loo_predict(g: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Leave-one-out linear regression on log10(best_k)."""
    from numpy.linalg import lstsq

    y = np.log10(g["best_k"].to_numpy(dtype=float))
    X = g[features].to_numpy(dtype=float)
    # drop non-finite
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    names = g["dataset"].to_numpy()[ok]
    # standardize
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd
    preds = np.zeros_like(y)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        A = np.column_stack([np.ones(mask.sum()), Xs[mask]])
        coef, _, _, _ = lstsq(A, y[mask], rcond=None)
        preds[i] = coef[0] + Xs[i] @ coef[1:]
    pred_k = np.clip(np.round(10**preds / 500) * 500, 500, 4000).astype(int)
    true_k = g["best_k"].to_numpy()[ok]
    return pd.DataFrame(
        {
            "dataset": names,
            "best_k": true_k,
            "pred_log10_k": preds,
            "pred_k_rounded": pred_k,
            "abs_err_k": np.abs(pred_k - true_k),
            "features": ",".join(features),
        }
    )


def main(datasets: list[str]):
    rows = []
    for name in datasets:
        if name not in LOADERS:
            print(f"skip {name}: no loader", flush=True)
            continue
        for seed in SEEDS:
            print(f"### {name} seed={seed}", flush=True)
            try:
                row = probe_one(name, seed=seed)
                rows.append(row)
                print(
                    f"  dens_pops={row.get('n_density_pops')} "
                    f"raw_peaks={row.get('n_raw_peaks')} "
                    f"leiden={row.get('n_leiden')} "
                    f"mean_stab={row.get('mean_stability')} "
                    f"valley_med={row.get('valley_median')} "
                    f"best_k={row.get('best_k')}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            pd.DataFrame(rows).to_csv(CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    g, corr = correlate(df)
    corr.to_csv(CORR, index=False)

    print("\n=== per-dataset means (key cols) ===", flush=True)
    cols = [
        "dataset",
        "label_kind",
        "best_k",
        "n_density_pops",
        "n_raw_peaks",
        "n_merged_pops",
        "peak_merge_ratio",
        "n_pops_depth_span",
        "valley_median",
        "valley_mean",
        "frac_shallow_valleys",
        "density_gini",
        "density_max_frac",
        "n_leiden",
        "mean_stability",
        "min_stability",
    ]
    cols = [c for c in cols if c in g.columns]
    print(g[cols].round(3).to_string(index=False), flush=True)

    print("\n=== Spearman with best_k (top |ρ|) ===", flush=True)
    print(corr.head(15).round(3).to_string(index=False), flush=True)

    # candidate feature sets for LOO
    candidates = [
        ["mean_stability"],
        ["min_stability"],
        ["n_density_pops"],
        ["valley_median"],
        ["peak_merge_ratio"],
        ["n_pops_depth_span"],
        ["mean_stability", "n_density_pops"],
        ["mean_stability", "valley_median"],
        ["mean_stability", "peak_merge_ratio"],
        ["min_stability", "n_density_pops"],
        ["valley_median", "n_density_pops", "peak_merge_ratio"],
    ]
    print("\n=== LOO linear pred on log10(best_k) ===", flush=True)
    for feats in candidates:
        if any(f not in g.columns for f in feats):
            continue
        if g[feats].isna().any().any():
            continue
        pred = loo_predict(g, feats)
        mae = pred["abs_err_k"].mean()
        # baseline: always 2000
        base_mae = np.mean(np.abs(g["best_k"] - 2000))
        sp = stats.spearmanr(pred["best_k"], pred["pred_k_rounded"])
        print(
            f"  feats={feats}  MAE_k={mae:.0f}  (baseline always-2000 MAE={base_mae:.0f})  "
            f"spearman(true,pred_round)={sp.correlation:.3f}",
            flush=True,
        )
        if feats == ["mean_stability"] or feats == ["mean_stability", "valley_median"]:
            print(pred.round(3).to_string(index=False), flush=True)

    print(f"\nwrote {CSV}", flush=True)
    print(f"wrote {CORR}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "quick":
        ds = ["duo4_pbmc", "duo8_pbmc", "crafted_base", "pancreas_smartseq2"]
    else:
        ds = list(BEST_K.keys())
    main(ds)
