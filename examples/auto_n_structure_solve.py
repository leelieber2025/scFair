#!/usr/bin/env python
"""Hunt for a *structure-aware* auto_n that beats fixed k=2000.

Ground truth
------------
``auto_n_populations.csv``: full k-grid ARI curves (seed × res averaged).
``best_k`` = argmax_k mean ARI. Datasets with incomplete grids are dropped.

Features (label-free, computed once on HVG@2000 intermediate space)
-------------------------------------------------------------------
- Leiden@0.5: n_clusters, size gini, NN bootstrap stability (same family as
  cap_merge / separability probe — **without** running full hybrid+cap)
- 3D UMAP density field: n_pops, multi-depth span, valley depth stats,
  peak-merge ratio (from umap3d_bestk_probe)

Evaluation (what matters for auto_n)
------------------------------------
Not just |pred_k - best_k|, but **ARI recovered**:
  ARI(pred_k) − ARI(2000)   on the measured curve
  and gap to oracle: ARI(best) − ARI(pred)

Also LOO rules / linear models and brain-storm discrete bins
  short≤1000 | mid∈{1500,2000} | long≥3000.

Outputs
-------
  examples/results/auto_n_structure_features.csv
  examples/results/auto_n_structure_eval.csv
  examples/results/auto_n_structure_corr.csv

Usage
-----
  python examples/auto_n_structure_solve.py
  python examples/auto_n_structure_solve.py --seeds 0        # faster
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys_path_src = str(ROOT.parent / "src")
import sys

sys.path.insert(0, sys_path_src)
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
FEAT = OUT / "auto_n_structure_features.csv"
EVAL = OUT / "auto_n_structure_eval.csv"
CORR = OUT / "auto_n_structure_corr.csv"
POP = OUT / "auto_n_populations.csv"

SHADOWED = {"paul15", "pbmc3k_louvain", "pancreas_smartseq2"}
K_GRID = [200, 300, 500, 1000, 1500, 2000, 3000, 4000]
SEEDS_DEFAULT = [0, 1]


def _gini(sizes) -> float:
    x = np.sort(np.asarray(list(sizes), dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def load_curves() -> tuple[pd.DataFrame, dict[str, int], set[str]]:
    """dataset × k → mean ARI; best_k; datasets with ≥6 k points."""
    raw = pd.read_csv(POP)
    g = (
        raw.groupby(["dataset", "k", "seed"])["ARI"]
        .mean()
        .groupby(["dataset", "k"])
        .mean()
        .reset_index()
    )
    pivot = g.pivot(index="dataset", columns="k", values="ARI")
    complete = []
    best = {}
    for d in pivot.index:
        row = pivot.loc[d].dropna()
        if len(row) < 6 or 2000 not in row.index:
            print(f"drop incomplete curve: {d} (n_k={len(row)})", flush=True)
            continue
        complete.append(d)
        best[d] = int(row.idxmax())
    return pivot.loc[complete], best, set(complete)


def valley_stats(rho: np.ndarray, nbrs: np.ndarray, depth: float) -> dict:
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

    n_raw = 0
    for i in order:
        nb = nbrs[i]
        higher = nb[rank[nb] < rank[i]]
        if higher.size == 0:
            parent[i] = i
            n_raw += 1
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
            if v < depth:
                parent[lo] = hi
            ri = find(i)
    labels = np.array([find(i) for i in range(n)])
    n_merged = int(pd.unique(labels).size)
    arr = np.asarray(valleys, dtype=float) if valleys else np.array([np.nan])
    return {
        "n_raw_peaks": n_raw,
        "n_merged_pops": n_merged,
        "valley_median": float(np.nanmedian(arr)),
        "valley_mean": float(np.nanmean(arr)),
        "valley_min": float(np.nanmin(arr)),
        "frac_shallow": float(np.mean(arr < depth)) if np.isfinite(arr).any() else np.nan,
        "peak_merge_ratio": float(n_merged / max(n_raw, 1)),
    }


def prepare_graph(adata, seed: int):
    ad = adata.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=2000,
        balance_method="none",
        random_state=seed,
        diagnose=False,
    )
    e = ad[:, ad.var["highly_variable"]].copy()
    if "counts" in e.layers:
        e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    n_pcs = min(30, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=n_pcs, random_state=seed)
    return e


def extract_features(e, seed: int) -> dict:
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
    feat = {
        "n_obs": int(e.n_obs),
        "n_leiden": int(labels.nunique()),
        "n_leiden_kept": len(active),
        "leiden_max_frac": float(sizes.max() / sizes.sum()),
        "leiden_gini": _gini(sizes.to_numpy()),
        "log10_n_obs": float(np.log10(e.n_obs)),
    }

    # stability on PCA (separability-probe family, independent of cap)
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
    feat["n_stab_pairs"] = len(scores)
    feat["mean_stability"] = float(np.mean(scores)) if scores else np.nan
    feat["min_stability"] = float(np.min(scores)) if scores else np.nan
    feat["frac_unstable"] = float(np.mean(np.asarray(scores) < 0.5)) if scores else np.nan

    # 3D density
    X3 = _embedding_3d(e, n_components=3, random_state=seed, neighbors_key=None)
    if X3 is None:
        feat["density_ok"] = 0
        return feat
    feat["density_ok"] = 1
    bw = default_bandwidth(X3.shape[0])
    nbrs = knn_graph(X3, 15)
    rho = knn_density(X3, bw)
    feat["bandwidth"] = bw
    feat.update(valley_stats(rho, nbrs, DEFAULT_DEPTH))
    est = population_count_from_embedding(X3, depth=DEFAULT_DEPTH, bandwidth=bw)
    feat["n_density_pops"] = est.n_populations
    if est.labels is not None:
        _, counts = np.unique(est.labels, return_counts=True)
        feat["density_max_frac"] = float(counts.max() / counts.sum())
        feat["density_gini"] = _gini(counts)
    for d in (0.3, 0.5, 0.7):
        est_d = population_count_from_embedding(X3, depth=d, bandwidth=bw)
        feat[f"n_pops_d{d}"] = est_d.n_populations
    feat["n_pops_depth_span"] = int((feat.get("n_pops_d0.3") or 0) - (feat.get("n_pops_d0.7") or 0))
    # under/over proxies relative to density vs Leiden
    nd = feat.get("n_density_pops") or np.nan
    nl = feat["n_leiden"]
    feat["leiden_over_density"] = float(nl / nd) if nd and nd > 0 else np.nan
    feat["density_over_leiden"] = float(nd / nl) if nl > 0 and nd else np.nan
    # composite scores (brainstorm)
    # coarse_stable: few density cores, shallow valleys, high merge of raw peaks
    vm = feat.get("valley_median", np.nan)
    fs = feat.get("frac_shallow", np.nan)
    feat["score_coarse_shallow"] = float(
        (1.0 if (nd is not None and nd <= 5) else 0.0)
        + (1.0 if (np.isfinite(fs) and fs >= 0.5) else 0.0)
        + (1.0 if (np.isfinite(vm) and vm < 0.2) else 0.0)
    )
    feat["score_fine_deep"] = float(
        (1.0 if (nd is not None and nd >= 6) else 0.0)
        + (1.0 if (np.isfinite(vm) and vm > 0.5) else 0.0)
        + (1.0 if (np.isfinite(fs) and fs < 0.3) else 0.0)
    )
    # residualize-ish: stability not available with same scale as user's 0.99
    # use "deep structure" vs "shallow residual structure"
    feat["shallow_minus_deep"] = feat["score_coarse_shallow"] - feat["score_fine_deep"]
    return feat


def nearest_k(k: float, grid: list[int] = K_GRID) -> int:
    return int(min(grid, key=lambda g: abs(g - k)))


def ari_at(curve: pd.Series, k: int) -> float:
    if k in curve.index and np.isfinite(curve[k]):
        return float(curve[k])
    # nearest available
    avail = curve.dropna()
    if avail.empty:
        return float("nan")
    kk = int(min(avail.index, key=lambda g: abs(g - k)))
    return float(avail[kk])


# ---------------------------------------------------------------------------
# predictors: each maps a feature row (Series) -> k
# ---------------------------------------------------------------------------
def pred_fixed(k: int):
    return lambda r, _k=k: _k


def pred_valley_rule(r) -> int:
    """Deep multi-core → short; shallow few-core → long."""
    vm = r.get("valley_median", np.nan)
    nd = r.get("n_density_pops", np.nan)
    fs = r.get("frac_shallow", np.nan)
    if np.isfinite(vm) and vm >= 0.6:
        return 500
    if np.isfinite(fs) and fs >= 0.8 and np.isfinite(nd) and nd <= 4:
        return 3000
    if np.isfinite(vm) and vm < 0.15 and np.isfinite(nd) and nd <= 5:
        return 3000
    if np.isfinite(nd) and nd >= 10:
        return 500
    return 2000


def pred_score_bins(r) -> int:
    s = r.get("shallow_minus_deep", 0)
    if s >= 2:
        return 3000
    if s <= -2:
        return 500
    if s == 1:
        return 3000
    if s == -1:
        return 1000
    return 2000


def pred_stab_valley(r) -> int:
    """Blend stab (if high) with valley."""
    vm = r.get("valley_median", np.nan)
    ms = r.get("mean_stability", np.nan)
    mu = r.get("min_stability", np.nan)
    # unstable splits → short
    if np.isfinite(mu) and mu < 0.3:
        return 500
    if np.isfinite(vm) and vm > 0.65:
        return 500
    if np.isfinite(vm) and vm < 0.1:
        return 3500 if (np.isfinite(ms) and ms > 0.7) else 3000
    if np.isfinite(ms) and ms > 0.85 and np.isfinite(vm) and vm < 0.4:
        return 3000
    return 2000


def pred_leiden_density(r) -> int:
    """Leiden much finer than density → over-split risk → short k."""
    ratio = r.get("leiden_over_density", np.nan)
    nd = r.get("n_density_pops", np.nan)
    if np.isfinite(ratio) and ratio >= 1.5 and np.isfinite(nd) and nd >= 6:
        return 500
    if np.isfinite(ratio) and ratio <= 1.2 and np.isfinite(nd) and nd <= 4:
        return 3000
    return 2000


def pred_linear_factory(coef: dict, intercept: float, grid=K_GRID):
    def _p(r):
        s = intercept
        for k, c in coef.items():
            v = r.get(k, np.nan)
            if not np.isfinite(v):
                return 2000
            s += c * float(v)
        return nearest_k(s, grid)

    return _p


def fit_loo_linear(g: pd.DataFrame, features: list[str], target="best_k"):
    """Return list of (dataset, pred_k) via LOO linear reg on target."""
    from numpy.linalg import lstsq

    y = g[target].to_numpy(dtype=float)
    X = g[features].to_numpy(dtype=float)
    names = g["dataset"].to_numpy()
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y, names = X[ok], y[ok], names[ok]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd
    # use log target for k
    y_fit = np.log10(np.clip(y, 200, 4000))
    out = {}
    for i in range(len(y)):
        m = np.ones(len(y), dtype=bool)
        m[i] = False
        A = np.column_stack([np.ones(m.sum()), Xs[m]])
        coef, _, _, _ = lstsq(A, y_fit[m], rcond=None)
        pred_log = coef[0] + Xs[i] @ coef[1:]
        out[names[i]] = nearest_k(10**pred_log)
    return out


def evaluate_predictor(name: str, pred_map: dict, curves: pd.DataFrame, best: dict) -> dict:
    rows = []
    for d, pk in pred_map.items():
        if d not in curves.index:
            continue
        curve = curves.loc[d]
        a_pred = ari_at(curve, int(pk))
        a_2000 = ari_at(curve, 2000)
        a_best = ari_at(curve, best[d])
        rows.append(
            {
                "predictor": name,
                "dataset": d,
                "pred_k": int(pk),
                "best_k": best[d],
                "ARI_pred": a_pred,
                "ARI_2000": a_2000,
                "ARI_best": a_best,
                "d_vs_2000": a_pred - a_2000,
                "gap_to_oracle": a_best - a_pred,
                "abs_err_k": abs(int(pk) - best[d]),
                "circular": d in SHADOWED,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"predictor": name, "n": 0}
    return {
        "predictor": name,
        "n": len(df),
        "mean_d_vs_2000": float(df["d_vs_2000"].mean()),
        "median_d_vs_2000": float(df["d_vs_2000"].median()),
        "mean_gap_to_oracle": float(df["gap_to_oracle"].mean()),
        "mean_abs_err_k": float(df["abs_err_k"].mean()),
        "frac_beat_2000": float((df["d_vs_2000"] > 0.005).mean()),
        "frac_lose_to_2000": float((df["d_vs_2000"] < -0.005).mean()),
        "spearman_k": float(stats.spearmanr(df["pred_k"], df["best_k"]).correlation or 0),
        "mean_ARI_pred": float(df["ARI_pred"].mean()),
        "mean_ARI_2000": float(df["ARI_2000"].mean()),
        "mean_ARI_best": float(df["ARI_best"].mean()),
    }, df


def main(seeds: list[int]):
    curves, best, complete = load_curves()
    print("complete datasets:", sorted(complete), flush=True)
    print("best_k:", best, flush=True)

    feat_rows = []
    for name in sorted(complete):
        if name not in LOADERS:
            print(f"no loader {name}", flush=True)
            continue
        print(f"\n######## {name} best_k={best[name]} ########", flush=True)
        a = LOADERS[name]()
        for seed in seeds:
            e = prepare_graph(a, seed)
            f = extract_features(e, seed)
            f.update(
                {
                    "dataset": name,
                    "seed": seed,
                    "best_k": best[name],
                    "log_best_k": float(np.log10(best[name])),
                    "circular": name in SHADOWED,
                    "ARI_2000": ari_at(curves.loc[name], 2000),
                    "ARI_best": ari_at(curves.loc[name], best[name]),
                }
            )
            feat_rows.append(f)
            print(
                f"  seed={seed} dens={f.get('n_density_pops')} "
                f"valley_med={f.get('valley_median'):.3f} "
                f"frac_shallow={f.get('frac_shallow'):.2f} "
                f"mean_stab={f.get('mean_stability'):.3f} "
                f"min_stab={f.get('min_stability'):.3f} "
                f"shallow-deep={f.get('shallow_minus_deep')}",
                flush=True,
            )
        del a

    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(FEAT, index=False)

    # mean over seeds
    num = [c for c in feat_df.columns if pd.api.types.is_numeric_dtype(feat_df[c])]
    g = feat_df.groupby("dataset", as_index=False)[num].mean()
    g["circular"] = g["dataset"].map(lambda d: d in SHADOWED)
    g["best_k"] = g["dataset"].map(best)

    # correlations
    corr_rows = []
    for c in num:
        if c in ("best_k", "log_best_k", "ARI_2000", "ARI_best", "seed", "n_obs", "bandwidth"):
            continue
        x = g[c].to_numpy(dtype=float)
        y = g["best_k"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5 or np.unique(x[m]).size < 2:
            continue
        sp = stats.spearmanr(x[m], y[m])
        corr_rows.append(
            {
                "feature": c,
                "spearman": float(sp.correlation) if sp.correlation is not None else np.nan,
                "p": float(sp.pvalue) if sp.pvalue is not None else np.nan,
            }
        )
    corr = pd.DataFrame(corr_rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(CORR, index=False)
    print("\n=== Spearman(feature, best_k) ===", flush=True)
    print(corr.head(12).round(3).to_string(index=False), flush=True)

    # residualize top features by log n_obs
    print("\n=== partial Spearman controlling log10_n_obs ===", flush=True)
    for c in corr.head(8)["feature"]:
        # residual y and x vs log n
        ln = g["log10_n_obs"].to_numpy(dtype=float)
        x = g[c].to_numpy(dtype=float)
        y = g["best_k"].to_numpy(dtype=float)
        m = np.isfinite(ln) & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            continue

        # simple residual
        def resid(a, b):
            A = np.column_stack([np.ones(b.size), b])
            coef, _, _, _ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ coef

        xr = resid(x[m], ln[m])
        yr = resid(y[m].astype(float), ln[m])
        sp = stats.spearmanr(xr, yr)
        print(f"  {c}: ρ_partial={sp.correlation:.3f} p={sp.pvalue:.3f}", flush=True)

    # predictors
    predictors = {
        "fixed_2000": pred_fixed(2000),
        "fixed_500": pred_fixed(500),
        "fixed_3000": pred_fixed(3000),
        "valley_rule": pred_valley_rule,
        "score_bins": pred_score_bins,
        "stab_valley": pred_stab_valley,
        "leiden_density": pred_leiden_density,
    }

    # LOO linear
    for feats, tag in [
        (["valley_median"], "loo_valley"),
        (["frac_shallow"], "loo_frac_shallow"),
        (["shallow_minus_deep"], "loo_shallow_deep"),
        (["valley_median", "n_density_pops"], "loo_valley_npop"),
        (["valley_median", "mean_stability"], "loo_valley_stab"),
        (["valley_median", "frac_shallow", "n_density_pops"], "loo_valley3"),
        (["score_coarse_shallow", "score_fine_deep"], "loo_scores"),
        (["min_stability", "valley_median"], "loo_minstab_valley"),
    ]:
        if all(f in g.columns for f in feats):
            predictors[tag] = None  # filled below

    eval_summaries = []
    eval_detail = []

    g_idx = g.set_index("dataset")

    def map_from_fn(fn):
        return {d: int(fn(g_idx.loc[d])) for d in g_idx.index}

    for name, fn in list(predictors.items()):
        if fn is None:
            continue
        pmap = map_from_fn(fn)
        summ, det = evaluate_predictor(name, pmap, curves, best)
        eval_summaries.append(summ)
        eval_detail.append(det)

    for feats, tag in [
        (["valley_median"], "loo_valley"),
        (["frac_shallow"], "loo_frac_shallow"),
        (["shallow_minus_deep"], "loo_shallow_deep"),
        (["valley_median", "n_density_pops"], "loo_valley_npop"),
        (["valley_median", "mean_stability"], "loo_valley_stab"),
        (["valley_median", "frac_shallow", "n_density_pops"], "loo_valley3"),
        (["score_coarse_shallow", "score_fine_deep"], "loo_scores"),
        (["min_stability", "valley_median"], "loo_minstab_valley"),
    ]:
        if not all(f in g.columns for f in feats):
            continue
        pmap = fit_loo_linear(g, feats)
        summ, det = evaluate_predictor(tag, pmap, curves, best)
        eval_summaries.append(summ)
        eval_detail.append(det)

    # oracle
    summ, det = evaluate_predictor("oracle_best_k", {d: best[d] for d in best}, curves, best)
    eval_summaries.append(summ)
    eval_detail.append(det)

    summary_df = pd.DataFrame(eval_summaries).sort_values("mean_d_vs_2000", ascending=False)
    detail_df = pd.concat(eval_detail, ignore_index=True)
    detail_df.to_csv(EVAL, index=False)

    print("\n=== predictor leaderboard (by mean ARI gain vs k=2000) ===", flush=True)
    show = summary_df[
        [
            "predictor",
            "n",
            "mean_d_vs_2000",
            "median_d_vs_2000",
            "mean_gap_to_oracle",
            "mean_abs_err_k",
            "frac_beat_2000",
            "frac_lose_to_2000",
            "spearman_k",
            "mean_ARI_pred",
        ]
    ]
    print(show.round(4).to_string(index=False), flush=True)

    # best non-oracle detail
    non_oracle = summary_df[summary_df["predictor"] != "oracle_best_k"]
    if len(non_oracle):
        top = non_oracle.iloc[0]["predictor"]
        print(f"\n=== detail: {top} ===", flush=True)
        print(
            detail_df[detail_df.predictor == top][
                [
                    "dataset",
                    "pred_k",
                    "best_k",
                    "ARI_pred",
                    "ARI_2000",
                    "d_vs_2000",
                    "gap_to_oracle",
                ]
            ]
            .round(4)
            .to_string(index=False),
            flush=True,
        )
        # independent only
        det_i = detail_df[(detail_df.predictor == top) & (~detail_df.circular.astype(bool))]
        if len(det_i):
            print(
                f"\nindependent only mean Δ vs 2000: {det_i['d_vs_2000'].mean():+.4f}",
                flush=True,
            )
        det_s = detail_df[(detail_df.predictor == top) & (detail_df.circular.astype(bool))]
        if len(det_s):
            print(
                f"SHADOWED only mean Δ vs 2000: {det_s['d_vs_2000'].mean():+.4f}",
                flush=True,
            )

    # per-dataset feature table for discussion
    print("\n=== feature snapshot ===", flush=True)
    cols = [
        "dataset",
        "best_k",
        "ARI_2000",
        "ARI_best",
        "n_density_pops",
        "valley_median",
        "frac_shallow",
        "mean_stability",
        "min_stability",
        "n_leiden",
        "shallow_minus_deep",
        "leiden_over_density",
    ]
    cols = [c for c in cols if c in g.columns]
    print(g[cols].round(3).to_string(index=False), flush=True)

    print(f"\nwrote {FEAT}\nwrote {EVAL}\nwrote {CORR}", flush=True)

    # save summary leaderboard too
    summary_df.to_csv(OUT / "auto_n_structure_leaderboard.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    args = ap.parse_args()
    main(args.seeds)
