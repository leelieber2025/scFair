#!/usr/bin/env python
"""Compare automatic n_top strategies on PBMC3k embedding quality.

Usage:
    python examples/auto_n_benchmark.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import scfair as scf
from scfair.pp._auto_n import (
    select_n_top_coverage,
    select_n_top_cumfrac,
    select_n_top_elbow,
    select_n_top_knee,
    select_n_top_silhouette,
)
from scfair.pp._highly_variable_genes import (
    _cluster_vs_rest_logfc,
    _lognorm_matrix_from_counts,
)

sc.settings.verbosity = 0
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
RS = 0
K_MIN, K_MAX = 500, 5000


def load_pbmc():
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    adata.layers["counts"] = adata.X.copy()
    return adata


def lineage_proxy(adata):
    ad_log = adata.copy()
    ad_log.X = ad_log.layers["counts"].copy()
    sc.pp.normalize_total(ad_log, target_sum=1e4)
    sc.pp.log1p(ad_log)
    LIN = {
        "T": ["CD3D", "CD3E", "IL7R", "CD8A", "CD8B"],
        "NK": ["NKG7", "GNLY", "KLRD1"],
        "B": ["MS4A1", "CD79A", "CD79B"],
        "Mono": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A"],
        "DC": ["FCER1A", "CST3"],
        "Platelet": ["PPBP"],
    }

    def mod(genes):
        p = [g for g in genes if g in ad_log.var_names]
        if not p:
            return np.zeros(ad_log.n_obs)
        X = ad_log[:, p].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, float).mean(1)

    for lin, gs in LIN.items():
        ad_log.obs[f"mod_{lin}"] = mod(gs)
    M = np.column_stack([ad_log.obs[f"mod_{lin}"].to_numpy() for lin in LIN])
    ad_log.obs["lineage_proxy"] = [list(LIN)[i] for i in M.argmax(1)]
    return ad_log, LIN


def build_rankings(adata):
    """Global seurat_v3 scores + hybrid-like score for top K_MAX genes."""
    # --- global ranking (seurat_v3) ---
    ad = adata.copy()
    n_cap = min(K_MAX, ad.n_vars)
    sc.pp.highly_variable_genes(
        ad, n_top_genes=n_cap, flavor="seurat_v3", layer="counts", subset=False
    )
    gscore = ad.var["variances_norm"].astype(float).fillna(0.0)
    global_order = list(gscore.sort_values(ascending=False).index.astype(str))
    global_scores_desc = gscore.loc[global_order].to_numpy()

    # --- hybrid ranking via scfair (n_top=K_MAX pool) ---
    ad_h = adata.copy()
    scf.pp.highly_variable_genes(
        ad_h,
        n_top_genes=n_cap,
        flavor="seurat_v3",
        layer="counts",
        balance_method="hybrid",
        blend_global=0.95,
        marker_mode="none",
        min_cluster_size=30,
        random_state=RS,
        balance_power=0.5,
    )
    # use scfair_score for ordering of all genes; prefer selected first
    hscore = ad_h.var["scfair_score"].astype(float).fillna(0.0)
    hybrid_order = list(hscore.sort_values(ascending=False).index.astype(str))
    hybrid_scores_desc = hscore.loc[hybrid_order].to_numpy()

    # cluster gene ranks for coverage (from hybrid intermediate clusters if present)
    cluster_gene_ranks = {}
    if "scfair_hvg_clusters" in ad_h.obs.columns:
        labels = ad_h.obs["scfair_hvg_clusters"].astype(str)
        sizes = labels.value_counts()
        valid = sizes[sizes >= 30]
        X_log = _lognorm_matrix_from_counts(ad_h, "counts")
        for cl in valid.index:
            mask = (labels == cl).to_numpy()
            if mask.sum() < 30 or mask.sum() >= ad_h.n_obs:
                continue
            logfc = _cluster_vs_rest_logfc(X_log, mask, one_sided=True)
            order = np.argsort(-logfc)
            cluster_gene_ranks[str(cl)] = list(ad_h.var_names[order].astype(str))

    return {
        "global_order": global_order,
        "global_scores_desc": global_scores_desc,
        "hybrid_order": hybrid_order,
        "hybrid_scores_desc": hybrid_scores_desc,
        "cluster_gene_ranks": cluster_gene_ranks,
        "adata_for_sil": adata,
    }


def embed_metrics(adata, genes, ad_log, LIN):
    ad = adata.copy()
    ad.X = ad.layers["counts"].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    genes = [g for g in genes if g in ad.var_names]
    ad = ad[:, genes].copy()
    sc.pp.scale(ad, max_value=10)
    n_comps = min(40, ad.n_vars - 1, ad.n_obs - 1)
    sc.tl.pca(ad, n_comps=n_comps, svd_solver="arpack", random_state=RS)
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=min(30, n_comps), random_state=RS)
    sc.tl.leiden(
        ad,
        resolution=0.8,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=RS,
    )
    ad.obs["lineage_proxy"] = ad_log.obs["lineage_proxy"].values
    for lin in LIN:
        ad.obs[f"mod_{lin}"] = ad_log.obs[f"mod_{lin}"].values

    X = ad.obsm["X_pca"][:, : min(20, n_comps)]
    lab = ad.obs["lineage_proxy"].astype(str)
    leid = ad.obs["leiden"].astype(str)
    out = {
        "n_genes": len(genes),
        "sil_lin": float(
            silhouette_score(X, lab, sample_size=min(2000, ad.n_obs), random_state=RS)
        ),
        "sil_leid": float(
            silhouette_score(X, leid, sample_size=min(2000, ad.n_obs), random_state=RS)
        ),
        "ARI": float(adjusted_rand_score(lab, leid)),
        "NMI": float(normalized_mutual_info_score(lab, leid)),
    }
    pur, gaps = [], []
    for cl in ad.obs["leiden"].unique():
        sub = ad.obs.loc[ad.obs["leiden"] == cl, "lineage_proxy"]
        vc = sub.value_counts(normalize=True)
        pur.append(float(vc.iloc[0]))
        cells = ad.obs_names[ad.obs["leiden"] == cl]
        scs = {lin: float(ad.obs.loc[cells, f"mod_{lin}"].mean()) for lin in LIN}
        vals = sorted(scs.values(), reverse=True)
        gaps.append(vals[0] - vals[1] if len(vals) > 1 else vals[0])
    out["purity"] = float(np.mean(pur))
    out["gap"] = float(np.mean(gaps))
    for r in ("NK", "DC", "Platelet", "T"):
        mask = ad.obs["lineage_proxy"] == r
        n = int(mask.sum())
        pure = 0
        for cl in ad.obs["leiden"].unique():
            sub = ad.obs.loc[ad.obs["leiden"] == cl, "lineage_proxy"]
            vc = sub.value_counts(normalize=True)
            if len(vc) and vc.index[0] == r and vc.iloc[0] >= 0.5:
                pure += int(((ad.obs["leiden"] == cl) & mask).sum())
        out[f"{r}_pure"] = pure / n if n else np.nan
    return out


def main():
    print("=== Load PBMC3k ===")
    adata = load_pbmc()
    print(adata.shape)
    ad_log, LIN = lineage_proxy(adata)

    print("=== Build rankings ===")
    R = build_rankings(adata)

    # Strategies to evaluate (on hybrid ranking — scFair path)
    order = R["hybrid_order"]
    scores = R["hybrid_scores_desc"]
    cgr = R["cluster_gene_ranks"]

    strategies = {}
    strategies["fixed_1000"] = 1000
    strategies["fixed_2000"] = 2000
    strategies["fixed_3000"] = 3000
    strategies["elbow"] = select_n_top_elbow(scores, k_min=K_MIN, k_max=K_MAX)
    strategies["knee"] = select_n_top_knee(scores, k_min=K_MIN, k_max=K_MAX)
    strategies["cumfrac_0.7"] = select_n_top_cumfrac(scores, frac=0.7, k_min=K_MIN, k_max=K_MAX)
    strategies["cumfrac_0.8"] = select_n_top_cumfrac(scores, frac=0.8, k_min=K_MIN, k_max=K_MAX)
    strategies["cumfrac_0.9"] = select_n_top_cumfrac(scores, frac=0.9, k_min=K_MIN, k_max=K_MAX)
    if cgr:
        strategies["coverage_m15"] = select_n_top_coverage(
            order, cgr, min_per_cluster=15, k_min=K_MIN, k_max=K_MAX
        )
        strategies["coverage_m20"] = select_n_top_coverage(
            order, cgr, min_per_cluster=20, k_min=K_MIN, k_max=K_MAX
        )
        strategies["coverage_m30"] = select_n_top_coverage(
            order, cgr, min_per_cluster=30, k_min=K_MIN, k_max=K_MAX
        )

    print("Computing silhouette grid (slow)...")
    k_sil, sil_curve = select_n_top_silhouette(
        adata,
        order,
        candidates=[500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000],
        k_min=K_MIN,
        k_max=K_MAX,
        counts_layer="counts",
        random_state=RS,
    )
    strategies["silhouette"] = k_sil
    print("silhouette picks", k_sil, "curve", {k: round(v, 4) for k, v in sil_curve.items()})

    # ensemble of cheap methods
    cheap = {m: strategies[m] for m in ("elbow", "knee", "cumfrac_0.8") if m in strategies}
    if "coverage_m20" in strategies:
        cheap["coverage_m20"] = strategies["coverage_m20"]
    strategies["ensemble_cheap"] = int(np.median(list(cheap.values())))
    # ensemble + silhouette
    with_sil = dict(cheap)
    with_sil["silhouette"] = k_sil
    strategies["ensemble_full"] = int(np.median(list(with_sil.values())))

    print("\n=== k picks ===")
    for name, k in sorted(strategies.items(), key=lambda x: x[1]):
        print(f"  {name:18s} k={k}")

    # Also global-ranking fixed for reference
    rows = []
    for name, k in strategies.items():
        print(f"embed {name} k={k}...", flush=True)
        genes = order[:k]
        m = embed_metrics(adata, genes, ad_log, LIN)
        m["method"] = name
        m["k"] = k
        m["ranking"] = "hybrid_score"
        rows.append(m)

    # global fixed 2000 baseline
    print("embed global_fixed_2000...", flush=True)
    m = embed_metrics(adata, R["global_order"][:2000], ad_log, LIN)
    m["method"] = "global_fixed_2000"
    m["k"] = 2000
    m["ranking"] = "seurat_v3"
    rows.append(m)

    df = pd.DataFrame(rows).set_index("method")
    metric_cols = [
        "k",
        "sil_lin",
        "sil_leid",
        "ARI",
        "NMI",
        "purity",
        "gap",
        "NK_pure",
        "DC_pure",
        "Platelet_pure",
        "T_pure",
    ]
    print("\n=== metrics (hybrid ranking) ===")
    print(df[metric_cols].round(4).to_string())

    # rank methods: higher better for all except we want balanced
    # score = sum of ranks (1=best) inverted
    higher_better = [
        "sil_lin",
        "sil_leid",
        "ARI",
        "NMI",
        "purity",
        "gap",
        "NK_pure",
        "DC_pure",
        "Platelet_pure",
        "T_pure",
    ]
    ranks = pd.DataFrame(index=df.index)
    for c in higher_better:
        ranks[c] = df[c].rank(ascending=False, method="average")
    df["rank_sum"] = ranks.sum(axis=1)
    df["rank_mean"] = ranks.mean(axis=1)
    # penalize extreme k? optional soft: prefer mid-range
    print("\n=== mean rank (lower better) ===")
    print(
        df[["k", "rank_mean", "rank_sum"] + higher_better]
        .sort_values("rank_mean")
        .round(4)
        .to_string()
    )

    best = df["rank_mean"].idxmin()
    print(f"\n*** BEST by mean rank: {best} (k={df.loc[best, 'k']}) ***")

    # vs fixed_2000
    print("\n=== delta vs hybrid fixed_2000 ===")
    if "fixed_2000" in df.index:
        base = df.loc["fixed_2000"]
        for name in df.index:
            if name == "fixed_2000":
                continue
            d = {c: df.loc[name, c] - base[c] for c in higher_better}
            nb = sum(v > 1e-6 for v in d.values())
            nw = sum(v < -1e-6 for v in d.values())
            print(
                f"{name:18s} k={int(df.loc[name, 'k']):4d} +{nb}/-{nw}  "
                + " ".join(f"{c}:{d[c]:+.4f}" for c in higher_better)
            )

    df.to_csv(OUT / "pbmc3k_auto_n_benchmark.csv")
    pd.Series(sil_curve).to_csv(OUT / "pbmc3k_auto_n_silhouette_curve.csv")
    print(f"\nWrote {OUT / 'pbmc3k_auto_n_benchmark.csv'}")
    print("DONE")
    return best, df


if __name__ == "__main__":
    main()
