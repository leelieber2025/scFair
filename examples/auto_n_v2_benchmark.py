#!/usr/bin/env python
"""Compare auto-n ensemble v2 vs HVG@2000 / hybrid@2000 on multi + rare.

Writes examples/results/auto_n_v2_*.csv|json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import scfair as scf

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
RARE_FRACS = [0.01, 0.02, 0.05]
METHODS = ["hvg", "scfair_hybrid", "scfair_auto"]


def load_cao() -> ad.AnnData:
    path = DATA / "Cao.h5"
    with h5py.File(path, "r") as f:
        X = np.array(f["X"]).astype(np.float32)
        y = np.array(f["Y"]).astype(int).ravel()
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(a.n_vars)]
    a.obs["cell_type"] = pd.Categorical(y.astype(str))
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.obs_names_make_unique()
    return a


def load_paul15() -> ad.AnnData:
    a = sc.datasets.paul15()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    a.obs["cell_type"] = a.obs["paul15_clusters"].astype(str)
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    return a


def load_pbmc3k_labeled() -> ad.AnnData:
    raw = sc.datasets.pbmc3k()
    raw.var_names_make_unique()
    raw.obs_names_make_unique()
    proc = sc.datasets.pbmc3k_processed()
    common = raw.obs_names.intersection(proc.obs_names)
    a = raw[common].copy()
    a.obs["cell_type"] = proc.obs.loc[common, "louvain"].astype(str).values
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs.n_genes_by_counts < 2500, :].copy()
    a = a[a.obs.pct_counts_mt < 5, :].copy()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    return a


def make_rare_mix(
    adata: ad.AnnData,
    *,
    rare_frac: float,
    rare_type: str,
    random_state: int = 0,
    max_majority: int = 3000,
) -> ad.AnnData:
    rng = np.random.default_rng(random_state)
    labels = adata.obs["cell_type"].astype(str)
    maj_idx = np.where((labels != rare_type).to_numpy())[0]
    rare_idx = np.where((labels == rare_type).to_numpy())[0]
    n_maj = min(len(maj_idx), max_majority)
    maj_sel = rng.choice(maj_idx, size=n_maj, replace=False)
    n_rare_target = max(5, int(round(rare_frac * n_maj / max(1e-9, 1.0 - rare_frac))))
    n_rare_target = min(n_rare_target, len(rare_idx))
    rare_sel = rng.choice(rare_idx, size=n_rare_target, replace=False)
    sel = np.concatenate([maj_sel, rare_sel])
    rng.shuffle(sel)
    out = adata[sel].copy()
    out.obs["is_rare"] = (out.obs["cell_type"].astype(str) == rare_type).astype(int)
    out.uns["rare_frac_actual"] = float(out.obs["is_rare"].mean())
    return out


def select_genes(adata: ad.AnnData, method: str, seed: int = 0) -> tuple[list[str], dict]:
    a = adata.copy()
    meta: dict = {}
    if method == "hvg":
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
        )
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    elif method == "scfair_hybrid":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=min(2000, a.n_vars - 1),
            flavor="seurat_v3",
            layer="counts",
            balance_method="hybrid",
            blend_global=0.95,
            marker_mode="none",
            random_state=seed,
        )
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
    elif method == "scfair_auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            n_top_min=500,
            n_top_max=min(5000, a.n_vars - 1),
            flavor="seurat_v3",
            layer="counts",
            balance_method="hybrid",
            blend_global=0.95,
            marker_mode="none",
            random_state=seed,
        )
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
    else:
        raise ValueError(method)
    return genes, meta


def cluster_metrics(
    adata: ad.AnnData, genes: list[str], *, seed: int = 0, label_key: str = "cell_type"
) -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    if len(genes) < 10:
        return {"n_genes": len(genes), "ARI": np.nan, "NMI": np.nan}
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a,
        resolution=0.8,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    y_true = a.obs[label_key].astype(str)
    y_pred = a.obs["leiden"].astype(str)
    X = a.obsm["X_pca"][:, : min(20, n_comps)]
    out = {
        "n_genes": len(genes),
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "sil_true": float(
            silhouette_score(X, y_true, sample_size=min(2000, a.n_obs), random_state=seed)
        ),
    }
    if "is_rare" in a.obs.columns:
        rare = a.obs["is_rare"].astype(bool)
        n_rare = int(rare.sum())
        pure = 0
        best_f1 = 0.0
        for cl in y_pred.unique():
            m = y_pred == cl
            frac = float(rare[m].mean()) if m.sum() else 0.0
            if frac >= 0.5:
                pure += int((rare & m).sum())
            tp = int((rare & m).sum())
            prec = tp / int(m.sum()) if m.sum() else 0.0
            rec = tp / n_rare if n_rare else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            best_f1 = max(best_f1, f1)
        out["rare_n"] = n_rare
        out["rare_recall_pure"] = pure / n_rare if n_rare else np.nan
        out["rare_best_cluster_f1"] = best_f1
    return out


def pbmc_module_metrics(adata: ad.AnnData, genes: list[str], seed: int = 0) -> dict:
    """Rough DC / NK purity proxies using processed louvain labels if present."""
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a, resolution=0.8, flavor="igraph", n_iterations=2, random_state=seed, key_added="leiden"
    )
    labels = a.obs["cell_type"].astype(str)
    pred = a.obs["leiden"].astype(str)

    def pure_frac(name_substr: str) -> float:
        mask = labels.str.contains(name_substr, case=False, regex=False)
        if mask.sum() == 0:
            return float("nan")
        best = 0.0
        for cl in pred.unique():
            m = pred == cl
            if m.sum() == 0:
                continue
            frac = float((mask & m).sum() / mask.sum())
            purity = float(mask[m].mean())
            if purity >= 0.5:
                best = max(best, frac)
        return best

    return {
        "DC_pure_frac": pure_frac("DC"),
        "NK_pure_frac": pure_frac("NK"),
        "ARI": float(adjusted_rand_score(labels, pred)),
        "NMI": float(normalized_mutual_info_score(labels, pred)),
    }


def main():
    print("=== auto-n v2 multi-dataset ===", flush=True)
    loaders = {
        "Cao": load_cao,
        "paul15": load_paul15,
        "pbmc3k_louvain": load_pbmc3k_labeled,
    }
    multi_rows = []
    for dname, loader in loaders.items():
        print(f"\n==== {dname} ====", flush=True)
        adata = loader()
        print(f"  {adata.n_obs}×{adata.n_vars}", flush=True)
        for method in METHODS:
            genes, meta = select_genes(adata, method, seed=0)
            scores = cluster_metrics(adata, genes, seed=0)
            auto = meta.get("auto_n") or {}
            ens = auto.get("ensemble") if isinstance(auto, dict) else None
            row = {
                "dataset": dname,
                "method": method,
                "n_genes": scores["n_genes"],
                "ARI": scores["ARI"],
                "NMI": scores["NMI"],
                "sil_true": scores["sil_true"],
                "n_top_used": meta.get("n_top_genes_used"),
                "auto_strategy": auto.get("strategy") if isinstance(auto, dict) else None,
                "k_floor": ens.get("k_floor") if isinstance(ens, dict) else None,
                "ensemble_votes": json.dumps(ens.get("votes")) if isinstance(ens, dict) else None,
                "ensemble_notes": json.dumps(ens.get("notes")) if isinstance(ens, dict) else None,
                "method_picks": json.dumps(auto.get("method_picks"))
                if isinstance(auto, dict)
                else None,
            }
            if dname == "pbmc3k_louvain":
                mod = pbmc_module_metrics(adata, genes, seed=0)
                row.update(mod)
            multi_rows.append(row)
            print(
                f"  {method}: ARI={scores['ARI']:.3f} n={scores['n_genes']} "
                f"k_used={row['n_top_used']} votes={row['ensemble_votes']}",
                flush=True,
            )

    multi = pd.DataFrame(multi_rows)
    multi.to_csv(OUT / "auto_n_v2_multi.csv", index=False)

    print("\n=== auto-n v2 Cao rare multi-seed ===", flush=True)
    base = load_cao()
    vc = base.obs["cell_type"].astype(str).value_counts()
    rare_type = None
    for t in vc.index[1:]:
        if vc[t] >= 50:
            rare_type = str(t)
            break
    rare_type = rare_type or str(vc.index[-1])
    print(f"rare_type={rare_type} n={vc[rare_type]}", flush=True)

    rare_rows = []
    for frac in RARE_FRACS:
        for seed in SEEDS:
            mix = make_rare_mix(base, rare_frac=frac, rare_type=rare_type, random_state=seed)
            for method in METHODS:
                genes, meta = select_genes(mix, method, seed=seed)
                scores = cluster_metrics(mix, genes, seed=seed)
                auto = meta.get("auto_n") or {}
                ens = auto.get("ensemble") if isinstance(auto, dict) else None
                rare_rows.append(
                    {
                        "rare_frac_target": frac,
                        "seed": seed,
                        "method": method,
                        "n_genes": scores["n_genes"],
                        "ARI": scores["ARI"],
                        "NMI": scores["NMI"],
                        "rare_recall_pure": scores.get("rare_recall_pure"),
                        "rare_best_cluster_f1": scores.get("rare_best_cluster_f1"),
                        "n_top_used": meta.get("n_top_genes_used"),
                        "k_floor": ens.get("k_floor") if isinstance(ens, dict) else None,
                        "ensemble_votes": json.dumps(ens.get("votes"))
                        if isinstance(ens, dict)
                        else None,
                    }
                )
        print(f"  frac={frac} done", flush=True)

    rare = pd.DataFrame(rare_rows)
    rare.to_csv(OUT / "auto_n_v2_rare.csv", index=False)
    rare_sum = rare.groupby(["rare_frac_target", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        rare_recall_mean=("rare_recall_pure", "mean"),
        rare_recall_std=("rare_recall_pure", "std"),
        rare_recall_gt0=("rare_recall_pure", lambda s: float((s > 0).mean())),
        k_mean=("n_genes", "mean"),
    )
    rare_sum.to_csv(OUT / "auto_n_v2_rare_summary.csv", index=False)

    print("\n======== MULTI ARI ========")
    print(multi.pivot_table(index="dataset", columns="method", values="ARI").round(3))
    print("\n======== MULTI n_genes ========")
    print(multi.pivot_table(index="dataset", columns="method", values="n_genes").round(0))
    print("\n======== RARE SUMMARY ========")
    print(rare_sum.round(3).to_string(index=False))

    # Compare to P2 baseline if present
    p2 = OUT / "p2_multi_dataset.csv"
    delta = {}
    if p2.exists():
        old = pd.read_csv(p2)
        for d in multi["dataset"].unique():
            for m in ("scfair_auto",):
                o = old[(old.dataset == d) & (old.method == m)]
                n = multi[(multi.dataset == d) & (multi.method == m)]
                if len(o) and len(n):
                    delta[f"{d}_{m}_ARI"] = {
                        "p2_old": float(o.ARI.iloc[0]),
                        "v2_new": float(n.ARI.iloc[0]),
                        "delta": float(n.ARI.iloc[0] - o.ARI.iloc[0]),
                        "p2_k": float(o.n_genes.iloc[0]) if "n_genes" in o else None,
                        "v2_k": float(n.n_genes.iloc[0]),
                    }

    summary = {
        "multi": multi.replace({np.nan: None}).to_dict(orient="records"),
        "rare_summary": rare_sum.replace({np.nan: None}).to_dict(orient="records"),
        "vs_p2_auto": delta,
        "note": "ensemble v2: single shape vote, k_floor, shape-vs-mass guard",
    }
    with open(OUT / "auto_n_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nvs P2 auto:", json.dumps(delta, indent=2))
    print(f"Wrote {OUT / 'auto_n_v2_multi.csv'}")
    print(f"Wrote {OUT / 'auto_n_v2_rare_summary.csv'}")
    print("DONE")


if __name__ == "__main__":
    main()
