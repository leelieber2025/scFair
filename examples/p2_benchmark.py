#!/usr/bin/env python
"""P2 benchmarks: multi-seed rare stability + multi-dataset ARI/NMI expansion.

Uses publicly available labeled data (Cao, paul15, pbmc3k+louvain).
Rare-cell protocol: CellBRF-style minority mixing on Cao and paul15
across multiple random seeds.

Outputs under examples/results/:
  p2_multi_dataset.csv
  p2_rare_multiseed.csv
  p2_summary.json
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
RARE_FRACS = [0.005, 0.01, 0.02, 0.05]
METHODS = ["hvg", "scfair_hybrid", "scfair_auto"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_cao() -> ad.AnnData:
    path = DATA / "Cao.h5"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/xuyp-csu/CellBRF/main/h5data/Cao.h5",
            path,
        )
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
    rare_type: str | None = None,
    random_state: int = 0,
    max_majority: int = 3000,
) -> ad.AnnData:
    rng = np.random.default_rng(random_state)
    labels = adata.obs["cell_type"].astype(str)
    vc = labels.value_counts()
    if rare_type is None:
        rare_type = None
        for t in vc.index[1:]:
            if vc[t] >= 50:
                rare_type = str(t)
                break
        if rare_type is None:
            rare_type = str(vc.index[-1])
    rare_type = str(rare_type)
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
    out.obs["rare_type"] = rare_type
    out.uns["rare_frac_target"] = rare_frac
    out.uns["rare_frac_actual"] = float(out.obs["is_rare"].mean())
    return out


# ---------------------------------------------------------------------------
# Select + cluster
# ---------------------------------------------------------------------------


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
        out["rare_frac"] = float(rare.mean())
        out["rare_recall_pure"] = pure / n_rare if n_rare else np.nan
        out["rare_best_cluster_f1"] = best_f1
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_multi_dataset() -> pd.DataFrame:
    loaders = {
        "Cao": load_cao,
        "paul15": load_paul15,
        "pbmc3k_louvain": load_pbmc3k_labeled,
    }
    rows = []
    for dname, loader in loaders.items():
        print(f"\n==== Dataset {dname} ====", flush=True)
        adata = loader()
        print(f"  {adata.n_obs}×{adata.n_vars} types={adata.obs['cell_type'].nunique()}")
        for method in METHODS:
            print(f"  {method}", flush=True)
            genes, meta = select_genes(adata, method, seed=0)
            scores = cluster_metrics(adata, genes, seed=0)
            scores.update(
                {
                    "dataset": dname,
                    "method": method,
                    "n_types": int(adata.obs["cell_type"].nunique()),
                    "n_top_used": meta.get("n_top_genes_used"),
                }
            )
            rows.append(scores)
            print(f"    ARI={scores['ARI']:.3f} NMI={scores['NMI']:.3f} n={scores['n_genes']}")
    return pd.DataFrame(rows)


def run_rare_multiseed() -> pd.DataFrame:
    """Rare mixes on Cao + paul15, multiple seeds."""
    bases = {
        "Cao": load_cao(),
        "paul15": load_paul15(),
    }
    rows = []
    for dname, base in bases.items():
        vc = base.obs["cell_type"].astype(str).value_counts()
        rare_type = None
        for t in vc.index[1:]:
            if vc[t] >= 50:
                rare_type = str(t)
                break
        if rare_type is None:
            rare_type = str(vc.index[-1])
        print(f"\n==== Rare multiseed {dname} rare_type={rare_type} (n={vc[rare_type]}) ====")
        for frac in RARE_FRACS:
            for seed in SEEDS:
                mix = make_rare_mix(base, rare_frac=frac, rare_type=rare_type, random_state=seed)
                for method in METHODS:
                    try:
                        genes, meta = select_genes(mix, method, seed=seed)
                        scores = cluster_metrics(mix, genes, seed=seed)
                        scores.update(
                            {
                                "source": dname,
                                "rare_type": rare_type,
                                "rare_frac_target": frac,
                                "rare_frac_actual": mix.uns["rare_frac_actual"],
                                "seed": seed,
                                "method": method,
                                "n_top_used": meta.get("n_top_genes_used"),
                                "n_cells": int(mix.n_obs),
                            }
                        )
                        rows.append(scores)
                    except Exception as e:
                        rows.append(
                            {
                                "source": dname,
                                "rare_frac_target": frac,
                                "seed": seed,
                                "method": method,
                                "error": str(e),
                            }
                        )
                # progress line per frac/seed once
            # aggregate progress
            sub = [
                r for r in rows if r.get("source") == dname and r.get("rare_frac_target") == frac
            ]
            if sub:
                print(
                    f"  frac={frac}: seeds done={len({r.get('seed') for r in sub})} "
                    f"rows={len(sub)}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize_rare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "rare_recall_pure" not in df.columns:
        return df
    g = (
        df.dropna(subset=["ARI"])
        .groupby(["source", "rare_frac_target", "method"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            ARI_mean=("ARI", "mean"),
            ARI_std=("ARI", "std"),
            NMI_mean=("NMI", "mean"),
            rare_recall_mean=("rare_recall_pure", "mean"),
            rare_recall_std=("rare_recall_pure", "std"),
            rare_recall_gt0=("rare_recall_pure", lambda s: float((s > 0).mean())),
            rare_f1_mean=("rare_best_cluster_f1", "mean"),
            rare_f1_std=("rare_best_cluster_f1", "std"),
        )
    )
    return g


def main():
    print("=== P2 multi-dataset ===", flush=True)
    multi = run_multi_dataset()
    multi.to_csv(OUT / "p2_multi_dataset.csv", index=False)

    print("\n=== P2 rare multi-seed (this takes a while) ===", flush=True)
    rare = run_rare_multiseed()
    rare.to_csv(OUT / "p2_rare_multiseed.csv", index=False)
    rare_sum = summarize_rare(rare)
    rare_sum.to_csv(OUT / "p2_rare_multiseed_summary.csv", index=False)

    print("\n======== MULTI DATASET ARI ========")
    print(multi.pivot_table(index="dataset", columns="method", values="ARI").round(3).to_string())
    print("\n======== MULTI DATASET NMI ========")
    print(multi.pivot_table(index="dataset", columns="method", values="NMI").round(3).to_string())

    print("\n======== RARE SUMMARY (mean±std over seeds) ========")
    if not rare_sum.empty:
        for src in rare_sum["source"].unique():
            print(f"\n--- {src} rare_recall_mean ---")
            sub = rare_sum[rare_sum["source"] == src]
            print(
                sub.pivot_table(
                    index="rare_frac_target", columns="method", values="rare_recall_mean"
                )
                .round(3)
                .to_string()
            )
            print(f"--- {src} rare_recall P(>0) ---")
            print(
                sub.pivot_table(
                    index="rare_frac_target", columns="method", values="rare_recall_gt0"
                )
                .round(2)
                .to_string()
            )
            print(f"--- {src} ARI_mean ---")
            print(
                sub.pivot_table(index="rare_frac_target", columns="method", values="ARI_mean")
                .round(3)
                .to_string()
            )

    # wins count: hybrid vs hvg on multi
    wins = {"hybrid_ari": 0, "hvg_ari": 0, "tie_ari": 0}
    for d in multi["dataset"].unique():
        sub = multi[multi["dataset"] == d].set_index("method")
        if "hvg" in sub.index and "scfair_hybrid" in sub.index:
            ha, hh = sub.loc["hvg", "ARI"], sub.loc["scfair_hybrid", "ARI"]
            if hh > ha + 1e-6:
                wins["hybrid_ari"] += 1
            elif ha > hh + 1e-6:
                wins["hvg_ari"] += 1
            else:
                wins["tie_ari"] += 1

    summary = {
        "multi_dataset_records": multi.replace({np.nan: None}).to_dict(orient="records"),
        "rare_summary_records": rare_sum.replace({np.nan: None}).to_dict(orient="records"),
        "wins_multi_ari": wins,
        "seeds": SEEDS,
        "rare_fracs": RARE_FRACS,
        "note": "Jurkat/293T public 10x matrices not downloaded (CDN 403); rare mixes simulated from Cao/paul15.",
    }
    with open(OUT / "p2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nMulti ARI wins hybrid vs hvg: {wins}")
    print(f"Wrote {OUT / 'p2_multi_dataset.csv'}")
    print(f"Wrote {OUT / 'p2_rare_multiseed.csv'}")
    print(f"Wrote {OUT / 'p2_rare_multiseed_summary.csv'}")
    print(f"Wrote {OUT / 'p2_summary.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
