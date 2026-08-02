#!/usr/bin/env python
"""P1 benchmarks for scFair (CellBRF-style + rare-cell protocol).

1) Multi-dataset labeled ARI/NMI
   - Cao.h5 (CellBRF)
   - paul15 (scanpy, hematopoietic clusters)
   - pbmc3k raw counts + louvain labels from pbmc3k_processed

2) Rare-cell mixes simulated from Cao
   - Hold out one minority type at target fractions of the final mix
   - Metrics: overall ARI/NMI + rare-type recovery (in pure rare clusters)

Usage
-----
  python examples/p1_multi_dataset_benchmark.py
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
RS = 0
RARE_FRACS = [0.005, 0.01, 0.02, 0.05]


# ---------------------------------------------------------------------------
# Data loaders
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
    # integer-like counts already
    a.layers["counts"] = a.X.copy() if not hasattr(a.X, "toarray") else a.X.copy()
    if hasattr(a.layers["counts"], "toarray"):
        a.layers["counts"] = a.layers["counts"].copy()
    sc.pp.filter_genes(a, min_cells=3)
    return a


def load_pbmc3k_labeled() -> ad.AnnData:
    """Raw counts with louvain labels transferred from processed object."""
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
) -> ad.AnnData:
    """Build a mix where one cell type is approximately ``rare_frac`` of cells.

    Keeps all non-rare cells as the majority background (subsampled if needed
    only to control size), and samples the rare type so that
    n_rare / (n_rare + n_maj) ≈ rare_frac.
    """
    rng = np.random.default_rng(random_state)
    labels = adata.obs["cell_type"].astype(str)
    vc = labels.value_counts()
    if rare_type is None:
        # prefer a mid-size type (not the largest) with enough cells
        ordered = vc.sort_values(ascending=False)
        candidates = ordered.index[1:] if len(ordered) > 1 else ordered.index
        rare_type = str(candidates[0])
        for t in candidates:
            if vc[t] >= 30:
                rare_type = str(t)
                break
    rare_type = str(rare_type)
    maj_mask = labels != rare_type
    rare_mask = labels == rare_type
    maj_idx = np.where(maj_mask.to_numpy())[0]
    rare_idx = np.where(rare_mask.to_numpy())[0]
    if len(rare_idx) < 5 or len(maj_idx) < 20:
        raise ValueError(f"Not enough cells for rare mix: rare={rare_type}")

    # n_rare / (n_rare + n_maj) = rare_frac  →  n_rare = rare_frac/(1-f) * n_maj
    # use all majority (or cap for speed)
    n_maj = min(len(maj_idx), 3000)
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
# Feature selection + clustering
# ---------------------------------------------------------------------------


def select_genes(adata: ad.AnnData, method: str) -> tuple[list[str], dict]:
    a = adata.copy()
    meta: dict = {}
    if method == "hvg":
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3", layer="counts")
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    elif method == "scfair_hybrid":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=2000,
            flavor="seurat_v3",
            layer="counts",
            balance_method="hybrid",
            blend_global=0.95,
            marker_mode="none",
            random_state=RS,
        )
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
    elif method == "scfair_auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            n_top_min=500,
            n_top_max=5000,
            flavor="seurat_v3",
            layer="counts",
            balance_method="hybrid",
            blend_global=0.95,
            marker_mode="none",
            random_state=RS,
        )
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
    else:
        raise ValueError(method)
    return genes, meta


def cluster_metrics(adata: ad.AnnData, genes: list[str], label_key: str = "cell_type") -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    if len(genes) < 10:
        return {"n_genes": len(genes), "ARI": np.nan, "NMI": np.nan, "sil_true": np.nan}
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=RS)
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), n_pcs=min(30, n_comps), random_state=RS)
    sc.tl.leiden(
        a, resolution=0.8, key_added="leiden", flavor="igraph", n_iterations=2, random_state=RS
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
            silhouette_score(X, y_true, sample_size=min(2000, a.n_obs), random_state=RS)
        ),
    }
    if "is_rare" in a.obs.columns:
        rare = a.obs["is_rare"].astype(bool)
        n_rare = int(rare.sum())
        # rare cells in clusters where rare is majority
        pure = 0
        for cl in y_pred.unique():
            m = y_pred == cl
            if rare[m].mean() >= 0.5:
                pure += int((rare & m).sum())
        out["rare_n"] = n_rare
        out["rare_frac"] = float(rare.mean())
        out["rare_recall_pure"] = pure / n_rare if n_rare else np.nan
        # also: best cluster precision/recall for rare
        best_f1 = 0.0
        for cl in y_pred.unique():
            m = y_pred == cl
            tp = int((rare & m).sum())
            prec = tp / int(m.sum()) if m.sum() else 0
            rec = tp / n_rare if n_rare else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            best_f1 = max(best_f1, f1)
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
    methods = ["hvg", "scfair_hybrid", "scfair_auto"]
    rows = []
    for dname, loader in loaders.items():
        print(f"\n==== Dataset {dname} ====", flush=True)
        try:
            adata = loader()
        except Exception as e:
            print(f"  LOAD FAIL: {e}")
            continue
        print(f"  {adata.n_obs}×{adata.n_vars}, types={adata.obs['cell_type'].nunique()}")
        for method in methods:
            print(f"  method={method}", flush=True)
            try:
                genes, meta = select_genes(adata, method)
                scores = cluster_metrics(adata, genes)
                scores["dataset"] = dname
                scores["method"] = method
                scores["n_types"] = int(adata.obs["cell_type"].nunique())
                if method == "scfair_auto":
                    scores["n_top_used"] = meta.get("n_top_genes_used")
                rows.append(scores)
                print(f"    ARI={scores['ARI']:.3f} NMI={scores['NMI']:.3f} n={scores['n_genes']}")
            except Exception as e:
                print(f"    FAIL: {e}")
                rows.append({"dataset": dname, "method": method, "error": str(e)})
    return pd.DataFrame(rows)


def run_rare_cell() -> pd.DataFrame:
    print("\n==== Rare-cell mixes from Cao ====", flush=True)
    base = load_cao()
    # choose rare type: second-largest with >= 50 cells
    vc = base.obs["cell_type"].astype(str).value_counts()
    rare_type = str(vc.index[1])
    for t in vc.index[1:]:
        if vc[t] >= 50:
            rare_type = str(t)
            break
    print(f"  rare_type={rare_type} (n={vc[rare_type]}), majority=others")

    methods = ["hvg", "scfair_hybrid", "scfair_auto"]
    rows = []
    for frac in RARE_FRACS:
        mix = make_rare_mix(base, rare_frac=frac, rare_type=rare_type, random_state=RS)
        print(
            f"\n  frac_target={frac} actual={mix.uns['rare_frac_actual']:.4f} "
            f"n={mix.n_obs} rare_n={int(mix.obs['is_rare'].sum())}",
            flush=True,
        )
        for method in methods:
            print(f"    {method}", flush=True)
            try:
                genes, meta = select_genes(mix, method)
                scores = cluster_metrics(mix, genes)
                scores["dataset"] = f"Cao_rare_{frac}"
                scores["rare_frac_target"] = frac
                scores["rare_frac_actual"] = mix.uns["rare_frac_actual"]
                scores["rare_type"] = rare_type
                scores["method"] = method
                if method == "scfair_auto":
                    scores["n_top_used"] = meta.get("n_top_genes_used")
                rows.append(scores)
                print(
                    f"      ARI={scores['ARI']:.3f} rare_recall={scores.get('rare_recall_pure', np.nan):.3f} "
                    f"rare_f1={scores.get('rare_best_cluster_f1', np.nan):.3f}"
                )
            except Exception as e:
                print(f"      FAIL: {e}")
                rows.append(
                    {
                        "dataset": f"Cao_rare_{frac}",
                        "method": method,
                        "rare_frac_target": frac,
                        "error": str(e),
                    }
                )
    return pd.DataFrame(rows)


def main():
    multi = run_multi_dataset()
    multi.to_csv(OUT / "p1_multi_dataset_ari.csv", index=False)
    print("\n=== MULTI DATASET SUMMARY ===")
    if "ARI" in multi.columns:
        print(
            multi.pivot_table(index="dataset", columns="method", values="ARI").round(3).to_string()
        )
        print("\nNMI:")
        print(
            multi.pivot_table(index="dataset", columns="method", values="NMI").round(3).to_string()
        )

    rare = run_rare_cell()
    rare.to_csv(OUT / "p1_rare_cell_cao.csv", index=False)
    print("\n=== RARE CELL SUMMARY (rare_recall_pure) ===")
    if "rare_recall_pure" in rare.columns:
        print(
            rare.pivot_table(index="rare_frac_target", columns="method", values="rare_recall_pure")
            .round(3)
            .to_string()
        )
        print("\nrare_best_cluster_f1:")
        print(
            rare.pivot_table(
                index="rare_frac_target", columns="method", values="rare_best_cluster_f1"
            )
            .round(3)
            .to_string()
        )
        print("\nARI:")
        print(
            rare.pivot_table(index="rare_frac_target", columns="method", values="ARI")
            .round(3)
            .to_string()
        )

    summary = {
        "multi_dataset": multi.replace({np.nan: None}).to_dict(orient="records"),
        "rare_cell": rare.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(OUT / "p1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT / 'p1_multi_dataset_ari.csv'}")
    print(f"Wrote {OUT / 'p1_rare_cell_cao.csv'}")
    print(f"Wrote {OUT / 'p1_summary.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
