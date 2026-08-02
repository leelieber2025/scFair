#!/usr/bin/env python
"""P3 validation: public *validated* labeled datasets.

Sources (already used in published benchmarks):
  - Cao.h5          — CellBRF (Xu et al. Bioinformatics 2023)
  - paul15          — scanpy / Paul et al. Cell 2015 (hematopoiesis)
  - pbmc3k_louvain  — 10x + scanpy processed labels
  - scIB pancreas   — Luecken et al. Nat Methods 2022 (scIB atlas task)
  - scIB lung       — same paper/figshare, layers['counts'] + obs['cell_type']

Outputs: examples/results/p3_public_validation.*
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
    a.uns["dataset_source"] = "CellBRF Cao.h5"
    return a


def load_paul15() -> ad.AnnData:
    a = sc.datasets.paul15()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    a.obs["cell_type"] = a.obs["paul15_clusters"].astype(str)
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.uns["dataset_source"] = "scanpy.datasets.paul15 (Paul 2015)"
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
    a.uns["dataset_source"] = "10x pbmc3k + scanpy processed louvain"
    return a


def load_scib_pancreas(*, max_cells: int | None = 8000, seed: int = 0) -> ad.AnnData:
    """scIB human pancreas (Luecken et al. 2022) — validated integration task.

    Has layers['counts'] and obs['celltype'] (14 types, multi-tech).
    Optionally subsample cells for runtime (stratified by celltype).
    """
    path = DATA / "human_pancreas_norm_complexBatch.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download from figshare article 12420968 "
            "(human_pancreas_norm_complexBatch.h5ad)."
        )
    a = ad.read_h5ad(path)
    if "counts" not in a.layers:
        raise ValueError("scIB pancreas expected layers['counts']")
    a.obs["cell_type"] = a.obs["celltype"].astype(str)
    # work on counts
    a.X = a.layers["counts"].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.filter_cells(a, min_genes=50)
    a.obs_names_make_unique()
    a.var_names_make_unique()

    if max_cells is not None and a.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        labels = a.obs["cell_type"].astype(str)
        # stratified sample
        idx = []
        groups = labels.groupby(labels, observed=False).indices
        # proportional
        for lab, ixs in groups.items():
            ixs = np.asarray(list(ixs))
            n = max(5, int(round(max_cells * len(ixs) / a.n_obs)))
            n = min(n, len(ixs))
            idx.append(rng.choice(ixs, size=n, replace=False))
        sel = np.concatenate(idx)
        if len(sel) > max_cells:
            sel = rng.choice(sel, size=max_cells, replace=False)
        a = a[sel].copy()
        a.uns["subsampled"] = True
        a.uns["subsample_n"] = int(a.n_obs)
    else:
        a.uns["subsampled"] = False

    a.uns["dataset_source"] = "scIB human_pancreas (figshare 12420968; Luecken Nat Methods 2022)"
    return a


def load_scib_pancreas_one_tech(tech: str = "smartseq2") -> ad.AnnData:
    """Single-technology subset (cleaner, still validated labels)."""
    path = DATA / "human_pancreas_norm_complexBatch.h5ad"
    a = ad.read_h5ad(path)
    a = a[a.obs["tech"].astype(str) == tech].copy()
    a.obs["cell_type"] = a.obs["celltype"].astype(str)
    a.X = a.layers["counts"].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.filter_cells(a, min_genes=50)
    a.obs_names_make_unique()
    a.var_names_make_unique()
    a.uns["dataset_source"] = f"scIB pancreas tech={tech}"
    return a


def load_scib_lung(*, max_cells: int | None = 8000, seed: int = 0) -> ad.AnnData:
    """scIB Lung atlas (Luecken et al. 2022) — validated integration task.

    layers['counts'], obs['cell_type'] (17 types), multi-protocol/batch.
    """
    path = DATA / "Lung_atlas_public.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download from figshare article 12420968 (Lung_atlas_public.h5ad)."
        )
    a = ad.read_h5ad(path)
    if "counts" not in a.layers:
        raise ValueError("scIB lung expected layers['counts']")
    if "cell_type" not in a.obs.columns:
        raise ValueError("scIB lung expected obs['cell_type']")
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.X = a.layers["counts"].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.filter_cells(a, min_genes=50)
    a.obs_names_make_unique()
    a.var_names_make_unique()

    if max_cells is not None and a.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        labels = a.obs["cell_type"].astype(str)
        idx = []
        for lab, ixs in labels.groupby(labels, observed=False).indices.items():
            ixs = np.asarray(list(ixs))
            n = max(5, int(round(max_cells * len(ixs) / a.n_obs)))
            n = min(n, len(ixs))
            idx.append(rng.choice(ixs, size=n, replace=False))
        sel = np.concatenate(idx)
        if len(sel) > max_cells:
            sel = rng.choice(sel, size=max_cells, replace=False)
        a = a[sel].copy()
        a.uns["subsampled"] = True
    else:
        a.uns["subsampled"] = False

    a.uns["dataset_source"] = "scIB Lung_atlas (figshare 12420968; Luecken Nat Methods 2022)"
    return a


def load_pbmc_10k_v3() -> ad.AnnData:
    """10x Chromium 3′ v3 PBMC 10k (mainstream chemistry + deeper UMI).

    Labels are Leiden + canonical marker modules (proxy), stored in
    ``examples/data/pbmc_10k_v3_labeled.h5ad``. Rebuild by re-running the
    labeling block in session notes if missing.
    """
    path = DATA / "pbmc_10k_v3_labeled.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download 10x pbmc_10k_v3 h5 and build labels "
            "(see docs/DEVELOPMENT_LOG.md §5.7.3)."
        )
    a = ad.read_h5ad(path)
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.uns["dataset_source"] = a.uns.get("label_source", "10x pbmc_10k_v3 + marker-proxy labels")
    return a


def load_scib_lung_10x(*, max_cells: int | None = 6000, seed: int = 0) -> ad.AnnData:
    """Lung atlas restricted to 10x v2 protocol (cleaner single-protocol slice)."""
    path = DATA / "Lung_atlas_public.h5ad"
    a = ad.read_h5ad(path)
    a = a[a.obs["protocol"].astype(str) == "10x v2"].copy()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.X = a.layers["counts"].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.filter_cells(a, min_genes=50)
    a.obs_names_make_unique()
    a.var_names_make_unique()
    if max_cells is not None and a.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        labels = a.obs["cell_type"].astype(str)
        idx = []
        for lab, ixs in labels.groupby(labels, observed=False).indices.items():
            ixs = np.asarray(list(ixs))
            n = max(5, int(round(max_cells * len(ixs) / a.n_obs)))
            n = min(n, len(ixs))
            idx.append(rng.choice(ixs, size=n, replace=False))
        sel = np.concatenate(idx)
        if len(sel) > max_cells:
            sel = rng.choice(sel, size=max_cells, replace=False)
        a = a[sel].copy()
    a.uns["dataset_source"] = "scIB Lung atlas protocol=10x v2"
    return a


def select_genes(adata: ad.AnnData, method: str, seed: int = 0) -> tuple[list[str], dict]:
    a = adata.copy()
    meta: dict = {}
    n_top = min(2000, a.n_vars - 1)
    if method == "hvg":
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3", layer="counts")
        genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    elif method == "scfair_hybrid":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=n_top,
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
    return {
        "n_genes": len(genes),
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "sil_true": float(
            silhouette_score(X, y_true, sample_size=min(2000, a.n_obs), random_state=seed)
        ),
    }


def main():
    loaders = {
        "Cao": load_cao,
        "paul15": load_paul15,
        "pbmc3k_louvain": load_pbmc3k_labeled,
        "scIB_pancreas_sub8k": lambda: load_scib_pancreas(max_cells=8000, seed=0),
        "scIB_pancreas_smartseq2": lambda: load_scib_pancreas_one_tech("smartseq2"),
        "scIB_lung_sub8k": lambda: load_scib_lung(max_cells=8000, seed=0),
        "scIB_lung_10x_sub6k": lambda: load_scib_lung_10x(max_cells=6000, seed=0),
        "pbmc_10k_v3": load_pbmc_10k_v3,
    }

    rows = []
    for dname, loader in loaders.items():
        print(f"\n==== {dname} ====", flush=True)
        try:
            adata = loader()
        except Exception as e:
            print(f"  LOAD FAIL: {e}", flush=True)
            rows.append({"dataset": dname, "method": "LOAD_FAIL", "error": str(e)})
            continue
        print(
            f"  {adata.n_obs}×{adata.n_vars} types={adata.obs['cell_type'].nunique()} "
            f"src={adata.uns.get('dataset_source')}",
            flush=True,
        )
        for method in METHODS:
            print(f"  {method} ...", flush=True)
            try:
                genes, meta = select_genes(adata, method, seed=0)
                scores = cluster_metrics(adata, genes, seed=0)
                scores.update(
                    {
                        "dataset": dname,
                        "method": method,
                        "n_types": int(adata.obs["cell_type"].nunique()),
                        "n_cells": int(adata.n_obs),
                        "n_vars": int(adata.n_vars),
                        "n_top_used": meta.get("n_top_genes_used"),
                        "source": adata.uns.get("dataset_source"),
                    }
                )
                rows.append(scores)
                print(
                    f"    ARI={scores['ARI']:.3f} NMI={scores['NMI']:.3f} n={scores['n_genes']}",
                    flush=True,
                )
            except Exception as e:
                print(f"    FAIL {e}", flush=True)
                rows.append({"dataset": dname, "method": method, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p3_public_validation.csv", index=False)

    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df
    print("\n======== ARI ========")
    if not ok.empty:
        print(ok.pivot_table(index="dataset", columns="method", values="ARI").round(3).to_string())
        print("\n======== NMI ========")
        print(ok.pivot_table(index="dataset", columns="method", values="NMI").round(3).to_string())

        wins = {"hybrid": 0, "hvg": 0, "tie": 0}
        for d in ok["dataset"].unique():
            sub = ok[ok["dataset"] == d].set_index("method")
            if "hvg" in sub.index and "scfair_hybrid" in sub.index:
                ha, hh = float(sub.loc["hvg", "ARI"]), float(sub.loc["scfair_hybrid", "ARI"])
                if hh > ha + 1e-6:
                    wins["hybrid"] += 1
                elif ha > hh + 1e-6:
                    wins["hvg"] += 1
                else:
                    wins["tie"] += 1
        print("\nARI wins hybrid vs hvg:", wins)
    else:
        wins = {}

    summary = {
        "records": df.replace({np.nan: None}).to_dict(orient="records"),
        "wins_ari_hybrid_vs_hvg": wins,
        "note": (
            "New public validated set: scIB pancreas (figshare 12420968). "
            "Classic CellBRF Rdata/S3 mirrors mostly 403/404; Zeisel counts "
            "downloaded but labels not in GEO series matrix."
        ),
    }
    with open(OUT / "p3_public_validation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT / 'p3_public_validation.csv'}")
    print("DONE")


if __name__ == "__main__":
    main()
