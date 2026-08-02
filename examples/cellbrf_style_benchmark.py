#!/usr/bin/env python
"""P0 benchmark in CellBRF style: feature selection → clustering → ARI/NMI.

Dataset
-------
Cao.h5 from CellBRF repo (https://github.com/xuyp-csu/CellBRF/h5data/Cao.h5)
  - X: cells × genes (counts-like)
  - Y: integer cell-type labels

Methods compared
----------------
  - scanpy HVG seurat_v3 (n_top=2000)
  - scFair hybrid fixed 2000
  - scFair hybrid n_top=auto (ensemble)
  - scFair none (global = scanpy-like path)

Downstream (aligned with CellBRF-style evaluation)
--------------------------------------------------
  normalize_total → log1p → scale → PCA → neighbors → Leiden
  metrics: ARI, NMI vs true labels; silhouette (PCA, true labels)

Usage
-----
  python examples/cellbrf_style_benchmark.py
  # expects examples/data/Cao.h5 (download if missing)
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
DATA = ROOT / "data" / "Cao.h5"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
RS = 0


def ensure_cao() -> Path:
    if DATA.exists():
        return DATA
    DATA.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/xuyp-csu/CellBRF/main/h5data/Cao.h5"
    import urllib.request

    print(f"Downloading {url} → {DATA}")
    urllib.request.urlretrieve(url, DATA)
    return DATA


def load_cao() -> ad.AnnData:
    path = ensure_cao()
    with h5py.File(path, "r") as f:
        X = np.array(f["X"])
        y = np.array(f["Y"]).astype(int).ravel()
    # CellBRF demo: X is cells × genes
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
    adata.obs["cell_type"] = pd.Categorical(y.astype(str))
    adata.layers["counts"] = adata.X.copy()
    # light gene filter
    sc.pp.filter_genes(adata, min_cells=3)
    print(
        f"Cao: {adata.n_obs} cells × {adata.n_vars} genes, {adata.obs['cell_type'].nunique()} types"
    )
    return adata


def select_genes(adata: ad.AnnData, method: str) -> list[str]:
    a = adata.copy()
    if method == "scanpy_hvg_2000":
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3", layer="counts")
        return a.var_names[a.var["highly_variable"]].astype(str).tolist()
    if method == "scfair_none_2000":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=2000,
            flavor="seurat_v3",
            layer="counts",
            balance_method="none",
            marker_mode="none",
        )
        return a.var_names[a.var["highly_variable"]].astype(str).tolist()
    if method == "scfair_hybrid_2000":
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
        return a.var_names[a.var["highly_variable"]].astype(str).tolist()
    if method == "scfair_hybrid_auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            n_top_min=500,
            n_top_max=5000,
            auto_n_method="ensemble",
            flavor="seurat_v3",
            layer="counts",
            balance_method="hybrid",
            blend_global=0.95,
            marker_mode="none",
            random_state=RS,
        )
        n_used = a.uns["scfair"]["hvg"].get("n_top_genes_used")
        auto = a.uns["scfair"]["hvg"].get("auto_n")
        print(f"  auto n_top_genes_used={n_used} meta_picks={auto and auto.get('method_picks')}")
        return a.var_names[a.var["highly_variable"]].astype(str).tolist()
    raise ValueError(method)


def cluster_and_score(adata: ad.AnnData, genes: list[str]) -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=RS)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=RS)
    sc.tl.leiden(
        a,
        resolution=0.8,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=RS,
    )
    y_true = a.obs["cell_type"].astype(str)
    y_pred = a.obs["leiden"].astype(str)
    X = a.obsm["X_pca"][:, : min(20, n_comps)]
    return {
        "n_genes": len(genes),
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "sil_true": float(
            silhouette_score(X, y_true, sample_size=min(2000, a.n_obs), random_state=RS)
        ),
        "sil_leiden": float(
            silhouette_score(X, y_pred, sample_size=min(2000, a.n_obs), random_state=RS)
        ),
    }


def main():
    print("=== CellBRF-style P0: Cao.h5 ===")
    adata = load_cao()
    methods = [
        "scanpy_hvg_2000",
        "scfair_none_2000",
        "scfair_hybrid_2000",
        "scfair_hybrid_auto",
    ]
    rows = []
    gene_sets = {}
    for m in methods:
        print(f"\n--- {m} ---", flush=True)
        genes = select_genes(adata, m)
        gene_sets[m] = set(genes)
        scores = cluster_and_score(adata, genes)
        scores["method"] = m
        rows.append(scores)
        print(scores)

    df = pd.DataFrame(rows).set_index("method")
    print("\n=== summary ===")
    print(df.round(4).to_string())

    # overlaps with scanpy HVG
    base = gene_sets["scanpy_hvg_2000"]
    print("\n=== gene-set overlap with scanpy HVG ===")
    for m, gs in gene_sets.items():
        ov = len(gs & base)
        print(f"  {m}: {ov}/{len(gs)} (Jaccard={ov / len(gs | base):.3f})")

    df.to_csv(OUT / "cao_cellbrf_style_p0.csv")
    summary = {
        "dataset": "Cao.h5 (CellBRF repo)",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_types": int(adata.obs["cell_type"].nunique()),
        "results": df.reset_index().to_dict(orient="records"),
        "note": "ARI/NMI vs true labels after Leiden on selected features",
    }
    with open(OUT / "cao_cellbrf_style_p0.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT / 'cao_cellbrf_style_p0.csv'}")
    print("DONE")


if __name__ == "__main__":
    main()
