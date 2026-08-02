#!/usr/bin/env python
"""PBMC3k benchmark: scanpy vs scFair none / score / reweight.

Usage:
    python examples/pbmc3k_benchmark.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import scfair as scf

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

MARKERS = {
    "T": ["CD3D", "CD3E", "CD3G", "IL7R", "CD8A", "CD8B"],
    "NK": ["NKG7", "GNLY", "KLRD1"],
    "B": ["MS4A1", "CD79A", "CD79B"],
    "Mono": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A"],
    "DC": ["FCER1A", "CST3"],
    "Platelet": ["PPBP"],
}
NOISE = ["RPL19", "RPS6", "MT-CO3", "TPT1", "FOS"]
N_TOP = 2000


def load_pbmc3k():
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


def geneset(ad):
    return set(ad.var_names[ad.var["highly_variable"]].astype(str))


def recovery(gs):
    rows = []
    for lin, ms in MARKERS.items():
        present = [m for m in ms if m in ad_ref.var_names]
        hit = sum(m in gs for m in present)
        rows.append(
            {
                "lineage": lin,
                "hit": hit,
                "n": len(present),
                "rate": hit / len(present) if present else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    global ad_ref
    print("=== Load pbmc3k ===")
    ad_ref = load_pbmc3k()
    print(f"After QC: {ad_ref.n_obs} × {ad_ref.n_vars}")

    results = {}

    # scanpy baseline
    print("\n=== scanpy seurat_v3 ===")
    ad_sc = ad_ref.copy()
    t0 = time.perf_counter()
    sc.pp.highly_variable_genes(ad_sc, n_top_genes=N_TOP, flavor="seurat_v3", layer="counts")
    t_sc = time.perf_counter() - t0
    g_sc = geneset(ad_sc)
    print(f"  n={len(g_sc)} time={t_sc:.1f}s")
    results["scanpy_seurat_v3"] = {"n_hvg": len(g_sc), "time_s": round(t_sc, 2)}

    methods = [
        ("scfair_none", dict(balance_method="none")),
        ("scfair_score", dict(balance_method="score", balance_power=0.5)),
        ("scfair_reweight", dict(balance_method="reweight", balance_power=0.5)),
        (
            "scfair_score_filter",
            dict(balance_method="score", balance_power=0.5, filter_ribo=True, filter_mito=True),
        ),
    ]

    gene_sets = {"scanpy_seurat_v3": g_sc}
    for name, kwargs in methods:
        print(f"\n=== {name} ===")
        ad = ad_ref.copy()
        t0 = time.perf_counter()
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=N_TOP,
            flavor="seurat_v3",
            layer="counts",
            resolution=1.0,
            min_cluster_size=30,
            random_state=0,
            **kwargs,
        )
        dt = time.perf_counter() - t0
        gs = geneset(ad)
        gene_sets[name] = gs
        meta = ad.uns.get("scfair", {}).get("hvg", {})
        ov = len(gs & g_sc)
        print(
            f"  n={len(gs)} time={dt:.1f}s overlap_scanpy={ov}/{N_TOP} "
            f"clusters={meta.get('n_clusters_used')} selection={meta.get('selection')}"
        )
        results[name] = {
            "n_hvg": len(gs),
            "time_s": round(dt, 2),
            "overlap_scanpy": ov,
            "jaccard_scanpy": round(ov / len(gs | g_sc), 3),
            "n_clusters_used": meta.get("n_clusters_used"),
            "selection": meta.get("selection"),
            "score_type": meta.get("score_type"),
            "balance_power": meta.get("balance_power"),
        }
        # stash scores for marker table
        ad.var[f"score_{name}"] = ad.var["scfair_score"]
        if name == "scfair_score":
            ad_score = ad
        if name == "scfair_reweight":
            ad_reweight = ad

    # pairwise overlaps among scfair methods
    print("\n=== pairwise overlap ===")
    keys = list(gene_sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ov = len(gene_sets[a] & gene_sets[b])
            print(f"  {a} ∩ {b} = {ov}")

    # marker table
    rows = []
    for lin, ms in MARKERS.items():
        for m in ms:
            if m not in ad_ref.var_names:
                continue
            row = {"lineage": lin, "gene": m}
            for name, gs in gene_sets.items():
                row[name] = m in gs
            if "ad_score" in dir() or True:
                try:
                    row["S_score"] = float(ad_score.var.loc[m, "scfair_score"])
                except Exception:
                    row["S_score"] = np.nan
                try:
                    row["S_reweight"] = float(ad_reweight.var.loc[m, "scfair_score"])
                except Exception:
                    row["S_reweight"] = np.nan
            rows.append(row)
    mt = pd.DataFrame(rows)
    print("\n=== markers ===")
    print(mt.to_string(index=False))
    mt.to_csv(OUT / "pbmc3k_markers_all_methods.csv", index=False)

    print("\n=== recovery rates ===")
    rec_rows = []
    for name, gs in gene_sets.items():
        for lin, ms in MARKERS.items():
            present = [m for m in ms if m in ad_ref.var_names]
            hit = sum(m in gs for m in present)
            rec_rows.append(
                {
                    "method": name,
                    "lineage": lin,
                    "rate": hit / len(present),
                    "hit": hit,
                    "n": len(present),
                }
            )
    rec = pd.DataFrame(rec_rows)
    print(rec.pivot(index="lineage", columns="method", values="rate").round(2).to_string())
    rec.to_csv(OUT / "pbmc3k_recovery_rates.csv", index=False)

    print("\n=== noise genes in HVG? ===")
    for g in NOISE:
        if g not in ad_ref.var_names:
            continue
        flags = {name: (g in gs) for name, gs in gene_sets.items()}
        print(f"  {g:8s} {flags}")

    summary = {
        "n_cells": int(ad_ref.n_obs),
        "n_genes": int(ad_ref.n_vars),
        "n_top": N_TOP,
        "methods": results,
    }
    with open(OUT / "pbmc3k_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT / 'pbmc3k_benchmark_summary.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
