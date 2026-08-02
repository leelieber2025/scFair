#!/usr/bin/env python
"""Protein gold standard: auto-n vs fixed-k, and hybrid vs scanpy HVG.

Answers two questions on the three CITE-seq panels of §5.12
(protein-space Leiden partition as gold standard — non-circular):

  1. Is auto-n better than fixed 2000 or fixed 3000?
  2. How much does hybrid gain over scanpy seurat_v3 HVG at the same k?

Datasets (examples/data/):
  pbmc_seurat_v4.h5ad   228 ADT, subsampled 20k
  sln_208.h5ad          198 ADT, mouse spleen+LN
  pbmc_5k_protein_v3.h5ad  29 ADT

Outputs: examples/results/adt_auto_k_protein_gold.csv / .json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

RARE_MAX_FRAC = 0.02
MIN_CLUSTER_FRAC = 0.003
LEIDEN_RES = 0.8

# Configs directly answer Q1 (k choice) and Q2 (hybrid vs HVG).
# hybrid uses shipped defaults: resolution=0.5, neighbor_contrast=0, blend=0.95
CONFIGS = {
    "hvg_2000": dict(kind="hvg", n_top=2000),
    "hvg_3000": dict(kind="hvg", n_top=3000),
    "hybrid_2000": dict(kind="hybrid", n_top=2000),
    "hybrid_3000": dict(kind="hybrid", n_top=3000),
    "hybrid_auto": dict(kind="hybrid", n_top="auto"),
}

DATASETS = {
    "pbmc_seurat_v4_20k": dict(
        path="pbmc_seurat_v4.h5ad",
        adt_key="protein_counts",
        ref_label="celltype.l2",
        subsample=20000,
        adt_res=0.4,
    ),
    "sln_208_mouse": dict(
        path="sln_208.h5ad",
        adt_key="protein_expression",
        ref_label="cell_types",
        subsample=None,
        adt_res=0.4,
    ),
    "pbmc_5k_v3": dict(
        path="pbmc_5k_protein_v3.h5ad",
        adt_key="protein_expression",
        ref_label=None,
        subsample=None,
        adt_res=0.4,
    ),
}


def clr_across_cells(P: np.ndarray) -> np.ndarray:
    L = np.log1p(np.asarray(P, dtype=float))
    return L - L.mean(axis=0, keepdims=True)


def build_protein_labels(adata, adt_key: str, ref_label: str | None, adt_res: float):
    """Protein-only Leiden partition; names from published annotation only for display."""
    P = adata.obsm[adt_key]
    names = (
        list(P.columns)
        if hasattr(P, "columns")
        else [str(x) for x in adata.uns.get("protein_names", [])]
    )
    mat = P.values if hasattr(P, "values") else np.asarray(P)
    adt = ad.AnnData(X=clr_across_cells(mat).astype(np.float32))
    adt.obs_names = adata.obs_names
    adt.var_names = names if len(names) == mat.shape[1] else [f"p{i}" for i in range(mat.shape[1])]
    sc.pp.scale(adt, max_value=10)
    n_comps = min(30, adt.n_vars - 1, adt.n_obs - 1)
    sc.pp.pca(adt, n_comps=n_comps, random_state=0)
    sc.pp.neighbors(adt, n_neighbors=20, n_pcs=n_comps, random_state=0)
    sc.tl.leiden(
        adt,
        resolution=adt_res,
        key_added="p",
        flavor="igraph",
        n_iterations=2,
        random_state=0,
    )
    cl = adt.obs["p"].astype(str)
    counts = cl.value_counts()
    keep = counts[counts / len(cl) >= MIN_CLUSTER_FRAC].index
    names_map = {}
    for c in counts.index:
        if c not in keep:
            names_map[c] = "unassigned"
        elif ref_label is not None and ref_label in adata.obs.columns:
            maj = adata.obs.loc[(cl == c).to_numpy(), ref_label].astype(str).mode()
            names_map[c] = f"p{c}_{maj.iloc[0]}" if len(maj) else f"p{c}"
        else:
            names_map[c] = f"p{c}"
    adata.obs["cell_type"] = cl.map(names_map).values
    adata.obs["adt_confident"] = adata.obs["cell_type"] != "unassigned"
    return adata


def load_cite(name: str):
    spec = DATASETS[name]
    a = ad.read_h5ad(DATA / spec["path"])
    if spec["subsample"] and a.n_obs > spec["subsample"]:
        rng = np.random.default_rng(0)
        strat = (
            a.obs[spec["ref_label"]].astype(str)
            if spec["ref_label"] and spec["ref_label"] in a.obs.columns
            else pd.Series("all", index=a.obs_names)
        )
        idx = []
        for _, ix in strat.groupby(strat).indices.items():
            ix = np.asarray(list(ix))
            n = min(len(ix), max(20, int(round(spec["subsample"] * len(ix) / a.n_obs))))
            idx.append(rng.choice(ix, size=n, replace=False))
        a = a[np.concatenate(idx)].copy()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.layers["counts"] = a.X.copy()
    a = build_protein_labels(a, spec["adt_key"], spec["ref_label"], spec["adt_res"])
    return a


def per_population_f1(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    out = {}
    for pop in sorted(y_true.unique()):
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in y_pred.unique():
            p = (y_pred == cl).to_numpy()
            tp = float(np.sum(t & p))
            if tp:
                prec, rec = tp / p.sum(), tp / t.sum()
                best = max(best, 2 * prec * rec / (prec + rec))
        out[pop] = best
    return out


def select_genes(adata, cfg: dict, seed: int):
    a = adata.copy()
    n_top = cfg["n_top"]
    if cfg["kind"] == "hvg":
        n = min(int(n_top), a.n_vars - 1)
        sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat_v3", layer="counts")
        k_used = n
    else:
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
        meta = a.uns.get("scfair", {}).get("hvg", {})
        k_used = int(meta.get("n_top_genes_used", a.var["highly_variable"].sum()))
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    return genes, k_used


def evaluate(adata, genes, seed: int):
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, [g for g in genes if g in a.var_names]].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a,
        resolution=LEIDEN_RES,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    conf = a.obs["adt_confident"].to_numpy(dtype=bool)
    y_true = a.obs["cell_type"].astype(str)[conf]
    y_pred = a.obs["leiden"].astype(str)[conf]
    f1 = per_population_f1(y_true, y_pred)
    prev = y_true.value_counts(normalize=True)
    rare = [p for p in f1 if prev.get(p, 0.0) < RARE_MAX_FRAC]
    dc = [p for p in f1 if "DC" in p.upper()]
    return {
        "n_genes": a.n_vars,
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1.values()))),
        "rare_f1_mean": float(np.mean([f1[p] for p in rare])) if rare else np.nan,
        "dc_f1_mean": float(np.mean([f1[p] for p in dc])) if dc else np.nan,
        "n_pop": len(f1),
        "n_rare": len(rare),
        **{f"f1_{pop}": float(v) for pop, v in f1.items()},
    }


def main(which=None, seeds=(0, 1, 2)):
    rows = []
    meta = {}
    for dname in which or DATASETS:
        print(f"\n################ {dname} ################", flush=True)
        adata = load_cite(dname)
        conf = adata.obs["adt_confident"].to_numpy(dtype=bool)
        vc = adata.obs.loc[conf, "cell_type"].astype(str).value_counts()
        frac = vc / vc.sum()
        print(
            f"{adata.n_obs} cells x {adata.n_vars} genes | "
            f"{int(conf.sum())} confident ({100 * conf.mean():.1f}%) | "
            f"{len(vc)} protein populations, {(frac < RARE_MAX_FRAC).sum()} rare(<2%)",
            flush=True,
        )
        print(pd.DataFrame({"n": vc, "pct": (100 * frac).round(2)}).to_string())
        meta[dname] = {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "populations": {str(k): int(v) for k, v in vc.items()},
            "rare": frac.index[frac < RARE_MAX_FRAC].tolist(),
        }
        for cfg_name, cfg in CONFIGS.items():
            for seed in seeds:
                try:
                    genes, k_used = select_genes(adata, cfg, seed)
                    res = evaluate(adata, genes, seed)
                    res.update(
                        {
                            "dataset": dname,
                            "config": cfg_name,
                            "seed": seed,
                            "k_used": k_used,
                        }
                    )
                    rows.append(res)
                    print(
                        f"  {cfg_name:14s} seed={seed} k={k_used:4d} "
                        f"ARI={res['ARI']:.3f} macroF1={res['macro_f1']:.3f} "
                        f"rareF1={res['rare_f1_mean']:.3f} dcF1={res['dc_f1_mean']:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"  {cfg_name} seed={seed} FAIL {type(e).__name__}: {e}",
                        flush=True,
                    )
                    rows.append(
                        {
                            "dataset": dname,
                            "config": cfg_name,
                            "seed": seed,
                            "error": str(e),
                        }
                    )
            # checkpoint after each config
            pd.DataFrame(rows).to_csv(OUT / "adt_auto_k_protein_gold.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_auto_k_protein_gold.csv", index=False)
    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df

    for metric in ("ARI", "macro_f1", "rare_f1_mean", "dc_f1_mean"):
        if metric not in ok.columns:
            continue
        print(f"\n======== {metric} (mean over seeds) ========")
        print(ok.pivot_table(index="dataset", columns="config", values=metric).round(4).to_string())

    if "k_used" in ok.columns:
        print("\n======== k_used (mean) ========")
        print(
            ok.pivot_table(index="dataset", columns="config", values="k_used").round(0).to_string()
        )

    print("\n======== Q1: auto / 2000 / 3000  (Δ vs hybrid_2000) ========")
    for metric in ("ARI", "macro_f1", "rare_f1_mean"):
        if metric not in ok.columns:
            continue
        piv = ok.pivot_table(index=["dataset", "seed"], columns="config", values=metric)
        if "hybrid_2000" not in piv.columns:
            continue
        print(f"  -- {metric} --")
        for c in ("hybrid_auto", "hybrid_3000", "hvg_2000", "hvg_3000"):
            if c not in piv.columns:
                continue
            d = (piv[c] - piv["hybrid_2000"]).groupby("dataset").mean()
            print(f"    {c:14s} " + "  ".join(f"{k}={v:+.4f}" for k, v in d.items()))

    print("\n======== Q2: hybrid vs HVG at same k  (Δ hybrid − hvg) ========")
    for metric in ("ARI", "macro_f1", "rare_f1_mean", "dc_f1_mean"):
        if metric not in ok.columns:
            continue
        piv = ok.pivot_table(index=["dataset", "seed"], columns="config", values=metric)
        print(f"  -- {metric} --")
        for k in (2000, 3000):
            h, s = f"hybrid_{k}", f"hvg_{k}"
            if h not in piv.columns or s not in piv.columns:
                continue
            d = (piv[h] - piv[s]).groupby("dataset").mean()
            overall = (piv[h] - piv[s]).mean()
            wins = int(((piv[h] - piv[s]) > 0).sum())
            n = int((piv[h] - piv[s]).notna().sum())
            print(
                f"    @{k}: overall={overall:+.4f}  wins={wins}/{n}  |  "
                + "  ".join(f"{ds}={v:+.4f}" for ds, v in d.items())
            )

    # pooled paired t-style summary (9 points)
    print("\n======== pooled over 3 datasets × 3 seeds ========")
    piv_ari = ok.pivot_table(index=["dataset", "seed"], columns="config", values="ARI")
    piv_f1 = ok.pivot_table(index=["dataset", "seed"], columns="config", values="macro_f1")
    pairs = [
        ("hybrid_auto", "hybrid_2000"),
        ("hybrid_3000", "hybrid_2000"),
        ("hybrid_2000", "hvg_2000"),
        ("hybrid_3000", "hvg_3000"),
        ("hybrid_auto", "hvg_2000"),
    ]
    for a, b in pairs:
        for name, piv in (("ARI", piv_ari), ("macro_f1", piv_f1)):
            if a not in piv.columns or b not in piv.columns:
                continue
            d = (piv[a] - piv[b]).dropna()
            print(
                f"  {a:14s} − {b:14s}  {name:8s}: Δ={d.mean():+.4f}  wins={(d > 0).sum()}/{len(d)}"
            )

    with open(OUT / "adt_auto_k_protein_gold.json", "w") as f:
        json.dump(
            {
                "datasets": meta,
                "configs": {k: {**v, "n_top": str(v["n_top"])} for k, v in CONFIGS.items()},
                "records": df.replace({np.nan: None}).to_dict(orient="records"),
            },
            f,
            indent=2,
            default=str,
        )
    print("\nDONE →", OUT / "adt_auto_k_protein_gold.csv")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(which=args or None)
