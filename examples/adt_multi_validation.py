#!/usr/bin/env python
"""Validate scFair's defaults on THREE CITE-seq panels (§5.12).

§5.9–5.11 were derived from a single 14-antibody PBMC panel. Everything shipped
out of that — the `resolution` 1.0→0.5 default change and the opt-in
`neighbor_contrast` — rests on one dataset, one tissue, one chemistry. This
script re-tests them on:

  pbmc_seurat_v4   161k PBMC, **228** antibodies (Hao 2021), subsampled to ~20k.
                   The only panel here carrying CD11c/CD123/CD1c/CD141/CD303,
                   so it is the first chance to test the DC claim in §5.5 that
                   has been shadow-bound since it was made.
  sln_208          15.8k **mouse** spleen + lymph node, 198 antibodies.
                   Different tissue *and* species — the real generalisation test
                   for `neighbor_contrast`, which so far is validated on exactly
                   one rare population (non-classical monocytes).
  pbmc_5k_v3       4k PBMC, 29 antibodies. Same assay family as §5.9 — a
                   near-replication, included to separate "panel depth" from
                   "different biology".

Gold standard
-------------
The **protein-space Leiden partition is the gold standard**, unnamed. ARI, NMI
and macro-F1 are all defined on unnamed partitions, so no hand-written rule
table is needed for a 228-plex, and rare populations are identified
automatically by prevalence rather than by whichever ones someone thought to
name. RNA is never used to build the partition.

Cluster *names* are the majority overlap with each dataset's published
annotation. Names are for reading the output only — they enter no metric. (For
pbmc_seurat_v4 that annotation is WNN-derived, i.e. partly RNA-based; that is
harmless for labelling but would not be acceptable for the partition itself.)

Outputs: examples/results/adt_multi_validation.csv / .json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

RARE_MAX_FRAC = 0.02
MIN_CLUSTER_FRAC = 0.003  # protein clusters below this are dropped as unassigned
LEIDEN_RES = 0.8  # downstream RNA clustering, same as §5.9

CONFIGS = {
    "hvg": None,
    "hybrid_res1.0": dict(balance_method="hybrid", resolution=1.0, neighbor_contrast=0.0),
    "hybrid_res0.5": dict(balance_method="hybrid", resolution=0.5, neighbor_contrast=0.0),
    "hybrid_res1.0_nc1": dict(balance_method="hybrid", resolution=1.0, neighbor_contrast=1.0),
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
    # adt_res is fixed at 0.4 for every dataset: the gold-standard granularity
    # must not be tuned per dataset, or it becomes a free parameter chosen
    # after seeing the scores.
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
    """Protein-only Leiden partition; names borrowed from published annotation."""
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
    # disambiguate clusters that borrowed the same published name
    seen: dict[str, int] = {}
    for c in counts.index:
        base = names_map[c]
        if base == "unassigned":
            continue
        seen[base] = seen.get(base, 0) + 1
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
            if spec["ref_label"] in a.obs.columns
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


def select_genes(adata, cfg, seed: int):
    a = adata.copy()
    n_top = min(2000, a.n_vars - 1)
    if cfg is None:
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3", layer="counts")
    else:
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=n_top,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            blend_global=0.95,
            random_state=seed,
            **cfg,
        )
    return a.var_names[a.var["highly_variable"]].astype(str).tolist()


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
    res = {
        "n_genes": a.n_vars,
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1.values()))),
        "rare_f1_mean": float(np.mean([f1[p] for p in rare])) if rare else np.nan,
        "dc_f1_mean": float(np.mean([f1[p] for p in dc])) if dc else np.nan,
        "n_pop": len(f1),
        "n_rare": len(rare),
    }
    for pop, v in f1.items():
        res[f"f1_{pop}"] = float(v)
    return res


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
            f"{len(vc)} protein populations, {(frac < RARE_MAX_FRAC).sum()} rare(<2%)"
        )
        print(pd.DataFrame({"n": vc, "pct": (100 * frac).round(2)}).to_string())
        meta[dname] = {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "populations": vc.to_dict(),
            "rare": frac.index[frac < RARE_MAX_FRAC].tolist(),
        }
        for cfg_name, cfg in CONFIGS.items():
            for seed in seeds:
                try:
                    genes = select_genes(adata, cfg, seed)
                    res = evaluate(adata, genes, seed)
                    res.update({"dataset": dname, "config": cfg_name, "seed": seed})
                    rows.append(res)
                    print(
                        f"  {cfg_name:18s} seed={seed} ARI={res['ARI']:.3f} "
                        f"macroF1={res['macro_f1']:.3f} rareF1={res['rare_f1_mean']:.3f} "
                        f"dcF1={res['dc_f1_mean']:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {cfg_name} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                    rows.append(
                        {"dataset": dname, "config": cfg_name, "seed": seed, "error": str(e)}
                    )
        pd.DataFrame(rows).to_csv(OUT / "adt_multi_validation.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_multi_validation.csv", index=False)
    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df
    for metric in ("ARI", "macro_f1", "rare_f1_mean", "dc_f1_mean"):
        if metric not in ok.columns:
            continue
        print(f"\n======== {metric} (mean over seeds) ========")
        print(ok.pivot_table(index="dataset", columns="config", values=metric).round(3).to_string())
    print("\n======== paired deltas vs hvg ========")
    for metric in ("ARI", "macro_f1", "rare_f1_mean", "dc_f1_mean"):
        if metric not in ok.columns:
            continue
        piv = ok.pivot_table(index=["dataset", "seed"], columns="config", values=metric)
        if "hvg" not in piv.columns:
            continue
        print(f"  -- {metric} --")
        for c in piv.columns:
            if c == "hvg":
                continue
            d = (piv[c] - piv["hvg"]).groupby("dataset").mean()
            print(f"    {c:18s} " + "  ".join(f"{k}={v:+.4f}" for k, v in d.round(4).items()))
    with open(OUT / "adt_multi_validation.json", "w") as f:
        json.dump(
            {"datasets": meta, "records": df.replace({np.nan: None}).to_dict(orient="records")},
            f,
            indent=2,
            default=str,
        )
    print("\nDONE")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(which=args or None)
