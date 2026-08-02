#!/usr/bin/env python
"""Definitive head-to-head: is auto-n worth it, and how big is scFair's gain?

Answers exactly two questions, with one protocol and one set of defaults:

  Q1. Is ``n_top_genes="auto"`` better than a fixed 2000 or a fixed 3000?
  Q2. How much does scFair improve on plain scanpy HVG?

Everything earlier in DEVELOPMENT_LOG mixes settings — some runs used the old
``resolution=1.0`` default, some 3 seeds, some 5, and two different gold-standard
protocols. Numbers assembled across those runs are not comparable. This script
re-runs everything at **current defaults**, same seeds, same evaluation.

Arms
----
hvg2000 / hvg3000    scanpy ``highly_variable_genes(flavor="seurat_v3")``.
                     hvg3000 exists to separate "scFair helps" from "more genes
                     help" — without it, any scFair-at-3000 win is confounded.
scfair2000/3000      scFair at the same fixed k.
scfair_auto          scFair with ensemble auto-k; the k it picks is recorded.

Datasets
--------
Non-circular (labels from surface protein, never from RNA):
  pbmc10k_adt14   pbmc5k_adt29   pbmc_seurat_v4_20k(228)   sln_208_mouse(198)
Shadowed (author labels produced by ~2000-HVG pipelines — biased toward the
baseline, so a scFair win here is conservative and a loss is ambiguous):
  Cao   paul15   pbmc3k_louvain   pancreas_smartseq2

Outputs: examples/results/final_comparison.csv / .json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adt_gold_benchmark import load_labeled as load_adt14  # noqa: E402
from adt_multi_validation import load_cite  # noqa: E402
from p3_public_validation import (  # noqa: E402
    load_cao,
    load_paul15,
    load_pbmc3k_labeled,
    load_scib_pancreas_one_tech,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
SEEDS = [0, 1, 2]
LEIDEN_RES = 0.8

ARMS = {
    "hvg2000": ("scanpy", 2000),
    "hvg3000": ("scanpy", 3000),
    "scfair2000": ("scfair", 2000),
    "scfair3000": ("scfair", 3000),
    "scfair_auto": ("scfair", "auto"),
}

NON_CIRCULAR = ["pbmc10k_adt14", "pbmc5k_adt29", "pbmc_seurat_v4_20k", "sln_208_mouse"]
SHADOWED = ["Cao", "paul15", "pbmc3k_louvain", "pancreas_smartseq2"]

LOADERS = {
    "pbmc10k_adt14": load_adt14,
    "pbmc5k_adt29": lambda: load_cite("pbmc_5k_v3"),
    "pbmc_seurat_v4_20k": lambda: load_cite("pbmc_seurat_v4_20k"),
    "sln_208_mouse": lambda: load_cite("sln_208_mouse"),
    "Cao": load_cao,
    "paul15": load_paul15,
    "pbmc3k_louvain": load_pbmc3k_labeled,
    "pancreas_smartseq2": lambda: load_scib_pancreas_one_tech("smartseq2"),
}


def select(adata, arm: str, seed: int):
    kind, k = ARMS[arm]
    a = adata.copy()
    meta = {}
    if kind == "scanpy":
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(int(k), a.n_vars - 1), flavor="seurat_v3", layer="counts"
        )
    else:
        kw = dict(flavor="seurat_v3", layer="counts", marker_mode="none", random_state=seed)
        if k == "auto":
            kw.update(n_top_genes="auto", n_top_min=500, n_top_max=min(5000, a.n_vars - 1))
        else:
            kw.update(n_top_genes=min(int(k), a.n_vars - 1))
        scf.pp.highly_variable_genes(a, **kw)
        meta = {"k_used": a.uns.get("scfair", {}).get("hvg", {}).get("n_top_genes_used")}
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    return genes, meta


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
    if "adt_confident" in a.obs.columns:
        conf = a.obs["adt_confident"].to_numpy(dtype=bool)
    else:
        conf = np.ones(a.n_obs, dtype=bool)
    y_true = a.obs["cell_type"].astype(str)[conf]
    y_pred = a.obs["leiden"].astype(str)[conf]

    f1s, prev = {}, y_true.value_counts(normalize=True)
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in y_pred.unique():
            p = (y_pred == cl).to_numpy()
            tp = float(np.sum(t & p))
            if tp:
                pr, rc = tp / p.sum(), tp / t.sum()
                best = max(best, 2 * pr * rc / (pr + rc))
        f1s[pop] = best
    rare = [p for p in f1s if prev.get(p, 0) < 0.02]
    return {
        "n_genes": a.n_vars,
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "rare_f1": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
        "n_pop": len(f1s),
    }


def main(which=None):
    rows = []
    for dname in which or LOADERS:
        print(f"\n################ {dname} ################", flush=True)
        try:
            adata = LOADERS[dname]()
        except Exception as e:
            print(f"  LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  {adata.n_obs} x {adata.n_vars}", flush=True)
        for arm in ARMS:
            for seed in SEEDS:
                try:
                    genes, meta = select(adata, arm, seed)
                    res = evaluate(adata, genes, seed)
                    res.update(
                        {
                            "dataset": dname,
                            "arm": arm,
                            "seed": seed,
                            "k_used": meta.get("k_used"),
                            "circular": dname in SHADOWED,
                        }
                    )
                    rows.append(res)
                    print(
                        f"  {arm:12s} seed={seed} k={res['n_genes']:5d} "
                        f"ARI={res['ARI']:.3f} macroF1={res['macro_f1']:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                    rows.append({"dataset": dname, "arm": arm, "seed": seed, "error": str(e)})
        pd.DataFrame(rows).to_csv(OUT / "final_comparison.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_comparison.csv", index=False)
    with open(OUT / "final_comparison.json", "w") as f:
        json.dump(df.replace({np.nan: None}).to_dict(orient="records"), f, indent=2, default=str)
    print("\nDONE")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(which=args or None)
