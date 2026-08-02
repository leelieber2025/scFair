#!/usr/bin/env python
"""Overfitting check: does the ADT-derived change regress the shadowed panel?

`neighbor_contrast` was designed and tuned on ONE dataset (the ADT gold
standard, §5.9). A single-dataset tune is trivially overfittable, so it must
be re-run on the labelled panel from §5.1–5.8 before it can be a default.

Those labels carry the protocol shadow (§5.7.5) and so are biased *toward*
the k=2000 seurat_v3 subspace. That makes them a **conservative** test for a
change that moves the gene set away from that subspace: "no regression here"
is meaningful evidence, while a win here would be strong evidence.

Outputs: examples/results/adt_crossval_shadowed.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

CONFIGS = {
    "hvg": None,
    "hybrid": dict(balance_method="hybrid", blend_global=0.95, neighbor_contrast=0.0),
    "hybrid_nc1": dict(balance_method="hybrid", blend_global=0.95, neighbor_contrast=1.0),
    "score": dict(balance_method="score", neighbor_contrast=0.0),
}


def evaluate(adata, genes, seed):
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
        a,
        resolution=0.8,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    y_true = a.obs["cell_type"].astype(str)
    y_pred = a.obs["leiden"].astype(str)
    # macro F1 over true types: rare types are not averaged away as in ARI
    f1s = []
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in y_pred.unique():
            p = (y_pred == cl).to_numpy()
            tp = float(np.sum(t & p))
            if tp:
                prec, rec = tp / p.sum(), tp / t.sum()
                best = max(best, 2 * prec * rec / (prec + rec))
        f1s.append(best)
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1s)),
    }


def main():
    loaders = {
        "Cao": load_cao,
        "paul15": load_paul15,
        "pbmc3k_louvain": load_pbmc3k_labeled,
        "pancreas_smartseq2": lambda: load_scib_pancreas_one_tech("smartseq2"),
    }
    rows = []
    for dname, loader in loaders.items():
        print(f"\n==== {dname} ====", flush=True)
        try:
            adata = loader()
        except Exception as e:
            print(f"  LOAD FAIL {e}", flush=True)
            continue
        n_top = min(2000, adata.n_vars - 1)
        for cfg_name, cfg in CONFIGS.items():
            for seed in SEEDS:
                try:
                    a = adata.copy()
                    if cfg is None:
                        sc.pp.highly_variable_genes(
                            a, n_top_genes=n_top, flavor="seurat_v3", layer="counts"
                        )
                    else:
                        scf.pp.highly_variable_genes(
                            a,
                            n_top_genes=n_top,
                            flavor="seurat_v3",
                            layer="counts",
                            marker_mode="none",
                            random_state=seed,
                            **cfg,
                        )
                    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
                    res = evaluate(adata, genes, seed)
                    res.update({"dataset": dname, "config": cfg_name, "seed": seed})
                    rows.append(res)
                    print(
                        f"  {cfg_name:11s} seed={seed} ARI={res['ARI']:.3f} "
                        f"macroF1={res['macro_f1']:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {cfg_name} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_crossval_shadowed.csv", index=False)
    print("\n======== ARI (mean over seeds) ========")
    print(df.pivot_table(index="dataset", columns="config", values="ARI").round(3).to_string())
    print("\n======== macro F1 (mean over seeds) ========")
    print(df.pivot_table(index="dataset", columns="config", values="macro_f1").round(3).to_string())
    print("\n======== paired delta: hybrid_nc1 - hybrid ========")
    for metric in ("ARI", "macro_f1"):
        piv = df.pivot_table(index=["dataset", "seed"], columns="config", values=metric)
        d = (piv["hybrid_nc1"] - piv["hybrid"]).groupby("dataset").mean()
        print(f"  {metric}: {d.round(4).to_dict()}  overall={d.mean():+.4f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
