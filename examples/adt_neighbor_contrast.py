#!/usr/bin/env python
"""Test the nearest-neighbour specificity contrast on the ADT gold standard.

Hypothesis (from §5.9): hybrid loses the classical/non-classical monocyte
boundary because cluster-vs-*rest* cannot see it — for a 1% subset the "rest"
is dominated by distant lineages, so every gene of the parent lineage scores
high and the genes separating the subset from its neighbour earn no credit.

``neighbor_contrast`` moves the peak-specificity term from vs-rest to
vs-nearest-cluster. If the hypothesis holds, ncMono F1 should recover toward
plain HVG's ~0.9 without giving up hybrid's ARI.

5 seeds throughout: the failure is seed-metastable (ncMono F1 is bimodal at
~0.35 / ~0.9), so 3-seed means mostly count how many seeds landed well.

Outputs: examples/results/adt_neighbor_contrast.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import scanpy as sc

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adt_gold_benchmark import cluster_metrics, load_labeled  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
SEEDS = [0, 1, 2, 3, 4]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    adata = load_labeled()
    rows = []
    for method in ("hybrid", "score"):
        for nc in GRID:
            for seed in SEEDS:
                a = adata.copy()
                kw = dict(
                    n_top_genes=2000,
                    flavor="seurat_v3",
                    layer="counts",
                    marker_mode="none",
                    balance_method=method,
                    neighbor_contrast=nc,
                    random_state=seed,
                )
                if method == "hybrid":
                    kw["blend_global"] = 0.95
                scf.pp.highly_variable_genes(a, **kw)
                genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
                res = cluster_metrics(adata, genes, seed=seed)
                res.update({"method": method, "neighbor_contrast": nc, "seed": seed})
                rows.append(res)
                print(
                    f"  {method:6s} nc={nc:<4} seed={seed} ARI={res['ARI']:.3f} "
                    f"macroF1={res['macro_f1']:.3f} rareF1={res['rare_f1_mean']:.3f} "
                    f"ncMono={res['f1_Mono_nonclassical']:.3f} Treg={res['f1_Treg']:.3f}",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_neighbor_contrast.csv", index=False)

    cols = ["ARI", "macro_f1", "rare_f1_mean", "f1_Mono_nonclassical", "f1_Treg"]
    print("\n======== mean over 5 seeds ========")
    print(df.groupby(["method", "neighbor_contrast"])[cols].mean().round(3).to_string())
    print("\n======== ncMono collapse count (F1 < 0.6) ========")
    coll = df.assign(collapsed=df["f1_Mono_nonclassical"] < 0.6)
    print(
        coll.groupby(["method", "neighbor_contrast"])["collapsed"].agg(["sum", "count"]).to_string()
    )
    print("\nDONE")


if __name__ == "__main__":
    main()
