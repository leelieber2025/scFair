#!/usr/bin/env python
"""Cross-validate the intermediate-clustering `resolution` default (§5.10 Open).

The ADT sweep hinted that `resolution=0.5` is Pareto-better than the current
default of 1.0 (macro-F1 0.749 vs 0.738, ncMono 0.901 vs 0.718) — but that was
3 seeds on one dataset, and §5.10 established that ncMono F1 is bimodal, so a
3-seed mean mostly counts how many seeds landed well. This script:

  A. re-tests resolution on ADT at **5 seeds**, including whether it stacks
     with `neighbor_contrast` (both target the same failure mode);
  B. re-runs it on the shadowed labelled panel, the overfitting check that
     §5.10 requires before any default changes.

`resolution` is an existing public default, so changing it is a bigger
commitment than adding an opt-in knob: it silently changes every current
user's results. The bar is correspondingly higher.

Outputs: examples/results/adt_resolution_{adt,shadowed}.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy import stats

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adt_crossval_shadowed import evaluate  # noqa: E402
from adt_gold_benchmark import cluster_metrics, load_labeled  # noqa: E402
from p3_public_validation import (  # noqa: E402
    load_cao,
    load_paul15,
    load_pbmc3k_labeled,
    load_scib_pancreas_one_tech,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
ADT_SEEDS = [0, 1, 2, 3, 4]
PANEL_SEEDS = [0, 1, 2]

CONFIGS = {
    "res1.0": dict(resolution=1.0, neighbor_contrast=0.0),
    "res0.5": dict(resolution=0.5, neighbor_contrast=0.0),
    "res0.75": dict(resolution=0.75, neighbor_contrast=0.0),
    "res0.5+nc1": dict(resolution=0.5, neighbor_contrast=1.0),
    "res1.0+nc1": dict(resolution=1.0, neighbor_contrast=1.0),
}


def select(adata, seed: int, **cfg):
    a = adata.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=min(2000, a.n_vars - 1),
        flavor="seurat_v3",
        layer="counts",
        marker_mode="none",
        balance_method="hybrid",
        blend_global=0.95,
        random_state=seed,
        **cfg,
    )
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    meta = a.uns.get("scfair", {}).get("hvg", {})
    return genes, int(meta.get("n_clusters_used", 0))


def part_a(adata):
    print("\n######## A. ADT gold standard, 5 seeds ########")
    rows = []
    for name, cfg in CONFIGS.items():
        for seed in ADT_SEEDS:
            genes, n_cl = select(adata, seed, **cfg)
            res = cluster_metrics(adata, genes, seed=seed)
            res.update({"config": name, "seed": seed, "n_clusters_used": n_cl})
            rows.append(res)
            print(
                f"  {name:11s} seed={seed} clusters={n_cl:2d} ARI={res['ARI']:.3f} "
                f"macroF1={res['macro_f1']:.3f} ncMono={res['f1_Mono_nonclassical']:.3f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_resolution_adt.csv", index=False)

    cols = ["ARI", "macro_f1", "rare_f1_mean", "f1_Mono_nonclassical", "n_clusters_used"]
    print("\n--- mean over 5 seeds ---")
    print(df.groupby("config")[cols].mean().round(3).to_string())
    print("\n--- ncMono F1: std and worst seed (the failure mode) ---")
    g = df.groupby("config")["f1_Mono_nonclassical"]
    print(
        pd.DataFrame(
            {
                "mean": g.mean().round(3),
                "std": g.std().round(3),
                "worst": g.min().round(3),
                "collapsed": df.assign(c=df["f1_Mono_nonclassical"] < 0.6)
                .groupby("config")["c"]
                .sum(),
            }
        ).to_string()
    )

    print("\n--- paired vs res1.0 (current default) ---")
    base = df[df.config == "res1.0"].set_index("seed")
    for name in CONFIGS:
        if name == "res1.0":
            continue
        sub = df[df.config == name].set_index("seed")
        bits = []
        for m in ("ARI", "macro_f1", "rare_f1_mean", "f1_Mono_nonclassical"):
            d = (sub[m] - base[m]).dropna()
            t, p = stats.ttest_1samp(d, 0)
            bits.append(f"{m}={d.mean():+.4f}(p={p:.3f},{(d > 0).sum()}/{len(d)})")
        print(f"  {name:11s} " + "  ".join(bits))
    return df


def part_b():
    print("\n######## B. shadowed labelled panel ########")
    loaders = {
        "Cao": load_cao,
        "paul15": load_paul15,
        "pbmc3k_louvain": load_pbmc3k_labeled,
        "pancreas_smartseq2": lambda: load_scib_pancreas_one_tech("smartseq2"),
    }
    panel_cfgs = {k: CONFIGS[k] for k in ("res1.0", "res0.5")}
    rows = []
    for dname, loader in loaders.items():
        print(f"\n==== {dname} ====", flush=True)
        try:
            adata = loader()
        except Exception as e:
            print(f"  LOAD FAIL {e}", flush=True)
            continue
        for name, cfg in panel_cfgs.items():
            for seed in PANEL_SEEDS:
                try:
                    genes, n_cl = select(adata, seed, **cfg)
                    res = evaluate(adata, genes, seed)
                    res.update(
                        {"dataset": dname, "config": name, "seed": seed, "n_clusters_used": n_cl}
                    )
                    rows.append(res)
                    print(
                        f"  {name:8s} seed={seed} clusters={n_cl:2d} "
                        f"ARI={res['ARI']:.3f} macroF1={res['macro_f1']:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {name} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_resolution_shadowed.csv", index=False)
    for metric in ("ARI", "macro_f1"):
        print(f"\n--- {metric} (mean over seeds) ---")
        print(df.pivot_table(index="dataset", columns="config", values=metric).round(3).to_string())
        piv = df.pivot_table(index=["dataset", "seed"], columns="config", values=metric)
        d = (piv["res0.5"] - piv["res1.0"]).groupby("dataset").mean()
        print(f"  delta res0.5-res1.0: {d.round(4).to_dict()}  overall={d.mean():+.4f}")
    return df


def main():
    adata = load_labeled()
    part_a(adata)
    part_b()
    print("\nDONE")


if __name__ == "__main__":
    main()
