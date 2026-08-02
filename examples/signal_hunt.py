#!/usr/bin/env python
"""Does any label-free signal predict whether scFair helps? (§5.14)

§5.13 measured scFair's benefit per dataset: +8.0% (Cao) down to −1.4% (mouse
sln). If a signal computable *at gene-selection time, without labels* tracked
that, scFair could stand down where it cannot help — falling back to plain
global HVG, i.e. `blend_global=1.0`, which §5.10 verified reproduces scanpy
exactly.

This script does **not** implement any gate. It only asks whether such a signal
exists, because the answer is binary and decides whether the feature is
buildable at all. Fitting a threshold here would be meaningless: n=8 datasets,
and mis-falling-back costs ~8× more than mis-intervening (−8.0% vs −1.1%).

Candidate signals, all computed from the intermediate clustering that scFair
already builds, with the shipped defaults (resolution=0.5, min_cluster_size=30):

  stability     Leiden run at two seeds, ARI between them. Directly measures
                whether the input driving the re-ranking is trustworthy — the
                failure mode §5.9/§5.10 diagnosed.
  silhouette    separation of the intermediate clusters in PCA space; the
                literal reading of "already well separated".
  gini/entropy  cluster-size imbalance; §5.5 observed the gain concentrates on
                imbalanced data.
  n_clusters    how much structure there is to be specific about.
  swap_frac     fraction of the 2000 genes hybrid actually changes. If scFair
                barely moves the set it cannot help or hurt much.

Outputs: examples/results/signal_hunt.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import adjusted_rand_score, silhouette_score

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_comparison import LOADERS, SHADOWED  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
N_TOP = 2000
RESOLUTION = 0.5  # shipped default
MIN_CLUSTER_SIZE = 30


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def intermediate_view(adata, genes, seed: int):
    """Reproduce scFair's intermediate clustering step and keep the embedding."""
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    a = a[:, [g for g in genes if g in a.var_names]].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    n_pcs = min(30, a.n_vars - 1, a.n_obs - 1)
    sc.pp.pca(a, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=n_pcs, random_state=seed)
    sc.tl.leiden(
        a,
        resolution=RESOLUTION,
        key_added="cl",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    return a.obsm["X_pca"][:, : min(20, n_pcs)], a.obs["cl"].astype(str)


def signals_for(adata) -> dict:
    a = adata.copy()
    sc.pp.highly_variable_genes(
        a, n_top_genes=min(N_TOP, a.n_vars - 1), flavor="seurat_v3", layer="counts"
    )
    global_genes = set(a.var_names[a.var["highly_variable"]].astype(str))

    b = adata.copy()
    scf.pp.highly_variable_genes(
        b,
        n_top_genes=min(N_TOP, b.n_vars - 1),
        flavor="seurat_v3",
        layer="counts",
        marker_mode="none",
        random_state=0,
    )
    scfair_genes = set(b.var_names[b.var["highly_variable"]].astype(str))
    swap_frac = len(scfair_genes - global_genes) / max(len(global_genes), 1)

    X0, l0 = intermediate_view(adata, sorted(global_genes), seed=0)
    _, l1 = intermediate_view(adata, sorted(global_genes), seed=1)

    sizes = l0.value_counts()
    valid = sizes[sizes >= MIN_CLUSTER_SIZE]
    sil = (
        float(silhouette_score(X0, l0, sample_size=min(3000, len(l0)), random_state=0))
        if l0.nunique() > 1
        else np.nan
    )
    p = (valid / valid.sum()).to_numpy()
    return {
        "stability": float(adjusted_rand_score(l0, l1)),
        "silhouette": sil,
        "gini": gini(valid.to_numpy()),
        "entropy": float(-(p * np.log(p)).sum()),
        "n_clusters": int(len(valid)),
        "min_cluster_frac": float(valid.min() / valid.sum()) if len(valid) else np.nan,
        "swap_frac": float(swap_frac),
    }


def main():
    prev = pd.read_csv(OUT / "final_comparison.csv").dropna(subset=["ARI"])
    piv_a = prev.pivot_table(index="dataset", columns="arm", values="ARI")
    piv_m = prev.pivot_table(index="dataset", columns="arm", values="macro_f1")
    outcome = pd.DataFrame(
        {
            "d_ARI": piv_a["scfair2000"] - piv_a["hvg2000"],
            "d_ARI_rel": 100 * (piv_a["scfair2000"] - piv_a["hvg2000"]) / piv_a["hvg2000"],
            "d_macro": piv_m["scfair2000"] - piv_m["hvg2000"],
        }
    )

    rows = []
    for dname, loader in LOADERS.items():
        print(f"  {dname} ...", flush=True)
        try:
            adata = loader()
            s = signals_for(adata)
            s.update({"dataset": dname, "shadowed": dname in SHADOWED})
            rows.append(s)
            print(f"    {s}", flush=True)
        except Exception as e:
            print(f"    FAIL {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows).set_index("dataset").join(outcome)
    df.to_csv(OUT / "signal_hunt.csv")
    print("\n======== signals vs measured benefit ========")
    print(df.round(3).to_string())

    sig_cols = [
        "stability",
        "silhouette",
        "gini",
        "entropy",
        "n_clusters",
        "min_cluster_frac",
        "swap_frac",
    ]
    print("\n======== correlation with benefit (n=8) ========")
    print(f"{'signal':17s} {'spearman(dARI)':>15s} {'p':>7s} {'spearman(dmacro)':>17s} {'p':>7s}")
    for c in sig_cols:
        sub = df[[c, "d_ARI", "d_macro"]].dropna()
        if len(sub) < 4:
            continue
        r1, p1 = stats.spearmanr(sub[c], sub["d_ARI"])
        r2, p2 = stats.spearmanr(sub[c], sub["d_macro"])
        star = "  <==" if min(p1, p2) < 0.05 else ""
        print(f"{c:17s} {r1:15.3f} {p1:7.3f} {r2:17.3f} {p2:7.3f}{star}")
    print("\nDONE")


if __name__ == "__main__":
    main()
