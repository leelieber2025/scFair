#!/usr/bin/env python
"""Does a second round of (cluster -> score -> select) buy anything?

Question (§5.17)
----------------
`hybrid` runs one intermediate clustering, scores specificity against it, and
selects. The obvious extension is to iterate: re-cluster on the genes just
selected, re-score, re-select, repeat. Two things have to be true for that to
be worth shipping, and they are separable:

  (a) it must *move* -- if round 2 returns round 1's genes, it is 20s of
      nothing;
  (b) where it moves, it must move toward the gold labels, not merely toward
      its own fixed point.

(b) is the real risk. Genes are chosen *because* they separate round-1
clusters, so clustering on them reproduces round-1's partition with more
confidence whether or not that partition was right. Convergence proves
self-consistency, not correctness -- so this script measures both, and reports
them separately.

Note the pool asymmetry that makes iteration well-defined at all: the
*clustering* pool narrows to the previous selection, while the *candidate* pool
stays global. Re-selecting k genes from a k-gene pool is the identity.

Protocol
--------
adt14 protein gold labels, k=1000 (the §5.16 optimum for this panel), 5 seeds,
3 rounds. Evaluation follows §5.16: a resolution sweep, reported at peak and
at the legacy single point res=0.8, never at res=0.8 alone.

Outputs (examples/results/):
  iterate_rounds.csv         per (seed, round, resolution) metrics
  iterate_rounds_genes.csv   per (seed, round) gene-set / partition movement
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sorted_gold_panel import load_dataset, per_population_f1  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "iterate_rounds.csv"
GCSV = OUT / "iterate_rounds_genes.csv"

RES_GRID = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
K = 1000
N_ROUNDS = 3
SEEDS = range(5)


def evaluate(a0, genes, info, seed, k, rnd, rows):
    a = a0.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, [g for g in genes if g in a0.var_names]].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    keep = (
        a.obs[info["conf_col"]].to_numpy(dtype=bool)
        if info["conf_col"]
        else np.ones(a.n_obs, dtype=bool)
    )
    y_true = a.obs[info["label_col"]].astype(str)[keep]
    prev = y_true.value_counts(normalize=True)
    smallest = prev.idxmin()

    for res in RES_GRID:
        sc.tl.leiden(
            a,
            resolution=res,
            key_added="L",
            flavor="igraph",
            n_iterations=2,
            random_state=seed,
        )
        y_pred = a.obs["L"].astype(str)[keep]
        f1 = per_population_f1(y_true, y_pred)
        rows.append(
            dict(
                dataset="adt14",
                k=k,
                round=rnd,
                seed=seed,
                resolution=res,
                n_genes=a.n_vars,
                n_leiden=int(y_pred.nunique()),
                ARI=float(adjusted_rand_score(y_true, y_pred)),
                NMI=float(normalized_mutual_info_score(y_true, y_pred)),
                macro_f1=float(np.mean(list(f1.values()))),
                min_pop_f1=float(f1[str(smallest)]),
                min_pop=str(smallest),
            )
        )


def main() -> None:
    a0, info = load_dataset("adt14")
    print(f"adt14: {a0.n_obs} cells x {a0.n_vars} genes", flush=True)

    rows: list[dict] = []
    grows: list[dict] = []

    for seed in SEEDS:
        prev_genes: list[str] | None = None
        prev_labels: pd.Series | None = None

        # round 0 = plain scanpy HVG, the baseline both rounds are judged against
        a = a0.copy()
        sc.pp.highly_variable_genes(a, n_top_genes=K, flavor="seurat_v3", layer="counts")
        hvg_genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
        evaluate(a0, hvg_genes, info, seed, K, 0, rows)

        for rnd in range(1, N_ROUNDS + 1):
            t0 = time.time()
            a = a0.copy()
            kw = dict(
                n_top_genes=K,
                flavor="seurat_v3",
                layer="counts",
                marker_mode="none",
                balance_method="hybrid",
                blend_global=0.95,
                resolution=0.5,
                random_state=seed,
            )
            if prev_genes is not None:
                kw["cluster_genes"] = prev_genes
            scf.pp.highly_variable_genes(a, **kw)
            genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
            labels = a.obs["scfair_hvg_clusters"].astype(str)
            diag = a.uns["scfair"]["hvg"]["clustering"]

            grows.append(
                dict(
                    seed=seed,
                    round=rnd,
                    n_genes=len(genes),
                    n_genes_clustered=diag["n_genes_clustered"],
                    n_clusters_total=diag["n_clusters_total"],
                    n_clusters_kept=diag["n_clusters_kept"],
                    # movement vs the previous round
                    overlap_prev=(len(set(genes) & set(prev_genes)) if prev_genes else np.nan),
                    ari_partition_prev=(
                        float(adjusted_rand_score(prev_labels, labels))
                        if prev_labels is not None
                        else np.nan
                    ),
                    # movement vs plain HVG, for scale
                    overlap_hvg=len(set(genes) & set(hvg_genes)),
                    sel_seconds=round(time.time() - t0, 1),
                )
            )
            evaluate(a0, genes, info, seed, K, rnd, rows)
            print(
                f"seed={seed} round={rnd} genes={len(genes)} "
                f"overlap_prev={grows[-1]['overlap_prev']} "
                f"ARI(part,prev)={grows[-1]['ari_partition_prev']} "
                f"({grows[-1]['sel_seconds']}s)",
                flush=True,
            )
            prev_genes, prev_labels = genes, labels

        pd.DataFrame(rows).to_csv(CSV, index=False)
        pd.DataFrame(grows).to_csv(GCSV, index=False)

    print("ITERATE ROUNDS DONE", flush=True)


if __name__ == "__main__":
    main()
