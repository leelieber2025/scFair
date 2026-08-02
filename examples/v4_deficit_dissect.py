#!/usr/bin/env python
"""Why does scFair lose on pbmc_seurat_v4_20k? A within-dataset dissection.

§5.22. The deficit is confirmed and robust: scFair loses to plain scanpy on the
largest non-circular panel at three of four k values (§5.21 probe), it is not a
k artifact, and it is not explained by the quality of the intermediate partition
(ARI to truth 0.485, mid-pack, higher than the biggest *winner*).

Six cross-dataset predictors have now failed on this question at n=7-8, where
nothing below |r|=0.754 can even reach p<0.05. So this script does not add a
dataset. It stays inside this one and varies seeds, following §5.18's design,
which is the only approach that has produced a clean answer so far.

The decisive split, asked first
-------------------------------
Is the loss a **subspace** effect or a **clustering** effect?

  - kNN purity is a property of the embedding alone -- no Leiden involved. If
    hybrid's genes give worse purity, the gene set is making a worse space.
  - If purity is equal or better while F1 is worse, the space is fine and the
    damage happens at the clustering step -- which is exactly duo4's mechanism
    (§5.16.3: hybrid resolved sub-structure, and a fixed resolution punished it).

These call for different fixes, so the answer is worth having before anything
else is measured.

Then, conditional on that:
  - per-population F1 and purity deltas, to see whether the loss is spread out
    or concentrated in a few cell types;
  - what hybrid actually swaps: global variability rank, detection rate, mean
    expression of the genes it adds vs the ones it drops.

Outputs (examples/results/):
  v4_dissect_populations.csv   per (seed, arm, population)
  v4_dissect_genes.csv         per (seed) gene-swap characterisation
  v4_dissect_summary.csv       per (seed) embedding-level summary
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cluster_pool_panel import LOADERS, select  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
POP_CSV = OUT / "v4_dissect_populations.csv"
GENE_CSV = OUT / "v4_dissect_genes.csv"
SUM_CSV = OUT / "v4_dissect_summary.csv"

TARGET = "pbmc_seurat_v4_20k"
SEEDS = list(range(10))
ARMS = ["hvg2000", "scfair2000"]
RES_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_NEIGHBORS = 15


def knn_purity(conn, labels: np.ndarray) -> np.ndarray:
    """Fraction of each cell's neighbours sharing its label.

    Uses the connectivity graph scanpy already built, so this costs nothing
    extra and -- crucially -- involves no clustering.
    """
    A = conn.tocsr()
    out = np.zeros(A.shape[0], dtype=float)
    for i in range(A.shape[0]):
        nb = A.indices[A.indptr[i] : A.indptr[i + 1]]
        if nb.size:
            out[i] = float(np.mean(labels[nb] == labels[i]))
    return out


def per_population_f1(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    out = {}
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in y_pred.unique():
            m = (y_pred == cl).to_numpy()
            tp = float(np.sum(t & m))
            if tp:
                pr, rc = tp / m.sum(), tp / t.sum()
                best = max(best, 2 * pr * rc / (pr + rc))
        out[pop] = best
    return out


def main() -> None:
    pop_rows = pd.read_csv(POP_CSV).to_dict("records") if POP_CSV.exists() else []
    gene_rows = pd.read_csv(GENE_CSV).to_dict("records") if GENE_CSV.exists() else []
    sum_rows = pd.read_csv(SUM_CSV).to_dict("records") if SUM_CSV.exists() else []
    done = {(r["seed"], r["arm"]) for r in sum_rows}

    adata = LOADERS[TARGET]()
    print(f"{TARGET}: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

    conf = (
        adata.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in adata.obs.columns
        else np.ones(adata.n_obs, dtype=bool)
    )
    y_true_all = adata.obs["cell_type"].astype(str)
    prev = y_true_all[conf].value_counts(normalize=True)

    # gene-level context computed once: global variability rank and how broadly
    # each gene is detected, so a swap can be described rather than just counted
    counts = adata.layers["counts"]
    counts = counts.tocsc() if sp.issparse(counts) else counts
    det = np.asarray((counts > 0).sum(axis=0)).ravel() / adata.n_obs
    mean_expr = np.asarray(counts.mean(axis=0)).ravel()
    ref = adata.copy()
    sc.pp.highly_variable_genes(
        ref, n_top_genes=adata.n_vars - 1, flavor="seurat_v3", layer="counts"
    )
    grank = pd.Series(ref.var["highly_variable_rank"].to_numpy(), index=adata.var_names)
    del ref
    gene_ctx = pd.DataFrame(
        {"detection": det, "mean_expr": mean_expr, "global_rank": grank.to_numpy()},
        index=adata.var_names.astype(str),
    )

    genes_by_seed: dict[int, dict[str, list[str]]] = {}

    for seed in SEEDS:
        for arm in ARMS:
            if (seed, arm) in done:
                continue
            t0 = time.time()
            genes, _ = select(adata, arm, seed)
            genes_by_seed.setdefault(seed, {})[arm] = genes

            a = adata.copy()
            a.X = a.layers["counts"].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            a = a[:, [g for g in genes if g in adata.var_names]].copy()
            sc.pp.scale(a, max_value=10)
            n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
            sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
            sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, n_pcs=min(30, n_comps), random_state=seed)

            lab = y_true_all.to_numpy()
            pur = knn_purity(a.obsp["connectivities"], lab)

            # F1 per population at each resolution; keep the per-population best
            # over the grid so a single bad resolution cannot masquerade as a
            # gene-selection effect (§5.16)
            y_true = y_true_all[conf]
            best_f1: dict[str, float] = {}
            n_leiden = {}
            for res in RES_GRID:
                sc.tl.leiden(
                    a,
                    resolution=res,
                    key_added="L",
                    flavor="igraph",
                    n_iterations=2,
                    random_state=seed,
                )
                y_pred = a.obs["L"].astype(str)[conf]
                n_leiden[res] = int(y_pred.nunique())
                for pop, f1 in per_population_f1(y_true, y_pred).items():
                    best_f1[pop] = max(best_f1.get(pop, 0.0), f1)

            for pop in y_true.unique():
                m = (y_true_all == pop).to_numpy() & conf
                pop_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "population": pop,
                        "prevalence": float(prev.get(pop, np.nan)),
                        "n_cells": int(m.sum()),
                        "best_f1": float(best_f1[pop]),
                        "knn_purity": float(pur[m].mean()),
                    }
                )

            sum_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "n_genes": a.n_vars,
                    "knn_purity_overall": float(pur[conf].mean()),
                    "macro_best_f1": float(np.mean(list(best_f1.values()))),
                    **{f"n_leiden_res{r}": n_leiden[r] for r in RES_GRID},
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(
                f"  seed={seed} {arm:12s} purity={sum_rows[-1]['knn_purity_overall']:.4f} "
                f"macroF1={sum_rows[-1]['macro_best_f1']:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            pd.DataFrame(pop_rows).to_csv(POP_CSV, index=False)
            pd.DataFrame(sum_rows).to_csv(SUM_CSV, index=False)

        # characterise what hybrid swapped, once both arms for this seed exist
        if seed in genes_by_seed and set(ARMS) <= set(genes_by_seed[seed]):
            base = set(genes_by_seed[seed]["hvg2000"])
            new = set(genes_by_seed[seed]["scfair2000"])
            added, dropped = sorted(new - base), sorted(base - new)
            for tag, gs in [("added", added), ("dropped", dropped)]:
                sub = gene_ctx.loc[[g for g in gs if g in gene_ctx.index]]
                gene_rows.append(
                    {
                        "seed": seed,
                        "direction": tag,
                        "n": len(sub),
                        "median_global_rank": float(sub.global_rank.median()),
                        "median_detection": float(sub.detection.median()),
                        "median_mean_expr": float(sub.mean_expr.median()),
                    }
                )
            pd.DataFrame(gene_rows).to_csv(GENE_CSV, index=False)
            print(f"  seed={seed} swapped in {len(added)} / out {len(dropped)}", flush=True)

    print("V4 DISSECT DONE", flush=True)


if __name__ == "__main__":
    main()
