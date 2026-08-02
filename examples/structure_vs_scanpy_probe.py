#!/usr/bin/env python
"""Quick check: does `n_top_genes="structure"` (density-valley rule v5,
now wired into the public API) beat plain scanpy HVG@2000 on 3 small,
label-independent datasets?

Datasets chosen deliberately non-SHADOWED (today's own audit, §15 in
DEVELOPMENT_LOG.md: ground truth from a clustering pipeline makes "2000
wins" comparisons suspect) and the 3 smallest by cell count for a fast
turnaround: `crafted_base` (2609, barcode-multiplexed), `duo4_pbmc` and
`duo8_pbmc` (3994 each, FACS-sorted).

Arms:
  scanpy_2000  -- plain scanpy HVG, n_top_genes=2000 (the baseline every
                  other measurement in this project compares against)
  scfair_2000  -- scfair hybrid, fixed k=2000 (reference point)
  scfair_structure -- scfair, n_top_genes="structure" (the new rule)

5 seeds (not 2 -- today's own separability-hypothesis probe found a
strong-looking n=4 correlation evaporate at n=11; a 2-seed holdout read
is too thin to trust for a rule about to become a default candidate),
4-point resolution grid, mean-over-resolution ARI/macro_f1 (§5.19.4
protocol). `allocation_method`/`cap_allocation` left at their current
default (off, per the CHANGELOG's §5.31-32 update) -- not exercised here.

Outputs (examples/results/):
  structure_vs_scanpy_probe.csv
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "structure_vs_scanpy_probe.csv"

SEEDS = [0, 1, 2, 3, 4]
# CLI dataset args write to a per-invocation CSV (safe to run in parallel
# with the default full-panel run, which owns the shared CSV) -- merge
# fragments back into the shared CSV once all runs finish.
RES_GRID = [0.3, 0.5, 0.8, 1.2]
DATASETS = [
    "paul15",
    "pbmc3k_louvain",
    "pancreas_smartseq2",
    "duo4_pbmc",
    "duo8_pbmc",
    "duo4un_pbmc",
    "pbmc5k_adt29",
    "pbmc10k_adt14",
    "sln_208_mouse",
    "pbmc_seurat_v4_20k",
    "crafted_base",
]
ARMS = ["scanpy_2000", "scfair_2000", "scfair_structure"]


def select(adata, arm, seed):
    a = adata.copy()
    k = min(2000, a.n_vars - 1)
    meta = {"pred_k": np.nan}
    if arm == "scanpy_2000":
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
    elif arm == "scfair_2000":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            random_state=seed,
            diagnose=False,
        )
    else:  # scfair_structure
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="structure",
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            random_state=seed,
            diagnose=False,
        )
        meta["pred_k"] = a.uns["scfair"]["hvg"]["auto_n"]["n_top_selected"]
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    return genes, meta


def evaluate(a, genes, seed, rows, base):
    e = a.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    e = e[:, [g for g in genes if g in e.var_names]].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
    y_true = e.obs["cell_type"].astype(str)
    prev = y_true.value_counts(normalize=True)

    for res in RES_GRID:
        sc.tl.leiden(
            e, resolution=res, key_added="L", flavor="igraph", n_iterations=2, random_state=seed
        )
        y_pred = e.obs["L"].astype(str)
        f1s = {}
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
        rows.append(
            {
                **base,
                "resolution": res,
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
            }
        )


def main(which=None):
    csv_path = CSV if which is None else OUT / f"structure_vs_scanpy_probe_{'_'.join(which)}.csv"
    names = which or DATASETS
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["dataset"], r["seed"], r["arm"]) for r in rows}
    print(f"resuming: {len(done)} blocks done  (writing to {csv_path.name})", flush=True)

    for name in names:
        if all((name, s, arm) in done for s in SEEDS for arm in ARMS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        for seed in SEEDS:
            for arm in ARMS:
                if (name, seed, arm) in done:
                    continue
                t0 = time.time()
                genes, meta = select(a, arm, seed)
                base = {"dataset": name, "arm": arm, "seed": seed, "n_genes": len(genes), **meta}
                evaluate(a, genes, seed, rows, base)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(
                    f"  {arm:16s} seed={seed} n_genes={len(genes):4d} "
                    f"pred_k={meta['pred_k']}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        del a
    print(f"\nwrote {csv_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
