#!/usr/bin/env python
"""Fill in the missing k-sweeps for the 15-dataset GOLD panel (docs/gold.md).

Same protocol as auto_n_holdout_eval.py (K_GRID, RES_GRID, 2 seeds, hybrid
balance_method, allocation_method="none"): reused directly from that module
so results are comparable to the existing auto_n_holdout_ksweep*.csv family.

Datasets covered here are exactly the 4 GOLD-15 entries with no prior
empirical ARI-vs-k sweep on record (checked against auto_n_holdout_ksweep*.csv
and docs/DEVELOPMENT_LOG.md §15 — both only ever swept PBMC/ADT/Duo/crafted/
lung/seurat_v4/cellxgene/gbm/cbmc8k/pbmc_cite_gse100866, never these four):

  - tm_limb_muscle_gold
  - tm_brain_myeloid_vs_nonmyeloid_gold
  - villani_dc_mono
  - zheng_facs9_gold (subsampled to 20k, same treatment as lung_atlas/seurat_v4)

Usage:
  python examples/gold15_ksweep_missing.py            # all 4
  python examples/gold15_ksweep_missing.py NAME...     # subset
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import auto_n_holdout_eval as base  # noqa: E402

warnings_ok = True

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SWEEP_CSV = OUT / "gold15_missing_ksweep.csv"
FEAT_CSV = OUT / "gold15_missing_features.csv"
SUM_CSV = OUT / "gold15_missing_summary.csv"


def _load(fname, label_col="cell_type", drop=(), subsample=None, seed=0):
    a = ad.read_h5ad(DATA / fname)
    a = base._ensure_counts(a)
    a.obs["cell_type"] = a.obs[label_col].astype(str)
    mask = a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")
    for d in drop:
        mask &= a.obs["cell_type"] != d
    a = a[mask].copy()
    if subsample and a.n_obs > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(a.n_obs, size=subsample, replace=False)
        a = a[idx].copy()
    return a


LOADERS = {
    "tm_limb_muscle_gold": lambda: _load("tm_limb_muscle_gold.h5ad"),
    "tm_brain_myeloid_vs_nonmyeloid_gold": lambda: _load(
        "tm_brain_myeloid_vs_nonmyeloid_gold.h5ad"
    ),
    "villani_dc_mono": lambda: _load("villani_dc_mono_gold.h5ad"),
    "zheng_facs9_gold": lambda: _load("zheng_facs9_gold.h5ad", subsample=20000),
}


def main(names):
    # redirect the reused module's incremental-write targets so we never
    # clobber the existing auto_n_holdout_ksweep*.csv files
    base.SWEEP_CSV = SWEEP_CSV

    all_sweep, all_feat = [], []
    for name in names:
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        print(f"shape={a.shape} n_types={a.obs['cell_type'].nunique()}", flush=True)
        sw, ft = base.run_ksweep(name, a)
        all_sweep.append(sw)
        all_feat.append(ft)
        del a

    sweep = pd.concat(all_sweep, ignore_index=True)
    feats = pd.concat(all_feat, ignore_index=True)
    sweep.to_csv(SWEEP_CSV, index=False)
    feats.to_csv(FEAT_CSV, index=False)

    g = sweep.groupby(["dataset", "k"], as_index=False)["ARI"].mean()
    summaries = []
    for name, sub in g.groupby("dataset"):
        best_row = sub.loc[sub["ARI"].idxmax()]
        a2000 = float(sub.loc[sub.k == 2000, "ARI"].iloc[0])
        pred_k = int(feats.loc[feats.dataset == name, "pred_k"].mode().iloc[0])
        a_pred = (
            float(sub.loc[sub.k == pred_k, "ARI"].iloc[0]) if (sub.k == pred_k).any() else np.nan
        )
        if not np.isfinite(a_pred):
            kk = int(sub.k.iloc[(sub.k - pred_k).abs().argmin()])
            a_pred = float(sub.loc[sub.k == kk, "ARI"].iloc[0])
            pred_k_used = kk
        else:
            pred_k_used = pred_k
        summaries.append(
            {
                "dataset": name,
                "pred_k": pred_k,
                "pred_k_used": pred_k_used,
                "best_k": int(best_row.k),
                "ARI_pred": a_pred,
                "ARI_2000": a2000,
                "ARI_best": float(best_row.ARI),
                "d_vs_2000": a_pred - a2000,
                "gap_to_oracle": float(best_row.ARI) - a_pred,
                "n_types": int(LOADERS[name]().obs["cell_type"].nunique()),
            }
        )
    out = pd.DataFrame(summaries)
    out.to_csv(SUM_CSV, index=False)
    print("\n=== SUMMARY (missing GOLD-15 datasets) ===", flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"\nwrote {SWEEP_CSV}\nwrote {FEAT_CSV}\nwrote {SUM_CSV}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(LOADERS)
    main(names)
