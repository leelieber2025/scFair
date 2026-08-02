#!/usr/bin/env python
"""Evaluate ARI at the *exact* n_top_genes auto_n picked, not a grid-snapped
neighbour.

Why this exists: `docs/PAPER_gold15_best_k_vs_auto.csv` compared auto_n's
picked k (a continuous value like 2995, 1833, 3177...) against `best_k`,
which can only be one of the ~500-spaced k-sweep grid points
({200,300,500,1000,1500,2000,3000,4000} or a 6-point subset). That comparison
is fair for the 9/15 GOLD datasets where auto happened to land exactly on a
grid point (2000 or 1000) — we already have real ARI there. It is NOT fair
for the 6 where auto's pick falls off-grid: duo4_pbmc (2995), duo4un_pbmc
(3177), duo8_pbmc (4395), crafted_base_3cellline (1333), tm_limb_muscle_gold
(1833), tm_brain_myeloid_vs_nonmyeloid_gold (1333) — for those the reported
"gap to best_k" was really measuring grid quantization, not auto's quality.

This script adds one extra k-sweep point, at auto's exact picked value, for
exactly those 6, under the *same* protocol/loader/seeds each dataset's
best_k came from (auto_n_populations.py's `evaluate()` for the 4 calibration
sets, this repo's gold15_ksweep_missing.py loaders for the 2 new-panel sets)
— so the new number is directly comparable to that dataset's own best_k row,
no cross-protocol confound.

Usage: python examples/gold15_auto_k_exact_eval.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import auto_n_holdout_eval as hbase  # noqa: E402
import auto_n_populations as cbase  # noqa: E402 (imports umap3d_smoke.LOADERS internally)
import gold15_ksweep_missing as newbase  # noqa: E402

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT_CSV = OUT / "gold15_auto_k_exact.csv"

# off-grid auto_base_k, rounded to nearest int, from docs/PAPER_gold15_best_k_vs_auto.csv
CALIBRATION_TARGETS = {  # protocol: auto_n_populations.py (3 seeds, RES_GRID, cbase.LOADERS)
    "duo4_pbmc": 2995,
    "duo4un_pbmc": 3177,
    "duo8_pbmc": 4395,
    "crafted_base": 1333,  # loader key differs from the GOLD-15 name (crafted_base_3cellline)
}
NEWPANEL_TARGETS = {  # protocol: gold15_ksweep_missing.py (2 seeds, newbase.LOADERS)
    "tm_limb_muscle_gold": 1833,
    "tm_brain_myeloid_vs_nonmyeloid_gold": 1333,
}


def run_calibration(name: str, k: int) -> pd.DataFrame:
    a = cbase.LOADERS[name]()
    rows = []
    for seed in cbase.SEEDS:
        sel = a.copy()
        scf.pp.highly_variable_genes(sel, n_top_genes=min(k, sel.n_vars - 1), random_state=seed)
        genes = sel.var_names[sel.var["highly_variable"]].astype(str).tolist()
        base = {"dataset": name, "k": k, "seed": seed, "n_genes": len(genes)}
        cbase.evaluate(a, genes, seed, rows, base)
        del sel
    return pd.DataFrame(rows)


def run_newpanel(name: str, k: int) -> pd.DataFrame:
    a = newbase.LOADERS[name]()
    rows = []
    for seed in hbase.SEEDS:
        ad_ = a.copy()
        scf.pp.highly_variable_genes(
            ad_,
            n_top_genes=k,
            balance_method="hybrid",
            resolution=hbase.INT_RES,
            allocation_method="none",
            random_state=seed,
            diagnose=False,
        )
        genes = list(ad_.var_names[ad_.var["highly_variable"]])
        ari = hbase.evaluate_genes(a, genes, seed)
        rows.append({"dataset": name, "k": k, "seed": seed, "ARI": ari, "n_genes": len(genes)})
    return pd.DataFrame(rows)


def main():
    all_rows = []

    for name, k in CALIBRATION_TARGETS.items():
        print(f"\n### calibration protocol: {name} @ k={k}", flush=True)
        df = run_calibration(name, k)
        mean_ari = df["ARI"].mean()
        print(f"  mean ARI over {len(df)} (seed x res) rows = {mean_ari:.4f}", flush=True)
        all_rows.append(
            {
                "dataset": "crafted_base_3cellline" if name == "crafted_base" else name,
                "auto_base_k_exact": k,
                "ARI_at_auto_k_exact": float(mean_ari),
                "protocol": "auto_n_populations (3 seeds x 4 res)",
            }
        )

    for name, k in NEWPANEL_TARGETS.items():
        print(f"\n### new-panel protocol: {name} @ k={k}", flush=True)
        df = run_newpanel(name, k)
        mean_ari = df["ARI"].mean()
        print(f"  mean ARI over {len(df)} seeds = {mean_ari:.4f}", flush=True)
        all_rows.append(
            {
                "dataset": name,
                "auto_base_k_exact": k,
                "ARI_at_auto_k_exact": float(mean_ari),
                "protocol": "gold15_ksweep_missing (2 seeds x 4 res)",
            }
        )

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False)
    print("\n=== exact auto_base_k ARI (off-grid GOLD-15 datasets) ===", flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"\nwrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
