#!/usr/bin/env python
"""Parameter sweeps against the ADT (protein) gold standard — see §5.9.

Purpose: the ADT labels are the first *non-circular* target scFair has, so the
knobs that were frozen on shadowed evidence can finally be checked. Sweeps
here are **development-set** experiments; anything that wins must then be
re-validated on the shadowed multi-dataset panel (p1/p2/p3) before it becomes
a default, because a single dataset is trivially overfittable.

Sweeps
------
k        oracle n_top_genes for hybrid (answers: is auto's k right?)
res      intermediate Leiden resolution (never swept — §7 open question 9)
mcs      intermediate min_cluster_size
blend    blend_global (frozen at 0.95 on PBMC embedding metrics)
smax     specificity mix 0.7*Σw·logFC + 0.3*max logFC (rare-cluster term)

Usage: python examples/adt_tuning_sweep.py [k res mcs blend smax ...]
Outputs: examples/results/adt_sweep_<name>.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import scanpy as sc

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adt_gold_benchmark import (  # noqa: E402
    RARE_MAX_FRAC,
    cluster_metrics,
    load_labeled,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]  # exploration; winners get re-run at 5 seeds
BASE = dict(
    n_top_genes=2000,
    flavor="seurat_v3",
    layer="counts",
    marker_mode="none",
    balance_method="hybrid",
    blend_global=0.95,
    resolution=1.0,
    min_cluster_size=30,
)


def run_one(adata, seed: int, **overrides) -> dict:
    """One scFair selection + downstream scoring against ADT labels."""
    kw = {**BASE, **overrides}
    a = adata.copy()
    scf.pp.highly_variable_genes(a, random_state=seed, **kw)
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
    res = cluster_metrics(adata, genes, seed=seed)
    res["k_used"] = meta.get("n_top_genes_used")
    return res


def scanpy_baseline(adata, seed: int, k: int = 2000) -> dict:
    a = adata.copy()
    sc.pp.highly_variable_genes(
        a, n_top_genes=min(k, a.n_vars - 1), flavor="seurat_v3", layer="counts"
    )
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    return cluster_metrics(adata, genes, seed=seed)


def sweep(adata, name: str, values: list, key: str, **fixed) -> pd.DataFrame:
    rows = []
    for v in values:
        for seed in SEEDS:
            try:
                if key == "__scanpy_k__":
                    res = scanpy_baseline(adata, seed, k=v)
                    res["method"] = "hvg"
                else:
                    res = run_one(adata, seed, **{key: v}, **fixed)
                    res["method"] = "hybrid"
                res.update({key: v, "seed": seed})
                rows.append(res)
                print(
                    f"  {key}={v!s:>6} seed={seed} ARI={res['ARI']:.3f} "
                    f"macroF1={res['macro_f1']:.3f} rareF1={res['rare_f1_mean']:.3f} "
                    f"ncMono={res.get('f1_Mono_nonclassical', float('nan')):.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  {key}={v} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                rows.append({key: v, "seed": seed, "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"adt_sweep_{name}.csv", index=False)
    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df
    if not ok.empty:
        cols = ["ARI", "macro_f1", "rare_f1_mean", "f1_Mono_nonclassical", "f1_Treg"]
        cols = [c for c in cols if c in ok.columns]
        print(f"\n--- {name}: mean over {len(SEEDS)} seeds ---")
        print(ok.groupby(key)[cols].mean().round(3).to_string())
    return df


def main(which: list[str]):
    adata = load_labeled()
    print(f"{adata.n_obs} cells x {adata.n_vars} genes | rare<{RARE_MAX_FRAC:.0%}")

    if "k" in which:
        ks = [500, 1000, 1500, 2000, 2500, 3000, 4000]
        print("\n======== oracle-k sweep: hybrid ========")
        sweep(adata, "k_hybrid", ks, "n_top_genes")
        print("\n======== oracle-k sweep: scanpy hvg ========")
        sweep(adata, "k_hvg", ks, "__scanpy_k__")

    if "knc" in which:
        # Same k sweep with the nearest-neighbour contrast on. If the rare
        # metric stops depending on k, then the k-sensitivity documented in
        # §5.9 was a symptom of the vs-rest blind spot, not of k itself —
        # which would make auto-n's anchor defensible again.
        ks = [500, 1000, 1500, 2000, 2500, 3000, 4000]
        print("\n======== oracle-k sweep: hybrid + neighbor_contrast=1.0 ========")
        sweep(adata, "k_hybrid_nc1", ks, "n_top_genes", neighbor_contrast=1.0)

    if "res" in which:
        print("\n======== intermediate Leiden resolution ========")
        sweep(adata, "res", [0.5, 1.0, 1.5, 2.0, 3.0], "resolution")

    if "mcs" in which:
        print("\n======== intermediate min_cluster_size ========")
        sweep(adata, "mcs", [10, 20, 30, 50], "min_cluster_size")

    if "blend" in which:
        print("\n======== blend_global ========")
        sweep(adata, "blend", [0.80, 0.90, 0.95, 0.99, 1.0], "blend_global")

    print("\nDONE")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or ["k", "res", "mcs", "blend"])
