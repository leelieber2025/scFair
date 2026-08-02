#!/usr/bin/env python3
"""P2 probe: Zheng FACS9 k-sweep (structure short edge vs fixed k).

Edge pack (structure_v6_vs_v7_edge_pack.csv) showed structure auto often
picks k=500 on zheng_facs9_gold_20k with mean ARI ≈ −0.07 vs scanpy@2000,
while k draws of 1500–2000 close the gap. This script measures hybrid and
scanpy ARI on a k grid so we can decide whether a narrow low-nd large-n
rule is justified — **not** a product default change.

Usage (from repo root)::

    python examples/zheng_k_sweep_probe.py
    python examples/zheng_k_sweep_probe.py --n-cells 10000 --seeds 0,1,2

Writes ``examples/results/zheng_k_sweep_probe.csv``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

import scfair as scf
from scfair.pp._auto_n import estimate_n_top_structure

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

DATA = Path("examples/data/zheng_facs9_gold.h5ad")
OUT = Path("examples/results/zheng_k_sweep_probe.csv")
RES = [0.3, 0.5, 0.8, 1.2]
K_GRID = [500, 1000, 1500, 2000, 2500]


def load_zheng(seed: int = 0, n: int = 20_000) -> ad.AnnData:
    a = ad.read_h5ad(DATA)
    a.obs_names_make_unique()
    a.var_names_make_unique()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    if a.n_obs > n:
        rng = np.random.default_rng(seed)
        idx: list = []
        for _, g in a.obs.groupby("cell_type"):
            take = max(50, int(round(n * len(g) / a.n_obs)))
            take = min(take, len(g))
            pick = rng.choice(g.index.to_numpy(), size=take, replace=False)
            idx.extend(pick.tolist())
        idx_arr = np.array(idx)
        if len(idx_arr) > n:
            idx_arr = rng.choice(idx_arr, size=n, replace=False)
        a = a[idx_arr].copy()
    return a


def hybrid_at_k(adata: ad.AnnData, k: int, seed: int) -> list[str]:
    a = adata.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=int(min(k, a.n_vars - 1)),
        balance_method="hybrid",
        flavor="seurat_v3",
        layer="counts",
        random_state=seed,
        diagnose=False,
        marker_mode="none",
    )
    return a.var_names[a.var["highly_variable"]].astype(str).tolist()


def scanpy_at_k(adata: ad.AnnData, k: int) -> list[str]:
    a = adata.copy()
    sc.pp.highly_variable_genes(
        a, n_top_genes=int(min(k, a.n_vars - 1)), flavor="seurat_v3", layer="counts"
    )
    return a.var_names[a.var["highly_variable"]].astype(str).tolist()


def mean_ari(adata: ad.AnnData, genes: list[str], seed: int) -> float:
    e = adata.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    keep = [g for g in genes if g in e.var_names]
    if len(keep) < 10:
        return float("nan")
    e = e[:, keep].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    if n_comps < 2:
        return float("nan")
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(
        e,
        n_neighbors=min(15, e.n_obs - 1),
        n_pcs=min(30, n_comps),
        random_state=seed,
    )
    y = e.obs["cell_type"].astype(str)
    aris = []
    for res in RES:
        sc.tl.leiden(
            e,
            resolution=res,
            key_added="L",
            flavor="igraph",
            n_iterations=2,
            random_state=seed,
        )
        aris.append(adjusted_rand_score(y, e.obs["L"].astype(str)))
    return float(np.mean(aris))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cells", type=int, default=20_000)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}")

    rows: list[dict] = []
    t_all = time.time()
    for seed in seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        a = load_zheng(seed=seed, n=args.n_cells)
        print(f"  shape={a.shape} n_types={a.obs.cell_type.nunique()}", flush=True)

        k_auto, detail = estimate_n_top_structure(a, random_state=seed, version="v7", n_seeds=3)
        feat = detail.get("features") or {}
        print(
            f"  structure auto k={k_auto} branch={detail.get('rule_branch')} "
            f"nd={feat.get('n_density_pops')} nl={feat.get('n_leiden')} "
            f"vm={feat.get('valley_median')} per={detail.get('per_seed_k')}",
            flush=True,
        )
        rx = detail.get("rule_explain") or {}
        if rx:
            print(f"  rule_explain={rx}", flush=True)

        for method, getter in (
            ("hybrid", lambda kk, s=seed: hybrid_at_k(a, kk, s)),
            ("scanpy", lambda kk: scanpy_at_k(a, kk)),
        ):
            for k in K_GRID:
                t0 = time.time()
                genes = getter(k)
                ari = mean_ari(a, genes, seed)
                rows.append(
                    dict(
                        seed=seed,
                        method=method,
                        k=k,
                        ARI=ari,
                        n_genes=len(genes),
                        structure_auto_k=k_auto,
                        rule_branch=detail.get("rule_branch"),
                        n_density_pops=feat.get("n_density_pops"),
                        n_leiden=feat.get("n_leiden"),
                        valley_median=feat.get("valley_median"),
                    )
                )
                print(
                    f"  {method}@{k}: ARI={ari:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} total {time.time() - t_all:.0f}s", flush=True)

    print("\n=== mean ARI by method × k ===", flush=True)
    g = df.groupby(["method", "k"])["ARI"].agg(["mean", "std", "count"]).reset_index().round(4)
    print(g.to_string(index=False), flush=True)

    print("\n=== Δ hybrid@k − scanpy@2000 (mean) ===", flush=True)
    scp = float(df[(df.method == "scanpy") & (df.k == 2000)]["ARI"].mean())
    for k in K_GRID:
        hm = float(df[(df.method == "hybrid") & (df.k == k)]["ARI"].mean())
        print(f"  k={k}: hybrid={hm:.4f}  Δ vs scanpy@2000={hm - scp:+.4f}", flush=True)


if __name__ == "__main__":
    main()
