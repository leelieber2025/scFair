#!/usr/bin/env python
"""L1: output gene-budget sweep — does a longer shortlist raise absolute
quality, and does hybrid's margin over scanpy shrink?

Protocol
--------
- Datasets: FACS-sorted Duo gold (labels independent of HVG clustering).
- Arms: ``scanpy`` = balance_method="none"; ``hybrid`` = hybrid +
  allocation_method="none" (product path).
- k grid: 2000, 3000, 4000 (``wide`` also 6000 for duo8 ceiling check).
- Intermediate clustering resolution fixed at 0.5 (isolate k, not auto-res).
- Downstream: Leiden grid {0.3, 0.5, 0.8, 1.2}, multi-seed; metrics ARI /
  macro-F1 / min_pop_F1 averaged over the res grid then seeds.

Outputs (examples/results/):
  l1_k_budget_panel.csv
  l1_k_budget_panel_pops.csv
  l1_k_budget_summary.csv   — three views: @2000, best-k, shared 3k/4k

Usage:
  python examples/l1_k_budget_panel.py           # duo ×3, k∈{2,3,4}k, 3 seeds
  python examples/l1_k_budget_panel.py wide      # + k=6000
  python examples/l1_k_budget_panel.py quick     # 2 seeds, k∈{2,3,4}k
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
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "l1_k_budget_panel.csv"
POPS = OUT / "l1_k_budget_panel_pops.csv"
SUMMARY = OUT / "l1_k_budget_summary.csv"

INT_RESOLUTION = 0.5
RES_GRID = [0.3, 0.5, 0.8, 1.2]
ARMS = ("hybrid", "scanpy")
ORDER = ["duo4_pbmc", "duo8_pbmc", "duo4un_pbmc"]


def evaluate(a, genes, seed, rows, pop_rows, base):
    e = a.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    keep = [g for g in genes if g in e.var_names]
    e = e[:, keep].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    y_true = e.obs["cell_type"].astype(str)
    prev = y_true.value_counts(normalize=True)

    for res in RES_GRID:
        sc.tl.leiden(
            e,
            resolution=res,
            key_added="L",
            flavor="igraph",
            n_iterations=2,
            random_state=seed,
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
        for pop, f1 in f1s.items():
            pop_rows.append(
                {
                    **{k: base[k] for k in ("dataset", "arm", "k", "seed")},
                    "resolution": res,
                    "population": pop,
                    "prevalence": float(prev.get(pop, np.nan)),
                    "f1": float(f1),
                }
            )
        rows.append(
            {
                **base,
                "resolution": res,
                "n_leiden": int(y_pred.nunique()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
            }
        )


def select_genes(ad, arm: str, k: int, seed: int) -> list[str]:
    if arm == "scanpy":
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=k,
            balance_method="none",
            random_state=seed,
            diagnose=False,
        )
    elif arm == "hybrid":
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=k,
            balance_method="hybrid",
            resolution=INT_RESOLUTION,
            allocation_method="none",
            random_state=seed,
            diagnose=False,
        )
    else:
        raise ValueError(arm)
    return list(ad.var_names[ad.var["highly_variable"]])


def per_seed_means(rows: list[dict]) -> pd.DataFrame:
    """Mean over downstream res grid → one row per dataset/arm/k/seed."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["dataset", "arm", "k", "seed"], as_index=False)[
        ["ARI", "NMI", "macro_f1", "min_pop_f1"]
    ].mean()


def write_summary(rows: list[dict]) -> pd.DataFrame:
    g = per_seed_means(rows)
    if g.empty:
        return g
    # mean over seeds
    ds = g.groupby(["dataset", "arm", "k"], as_index=False)[
        ["ARI", "NMI", "macro_f1", "min_pop_f1"]
    ].mean()

    views = []
    # View 1: fixed k=2000 margin
    for name, sub in ds.groupby("dataset"):
        h = sub[(sub.arm == "hybrid") & (sub.k == 2000)]
        s = sub[(sub.arm == "scanpy") & (sub.k == 2000)]
        if len(h) and len(s):
            views.append(
                {
                    "view": "fixed_k2000",
                    "dataset": name,
                    "k": 2000,
                    "ARI_hybrid": float(h.ARI.iloc[0]),
                    "ARI_scanpy": float(s.ARI.iloc[0]),
                    "d_hybrid_scanpy": float(h.ARI.iloc[0] - s.ARI.iloc[0]),
                    "macro_f1_hybrid": float(h.macro_f1.iloc[0]),
                    "macro_f1_scanpy": float(s.macro_f1.iloc[0]),
                }
            )

    # View 2: each arm's best k (by ARI)
    for name, sub in ds.groupby("dataset"):
        row = {"view": "best_k", "dataset": name}
        for arm in ARMS:
            a = sub[sub.arm == arm]
            if a.empty:
                continue
            best = a.loc[a.ARI.idxmax()]
            row[f"k_{arm}"] = int(best.k)
            row[f"ARI_{arm}"] = float(best.ARI)
            row[f"macro_f1_{arm}"] = float(best.macro_f1)
        if "ARI_hybrid" in row and "ARI_scanpy" in row:
            row["d_hybrid_scanpy"] = row["ARI_hybrid"] - row["ARI_scanpy"]
            row["k"] = row.get("k_hybrid")
        views.append(row)

    # View 3: shared k in {3000, 4000, ...}
    for k in sorted(ds.k.unique()):
        if k < 2000:
            continue
        for name, sub in ds.groupby("dataset"):
            h = sub[(sub.arm == "hybrid") & (sub.k == k)]
            s = sub[(sub.arm == "scanpy") & (sub.k == k)]
            if len(h) and len(s):
                views.append(
                    {
                        "view": f"shared_k{k}",
                        "dataset": name,
                        "k": int(k),
                        "ARI_hybrid": float(h.ARI.iloc[0]),
                        "ARI_scanpy": float(s.ARI.iloc[0]),
                        "d_hybrid_scanpy": float(h.ARI.iloc[0] - s.ARI.iloc[0]),
                        "macro_f1_hybrid": float(h.macro_f1.iloc[0]),
                        "macro_f1_scanpy": float(s.macro_f1.iloc[0]),
                    }
                )

    # View 4: hybrid ARI vs k (absolute budget curve)
    for _, r in ds[ds.arm == "hybrid"].iterrows():
        views.append(
            {
                "view": "hybrid_k_curve",
                "dataset": r.dataset,
                "k": int(r.k),
                "ARI_hybrid": float(r.ARI),
                "macro_f1_hybrid": float(r.macro_f1),
                "min_pop_f1_hybrid": float(r.min_pop_f1),
            }
        )

    out = pd.DataFrame(views)
    out.to_csv(SUMMARY, index=False)
    return out


def print_tables(rows: list[dict]) -> None:
    g = per_seed_means(rows)
    if g.empty:
        return
    ds = g.groupby(["dataset", "arm", "k"], as_index=False)[
        ["ARI", "macro_f1", "min_pop_f1"]
    ].mean()

    print("\n=== hybrid ARI by k (mean over seeds × res) ===", flush=True)
    h = ds[ds.arm == "hybrid"].pivot(index="dataset", columns="k", values="ARI")
    print(h.round(4).to_string(), flush=True)

    print("\n=== scanpy ARI by k ===", flush=True)
    s = ds[ds.arm == "scanpy"].pivot(index="dataset", columns="k", values="ARI")
    print(s.round(4).to_string(), flush=True)

    print("\n=== hybrid − scanpy (matched k) ===", flush=True)
    # align
    d = h - s
    print(d.round(4).to_string(), flush=True)

    print("\n=== vs k=2000 hybrid (Δ ARI) ===", flush=True)
    if 2000 in h.columns:
        print((h.subtract(h[2000], axis=0)).round(4).to_string(), flush=True)

    # best k
    print("\n=== best k by ARI ===", flush=True)
    for name in ORDER:
        sub = ds[ds.dataset == name]
        for arm in ARMS:
            a = sub[sub.arm == arm]
            if a.empty:
                continue
            best = a.loc[a.ARI.idxmax()]
            print(
                f"  {name:14s} {arm:7s}  k={int(best.k)}  "
                f"ARI={best.ARI:.4f}  macro_f1={best.macro_f1:.4f}",
                flush=True,
            )


def run_panel(k_grid: list[int], seeds: list[int]) -> None:
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pop_rows = pd.read_csv(POPS).to_dict("records") if POPS.exists() else []
    seen: dict[tuple, set] = {}
    for r in rows:
        key = (r["dataset"], r["seed"], int(r["k"]))
        seen.setdefault(key, set()).add(r["arm"])
    done = {k for k, v in seen.items() if set(ARMS) <= v}
    print(
        f"resuming: {len(done)} (dataset, seed, k) blocks done; k_grid={k_grid} seeds={seeds}",
        flush=True,
    )

    for name in ORDER:
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        for seed in seeds:
            for k in k_grid:
                key = (name, seed, k)
                if key in done:
                    continue
                t0 = time.time()
                for arm in ARMS:
                    ad = a.copy()
                    genes = select_genes(ad, arm, k, seed)
                    base = {
                        "dataset": name,
                        "arm": arm,
                        "k": k,
                        "seed": seed,
                        "n_genes": len(genes),
                    }
                    evaluate(a, genes, seed, rows, pop_rows, base)
                done.add(key)
                pd.DataFrame(rows).to_csv(CSV, index=False)
                pd.DataFrame(pop_rows).to_csv(POPS, index=False)
                got = {
                    arm: np.mean(
                        [
                            r["ARI"]
                            for r in rows
                            if r["dataset"] == name
                            and r["seed"] == seed
                            and int(r["k"]) == k
                            and r["arm"] == arm
                        ]
                    )
                    for arm in ARMS
                }
                print(
                    f"  seed={seed} k={k}  ARI hybrid={got['hybrid']:.3f} "
                    f"scanpy={got['scanpy']:.3f}  Δ={got['hybrid'] - got['scanpy']:+.3f}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        del a
        print_tables(rows)

    write_summary(rows)
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)
    print(f"wrote {SUMMARY}", flush=True)
    print_tables(rows)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    if mode == "quick":
        run_panel([2000, 3000, 4000], seeds=[0, 1])
    elif mode == "wide":
        run_panel([2000, 3000, 4000, 6000], seeds=[0, 1, 2])
    elif mode == "default":
        run_panel([2000, 3000, 4000], seeds=[0, 1, 2])
    else:
        raise SystemExit("use quick | default | wide")
