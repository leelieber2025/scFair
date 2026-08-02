#!/usr/bin/env python
"""Panel 1: sorted / protein gold standards × cluster-pool decoupling.

Two questions, one protocol (DEVELOPMENT_LOG §5.15 items 2 and 4):

Q1  Does §5.15's "k=1000 beats k=2000" hold on **cell-sorting** labels, or was
    it specific to protein-Leiden labels? duo4/duo8/duo4un are FACS-sorted
    (Zhengmix), the cleanest non-circular labels available here: no clustering
    choice enters the ground truth at all.

Q2  Is the rare-cell advantage `auto` showed over `hybrid@2000` reproducible by
    decoupling the intermediate-clustering pool from k? §5.15 traced auto's
    behaviour to clustering its `n_top_max` pool. There are two distinct
    decouplings and this panel separates them:

      B1  `cluster_pool=5000`  -> cluster on the **global top-5000**
                                  (clean, independent of balance_method)
      B2  `n_top_genes="auto"` -> cluster on the **hybrid-selected 5000**
                                  (what auto does today; two-stage, and
                                  circular in that hybrid genes feed the
                                  clustering that scores hybrid genes)

    Verified before writing this: ARI(auto clusters, hybrid@5000 clusters)
    = 1.000000, while ARI(cluster_pool=5000 clusters, auto clusters) = 0.8595 —
    so B1 and B2 are genuinely different arms, not two names for one thing.

Tier note: duo* are Easy/Medium (10 seeds); the ADT panel is the Hard one and
keeps the review.md §9 floor of 20 seeds.

Usage
-----
  python examples/sorted_gold_panel.py --smoke
  python examples/sorted_gold_panel.py                 # full, resumable
  python examples/sorted_gold_panel.py --datasets duo4_pbmc,adt14

Outputs under examples/results/:
  sorted_gold_panel.csv / _summary.csv / _summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from adt_gold_benchmark import load_labeled as load_adt_labeled  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

LEIDEN_RES = 0.8  # evaluation clustering, same as §5.9-5.15
K_GRID = [1000, 2000]  # the §5.15 comparison that mattered
CLUSTER_POOL = 5000
SEEDS_EASY = 10
SEEDS_HARD = 20

SORTED_SETS = ["duo4_pbmc", "duo8_pbmc", "duo4un_pbmc"]
ALL_SETS = SORTED_SETS + ["adt14"]
CONFIGS = ["hvg", "hybrid", "hybrid_cp5000", "auto"]


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> tuple[ad.AnnData, dict]:
    """Return (adata, info). info carries label column, mask and tier."""
    if name == "adt14":
        a = load_adt_labeled()
        return a, {
            "label_col": "cell_type",
            "conf_col": "adt_confident" if "adt_confident" in a.obs else None,
            "tier": "Hard",
            "label_kind": "protein_leiden",
            "n_seeds": SEEDS_HARD,
        }
    path = DATA / f"{name}.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    a = ad.read_h5ad(path)
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    tier = "Medium" if name != "duo4_pbmc" else "Easy"
    return a, {
        "label_col": "cell_type",
        "conf_col": None,
        "tier": tier,
        "label_kind": "cell_sorting",
        "n_seeds": SEEDS_EASY,
    }


# ---------------------------------------------------------------------------
# feature selection arms
# ---------------------------------------------------------------------------


def select_genes(adata, config: str, *, n_top, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    common = dict(flavor="seurat_v3", layer="counts", marker_mode="none", random_state=seed)
    if config == "hvg":
        k = min(int(n_top), a.n_vars - 1)
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
        meta = {"n_top_genes_used": k}
    else:
        kw = dict(
            balance_method="hybrid",
            blend_global=0.95,
            resolution=0.5,
            neighbor_contrast=0.0,
            **common,
        )
        if config == "hybrid":
            scf.pp.highly_variable_genes(a, n_top_genes=min(int(n_top), a.n_vars - 1), **kw)
        elif config == "hybrid_cp5000":
            scf.pp.highly_variable_genes(
                a,
                n_top_genes=min(int(n_top), a.n_vars - 1),
                cluster_pool=min(CLUSTER_POOL, a.n_vars - 1),
                **kw,
            )
        elif config == "auto":
            scf.pp.highly_variable_genes(
                a,
                n_top_genes="auto",
                n_top_min=100,
                n_top_max=min(CLUSTER_POOL, a.n_vars - 1),
                **kw,
            )
        else:
            raise ValueError(config)
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
        meta.pop("selected_genes", None)
        meta.pop("cluster_weights", None)
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    return genes, meta


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def per_population_f1(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Best-matching-cluster F1 per true population (Hungarian-free, per-pop max)."""
    out: dict[str, float] = {}
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        n_t = int(t.sum())
        best = 0.0
        for cl in y_pred.unique():
            m = (y_pred == cl).to_numpy()
            tp = int((t & m).sum())
            if tp == 0:
                continue
            prec = tp / int(m.sum())
            rec = tp / n_t
            best = max(best, 2 * prec * rec / (prec + rec))
        out[str(pop)] = best
    return out


def evaluate(adata, genes: list[str], info: dict, *, seed: int) -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    if len(genes) < 10:
        return {"n_genes": len(genes), "ARI": np.nan}
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a,
        resolution=LEIDEN_RES,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    keep = (
        a.obs[info["conf_col"]].to_numpy(dtype=bool)
        if info["conf_col"]
        else np.ones(a.n_obs, dtype=bool)
    )
    y_true = a.obs[info["label_col"]].astype(str)[keep]
    y_pred = a.obs["leiden"].astype(str)[keep]
    f1 = per_population_f1(y_true, y_pred)
    prev = y_true.value_counts(normalize=True)
    smallest = prev.idxmin()
    res = {
        "n_genes": len(genes),
        "n_leiden": int(y_pred.nunique()),
        "n_eval_cells": int(keep.sum()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1.values()))),
        # imbalance-sensitive and defined on every panel (duo's smallest class is
        # 7.7%, so a <2% "rare" threshold would select nothing there)
        "min_pop_f1": float(f1[str(smallest)]),
        "min_pop": str(smallest),
        "min_pop_frac": float(prev.min()),
    }
    rare = [p for p in f1 if prev.get(p, 0.0) < 0.02]
    res["rare_f1_mean"] = float(np.mean([f1[p] for p in rare])) if rare else np.nan
    for pop, v in f1.items():
        res[f"f1_{pop}"] = float(v)
    return res


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

KEY = ["dataset", "k_label", "config", "seed"]


def _json_safe(o):
    """numpy/pandas scalars are not JSON-serialisable by default."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def _jsonl_path(path: Path) -> Path:
    return path.with_suffix(".jsonl")


def append_row(path: Path, row: dict) -> None:
    """Append one result as a JSON line.

    Rows do not all carry the same keys - per-population F1 columns depend on the
    dataset, and only some k values get the downstream clustering - so appending
    with ``to_csv(mode="a", header=not exists)`` wrote the header from the *first*
    row and then silently emitted rows with more fields than the header. Both of
    this repo's earlier panel CSVs came out unreadable that way. JSON lines have no
    header to disagree with; the CSV is materialised from the union at the end.
    """
    with open(_jsonl_path(path), "a") as fh:
        fh.write(json.dumps(row, default=_json_safe) + "\n")


def read_rows(path: Path) -> pd.DataFrame:
    jl = _jsonl_path(path)
    if not jl.exists():
        return pd.DataFrame()
    with open(jl) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(rows)


def load_done(path: Path) -> set[tuple]:
    df = read_rows(path)
    if df.empty or any(c not in df.columns for c in KEY):
        return set()
    return set(tuple(r) for r in df[KEY].itertuples(index=False, name=None))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", default=",".join(ALL_SETS))
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--smoke", action="store_true", help="2 seeds, k=2000 only")
    args = p.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    k_grid = [2000] if args.smoke else list(K_GRID)
    out_path = OUT / ("sorted_gold_panel_smoke.csv" if args.smoke else "sorted_gold_panel.csv")

    print(
        f"Sorted/protein gold panel | datasets={datasets} configs={configs} "
        f"k={k_grid} smoke={args.smoke}",
        flush=True,
    )
    print(
        "Arms: hybrid=cluster on top-k (current default) | "
        "hybrid_cp5000=cluster on global top-5000 (B1) | "
        "auto=cluster on hybrid-selected 5000 (B2)",
        flush=True,
    )

    done = load_done(out_path)
    t0 = time.time()
    n_run = 0
    for name in datasets:
        adata, info = load_dataset(name)
        n_seeds = 2 if args.smoke else info["n_seeds"]
        vc = adata.obs[info["label_col"]].astype(str).value_counts()
        print(
            f"\n==== {name} | {adata.n_obs}x{adata.n_vars} | {len(vc)} pops "
            f"| labels={info['label_kind']} | tier={info['tier']} "
            f"| min={vc.min()} ({100 * vc.min() / adata.n_obs:.2f}%) "
            f"| seeds={n_seeds} ====",
            flush=True,
        )
        units = []
        for cfg in configs:
            if cfg == "auto":
                units += [("auto", "auto", cfg, s) for s in range(n_seeds)]
            else:
                for k in k_grid:
                    units += [(str(k), k, cfg, s) for s in range(n_seeds)]

        for k_label, n_top, cfg, seed in units:
            key = (name, k_label, cfg, seed)
            if key in done:
                continue
            row = {
                "dataset": name,
                "tier": info["tier"],
                "label_kind": info["label_kind"],
                "k_label": k_label,
                "config": cfg,
                "seed": seed,
                "n_cells": int(adata.n_obs),
            }
            try:
                genes, meta = select_genes(adata, cfg, n_top=n_top, seed=seed)
                scores = evaluate(adata, genes, info, seed=seed)
                row.update(
                    n_top_used=meta.get("n_top_genes_used", scores.get("n_genes")),
                    cluster_pool_used=meta.get("cluster_pool"),
                    **scores,
                )
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            append_row(out_path, row)
            done.add(key)
            n_run += 1
            if n_run % 5 == 0:
                print(
                    f"  [{n_run}] {name} k={k_label} {cfg} s={seed} "
                    f"ARI={row.get('ARI', float('nan')):.4f} "
                    f"mF1={row.get('macro_f1', float('nan')):.4f} "
                    f"minF1={row.get('min_pop_f1', float('nan')):.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    df = read_rows(out_path)
    if df.empty:
        print("no rows written")
        return
    df.to_csv(out_path, index=False)  # union of all keys, aligned
    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df
    if ok.empty:
        print("no successful rows")
        return
    summ = ok.groupby(["dataset", "label_kind", "k_label", "config"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        min_pop_f1_mean=("min_pop_f1", "mean"),
        min_pop_f1_std=("min_pop_f1", "std"),
        n_genes_mean=("n_genes", "mean"),
    )
    spath = OUT / (
        "sorted_gold_panel_smoke_summary.csv" if args.smoke else "sorted_gold_panel_summary.csv"
    )
    summ.to_csv(spath, index=False)

    for metric in ("ARI_mean", "macro_f1_mean", "min_pop_f1_mean"):
        print(f"\n======== {metric} ========")
        print(
            summ.pivot_table(index=["dataset", "k_label"], columns="config", values=metric)
            .round(4)
            .to_string()
        )
    with open(OUT / "sorted_gold_panel_summary.json", "w") as f:
        json.dump(
            {
                "k_grid": k_grid,
                "cluster_pool": CLUSTER_POOL,
                "seeds": {"easy_medium": SEEDS_EASY, "hard": SEEDS_HARD},
                "arms": {
                    "hybrid": "cluster on top-k (default)",
                    "hybrid_cp5000": "B1 cluster on global top-5000",
                    "auto": "B2 cluster on hybrid-selected 5000",
                },
                "rows": int(len(df)),
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}\nwrote {spath}\nDONE")


if __name__ == "__main__":
    main()
