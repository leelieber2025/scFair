#!/usr/bin/env python
"""Hard-task evaluation package (docs/review.md §9–§10).

Two Hard axes, multi-seed raised to ≥20 (was 5):

1. **k-sweep + short-list→fair** on Cao rare mixes (1%, 2%, 5%).
   At each k, HVG@k vs hybrid@k tests “fair re-rank on a short list”
   without productizing mixHVG/BigSur (global base stays seurat_v3).

2. **ADT protein-gold** k-sweep for ncMono / Treg (orthogonal labels),
   including hybrid + neighbor_contrast@res=1.0.

Tier notes (locked):
  - Cao *full* multi-type = Medium (not run here).
  - Hard success = rare-F1 / pure recall / collapse rate; ARI secondary.
  - auto narrative = k control, not gold-ARI wins.

Usage
-----
  # smoke (3 seeds, one frac)
  python examples/hard_task_benchmark.py --panel both --smoke

  # full Hard (20 seeds; long)
  python examples/hard_task_benchmark.py --panel both --seeds 20

  # resume / subset
  python examples/hard_task_benchmark.py --panel cao --seeds 20
  python examples/hard_task_benchmark.py --panel adt --seeds 20

Outputs under examples/results/:
  hard_cao_rare_ksweep.csv
  hard_cao_rare_ksweep_summary.csv
  hard_adt_ksweep.csv
  hard_adt_ksweep_summary.csv
  hard_task_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adt_gold_benchmark import (  # noqa: E402
    cluster_metrics as adt_cluster_metrics,
)
from adt_gold_benchmark import (
    load_labeled as load_adt_labeled,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Hard multi-seed floor (review.md §9 decision 6)
DEFAULT_SEEDS = 20
SMOKE_SEEDS = 3

K_GRID = [100, 500, 1000, 2000]  # short-list → 2000; auto separate
RARE_FRACS = [0.01, 0.02, 0.05]  # 0.5% historically all-fail
NCMONO_COLLAPSE = 0.6  # F1 below this counts as collapsed seed


# ---------------------------------------------------------------------------
# Cao loaders / rare mix (same protocol as p2)
# ---------------------------------------------------------------------------


def load_cao() -> ad.AnnData:
    path = DATA / "Cao.h5"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/xuyp-csu/CellBRF/main/h5data/Cao.h5",
            path,
        )
    with h5py.File(path, "r") as f:
        X = np.array(f["X"]).astype(np.float32)
        y = np.array(f["Y"]).astype(int).ravel()
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(a.n_vars)]
    a.obs["cell_type"] = pd.Categorical(y.astype(str))
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.obs_names_make_unique()
    return a


def make_rare_mix(
    adata: ad.AnnData,
    *,
    rare_frac: float,
    rare_type: str,
    random_state: int = 0,
    max_majority: int = 3000,
) -> ad.AnnData:
    rng = np.random.default_rng(random_state)
    labels = adata.obs["cell_type"].astype(str)
    maj_idx = np.where((labels != rare_type).to_numpy())[0]
    rare_idx = np.where((labels == rare_type).to_numpy())[0]
    n_maj = min(len(maj_idx), max_majority)
    maj_sel = rng.choice(maj_idx, size=n_maj, replace=False)
    n_rare_target = max(5, int(round(rare_frac * n_maj / max(1e-9, 1.0 - rare_frac))))
    n_rare_target = min(n_rare_target, len(rare_idx))
    rare_sel = rng.choice(rare_idx, size=n_rare_target, replace=False)
    sel = np.concatenate([maj_sel, rare_sel])
    rng.shuffle(sel)
    out = adata[sel].copy()
    out.obs["is_rare"] = (out.obs["cell_type"].astype(str) == rare_type).astype(int)
    out.obs["rare_type"] = rare_type
    out.uns["rare_frac_target"] = rare_frac
    out.uns["rare_frac_actual"] = float(out.obs["is_rare"].mean())
    return out


def pick_rare_type(adata: ad.AnnData) -> str:
    vc = adata.obs["cell_type"].astype(str).value_counts()
    for t in vc.index[1:]:
        if vc[t] >= 50:
            return str(t)
    return str(vc.index[-1])


# ---------------------------------------------------------------------------
# Feature selection configs
# ---------------------------------------------------------------------------


def select_genes(
    adata: ad.AnnData,
    config: str,
    *,
    n_top: int | str,
    seed: int,
) -> tuple[list[str], dict]:
    """Return (genes, meta).

    Configs
    -------
    hvg              scanpy seurat_v3 @ n_top
    hybrid           scFair cluster-balanced (hybrid, res=0.5 default)
    hybrid_nc1       hybrid + neighbor_contrast=1.0 + resolution=1.0
    auto             n_top_genes='auto' + hybrid (n_top arg ignored)
    """
    a = adata.copy()
    meta: dict = {}
    common = dict(
        flavor="seurat_v3",
        layer="counts",
        marker_mode="none",
        random_state=seed,
    )

    if config == "hvg":
        k = min(int(n_top), a.n_vars - 1)
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
        meta = {"n_top_genes_used": k, "config": config}
    elif config == "hybrid":
        k = min(int(n_top), a.n_vars - 1)
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            balance_method="hybrid",
            blend_global=0.95,
            resolution=0.5,
            neighbor_contrast=0.0,
            **common,
        )
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
        meta["config"] = config
    elif config == "hybrid_nc1":
        k = min(int(n_top), a.n_vars - 1)
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            balance_method="hybrid",
            blend_global=0.95,
            resolution=1.0,
            neighbor_contrast=1.0,
            **common,
        )
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
        meta["config"] = config
    elif config == "auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            n_top_min=100,
            n_top_max=min(5000, a.n_vars - 1),
            balance_method="hybrid",
            blend_global=0.95,
            resolution=0.5,
            neighbor_contrast=0.0,
            **common,
        )
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
        meta["config"] = config
    else:
        raise ValueError(config)

    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    meta.pop("selected_genes", None)
    return genes, meta


def cao_cluster_metrics(adata: ad.AnnData, genes: list[str], *, seed: int = 0) -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    if len(genes) < 10:
        return {"n_genes": len(genes), "ARI": np.nan, "NMI": np.nan}
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a,
        resolution=0.8,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    y_true = a.obs["cell_type"].astype(str)
    y_pred = a.obs["leiden"].astype(str)
    out: dict = {
        "n_genes": len(genes),
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
    }
    if "is_rare" in a.obs.columns:
        rare = a.obs["is_rare"].astype(bool)
        n_rare = int(rare.sum())
        pure = 0
        best_f1 = 0.0
        for cl in y_pred.unique():
            m = y_pred == cl
            frac = float(rare[m].mean()) if m.sum() else 0.0
            if frac >= 0.5:
                pure += int((rare & m).sum())
            tp = int((rare & m).sum())
            prec = tp / int(m.sum()) if m.sum() else 0.0
            rec = tp / n_rare if n_rare else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            best_f1 = max(best_f1, f1)
        out["rare_n"] = n_rare
        out["rare_frac"] = float(rare.mean())
        out["rare_recall_pure"] = pure / n_rare if n_rare else np.nan
        out["rare_best_cluster_f1"] = best_f1
    return out


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _key_cols_cao() -> list[str]:
    return ["rare_frac_target", "k_label", "config", "seed"]


def _key_cols_adt() -> list[str]:
    return ["k_label", "config", "seed"]


def load_done(path: Path, key_cols: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or any(c not in df.columns for c in key_cols):
        return set()
    return set(tuple(r) for r in df[key_cols].itertuples(index=False, name=None))


def append_row(path: Path, row: dict, columns: list[str] | None = None) -> None:
    df = pd.DataFrame([row])
    if columns is not None:
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[columns]
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def run_cao_panel(
    *,
    n_seeds: int,
    fracs: list[float],
    k_grid: list[int],
    configs: list[str],
) -> pd.DataFrame:
    """Hard: Cao rare k-sweep / short-list→fair. configs e.g. hvg, hybrid, auto."""
    out_path = OUT / "hard_cao_rare_ksweep.csv"
    base = load_cao()
    rare_type = pick_rare_type(base)
    vc = base.obs["cell_type"].astype(str).value_counts()
    print(
        f"\n==== Cao rare Hard | rare_type={rare_type} (n={vc[rare_type]}) "
        f"| seeds=0..{n_seeds - 1} ====",
        flush=True,
    )

    # work units: (frac, k_label, n_top_or_auto, config, seed)
    units: list[tuple] = []
    for frac in fracs:
        for k in k_grid:
            for cfg in configs:
                if cfg == "auto":
                    continue  # auto once per frac/seed, not per k
                for seed in range(n_seeds):
                    units.append((frac, str(k), k, cfg, seed))
        if "auto" in configs:
            for seed in range(n_seeds):
                units.append((frac, "auto", "auto", "auto", seed))

    done = load_done(out_path, _key_cols_cao())
    n_todo = sum(
        1 for frac, k_label, _nt, cfg, seed in units if (frac, k_label, cfg, seed) not in done
    )
    print(f"  units total={len(units)} done={len(units) - n_todo} todo={n_todo}", flush=True)

    t0 = time.time()
    n_run = 0
    for frac, k_label, n_top, cfg, seed in units:
        key = (frac, k_label, cfg, seed)
        if key in done:
            continue
        mix = make_rare_mix(base, rare_frac=frac, rare_type=rare_type, random_state=seed)
        try:
            genes, meta = select_genes(mix, cfg, n_top=n_top, seed=seed)
            scores = cao_cluster_metrics(mix, genes, seed=seed)
            row = {
                "tier": "Hard",
                "panel": "cao_rare",
                "source": "Cao",
                "rare_type": rare_type,
                "rare_frac_target": frac,
                "rare_frac_actual": mix.uns["rare_frac_actual"],
                "k_label": k_label,
                "n_top_request": n_top if n_top != "auto" else "auto",
                "n_top_used": meta.get("n_top_genes_used", scores.get("n_genes")),
                "config": cfg,
                "seed": seed,
                "n_cells": int(mix.n_obs),
                **{k: scores[k] for k in scores},
            }
        except Exception as e:
            row = {
                "tier": "Hard",
                "panel": "cao_rare",
                "source": "Cao",
                "rare_type": rare_type,
                "rare_frac_target": frac,
                "k_label": k_label,
                "config": cfg,
                "seed": seed,
                "error": f"{type(e).__name__}: {e}",
            }
        append_row(out_path, row)
        done.add(key)
        n_run += 1
        if n_run % 10 == 0 or n_run == n_todo:
            elapsed = time.time() - t0
            print(
                f"  [{n_run}/{n_todo}] frac={frac} k={k_label} cfg={cfg} seed={seed} "
                f"ARI={row.get('ARI', float('nan'))} "
                f"rareF1={row.get('rare_best_cluster_f1', float('nan'))} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    return df


def summarize_cao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ARI" not in df.columns:
        return df
    ok = df.dropna(subset=["ARI"])
    g = ok.groupby(["rare_frac_target", "k_label", "config"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        rare_recall_mean=("rare_recall_pure", "mean"),
        rare_recall_std=("rare_recall_pure", "std"),
        rare_recall_gt0=("rare_recall_pure", lambda s: float((s > 0).mean())),
        rare_f1_mean=("rare_best_cluster_f1", "mean"),
        rare_f1_std=("rare_best_cluster_f1", "std"),
        n_genes_mean=("n_genes", "mean"),
    )
    return g


def run_adt_panel(
    *,
    n_seeds: int,
    k_grid: list[int],
    configs: list[str],
) -> pd.DataFrame:
    """Hard: ADT protein gold k-sweep (ncMono / Treg)."""
    out_path = OUT / "hard_adt_ksweep.csv"
    adata = load_adt_labeled()
    print(
        f"\n==== ADT protein-gold Hard | cells={adata.n_obs} | seeds=0..{n_seeds - 1} ====",
        flush=True,
    )

    units: list[tuple] = []
    for k in k_grid:
        for cfg in configs:
            if cfg == "auto":
                continue
            for seed in range(n_seeds):
                units.append((str(k), k, cfg, seed))
    if "auto" in configs:
        for seed in range(n_seeds):
            units.append(("auto", "auto", "auto", seed))

    done = load_done(out_path, _key_cols_adt())
    n_todo = sum(1 for k_label, _nt, cfg, seed in units if (k_label, cfg, seed) not in done)
    print(f"  units total={len(units)} done={len(units) - n_todo} todo={n_todo}", flush=True)

    t0 = time.time()
    n_run = 0
    for k_label, n_top, cfg, seed in units:
        key = (k_label, cfg, seed)
        if key in done:
            continue
        try:
            genes, meta = select_genes(adata, cfg, n_top=n_top, seed=seed)
            scores = adt_cluster_metrics(adata, genes, seed=seed)
            row = {
                "tier": "Hard",
                "panel": "adt_protein",
                "k_label": k_label,
                "n_top_request": n_top if n_top != "auto" else "auto",
                "n_top_used": meta.get("n_top_genes_used", scores.get("n_genes")),
                "config": cfg,
                "seed": seed,
                **{k: scores[k] for k in scores},
            }
            # collapse flag for ncMono
            nc = scores.get("f1_Mono_nonclassical", np.nan)
            row["ncMono_collapsed"] = int(nc < NCMONO_COLLAPSE) if nc == nc else np.nan
        except Exception as e:
            row = {
                "tier": "Hard",
                "panel": "adt_protein",
                "k_label": k_label,
                "config": cfg,
                "seed": seed,
                "error": f"{type(e).__name__}: {e}",
            }
        append_row(out_path, row)
        done.add(key)
        n_run += 1
        if n_run % 5 == 0 or n_run == n_todo:
            elapsed = time.time() - t0
            print(
                f"  [{n_run}/{n_todo}] k={k_label} cfg={cfg} seed={seed} "
                f"ARI={row.get('ARI', float('nan'))} "
                f"ncMono={row.get('f1_Mono_nonclassical', float('nan'))} "
                f"Treg={row.get('f1_Treg', float('nan'))} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    return df


def summarize_adt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ARI" not in df.columns:
        return df
    ok = df.dropna(subset=["ARI"])
    agg_kw = {
        "n_seeds": ("seed", "nunique"),
        "ARI_mean": ("ARI", "mean"),
        "ARI_std": ("ARI", "std"),
        "macro_f1_mean": ("macro_f1", "mean"),
        "rare_f1_mean_m": ("rare_f1_mean", "mean"),
        "ncMono_mean": ("f1_Mono_nonclassical", "mean"),
        "ncMono_std": ("f1_Mono_nonclassical", "std"),
        "Treg_mean": ("f1_Treg", "mean"),
        "Treg_std": ("f1_Treg", "std"),
        "n_genes_mean": ("n_genes", "mean"),
    }
    if "ncMono_collapsed" in ok.columns:
        agg_kw["ncMono_collapse_rate"] = ("ncMono_collapsed", "mean")
    g = ok.groupby(["k_label", "config"], as_index=False).agg(**agg_kw)
    return g


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--panel",
        choices=["cao", "adt", "both"],
        default="both",
        help="Which Hard panel(s) to run",
    )
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Leiden seeds (default 20)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"Quick path: {SMOKE_SEEDS} seeds, Cao frac=0.02 only, k in {{500,2000}}",
    )
    p.add_argument(
        "--cao-configs",
        default="hvg,hybrid,auto",
        help="Comma list: hvg,hybrid,hybrid_nc1,auto",
    )
    p.add_argument(
        "--adt-configs",
        default="hvg,hybrid,hybrid_nc1,auto",
        help="Comma list: hvg,hybrid,hybrid_nc1,auto",
    )
    args = p.parse_args(argv)

    n_seeds = SMOKE_SEEDS if args.smoke else args.seeds
    k_grid = [500, 2000] if args.smoke else list(K_GRID)
    fracs = [0.02] if args.smoke else list(RARE_FRACS)
    cao_cfgs = [c.strip() for c in args.cao_configs.split(",") if c.strip()]
    adt_cfgs = [c.strip() for c in args.adt_configs.split(",") if c.strip()]

    print(
        f"Hard-task benchmark | panel={args.panel} seeds={n_seeds} k={k_grid} smoke={args.smoke}",
        flush=True,
    )
    print(
        "Policy: Cao full=Medium (skipped); Hard multi-seed≥20; "
        "auto≠gold-ARI win; hybrid docs-name=cluster-balanced",
        flush=True,
    )

    summary: dict = {
        "seeds": n_seeds,
        "k_grid": k_grid,
        "rare_fracs": fracs,
        "smoke": args.smoke,
        "policy": {
            "cao_full": "Medium",
            "hard_seed_floor": 20,
            "auto_narrative": "k_control_not_gold_ari",
            "hybrid_docs_alias": "cluster-balanced",
        },
    }

    if args.panel in ("cao", "both"):
        cao = run_cao_panel(n_seeds=n_seeds, fracs=fracs, k_grid=k_grid, configs=cao_cfgs)
        cao_sum = summarize_cao(cao)
        if not cao_sum.empty:
            cao_sum.to_csv(OUT / "hard_cao_rare_ksweep_summary.csv", index=False)
            print("\n======== Cao rare summary (mean over seeds) ========")
            print(
                cao_sum.pivot_table(
                    index=["rare_frac_target", "k_label"],
                    columns="config",
                    values="rare_f1_mean",
                )
                .round(3)
                .to_string()
            )
            print("\n--- rare_recall_pure mean ---")
            print(
                cao_sum.pivot_table(
                    index=["rare_frac_target", "k_label"],
                    columns="config",
                    values="rare_recall_mean",
                )
                .round(3)
                .to_string()
            )
            print("\n--- ARI mean ---")
            print(
                cao_sum.pivot_table(
                    index=["rare_frac_target", "k_label"],
                    columns="config",
                    values="ARI_mean",
                )
                .round(3)
                .to_string()
            )
        summary["cao_rows"] = int(len(cao)) if cao is not None else 0

    if args.panel in ("adt", "both"):
        adt = run_adt_panel(n_seeds=n_seeds, k_grid=k_grid, configs=adt_cfgs)
        adt_sum = summarize_adt(adt)
        if not adt_sum.empty:
            adt_sum.to_csv(OUT / "hard_adt_ksweep_summary.csv", index=False)
            print("\n======== ADT ncMono F1 (mean over seeds) ========")
            print(
                adt_sum.pivot_table(index="k_label", columns="config", values="ncMono_mean")
                .round(3)
                .to_string()
            )
            if "ncMono_collapse_rate" in adt_sum.columns:
                print("\n--- ncMono collapse rate (F1 < 0.6) ---")
                print(
                    adt_sum.pivot_table(
                        index="k_label",
                        columns="config",
                        values="ncMono_collapse_rate",
                    )
                    .round(2)
                    .to_string()
                )
            print("\n--- ARI mean ---")
            print(
                adt_sum.pivot_table(index="k_label", columns="config", values="ARI_mean")
                .round(3)
                .to_string()
            )
            print("\n--- Treg F1 mean (report separately; often uninformative) ---")
            print(
                adt_sum.pivot_table(index="k_label", columns="config", values="Treg_mean")
                .round(3)
                .to_string()
            )
        summary["adt_rows"] = int(len(adt)) if adt is not None else 0

    with open(OUT / "hard_task_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary → {OUT / 'hard_task_summary.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
