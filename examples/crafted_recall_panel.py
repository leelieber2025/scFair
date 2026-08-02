#!/usr/bin/env python
"""Panel 2: crafted-signal gene recall as a function of k (P2 x scFair).

**Not on the product path.** scFair defaults are hybrid + structure auto_n
for clustering fairness; this script is archival research only. Matrices
live under ``examples/data/crafted_p2/`` and are **not** kept in the default
local cache — re-download from CraftedExperiment / Zenodo before running.

Motivation
----------
Liu et al. (NAR Genom Bioinform 2025, PMC11920870) fix the feature count at 2000
and name it as their own open limitation:

    "GOF still relies on the user's choice of several parameters. These include
     the number of genes to select that was arbitrarily set at 2000 in our
     examples here."

They report the proportion of crafted genes each method recovers, but only at that
single k, as bar plots. The recall-versus-k curve is therefore unfilled space, and
it is exactly what auto-n has never had: **a criterion for k that does not use cell
labels at all.**

Every previous attempt to judge auto-n went through ARI / macro-F1 against cell
labels, where the margins are ~0.01 and label provenance confounds the comparison
(DEVELOPMENT_LOG §5.7.5), and §5.14 then showed that no label-free summary
statistic predicts the benefit. Crafted data sidesteps both: the injected genes are
known exactly, so

    k*  =  the smallest k that recovers the injected signal

is a ground-truth objective, immune to the protocol shadow. This panel measures
k*, then asks how far `auto` lands from it.

Falsifiable prediction being tested
-----------------------------------
auto-n v2.2 sets its floor/ceiling/anchor from sequencing **depth** (median counts
and genes per cell). P2's central result is that different methods win in sparse
vs dense regions, and the mechanism carries over to k: sparse signal genes rank low
on variance-based scores, so recovering them should need a *larger* k. Hence

    k*(Sparse)  >  k*(Medium)  >  k*(Dense)      at matched signal size

If that holds, auto-n has a principled **sparsity-aware** bound, which is closer to
the mechanism than depth tiers. If it fails, that is a cheap negative.

Watch for a collision with §5.13: if sparse signal needs k>3000, but k>=3000 is
where scFair's advantage over scanpy HVG disappears, then scFair is structurally
weak on sparse signal — a real limitation, not a tuning problem.

Arms
----
  hvg             scanpy seurat_v3 @ k
  hybrid          scFair cluster-balanced @ k (default; clusters the top-k mask)
  hybrid_cp5000   B1 — clusters the global top-5000 regardless of k (§5.15 item 4)
  auto            k chosen by the ensemble (also B2: clusters its n_top_max pool)

Caveats recorded in the output
------------------------------
The 24 crafted matrices are NOT 24 independent datasets: one base matrix, and the
gene lists are nested (50 subset 100 subset 300 subset 600). This panel can
establish mechanism; it cannot produce dataset-level statistics and does not
satisfy §5.14's n>=20 requirement. The signal is also an additive perturbation, not
biology, so "recall of injected genes" is not "recall of real markers".

Usage
-----
  python examples/crafted_recall_panel.py --smoke
  python examples/crafted_recall_panel.py                  # full, resumable
  python examples/crafted_recall_panel.py --all-perturbations

Outputs under examples/results/:
  crafted_recall_panel.csv / _summary.csv / _kstar.csv / _summary.json
"""

from __future__ import annotations

import argparse
import json
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
DATA = ROOT / "data"
CRAFTED = DATA / "crafted_p2"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

K_GRID = [100, 500, 1000, 2000, 3000]
EVAL_K = {"500", "2000", "auto"}  # downstream clustering only where it is informative
CLUSTER_POOL = 5000
CONFIGS = ["hvg", "hybrid", "hybrid_cp5000", "auto"]
SEEDS = 3
LEIDEN_RES = 0.8
RECALL_TARGET = 0.9  # k* = smallest k reaching this recall

# One perturbation type per sparsity regime for the main k sweep; the remaining
# types are a robustness check (--all-perturbations widens it). Matched as
# prefixes: upstream filenames are inconsistent — the Medium files are spelled
# "craftedata" while Sparse/Dense are "crafteddata".
PRIMARY_PREFIXES = ("Sparse_AddPois0.5", "Medium_AddPois1.5", "Dense_AddPoisF1.5")


def crafted_datasets(primary_only: bool = True) -> list[tuple[str, Path, Path]]:
    out = []
    for h5 in sorted(CRAFTED.glob("*.h5ad")):
        name = h5.stem
        truth = CRAFTED / f"{name}.crafted_genes.csv"
        if not truth.exists():
            continue
        family = name.split("__")[0]
        if primary_only and not family.startswith(PRIMARY_PREFIXES):
            continue
        out.append((name, h5, truth))
    return out


def load_crafted(h5: Path, truth: Path) -> tuple[ad.AnnData, list[str], dict]:
    a = ad.read_h5ad(h5)
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    genes = pd.read_csv(truth)["crafted_gene"].astype(str).tolist()
    present = [g for g in genes if g in set(a.var_names.astype(str))]
    lab = a.obs["cell_type"].astype(str)
    crafted_pop = next((p for p in lab.unique() if p.startswith("crafted")), None)
    info = {
        "n_crafted_truth": len(genes),
        "n_crafted_present": len(present),
        "crafted_pop": crafted_pop,
        "crafted_frac": float((lab == crafted_pop).mean()) if crafted_pop else np.nan,
        "regime": h5.stem.split("_")[0],
        "family": h5.stem.split("__")[0],
        "n_signal": int(h5.stem.split(".")[-1]),
    }
    return a, present, info


def select_genes(adata, config: str, *, n_top, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    common = dict(
        flavor="seurat_v3",
        layer="counts",
        marker_mode="none",
        random_state=seed,
        progress=False,
    )
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
        for drop in ("selected_genes", "cluster_weights", "auto_n"):
            meta.pop(drop, None)
    return a.var_names[a.var["highly_variable"]].astype(str).tolist(), meta


def gene_scores(selected: list[str], truth: list[str]) -> dict:
    """Recall / precision / F1 of the injected signal genes — the label-free part."""
    s, t = set(selected), set(truth)
    tp = len(s & t)
    rec = tp / len(t) if t else np.nan
    prec = tp / len(s) if s else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else 0.0
    # enrichment over picking k genes at random
    return {
        "n_selected": len(s),
        "crafted_recovered": tp,
        "gene_recall": rec,
        "gene_precision": prec,
        "gene_f1": f1,
    }


def per_population_f1(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for pop in y_true.unique():
        t = (y_true == pop).to_numpy()
        n_t = int(t.sum())
        best = 0.0
        for cl in y_pred.unique():
            m = (y_pred == cl).to_numpy()
            tp = int((t & m).sum())
            if tp:
                prec, rec = tp / int(m.sum()), tp / n_t
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
        return {}
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
    y_true = a.obs["cell_type"].astype(str)
    y_pred = a.obs["leiden"].astype(str)
    f1 = per_population_f1(y_true, y_pred)
    return {
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1.values()))),
        "crafted_pop_f1": float(f1.get(str(info["crafted_pop"]), np.nan)),
    }


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


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = df.dropna(subset=["gene_recall"])
    summ = ok.groupby(
        ["regime", "family", "n_signal", "dataset", "k_label", "config"], as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        n_top_used_mean=("n_top_used", "mean"),
        gene_recall_mean=("gene_recall", "mean"),
        gene_recall_std=("gene_recall", "std"),
        gene_precision_mean=("gene_precision", "mean"),
        gene_f1_mean=("gene_f1", "mean"),
        ARI_mean=("ARI", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        crafted_pop_f1_mean=("crafted_pop_f1", "mean"),
    )

    # k*: smallest fixed k reaching RECALL_TARGET, and the k auto actually chose
    rows = []
    fixed = summ[summ.k_label != "auto"].copy()
    fixed["k"] = fixed.k_label.astype(int)
    for (regime, fam, nsig, cfg), g in fixed.groupby(["regime", "family", "n_signal", "config"]):
        g = g.sort_values("k")
        hit = g[g.gene_recall_mean >= RECALL_TARGET]
        best = g.loc[g.gene_f1_mean.idxmax()]
        auto = summ[
            (summ.regime == regime)
            & (summ.family == fam)
            & (summ.n_signal == nsig)
            & (summ.config == "auto")
        ]
        rows.append(
            dict(
                regime=regime,
                family=fam,
                n_signal=nsig,
                config=cfg,
                k_star_recall90=int(hit.iloc[0]["k"]) if len(hit) else None,
                recall_at_3000=float(g[g.k == 3000].gene_recall_mean.mean())
                if (g.k == 3000).any()
                else np.nan,
                k_best_gene_f1=int(best["k"]),
                best_gene_f1=round(float(best.gene_f1_mean), 4),
                auto_k=float(auto.n_top_used_mean.mean()) if len(auto) else np.nan,
                auto_recall=float(auto.gene_recall_mean.mean()) if len(auto) else np.nan,
            )
        )
    return summ, pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--smoke", action="store_true", help="1 seed, k in {500,2000}, 3 datasets")
    p.add_argument(
        "--all-perturbations",
        action="store_true",
        help="include every perturbation type, not just one per regime",
    )
    p.add_argument("--configs", default=",".join(CONFIGS))
    args = p.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    k_grid = [500, 2000] if args.smoke else list(K_GRID)
    n_seeds = 1 if args.smoke else SEEDS
    sets = crafted_datasets(primary_only=not args.all_perturbations)
    if args.smoke:
        sets = [s for s in sets if s[0].endswith(".100")][:3]
    out_path = OUT / ("crafted_recall_smoke.csv" if args.smoke else "crafted_recall_panel.csv")

    print(f"Crafted recall panel | {len(sets)} datasets | k={k_grid} | seeds={n_seeds}", flush=True)
    print("Arms: hvg | hybrid (clusters top-k) | hybrid_cp5000 (B1) | auto (B2)", flush=True)
    print(f"k* = smallest k with gene recall >= {RECALL_TARGET}", flush=True)
    print(
        "NOTE: nested gene lists on one base matrix — mechanism only, not "
        "dataset-level statistics (§5.14 still needs n>=20).",
        flush=True,
    )

    done = load_done(out_path)
    t0, n_run = time.time(), 0
    for name, h5, truth in sets:
        adata, present, info = load_crafted(h5, truth)
        print(
            f"\n==== {name} | {adata.n_obs}x{adata.n_vars} | regime={info['regime']} "
            f"| signal={info['n_signal']} ({info['n_crafted_present']} present) "
            f"| crafted pop {info['crafted_frac'] * 100:.1f}% ====",
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
                "tier": "Hard_crafted",
                **{
                    k: info[k]
                    for k in (
                        "regime",
                        "family",
                        "n_signal",
                        "n_crafted_truth",
                        "n_crafted_present",
                        "crafted_pop",
                        "crafted_frac",
                    )
                },
                "k_label": k_label,
                "config": cfg,
                "seed": seed,
                "n_cells": int(adata.n_obs),
                "n_genes_total": int(adata.n_vars),
            }
            try:
                genes, meta = select_genes(adata, cfg, n_top=n_top, seed=seed)
                row["n_top_used"] = meta.get("n_top_genes_used", len(genes))
                row["cluster_pool_used"] = meta.get("cluster_pool")
                row.update(gene_scores(genes, present))
                if k_label in EVAL_K:
                    row.update(evaluate(adata, genes, info, seed=seed))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            append_row(out_path, row)
            done.add(key)
            n_run += 1
            if n_run % 5 == 0:
                print(
                    f"  [{n_run}] {info['regime']}/{info['n_signal']} k={k_label} "
                    f"{cfg} s={seed} recall={row.get('gene_recall', float('nan')):.3f} "
                    f"prec={row.get('gene_precision', float('nan')):.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    df = read_rows(out_path)
    if df.empty:
        print("no rows written — check the dataset filter (0 datasets selected?)")
        return
    df.to_csv(out_path, index=False)  # union of all keys, aligned
    if "gene_recall" not in df.columns or df.gene_recall.dropna().empty:
        print("no successful rows")
        return
    summ, kstar = summarize(df)
    sp = OUT / (
        "crafted_recall_smoke_summary.csv" if args.smoke else "crafted_recall_panel_summary.csv"
    )
    summ.to_csv(sp, index=False)
    kp = OUT / (
        "crafted_recall_smoke_kstar.csv" if args.smoke else "crafted_recall_panel_kstar.csv"
    )
    kstar.to_csv(kp, index=False)

    print("\n======== gene recall by regime x k (mean over seeds) ========")
    print(
        summ.pivot_table(
            index=["regime", "n_signal", "k_label"], columns="config", values="gene_recall_mean"
        )
        .round(3)
        .to_string()
    )
    print(
        "\n======== k* (smallest k with recall >= %.2f) and what auto chose ========"
        % RECALL_TARGET
    )
    print(kstar.to_string(index=False))
    print("\n--- prediction under test: k*(Sparse) > k*(Medium) > k*(Dense) ---")
    ks = kstar[kstar.config == "hybrid"].groupby("regime").k_star_recall90.mean()
    print(ks.to_string())

    with open(OUT / "crafted_recall_panel_summary.json", "w") as f:
        json.dump(
            {
                "k_grid": k_grid,
                "seeds": n_seeds,
                "recall_target": RECALL_TARGET,
                "n_datasets": len(sets),
                "rows": int(len(df)),
                "arms": {
                    "hybrid": "clusters top-k (default)",
                    "hybrid_cp5000": "B1 clusters global top-5000",
                    "auto": "B2 clusters its n_top_max pool",
                },
                "caveat": "nested gene lists on one base matrix; mechanism only",
                "k_star_by_regime": {k: (None if pd.isna(v) else float(v)) for k, v in ks.items()},
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}\nwrote {sp}\nwrote {kp}\nDONE")


if __name__ == "__main__":
    main()
