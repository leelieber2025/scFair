#!/usr/bin/env python
"""Ablation: intermediate-clustering scaling and the fold-change space.

One factor at a time, deliberately kept out of the k-sweep panels. §5.15 conflated
sparsity regime with perturbation type and could not attribute the result; mixing
these two knobs into a 7-arm panel would repeat that mistake.

Factors (DEVELOPMENT_LOG §5.16), each measured against the shipped default:

  baseline            resolution=0.5, no scaling, logfc_space="log1p"
  +scale              scale_clustering=True
  linear              logfc_space="linear"              (scanpy convention, pseudo 1e-9)
  linear_regularised  logfc_space="linear_regularised"  (pseudo 1.0)

Why these are worth measuring rather than "fixing":

* Omitting ``sc.pp.scale`` before the intermediate PCA changes the intermediate
  populations themselves — ARI 0.885 between on and off — while this repo's
  *evaluation* pipeline does scale. Nobody had quantified that asymmetry.
* The shipped contrast is ``log2(mean log1p / mean log1p)``, not a linear-space
  fold change. The two disagree on **more than half** the top-2000 specificity
  genes (Spearman 0.465, 967/2000 shared). But the conventional version pulls in
  genes with a mean zero-fraction of 0.998, because in linear space ``mu_out``
  approaches zero and scanpy's 1e-9 epsilon barely regularises it. So the shipped
  deviation may be the better-behaved choice, and the third arm exists to separate
  "log vs linear" from "pseudocount strength".

Both knobs default to off/log1p and are bit-identical to the previous behaviour
(test-enforced), so this measures alternatives, it does not change anything.

Pre-registered decision rule — fixed before running, primary metric macro-F1
(§5.11; ARI is size-weighted and would under-weight the sparse-gene effects these
knobs act on):

    delta macro-F1 >= +0.010    consider a default change, then the §5.11 bar
    delta macro-F1 <= -0.010    alternative clearly worse; shipped default vindicated
    |delta| 0.005 - 0.010       keep off; document as a data-dependent knob
    |delta| < 0.005             close the line; documentation only

Labels are non-circular throughout: FACS-sorted (duo*) and protein-derived (adt14).

Usage
-----
  python examples/ablation_scale_logfc.py --smoke
  python examples/ablation_scale_logfc.py            # full, resumable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import scanpy as sc

import scfair as scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sorted_gold_panel import (  # noqa: E402
    OUT,
    append_row,
    evaluate,
    load_dataset,
    load_done,
    read_rows,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

K = 2000  # fixed: this ablation is about the knobs, not about k
DATASETS = ["duo4_pbmc", "duo8_pbmc", "duo4un_pbmc", "adt14"]
ARMS = {
    "baseline": {},
    "scale": dict(scale_clustering=True),
    "linear": dict(logfc_space="linear"),
    "linear_regularised": dict(logfc_space="linear_regularised"),
}
DECISION = {"consider_default": 0.010, "keep_as_knob": 0.005}


def select_genes(adata, arm: str, *, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=min(K, a.n_vars - 1),
        balance_method="hybrid",
        blend_global=0.95,
        resolution=0.5,
        neighbor_contrast=0.0,
        flavor="seurat_v3",
        layer="counts",
        marker_mode="none",
        random_state=seed,
        progress=False,
        **ARMS[arm],
    )
    meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
    for drop in ("selected_genes", "cluster_weights", "auto_n"):
        meta.pop(drop, None)
    return a.var_names[a.var["highly_variable"]].astype(str).tolist(), meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--smoke", action="store_true", help="2 seeds, duo4 only")
    p.add_argument("--datasets", default=",".join(DATASETS))
    args = p.parse_args(argv)

    datasets = (
        ["duo4_pbmc"] if args.smoke else [d.strip() for d in args.datasets.split(",") if d.strip()]
    )
    out_path = OUT / (
        "ablation_scale_logfc_smoke.csv" if args.smoke else "ablation_scale_logfc.csv"
    )

    print(f"Ablation | k={K} | arms={list(ARMS)} | datasets={datasets}", flush=True)
    print(
        "Pre-registered: primary macro-F1; |delta|>=0.010 consider default, "
        ">=0.005 keep as knob, else close.",
        flush=True,
    )

    done = load_done(out_path)
    t0, n_run = time.time(), 0
    for name in datasets:
        adata, info = load_dataset(name)
        n_seeds = 2 if args.smoke else info["n_seeds"]
        print(
            f"\n==== {name} | {adata.n_obs}x{adata.n_vars} | {info['label_kind']} "
            f"| seeds={n_seeds} ====",
            flush=True,
        )
        for arm in ARMS:
            for seed in range(n_seeds):
                key = (name, str(K), arm, seed)
                if key in done:
                    continue
                row = {
                    "dataset": name,
                    "tier": info["tier"],
                    "label_kind": info["label_kind"],
                    "k_label": str(K),
                    "config": arm,
                    "seed": seed,
                    "n_cells": int(adata.n_obs),
                }
                try:
                    genes, meta = select_genes(adata, arm, seed=seed)
                    row.update(
                        n_top_used=meta.get("n_top_genes_used", len(genes)),
                        scale_clustering=meta.get("scale_clustering"),
                        logfc_space=meta.get("logfc_space"),
                        **evaluate(adata, genes, info, seed=seed),
                    )
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                append_row(out_path, row)
                done.add(key)
                n_run += 1
                if n_run % 5 == 0:
                    print(
                        f"  [{n_run}] {name} {arm} s={seed} "
                        f"ARI={row.get('ARI', float('nan')):.4f} "
                        f"mF1={row.get('macro_f1', float('nan')):.4f} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )

    df = read_rows(out_path)
    if df.empty:
        print("no rows")
        return
    df.to_csv(out_path, index=False)
    ok = df.dropna(subset=["macro_f1"])
    summ = ok.groupby(["dataset", "config"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ARI=("ARI", "mean"),
        macro_f1=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        min_pop_f1=("min_pop_f1", "mean"),
    )
    summ.to_csv(OUT / "ablation_scale_logfc_summary.csv", index=False)
    print("\n======== macro-F1 by dataset x arm ========")
    print(
        summ.pivot_table(index="dataset", columns="config", values="macro_f1").round(4).to_string()
    )
    print("\n======== min-population F1 ========")
    print(
        summ.pivot_table(index="dataset", columns="config", values="min_pop_f1")
        .round(4)
        .to_string()
    )

    base = summ[summ.config == "baseline"].set_index("dataset")
    verdict = {}
    print("\n======== pre-registered decision ========")
    for arm in ARMS:
        if arm == "baseline":
            continue
        cur = summ[summ.config == arm].set_index("dataset")
        shared = base.index.intersection(cur.index)
        d = float((cur.loc[shared, "macro_f1"] - base.loc[shared, "macro_f1"]).mean())
        wins = int((cur.loc[shared, "macro_f1"] > base.loc[shared, "macro_f1"]).sum())
        # Sign matters: a large *negative* delta means the alternative is worse,
        # i.e. the shipped default is vindicated — not a reason to change it.
        if d >= DECISION["consider_default"]:
            call = "CONSIDER DEFAULT CHANGE (then the §5.11 bar)"
        elif -d >= DECISION["consider_default"]:
            call = "alternative is clearly WORSE; current default vindicated"
        elif abs(d) >= DECISION["keep_as_knob"]:
            call = "keep off; document as a data-dependent knob"
        else:
            call = "CLOSE the line; documentation only"
        verdict[arm] = {
            "delta_macro_f1": round(d, 4),
            "wins": f"{wins}/{len(shared)}",
            "call": call,
        }
        print(f"  {arm:20s} delta macro-F1 = {d:+.4f}  wins {wins}/{len(shared)}  -> {call}")

    with open(OUT / "ablation_scale_logfc_summary.json", "w") as f:
        json.dump(
            {
                "k": K,
                "arms": {a: str(v) for a, v in ARMS.items()},
                "decision_rule": DECISION,
                "verdict": verdict,
                "rows": int(len(df)),
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}\nDONE")


if __name__ == "__main__":
    main()
