#!/usr/bin/env python
"""Analyse cluster_pool_panel.csv -- runs on partial results.

Reports, per dataset and pooled, the two arms against the scanpy baseline and
against each other, under the §5.16 protocol (peak- and mean-over-resolution,
plus the legacy single point for comparison only).

Flags grid truncation explicitly: if a seed's argmax resolution sits on a grid
boundary, that arm's peak is censored and the margin is not trustworthy. That
is the caveat §5.16.6 had to attach after the fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path(__file__).resolve().parent / "results" / "cluster_pool_panel.csv"
METRICS = ["macro_f1", "min_pop_f1", "ARI"]
BASE = "hvg2000"
ARMS = ["scfair2000", "scfair2000_cp5000"]

# The grid extension (cluster_pool_extend.py) only targets the peak-censored
# datasets, so their grids run to 6.0 while the rest stop at 2.5. Per-dataset
# deltas stay valid either way -- all three arms share a dataset's grid -- but
# "mean over resolution" means a different thing on a longer grid, so pooling
# mixed grids silently reweights the average. Default to the common grid; pass
# a larger --max-res to inspect the de-censoring on the extended datasets.
COMMON_MAX_RES = 2.5


def paired(a: pd.Series, b: pd.Series) -> tuple[float, int, int, float]:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 2:
        return np.nan, 0, len(j), np.nan
    d = j["a"] - j["b"]
    return d.mean(), int((d > 0).sum()), len(d), stats.ttest_rel(j["a"], j["b"]).pvalue


def agg(df: pd.DataFrame, arm: str, metric: str, how: str) -> pd.Series:
    s = df[df.arm == arm]
    if how == "peak":
        return s.groupby("seed")[metric].max()
    if how == "mean":
        return s.groupby("seed")[metric].mean()
    return s[s.resolution == 0.8].set_index("seed")[metric]


def edge_fraction(df: pd.DataFrame, arm: str, metric: str) -> float:
    """Fraction of seeds whose best resolution is on a grid boundary."""
    s = df[df.arm == arm]
    if s.empty:
        return np.nan
    lo, hi = s.resolution.min(), s.resolution.max()
    best = s.loc[s.groupby("seed")[metric].idxmax(), "resolution"]
    return float(((best == lo) | (best == hi)).mean())


def main() -> None:
    max_res = COMMON_MAX_RES
    for a in sys.argv[1:]:
        if a.startswith("--max-res="):
            max_res = float(a.split("=", 1)[1])

    d = pd.read_csv(CSV)
    d = d[d.get("ARI").notna()] if "ARI" in d else d
    grids = d.groupby("dataset").resolution.max()
    d = d[d.resolution <= max_res]
    print(f"resolution grid capped at {max_res} (per-dataset maxima on disk: {grids.to_dict()})")
    complete = d.groupby(["dataset", "seed"])["arm"].nunique().eq(len(ARMS) + 1)
    keep = complete[complete].index
    d = d.set_index(["dataset", "seed"]).loc[keep].reset_index()
    print(f"{len(d)} rows; complete (dataset, seed) blocks: {len(keep)}")
    print(f"datasets: {sorted(d.dataset.unique())}")
    print(f"seeds per dataset: {d.groupby('dataset').seed.nunique().to_dict()}\n")

    for how in ["peak", "mean", "res0.8"]:
        print(f"\n{'=' * 74}\n  {how}-over-resolution\n{'=' * 74}")
        for metric in METRICS:
            print(f"\n--- {metric} ---")
            print(
                f"{'dataset':22s} {'scfair vs hvg':>22s} {'cp5000 vs hvg':>22s} "
                f"{'cp5000 vs scfair':>22s}"
            )
            pooled: dict[str, list] = {a: [] for a in ARMS + ["delta"]}
            for ds, g in d.groupby("dataset"):
                cells = []
                for arm in ARMS:
                    mu, w, n, p = paired(agg(g, arm, metric, how), agg(g, BASE, metric, how))
                    cells.append(f"{mu:+.4f} {w}/{n} p={p:.2g}")
                    pooled[arm].append(agg(g, arm, metric, how) - agg(g, BASE, metric, how))
                mu, w, n, p = paired(
                    agg(g, "scfair2000_cp5000", metric, how),
                    agg(g, "scfair2000", metric, how),
                )
                cells.append(f"{mu:+.4f} {w}/{n} p={p:.2g}")
                pooled["delta"].append(
                    agg(g, "scfair2000_cp5000", metric, how) - agg(g, "scfair2000", metric, how)
                )
                print(f"{ds:22s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")

            # dataset-level sign counts: the §5.11 bar is "wins on N/8 datasets"
            line = []
            for key in ARMS + ["delta"]:
                means = [s.mean() for s in pooled[key]]
                line.append(f"{np.mean(means):+.4f} {sum(m > 0 for m in means)}/{len(means)}")
            print(f"{'POOLED (per-dataset)':22s} {line[0]:>22s} {line[1]:>22s} {line[2]:>22s}")

    print(
        f"\n{'=' * 74}\n  grid truncation check (fraction of seeds peaking on a boundary)\n{'=' * 74}"
    )
    print(f"{'dataset':22s} " + " ".join(f"{a:>20s}" for a in [BASE] + ARMS))
    for ds, g in d.groupby("dataset"):
        fr = [edge_fraction(g, a, "macro_f1") for a in [BASE] + ARMS]
        flag = "  <-- CENSORED" if max(fr) > 0.25 else ""
        print(f"{ds:22s} " + " ".join(f"{f:>20.2f}" for f in fr) + flag)

    if "n_clusters_dropped" in d.columns:
        print(f"\n{'=' * 74}\n  intermediate clustering (new diagnostics)\n{'=' * 74}")
        sub = d[d.arm != BASE].drop_duplicates(["dataset", "arm", "seed"])
        print(
            sub.groupby(["dataset", "arm"])[
                [
                    "cluster_pool_effective",
                    "n_clusters_total",
                    "n_clusters_kept",
                    "n_clusters_dropped",
                ]
            ]
            .mean()
            .round(2)
            .to_string()
        )


if __name__ == "__main__":
    sys.exit(main())
