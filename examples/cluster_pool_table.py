#!/usr/bin/env python
"""Detailed comparison table for cluster_pool_panel.csv.

Emits absolute metric values per (dataset, arm) alongside the paired deltas and
their significance, rather than deltas alone -- a +0.007 margin means something
different on a dataset scoring 0.85 than on one scoring 0.25.

Only (dataset, seed) blocks with all three arms present are used, so every
comparison is paired. Datasets still running are included at whatever seed
count they have reached; the seed count is printed per dataset.

Outputs (examples/results/):
  cluster_pool_table.csv   long form, one row per (dataset, arm, metric, protocol)
  cluster_pool_table.md    the rendered tables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "results"
CSV = OUT / "cluster_pool_panel.csv"

BASE = "hvg2000"
ARMS = ["scfair2000", "scfair2000_cp5000"]
ALL_ARMS = [BASE] + ARMS
METRICS = ["macro_f1", "min_pop_f1", "ARI"]
PROTOCOLS = {"peak": "max", "mean": "mean"}
LABEL = {
    BASE: "scanpy",
    "scfair2000": "scFair (current)",
    "scfair2000_cp5000": "scFair cp5000",
}


COMMON_MAX_RES = 2.5  # see cluster_pool_analysis.COMMON_MAX_RES


def load() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    d = d[d["ARI"].notna()]
    # pool on the common grid: the extension only lengthens the censored
    # datasets, and "mean over resolution" is not the same quantity on a
    # longer grid
    d = d[d.resolution <= COMMON_MAX_RES]
    ok = d.groupby(["dataset", "seed"])["arm"].nunique().eq(len(ALL_ARMS))
    keep = ok[ok].index
    return d.set_index(["dataset", "seed"]).loc[keep].reset_index()


def per_seed(g: pd.DataFrame, arm: str, metric: str, how: str) -> pd.Series:
    s = g[g.arm == arm].groupby("seed")[metric]
    return s.max() if how == "max" else s.mean()


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> None:
    d = load()
    seeds = d.groupby("dataset").seed.nunique()
    rows, md = [], []

    md.append("# `cluster_pool=5000` panel — detailed comparison\n")
    md.append(
        "Paired per seed at matched k=2000. Absolute values are the mean over "
        "seeds of each seed's peak (or mean) over the resolution grid.\n"
    )
    md.append("Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 (paired t-test).\n")
    md.append("| dataset | seeds | " + " | ".join(f"n_clusters ({LABEL[a]})" for a in ARMS) + " |")
    md.append("|---|---|" + "---|" * len(ARMS))
    for ds, g in d.groupby("dataset"):
        cells = []
        for a in ARMS:
            sub = g[g.arm == a].drop_duplicates("seed")
            cells.append(
                f"{sub.n_clusters_total.mean():.1f} total / {sub.n_clusters_kept.mean():.1f} kept"
            )
        md.append(f"| {ds} | {seeds[ds]} | " + " | ".join(cells) + " |")
    md.append("")

    for proto, how in PROTOCOLS.items():
        md.append(f"\n## {proto}-over-resolution\n")
        for metric in METRICS:
            md.append(f"\n### {metric}\n")
            md.append(
                "| dataset | scanpy | scFair (current) | scFair cp5000 | "
                "scFair−scanpy | cp5000−scanpy | **cp5000−scFair** |"
            )
            md.append("|---|---:|---:|---:|---:|---:|---:|")
            for ds, g in d.groupby("dataset"):
                vals = {a: per_seed(g, a, metric, how) for a in ALL_ARMS}
                absol = [f"{vals[a].mean():.4f}" for a in ALL_ARMS]
                deltas = []
                for lhs, rhs in [
                    ("scfair2000", BASE),
                    ("scfair2000_cp5000", BASE),
                    ("scfair2000_cp5000", "scfair2000"),
                ]:
                    a, b = vals[lhs].align(vals[rhs], join="inner")
                    dd = a - b
                    p = stats.ttest_rel(a, b).pvalue if dd.std() > 0 else np.nan
                    deltas.append(f"{dd.mean():+.4f}{stars(p)} {int((dd > 0).sum())}/{len(dd)}")
                    rows.append(
                        {
                            "dataset": ds,
                            "metric": metric,
                            "protocol": proto,
                            "comparison": f"{lhs}-{rhs}",
                            "delta": dd.mean(),
                            "wins": int((dd > 0).sum()),
                            "n": len(dd),
                            "p": p,
                            "mean_lhs": a.mean(),
                            "mean_rhs": b.mean(),
                        }
                    )
                md.append(
                    f"| {ds} | "
                    + " | ".join(absol)
                    + " | "
                    + " | ".join(f"{x}" if i < 2 else f"**{x}**" for i, x in enumerate(deltas))
                    + " |"
                )
            # per-dataset sign tally: the §5.11 bar is stated in datasets, not seeds
            tal = []
            for lhs, rhs in [
                ("scfair2000", BASE),
                ("scfair2000_cp5000", BASE),
                ("scfair2000_cp5000", "scfair2000"),
            ]:
                per_ds = [
                    (per_seed(g, lhs, metric, how) - per_seed(g, rhs, metric, how)).mean()
                    for _, g in d.groupby("dataset")
                ]
                tal.append(f"{np.mean(per_ds):+.4f} {sum(x > 0 for x in per_ds)}/{len(per_ds)}")
            md.append(
                "| **pooled (datasets)** | | | | "
                + " | ".join(f"{x}" if i < 2 else f"**{x}**" for i, x in enumerate(tal))
                + " |"
            )

    md.append("\n## Grid truncation\n")
    md.append(
        "Fraction of seeds whose best resolution sits on a grid boundary; "
        "above ~0.25 the peak is censored and that row's peak margins are "
        "not trustworthy.\n"
    )
    md.append("| dataset | " + " | ".join(LABEL[a] for a in ALL_ARMS) + " | |")
    md.append("|---|---:|---:|---:|---|")
    for ds, g in d.groupby("dataset"):
        fr = []
        for a in ALL_ARMS:
            s = g[g.arm == a]
            lo, hi = s.resolution.min(), s.resolution.max()
            best = s.loc[s.groupby("seed")["macro_f1"].idxmax(), "resolution"]
            fr.append(float(((best == lo) | (best == hi)).mean()))
        flag = "**CENSORED**" if max(fr) > 0.25 else ""
        md.append(f"| {ds} | " + " | ".join(f"{x:.2f}" for x in fr) + f" | {flag} |")

    (OUT / "cluster_pool_table.md").write_text("\n".join(md) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "cluster_pool_table.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
