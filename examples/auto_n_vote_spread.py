#!/usr/bin/env python
"""Do auto-n's raw votes carry any dataset-to-dataset signal at all?

The question, and why it is answerable at n=12
-----------------------------------------------
The measured optimum for `n_top_genes` is not one number. From the archived
k-sweeps: ADT panel k=1000, Cao rare panel k=2000, pbmc_seurat_v4_20k k=3000 --
a 3x spread. `n_top_genes="auto"` returns ~2000 on almost everything (§5.13:
2000 on 6/8 datasets, statistically indistinguishable from fixed 2000).

Reading `_auto_n.py:399` shows why that is structural rather than empirical: the
ensemble injects an **anchor of 2000 as a vote** and clips every vote into
`[k_floor, k_ceiling=2500]`, so the median is pinned near 2000 by construction.

Direction A in the improvement list is "remove the anchor and the ceiling so it
can leave the 2000 neighbourhood at all". That is only worth doing if the raw
votes carry signal for the anchor to be suppressing. Correlating votes against
measured optima would need n>=20 datasets (§5.14's bar). But a weaker question
is decisive at n=12 and needs no labels:

    **do the raw votes vary across datasets at all?**

If elbow / knee / cumfrac / coverage return near-identical k on 12 datasets
spanning 2.4k-30.7k cells, 3.4k-20k genes, two species and four tissues, then
there is no signal for the anchor to be hiding, and A cannot work no matter how
the votes are combined.

Cheap: the votes are functions of the global HVG score curve, so this runs one
`highly_variable_genes(flavor="seurat_v3")` pass per dataset and no clustering.

Outputs (examples/results/):
  auto_n_vote_spread.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from umap3d_smoke import LOADERS, ORDER  # noqa: E402

from scfair.pp._auto_n import (  # noqa: E402
    resolve_n_top_genes,
    select_n_top_cumfrac,
    select_n_top_elbow,
    select_n_top_knee,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
CSV = OUT / "auto_n_vote_spread.csv"

K_MIN, K_MAX = 500, 5000


def run_one(name: str) -> dict:
    a = LOADERS[name]()
    sc.pp.highly_variable_genes(
        a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
    )
    v = a.var
    col = "variances_norm" if "variances_norm" in v else "dispersions_norm"
    scores = np.asarray(v[col], dtype=float)
    order = np.argsort(-scores)
    scores_desc = scores[order]
    gene_order = list(v.index[order].astype(str))

    row = {"dataset": name, "n_cells": int(a.n_obs), "n_genes": int(a.n_vars)}
    row["elbow"] = select_n_top_elbow(scores_desc, k_min=K_MIN, k_max=K_MAX)
    row["knee"] = select_n_top_knee(scores_desc, k_min=K_MIN, k_max=K_MAX)
    row["cumfrac"] = select_n_top_cumfrac(scores_desc, k_min=K_MIN, k_max=K_MAX)
    # `coverage` is deliberately absent: it is the only estimator that looks at
    # cluster structure, and it needs `cluster_gene_ranks` -- i.e. a clustering
    # that does not exist yet. That requirement is exactly why `auto` runs the
    # intermediate clustering twice (§5.17 B4).

    k, meta = resolve_n_top_genes(
        "auto",
        scores_desc,
        gene_order,
        k_min=K_MIN,
        k_max=K_MAX,
        adata=a,
    )
    row["ensemble_shipped"] = int(k)
    row["anchor_used"] = meta.get("anchor")
    row["k_floor"] = meta.get("k_floor")
    row["k_ceiling"] = meta.get("k_ceiling")

    # Direction A, simulated: no anchor, no soft ceiling
    k_free, _ = resolve_n_top_genes(
        "auto",
        scores_desc,
        gene_order,
        k_min=K_MIN,
        k_max=K_MAX,
        adata=a,
        anchor=None,
        soft_ceiling=K_MAX,
        depth_aware=False,
    )
    row["ensemble_no_anchor"] = int(k_free)
    return row


def main(which=None) -> None:
    rows = []
    for name in which or ORDER:
        try:
            rows.append(run_one(name))
            print("  " + "  ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    d = pd.DataFrame(rows)
    print("\n=== spread across datasets (the whole question) ===", flush=True)
    for c in ("elbow", "knee", "cumfrac", "ensemble_shipped", "ensemble_no_anchor"):
        if c in d and d[c].notna().any():
            s = d[c].dropna()
            print(
                f"  {c:20s} min={s.min():>5.0f} median={s.median():>6.0f} "
                f"max={s.max():>5.0f}  ratio={s.max() / max(s.min(), 1):.2f}x",
                flush=True,
            )
    print(f"\nwrote {CSV}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
