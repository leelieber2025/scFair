#!/usr/bin/env python
"""Does raising specificity weight (lowering `blend_global` below the
shipped 0.95) actually reduce cluster deprivation -- not just move ARI?

Motivated mid-session: §13 (DEVELOPMENT_LOG.md) swept `blend_global`
against downstream ARI/macro_f1 with `cap_allocation` stacked on top and
found no global value that helps both `duo4_pbmc` and `duo8_pbmc` --
closed. But that tested the wrong dependent variable for this question:
`blend_global` is a different lever from `cap_allocation` (which is now
OFF by default per the user's separate §5.31-32 decision) -- it sets how
much the *global ranking pool* weighs specificity, not a targeted
per-cluster fix. Cluster deprivation (§5.29's metric: some cluster gets
under 50% of its equal share of "own" genes) was never re-measured
directly against blend_global; ARI and deprivation are not the same
quantity, and cap_allocation's own ARI effect was modest even though it
targeted deprivation directly.

Method, reusing §5.29's own yardstick (`deprivation_topup.py`):
1. One `highly_variable_genes` call (defaults, blend_global=0.95) per
   (dataset, seed) builds the intermediate partition and its per-cluster
   one-sided logFC+ order (`_build_cluster_gene_ranks`) -- the fixed
   "whose gene is this really" yardstick every arm is graded against, so
   changing blend_global changes what's *selected*, not what counts as
   a cluster's "own" gene.
2. For each blend_global in {0.85, 0.90, 0.95, 0.98}: a fresh selection
   call at that blend_global, `peak_cluster_map` attributes each selected
   gene to its best-fit cluster by the fixed yardstick, counted per
   cluster, compared to equal_share = k / n_clusters.
3. Report: n_starved (< TRIGGER=0.5x equal_share, §5.29's own threshold),
   mean/min share ratio -- does starvation actually shrink as blend_global
   drops, or does it move ARI without moving deprivation (or vice versa)?

Same 10-dataset panel as `deprivation_topup.py`, 5 seeds.

Outputs (examples/results/):
  blend_global_deprivation.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402
from scfair.pp._highly_variable_genes import _build_cluster_gene_ranks  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "blend_global_deprivation.csv"

K = 2000
SEEDS = [0, 1, 2, 3, 4]
MIN_CLUSTER_SIZE = 30
TRIGGER = 0.5  # §5.29/deprivation_topup.py's own starvation threshold
ALPHAS = [0.85, 0.90, 0.95, 0.98]

ORDER = [
    "paul15",
    "pbmc3k_louvain",
    "pancreas_smartseq2",
    "duo4_pbmc",
    "duo8_pbmc",
    "duo4un_pbmc",
    "pbmc5k_adt29",
    "pbmc10k_adt14",
    "sln_208_mouse",
    "pbmc_seurat_v4_20k",
]


def peak_cluster_map(genes, ranks):
    """Attribute each gene to the cluster whose own logFC+ order ranks it best."""
    pos = {c: {g: i for i, g in enumerate(order_)} for c, order_ in ranks.items()}
    out = {}
    for g in genes:
        best_c, best_p = None, None
        for c, p_ in pos.items():
            p = p_.get(g)
            if p is not None and (best_p is None or p < best_p):
                best_p, best_c = p, c
        if best_c is not None:
            out[g] = best_c
    return out


def deprivation_stats(genes, ranks, k):
    n_clusters = len(ranks)
    equal_share = k / n_clusters
    peak = peak_cluster_map(genes, ranks)
    count = {c: 0 for c in ranks}
    for g, c in peak.items():
        count[c] += 1
    shares = {c: count[c] / equal_share for c in ranks}
    starved = [c for c, s in shares.items() if s < TRIGGER]
    return {
        "n_clusters": n_clusters,
        "n_starved": len(starved),
        "mean_share": float(np.mean(list(shares.values()))),
        "min_share": float(np.min(list(shares.values()))),
    }


def main(which=None):
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    done = {(r["dataset"], r["seed"], r["alpha"]) for r in rows}
    print(f"resuming: {len(done)} blocks done", flush=True)

    for name in which or ORDER:
        if all((name, s, a) in done for s in SEEDS for a in ALPHAS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        for seed in SEEDS:
            if all((name, seed, al) in done for al in ALPHAS):
                continue
            t0 = time.time()
            sel = a.copy()
            scf.pp.highly_variable_genes(
                sel,
                n_top_genes=K,
                random_state=seed,
                diagnose=False,
            )
            labels = sel.obs["scfair_hvg_clusters"]
            ranks = _build_cluster_gene_ranks(
                sel,
                cluster_labels=labels,
                counts_layer="counts",
                min_cluster_size=MIN_CLUSTER_SIZE,
                logfc_space="log1p",
            )
            if not ranks:
                print(f"  seed={seed}: no cluster ranks; skipping", flush=True)
                continue

            for alpha in ALPHAS:
                if (name, seed, alpha) in done:
                    continue
                ad = a.copy()
                scf.pp.highly_variable_genes(
                    ad,
                    n_top_genes=K,
                    balance_method="hybrid",
                    blend_global=alpha,
                    random_state=seed,
                    diagnose=False,
                )
                genes = [str(g) for g in ad.var_names[ad.var["highly_variable"]]]
                stats = deprivation_stats(genes, ranks, K)
                row = {"dataset": name, "seed": seed, "alpha": alpha, **stats}
                rows.append(row)
                print(
                    f"  seed={seed} alpha={alpha:.2f}  "
                    f"n_starved={stats['n_starved']}/{stats['n_clusters']}  "
                    f"mean_share={stats['mean_share']:.2f} "
                    f"min_share={stats['min_share']:.2f}",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f"    ({time.time() - t0:.0f}s)", flush=True)
        del a
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
