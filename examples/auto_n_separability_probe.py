#!/usr/bin/env python
"""Does population *separability* (not just count) predict the optimal
n_top_genes better than auto_n's existing signals?

Motivated by a hypothesis raised mid-investigation (2026-07-31): datasets
with many close/confusable real subpopulations should need more genes to
resolve them, while datasets with few well-separated types should plateau
early. `duo4_pbmc` (4 broad FACS types) plateaus by k~500; `duo8_pbmc` (8
types, several closely-related T-cell subsets) is still climbing at
k=4000, the top of the tested grid -- consistent with the hypothesis.
`crafted_base` (3 cell lines, but 71/21/8 imbalanced) doesn't fit cleanly,
suggesting imbalance is a separate confound from closeness/subtlety.

auto_n's existing methods (elbow/knee/cumfrac/coverage, §5.28) read only
the *univariate* global gene-variance curve -- structurally blind to
"how many real populations are there, and how close together are they."
`select_n_top_from_populations` (§5.28 §5) added population *count* but
that alone didn't predict optimal k either (per-metric disagreement,
never wired into the public API). This probe adds *separability*: reuse
the cap_allocation merge machinery's bootstrap pairwise stability score
(`_pair_bootstrap_stability`, §5.30 §7) -- already computed for every
nearest-neighbour cluster pair, just never used as an auto_n signal --
and see whether it correlates with the actual measured-optimal k better
than population count alone.

Expanded to the full panel k-swept in `auto_n_populations.csv` (all 10
canonical datasets + `crafted_base`, 11 total) per the user's instruction
to widen past the original n=4. `best_k_ari` is read dynamically from
that CSV (argmax mean ARI over k) rather than hardcoded.

Outputs (examples/results/):
  auto_n_separability_probe.csv
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
from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402
import scfair.pp._highly_variable_genes as _hvg  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "auto_n_separability_probe.csv"
POPULATIONS_CSV = OUT / "auto_n_populations.csv"

DATASETS = [
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
    "crafted_base",
]
SEEDS = [0, 1, 2]

# measured optimum (ARI-argmax) read live from today's real k-sweep
_pop_df = pd.read_csv(POPULATIONS_CSV)
_k_means = _pop_df.groupby(["dataset", "k"])["ARI"].mean().reset_index()
BEST_K_ARI = {d: int(sub.loc[sub["ARI"].idxmax(), "k"]) for d, sub in _k_means.groupby("dataset")}
SHADOWED = {"paul15", "pbmc3k_louvain", "pancreas_smartseq2"}

_orig = _hvg._pair_bootstrap_stability
_log: list[float] = []


def _wrapped(X_pca, mask_a, mask_b, **kw):
    score = _orig(X_pca, mask_a, mask_b, **kw)
    _log.append(float(score))
    return score


def main():
    rows = []
    _hvg._pair_bootstrap_stability = _wrapped
    try:
        for name in DATASETS:
            if name not in BEST_K_ARI:
                print(f"### {name}: no k-sweep data yet, skipping", flush=True)
                continue
            print(f"\n######## {name} ########", flush=True)
            a = LOADERS[name]()
            for seed in SEEDS:
                _log.clear()
                ad = a.copy()
                scf.pp.highly_variable_genes(
                    ad,
                    n_top_genes=2000,
                    flavor="seurat_v3",
                    layer="counts",
                    marker_mode="none",
                    balance_method="hybrid",
                    random_state=seed,
                    diagnose=True,
                    cap_allocation=True,
                    cap_merge_threshold=0.5,
                )
                n_pop = ad.uns["scfair"]["hvg"]["clustering"]["n_clusters_kept"]
                scores = np.array(_log)
                row = {
                    "dataset": name,
                    "circular": name in SHADOWED,
                    "seed": seed,
                    "n_pairs": len(scores),
                    "n_populations": n_pop,
                    "mean_stability": float(scores.mean()) if len(scores) else np.nan,
                    "min_stability": float(scores.min()) if len(scores) else np.nan,
                    "frac_below_0.5": float((scores < 0.5).mean()) if len(scores) else np.nan,
                    "best_k_ari": BEST_K_ARI[name],
                }
                rows.append(row)
                print(
                    f"  seed={seed}  n_pop={n_pop}  n_pairs={len(scores)}  "
                    f"mean_stab={row['mean_stability']:.3f}  "
                    f"min_stab={row['min_stability']:.3f}  "
                    f"frac<0.5={row['frac_below_0.5']:.2f}",
                    flush=True,
                )
            del a
    finally:
        _hvg._pair_bootstrap_stability = _orig

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print(f"\nwrote {CSV} ({len(df)} rows)")

    summary = df.groupby("dataset")[
        ["n_populations", "mean_stability", "min_stability", "frac_below_0.5", "best_k_ari"]
    ].mean()
    print("\nper-dataset means:")
    print(summary.round(3))

    print("\ncorrelation with best_k_ari (n=4 datasets, directional only):")
    for col in ["n_populations", "mean_stability", "min_stability", "frac_below_0.5"]:
        r = summary[col].corr(summary["best_k_ari"], method="spearman")
        print(f"  spearman(best_k_ari, {col}) = {r:.3f}")


if __name__ == "__main__":
    main()
