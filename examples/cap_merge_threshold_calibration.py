#!/usr/bin/env python
"""Re-check `cap_merge_threshold=0.5` against the *current* (single-pass,
non-chained) `_merge_unstable_clusters` -- the original 0.5 was eyeballed
from §7's table, gap between stable pairs (0.72-1.00) and unstable/fragment
pairs (-0.07-0.16), using the *prototype* chained merge. The scoring
function (`_pair_bootstrap_stability`) itself didn't change in the fix, but
it's never been looked at again since the cascade bug was found and fixed,
so re-derive it honestly rather than assume it still holds.

Monkeypatches `_pair_bootstrap_stability` to log *every* nearest-neighbour
candidate pair it scores (not just the ones that end up merged -- the
public `cap_merges` diagnostic only reports post-threshold), tagged with
each side's dominant ground-truth type and purity so we can tell real
pairs (different type, or same type but both legitimately real) from
fragment pairs (same type, low purity split) independent of the
threshold being tested.

3 datasets -- the ones §7's table actually had unstable/fragment examples
on: pancreas_smartseq2 (alpha fragments), sln_208_mouse (p4_cDC2s,
p0_CD8T), pbmc_seurat_v4_20k (CD4Naive, CD14Mono). duo4/duo8 are left out
here on purpose -- they were the zero-false-positive controls (all pairs
0.98-1.00), not part of the gap this threshold was set from.

Outputs (examples/results/):
  cap_merge_threshold_calibration.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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
CSV = OUT / "cap_merge_threshold_calibration.csv"

DATASETS = ["pancreas_smartseq2", "sln_208_mouse", "pbmc_seurat_v4_20k"]
SEEDS = [0, 1, 2]

_orig = _hvg._pair_bootstrap_stability
_log: list[dict] = []
_current: dict = {}


def _wrapped(X_pca, mask_a, mask_b, **kw):
    score = _orig(X_pca, mask_a, mask_b, **kw)
    ct = _current["cell_type"]
    ta = pd.Series(ct[mask_a]).value_counts(normalize=True)
    tb = pd.Series(ct[mask_b]).value_counts(normalize=True)
    _log.append(
        {
            "dataset": _current["dataset"],
            "seed": _current["seed"],
            "size_a": int(mask_a.sum()),
            "size_b": int(mask_b.sum()),
            "type_a": str(ta.index[0]),
            "purity_a": float(ta.iloc[0]),
            "type_b": str(tb.index[0]),
            "purity_b": float(tb.iloc[0]),
            "same_type": bool(ta.index[0] == tb.index[0]),
            "score": float(score),
        }
    )
    return score


def main():
    _hvg._pair_bootstrap_stability = _wrapped
    try:
        for name in DATASETS:
            print(f"\n######## {name} ########", flush=True)
            a = LOADERS[name]()
            for seed in SEEDS:
                _current["dataset"] = name
                _current["seed"] = seed
                _current["cell_type"] = a.obs["cell_type"].astype(str).to_numpy()
                n_before = len(_log)
                scf.pp.highly_variable_genes(
                    a.copy(),
                    n_top_genes=2000,
                    flavor="seurat_v3",
                    layer="counts",
                    marker_mode="none",
                    balance_method="hybrid",
                    random_state=seed,
                    cap_allocation=True,
                    cap_merge_threshold=0.5,
                    diagnose=False,
                )
                print(f"  seed={seed}: {len(_log) - n_before} candidate pairs scored", flush=True)
            pd.DataFrame(_log).to_csv(CSV, index=False)
    finally:
        _hvg._pair_bootstrap_stability = _orig

    df = pd.DataFrame(_log)
    print(f"\nwrote {CSV} ({len(df)} rows)")
    print("\nsame_type pairs (candidate fragments) sorted by score:")
    print(
        df[df.same_type]
        .sort_values("score")[
            ["dataset", "seed", "type_a", "purity_a", "type_b", "purity_b", "score"]
        ]
        .to_string(index=False)
    )
    print("\ndifferent_type pairs (should never merge) sorted by score:")
    print(
        df[~df.same_type]
        .sort_values("score")[["dataset", "seed", "type_a", "type_b", "score"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
