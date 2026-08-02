#!/usr/bin/env python
"""Extend the resolution grid past 2.5 for the three peak-censored datasets.

§5.19.4: even with the grid at 0.2-2.5, three of eight datasets have a large
fraction of seeds whose best resolution sits on the top boundary --
Cao (0.80 on the current arm), pancreas_smartseq2 (0.55 on cp5000) and
pbmc_seurat_v4_20k (0.35 on scanpy). Their peak margins are censored, and
because censoring is arm-specific it biases the comparison asymmetrically.

This appends the missing high-resolution points to the same CSV. The panel is
long-form with a `resolution` column, so new rows simply extend each block and
every downstream script picks them up with no changes. Gene selection is
deterministic given (dataset, arm, seed), so the added points are paired with
the existing ones by construction.

Does not revisit §5.19's verdict, which rests on mean-over-resolution -- that
protocol never depended on where the peak sits. This only makes the *peak*
numbers quotable for these three.

Resumable per (dataset, seed, arm, resolution).

Output: appends to examples/results/cluster_pool_panel.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cluster_pool_panel import (  # noqa: E402
    ARMS,
    CSV,
    LOADERS,
    SHADOWED,
    evaluate,
    select,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# The existing grid tops out at 2.5. Cao's current arm peaks on that edge on
# 16/20 seeds, so the true optimum may be well beyond it; go wide enough that a
# second censoring round is unlikely.
NEW_RES = [3.0, 3.5, 4.0, 5.0, 6.0]
# Cao was censored too (0.80 on the current arm) but is excluded from the panel
# entirely as of §5.21 -- see cluster_pool_panel.EXCLUDED.
DATASETS = ["pancreas_smartseq2", "pbmc_seurat_v4_20k"]
SEEDS = list(range(20))


def main() -> None:
    existing = pd.read_csv(CSV)
    rows = existing.to_dict("records")
    done = {
        (r["dataset"], r["seed"], r["arm"], float(r["resolution"]))
        for r in rows
        if pd.notna(r.get("resolution"))
    }
    print(f"CSV has {len(existing)} rows; extending grid with {NEW_RES}", flush=True)

    for dname in DATASETS:
        todo = [
            (s, arm)
            for s in SEEDS
            for arm in ARMS
            if any((dname, s, arm, r) not in done for r in NEW_RES)
        ]
        if not todo:
            print(f"### {dname}: already extended, skipping", flush=True)
            continue
        print(f"\n############ {dname} ({len(todo)} blocks) ############", flush=True)
        adata = LOADERS[dname]()
        print(f"  {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

        for seed, arm in todo:
            t0 = time.time()
            missing = [r for r in NEW_RES if (dname, seed, arm, r) not in done]
            try:
                genes, meta = select(adata, arm, seed)
                new_rows: list[dict] = []
                evaluate(
                    adata,
                    genes,
                    seed,
                    new_rows,
                    base={
                        "dataset": dname,
                        "arm": arm,
                        "seed": seed,
                        "circular": dname in SHADOWED,
                        **meta,
                    },
                    res_grid=missing,
                )
                rows.extend(new_rows)
                print(
                    f"  {arm:18s} seed={seed:2d} +{len(new_rows)} res points "
                    f"({time.time() - t0:5.1f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
            pd.DataFrame(rows).to_csv(CSV, index=False)
        print(f"===== {dname} EXTENDED =====", flush=True)

    print("GRID EXTENSION DONE", flush=True)


if __name__ == "__main__":
    main()
