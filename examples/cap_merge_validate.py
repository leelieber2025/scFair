#!/usr/bin/env python
"""Validate `cap_merge_threshold` (§5.30 §7, landed) through the real API:
does merging unstable nearest-neighbour pairs before `cap_allocation`
computes equal share actually help, on top of `cap_allocation` itself?

Arms
----
- ``nocap`` -- ``cap_allocation=False``.
- ``cap_nomerge`` -- ``cap_allocation=True, cap_merge_threshold=None``
  (§5.30 §1-6's shipped behaviour before this follow-up).
- ``cap_merge`` -- ``cap_allocation=True, cap_merge_threshold=0.5``
  (the new default).

Outputs (examples/results/):
  cap_merge_validate.csv       one row per (dataset, arm, seed, resolution)
  cap_merge_validate_pops.csv  per-population, long format
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "cap_merge_validate.csv"
POPS = OUT / "cap_merge_validate_pops.csv"

K = 2000
SEEDS = [0, 1, 2, 3, 4]
RES_GRID = [0.3, 0.5, 0.8, 1.2]
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
ARMS = {
    "nocap": dict(cap_allocation=False),
    "cap_nomerge": dict(cap_allocation=True, cap_merge_threshold=None),
    "cap_merge": dict(cap_allocation=True, cap_merge_threshold=0.5),
}


def evaluate(a, genes, seed, rows, pop_rows, base):
    e = a.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    e = e[:, [g for g in genes if g in e.var_names]].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    conf = (
        e.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in e.obs.columns
        else np.ones(e.n_obs, dtype=bool)
    )
    y_true = e.obs["cell_type"].astype(str)[conf]
    prev = y_true.value_counts(normalize=True)

    for res in RES_GRID:
        sc.tl.leiden(
            e, resolution=res, key_added="L", flavor="igraph", n_iterations=2, random_state=seed
        )
        y_pred = e.obs["L"].astype(str)[conf]
        f1s = {}
        for pop in y_true.unique():
            t = (y_true == pop).to_numpy()
            best = 0.0
            for cl in y_pred.unique():
                p = (y_pred == cl).to_numpy()
                tp = float(np.sum(t & p))
                if tp:
                    pr, rc = tp / p.sum(), tp / t.sum()
                    best = max(best, 2 * pr * rc / (pr + rc))
            f1s[pop] = best
        for pop, f1 in f1s.items():
            pop_rows.append(
                {
                    "dataset": base["dataset"],
                    "arm": base["arm"],
                    "seed": base["seed"],
                    "resolution": res,
                    "population": pop,
                    "prevalence": float(prev.get(pop, np.nan)),
                    "f1": float(f1),
                }
            )
        rows.append(
            {
                **base,
                "resolution": res,
                "n_leiden": int(y_pred.nunique()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
            }
        )


def main():
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pop_rows = pd.read_csv(POPS).to_dict("records") if POPS.exists() else []
    seen: dict[tuple, set] = {}
    for r in rows:
        seen.setdefault((r["dataset"], r["seed"]), set()).add(r["arm"])
    done = {k for k, v in seen.items() if set(ARMS) <= v}
    print(f"resuming: {len(done)} (dataset, seed) blocks done", flush=True)

    for name in ORDER:
        if all((name, s) in done for s in SEEDS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        a = LOADERS[name]()
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.time()
            merges_seen = {}
            for arm, kwargs in ARMS.items():
                ad = a.copy()
                scf.pp.highly_variable_genes(
                    ad,
                    n_top_genes=K,
                    balance_method="hybrid",
                    random_state=seed,
                    diagnose=False,
                    **kwargs,
                )
                genes = list(ad.var_names[ad.var["highly_variable"]])
                if arm == "cap_merge":
                    cl = ad.uns.get("scfair", {}).get("hvg", {}).get("clustering", {})
                    merges_seen = cl.get("cap_merges")
                base = {"dataset": name, "arm": arm, "seed": seed, "n_genes": len(genes)}
                evaluate(a, genes, seed, rows, pop_rows, base)
            done.add((name, seed))
            pd.DataFrame(rows).to_csv(CSV, index=False)
            pd.DataFrame(pop_rows).to_csv(POPS, index=False)
            got = {
                arm: np.mean(
                    [
                        r["ARI"]
                        for r in rows
                        if r["dataset"] == name and r["seed"] == seed and r["arm"] == arm
                    ]
                )
                for arm in ARMS
            }
            print(
                f"  seed={seed}  ARI nocap={got['nocap']:.3f} "
                f"cap_nomerge={got['cap_nomerge']:.3f} cap_merge={got['cap_merge']:.3f}  "
                f"merges={merges_seen}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        del a
    print(f"\nwrote {CSV} ({len(rows)} rows) and {POPS} ({len(pop_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
