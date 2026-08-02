#!/usr/bin/env python
"""Re-validate `cap_allocation` through the actual shipped API, not the
`examples/deprivation_topup.py` standalone reimplementation.

Why this exists
----------------
`deprivation_topup.py`'s `cap_select` backfilled from the *entire* gene
universe by `scfair_score`. Landing the mechanism in
`scfair/pp/_highly_variable_genes.py` (`_cap_over_represented`) surfaced a
real bug the standalone script never could: hybrid's defining invariant is
that the selection never leaves the global top-``2*n_top`` candidate pool,
and unrestricted backfill broke it (caught by
``test_hybrid_pool_is_the_true_top_2k``). Backfill is now restricted to that
same pool, which can only make it *more* conservative than what
`deprivation_topup.py` measured -- this script re-runs the panel against the
real, pool-restricted, in-package behaviour to get the authoritative numbers.

Arms
----
- ``nocap`` -- ``scf.pp.highly_variable_genes(..., cap_allocation=False)``.
- ``cap`` -- same call, ``cap_allocation=True`` (the new default).

Both otherwise default (``n_top_genes=2000``, ``balance_method="hybrid"``).

Also sweeps ``flavor`` (``seurat_v3`` default, plus ``cell_ranger`` and
``seurat_v3_paper``) on a smaller subset, since ``cap_allocation``'s
cell-cycle guard and backfill pool are keyed off whatever `flavor` produces
-- worth checking it is not a seurat_v3-only effect. ``seurat`` needs
log-normalized input (different preprocessing contract), skipped here.

Outputs (examples/results/):
  cap_allocation_validate.csv       full panel, flavor=seurat_v3
  cap_allocation_validate_pops.csv  per-population, flavor=seurat_v3
  cap_allocation_flavors.csv        flavor sweep, subset of datasets
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
CSV = OUT / "cap_allocation_validate.csv"
POPS = OUT / "cap_allocation_validate_pops.csv"
FLAVOR_CSV = OUT / "cap_allocation_flavors.csv"

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
FLAVOR_SUBSET = ["pancreas_smartseq2", "duo4_pbmc", "duo8_pbmc", "pbmc10k_adt14"]
FLAVORS = ["seurat_v3", "cell_ranger", "seurat_v3_paper"]


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


def run_panel():
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pop_rows = pd.read_csv(POPS).to_dict("records") if POPS.exists() else []
    seen = {(r["dataset"], r["seed"]): set() for r in rows}
    for r in rows:
        seen[(r["dataset"], r["seed"])].add(r["arm"])
    done = {k for k, v in seen.items() if {"nocap", "cap"} <= v}
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
            for arm, cap in (("nocap", False), ("cap", True)):
                ad = a.copy()
                scf.pp.highly_variable_genes(
                    ad,
                    n_top_genes=K,
                    balance_method="hybrid",
                    cap_allocation=cap,
                    random_state=seed,
                    diagnose=False,
                )
                genes = list(ad.var_names[ad.var["highly_variable"]])
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
                for arm in ("nocap", "cap")
            }
            print(
                f"  seed={seed}  ARI nocap={got['nocap']:.3f} cap={got['cap']:.3f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        del a
    print(f"\nwrote {CSV} ({len(rows)} rows) and {POPS} ({len(pop_rows)} rows)", flush=True)


def run_flavor_sweep():
    rows = pd.read_csv(FLAVOR_CSV).to_dict("records") if FLAVOR_CSV.exists() else []
    seen_arms: dict[tuple, set] = {}
    for r in rows:
        seen_arms.setdefault((r["dataset"], r["flavor"], r["seed"]), set()).add(r["arm"])
    done = {k for k, v in seen_arms.items() if {"nocap", "cap"} <= v}

    for name in FLAVOR_SUBSET:
        a = LOADERS[name]()
        for flavor in FLAVORS:
            for seed in [0, 1]:
                if (name, flavor, seed) in done:
                    continue
                t0 = time.time()
                for arm, cap in (("nocap", False), ("cap", True)):
                    ad = a.copy()
                    try:
                        scf.pp.highly_variable_genes(
                            ad,
                            n_top_genes=K,
                            balance_method="hybrid",
                            flavor=flavor,
                            cap_allocation=cap,
                            random_state=seed,
                            diagnose=False,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"  FAIL {name}/{flavor}/seed={seed}/{arm}: {type(e).__name__}: {e}",
                            flush=True,
                        )
                        continue
                    genes = list(ad.var_names[ad.var["highly_variable"]])
                    base = {
                        "dataset": name,
                        "flavor": flavor,
                        "arm": arm,
                        "seed": seed,
                        "n_genes": len(genes),
                    }
                    evaluate(a, genes, seed, rows, [], base)
                pd.DataFrame(rows).to_csv(FLAVOR_CSV, index=False)
                print(f"  {name}/{flavor}/seed={seed}  ({time.time() - t0:.0f}s)", flush=True)
        del a
    print(f"\nwrote {FLAVOR_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "panel"
    if which == "panel":
        run_panel()
    elif which == "flavors":
        run_flavor_sweep()
    else:
        raise SystemExit(f"unknown mode {which!r}: use 'panel' or 'flavors'")
