#!/usr/bin/env python
"""Real-ARI check on the 9 non-PBMC datasets added 2026-08-01 (DATA_SOURCES.md
"Non-PBMC gold-standard expansion"). Same protocol as
`structure_vs_scanpy_probe.py` (scanpy_2000 / scfair_2000 / scfair_structure,
downstream Leiden ARI/NMI/macro_f1/min_pop_f1), just pointed at the new files
and a lighter seed/resolution grid since several of these are large
(marrow ~4.9k cells, brain ~10k, haber ~7.2k) and untested end-to-end.

Outputs (examples/results/):
  gold_expansion_probe.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "gold_expansion_probe.csv"

SEEDS = [0, 1, 2]
RES_GRID = [0.3, 0.5, 0.8, 1.2]
ARMS = ["scanpy_2000", "scfair_2000", "scfair_structure"]

DATASETS = {
    "villani_dc_mono_gold": ("villani_dc_mono_gold.h5ad", "gold_facs_sort"),
    "haber_intestine_atlas": ("haber_intestine_atlas.h5ad", "mixed_broad_gold_fine_shadowed"),
    "tm_limb_muscle_gold": ("tm_limb_muscle_gold.h5ad", "gold_facs_sort"),
    "tm_brain_myeloid_vs_nonmyeloid_gold": (
        "tm_brain_myeloid_vs_nonmyeloid_gold.h5ad",
        "gold_facs_sort",
    ),
    "tm_lung_shadowed": ("tm_lung_shadowed.h5ad", "SHADOWED"),
    "tm_kidney_shadowed": ("tm_kidney_shadowed.h5ad", "SHADOWED"),
    "tm_marrow_shadowed": ("tm_marrow_shadowed.h5ad", "SHADOWED"),
    "tm_spleen_shadowed": ("tm_spleen_shadowed.h5ad", "SHADOWED"),
    "tm_thymus_shadowed": ("tm_thymus_shadowed.h5ad", "SHADOWED"),
}
# smallest / fastest first so partial runs still cover breadth
ORDER = [
    "tm_limb_muscle_gold",
    "tm_kidney_shadowed",
    "villani_dc_mono_gold",
    "tm_thymus_shadowed",
    "tm_lung_shadowed",
    "tm_spleen_shadowed",
    "haber_intestine_atlas",
    "tm_marrow_shadowed",
    "tm_brain_myeloid_vs_nonmyeloid_gold",
]


def load(name: str) -> ad.AnnData:
    fname, _tier = DATASETS[name]
    a = ad.read_h5ad(DATA / fname)
    a.obs_names_make_unique()
    a.var_names_make_unique()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a = a[a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")].copy()
    return a


def select(adata, arm, seed):
    a = adata.copy()
    k = min(2000, a.n_vars - 1)
    meta = {"pred_k": np.nan}
    if arm == "scanpy_2000":
        sc.pp.highly_variable_genes(a, n_top_genes=k, flavor="seurat_v3", layer="counts")
    elif arm == "scfair_2000":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=k,
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            random_state=seed,
            diagnose=False,
        )
    else:  # scfair_structure
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="structure",
            flavor="seurat_v3",
            layer="counts",
            marker_mode="none",
            balance_method="hybrid",
            random_state=seed,
            diagnose=False,
        )
        meta["pred_k"] = a.uns["scfair"]["hvg"]["auto_n"]["n_top_selected"]
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    return genes, meta


def evaluate(a, genes, seed, rows, base):
    e = a.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    e = e[:, [g for g in genes if g in e.var_names]].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
    y_true = e.obs["cell_type"].astype(str)
    prev = y_true.value_counts(normalize=True)

    for res in RES_GRID:
        sc.tl.leiden(
            e, resolution=res, key_added="L", flavor="igraph", n_iterations=2, random_state=seed
        )
        y_pred = e.obs["L"].astype(str)
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
        rows.append(
            {
                **base,
                "resolution": res,
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
            }
        )


def main(which=None):
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    done = {(r["dataset"], r["seed"], r["arm"]) for r in rows}
    print(f"resuming: {len(done)} blocks done", flush=True)

    for name in which or ORDER:
        if all((name, s, arm) in done for s in SEEDS for arm in ARMS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ({DATASETS[name][1]}) ########", flush=True)
        a = load(name)
        n_types = a.obs["cell_type"].nunique()
        print(f"  shape={a.shape}  n_types={n_types}", flush=True)
        for seed in SEEDS:
            for arm in ARMS:
                if (name, seed, arm) in done:
                    continue
                t0 = time.time()
                genes, meta = select(a, arm, seed)
                base = {"dataset": name, "arm": arm, "seed": seed, "n_genes": len(genes), **meta}
                evaluate(a, genes, seed, rows, base)
                pd.DataFrame(rows).to_csv(CSV, index=False)
                print(
                    f"  {arm:16s} seed={seed} n_genes={len(genes):4d} "
                    f"pred_k={meta['pred_k']}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        del a
    print(f"\nwrote {CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
