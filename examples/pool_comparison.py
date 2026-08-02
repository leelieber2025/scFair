#!/usr/bin/env python
"""Post-0.2.0 head-to-head across the full local data pool.

Same protocol as final_comparison.py (which predates the 0.2.0 API/default
changes: scale_clustering default, resolution floor, cap_allocation removal,
auto reverted to opt-in), re-run against every locally cached dataset with
raw counts + a usable cell-type label — no network downloads, no dependency
on other example scripts' loader side effects.

Arms
----
hvg2000       scanpy ``highly_variable_genes(flavor="seurat_v3")`` @ 2000 — the
              baseline everything is measured against.
scfair2000    scFair product path at fixed base k=2000,
              ``balance_method="append"`` (freeze global top-k + secondary
              ``append_budget`` genes, default 200 → final ~2200). No intermediate
              clustering / no hybrid re-rank.
scfair_auto   same product path with ``n_top_genes="auto"`` (structure v7 +
              soft k-buffer 500→1000 / 1000→1500 / 1500→2000) then append.
              Base k and final gene count recorded in ``k_used`` / ``n_genes``.

Usage
-----
  python examples/pool_comparison.py                      # every registered dataset
  python examples/pool_comparison.py --out small NAME...   # only NAME... datasets
  python examples/pool_comparison.py --list                # print dataset names + sizes

Each invocation writes examples/results/pool_comparison_<out>.csv incrementally
(one row appended per arm/seed, flushed to disk after every dataset) so a killed
process still leaves partial results.
"""

from __future__ import annotations

import argparse
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
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT.parent / "src"))

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ARMS = ["hvg2000", "scfair2000", "scfair_auto"]
SEEDS_DEFAULT = [0, 1, 2]
SEEDS_BIG = [0, 1]
# Fallback only; evaluate() uses scf.pp.resolve_cluster_resolution(auto).
LEIDEN_RES = 0.8


def _generic(fname, label_col, drop=(), subsample=None):
    def _load():
        a = ad.read_h5ad(DATA / fname)
        a = a.copy()
        a.obs_names_make_unique()
        a.var_names_make_unique()
        if "counts" not in a.layers:
            a.layers["counts"] = a.X.copy()
        sc.pp.filter_genes(a, min_cells=3)
        a.obs["cell_type"] = a.obs[label_col].astype(str)
        mask = a.obs["cell_type"].notna() & (a.obs["cell_type"] != "nan")
        for d in drop:
            mask &= a.obs["cell_type"] != d
        a = a[mask].copy()
        if subsample and a.n_obs > subsample:
            rng = np.random.default_rng(0)
            idx = rng.choice(a.n_obs, size=subsample, replace=False)
            a = a[idx].copy()
        return a

    return _load


LOADERS = {
    "duo4_pbmc": _generic("duo4_pbmc.h5ad", "cell_type"),
    "duo8_pbmc": _generic("duo8_pbmc.h5ad", "cell_type"),
    "duo4un_pbmc": _generic("duo4un_pbmc.h5ad", "cell_type"),
    "tm_kidney_shadowed": _generic("tm_kidney_shadowed.h5ad", "cell_type"),
    "tm_limb_muscle_gold": _generic("tm_limb_muscle_gold.h5ad", "cell_type"),
    "tm_thymus_shadowed": _generic("tm_thymus_shadowed.h5ad", "cell_type"),
    "tm_spleen_shadowed": _generic("tm_spleen_shadowed.h5ad", "cell_type"),
    "tm_lung_shadowed": _generic("tm_lung_shadowed.h5ad", "cell_type"),
    "tm_marrow_shadowed": _generic("tm_marrow_shadowed.h5ad", "cell_type"),
    "tm_brain_myeloid_vs_nonmyeloid_gold": _generic(
        "tm_brain_myeloid_vs_nonmyeloid_gold.h5ad", "cell_type"
    ),
    "crafted_base_3cellline": _generic("crafted_base_3cellline_GSE136148.h5ad", "cell_type"),
    "pbmc_10k_adt_labeled": _generic(
        "pbmc_10k_adt_labeled.h5ad", "cell_type", drop=("unassigned",)
    ),
    "baron_pancreas": _generic("baron_pancreas_human_author.h5ad", "cell_type"),
    "villani_dc_mono": _generic("villani_dc_mono_gold.h5ad", "cell_type"),
    "pbmc_cite_gse100866": _generic("pbmc_cite_gse100866_holdout.h5ad", "cell_type"),
    "gbm_sd": _generic("gbm_sd_gse84465_holdout.h5ad", "cell_type", drop=("Unpanned",)),
    "cbmc8k_cite": _generic("cbmc8k_cite_holdout.h5ad", "cell_type"),
    "haber_intestine": _generic("haber_intestine_atlas.h5ad", "cell_type"),
    "pbmc_cellxgene_gold": _generic(
        "pbmc_10x_v3_cellxgene_gold.h5ad",
        "author_cell_type",
        drop=("unknown", "multiplet"),
    ),
    "pbmc_10k_v3_labeled": _generic("pbmc_10k_v3_labeled.h5ad", "cell_type"),
    # big — own process recommended
    "sln_208_mouse": _generic("sln_208.h5ad", "cell_types"),
    "human_pancreas_complexBatch": _generic("human_pancreas_norm_complexBatch.h5ad", "celltype"),
    "lung_atlas": _generic("Lung_atlas_public.h5ad", "cell_type"),
    "pbmc_seurat_v4_20k": _generic("pbmc_seurat_v4.h5ad", "celltype.l2", subsample=20000),
    "zheng_facs9_gold": _generic("zheng_facs9_gold.h5ad", "cell_type"),
}

BIG = {
    "sln_208_mouse",
    "human_pancreas_complexBatch",
    "lung_atlas",
    "pbmc_seurat_v4_20k",
    "zheng_facs9_gold",
}


def select(adata, arm: str, seed: int):
    """Product path: append + mode='auto' (fine/compact/balanced from data)."""
    from scfair.pp import HVGOptions

    a = adata.copy()
    lab = "cell_type" if "cell_type" in a.obs.columns else None
    opt = HVGOptions(label_key=lab) if lab else None
    if arm == "hvg2000":
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
        )
    elif arm == "scfair2000":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=min(2000, a.n_vars - 1),
            flavor="seurat_v3",
            layer="counts",
            balance_method="append",
            mode="auto",
            options=opt,
            random_state=seed,
            diagnose=False,
            progress=False,
        )
    elif arm == "scfair_auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            flavor="seurat_v3",
            layer="counts",
            balance_method="append",
            mode="auto",
            options=opt,
            random_state=seed,
            diagnose=False,
            progress=False,
        )
    else:
        raise ValueError(arm)
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    h = a.uns.get("scfair", {}).get("hvg", {}) or {}
    append_info = h.get("append") or {}
    # Prefer frozen base k for append; fall back to n_top_genes_used / mask size.
    k_used = (
        append_info.get("n_base")
        if append_info.get("n_base") is not None
        else h.get("n_top_genes_used")
    )
    auto_n = h.get("auto_n") or {}
    st = auto_n.get("structure") if isinstance(auto_n, dict) else None
    if not isinstance(st, dict):
        st = {}
    dc = h.get("downstream_clustering") or {}
    meta_extra = {
        "balance_method": h.get("balance_method"),
        "hvg_mode": h.get("mode"),
        "append_budget": h.get("append_budget"),
        "n_base": append_info.get("n_base"),
        "n_append_used": append_info.get("n_append_used"),
        "rule_branch": (
            auto_n.get("rule_branch")
            or st.get("rule_branch")
            or (st.get("rule_explain") or {}).get("rule_branch")
        ),
        "k_source": auto_n.get("k_source") or st.get("k_source"),
        "k_buffer_raw": st.get("k_buffer_raw")
        or (st.get("rule_explain") or {}).get("k_buffer_raw")
        or auto_n.get("k_buffer_raw"),
        "mode_cluster_res": dc.get("resolution"),
    }
    return genes, (k_used if k_used is not None else len(genes)), meta_extra


def evaluate(adata, genes, seed: int, *, label_key: str = "cell_type"):
    """Downstream PCA→neighbours→Leiden with **auto** fine/coarse resolution.

    Fine (res=1.5) when n_types≥15 (or structure fine-atlas); else 0.8.
    """
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, [g for g in genes if g in a.var_names]].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    # Prefer type count on the evaluation object; structure meta if HVG wrote uns.
    rec = scf.pp.resolve_cluster_resolution(
        a if "scfair" in getattr(a, "uns", {}) else adata,
        resolution="auto",
        label_key=label_key if label_key in a.obs.columns else None,
        n_types=int(a.obs[label_key].nunique()) if label_key in a.obs.columns else None,
    )
    leiden_res = float(rec["resolution"])
    sc.tl.leiden(
        a,
        resolution=leiden_res,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    y_true = a.obs[label_key].astype(str)
    y_pred = a.obs["leiden"].astype(str)

    f1s, prev = {}, y_true.value_counts(normalize=True)
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
    rare = [p for p in f1s if prev.get(p, 0) < 0.02]
    return {
        "n_genes": a.n_vars,
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "rare_f1": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
        "n_pop": len(f1s),
        "leiden_resolution": leiden_res,
        "leiden_tier": rec.get("tier"),
        "n_leiden": int(y_pred.nunique()),
    }


def main(names, out_tag: str):
    csv_path = OUT / f"pool_comparison_{out_tag}.csv"
    rows = []
    for dname in names:
        t_ds = time.time()
        print(f"\n################ {dname} ################", flush=True)
        try:
            adata = LOADERS[dname]()
        except Exception as e:
            print(f"  LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        seeds = SEEDS_BIG if dname in BIG else SEEDS_DEFAULT
        print(
            f"  {adata.n_obs} x {adata.n_vars}  types={adata.obs['cell_type'].nunique()}  seeds={seeds}",
            flush=True,
        )
        for arm in ARMS:
            for seed in seeds:
                t0 = time.time()
                try:
                    genes, k_used, meta_extra = select(adata, arm, seed)
                    res = evaluate(adata, genes, seed)
                    res.update(
                        {
                            "dataset": dname,
                            "arm": arm,
                            "seed": seed,
                            "k_used": k_used,
                            "n_obs": int(adata.n_obs),
                            **{k: v for k, v in meta_extra.items() if v is not None},
                        }
                    )
                    rows.append(res)
                    base_note = (
                        f" base={meta_extra.get('n_base')}"
                        if meta_extra.get("n_base") is not None
                        else ""
                    )
                    print(
                        f"  {arm:12s} seed={seed} k={res['n_genes']:5d}{base_note} "
                        f"res={res.get('leiden_resolution', LEIDEN_RES)}"
                        f"({res.get('leiden_tier', '?')}) "
                        f"ARI={res['ARI']:.3f} macroF1={res['macro_f1']:.3f} "
                        f"rareF1={res['rare_f1']:.3f} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                    rows.append({"dataset": dname, "arm": arm, "seed": seed, "error": str(e)})
                pd.DataFrame(rows).to_csv(csv_path, index=False)
        del adata
        print(f"  [{dname} done in {time.time() - t_ds:.0f}s]", flush=True)
    print(f"\nDONE -> {csv_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help="dataset names (default: all)")
    p.add_argument("--out", default="all", help="output CSV suffix")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for n in LOADERS:
            tag = "BIG" if n in BIG else "small"
            print(f"{n:36s} {tag}")
        sys.exit(0)

    main(args.names or list(LOADERS), args.out)
