#!/usr/bin/env python
"""Targeted probes (no docs): (1) seurat_v4 res × label; (2) v3/pancreas short-k.

1) pbmc_seurat_v4 (20k subsample):
   arms: hvg2000, append2000
   labels: celltype.l1, celltype.l2
   Leiden res: 0.8, 1.2, 1.5, 2.0
   seeds: 0, 1
   Gene selection once per (arm, seed); reuse for all label×res.

2) pbmc_10k_v3_labeled + human_pancreas_complexBatch:
   arms: hvg2000, append2000, append_auto
   Leiden res: 0.8 (same as pool_comparison)
   seeds: 0, 1, 2
   Check whether structure soft-buffer auto (often base≈1000) beats fixed 2200.

Usage:
  python examples/fine_res_and_shortk_probe.py              # both probes
  python examples/fine_res_and_shortk_probe.py seurat
  python examples/fine_res_and_shortk_probe.py shortk
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
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT.parent / "src"))

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

SEEDS_SEURAT = [0, 1]
SEEDS_SHORTK = [0, 1, 2]
RES_SWEEP = [0.8, 1.2, 1.5, 2.0]
N_SUB = 20_000


def _ensure_counts(a: ad.AnnData) -> ad.AnnData:
    a = a.copy()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    return a


def select_genes(adata: ad.AnnData, arm: str, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    meta: dict = {}
    if arm == "hvg2000":
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
        )
    elif arm == "append2000":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=min(2000, a.n_vars - 1),
            flavor="seurat_v3",
            layer="counts",
            balance_method="append",
            random_state=seed,
            diagnose=False,
            progress=False,
        )
    elif arm == "append_auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            flavor="seurat_v3",
            layer="counts",
            balance_method="append",
            random_state=seed,
            diagnose=False,
            progress=False,
        )
    else:
        raise ValueError(arm)
    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    h = a.uns.get("scfair", {}).get("hvg", {}) or {}
    app = h.get("append") or {}
    auto = h.get("auto_n") or {}
    st = auto.get("structure") if isinstance(auto, dict) else {}
    if not isinstance(st, dict):
        st = {}
    meta = {
        "n_genes": len(genes),
        "k_used": app.get("n_base", h.get("n_top_genes_used", len(genes))),
        "n_base": app.get("n_base"),
        "n_append_used": app.get("n_append_used"),
        "balance_method": h.get("balance_method"),
        "rule_branch": auto.get("rule_branch") or st.get("rule_branch"),
        "k_buffer_raw": st.get("k_buffer_raw")
        or (st.get("rule_explain") or {}).get("k_buffer_raw"),
    }
    return genes, meta


def evaluate(
    adata: ad.AnnData,
    genes: list[str],
    *,
    label_col: str,
    resolution: float,
    seed: int,
) -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    keep = [g for g in genes if g in a.var_names]
    a = a[:, keep].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a,
        resolution=float(resolution),
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )
    y_true = a.obs[label_col].astype(str)
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
                best = max(best, 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
        f1s[pop] = best
    rare = [p for p in f1s if prev.get(p, 0) < 0.02]
    return {
        "n_genes": a.n_vars,
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "rare_f1": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
        "n_pop": int(y_true.nunique()),
    }


def run_seurat() -> Path:
    path = DATA / "pbmc_seurat_v4.h5ad"
    csv_path = OUT / "fine_res_seurat_v4.csv"
    print(f"\n######## seurat_v4 res×label probe → {csv_path}", flush=True)
    t0 = time.time()
    raw = ad.read_h5ad(path)
    raw = _ensure_counts(raw)
    # drop missing labels
    raw = raw[raw.obs["celltype.l1"].notna() & raw.obs["celltype.l2"].notna()].copy()
    rng = np.random.default_rng(0)
    if raw.n_obs > N_SUB:
        idx = rng.choice(raw.n_obs, size=N_SUB, replace=False)
        raw = raw[idx].copy()
    print(
        f"  {raw.n_obs} x {raw.n_vars}  l1={raw.obs['celltype.l1'].nunique()} "
        f"l2={raw.obs['celltype.l2'].nunique()}",
        flush=True,
    )
    rows: list[dict] = []
    for arm in ("hvg2000", "append2000"):
        for seed in SEEDS_SEURAT:
            ts = time.time()
            genes, meta = select_genes(raw, arm, seed)
            print(
                f"  select {arm} seed={seed} n_genes={len(genes)} "
                f"base={meta.get('n_base')} ({time.time() - ts:.0f}s)",
                flush=True,
            )
            for label_col, label_tag in (
                ("celltype.l1", "l1"),
                ("celltype.l2", "l2"),
            ):
                for res in RES_SWEEP:
                    te = time.time()
                    try:
                        m = evaluate(raw, genes, label_col=label_col, resolution=res, seed=seed)
                        row = {
                            "dataset": "pbmc_seurat_v4_20k",
                            "arm": arm,
                            "seed": seed,
                            "label": label_tag,
                            "label_col": label_col,
                            "resolution": res,
                            **m,
                            **{k: v for k, v in meta.items() if v is not None},
                        }
                        rows.append(row)
                        print(
                            f"    {arm:11s} {label_tag} res={res:.1f} seed={seed} "
                            f"ARI={m['ARI']:.3f} macroF1={m['macro_f1']:.3f} "
                            f"n_leiden={m['n_leiden']} ({time.time() - te:.0f}s)",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"    FAIL {arm} {label_tag} res={res} seed={seed}: {e}", flush=True)
                        rows.append(
                            {
                                "dataset": "pbmc_seurat_v4_20k",
                                "arm": arm,
                                "seed": seed,
                                "label": label_tag,
                                "resolution": res,
                                "error": str(e),
                            }
                        )
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  seurat done in {time.time() - t0:.0f}s → {csv_path}", flush=True)
    return csv_path


def run_shortk() -> Path:
    csv_path = OUT / "shortk_v3_pancreas.csv"
    print(f"\n######## short-k probe (v3 + pancreas) → {csv_path}", flush=True)
    specs = [
        (
            "pbmc_10k_v3_labeled",
            DATA / "pbmc_10k_v3_labeled.h5ad",
            "cell_type",
            (),
        ),
        (
            "human_pancreas_complexBatch",
            DATA / "human_pancreas_norm_complexBatch.h5ad",
            "celltype",
            (),
        ),
    ]
    rows: list[dict] = []
    for dname, path, lab, drop in specs:
        print(f"\n## {dname}", flush=True)
        raw = ad.read_h5ad(path)
        raw = _ensure_counts(raw)
        raw.obs["cell_type"] = raw.obs[lab].astype(str)
        mask = raw.obs["cell_type"].notna() & (raw.obs["cell_type"] != "nan")
        for d in drop:
            mask &= raw.obs["cell_type"] != d
        raw = raw[mask].copy()
        print(f"  {raw.n_obs} x {raw.n_vars} types={raw.obs['cell_type'].nunique()}", flush=True)
        for arm in ("hvg2000", "append2000", "append_auto"):
            for seed in SEEDS_SHORTK:
                t0 = time.time()
                try:
                    genes, meta = select_genes(raw, arm, seed)
                    m = evaluate(raw, genes, label_col="cell_type", resolution=0.8, seed=seed)
                    row = {
                        "dataset": dname,
                        "arm": arm,
                        "seed": seed,
                        "label": "author",
                        "resolution": 0.8,
                        **m,
                        **{k: v for k, v in meta.items() if v is not None},
                    }
                    rows.append(row)
                    print(
                        f"  {arm:12s} seed={seed} genes={m['n_genes']} "
                        f"base={meta.get('n_base')} branch={str(meta.get('rule_branch'))[:40]} "
                        f"ARI={m['ARI']:.3f} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  FAIL {arm} seed={seed}: {e}", flush=True)
                    rows.append({"dataset": dname, "arm": arm, "seed": seed, "error": str(e)})
                pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  shortk done → {csv_path}", flush=True)
    return csv_path


def summarize():
    print("\n######## SUMMARY ########")
    sp = OUT / "fine_res_seurat_v4.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        if "error" in df.columns:
            df = df[df.get("error").isna()] if "error" in df else df
        g = df.groupby(["label", "resolution", "arm"], as_index=False).agg(
            ARI=("ARI", "mean"), macro_f1=("macro_f1", "mean"), n_leiden=("n_leiden", "mean")
        )
        print("\n--- seurat mean ARI (seed avg) ---")
        for label in ["l1", "l2"]:
            sub = g[g.label == label]
            print(f"\n  label={label}")
            print(
                f"  {'res':>5}  {'hvg':>7}  {'append':>7}  {'dARI':>8}  {'hvg_mF1':>7}  {'app_mF1':>7}"
            )
            for res in RES_SWEEP:
                h = sub[(sub.resolution == res) & (sub.arm == "hvg2000")]
                a = sub[(sub.resolution == res) & (sub.arm == "append2000")]
                if h.empty or a.empty:
                    continue
                dh = float(a.ARI.iloc[0] - h.ARI.iloc[0])
                print(
                    f"  {res:5.1f}  {h.ARI.iloc[0]:7.3f}  {a.ARI.iloc[0]:7.3f}  {dh:+8.4f}  "
                    f"{h.macro_f1.iloc[0]:7.3f}  {a.macro_f1.iloc[0]:7.3f}"
                )
    sk = OUT / "shortk_v3_pancreas.csv"
    if sk.exists():
        df = pd.read_csv(sk)
        print("\n--- v3 / pancreas short-k (seed mean) ---")
        g = df.groupby(["dataset", "arm"], as_index=False).agg(
            ARI=("ARI", "mean"),
            n_genes=("n_genes", "mean"),
            n_base=("n_base", "mean"),
            macro_f1=("macro_f1", "mean"),
        )
        for d, sub in g.groupby("dataset"):
            print(f"\n  {d}")
            hvg = sub[sub.arm == "hvg2000"]
            href = float(hvg.ARI.iloc[0]) if len(hvg) else np.nan
            for _, r in sub.iterrows():
                dlt = r.ARI - href if np.isfinite(href) else np.nan
                print(
                    f"    {r.arm:12s} ARI={r.ARI:.3f} dARI={dlt:+.4f} "
                    f"genes={r.n_genes:.0f} base={r.n_base if pd.notna(r.n_base) else '-'} "
                    f"mF1={r.macro_f1:.3f}"
                )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "seurat"):
        run_seurat()
    if which in ("all", "shortk"):
        run_shortk()
    summarize()
    print("\nDONE", flush=True)
