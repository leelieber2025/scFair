#!/usr/bin/env python
"""Fine-type gene-side probe on seurat_v4 (20k).

Arms (gene selection):
  hvg2000       scanpy seurat_v3 @ 2000
  append_m200   scFair append, base 2000 + budget 200  (product default)
  append_m500   scFair append, base 2000 + budget 500
  hvg3000       scanpy seurat_v3 @ 3000  (matched-length control vs m500)

Downstream (P1-oriented):
  labels: celltype.l1 (coarse control), celltype.l2 (fine)
  Leiden res: 0.8, 1.5, 2.0
  seeds: 0, 1

Gene selection once per (arm, seed); reused for all label×res.

Output: examples/results/fine_gene_budget_seurat.csv
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
from scfair.pp import HVGOptions  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

SEEDS = [0, 1]
RES_LIST = [0.8, 1.5, 2.0]
N_SUB = 20_000
CSV = OUT / "fine_gene_budget_seurat.csv"


def ensure_counts(a: ad.AnnData) -> ad.AnnData:
    a = a.copy()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    return a


def select(adata: ad.AnnData, arm: str, seed: int) -> tuple[list[str], dict]:
    a = adata.copy()
    meta: dict = {"arm": arm}
    if arm == "hvg2000":
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
        )
        meta.update(n_base=2000, n_append_used=0, balance_method="none_scanpy")
    elif arm == "hvg3000":
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(3000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
        )
        meta.update(n_base=3000, n_append_used=0, balance_method="none_scanpy")
    elif arm == "append_m200":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=min(2000, a.n_vars - 1),
            flavor="seurat_v3",
            layer="counts",
            balance_method="append",
            options=HVGOptions(append_budget=200),
            random_state=seed,
            diagnose=False,
            progress=False,
        )
    elif arm == "append_m500":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=min(2000, a.n_vars - 1),
            flavor="seurat_v3",
            layer="counts",
            balance_method="append",
            options=HVGOptions(append_budget=500),
            random_state=seed,
            diagnose=False,
            progress=False,
        )
    else:
        raise ValueError(arm)

    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    h = a.uns.get("scfair", {}).get("hvg", {}) or {}
    app = h.get("append") or {}
    if arm.startswith("append"):
        meta.update(
            n_base=app.get("n_base"),
            n_append_used=app.get("n_append_used"),
            balance_method=h.get("balance_method"),
            append_budget=h.get("append_budget"),
        )
    meta["n_genes"] = len(genes)
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
    a = a[:, [g for g in genes if g in a.var_names]].copy()
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
        "n_genes_eval": int(a.n_vars),
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "rare_f1": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
        "n_pop": int(y_true.nunique()),
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main() -> None:
    print(f"→ {CSV}", flush=True)
    raw = ad.read_h5ad(DATA / "pbmc_seurat_v4.h5ad")
    raw = ensure_counts(raw)
    raw = raw[raw.obs["celltype.l1"].notna() & raw.obs["celltype.l2"].notna()].copy()
    rng = np.random.default_rng(0)
    if raw.n_obs > N_SUB:
        idx = rng.choice(raw.n_obs, size=N_SUB, replace=False)
        raw = raw[idx].copy()
    print(
        f"  {raw.n_obs}×{raw.n_vars}  l1={raw.obs['celltype.l1'].nunique()} "
        f"l2={raw.obs['celltype.l2'].nunique()}",
        flush=True,
    )

    arms = ["hvg2000", "append_m200", "append_m500", "hvg3000"]
    rows: list[dict] = []
    gene_sets: dict[tuple[str, int], set[str]] = {}

    for arm in arms:
        for seed in SEEDS:
            t0 = time.time()
            genes, meta = select(raw, arm, seed)
            gene_sets[(arm, seed)] = set(genes)
            print(
                f"  select {arm:12s} seed={seed} n={len(genes)} "
                f"base={meta.get('n_base')} +{meta.get('n_append_used')} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            for label_col, tag in (("celltype.l1", "l1"), ("celltype.l2", "l2")):
                for res in RES_LIST:
                    te = time.time()
                    m = evaluate(raw, genes, label_col=label_col, resolution=res, seed=seed)
                    row = {
                        "dataset": "pbmc_seurat_v4_20k",
                        "arm": arm,
                        "seed": seed,
                        "label": tag,
                        "resolution": res,
                        **m,
                        **{k: v for k, v in meta.items() if v is not None},
                    }
                    rows.append(row)
                    print(
                        f"    {arm:12s} {tag} res={res:.1f} s={seed} "
                        f"ARI={m['ARI']:.3f} mF1={m['macro_f1']:.3f} "
                        f"rareF1={m['rare_f1']:.3f} nL={m['n_leiden']} "
                        f"({time.time() - te:.0f}s)",
                        flush=True,
                    )
                    pd.DataFrame(rows).to_csv(CSV, index=False)

    # gene-set overlap (seed 0)
    print("\n  gene-set Jaccard (seed=0):", flush=True)
    s0 = {a: gene_sets[(a, 0)] for a in arms}
    for a in arms:
        for b in arms:
            if a < b:
                print(f"    {a} ∩ {b} = {jaccard(s0[a], s0[b]):.3f}", flush=True)

    # summary tables
    df = pd.DataFrame(rows)
    print("\n######## SUMMARY (mean over seeds) ########", flush=True)
    g = df.groupby(["label", "resolution", "arm"], as_index=False).agg(
        ARI=("ARI", "mean"),
        macro_f1=("macro_f1", "mean"),
        rare_f1=("rare_f1", "mean"),
        n_genes=("n_genes", "mean"),
    )
    for label in ("l1", "l2"):
        print(f"\n=== label={label} ===", flush=True)
        print(
            f"{'res':>5} {'arm':12s} {'ARI':>7} {'dARI_hvg':>9} "
            f"{'mF1':>7} {'dmF1':>8} {'rareF1':>7} {'genes':>6}",
            flush=True,
        )
        for res in RES_LIST:
            sub = g[(g.label == label) & (g.resolution == res)]
            href = sub[sub.arm == "hvg2000"]
            h_ari = float(href.ARI.iloc[0]) if len(href) else np.nan
            h_mf = float(href.macro_f1.iloc[0]) if len(href) else np.nan
            for arm in arms:
                r = sub[sub.arm == arm]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(
                    f"{res:5.1f} {arm:12s} {r.ARI:7.3f} {r.ARI - h_ari:+9.4f} "
                    f"{r.macro_f1:7.3f} {r.macro_f1 - h_mf:+8.4f} "
                    f"{r.rare_f1:7.3f} {r.n_genes:6.0f}",
                    flush=True,
                )
    print(f"\nDONE → {CSV}", flush=True)


if __name__ == "__main__":
    main()
