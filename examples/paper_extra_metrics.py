#!/usr/bin/env python
"""P0/P1 quantitative metrics for the paper (UMAP left to the user).

Extends the pool_comparison protocol with:

P0
  - ASW (average silhouette width on PCA, true labels)
  - k-NN classification accuracy (15-NN, true labels)
  - Fair-k arm ``hvg_match_auto``: scanpy seurat_v3 with n_top = |scfair_auto genes|

P1
  - Variance ratio (between / within cell-type on PCA)
  - CITE: distance correlation RNA↔protein + ARI(Leiden_RNA vs Leiden_protein)
    when ``obsm`` carries protein/ADT

Usage
-----
  python examples/paper_extra_metrics.py --list
  python examples/paper_extra_metrics.py --out gold_p0p1          # all GOLD tier
  python examples/paper_extra_metrics.py --out gold_p0p1 duo4_pbmc gbm_sd
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

# Reuse loaders / HVG select / BIG flags from pool_comparison
from pool_comparison import (  # noqa: E402
    BIG,
    LOADERS,
    SEEDS_BIG,
    SEEDS_DEFAULT,
    select,
)

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# GOLD tier from docs/PAPER_pool25_by_label_tier.csv (keep in sync)
GOLD_DATASETS = [
    "pbmc_10k_adt_labeled",
    "duo4_pbmc",
    "tm_limb_muscle_gold",
    "pbmc_cellxgene_gold",
    "pbmc_seurat_v4_20k",
    "duo4un_pbmc",
    "zheng_facs9_gold",
    "gbm_sd",
    "crafted_base_3cellline",
    "sln_208_mouse",
    "villani_dc_mono",
    "cbmc8k_cite",
    "duo8_pbmc",
    "tm_brain_myeloid_vs_nonmyeloid_gold",
    "pbmc_cite_gse100866",
]

ARMS = ["hvg2000", "scfair_auto", "hvg_match_auto"]


def _protein_matrix(adata) -> np.ndarray | None:
    """Return dense protein/ADT matrix (cells × proteins) or None."""
    for key in (
        "protein_expression",
        "protein",
        "protein_counts",
        "X_protein",
        "ADT",
        "adt",
    ):
        if key in adata.obsm:
            X = adata.obsm[key]
            if sparse.issparse(X):
                X = X.toarray()
            return np.asarray(X, dtype=float)
    return None


def _log1p_normalize_counts(adata):
    a = adata.copy()
    if "counts" in a.layers:
        a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def embed_on_genes(adata, genes: list[str], seed: int, *, label_key: str = "cell_type"):
    """Normalize → subset HVGs → scale → PCA → neighbours → Leiden (auto res)."""
    a = _log1p_normalize_counts(adata)
    keep = [g for g in genes if g in a.var_names]
    a = a[:, keep].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    n_pcs = min(30, n_comps)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=n_pcs, random_state=seed)
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
    return a, n_pcs, leiden_res, rec.get("tier")


def _variance_ratio(X: np.ndarray, labels: np.ndarray) -> float:
    """Between-cluster / within-cluster variance (trace form on features)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    overall = X.mean(axis=0)
    total = float(np.sum((X - overall) ** 2))
    if total <= 0:
        return float("nan")
    within = 0.0
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() < 2:
            continue
        mu = X[mask].mean(axis=0)
        within += float(np.sum((X[mask] - mu) ** 2))
    between = max(total - within, 0.0)
    if within <= 0:
        return float("inf") if between > 0 else float("nan")
    return between / within


def metrics_label_space(
    a,
    *,
    label_key: str = "cell_type",
    n_pcs: int = 30,
) -> dict:
    """P0/P1 metrics that need true labels + PCA embedding."""
    y = a.obs[label_key].astype(str).to_numpy()
    y_pred = a.obs["leiden"].astype(str).to_numpy()
    X = np.asarray(a.obsm["X_pca"][:, :n_pcs], dtype=float)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Silhouette needs ≥2 labels and each cluster size constraints — sklearn handles
    try:
        asw = float(silhouette_score(X, y_enc, metric="euclidean"))
    except Exception:
        asw = float("nan")

    # k-NN accuracy (leave-one-out style via sklearn; for large n use 5-fold)
    knn_acc = float("nan")
    try:
        clf = KNeighborsClassifier(n_neighbors=min(15, max(2, len(np.unique(y_enc)))))
        if a.n_obs <= 8000:
            from sklearn.model_selection import cross_val_score

            knn_acc = float(
                cross_val_score(clf, X, y_enc, cv=min(5, len(np.unique(y_enc))), n_jobs=1).mean()
            )
        else:
            # stratified holdout for large objects
            from sklearn.model_selection import train_test_split

            Xtr, Xte, ytr, yte = train_test_split(
                X, y_enc, test_size=0.25, random_state=0, stratify=y_enc
            )
            clf.fit(Xtr, ytr)
            knn_acc = float(clf.score(Xte, yte))
    except Exception:
        knn_acc = float("nan")

    var_ratio = _variance_ratio(X, y)

    # cluster-vs-label F1 (same as pool_comparison)
    f1s, prev = {}, pd.Series(y).value_counts(normalize=True)
    for pop in np.unique(y):
        t = y == pop
        best = 0.0
        for cl in np.unique(y_pred):
            p = y_pred == cl
            tp = float(np.sum(t & p))
            if tp:
                pr, rc = tp / p.sum(), tp / t.sum()
                best = max(best, 2 * pr * rc / (pr + rc))
        f1s[pop] = best
    rare = [p for p in f1s if prev.get(p, 0) < 0.02]

    return {
        "n_genes": int(a.n_vars),
        "ARI": float(adjusted_rand_score(y, y_pred)),
        "NMI": float(normalized_mutual_info_score(y, y_pred)),
        "ASW": asw,
        "knn_acc": knn_acc,
        "var_ratio": float(var_ratio) if np.isfinite(var_ratio) else np.nan,
        "macro_f1": float(np.mean(list(f1s.values()))),
        "rare_f1": float(np.mean([f1s[p] for p in rare])) if rare else np.nan,
        "n_pop": len(f1s),
        "n_leiden": int(len(np.unique(y_pred))),
    }


def metrics_cite(a, protein: np.ndarray, *, n_pcs: int = 30, seed: int = 0) -> dict:
    """P1 CITE: RNA embedding vs protein (subsample distances if needed)."""
    out = {
        "cite_dist_corr": np.nan,
        "cite_ari_rna_vs_adt": np.nan,
        "cite_n_protein": int(protein.shape[1]),
    }
    try:
        from scipy.spatial.distance import pdist
        from scipy.stats import pearsonr

        Xr = np.asarray(a.obsm["X_pca"][:, :n_pcs], dtype=float)
        Xp = np.asarray(protein, dtype=float)
        # align rows (same obs order)
        if Xp.shape[0] != a.n_obs:
            return out
        # protein log1p library-normalize lightly
        lib = Xp.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        Xp = np.log1p(Xp / lib * 1e4)

        n = a.n_obs
        rng = np.random.default_rng(seed)
        if n > 3000:
            idx = rng.choice(n, size=3000, replace=False)
            Xr_s, Xp_s = Xr[idx], Xp[idx]
        else:
            Xr_s, Xp_s = Xr, Xp
        dr = pdist(Xr_s, metric="euclidean")
        dp = pdist(Xp_s, metric="euclidean")
        if dr.std() > 0 and dp.std() > 0:
            out["cite_dist_corr"] = float(pearsonr(dr, dp)[0])

        # Leiden on protein (PCA of protein → neighbours → leiden)
        import anndata as ad

        ap = ad.AnnData(X=Xp)
        ap.obs_names = a.obs_names.to_numpy()
        sc.pp.scale(ap, max_value=10)
        n_comp = min(20, ap.n_vars - 1, ap.n_obs - 1)
        if n_comp >= 2:
            sc.tl.pca(ap, n_comps=n_comp, random_state=seed)
            sc.pp.neighbors(ap, n_neighbors=15, n_pcs=min(15, n_comp), random_state=seed)
            sc.tl.leiden(
                ap,
                resolution=0.8,
                key_added="leiden_adt",
                flavor="igraph",
                n_iterations=2,
                random_state=seed,
            )
            out["cite_ari_rna_vs_adt"] = float(
                adjusted_rand_score(a.obs["leiden"].astype(str), ap.obs["leiden_adt"].astype(str))
            )
    except Exception as e:
        out["cite_error"] = f"{type(e).__name__}: {e}"
    return out


def select_hvg_match(adata, n_top: int, seed: int) -> list[str]:
    a = adata.copy()
    sc.pp.highly_variable_genes(
        a,
        n_top_genes=min(int(n_top), a.n_vars - 1),
        flavor="seurat_v3",
        layer="counts",
    )
    return a.var_names[a.var["highly_variable"]].astype(str).tolist()


def run_one(dname: str, arm: str, seed: int, adata, protein, genes=None, meta_extra=None):
    t0 = time.time()
    meta_extra = meta_extra or {}
    if genes is None:
        if arm == "hvg_match_auto":
            raise ValueError("hvg_match_auto needs genes precomputed")
        genes, k_used, meta_extra = select(
            adata, arm if arm != "hvg_match_auto" else "hvg2000", seed
        )
    else:
        k_used = len(genes)

    a_emb, n_pcs, leiden_res, leiden_tier = embed_on_genes(adata, genes, seed)
    row = {
        "dataset": dname,
        "arm": arm,
        "seed": seed,
        "k_used": k_used,
        "n_obs": int(adata.n_obs),
        "leiden_resolution": leiden_res,
        "leiden_tier": leiden_tier,
        **{k: v for k, v in meta_extra.items() if v is not None},
    }
    row.update(metrics_label_space(a_emb, n_pcs=n_pcs))
    if protein is not None:
        # protein rows must match original adata obs; emb is same cells
        # reindex protein if needed
        row.update(metrics_cite(a_emb, protein, n_pcs=n_pcs, seed=seed))
    else:
        row["cite_dist_corr"] = np.nan
        row["cite_ari_rna_vs_adt"] = np.nan
        row["cite_n_protein"] = 0
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main(names: list[str], out_tag: str):
    csv_path = OUT / f"paper_extra_metrics_{out_tag}.csv"
    rows: list[dict] = []
    for dname in names:
        t_ds = time.time()
        print(f"\n################ {dname} ################", flush=True)
        try:
            adata = LOADERS[dname]()
        except Exception as e:
            print(f"  LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        protein = _protein_matrix(adata)
        # Align protein to filtered adata (loader may have subset cells)
        if protein is not None and protein.shape[0] != adata.n_obs:
            print(
                f"  protein shape {protein.shape[0]} != n_obs {adata.n_obs}; "
                "skipping CITE metrics for this dataset",
                flush=True,
            )
            protein = None
        seeds = SEEDS_BIG if dname in BIG else SEEDS_DEFAULT
        print(
            f"  {adata.n_obs}x{adata.n_vars} types={adata.obs['cell_type'].nunique()} "
            f"seeds={seeds} protein={'yes' if protein is not None else 'no'}",
            flush=True,
        )
        for seed in seeds:
            # --- product arms ---
            for arm in ("hvg2000", "scfair_auto"):
                try:
                    genes, k_used, meta_extra = select(adata, arm, seed)
                    row = run_one(
                        dname,
                        arm,
                        seed,
                        adata,
                        protein,
                        genes=genes,
                        meta_extra={**meta_extra, "k_used_select": k_used},
                    )
                    rows.append(row)
                    print(
                        f"  {arm:14s} seed={seed} n={row['n_genes']:5d} "
                        f"ARI={row['ARI']:.3f} ASW={row['ASW']:.3f} "
                        f"kNN={row['knn_acc']:.3f} varR={row['var_ratio']:.3f} "
                        f"cite_r={row.get('cite_dist_corr', float('nan'))} "
                        f"({row['seconds']:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {arm} seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                    rows.append({"dataset": dname, "arm": arm, "seed": seed, "error": str(e)})
                pd.DataFrame(rows).to_csv(csv_path, index=False)

            # --- fair-k: scanpy at same final n as scfair_auto this seed ---
            try:
                auto_rows = [
                    r
                    for r in rows
                    if r.get("dataset") == dname
                    and r.get("arm") == "scfair_auto"
                    and r.get("seed") == seed
                    and "n_genes" in r
                ]
                if not auto_rows:
                    raise RuntimeError("no scfair_auto row for fair-k")
                n_match = int(auto_rows[-1]["n_genes"])
                genes_m = select_hvg_match(adata, n_match, seed)
                row = run_one(
                    dname,
                    "hvg_match_auto",
                    seed,
                    adata,
                    protein,
                    genes=genes_m,
                    meta_extra={"n_base": n_match, "fair_k_matched_to": n_match},
                )
                rows.append(row)
                print(
                    f"  {'hvg_match_auto':14s} seed={seed} n={row['n_genes']:5d} "
                    f"ARI={row['ARI']:.3f} ASW={row['ASW']:.3f} "
                    f"kNN={row['knn_acc']:.3f} ({row['seconds']:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  hvg_match_auto seed={seed} FAIL {type(e).__name__}: {e}", flush=True)
                rows.append(
                    {"dataset": dname, "arm": "hvg_match_auto", "seed": seed, "error": str(e)}
                )
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        del adata
        print(f"  [{dname} done in {time.time() - t_ds:.0f}s]", flush=True)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _write_summary(csv_path)
    print(f"\nDONE -> {csv_path}", flush=True)


def _write_summary(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty or "ARI" not in df.columns:
        return
    ok = df[df["ARI"].notna()].copy()
    g = ok.groupby(["dataset", "arm"], as_index=False)[
        [
            "ARI",
            "NMI",
            "ASW",
            "knn_acc",
            "var_ratio",
            "macro_f1",
            "rare_f1",
            "cite_dist_corr",
            "cite_ari_rna_vs_adt",
            "n_genes",
        ]
    ].mean(numeric_only=True)
    # pivot deltas vs hvg2000
    rows = []
    for ds, sub in g.groupby("dataset"):
        base = sub[sub.arm == "hvg2000"]
        auto = sub[sub.arm == "scfair_auto"]
        match = sub[sub.arm == "hvg_match_auto"]
        if base.empty or auto.empty:
            continue
        b, a = base.iloc[0], auto.iloc[0]
        rec = {
            "dataset": ds,
            "n_genes_hvg2000": b["n_genes"],
            "n_genes_auto": a["n_genes"],
            "ARI_hvg": b["ARI"],
            "ARI_auto": a["ARI"],
            "dARI": a["ARI"] - b["ARI"],
            "ASW_hvg": b["ASW"],
            "ASW_auto": a["ASW"],
            "dASW": a["ASW"] - b["ASW"],
            "knn_hvg": b["knn_acc"],
            "knn_auto": a["knn_acc"],
            "d_knn": a["knn_acc"] - b["knn_acc"],
            "varR_hvg": b["var_ratio"],
            "varR_auto": a["var_ratio"],
            "d_varR": a["var_ratio"] - b["var_ratio"],
            "cite_r_hvg": b.get("cite_dist_corr", np.nan),
            "cite_r_auto": a.get("cite_dist_corr", np.nan),
        }
        if not match.empty:
            m = match.iloc[0]
            rec["ARI_hvg_match"] = m["ARI"]
            rec["dARI_auto_vs_match"] = a["ARI"] - m["ARI"]
            rec["ASW_hvg_match"] = m["ASW"]
            rec["dASW_auto_vs_match"] = a["ASW"] - m["ASW"]
        rows.append(rec)
    summ = pd.DataFrame(rows)
    out = csv_path.with_name(csv_path.stem + "_SUMMARY.csv")
    summ.to_csv(out, index=False)
    # also docs
    docs = ROOT.parent / "docs"
    if docs.is_dir() and len(summ):
        summ.to_csv(docs / "PAPER_pool_extra_metrics_SUMMARY.csv", index=False)
    print(f"SUMMARY -> {out}", flush=True)
    if len(summ):
        print(
            f"  mean dARI={summ['dARI'].mean():+.4f}  dASW={summ['dASW'].mean():+.4f}  "
            f"d_knn={summ['d_knn'].mean():+.4f}",
            flush=True,
        )
        if "dARI_auto_vs_match" in summ:
            print(
                f"  fair-k mean dARI(auto−match)={summ['dARI_auto_vs_match'].mean():+.4f}",
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="*", help="datasets (default: all GOLD)")
    p.add_argument("--out", default="gold_p0p1", help="output CSV suffix")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for n in GOLD_DATASETS:
            print(f"{n:40s} {'BIG' if n in BIG else 'small'}")
        sys.exit(0)
    names = args.names if args.names else list(GOLD_DATASETS)
    unknown = [n for n in names if n not in LOADERS]
    if unknown:
        sys.exit(f"unknown datasets: {unknown}")
    main(names, args.out)
