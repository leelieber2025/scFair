#!/usr/bin/env python
"""Is scFair's gain real once the baseline is a *good* one? (§5.15)

Zhao et al. (*Genome Biology* 2025, `mixhvg`) benchmarked 21 HVG baselines over
19 datasets. Their top four are ``poisson_scran``, ``mv_lognc_scran``,
``disp_nc_seuratv1`` and ``mv_nc`` — and **`seurat_v3`, which scFair has used as
its baseline in every comparison from §5.1 to §5.14, is not among them.** Every
"scFair beats scanpy" number in this project may therefore have been measured
against a mediocre reference.

Three things are tested here, all pure Python (no rpy2):

  1. **Baseline check.** Where does `seurat_v3` sit against the stronger
     baselines, and does scFair's margin survive when they replace it?
  2. **Combination operator.** Zhao et al.'s hybrid rule is *best-rank* — each
     gene takes its best rank across methods, which can only ever promote a
     gene. scFair's rule is a linear blend, which can *demote*, and §5.10 traced
     the rare-population collapse to exactly that. Tested as a 2×3 factorial:
     {seurat_v3, best-rank mix} × {no specificity, scFair blend, scFair best-rank}.
  3. **Clustering-free metrics.** Everything in §5.9–5.14 ran Leiden at
     resolution 0.8 and scored ARI, and §5.9 showed that hides populations the
     resolution cannot resolve (Treg: F1 0.10–0.24 for *every* method). Zhao et
     al. avoid this with metrics that never cluster. Added here, alongside a
     **random-gene negative control** which this project has never had — without
     it there is no way to know what fraction of the achievable range a +2.5%
     gain represents.

Outputs: examples/results/mixhvg_comparison.csv / .json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

import scfair as scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adt_gold_benchmark import load_labeled as load_adt14  # noqa: E402
from adt_multi_validation import load_cite  # noqa: E402
from hvg_baselines import BASELINES, best_rank_combine  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = Path(__file__).resolve().parent / "results"
SEEDS = [0, 1, 2]
N_TOP = 2000
MIX = ["mv_lognc_scran", "poisson_scran", "disp_nc_seuratv1"]

DATASETS = {
    "pbmc10k_adt14": (load_adt14, "protein_expression"),
    "pbmc5k_adt29": (lambda: load_cite("pbmc_5k_v3"), "protein_expression"),
    "sln_208_mouse": (lambda: load_cite("sln_208_mouse"), "protein_expression"),
    "pbmc_seurat_v4_20k": (lambda: load_cite("pbmc_seurat_v4_20k"), "protein_counts"),
}


def clr(P: np.ndarray) -> np.ndarray:
    L = np.log1p(np.asarray(P, dtype=float))
    return L - L.mean(axis=0, keepdims=True)


def protein_space(adata, key: str) -> np.ndarray:
    P = adata.obsm[key]
    mat = P.values if hasattr(P, "values") else np.asarray(P)
    C = clr(mat)
    return (C - C.mean(0)) / (C.std(0) + 1e-9)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def genes_from_score(score: pd.Series, k: int) -> list[str]:
    return list(score.sort_values(ascending=False).index[:k])


def build_arms(adata, base_scores: dict[str, pd.Series], seed: int):
    """Return {arm_name: gene list}. Baselines are seed-free except `random`."""
    k = min(N_TOP, adata.n_vars - 1)
    mix_score = best_rank_combine({m: base_scores[m] for m in MIX})
    arms: dict[str, list[str]] = {
        "random": genes_from_score(BASELINES["random"](adata, seed=seed), k),
        "seurat_v3": genes_from_score(base_scores["logmv_ct_seuratv3"], k),
        "poisson_scran": genes_from_score(base_scores["poisson_scran"], k),
        "mv_lognc_scran": genes_from_score(base_scores["mv_lognc_scran"], k),
        "disp_nc_seuratv1": genes_from_score(base_scores["disp_nc_seuratv1"], k),
        "mix3": genes_from_score(mix_score, k),
    }
    # 2x2 factorial: anchor x scFair combination rule
    for anchor_name, anchor in (("sv3", None), ("mix3", mix_score)):
        for comb in ("blend", "best_rank"):
            a = adata.copy()
            kw = dict(
                n_top_genes=k,
                flavor="seurat_v3",
                layer="counts",
                marker_mode="none",
                balance_method="hybrid",
                combine=comb,
                random_state=seed,
            )
            if anchor is not None:
                kw["global_score"] = anchor
            scf.pp.highly_variable_genes(a, **kw)
            arms[f"scfair_{anchor_name}_{comb}"] = (
                a.var_names[a.var["highly_variable"]].astype(str).tolist()
            )
    return arms


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def rna_embedding(adata, genes, seed):
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, [g for g in genes if g in a.var_names]].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    return a, a.obsm["X_pca"][:, : min(30, n_comps)]


def clustering_free_metrics(emb: np.ndarray, prot: np.ndarray, seed: int, k: int = 30):
    """Metrics that never cluster, so no Leiden resolution is baked in.

    dist_corr   Pearson r between RNA-space and protein-space pairwise distances
                over a cell subsample. Higher = the selected genes preserve the
                geometry the protein sees.
    knn_mse     Predict each cell's protein vector from the mean of its RNA
                neighbours. Lower = RNA neighbourhoods are protein-coherent.
    knn_ratio   Protein distance to RNA-kNN over protein distance to random
                cells. Lower = better.
    """
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    sub = rng.choice(n, size=min(1500, n), replace=False)
    de = np.linalg.norm(emb[sub][:, None, :] - emb[sub][None, :, :], axis=-1)
    dp = np.linalg.norm(prot[sub][:, None, :] - prot[sub][None, :, :], axis=-1)
    iu = np.triu_indices(len(sub), k=1)
    dist_corr = float(np.corrcoef(de[iu], dp[iu])[0, 1])

    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    pred = prot[idx].mean(axis=1)
    knn_mse = float(np.mean((pred - prot) ** 2))

    d_knn = float(np.mean(np.linalg.norm(prot[idx] - prot[:, None, :], axis=-1)))
    rand_idx = rng.integers(0, n, size=idx.shape)
    d_rand = float(np.mean(np.linalg.norm(prot[rand_idx] - prot[:, None, :], axis=-1)))
    return {
        "dist_corr": dist_corr,
        "knn_mse": knn_mse,
        "knn_ratio": d_knn / d_rand if d_rand else np.nan,
    }


def label_metrics(a, seed):
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, a.obsm["X_pca"].shape[1]), random_state=seed)
    sc.tl.leiden(
        a, resolution=0.8, key_added="leiden", flavor="igraph", n_iterations=2, random_state=seed
    )
    conf = (
        a.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in a.obs.columns
        else np.ones(a.n_obs, bool)
    )
    y_true = a.obs["cell_type"].astype(str)[conf]
    y_pred = a.obs["leiden"].astype(str)[conf]
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
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1s.values()))),
    }


def main(which=None):
    rows = []
    for dname in which or DATASETS:
        loader, pkey = DATASETS[dname]
        print(f"\n################ {dname} ################", flush=True)
        adata = loader()
        prot = protein_space(adata, pkey)
        print(f"  {adata.n_obs} x {adata.n_vars}, {prot.shape[1]} proteins", flush=True)
        base_scores = {
            n: (BASELINES[n](adata, seed=0) if n == "poisson_scran" else BASELINES[n](adata))
            for n in ("logmv_ct_seuratv3", "poisson_scran", "mv_lognc_scran", "disp_nc_seuratv1")
        }
        for seed in SEEDS:
            arms = build_arms(adata, base_scores, seed)
            for arm, genes in arms.items():
                try:
                    a, emb = rna_embedding(adata, genes, seed)
                    res = clustering_free_metrics(emb, prot, seed)
                    res.update(label_metrics(a, seed))
                    res.update({"dataset": dname, "arm": arm, "seed": seed, "n_genes": len(genes)})
                    rows.append(res)
                    print(
                        f"  {arm:22s} s={seed} distcorr={res['dist_corr']:.3f} "
                        f"knnMSE={res['knn_mse']:.4f} ratio={res['knn_ratio']:.3f} "
                        f"ARI={res['ARI']:.3f} macroF1={res['macro_f1']:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {arm} s={seed} FAIL {type(e).__name__}: {e}", flush=True)
                    rows.append({"dataset": dname, "arm": arm, "seed": seed, "error": str(e)})
            pd.DataFrame(rows).to_csv(OUT / "mixhvg_comparison.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mixhvg_comparison.csv", index=False)
    with open(OUT / "mixhvg_comparison.json", "w") as f:
        json.dump(df.replace({np.nan: None}).to_dict(orient="records"), f, indent=2, default=str)
    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df
    for m in ("dist_corr", "knn_mse", "knn_ratio", "ARI", "macro_f1"):
        if m in ok.columns:
            print(f"\n======== {m} ========")
            print(ok.pivot_table(index="arm", columns="dataset", values=m).round(4).to_string())
    print("\nDONE")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(which=args or None)
