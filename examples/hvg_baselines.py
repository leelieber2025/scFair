#!/usr/bin/env python
"""Pure-Python reimplementations of the HVG baselines ranked best by Zhao et al.

Zhao, Lu, Li, Zhou, Zhao & Ji, *Genome Biology* 2025, "A systematic evaluation
of highly variable gene selection methods for single-cell RNA-sequencing"
(the `mixhvg` paper) benchmarked 21 baselines over 19 datasets and found the
top four to be ``poisson_scran``, ``mv_lognc_scran``, ``disp_nc_seuratv1`` and
``mv_nc``. **`seurat_v3` — scFair's baseline everywhere in §5.1–5.14 — is not
among them.** So every scFair-vs-baseline number in this project may have been
measured against a mediocre reference. This module exists to find out.

These are **reimplementations from the method description, not ports of the R
originals** (no rpy2 anywhere in this project). They are intended to be faithful
in mechanism, not bit-identical to scran/Seurat. Treat absolute values as
indicative; the comparisons they enable are between methods run through the
same code path, which is what matters here.

Taxonomy follows the paper: <adjustment>_<transform>, where transform is
ct (raw counts), nc (normalized counts) or lognc (log-normalized counts), and
adjustment is mv (variance vs mean trend residual), logmv (log-variance vs
log-mean), disp (variance/mean, binned z-score) or mean_max (mean expression).

Every function returns a pandas Series of scores indexed by var_names, higher =
more variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.nonparametric.smoothers_lowess import lowess

TARGET_SUM = 1e4


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _counts(adata) -> sp.csr_matrix:
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()


def _size_factors(X: sp.csr_matrix) -> np.ndarray:
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    tot[tot == 0] = 1.0
    return tot


def _normalize(X: sp.csr_matrix, target_sum: float = TARGET_SUM) -> sp.csr_matrix:
    s = _size_factors(X)
    inv = sp.diags(target_sum / s)
    return (inv @ X).tocsr()


def _mean_var(X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and (population) variance, sparse-safe."""
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = X.copy()
    sq.data = sq.data**2
    mean_sq = np.asarray(sq.mean(axis=0)).ravel()
    var = mean_sq - mean**2
    var *= n / max(n - 1, 1)
    return mean, np.maximum(var, 0.0)


def _trend_residual(mean: np.ndarray, var: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """var - loess_trend(mean); scran's 'biological component' in spirit."""
    keep = mean > 0
    resid = np.zeros_like(var)
    if keep.sum() < 10:
        return var
    x = np.log10(mean[keep])
    y = var[keep]
    fitted = lowess(y, x, frac=frac, return_sorted=False, it=1)
    resid[keep] = y - fitted
    return resid


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def mv_lognc_scran(adata) -> pd.Series:
    """scran ``modelGeneVar``: variance-vs-mean trend residual on log-norm counts."""
    X = _normalize(_counts(adata))
    X.data = np.log1p(X.data)
    mean, var = _mean_var(X)
    return pd.Series(_trend_residual(mean, var), index=adata.var_names)


def mv_nc(adata) -> pd.Series:
    """Variance-vs-mean trend residual on normalized (not logged) counts."""
    X = _normalize(_counts(adata))
    mean, var = _mean_var(X)
    return pd.Series(_trend_residual(mean, var), index=adata.var_names)


def poisson_scran(adata, n_grid: int = 100, seed: int = 0) -> pd.Series:
    """scran ``modelGeneVarByPoisson``: trend from *simulated* Poisson noise.

    The technical variance curve is obtained by simulating Poisson counts at a
    grid of true means using the observed library sizes, log-normalizing them
    exactly as the real data, and recording the resulting (mean, variance).
    A gene's score is its observed variance minus that curve — variance beyond
    what pure counting noise would produce.
    """
    C = _counts(adata)
    s = _size_factors(C)
    Xl = _normalize(C)
    Xl.data = np.log1p(Xl.data)
    mean, var = _mean_var(Xl)

    rng = np.random.default_rng(seed)
    n_cells = C.shape[0]
    # grid over per-cell expected counts, spanning the observed depth range
    gene_tot = np.asarray(C.sum(axis=0)).ravel()
    lam_hi = max(gene_tot.max() / n_cells, 1e-3)
    lams = np.logspace(-4, np.log10(lam_hi * 2 + 1e-3), n_grid)
    sim_mean, sim_var = np.empty(n_grid), np.empty(n_grid)
    # cap simulated cells for speed; variance estimate stays stable
    idx = rng.choice(n_cells, size=min(n_cells, 4000), replace=False)
    s_sub = s[idx]
    for i, lam in enumerate(lams):
        counts = rng.poisson(lam * s_sub / s_sub.mean())
        vals = np.log1p(counts * TARGET_SUM / s_sub)
        sim_mean[i] = vals.mean()
        sim_var[i] = vals.var(ddof=1)

    order = np.argsort(sim_mean)
    technical = np.interp(mean, sim_mean[order], sim_var[order])
    return pd.Series(var - technical, index=adata.var_names)


def disp_nc_seuratv1(adata, n_bins: int = 20) -> pd.Series:
    """Seurat v1 dispersion on normalized counts: binned z-score of var/mean."""
    X = _normalize(_counts(adata))
    mean, var = _mean_var(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(mean > 0, var / mean, np.nan)
    df = pd.DataFrame({"mean": mean, "disp": disp})
    df["bin"] = pd.cut(df["mean"].rank(method="first"), bins=n_bins, labels=False)
    grp = df.groupby("bin")["disp"]
    z = (df["disp"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    return pd.Series(z.fillna(0.0).to_numpy(), index=adata.var_names)


def logmv_ct_seuratv3(adata) -> pd.Series:
    """scanpy ``flavor='seurat_v3'`` — the baseline scFair has used throughout."""
    import scanpy as sc

    a = adata.copy()
    sc.pp.highly_variable_genes(
        a, n_top_genes=min(2000, a.n_vars - 1), flavor="seurat_v3", layer="counts"
    )
    return pd.Series(
        pd.to_numeric(a.var["variances_norm"], errors="coerce").fillna(0.0).to_numpy(),
        index=adata.var_names,
    )


def mean_max_nc(adata) -> pd.Series:
    """Highest average expression. Weak alone; a diversity term in ensembles."""
    X = _normalize(_counts(adata))
    mean, _ = _mean_var(X)
    return pd.Series(mean, index=adata.var_names)


def random_genes(adata, seed: int = 0) -> pd.Series:
    """Negative control. Calibrates how much of the achievable range a gain covers."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(adata.n_vars).astype(float), index=adata.var_names)


BASELINES = {
    "poisson_scran": poisson_scran,
    "mv_lognc_scran": mv_lognc_scran,
    "disp_nc_seuratv1": disp_nc_seuratv1,
    "mv_nc": mv_nc,
    "logmv_ct_seuratv3": logmv_ct_seuratv3,
    "mean_max_nc": mean_max_nc,
    "random": random_genes,
}

# The paper's best 3- and 4-way mixes, by their naming.
MIXES = {
    "mix_1mv3pos": ["mv_lognc_scran", "poisson_scran"],
    "mix_1mv3pos4dis": ["mv_lognc_scran", "poisson_scran", "disp_nc_seuratv1"],
    "mix_all4": ["mv_lognc_scran", "poisson_scran", "disp_nc_seuratv1", "mv_nc"],
}


def best_rank_combine(scores: dict[str, pd.Series]) -> pd.Series:
    """Zhao et al.'s hybrid rule: each gene takes its **best** rank across methods.

    Ties on the best rank are broken by the second-best rank. Returned as a
    score (higher = better) so it drops into the same ranking machinery as the
    individual baselines.

    Structurally this operator can only ever *promote* a gene relative to its
    best single method — it can never push a gene below where every method put
    it. That matters for scFair: §5.10 traced the rare-population failure to the
    linear blend *demoting* genes that carried a fine boundary.
    """
    ranks = pd.DataFrame({k: v.rank(ascending=False, method="average") for k, v in scores.items()})
    best = ranks.min(axis=1)
    second = ranks.apply(lambda r: np.sort(r.to_numpy())[1] if len(r) > 1 else r.iloc[0], axis=1)
    combined = best + 1e-6 * second
    return -combined


def compute_all(adata, names=None, seed: int = 0) -> dict[str, pd.Series]:
    out = {}
    for name in names or BASELINES:
        fn = BASELINES[name]
        out[name] = fn(adata, seed=seed) if name in ("random", "poisson_scran") else fn(adata)
    return out
