"""Automatic selection of n_top_genes for HVG.

All methods operate on a **descending** gene score vector (higher = more
variable / preferred) and optional cell data for downstream criteria.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _progress(on: bool, msg: str, *args: Any) -> None:
    """Mirrors ``_highly_variable_genes._progress`` -- kept local to avoid a
    circular import (that module already imports from this one)."""
    text = msg % args if args else msg
    logger.info(text)
    if on:
        print(f"scfair: {text}", file=sys.stderr, flush=True)


AutoNMethod = Literal[
    "elbow",
    "knee",
    "cumfrac",
    "silhouette",
    "coverage",
    "ensemble",
    "populations",
    "structure",
]

# Genes-per-population multiplier for the `populations` estimator.
# Provisional: the direction (more populations need more genes) is
# established but the exact constant is not well-calibrated yet; see
# `examples/auto_n_populations.py`.
GENES_PER_POPULATION = 150


def _clip_k(k: int, k_min: int, k_max: int, n_genes: int) -> int:
    k_max_eff = min(k_max, n_genes)
    k_min_eff = min(k_min, k_max_eff)
    return int(max(k_min_eff, min(int(k), k_max_eff)))


def select_n_top_elbow(
    scores_desc: np.ndarray,
    *,
    k_min: int = 500,
    k_max: int = 5000,
) -> int:
    """Largest-gap / max perpendicular distance from first-last chord (elbow)."""
    s = np.asarray(scores_desc, dtype=float)
    n = s.size
    if n == 0:
        return k_min
    lo = min(k_min, n) - 1  # 0-based inclusive start for search
    hi = min(k_max, n)  # exclusive end index for candidates 1..hi
    lo = max(lo, 0)
    if hi <= lo + 1:
        return _clip_k(hi, k_min, k_max, n)

    # Work on ranks 0..hi-1 of the sorted curve
    y = s[:hi]
    # normalize to [0,1] for geometry
    x = np.linspace(0.0, 1.0, hi)
    y0, y1 = y[0], y[-1]
    y_line = y0 + (y1 - y0) * x
    # distance above the chord (elbow for decreasing curve)
    dist = y_line - y
    # only search from k_min
    start = max(lo, 1)
    idx = start + int(np.argmax(dist[start:hi]))
    k = idx + 1  # convert to count
    return _clip_k(k, k_min, k_max, n)


def select_n_top_knee(
    scores_desc: np.ndarray,
    *,
    k_min: int = 500,
    k_max: int = 5000,
) -> int:
    """KneeLocator on the decreasing score curve (falls back to elbow)."""
    s = np.asarray(scores_desc, dtype=float)
    n = s.size
    hi = min(k_max, n)
    if hi < max(k_min, 10):
        return _clip_k(hi, k_min, k_max, n)
    try:
        from kneed import KneeLocator

        x = np.arange(1, hi + 1)
        y = s[:hi]
        # decreasing convex-looking curve
        kl = KneeLocator(x, y, curve="convex", direction="decreasing", interp_method="interp1d")
        k = int(kl.knee) if kl.knee is not None else select_n_top_elbow(s, k_min=k_min, k_max=k_max)
    except Exception as exc:
        logger.debug("KneeLocator failed (%s); using elbow.", exc)
        k = select_n_top_elbow(s, k_min=k_min, k_max=k_max)
    return _clip_k(k, k_min, k_max, n)


def select_n_top_cumfrac(
    scores_desc: np.ndarray,
    *,
    frac: float = 0.8,
    k_min: int = 500,
    k_max: int = 5000,
) -> int:
    """Smallest k such that cumulative score reaches ``frac`` of total (clipped)."""
    s = np.asarray(scores_desc, dtype=float)
    n = s.size
    if n == 0:
        return k_min
    s = np.maximum(s, 0.0)
    hi = min(k_max, n)
    total = float(s[:hi].sum())
    if total <= 0:
        return _clip_k(k_min, k_min, k_max, n)
    c = np.cumsum(s[:hi]) / total
    # first index where cum >= frac
    idx = int(np.searchsorted(c, frac, side="left"))
    k = min(idx + 1, hi)
    return _clip_k(k, k_min, k_max, n)


def select_n_top_coverage(
    gene_order: Sequence[str],
    cluster_gene_ranks: dict[str, Sequence[str]],
    *,
    min_per_cluster: int = 20,
    k_min: int = 500,
    k_max: int = 5000,
) -> int:
    """Minimal k so each cluster has ``min_per_cluster`` of its top genes in top-k.

    ``cluster_gene_ranks[cl]`` is genes ordered best→worst for that cluster
    (e.g. by one-sided logFC).
    """
    n = len(gene_order)
    if n == 0 or not cluster_gene_ranks:
        return k_min
    # map gene -> position in global order (0-based)
    pos = {str(g): i for i, g in enumerate(gene_order)}
    # for each cluster, positions of its top genes that appear in global order
    need_positions: list[int] = []
    for cl, ranked in cluster_gene_ranks.items():
        hits = []
        for g in ranked:
            gs = str(g)
            if gs in pos:
                hits.append(pos[gs])
            if len(hits) >= min_per_cluster:
                break
        if len(hits) < min_per_cluster:
            # not enough genes — take whatever we have
            if hits:
                need_positions.append(max(hits))
        else:
            need_positions.append(hits[min_per_cluster - 1])
    if not need_positions:
        return _clip_k(k_min, k_min, k_max, n)
    k = max(need_positions) + 1  # inclusive count
    return _clip_k(k, k_min, k_max, n)


def select_n_top_silhouette(
    adata: Any,
    gene_order: Sequence[str],
    *,
    candidates: Sequence[int] | None = None,
    k_min: int = 500,
    k_max: int = 5000,
    counts_layer: str = "counts",
    n_pcs: int = 30,
    n_neighbors: int = 15,
    resolution: float = 0.8,
    random_state: int = 0,
) -> tuple[int, dict[int, float]]:
    """Pick k maximizing Leiden silhouette in PCA space (downstream proxy)."""
    import scanpy as sc
    from sklearn.metrics import silhouette_score

    n = len(gene_order)
    k_max_eff = min(k_max, n)
    k_min_eff = min(k_min, k_max_eff)
    if candidates is None:
        # log-ish grid
        raw = [500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]
        candidates = sorted({_clip_k(k, k_min_eff, k_max_eff, n) for k in raw})
    else:
        candidates = sorted({_clip_k(int(k), k_min_eff, k_max_eff, n) for k in candidates})

    # prepare log-normalized full object once
    ad0 = adata.copy()
    if counts_layer in ad0.layers:
        ad0.X = ad0.layers[counts_layer].copy()
    ad0.uns.pop("log1p", None)
    sc.pp.normalize_total(ad0, target_sum=1e4)
    sc.pp.log1p(ad0)

    scores: dict[int, float] = {}
    best_k = candidates[0]
    best_s = -np.inf
    for k in candidates:
        genes = list(gene_order[:k])
        ad = ad0[:, genes].copy()
        sc.pp.scale(ad, max_value=10)
        n_comps = min(n_pcs, ad.n_vars - 1, ad.n_obs - 1)
        if n_comps < 2:
            scores[k] = -1.0
            continue
        sc.tl.pca(ad, n_comps=n_comps, svd_solver="arpack", random_state=random_state)
        sc.pp.neighbors(
            ad,
            n_neighbors=min(n_neighbors, ad.n_obs - 1),
            n_pcs=n_comps,
            random_state=random_state,
        )
        sc.tl.leiden(
            ad,
            resolution=resolution,
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            key_added="leiden",
        )
        X = ad.obsm["X_pca"][:, : min(20, n_comps)]
        labels = ad.obs["leiden"].astype(str)
        if labels.nunique() < 2:
            s = -1.0
        else:
            s = float(
                silhouette_score(
                    X, labels, sample_size=min(2000, ad.n_obs), random_state=random_state
                )
            )
        scores[k] = s
        if s > best_s:
            best_s = s
            best_k = k
    return best_k, scores


def effective_k_floor(
    n_genes: int,
    *,
    k_min: int = 500,
    k_max: int = 5000,
    soft_floor: int = 1000,
    soft_frac: float = 0.2,
) -> int:
    """Protective lower bound for auto-n (guards against collapsing rare,
    low-abundance cell types when k is small).

    ``k_floor = max(k_min, min(soft_floor, floor(n_genes * soft_frac)))``,
    then clipped to ``[1, k_max, n_genes]``.
    """
    n_genes = max(int(n_genes), 1)
    k_max_eff = min(int(k_max), n_genes)
    frac_cap = max(1, int(n_genes * float(soft_frac)))
    floor = max(int(k_min), min(int(soft_floor), frac_cap))
    return int(max(1, min(floor, k_max_eff)))


def effective_k_ceiling(
    n_genes: int,
    *,
    k_min: int = 500,
    k_max: int = 5000,
    soft_ceiling: int = 2500,
) -> int:
    """Soft upper bound for auto-n final k.

    Large k rarely improves downstream separation further and can erase
    the advantage of a well-chosen gene set. Default ceiling 2500 allows
    modest excursions above 2000 without permitting very large k.
    """
    n_genes = max(int(n_genes), 1)
    hi = min(int(soft_ceiling), int(k_max), n_genes)
    lo = min(int(k_min), hi)
    return int(max(lo, hi))


def library_depth_stats(
    adata: Any,
    counts_layer: str = "counts",
) -> dict[str, float]:
    """Median library size and genes/cell from a counts matrix."""
    from scipy import sparse

    if counts_layer in getattr(adata, "layers", {}):
        X = adata.layers[counts_layer]
    else:
        X = adata.X
    if sparse.issparse(X):
        n_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(float)
    else:
        Xa = np.asarray(X, dtype=float)
        n_counts = Xa.sum(axis=1)
        n_genes = (Xa > 0).sum(axis=1)
    return {
        "median_counts": float(np.median(n_counts)) if n_counts.size else 0.0,
        "mean_counts": float(np.mean(n_counts)) if n_counts.size else 0.0,
        "median_genes": float(np.median(n_genes)) if n_genes.size else 0.0,
        "mean_genes": float(np.mean(n_genes)) if n_genes.size else 0.0,
        "n_cells": float(n_counts.size),
    }


def depth_aware_auto_knobs(
    *,
    median_counts: float,
    median_genes: float,
    n_genes: int,
    k_min: int = 500,
    k_max: int = 5000,
) -> dict[str, Any]:
    """Map library depth → auto-n prior (anchor / floor / ceiling).

    - **shallow** (median counts ≲500 or genes/cell ≲300): fewer
      informative genes → lower ceiling; avoid over-picking noise.
    - **medium** (typical 10x ~1–5k UMI): anchor 2000, ceiling 2500.
    - **deepish** (high genes or ~10k+ counts): anchor/ceiling pulled down
      slightly.
    - **deep** (full-length-like, counts ≳5e4 or genes/cell ≳4000): smaller
      k tends to work better → anchor/ceiling pulled down further.

    Floor is never allowed to reach very low k, to avoid losing rare,
    low-abundance cell types.
    """
    c = float(median_counts)
    g = float(median_genes)
    n_genes = max(int(n_genes), 1)

    if c < 500.0 or g < 300.0:
        # Shallow: cap high-k noise, but keep ceiling/anchor at 2000 rather
        # than pulling them lower.
        tier = "shallow"
        soft_floor, soft_ceiling, anchor = 1000, 2000, 2000
    elif c >= 5.0e4 or g >= 4000.0:
        # Deep / full-length: smaller k tends to work better here.
        tier = "deep"
        soft_floor, soft_ceiling, anchor = 1000, 2000, 1500
    elif c >= 1.0e4 or g >= 2500.0:
        tier = "deepish"
        soft_floor, soft_ceiling, anchor = 1000, 2200, 1800
    else:
        # Typical 10x UMI band.
        tier = "medium"
        soft_floor, soft_ceiling, anchor = 1000, 2500, 2000

    # Respect pool size and user k_max / k_min
    soft_ceiling = int(min(soft_ceiling, k_max, n_genes))
    soft_floor = int(min(max(soft_floor, k_min), soft_ceiling))
    anchor = int(min(max(anchor, soft_floor), soft_ceiling))

    return {
        "depth_tier": tier,
        "median_counts": c,
        "median_genes": g,
        "soft_floor": soft_floor,
        "soft_ceiling": soft_ceiling,
        "anchor": anchor,
    }


def select_n_top_from_populations(
    n_populations: int,
    *,
    sizes: Sequence[int] | None = None,
    size_power: float = 0.0,
    genes_per_population: int = GENES_PER_POPULATION,
    k_min: int = 500,
    k_max: int = 5000,
    n_genes: int,
) -> int:
    """``k`` from how many populations have to be separated.

    Other estimators in this module infer k from the shape of the global
    score curve, a property of the HVG flavor's variance model rather than
    of the data's cluster structure. This estimator instead uses the
    number of populations the density field resolves
    (:mod:`scfair.pp._granularity`), which is free when
    ``resolution="auto"`` already ran since that computes the same
    embedding.

    Equal share, or more genes for bigger populations?
    ---------------------------------------------------
    ``size_power=0`` (default) gives every population the same share.
    ``size_power=b`` makes the effective count ``sum((s_p / median_s) ** b)``,
    so a population an order of magnitude above the median contributes
    ``10**b`` shares instead of one.

    The default is equal share, for three reasons:

    1. Characterising a cell type takes about as many genes whether it holds
       100 cells or 10,000.
    2. Internal heterogeneity is already accounted for: if the density field
       resolves the sub-structure it counts as more populations, and if it
       does not, there is no basis for allocating more.
    3. **Size-dependence already exists elsewhere.** ``balance_power`` weights
       clusters by ``size ** 0.5`` when aggregating specificity, so larger
       populations already claim more of the list. Making ``k`` scale with
       size too would apply the same prior twice, and this package's
       ``balance_power=0.5`` (rather than 1.0) exists precisely to stop
       common populations from dominating.

    ``select_n_top_coverage``'s ``min_per_cluster`` is the same equal-share
    rule already present in this module.
    """
    n_pop = max(1, int(n_populations))
    effective = float(n_pop)
    if sizes is not None and size_power:
        s = np.asarray([x for x in sizes if x > 0], dtype=float)
        if s.size:
            ref = float(np.median(s))
            if ref > 0:
                effective = float(np.sum((s / ref) ** float(size_power)))
    return _clip_k(int(round(effective * float(genes_per_population))), k_min, k_max, n_genes)


def select_n_top_from_structure(
    *,
    valley_median: float,
    frac_shallow: float,
    n_density_pops: int,
    mean_stability: float | None = None,
    min_stability: float | None = None,
    n_obs: int | None = None,
    n_leiden: int | None = None,
    version: str = "v7",
    k_min: int = 500,
    k_max: int = 4000,
    n_genes: int = 50_000,
    # guards; exposed for experiments
    atlas_n_obs: int = 10_000,
    atlas_n_pops: int = 15,
    atlas_k_floor: int = 2000,
    density_surplus_ratio: float = 1.0,
    # v7 fine-atlas band parameters
    fine_nd_lo: int = 12,
    fine_nd_hi: int = 20,
    fine_vm_lo: float = 0.78,
    fine_ratio_hi: float = 1.12,
) -> int:
    """Structure-aware ``n_top`` (density valleys + pop count).

    Default for ``n_top_genes="auto"`` / ``auto_n_method="structure"`` is
    v7; older versions (v4-v6) remain selectable via ``version=``.

    Rationale
    ---------
    Global variance elbows are blind to structure. 3D density **valley
    geometry** tracks best-k better: few cores + shallow valleys → longer
    lists; many deep valleys on compact data → shorter lists.

    v7 adds a **fine-atlas band** guard: on large ``n_obs`` datasets with
    density cores in a mid band, deep valleys, and Leiden not far above
    density (``n_leiden/n_density_pops <= fine_ratio_hi``), it allocates a
    per-population budget instead of falling through to the short-k rules
    below.

    Pass ``n_obs`` and ``n_leiden`` for v5+; without them behaviour
    approximates v4.
    """
    ver = str(version).lower().strip()
    if ver not in ("v4", "v5", "v6", "v7"):
        raise ValueError(f"Unknown structure auto_n version={version!r}; use 'v4'–'v7'.")

    vm = float(valley_median)
    fs = float(frac_shallow)
    nd = float(n_density_pops)
    ms = float(mean_stability) if mean_stability is not None else float("nan")
    mu = float(min_stability) if min_stability is not None else float("nan")
    n_cells = int(n_obs) if n_obs is not None else None
    nl = float(n_leiden) if n_leiden is not None else float("nan")
    ratio = (nl / nd) if (np.isfinite(nl) and np.isfinite(nd) and nd > 0) else float("nan")

    # LONG: few density cores + almost-all shallow valleys
    if np.isfinite(fs) and fs >= 0.85 and np.isfinite(nd) and nd <= 4.5:
        k = 4000 if (np.isfinite(ms) and ms >= 0.8) else 3000
        return _clip_k(k, k_min, k_max, n_genes)

    # --- atlas / fine-structure guards (skip SHORT-hard) ---
    # v5: any large multi-core atlas
    v5_guard = (
        ver == "v5"
        and n_cells is not None
        and n_cells >= int(atlas_n_obs)
        and np.isfinite(nd)
        and nd >= float(atlas_n_pops)
    )
    # v6: strict density surplus (nl/nd < 1)
    density_surplus = np.isfinite(ratio) and ratio < float(density_surplus_ratio)
    v6_guard = (
        ver == "v6"
        and density_surplus
        and n_cells is not None
        and n_cells >= int(atlas_n_obs)
        and np.isfinite(nd)
        and nd >= 12
    )
    # v7: fine-atlas band — mid density-pop count, deep valleys, and
    # near-parity leiden/density ratio (excludes higher-nd / higher-ratio
    # cases).
    v7_guard = (
        ver == "v7"
        and n_cells is not None
        and n_cells >= int(atlas_n_obs)
        and np.isfinite(nd)
        and float(fine_nd_lo) <= nd <= float(fine_nd_hi)
        and np.isfinite(vm)
        and vm >= float(fine_vm_lo)
        and np.isfinite(ratio)
        and ratio <= float(fine_ratio_hi)
    )
    if v5_guard or v6_guard or v7_guard:
        k_budget = int(round(100.0 * nd))
        k = max(int(atlas_k_floor), min(int(k_max), k_budget))
        return _clip_k(k, k_min, k_max, n_genes)

    # SHORT-hard: many deep density cores (compact / classical short-k)
    if np.isfinite(vm) and vm >= 0.80 and np.isfinite(nd) and nd >= 6:
        return _clip_k(500, k_min, k_max, n_genes)
    if np.isfinite(vm) and vm >= 0.70 and np.isfinite(nd) and nd >= 12:
        return _clip_k(500, k_min, k_max, n_genes)

    # MID: deep valleys, moderate pop count (CITE-like)
    # unstable mid (low min/mean pair-stability) bumps one rung up (1500->2000)
    if np.isfinite(vm) and vm >= 0.65 and np.isfinite(nd) and 6 <= nd < 12:
        k_mid = 1500
        if np.isfinite(mu) and mu < 0.1 and np.isfinite(ms) and ms < 0.55:
            k_mid = 2000
        return _clip_k(k_mid, k_min, k_max, n_genes)

    # unstable coarse (false-split risk) → short, unless the LONG pattern
    # above already applied
    if np.isfinite(mu) and mu < 0.25 and np.isfinite(nd) and nd <= 5:
        if not (np.isfinite(fs) and fs >= 0.85):
            return _clip_k(500, k_min, k_max, n_genes)
        if mu < 0.2:
            return _clip_k(500, k_min, k_max, n_genes)

    if np.isfinite(vm) and vm >= 0.65:
        return _clip_k(1000, k_min, k_max, n_genes)
    if np.isfinite(ms) and ms < 0.45 and np.isfinite(vm) and vm >= 0.5:
        return _clip_k(1000, k_min, k_max, n_genes)

    return _clip_k(2000, k_min, k_max, n_genes)


def _prepare_structure_embedding(
    adata: Any,
    *,
    counts_layer: str = "counts",
    random_state: int = 0,
    n_hvg: int = 2000,
) -> tuple[Any | None, dict[str, Any]]:
    """HVG@``n_hvg`` + log1p + PCA + neighbors once (shared multi-seed base).

    Returns ``(embedding_adata, fail_feat_or_empty)``. On failure,
    ``embedding_adata`` is None and the second value is a feature dict with
    ``reason``.
    """
    import scanpy as sc

    ad0 = adata.copy()
    ad0.obs_names_make_unique()
    ad0.var_names_make_unique()
    if counts_layer in getattr(ad0, "layers", {}):
        ad0.X = ad0.layers[counts_layer].copy()
    elif "counts" not in ad0.layers:
        ad0.layers["counts"] = ad0.X.copy()

    sc.pp.highly_variable_genes(
        ad0,
        n_top_genes=min(int(n_hvg), ad0.n_vars),
        flavor="seurat_v3",
        layer=counts_layer if counts_layer in ad0.layers else None,
        subset=False,
    )
    if "highly_variable" not in ad0.var.columns:
        mask = np.ones(ad0.n_vars, dtype=bool)
    else:
        mask = ad0.var["highly_variable"].to_numpy()
        if not mask.any():
            mask = np.ones(ad0.n_vars, dtype=bool)
    e = ad0[:, mask].copy()
    if counts_layer in e.layers:
        e.X = e.layers[counts_layer].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    n_pcs = min(30, e.n_vars - 1, e.n_obs - 1)
    if n_pcs < 2:
        return None, {
            "n_obs": int(adata.n_obs),
            "n_leiden": 1,
            "n_density_pops": 1,
            "valley_median": float("nan"),
            "frac_shallow": float("nan"),
            "mean_stability": float("nan"),
            "min_stability": float("nan"),
            "reason": "too_few_pcs",
        }
    sc.tl.pca(e, n_comps=n_pcs, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(
        e,
        n_neighbors=min(15, e.n_obs - 1),
        n_pcs=n_pcs,
        random_state=random_state,
    )
    return e, {}


# Bootstrap draws for structure-feature pair stability only -- lighter than
# cap-merge's default of 15, since structure only feeds coarse rule
# thresholds.
STRUCTURE_STABILITY_N_BOOT = 5

# Number of seeds the product ``n_top_genes="auto"`` / structure path uses
# for pred_k stability. ``estimate_n_top_structure`` itself still defaults
# to n_seeds=1 for fast one-shot / exploratory calls; product callers pass
# this constant explicitly.
PRODUCT_STRUCTURE_N_SEEDS = 3


def _structure_features_from_embedding(
    e: Any,
    *,
    n_obs: int,
    random_state: int = 0,
    intermediate_resolution: float = 0.5,
    min_cluster_size: int = 30,
    stability_n_boot: int = STRUCTURE_STABILITY_N_BOOT,
) -> dict[str, Any]:
    """Leiden + stability + 3D density valleys on a prepared embedding.

    PCA/neighbors are assumed already present on ``e``. Only steps that
    depend on ``random_state`` are re-run (cheap multi-seed path).

    ``stability_n_boot`` defaults to :data:`STRUCTURE_STABILITY_N_BOOT` (5),
    lighter than cap-merge's 15 — structure only needs coarse thresholds.
    """
    import scanpy as sc

    from scfair.pp._granularity import (
        DEFAULT_DEPTH,
        default_bandwidth,
        knn_density,
        knn_graph,
        population_count_from_embedding,
        _embedding_3d,
    )

    n_boot = max(1, int(stability_n_boot))

    # Work on a shallow-ish copy of obs so parallel seeds don't clobber
    # each other's Leiden keys if we ever thread this; embeddings stay shared.
    e = e.copy()
    sc.tl.leiden(
        e,
        resolution=float(intermediate_resolution),
        key_added="L",
        flavor="igraph",
        n_iterations=2,
        random_state=random_state,
    )
    labels = e.obs["L"].astype(str)
    sizes = labels.value_counts()
    active = [c for c in sizes.index if sizes[c] >= int(min_cluster_size)]
    X_pca = np.asarray(e.obsm["X_pca"])

    mean_stab = float("nan")
    min_stab = float("nan")
    if len(active) >= 2:
        from scfair.pp._highly_variable_genes import (
            _nearest_cluster_map,
            _pair_bootstrap_stability,
        )

        masks = {c: (labels == c).to_numpy() for c in active}
        try:
            nn = _nearest_cluster_map(X_pca, masks)
        except Exception:
            nn = {}
        scores: list[float] = []
        seen: set[tuple[str, str]] = set()
        for c, nbr in nn.items():
            key = (c, nbr) if c < nbr else (nbr, c)
            if key in seen:
                continue
            seen.add(key)
            try:
                scores.append(
                    _pair_bootstrap_stability(
                        X_pca,
                        masks[c],
                        masks[nbr],
                        n_boot=n_boot,
                        random_state=random_state,
                    )
                )
            except Exception:
                continue
        if scores:
            mean_stab = float(np.mean(scores))
            min_stab = float(np.min(scores))

    X3 = _embedding_3d(e, n_components=3, random_state=random_state, neighbors_key=None)
    if X3 is None:
        return {
            "n_obs": int(n_obs),
            "n_leiden": int(labels.nunique()),
            "n_density_pops": 0,
            "valley_median": float("nan"),
            "valley_mean": float("nan"),
            "frac_shallow": float("nan"),
            "mean_stability": mean_stab,
            "min_stability": min_stab,
            "stability_n_boot": n_boot,
            "bandwidth": None,
            "reason": "no_embedding",
        }
    bw = default_bandwidth(X3.shape[0])
    nbrs = knn_graph(X3, min(15, X3.shape[0] - 1))
    rho = knn_density(X3, bw)
    n = rho.size
    order = np.argsort(-rho, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    parent = np.full(n, -1, dtype=np.int64)
    valleys: list[float] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in order:
        higher = nbrs[i][rank[nbrs[i]] < rank[i]]
        if higher.size == 0:
            parent[i] = i
            continue
        ri = find(int(higher[np.argmin(rank[higher])]))
        parent[i] = ri
        for j in higher:
            rj = find(int(j))
            if rj == ri:
                continue
            lo, hi = (rj, ri) if rho[rj] < rho[ri] else (ri, rj)
            v = float((rho[lo] - rho[i]) / max(rho[lo], 1e-12))
            valleys.append(max(0.0, min(1.0, v)))
            if v < DEFAULT_DEPTH:
                parent[lo] = hi
            ri = find(i)

    arr = np.asarray(valleys, dtype=float) if valleys else np.array([np.nan])
    est = population_count_from_embedding(X3, depth=DEFAULT_DEPTH, bandwidth=bw)
    return {
        "n_obs": int(n_obs),
        "n_leiden": int(labels.nunique()),
        "n_density_pops": int(est.n_populations or 0),
        "valley_median": float(np.nanmedian(arr)),
        "valley_mean": float(np.nanmean(arr)),
        "frac_shallow": (
            float(np.mean(arr < DEFAULT_DEPTH)) if np.isfinite(arr).any() else float("nan")
        ),
        "mean_stability": mean_stab,
        "min_stability": min_stab,
        "stability_n_boot": n_boot,
        "bandwidth": int(bw) if bw is not None else None,
        "reason": "ok",
    }


def estimate_structure_features(
    adata: Any,
    *,
    counts_layer: str = "counts",
    random_state: int = 0,
    intermediate_resolution: float = 0.5,
    n_hvg: int = 2000,
    min_cluster_size: int = 30,
    stability_n_boot: int = STRUCTURE_STABILITY_N_BOOT,
) -> dict[str, Any]:
    """Label-free structure features for :func:`select_n_top_from_structure`.

    Builds an HVG@``n_hvg`` intermediate graph, Leiden clusters, bootstrap
    pair stability, and 3D density valley statistics (the same family used
    by ``resolution="auto"``).

    For multi-seed estimation prefer :func:`estimate_n_top_structure`.

    ``stability_n_boot`` (default 5) is lighter than cap-merge's 15 boots;
    pass 15 only if you need merge-grade precision on stability.

    Returns a dict with at least:
    ``n_obs``, ``n_density_pops``, ``valley_median``, ``frac_shallow``,
    ``mean_stability``, ``min_stability``, ``n_leiden``.
    """
    e, fail = _prepare_structure_embedding(
        adata,
        counts_layer=counts_layer,
        random_state=random_state,
        n_hvg=n_hvg,
    )
    if e is None:
        return fail
    return _structure_features_from_embedding(
        e,
        n_obs=int(adata.n_obs),
        random_state=random_state,
        intermediate_resolution=intermediate_resolution,
        min_cluster_size=min_cluster_size,
        stability_n_boot=stability_n_boot,
    )


def _aggregate_structure_features(feats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median-aggregate structure features across seeds (label-free)."""
    if not feats:
        return {}
    if len(feats) == 1:
        return dict(feats[0])

    def _nanmed(key: str) -> float:
        vals = [float(f[key]) for f in feats if key in f and np.isfinite(float(f.get(key, np.nan)))]
        return float(np.median(vals)) if vals else float("nan")

    def _nanmed_int(key: str) -> int:
        vals = [float(f[key]) for f in feats if key in f and np.isfinite(float(f.get(key, np.nan)))]
        return int(round(float(np.median(vals)))) if vals else 0

    out: dict[str, Any] = {
        "n_obs": int(feats[0].get("n_obs") or 0),
        "n_leiden": _nanmed_int("n_leiden"),
        "n_density_pops": _nanmed_int("n_density_pops"),
        "valley_median": _nanmed("valley_median"),
        "valley_mean": _nanmed("valley_mean"),
        "frac_shallow": _nanmed("frac_shallow"),
        "mean_stability": _nanmed("mean_stability"),
        "min_stability": _nanmed("min_stability"),
        "bandwidth": feats[0].get("bandwidth"),
        "reason": "ok_aggregated",
        "n_seeds_aggregated": len(feats),
    }
    return out


def _vote_structure_k(preds: Sequence[int]) -> int:
    """Mode of per-seed k; ties → median of all preds (integer).

    Diagnostic / optional consensus only. Final k is **not** taken from
    this alone — thin-margin guards (density surplus) can break under a
    mode vote when most seeds barely miss the guard.
    """
    if not preds:
        return 2000
    arr = [int(p) for p in preds]
    if len(arr) == 1:
        return arr[0]
    # mode
    vals, counts = np.unique(arr, return_counts=True)
    max_c = int(counts.max())
    modes = vals[counts == max_c]
    if modes.size == 1:
        return int(modes[0])
    return int(round(float(np.median(arr))))


def _combine_structure_k(
    *,
    k_from_agg: int,
    k_vote: int,
    per_seed_k: Sequence[int],
    n_obs: int | None,
) -> tuple[int, str]:
    """Choose final structure k after multi-seed extraction.

    **Primary:** rule on **median-aggregated** features (``k_from_agg``).
    That keeps density-surplus as ``median(n_leiden)/median(n_density)``
    rather than a majority of fragile per-seed branch decisions.

    **Unanimous vote:** if every seed agrees, trust that k (strong
    consensus, helps when jitter pushes all seeds to the same short
    verdict).

    **Anti-short veto (large n):** if the vote is hard-short (≤500) but the
    aggregated rule is mid/long (≥1500) on large data, keep ``k_from_agg`` —
    a large multi-core atlas should not fall to a short list just because a
    minority of seeds individually tipped short.
    """
    k_agg = int(k_from_agg)
    k_v = int(k_vote)
    preds = [int(p) for p in per_seed_k]
    n_cells = int(n_obs) if n_obs is not None else 0

    # Before unanimous: large-n short vote must not override mid/long aggregate
    # (seurat: all seeds may SHORT while median features still density-surplus).
    if n_cells >= 10_000 and k_v <= 500 and k_agg >= 1500:
        return k_agg, "anti_short_veto_large_n"

    if preds and len(set(preds)) == 1:
        return preds[0], "unanimous_seed_vote"

    return k_agg, "aggregated_features"


def estimate_n_top_structure(
    adata: Any,
    *,
    counts_layer: str = "counts",
    random_state: int = 0,
    version: str = "v7",
    k_min: int = 500,
    k_max: int = 4000,
    n_genes: int | None = None,
    n_seeds: int = 1,
    progress: bool = False,
    **feature_kwargs: Any,
) -> tuple[int, dict[str, Any]]:
    """End-to-end structure-aware ``n_top``: features → rule v7 (default).

    The product path (``highly_variable_genes(..., n_top_genes="auto")``)
    passes ``n_seeds=``:data:`PRODUCT_STRUCTURE_N_SEEDS` (3) so pred_k is
    multi-seed stable. This function's own default remains ``n_seeds=1``
    for fast one-shot / probe calls only — do not rely on the default for
    shipped auto behaviour.

    When ``n_seeds>1``, each seed rebuilds its own HVG@2000 + PCA +
    neighbors graph from scratch, rather than sharing one graph across
    seeds and only reseeding Leiden/density: sharing narrows the sampled
    feature variance enough to flip borderline threshold decisions,
    reintroducing the pred_k instability multi-seeding is meant to
    prevent. :func:`_prepare_structure_embedding` /
    :func:`_structure_features_from_embedding` remain available
    separately for callers who explicitly want the cheaper,
    narrower-variance version. Pair-stability uses
    ``stability_n_boot=5`` by default. Final k via
    :func:`_combine_structure_k`.

    Returns
    -------
    k, detail
        Chosen shortlist size and feature / rule diagnostics.
    """
    n_seeds = max(1, int(n_seeds))
    n_genes_eff = int(n_genes) if n_genes is not None else int(getattr(adata, "n_vars", 50_000))
    intermediate_resolution = float(feature_kwargs.pop("intermediate_resolution", 0.5))
    n_hvg = int(feature_kwargs.pop("n_hvg", 2000))
    min_cluster_size = int(feature_kwargs.pop("min_cluster_size", 30))
    stability_n_boot = int(feature_kwargs.pop("stability_n_boot", STRUCTURE_STABILITY_N_BOOT))
    if feature_kwargs:
        logger.debug("estimate_n_top_structure ignoring kwargs %s", feature_kwargs)

    # Each seed rebuilds its own HVG@2000 + PCA + neighbors graph.
    # (The shared-graph variant is available via _prepare_structure_embedding
    # + _structure_features_from_embedding directly, for callers who explicitly
    # want the cheaper, narrower-variance measurement -- see docstring.)
    if n_seeds > 1:
        _progress(
            progress,
            "structure auto_n: extracting features from %d seed(s) -- each "
            "rebuilds its own PCA + neighbour graph, so this can take a few "
            "minutes on larger datasets. Please wait...",
            n_seeds,
        )
    per_seed_feats: list[dict[str, Any]] = []
    per_seed_k: list[int] = []
    for i in range(n_seeds):
        seed_i = int(random_state) + i
        feat_i = estimate_structure_features(
            adata,
            counts_layer=counts_layer,
            random_state=seed_i,
            intermediate_resolution=intermediate_resolution,
            n_hvg=n_hvg,
            min_cluster_size=min_cluster_size,
            stability_n_boot=stability_n_boot,
        )
        if n_seeds > 1:
            _progress(
                progress,
                "structure auto_n: seed %d/%d done (%d%%)...",
                i + 1,
                n_seeds,
                round((i + 1) / n_seeds * 100),
            )
        k_i = select_n_top_from_structure(
            valley_median=feat_i.get("valley_median", float("nan")),
            frac_shallow=feat_i.get("frac_shallow", float("nan")),
            n_density_pops=int(feat_i.get("n_density_pops") or 0),
            mean_stability=feat_i.get("mean_stability"),
            min_stability=feat_i.get("min_stability"),
            n_obs=feat_i.get("n_obs"),
            n_leiden=feat_i.get("n_leiden"),
            version=version,
            k_min=k_min,
            k_max=k_max,
            n_genes=n_genes_eff,
        )
        per_seed_feats.append(feat_i)
        per_seed_k.append(int(k_i))

    feat = _aggregate_structure_features(per_seed_feats)
    k_vote = _vote_structure_k(per_seed_k)
    k_from_agg = select_n_top_from_structure(
        valley_median=feat.get("valley_median", float("nan")),
        frac_shallow=feat.get("frac_shallow", float("nan")),
        n_density_pops=int(feat.get("n_density_pops") or 0),
        mean_stability=feat.get("mean_stability"),
        min_stability=feat.get("min_stability"),
        n_obs=feat.get("n_obs"),
        n_leiden=feat.get("n_leiden"),
        version=version,
        k_min=k_min,
        k_max=k_max,
        n_genes=n_genes_eff,
    )
    k, k_source = _combine_structure_k(
        k_from_agg=int(k_from_agg),
        k_vote=int(k_vote),
        per_seed_k=per_seed_k,
        n_obs=feat.get("n_obs"),
    )
    detail = {
        "strategy": "structure",
        "version": version,
        "n_top_selected": int(k),
        "features": feat,
        "n_seeds": n_seeds,
        "per_seed_k": list(per_seed_k),
        "k_from_aggregated_features": int(k_from_agg),
        "k_vote": int(k_vote),
        "k_source": k_source,
        "shared_embedding": False,
        "stability_n_boot": int(stability_n_boot),
    }
    return int(k), detail


def select_n_top_ensemble(
    picks: dict[str, int],
    *,
    k_min: int = 500,
    k_max: int = 5000,
    n_genes: int,
    k_floor: int | None = None,
    k_ceiling: int | None = None,
    shape_mass_ratio: float = 0.5,
    anchor: int | None = 2000,
) -> int:
    """Robust ensemble v2.1 for auto ``n_top_genes``.

    Changes vs v1 (plain median of all picks):

    1. **Single shape vote** — ``max(elbow, knee)`` counts once.
    2. **Protective floor** — final k ≥ ``k_floor``.
    3. **Untrustworthy shape** — if ``shape < shape_mass_ratio * cumfrac``,
       raise shape to ``k_floor``.
    4. **Coverage is a soft floor, not a high vote**.
    5. **Anchor** (default 2000) stabilizes median near classical HVG size.
    6. **Ceiling** (default 2500) + **clipped cumfrac vote** — keeps very
       large cumfrac picks from dominating the median.
    7. Silhouette is never included here.

    Generic keys (not elbow/knee/cumfrac/coverage) are still included once.
    """
    detail = select_n_top_ensemble_detail(
        picks,
        k_min=k_min,
        k_max=k_max,
        n_genes=n_genes,
        k_floor=k_floor,
        k_ceiling=k_ceiling,
        shape_mass_ratio=shape_mass_ratio,
        anchor=anchor,
    )
    return int(detail["k"])


def select_n_top_ensemble_detail(
    picks: dict[str, int],
    *,
    k_min: int = 500,
    k_max: int = 5000,
    n_genes: int,
    k_floor: int | None = None,
    k_ceiling: int | None = None,
    shape_mass_ratio: float = 0.5,
    anchor: int | None = 2000,
) -> dict[str, Any]:
    """Same as :func:`select_n_top_ensemble` but returns vote diagnostics."""
    if k_floor is None:
        k_floor = effective_k_floor(n_genes, k_min=k_min, k_max=k_max)
    else:
        k_floor = _clip_k(int(k_floor), k_min, k_max, n_genes)

    if k_ceiling is None:
        k_ceiling = effective_k_ceiling(n_genes, k_min=k_min, k_max=k_max)
    else:
        k_ceiling = min(int(k_ceiling), int(k_max), max(int(n_genes), 1))
    if k_ceiling < k_floor:
        k_ceiling = k_floor

    raw = {str(m): int(v) for m, v in picks.items() if v is not None}
    votes: dict[str, int] = {}
    notes: list[str] = []

    # --- shape: one vote from elbow / knee ---
    shape_raw = None
    shape_parts = [raw[k] for k in ("elbow", "knee") if k in raw]
    if shape_parts:
        shape_raw = int(max(shape_parts))
        shape = shape_raw
        mass_ref = raw.get("cumfrac")
        if (
            mass_ref is not None
            and shape_mass_ratio > 0
            and shape < float(shape_mass_ratio) * float(mass_ref)
        ):
            shape = max(shape, k_floor)
            notes.append(
                f"shape_raised:{shape_raw}->{shape}(<{shape_mass_ratio}*cumfrac={mass_ref})"
            )
        else:
            shape = max(shape, k_floor) if shape < k_floor else shape
            if shape != shape_raw:
                notes.append(f"shape_floor:{shape_raw}->{shape}")
        # shape also should not exceed ceiling as a vote
        if shape > k_ceiling:
            notes.append(f"shape_ceiling:{shape}->{k_ceiling}")
            shape = k_ceiling
        votes["shape"] = int(shape)

    # --- mass: clip cumfrac vote to ceiling (do not let 3.5k+ dominate) ---
    if "cumfrac" in raw:
        cf_raw = int(raw["cumfrac"])
        cf = min(cf_raw, k_ceiling)
        if cf != cf_raw:
            notes.append(f"cumfrac_clipped:{cf_raw}->{cf}")
        votes["cumfrac"] = int(cf)

    # --- classical / depth-aware size anchor (always, when provided) ---
    # Previously only injected when strictly between shape and mass; after
    # cumfrac is clipped to ceiling that often equals anchor, the guard
    # skipped injection and median(shape_floor, ceiling) collapsed to
    # (1000+2000)/2=1500. Always vote the anchor for stability.
    if anchor is not None and ("shape" in votes or "cumfrac" in votes):
        a = _clip_k(int(anchor), k_min, k_max, n_genes)
        a = min(max(a, k_floor), k_ceiling)
        votes["anchor"] = a
        notes.append(f"anchor:{a}")

    # --- any other custom keys (exclude already-handled) ---
    skip = {"elbow", "knee", "cumfrac", "coverage", "silhouette"}
    for key, val in raw.items():
        if key not in skip and key not in votes:
            votes[key] = int(min(int(val), k_ceiling))

    if not votes:
        k = _clip_k(max(int(anchor or 2000), k_floor), k_min, k_max, n_genes)
        k = min(k, k_ceiling)
        return {
            "k": k,
            "k_floor": k_floor,
            "k_ceiling": k_ceiling,
            "votes": {},
            "raw_picks": raw,
            "notes": notes + ["fallback_2000"],
            "version": "ensemble_v2.2",
        }

    vals = list(votes.values())
    k_med = int(np.median(np.asarray(vals, dtype=float)))
    k = max(k_med, k_floor)

    # coverage: soft *floor* only when *raw* coverage is modest (≤1.25×k).
    # Clipping a huge coverage (4k→ceiling) must not force k up to ceiling.
    if "coverage" in raw:
        cover_raw = int(raw["coverage"])
        soft_cap = int(1.25 * max(k, 1))
        if cover_raw > k and cover_raw <= soft_cap:
            k_new = min(cover_raw, k_ceiling)
            notes.append(f"coverage_soft_raise:{k}->{k_new}")
            k = k_new
        else:
            notes.append(
                f"coverage_ignored_as_vote:{cover_raw}(soft_cap={soft_cap}, k_med={k_med})"
            )

    k = min(k, k_ceiling)
    k = _clip_k(k, k_min, k_max, n_genes)
    if k != k_med:
        notes.append(f"final_adjust:{k_med}->{k}")

    return {
        "k": k,
        "k_floor": k_floor,
        "k_ceiling": k_ceiling,
        "votes": votes,
        "raw_picks": raw,
        "notes": notes,
        "version": "ensemble_v2.2",
    }


def resolve_n_top_genes(
    n_top_genes: int | str | None,
    scores_desc: np.ndarray,
    gene_order: Sequence[str],
    *,
    method: str | None = None,
    k_min: int = 500,
    k_max: int = 5000,
    cumfrac: float = 0.8,
    adata: Any | None = None,
    counts_layer: str = "counts",
    cluster_gene_ranks: dict[str, Sequence[str]] | None = None,
    min_per_cluster: int = 20,
    silhouette_candidates: Sequence[int] | None = None,
    random_state: int = 0,
    shape_mass_ratio: float = 0.5,
    soft_floor: int = 1000,
    soft_ceiling: int = 2500,
    anchor: int | None = 2000,
    depth_aware: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Resolve ``n_top_genes`` int or auto strategy name into an integer k.

    Parameters
    ----------
    n_top_genes
        int, or one of ``auto`` / ``elbow`` / ``knee`` / ``cumfrac`` /
        ``silhouette`` / ``coverage`` / ``ensemble`` / ``structure``.
    method
        Override strategy when ``n_top_genes='auto'`` (default
        ``structure`` via HVG). ``"structure"`` needs ``adata``.
    shape_mass_ratio, soft_floor, soft_ceiling, anchor
        Ensemble knobs (see :func:`select_n_top_ensemble_detail`). When
        ``depth_aware`` and ``adata`` are set, floor/ceiling/anchor are
        overridden from library depth (see :func:`depth_aware_auto_knobs`).
    depth_aware
        If True (default) and ``adata`` is provided for ensemble, set
        anchor/ceiling/floor from median counts and genes per cell.
    """
    n_genes = len(gene_order)
    meta: dict[str, Any] = {"k_min": k_min, "k_max": k_max}

    if isinstance(n_top_genes, (int, np.integer)):
        k = _clip_k(int(n_top_genes), 1, k_max, n_genes)
        meta.update({"strategy": "fixed", "n_top_selected": k})
        return k, meta

    if n_top_genes is None:
        n_top_genes = "auto"

    name = str(n_top_genes).lower()
    if name == "auto":
        name = (method or "ensemble").lower()

    picks: dict[str, int] = {}
    picks["elbow"] = select_n_top_elbow(scores_desc, k_min=k_min, k_max=k_max)
    picks["knee"] = select_n_top_knee(scores_desc, k_min=k_min, k_max=k_max)
    picks["cumfrac"] = select_n_top_cumfrac(scores_desc, frac=cumfrac, k_min=k_min, k_max=k_max)

    if name == "coverage" or name == "ensemble":
        if cluster_gene_ranks:
            picks["coverage"] = select_n_top_coverage(
                gene_order,
                cluster_gene_ranks,
                min_per_cluster=min_per_cluster,
                k_min=k_min,
                k_max=k_max,
            )
        elif name == "coverage":
            logger.warning("coverage requested but no cluster ranks; falling back to knee.")
            name = "knee"

    sil_curve: dict[int, float] | None = None
    structure_detail: dict[str, Any] | None = None
    if name == "silhouette":
        if adata is None:
            logger.warning("silhouette needs adata; falling back to knee.")
            name = "knee"
        else:
            k_sil, sil_curve = select_n_top_silhouette(
                adata,
                gene_order,
                candidates=silhouette_candidates,
                k_min=k_min,
                k_max=k_max,
                counts_layer=counts_layer,
                random_state=random_state,
            )
            picks["silhouette"] = k_sil

    if name == "structure":
        if adata is None:
            logger.warning("structure auto_n needs adata; falling back to ensemble.")
            name = "ensemble"
        else:
            try:
                k_struct, structure_detail = estimate_n_top_structure(
                    adata,
                    counts_layer=counts_layer,
                    random_state=random_state,
                    version="v7",
                    k_min=k_min,
                    k_max=min(k_max, 4000),
                    n_genes=n_genes,
                    # Product stability: not the library default n_seeds=1.
                    n_seeds=PRODUCT_STRUCTURE_N_SEEDS,
                )
                picks["structure"] = int(k_struct)
                k = int(k_struct)
                meta.update(
                    {
                        "strategy": "structure",
                        "n_top_selected": k,
                        "method_picks": picks,
                        "structure": structure_detail,
                        "depth": None,
                        "silhouette_curve": None,
                        "cumfrac": None,
                        "ensemble": None,
                    }
                )
                logger.info(
                    "Auto n_top_genes=%d (strategy=structure, features=%s)",
                    k,
                    (structure_detail or {}).get("features"),
                )
                return k, meta
            except Exception as exc:
                logger.warning("structure auto_n failed (%s); falling back to ensemble.", exc)
                structure_detail = {"error": str(exc)}
                name = "ensemble"

    ensemble_detail: dict[str, Any] | None = None
    depth_meta: dict[str, Any] | None = None
    if name == "ensemble":
        # Ensemble v2.2: v2.1 + optional depth-aware anchor/ceiling/floor.
        # Silhouette never included (slow; tends to pick very small k and
        # can lose rare, low-abundance clusters).
        use = {m: picks[m] for m in ("elbow", "knee", "cumfrac") if m in picks}
        if "coverage" in picks:
            use["coverage"] = picks["coverage"]

        ens_floor = soft_floor
        ens_ceiling = soft_ceiling
        ens_anchor = anchor
        if depth_aware and adata is not None:
            try:
                stats = library_depth_stats(adata, counts_layer=counts_layer)
                knobs = depth_aware_auto_knobs(
                    median_counts=stats["median_counts"],
                    median_genes=stats["median_genes"],
                    n_genes=n_genes,
                    k_min=k_min,
                    k_max=k_max,
                )
                ens_floor = int(knobs["soft_floor"])
                ens_ceiling = int(knobs["soft_ceiling"])
                ens_anchor = int(knobs["anchor"])
                depth_meta = {**stats, **knobs}
            except Exception as exc:
                logger.warning("depth-aware auto knobs failed (%s); using defaults.", exc)
                depth_meta = {"error": str(exc)}

        k_floor = effective_k_floor(n_genes, k_min=k_min, k_max=k_max, soft_floor=ens_floor)
        k_ceiling = effective_k_ceiling(n_genes, k_min=k_min, k_max=k_max, soft_ceiling=ens_ceiling)
        ensemble_detail = select_n_top_ensemble_detail(
            use,
            k_min=k_min,
            k_max=k_max,
            n_genes=n_genes,
            k_floor=k_floor,
            k_ceiling=k_ceiling,
            shape_mass_ratio=shape_mass_ratio,
            anchor=ens_anchor,
        )
        k = int(ensemble_detail["k"])
    elif name in picks:
        k = picks[name]
    else:
        raise ValueError(
            f"Unknown n_top_genes={n_top_genes!r}. Use int or one of "
            f"auto/elbow/knee/cumfrac/silhouette/coverage/ensemble/structure."
        )

    meta.update(
        {
            "strategy": name,
            "n_top_selected": k,
            "method_picks": picks,
            "depth": depth_meta,
            "silhouette_curve": sil_curve,
            "cumfrac": cumfrac if name in ("cumfrac", "ensemble", "auto") else None,
            "ensemble": ensemble_detail,
            "structure": structure_detail,
        }
    )
    logger.info(
        "Auto n_top_genes=%d (strategy=%s, picks=%s, ensemble=%s)",
        k,
        name,
        picks,
        ensemble_detail,
    )
    return k, meta
