"""Estimate how many populations a 3D embedding's density field supports.

Advisory. Nothing here changes gene selection unless a caller uses the count.

Method:

1. 3D UMAP from an existing kNN graph (``sc.pp.neighbors``).
2. Adaptive kNN density: ``rho_i ~ k / r_k(i)**d``, with ``r_k`` the distance
   to the k-th neighbor.
3. ToMATo peak merge: two peaks are one population when the pass between
   them does not drop ``depth`` below the shorter peak. Default
   ``depth=0.5`` means the valley must fall below half the shorter core.

d=3 is a compromise: a 2D density field has a planar adjacency graph and
tends to undercount as the true number of populations grows; density
estimation in 50-PC space is too sparse for this geometry. Bandwidth is a
fraction of ``n_obs`` (default 2%, floor 30 cells) so it transfers across
dataset sizes. Merge is required because sub-states form their own small
cores; without it the field shatters.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# `depth` is the interpretable knob; the bandwidth is the calibrated constant.
DEFAULT_DEPTH = 0.5
DEFAULT_BANDWIDTH_FRAC = 0.02  # k-th neighbor, as a fraction of n_obs
MIN_BANDWIDTH = 30  # floor, so small inputs still get a density
DEFAULT_K_GRAPH = 15
DEFAULT_N_COMPONENTS = 3


@dataclass
class GranularityEstimate:
    """How many populations the embedding's density field resolves.

    Attributes
    ----------
    n_populations
        The count, or ``None`` when the estimate could not run at all.
    reason
        ``ok`` | ``too_few_cells`` | ``no_embedding``. ``ok`` only means the
        estimator produced a count — not that the count is reliable. Check
        ``confidence`` / ``depth_sensitivity``.
    labels
        Cell -> population; ``None`` when there is no count.
    bandwidth
        The kNN bandwidth actually used, in cells.
    depth
        The merge threshold actually used.
    confidence
        ``high`` | ``moderate`` | ``low`` | ``none``. Depth-perturbation
        stability of the count; ``none`` when no estimate ran.
    depth_sensitivity
        ``n_populations`` at a looser depth minus the count at a stricter
        depth. Large values mean the count moves when the merge threshold
        moves — treat ``n_populations`` as uncertain.
    n_populations_loose, n_populations_strict
        Counts at depth×0.5 and min(0.95, depth×1.5) for the margin.
    """

    n_populations: int | None
    reason: str
    labels: np.ndarray | None = None
    bandwidth: int | None = None
    depth: float = DEFAULT_DEPTH
    confidence: str = "none"
    depth_sensitivity: int | None = None
    n_populations_loose: int | None = None
    n_populations_strict: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics-shaped view, for ``uns``. Omits the per-cell labels."""
        out: dict[str, Any] = {
            "n_populations": self.n_populations,
            "reason": self.reason,
            "bandwidth": self.bandwidth,
            "depth": float(self.depth),
            "confidence": self.confidence,
            "depth_sensitivity": self.depth_sensitivity,
            "n_populations_loose": self.n_populations_loose,
            "n_populations_strict": self.n_populations_strict,
        }
        if self.labels is not None:
            # Sizes of the density-field populations -- not of the Leiden
            # partition, which is a different object. Recorded because whether
            # `n_top_genes` should scale with population size is an open
            # measurable question.
            _, counts = np.unique(self.labels, return_counts=True)
            sizes = sorted((int(c) for c in counts), reverse=True)
            out["population_sizes"] = sizes
            if len(sizes) >= 2 and sizes[-1] > 0:
                out["size_max_min_ratio"] = float(sizes[0] / sizes[-1])
            elif len(sizes) == 1:
                out["size_max_min_ratio"] = 1.0
        return out


# ---------------------------------------------------------------------------
# density and peak merging
# ---------------------------------------------------------------------------
# Floor on k-th neighbor distance. ``np.finfo(float).tiny ** 3`` underflows
# to 0 → rho=inf → rho/rho.max() = NaN on duplicate / near-duplicate points.
_KNN_DIST_FLOOR = 1e-12


def knn_density(X: np.ndarray, k: int) -> np.ndarray:
    """Adaptive kNN density on an embedding, normalized to max 1.

    ``rho_i ~ k / r_k(i)**d`` with ``r_k`` the distance to the k-th neighbor:
    the smoothing length widens in sparse regions and narrows in dense ones,
    and ``k`` is in units of cells rather than a fraction of the layout's
    extent. See the module docstring for why a voxel grid was dropped.

    Duplicate coordinates (common in integer embeddings / collapsed UMAP) are
    handled by flooring ``r_k`` so density stays finite; callers should treat a
    near-flat field as low confidence.
    """
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    r = np.maximum(dist[:, k].astype(float, copy=False), _KNN_DIST_FLOOR)
    dim = max(int(X.shape[1]), 1)
    # Avoid overflow to inf for very small r and high d before the floor bites.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rho = 1.0 / np.power(r, float(dim))
    rho = np.asarray(rho, dtype=float)
    finite = np.isfinite(rho) & (rho > 0)
    if not np.any(finite):
        # Fully degenerate (should be rare after the distance floor).
        return np.ones(X.shape[0], dtype=float)
    rho = np.where(finite, rho, 0.0)
    m = float(rho.max())
    if m <= 0.0 or not np.isfinite(m):
        return np.ones(X.shape[0], dtype=float)
    return rho / m


def knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def merge_peaks(rho: np.ndarray, nbrs: np.ndarray, depth: float) -> np.ndarray:
    """ToMATo: assign every cell to a density peak, merging shallow passes.

    Cells are visited in descending density. A cell with no denser neighbor
    starts a peak; otherwise it joins its densest neighbor's peak. When a cell
    is adjacent to two peaks it *is* the pass between them, so the valley depth
    is known at that moment: merge when ``(rho_peak_lo - rho_here) / rho_peak_lo
    < depth``, i.e. when the pass never falls ``depth`` below the shorter peak.
    """
    n = rho.size
    order = np.argsort(-rho, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    parent = np.full(n, -1, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for i in order:
        nb = nbrs[i]
        higher = nb[rank[nb] < rank[i]]
        if higher.size == 0:
            parent[i] = i  # local maximum: a candidate population
            continue
        ri = find(higher[np.argmin(rank[higher])])
        parent[i] = ri
        for j in higher:
            rj = find(j)
            if rj == ri:
                continue
            lo, hi = (rj, ri) if rho[rj] < rho[ri] else (ri, rj)
            if (rho[lo] - rho[i]) / rho[lo] < depth:
                parent[lo] = hi  # pass too shallow: one population
            ri = find(i)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def default_bandwidth(n_obs: int) -> int:
    """The calibrated bandwidth for a dataset of this size (see module docs)."""
    return int(max(MIN_BANDWIDTH, round(DEFAULT_BANDWIDTH_FRAC * int(n_obs))))


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
def population_count_from_embedding(
    embedding: np.ndarray,
    *,
    depth: float = DEFAULT_DEPTH,
    bandwidth: int | None = None,
    k_graph: int = DEFAULT_K_GRAPH,
) -> GranularityEstimate:
    """How many populations the embedding's density field resolves.

    Parameters
    ----------
    embedding
        ``(n_obs, d)`` array, ``d=3`` intended. 2D is accepted but undercounts
        by construction (see the planarity note in the module docstring).
    depth
        Merge threshold. The pass between two peaks must fall this far below the
        shorter peak, relative to it, for them to count as two populations.
    bandwidth
        kNN bandwidth in cells. ``None`` uses :func:`default_bandwidth`, i.e.
        2% of ``n_obs`` — the calibrated constant.
    k_graph
        Neighbours used for the merge graph.

    Returns
    -------
    GranularityEstimate
    """
    X = np.asarray(embedding, dtype=float)
    k = int(bandwidth) if bandwidth is not None else default_bandwidth(X.shape[0])
    if X.ndim != 2 or X.shape[0] < k + 2:
        return GranularityEstimate(None, "too_few_cells", depth=depth, bandwidth=k)

    nbrs = knn_graph(X, k_graph)
    rho = knn_density(X, k)
    # Near-flat density (duplicates / collapsed embedding): merge_peaks with
    # NaNs used to never merge and inflate n_pop while still reporting high
    # confidence. Force low confidence when the field has no usable contrast.
    rho_finite = bool(np.all(np.isfinite(rho)))
    rho_span = float(np.nanmax(rho) - np.nanmin(rho)) if rho.size else 0.0
    density_degenerate = (not rho_finite) or rho_span < 1e-12
    if not rho_finite:
        rho = np.nan_to_num(np.asarray(rho, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        m = float(rho.max()) if rho.size else 0.0
        rho = (rho / m) if m > 0 else np.ones(X.shape[0], dtype=float)

    labels = merge_peaks(rho, nbrs, depth)
    n_pop = int(pd.unique(labels).size)

    # Depth-perturbation margin: the same field at looser / stricter merges.
    # reason="ok" alone is not a reliability claim — if the count swings with
    # depth, the density field is not resolving stable valleys.
    depth_lo = max(0.05, float(depth) * 0.5)
    depth_hi = min(0.95, float(depth) * 1.5)
    n_loose = int(pd.unique(merge_peaks(rho, nbrs, depth_lo)).size)
    n_strict = int(pd.unique(merge_peaks(rho, nbrs, depth_hi)).size)
    sensitivity = int(n_loose - n_strict)
    if density_degenerate:
        confidence = "low"
    elif sensitivity <= 0 and n_pop >= 2:
        confidence = "high"
    elif sensitivity <= 2:
        confidence = "moderate"
    else:
        confidence = "low"
    # Single blob is always low-confidence as a multi-type claim.
    if n_pop < 2:
        confidence = "low"

    return GranularityEstimate(
        n_populations=n_pop,
        reason="degenerate_density" if density_degenerate else "ok",
        labels=labels,
        bandwidth=k,
        depth=depth,
        confidence=confidence,
        depth_sensitivity=sensitivity,
        n_populations_loose=n_loose,
        n_populations_strict=n_strict,
    )


def _embedding_3d(
    adata: Any,
    *,
    n_components: int,
    random_state: int,
    neighbors_key: str | None,
) -> np.ndarray | None:
    """A d-dimensional UMAP, without clobbering the user's 2D one.

    ``sc.tl.umap`` writes ``obsm['X_umap']`` and ``uns['umap']`` in place, and
    the 2D layout sitting there is usually the one the user plots. So both are
    saved and restored, including the "was absent" case, and the restore runs
    even if UMAP raises.
    """
    import scanpy as sc

    key = neighbors_key or "neighbors"
    if key not in adata.uns and "connectivities" not in adata.obsp:
        warnings.warn(
            "estimate_n_populations needs a neighbor graph; run "
            "sc.pp.neighbors(adata) first (or pass embedding=...). "
            "Returning n_populations=None.",
            UserWarning,
            stacklevel=3,
        )
        return None

    had_umap = "X_umap" in adata.obsm
    saved_obsm = adata.obsm["X_umap"].copy() if had_umap else None
    had_uns = "umap" in adata.uns
    saved_uns = adata.uns["umap"] if had_uns else None
    try:
        sc.tl.umap(
            adata,
            n_components=n_components,
            random_state=random_state,
            neighbors_key=neighbors_key,
        )
        return np.asarray(adata.obsm["X_umap"], dtype=float)
    finally:
        if had_umap:
            adata.obsm["X_umap"] = saved_obsm
        else:
            adata.obsm.pop("X_umap", None)
        if had_uns:
            adata.uns["umap"] = saved_uns
        else:
            adata.uns.pop("umap", None)


def estimate_n_populations(
    adata: Any,
    *,
    embedding: np.ndarray | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    depth: float = DEFAULT_DEPTH,
    bandwidth: int | None = None,
    k_graph: int = DEFAULT_K_GRAPH,
    random_state: int = 0,
    neighbors_key: str | None = None,
    key_added: str = "granularity",
    copy_labels_to_obs: str | None = None,
) -> GranularityEstimate:
    """Report how many populations this dataset's density field supports.

    **This is the label-free alternative to sweeping Leiden resolution** for
    the question *how many populations are there?* — not which cell belongs
    where. On well-separated synthetic blobs it recovers the true count with
    high confidence up to roughly **~20 populations**; with many tiny groups
    (e.g. 30 types × ~80 cells) it can under-count. Always run
    ``sc.pp.neighbors`` first (or pass ``embedding=``).

    **Advisory.** Nothing here changes gene selection, and nothing in
    :func:`~scfair.pp.highly_variable_genes` consults it. The count, bandwidth
    and merge threshold are written to ``adata.uns['scfair'][key_added]``.

    Parameters
    ----------
    adata
        AnnData. Without ``embedding=``, **requires** a neighbor graph from
        ``sc.pp.neighbors(adata)`` (``uns['neighbors']`` or
        ``obsp['connectivities']``). Missing graph → ``UserWarning`` and
        ``n_populations=None``. Existing ``obsm['X_umap']`` is preserved.
    embedding
        Use this ``(n_obs, d)`` array instead of computing a UMAP.
    n_components
        Embedding dimension. 3 by default; see the module docstring on why not
        2 and why not 50.
    depth
        Merge threshold — the one interpretable knob.
    bandwidth
        kNN bandwidth in cells; ``None`` uses the calibrated 2% of ``n_obs``.
    copy_labels_to_obs
        If given, write the per-cell population assignment to that ``obs``
        column. Off by default: the assignment is measurably worse than
        Leiden's, so this estimate is useful for *how many*, not for
        *which cell goes where*.

    Returns
    -------
    GranularityEstimate
    """
    X = embedding
    if X is None:
        X = _embedding_3d(
            adata,
            n_components=n_components,
            random_state=random_state,
            neighbors_key=neighbors_key,
        )
    if X is None:
        est = GranularityEstimate(None, "no_embedding", depth=depth)
    else:
        est = population_count_from_embedding(
            np.asarray(X, dtype=float),
            depth=depth,
            bandwidth=bandwidth,
            k_graph=k_graph,
        )

    store = adata.uns.setdefault("scfair", {})
    store[key_added] = {
        **est.to_dict(),
        "n_components": int(n_components),
        "k_graph": int(k_graph),
        "random_state": int(random_state),
    }
    if copy_labels_to_obs and est.labels is not None:
        adata.obs[copy_labels_to_obs] = pd.Categorical([str(v) for v in est.labels])

    if est.n_populations is not None:
        logger.info(
            "Density field resolves %d populations (bandwidth=%s cells, depth=%.2f).",
            est.n_populations,
            est.bandwidth,
            est.depth,
        )
    return est
