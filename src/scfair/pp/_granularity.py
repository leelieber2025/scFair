"""Read the population count off a 3D embedding, so nobody has to pick a resolution.

**Experimental.** Nothing here changes gene selection unless the caller asks for
it.

The intermediate clustering elsewhere in this package runs at a hardcoded
``resolution=0.5``, which corresponds to a different granularity on every
dataset since it does not account for dataset size or structure. This module
estimates population count directly from an embedding's density field, so a
Leiden resolution can be derived from a target cluster count instead of
guessed.

Why a 3D embedding, and not the PC space or the 2D plot
---------------------------------------------------------
Other automatic-granularity methods (sc-SHC, CHOIR, chooseR, MultiK) run
statistical tests in PC space rather than an embedding, because 2D embeddings
are known to distort. But 2D distorts partly *because it is made for eyes*,
and when software is the consumer the embedding dimension is free to choose.

Two dimension-specific facts pick d=3:

- **Planarity.** Segment a 2D density field into connected regions and the
  region-adjacency graph is necessarily planar: at most ``3n-6`` of the
  ``n(n-1)/2`` possible adjacencies. A 2D density field therefore
  structurally undercounts populations, and the undercount worsens as the
  true population count grows.
- **Density estimation still works.** Per-axis resolution goes as
  ``n**(1/d)``: far fewer bins are available in the 50-PC space where other
  methods operate, which is why they use classifiers and permutation tests
  instead of geometry.

The method
----------
1. 3D UMAP from the kNN graph that has already been built.
2. Adaptive kNN density: ``rho_i ~ k / r_k(i)**d``. The bandwidth is the
   distance to the k-th neighbour, so it is set by the data at each point.
3. Merge density peaks (ToMATo): two peaks are one population when the pass
   between them does not drop ``depth`` below the shorter peak. Dimensionless,
   so it needs no per-dataset tuning; ``depth=0.5`` reads as "the valley must
   fall below half the shorter core".

Cores outnumber cell types -- sub-states have their own small cores -- so step 3
is not optional. Without it the field shatters.

Why the bandwidth is a fraction of the cells
--------------------------------------------
The bandwidth is the one calibrated constant here, fixed once against data
rather than re-derived per call. A voxel-grid density field ties its
smoothing to the layout's bounding box, which is an artifact of UMAP's
arbitrary output scale rather than a property of the data; the kNN form used
here ties the bandwidth to a neighbour count instead, which is stable across
layouts. An absolute neighbour count still does not transfer across dataset
sizes, so the bandwidth is expressed as a fraction of ``n_obs``:
``DEFAULT_BANDWIDTH_FRAC = 0.02``, with a floor for small inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# `depth` is the interpretable knob; the bandwidth is the calibrated constant.
DEFAULT_DEPTH = 0.5
DEFAULT_BANDWIDTH_FRAC = 0.02  # k-th neighbour, as a fraction of n_obs
MIN_BANDWIDTH = 30  # floor, so small inputs still get a density
DEFAULT_K_GRAPH = 15
DEFAULT_N_COMPONENTS = 3
# What `resolution="auto"` falls back to when the density field yields no
# usable count. 0.5 is the value that was the default before auto existed,
# so the fallback path reproduces the historical behaviour exactly.
DEFAULT_RESOLUTION_FALLBACK = 0.5


@dataclass
class GranularityEstimate:
    """How many populations the embedding's density field resolves.

    Attributes
    ----------
    n_populations
        The count, or ``None`` when the estimate could not run at all.
    reason
        ``ok`` | ``too_few_cells`` | ``no_embedding``.
    labels
        Cell -> population; ``None`` when there is no count.
    bandwidth
        The kNN bandwidth actually used, in cells.
    depth
        The merge threshold actually used.
    """

    n_populations: int | None
    reason: str
    labels: np.ndarray | None = None
    bandwidth: int | None = None
    depth: float = DEFAULT_DEPTH

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics-shaped view, for ``uns``. Omits the per-cell labels."""
        out: dict[str, Any] = {
            "n_populations": self.n_populations,
            "reason": self.reason,
            "bandwidth": self.bandwidth,
            "depth": float(self.depth),
        }
        if self.labels is not None:
            # Sizes of the density-field populations -- not of the Leiden
            # partition, which is a different object. Recorded because whether
            # `n_top_genes` should scale with population size is an open,
            # measurable question (see _auto_n.select_n_top_from_populations).
            _, counts = np.unique(self.labels, return_counts=True)
            out["population_sizes"] = sorted((int(c) for c in counts), reverse=True)
        return out


# ---------------------------------------------------------------------------
# density and peak merging
# ---------------------------------------------------------------------------
def knn_density(X: np.ndarray, k: int) -> np.ndarray:
    """Adaptive kNN density on an embedding, normalised to max 1.

    ``rho_i ~ k / r_k(i)**d`` with ``r_k`` the distance to the k-th neighbour:
    the smoothing length widens in sparse regions and narrows in dense ones,
    and ``k`` is in units of cells rather than a fraction of the layout's
    extent. That independence from the layout's extent is the whole point —
    see the module docstring on why the voxel formulation failed.
    """
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    r = np.maximum(dist[:, k], np.finfo(float).tiny)
    rho = 1.0 / r ** X.shape[1]
    return rho / rho.max()


def knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def merge_peaks(rho: np.ndarray, nbrs: np.ndarray, depth: float) -> np.ndarray:
    """ToMATo: assign every cell to a density peak, merging shallow passes.

    Cells are visited in descending density. A cell with no denser neighbour
    starts a peak; otherwise it joins its densest neighbour's peak. When a cell
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
    labels = merge_peaks(rho, nbrs, depth)
    return GranularityEstimate(
        n_populations=int(pd.unique(labels).size),
        reason="ok",
        labels=labels,
        bandwidth=k,
        depth=depth,
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
        logger.warning(
            "estimate_n_populations needs a neighbour graph; run "
            "sc.pp.neighbors(adata) first (or pass embedding=...)."
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

    **Advisory. This does not run or change gene selection**, and nothing in
    :func:`~scfair.pp.highly_variable_genes` consults it. It answers the
    question a resolution parameter is usually a proxy for — *how many
    populations are actually there* — by reading the density field of a 3D
    embedding rather than by asking the caller to guess a number.

    The count, the bandwidth it used and the merge threshold are written to
    ``adata.uns['scfair'][key_added]``, so a partition can be reproduced from
    the record rather than from the defaults in force at the time.

    Parameters
    ----------
    adata
        AnnData with a neighbour graph (``sc.pp.neighbors``). Its existing
        ``obsm['X_umap']`` is preserved.
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


def resolution_from_density_field(
    adata: Any,
    *,
    fallback: float,
    n_components: int = DEFAULT_N_COMPONENTS,
    depth: float = DEFAULT_DEPTH,
    bandwidth: int | None = None,
    k_graph: int = DEFAULT_K_GRAPH,
    random_state: int = 0,
    leiden_key: str = "_scfair_res_probe",
) -> tuple[float, dict[str, Any]]:
    """Pick the Leiden resolution that yields the count the density field shows.

    This is what ``resolution="auto"`` does, and the ordering is the point: the
    count comes first and the resolution is derived from it, rather than a
    resolution being guessed and the count falling out of it.

    ``adata`` must already carry a neighbour graph; the same graph is reused for
    both the embedding and every Leiden call, so the cost is one UMAP plus a
    handful of Leiden runs, not a re-clustering per candidate.

    Falls back to ``fallback`` — with the reason recorded — when no count could
    be produced at all (too few cells, or no neighbour graph to embed from).
    """
    import scanpy as sc

    est = estimate_n_populations(
        adata,
        n_components=n_components,
        depth=depth,
        bandwidth=bandwidth,
        k_graph=k_graph,
        random_state=random_state,
        key_added="_granularity_probe",
    )
    adata.uns.get("scfair", {}).pop("_granularity_probe", None)
    diag: dict[str, Any] = {"granularity": est.to_dict()}

    # A target below 2 would drive the resolution to the bottom of the bracket
    # and leave a single intermediate cluster -- cluster-vs-rest then has no
    # "rest", every specificity score collapses, and the balanced path silently
    # becomes plain global HVG (the `insufficient_clusters` regime in
    # _diagnosis.py). Never let the estimate do that; fall back instead.
    if est.n_populations is None or est.n_populations < 2:
        diag.update({"resolution_source": "fallback", "n_leiden_calls": 0})
        logger.info(
            "resolution='auto': no usable population count (%s, n=%s); falling "
            "back to resolution=%.3g.",
            est.reason,
            est.n_populations,
            fallback,
        )
        return float(fallback), diag

    def leiden_fn(res: float) -> int:
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", category=FutureWarning)
            sc.tl.leiden(
                adata,
                resolution=float(res),
                random_state=random_state,
                key_added=leiden_key,
                flavor="igraph",
                n_iterations=2,
            )
        return int(adata.obs[leiden_key].nunique())

    res, n_got, calls = resolution_for_n_clusters(leiden_fn, est.n_populations)
    if leiden_key in adata.obs:  # DataFrame.pop takes no default
        del adata.obs[leiden_key]
    diag.update(
        {
            "resolution_source": "density_field",
            "n_populations_target": int(est.n_populations),
            "n_clusters_achieved": int(n_got),
            "n_leiden_calls": int(calls),
        }
    )
    if n_got != est.n_populations:
        # Leiden's cluster count is a step function of resolution and can skip
        # values outright, so "closest reachable" is a normal outcome, not a
        # failure -- but it is recorded rather than silently accepted.
        logger.info(
            "resolution='auto': target %d clusters not reachable; closest is %d "
            "at resolution=%.4g.",
            est.n_populations,
            n_got,
            res,
        )
    return float(res), diag


def resolution_for_n_clusters(
    leiden_fn,
    n_target: int,
    *,
    lo: float = 0.05,
    hi: float = 4.0,
    max_calls: int = 12,
) -> tuple[float, int, int]:
    """Bisect Leiden's resolution until it yields ``n_target`` communities.

    ``leiden_fn(resolution) -> n_clusters``. Kept as a callback so this module
    never touches AnnData and stays unit-testable without scanpy.

    Leiden's cluster count is a non-strict step function of resolution, so the
    target can be unreachable; the closest resolution found is returned along
    with the count actually achieved, and the caller decides whether that is
    good enough. Returns ``(resolution, n_clusters, n_calls)``.
    """
    best: tuple[float, int] | None = None
    calls = 0

    def evaluate(res: float) -> int:
        nonlocal best, calls
        n = int(leiden_fn(res))
        calls += 1
        if best is None or abs(n - n_target) < abs(best[1] - n_target):
            best = (res, n)
        return n

    n_lo, n_hi = evaluate(lo), evaluate(hi)
    if n_lo > n_target or n_hi < n_target:
        # target outside the bracket: the closest endpoint is the honest answer
        logger.info(
            "Target %d clusters is outside the resolution bracket [%g, %g] "
            "(%d..%d); using the closest resolution found.",
            n_target,
            lo,
            hi,
            n_lo,
            n_hi,
        )
        return (*best, calls)  # type: ignore[return-value]

    while calls < max_calls and best is not None and best[1] != n_target:
        mid = float(np.sqrt(lo * hi))  # bisect in log space: resolution is a scale
        n_mid = evaluate(mid)
        if n_mid < n_target:
            lo = mid
        elif n_mid > n_target:
            hi = mid
        else:
            break
        if hi / lo < 1.02:  # bracket collapsed; the count skips n_target
            break
    return (*best, calls)  # type: ignore[return-value]
