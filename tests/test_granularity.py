"""Tests for the 3D density-field population count.

The properties worth pinning down are the ones the design rests on:

- separated populations are counted, and the count is the *number*, not the
  assignment (which is measurably worse than Leiden's — see the module docs)
- `depth` is monotone: raising it can only merge, never split
- the bandwidth is a fraction of `n_obs`, so the same shape at a different
  scale gives the same count
- the estimate never clobbers the user's UMAP, and never touches selection
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scfair.pp._granularity import (
    AUTO_RESOLUTION_LO,
    DEFAULT_BANDWIDTH_FRAC,
    MIN_BANDWIDTH,
    default_bandwidth,
    estimate_n_populations,
    knn_density,
    knn_graph,
    merge_peaks,
    population_count_from_embedding,
    resolution_for_n_clusters,
)


def blobs(n_per: int = 300, centres=((0, 0, 0), (6, 0, 0), (0, 6, 0)), sd=0.35):
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(c, sd, size=(n_per, 3)) for c in centres])


# ---------------------------------------------------------------------------
# the core estimate
# ---------------------------------------------------------------------------
def test_separated_blobs_are_counted():
    est = population_count_from_embedding(blobs())
    assert est.reason == "ok"
    assert est.n_populations == 3
    assert est.bandwidth == default_bandwidth(900)


def test_labels_are_returned_and_match_the_count():
    est = population_count_from_embedding(blobs())
    assert est.labels is not None
    assert np.unique(est.labels).size == est.n_populations
    # each blob should land in one population: 300 cells, contiguous by
    # construction
    for start in (0, 300, 600):
        block = est.labels[start : start + 300]
        majority = np.bincount(np.unique(block, return_inverse=True)[1]).max()
        assert majority / block.size > 0.95


def test_one_blob_gives_one_population():
    est = population_count_from_embedding(blobs(centres=((0, 0, 0),)))
    assert est.n_populations == 1


def test_bandwidth_is_a_fraction_of_n_not_an_absolute_count():
    """Calibrated as 2% of n_obs; an absolute neighbour count does not transfer
    across dataset sizes."""
    assert default_bandwidth(20_000) == round(DEFAULT_BANDWIDTH_FRAC * 20_000)
    assert default_bandwidth(100) == MIN_BANDWIDTH  # floor for small data
    assert default_bandwidth(5_000) == 100


def test_same_shape_at_a_different_scale_gives_the_same_count():
    """Because the bandwidth scales with n, tripling the cells must not change
    how many populations are seen."""
    # both sizes must clear MIN_BANDWIDTH, or the floor -- not the fraction --
    # is what is being tested
    small = population_count_from_embedding(blobs(n_per=600))
    large = population_count_from_embedding(blobs(n_per=1800))
    assert small.n_populations == large.n_populations == 3
    assert large.bandwidth == 3 * small.bandwidth


def test_depth_is_monotone():
    """Raising `depth` merges; it must never split."""
    X = blobs(centres=((0, 0, 0), (2.2, 0, 0), (0, 6, 0)))
    nbrs = knn_graph(X, 15)
    rho = knn_density(X, default_bandwidth(X.shape[0]))
    counts = [np.unique(merge_peaks(rho, nbrs, d)).size for d in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert counts == sorted(counts, reverse=True), counts


def test_too_few_cells_is_refused_not_guessed():
    est = population_count_from_embedding(blobs(n_per=5))
    assert est.n_populations is None
    assert est.reason == "too_few_cells"


# ---------------------------------------------------------------------------
# the AnnData entry point
# ---------------------------------------------------------------------------
@pytest.fixture
def adata_blobs():
    X = blobs()
    a = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    a.obsm["X_precomputed"] = X
    return a


def test_precomputed_embedding_needs_no_neighbours(adata_blobs):
    est = estimate_n_populations(
        adata_blobs,
        embedding=adata_blobs.obsm["X_precomputed"],
    )
    assert est.n_populations == 3
    stored = adata_blobs.uns["scfair"]["granularity"]
    assert stored["n_populations"] == 3
    assert stored["bandwidth"] == default_bandwidth(adata_blobs.n_obs)
    assert stored["n_components"] == 3


def test_missing_neighbours_is_reported_not_raised(adata_blobs):
    est = estimate_n_populations(adata_blobs)
    assert est.n_populations is None
    assert est.reason == "no_embedding"
    assert adata_blobs.uns["scfair"]["granularity"]["reason"] == "no_embedding"


def test_labels_go_to_obs_only_when_asked(adata_blobs):
    estimate_n_populations(adata_blobs, embedding=adata_blobs.obsm["X_precomputed"])
    assert "population" not in adata_blobs.obs
    estimate_n_populations(
        adata_blobs,
        embedding=adata_blobs.obsm["X_precomputed"],
        copy_labels_to_obs="population",
    )
    assert adata_blobs.obs["population"].nunique() == 3


def test_does_not_touch_selection_or_the_users_umap(adata_blobs):
    adata_blobs.obsm["X_umap"] = np.zeros((adata_blobs.n_obs, 2), dtype=np.float32)
    before = adata_blobs.obsm["X_umap"].copy()
    estimate_n_populations(adata_blobs, embedding=adata_blobs.obsm["X_precomputed"])
    np.testing.assert_array_equal(adata_blobs.obsm["X_umap"], before)
    assert "highly_variable" not in adata_blobs.var
    assert "hvg" not in adata_blobs.uns.get("scfair", {})


# ---------------------------------------------------------------------------
# resolution bisection
# ---------------------------------------------------------------------------
def test_bisection_hits_an_reachable_target():
    # a monotone step function standing in for Leiden
    def leiden(res: float) -> int:
        return int(np.clip(round(2 + 12 * res), 1, 40))

    res, n, calls = resolution_for_n_clusters(leiden, 8)
    assert n == 8
    assert calls <= 12
    assert leiden(res) == 8


def test_bisection_returns_the_closest_when_the_target_is_skipped():
    def leiden(res: float) -> int:
        return 5 if res < 1.0 else 9  # 8 is never produced

    res, n, calls = resolution_for_n_clusters(leiden, 8)
    assert n == 9  # closest to 8
    assert calls <= 12


def test_bisection_reports_targets_outside_the_bracket():
    def leiden(res: float) -> int:
        return int(np.clip(round(2 + 2 * res), 1, 10))

    _, n, calls = resolution_for_n_clusters(leiden, 40, lo=0.05, hi=4.0)
    assert n == 10  # the bracket's best
    assert calls == 2  # gives up immediately, no search


def test_bisection_floors_lo_to_auto_resolution_lo():
    """Ultra-low lo (e.g. 0.05) is raised to AUTO_RESOLUTION_LO so rare types
    are not erased by a near-zero resolution that still reports high ARI on
    the coarse majority partition."""
    seen: list[float] = []

    def leiden(res: float) -> int:
        seen.append(float(res))
        return int(np.clip(round(2 + 12 * res), 1, 40))

    res, n, _ = resolution_for_n_clusters(leiden, 8, lo=0.05, hi=4.0)
    assert res >= AUTO_RESOLUTION_LO - 1e-12
    assert min(seen) >= AUTO_RESOLUTION_LO - 1e-12
    assert n == 8


def test_population_count_exposes_confidence_margin():
    est = population_count_from_embedding(blobs())
    assert est.reason == "ok"
    assert est.confidence in ("high", "moderate", "low")
    assert est.depth_sensitivity is not None
    assert est.n_populations_loose is not None
    assert est.n_populations_strict is not None
    d = est.to_dict()
    assert "confidence" in d
    assert "depth_sensitivity" in d


def test_knn_density_finite_on_duplicate_points():
    """Identical coordinates must not yield all-NaN density (loess of merge_peaks)."""
    # Two collapsed clusters of 25 identical points each.
    a = np.tile([0.0, 0.0, 0.0], (25, 1))
    b = np.tile([10.0, 0.0, 0.0], (25, 1))
    X = np.vstack([a, b])
    rho = knn_density(X, k=5)
    assert np.all(np.isfinite(rho)), f"non-finite dens: nan={np.isnan(rho).sum()}"
    assert rho.max() > 0
    est = population_count_from_embedding(X, bandwidth=5)
    # Flat / floored field → low confidence (must not claim high).
    assert est.confidence == "low"
    assert est.n_populations is not None
    # With two graph-separated components, count should not explode to n_obs.
    assert est.n_populations <= 10


def test_bisection_tie_prefers_higher_resolution():
    """Equal |n−target| must keep the higher resolution (rare-type bias)."""

    # lo→n=6, hi→n=10; target=8 is equidistant (gap 2). Prefer hi.
    def leiden(res: float) -> int:
        return 6 if res < 1.0 else 10

    res, n, _ = resolution_for_n_clusters(leiden, 8, lo=0.2, hi=4.0)
    assert n == 10
    assert res >= 1.0
