"""Tests for the public scfair.pp.highly_variable_genes API."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import scfair as scf
from scfair._utils import UNS_KEY
from scfair.pp._highly_variable_genes import (
    _cluster_size_weights,
    _cluster_vs_rest_logfc,
    _is_mito_name,
    _is_ribo_name,
    _nearest_cluster_map,
)


@pytest.fixture
def adata_for_hvg():
    """Synthetic counts with three groups and private markers."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_obs, n_vars = 200, 80
    X = rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    labels = np.array([0] * 70 + [1] * 70 + [2] * 60)
    for g, gene_idx in enumerate([0, 1, 2]):
        cells = labels == g
        X[cells, gene_idx] = rng.poisson(20, size=cells.sum()).astype(np.float32)
        for j in range(3):
            X[cells, 10 + g * 3 + j] = rng.poisson(12, size=cells.sum()).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"c{i}" for i in range(n_obs)]
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    adata.obs["true_group"] = labels.astype(str)
    return adata


def test_cluster_weights_not_equal_votes():
    sizes = __import__("pandas").Series({"big": 900, "small": 100})
    w_eq = _cluster_size_weights(sizes, power=0.0)
    w05 = _cluster_size_weights(sizes, power=0.5)
    w_prop = _cluster_size_weights(sizes, power=1.0)

    assert abs(w_eq["big"] - w_eq["small"]) < 1e-12
    assert w05["big"] > w05["small"]
    assert w_prop["big"] > w_prop["small"]
    assert w05["small"] > w_prop["small"]
    assert w05["small"] < w_eq["small"]


def test_cluster_vs_rest_logfc_one_sided():
    X = np.array(
        [
            [4.0, 1.0],
            [4.0, 1.0],
            [4.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    mask = np.array([True, True, True, False, False, False])
    logfc = _cluster_vs_rest_logfc(X, mask, pseudo=1e-2, one_sided=True)
    assert logfc[0] > 1.0
    assert logfc[1] == 0.0


def test_cluster_vs_rest_logfc_explicit_out_mask():
    """An explicit reference set scores against *that* set, not the complement."""
    X = np.array(
        [
            [8.0, 4.0],  # cluster A
            [8.0, 4.0],
            [4.0, 4.0],  # cluster B — close to A on gene 0
            [4.0, 4.0],
            [0.0, 4.0],  # cluster C — far from A
            [0.0, 4.0],
        ],
        dtype=float,
    )
    a = np.array([True, True, False, False, False, False])
    b = np.array([False, False, True, True, False, False])

    vs_rest = _cluster_vs_rest_logfc(X, a, one_sided=True)
    vs_b = _cluster_vs_rest_logfc(X, a, one_sided=True, out_mask=b)
    # against everything (mean 2.0) the gap looks larger than against the
    # adjacent cluster B (mean 4.0) — this is the blind spot the neighbour
    # contrast exists to remove
    assert vs_rest[0] > vs_b[0] > 0.0
    assert vs_b[1] == 0.0


def test_nearest_cluster_map_picks_closest_profile():
    # genes: A and B share a profile shape, C is anti-correlated
    X = np.array(
        [
            [10.0, 1.0, 1.0],
            [10.0, 1.0, 1.0],
            [8.0, 2.0, 1.0],
            [8.0, 2.0, 1.0],
            [1.0, 1.0, 10.0],
            [1.0, 1.0, 10.0],
        ],
        dtype=float,
    )
    masks = {
        "A": np.array([True, True, False, False, False, False]),
        "B": np.array([False, False, True, True, False, False]),
        "C": np.array([False, False, False, False, True, True]),
    }
    nn = _nearest_cluster_map(X, masks)
    assert nn["A"] == "B"
    assert nn["B"] == "A"
    assert set(nn) == {"A", "B", "C"}
    assert all(v != k for k, v in nn.items())


def test_nearest_cluster_map_needs_two_clusters():
    X = np.ones((4, 3), dtype=float)
    assert _nearest_cluster_map(X, {"only": np.ones(4, dtype=bool)}) == {}


def test_hvg_neighbor_contrast_runs_and_is_recorded(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=40,
        balance_method="hybrid",
        neighbor_contrast=1.0,
        resolution=1.0,
        min_cluster_size=20,
        random_state=0,
    )
    assert int(ad.var["highly_variable"].sum()) == 40
    assert ad.uns[UNS_KEY]["hvg"]["neighbor_contrast"] == 1.0


def test_neighbor_contrast_warns_at_low_resolution(adata_for_hvg, caplog):
    """The two fixes cancel; the combination must not fail silently."""
    ad = adata_for_hvg.copy()
    with caplog.at_level("WARNING", logger="scfair.pp._diagnosis"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=20,
            balance_method="hybrid",
            neighbor_contrast=1.0,
            resolution=0.5,
            min_cluster_size=20,
        )
    assert any("neighbor_contrast" in r.message for r in caplog.records)


def test_large_k_warns(adata_for_hvg, caplog):
    """k>=3000 is advisory-only: warn, never refuse."""
    ad = adata_for_hvg.copy()
    with caplog.at_level("WARNING", logger="scfair.pp._diagnosis"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=3000, balance_method="hybrid", min_cluster_size=20
        )
    assert any(">= 3000" in r.message for r in caplog.records)
    # advisory, not a refusal: it still selects every gene it can
    assert int(ad.var["highly_variable"].sum()) == ad.n_vars


def test_large_k_warns_before_the_expensive_step(adata_for_hvg, caplog):
    """The k>=3000 advice must precede the intermediate clustering.

    The balanced path spends ~90% of its runtime there, so advice delivered
    afterwards has already cost the caller the thing it is advising against.
    """
    ad = adata_for_hvg.copy()
    with caplog.at_level("INFO"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=3000,
            balance_method="hybrid",
            min_cluster_size=20,
            progress=False,
        )
    msgs = [r.message for r in caplog.records]
    k_warn = next(i for i, m in enumerate(msgs) if ">= 3000" in m)
    clustering = next(i for i, m in enumerate(msgs) if "intermediate clustering" in m)
    assert k_warn < clustering


def test_large_k_says_nothing_on_the_scanpy_path(adata_for_hvg, caplog):
    """balance_method='none' has no re-ranking, so "no advantage" is vacuous."""
    ad = adata_for_hvg.copy()
    with caplog.at_level("WARNING", logger="scfair.pp._diagnosis"):
        scf.pp.highly_variable_genes(ad, n_top_genes=3000, balance_method="none")
    assert not any(">= 3000" in r.message for r in caplog.records)


def test_each_config_finding_is_reported_once(adata_for_hvg, caplog):
    """One finding, one message: the pre-flight check and the post-run diagnosis
    must not both announce the same thing."""
    ad = adata_for_hvg.copy()
    with caplog.at_level("WARNING"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=3000, balance_method="hybrid", min_cluster_size=20
        )
    assert len([r for r in caplog.records if ">= 3000" in r.message]) == 1
    # ...and it is still recorded in the machine-readable diagnosis
    assert "k_ge_3000" in ad.uns[UNS_KEY]["hvg"]["diagnosis"]["flags"]


def test_moderate_k_does_not_warn(adata_for_hvg, caplog):
    ad = adata_for_hvg.copy()
    with caplog.at_level("WARNING", logger="scfair.pp._diagnosis"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=2000, balance_method="none", min_cluster_size=20
        )
    assert not any(">= 3000" in r.message for r in caplog.records)


def test_hvg_default_resolution_is_auto(adata_for_hvg):
    """Default resolution is "auto". Pinned so a revert is deliberate.

    ``resolution`` records the request and ``resolution_used`` the number that
    actually ran; with "auto" they differ, and only the second reproduces the
    partition.
    """
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="hybrid", min_cluster_size=20)
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["resolution"] == "auto"
    assert isinstance(meta["resolution_used"], float)


def test_auto_resolution_falls_back_when_it_cannot_decide(adata_for_hvg):
    """A target below 2 must not be honoured -> the pre-auto default.

    On this fixture the density field resolves a single population; honouring
    that would collapse cluster-vs-rest to plain global HVG. The fallback
    keeps "auto" safe by default, and the reason is recorded rather than
    inferred from the number.
    """
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="hybrid", min_cluster_size=20)
    clustering = ad.uns[UNS_KEY]["hvg"]["clustering"]
    assert clustering["resolution"] == 0.5
    assert clustering["resolution_source"] == "fallback"
    assert clustering["granularity"]["n_populations"] < 2
    assert clustering["n_clusters_kept"] >= 2  # the collapse did not happen


def test_explicit_resolution_still_wins_and_runs_no_probe(adata_for_hvg):
    """A number passed in is used as-is; no density field, no Leiden probes."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=1.0,
    )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["resolution"] == 1.0
    assert meta["resolution_used"] == 1.0
    assert "granularity" not in meta["clustering"]


def test_unknown_resolution_string_is_rejected(adata_for_hvg):
    with pytest.raises(ValueError, match="resolution must be"):
        scf.pp.highly_variable_genes(
            adata_for_hvg.copy(),
            n_top_genes=20,
            balance_method="hybrid",
            min_cluster_size=20,
            resolution="fine",
        )


def test_hvg_neighbor_contrast_zero_matches_default(adata_for_hvg):
    """Default stays bit-identical: nc=0 must not perturb existing results."""
    ad_a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_a, n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0
    )
    ad_b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_b,
        n_top_genes=40,
        balance_method="hybrid",
        neighbor_contrast=0.0,
        min_cluster_size=20,
        random_state=0,
    )
    assert list(ad_a.var_names[ad_a.var["highly_variable"]]) == list(
        ad_b.var_names[ad_b.var["highly_variable"]]
    )


def test_hvg_neighbor_contrast_out_of_range(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="neighbor_contrast"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, neighbor_contrast=1.5)


def _hvg_names(a):
    return list(a.var_names[a.var["highly_variable"]])


def test_cluster_pool_none_matches_default(adata_for_hvg):
    """Default stays bit-identical: cluster_pool=None must change nothing."""
    kw = dict(n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0)
    ad_a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_a, **kw)
    ad_b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_b, cluster_pool=None, **kw)
    assert _hvg_names(ad_a) == _hvg_names(ad_b)


def test_cluster_pool_equal_to_k_matches_default(adata_for_hvg):
    """cluster_pool=k reproduces the default, which clusters the top-k mask.

    This is the invariant that makes the parameter interpretable: it only ever
    changes *which* genes seed the intermediate clustering, and setting it to the
    value the default implies must be a no-op.
    """
    kw = dict(n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0)
    ad_a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_a, **kw)
    ad_b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_b, cluster_pool=40, **kw)
    assert _hvg_names(ad_a) == _hvg_names(ad_b)


def test_cluster_pool_larger_changes_selection_and_is_recorded(adata_for_hvg):
    """A wider clustering pool is allowed to move the selection, and is logged."""
    # n_pcs kept below the smaller clustering space so the two arms differ only
    # in the pool, not in whether PCA is feasible
    kw = dict(
        n_top_genes=30,
        balance_method="hybrid",
        min_cluster_size=20,
        n_pcs=10,
        random_state=0,
    )
    ad_a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_a, **kw)
    ad_b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_b, cluster_pool=70, **kw)
    assert ad_b.uns[UNS_KEY]["hvg"]["cluster_pool"] == 70
    assert ad_a.uns[UNS_KEY]["hvg"]["cluster_pool"] is None
    # same requested k either way
    assert int(ad_b.var["highly_variable"].sum()) == int(ad_a.var["highly_variable"].sum())


def test_marker_genes_already_selected_diagnostic(adata_for_hvg):
    """Reports how many supplied markers the algorithm chose unaided.

    This is the number that tells a user whether injecting is worth the subspace
    cost, so it must be counted *before* marker handling runs and must not be
    inflated by forcing.
    """
    # g0/g1/g2 are the fixture's strong private markers, so they are selected on
    # their own; g79 is background noise and is not.
    panel = ["g0", "g1", "g2", "g79"]

    ad_none = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_none,
        n_top_genes=20,
        balance_method="hybrid",
        marker_genes=panel,
        marker_mode="none",
        min_cluster_size=20,
        random_state=0,
    )
    m = ad_none.uns[UNS_KEY]["hvg"]
    assert m["n_marker_genes"] == 4  # all four exist in var_names
    already = m["n_marker_genes_already_selected"]
    assert 0 <= already <= 4
    selected = set(_hvg_names(ad_none))
    assert already == len(set(panel) & selected)

    # forcing must not change the diagnostic: it is measured pre-injection
    ad_force = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_force,
        n_top_genes=20,
        balance_method="hybrid",
        marker_genes=panel,
        marker_mode="force",
        marker_extra=True,
        min_cluster_size=20,
        random_state=0,
    )
    mf = ad_force.uns[UNS_KEY]["hvg"]
    assert mf["n_marker_genes_already_selected"] == already
    # ...while the panel really is present after forcing
    assert set(panel) <= set(_hvg_names(ad_force))


def test_progress_true_announces_stages_before_they_run(adata_for_hvg, capsys):
    """Stage messages must reach stderr, and must name the slow stage."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="hybrid",
        min_cluster_size=20,
        progress=True,
    )
    err = capsys.readouterr().err
    assert "global HVG pass" in err
    # the intermediate clustering is what actually blocks; it must be announced
    assert "intermediate clustering" in err
    assert "specificity scoring" in err


def test_progress_reports_the_result_not_only_the_stages(adata_for_hvg, capsys):
    """The channel must say the call *finished*, and what it produced.

    Stage messages are emitted before each slow step, so without a closing line
    a completed run and a run still inside the slow step look identical on
    stderr. The equivalent summary existed only on the logger — invisible to
    anyone who has not configured logging.
    """
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="hybrid",
        min_cluster_size=20,
        progress=True,
    )
    err = capsys.readouterr().err
    assert "done: 20 genes" in err
    assert "intermediate clusters" in err
    assert "resolution" in err


def test_progress_result_line_on_the_scanpy_path(adata_for_hvg, capsys):
    """balance_method='none' runs no clustering, so it must claim none."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="none", progress=True)
    err = capsys.readouterr().err
    assert "done: 20 genes (global HVG, no clustering)." in err
    assert "intermediate clusters" not in err


def test_progress_false_is_silent(adata_for_hvg, capsys):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="hybrid",
        min_cluster_size=20,
        progress=False,
    )
    assert capsys.readouterr().err == ""


def test_progress_default_quiet_on_small_data(adata_for_hvg, capsys):
    """Default must not chatter on small inputs (200x80 here)."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="hybrid", min_cluster_size=20)
    assert capsys.readouterr().err == ""


def test_progress_structure_auto_announces_single_hybrid(adata_for_hvg, capsys):
    """Default structure auto: one hybrid after k (no double-intermediate message)."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes="auto",
        n_top_min=10,
        n_top_max=40,
        balance_method="hybrid",
        min_cluster_size=20,
        progress=True,
    )
    err = capsys.readouterr().err
    assert "structure" in err.lower()
    assert "one hybrid" in err.lower() or "no double" in err.lower()
    # physical intermediate clustering once (hybrid@k), not twice
    assert err.count("intermediate clustering (the slow step)") == 1


def test_progress_ensemble_auto_mentions_shared_graph(adata_for_hvg, capsys):
    """Ensemble auto: progress notes protocol-matched reuse, not blind 'twice'."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes="auto",
        auto_n_method="ensemble",
        n_top_min=10,
        n_top_max=40,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        progress=True,
    )
    err = capsys.readouterr().err
    assert "shared" in err.lower() or "reus" in err.lower()


def test_marker_diagnostic_is_none_without_markers(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="hybrid", min_cluster_size=20)
    assert ad.uns[UNS_KEY]["hvg"]["n_marker_genes_already_selected"] is None


def test_cluster_pool_invalid(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="cluster_pool"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, cluster_pool=1)


def test_best_rank_score_only_promotes():
    """best_rank can lift a gene above its rank in either input, never below."""
    from scfair.pp._highly_variable_genes import _best_rank_score

    a = pd.Series({"g0": 10.0, "g1": 5.0, "g2": 1.0})
    b = pd.Series({"g0": 1.0, "g1": 5.0, "g2": 10.0})
    s = _best_rank_score(a, b)
    order = list(s.sort_values(ascending=False).index)
    # g0 is best in a, g2 best in b -> both outrank the middling g1
    assert order[-1] == "g1"
    assert set(order[:2]) == {"g0", "g2"}


def test_hvg_combine_best_rank(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=30,
        balance_method="hybrid",
        combine="best_rank",
        min_cluster_size=20,
        random_state=0,
    )
    assert int(ad.var["highly_variable"].sum()) == 30
    assert ad.uns[UNS_KEY]["hvg"]["combine"] == "best_rank"


def test_hvg_invalid_combine(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="combine"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, combine="nope")


def test_hvg_injected_global_score(adata_for_hvg):
    """An injected anchor must actually drive selection, not be ignored."""
    ad = adata_for_hvg.copy()
    # rank g0..g9 top by construction
    score = pd.Series([100.0 - i for i in range(ad.n_vars)], index=ad.var_names.astype(str))
    scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="none", global_score=score)
    assert list(ad.var_names[ad.var["highly_variable"]]) == [f"g{i}" for i in range(10)]
    assert ad.uns[UNS_KEY]["hvg"]["global_score"] == "injected"


def test_hvg_injected_global_score_misaligned(adata_for_hvg):
    ad = adata_for_hvg.copy()
    bad = pd.Series([1.0, 2.0], index=["not_a_gene", "also_not"])
    with pytest.raises(ValueError, match="align"):
        scf.pp.highly_variable_genes(ad, n_top_genes=5, global_score=bad)


def test_mito_ribo_name_helpers():
    assert _is_mito_name("MT-CO3")
    assert _is_mito_name("mt-Nd1")
    assert not _is_mito_name("COX4I1")
    assert not _is_mito_name("MTOR")
    assert _is_ribo_name("RPL19")
    assert _is_ribo_name("RPS6")
    assert _is_ribo_name("RPSA")
    assert _is_ribo_name("RPLP0")
    assert not _is_ribo_name("RPA1")
    # kinases share RPS6 prefix but are not ribosomal proteins
    assert not _is_ribo_name("RPS6KA1")
    assert not _is_ribo_name("RPS6KB1")
    assert not _is_ribo_name("RPS6KA3")


def test_hvg_global_only(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=30, balance_method="none", flavor="seurat_v3")
    assert int(ad.var["highly_variable"].sum()) == 30
    assert "counts" in ad.layers
    # Default store_raw=False: layers['counts'] only, no second full matrix in uns.
    assert "raw_snapshot" not in ad.uns.get(UNS_KEY, {})
    assert ad.uns[UNS_KEY]["hvg"]["balance_method"] == "none"
    assert ad.uns[UNS_KEY]["hvg"]["store_raw"] is False
    assert "scfair_score" in ad.var


def test_hvg_store_raw_true_writes_snapshot(adata_for_hvg):
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="none",
        options=HVGOptions(store_raw=True),
        diagnose=False,
    )
    assert "raw_snapshot" in ad.uns[UNS_KEY]
    assert ad.uns[UNS_KEY]["raw_snapshot"]["backend"] == "inline"
    assert ad.uns[UNS_KEY]["hvg"]["store_raw"] is True


def test_hvg_score_mode(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=40,
        balance_method="score",
        balance_power=0.5,
        resolution=0.8,
        min_cluster_size=20,
        flavor="seurat_v3",
        random_state=0,
    )
    assert int(ad.var["highly_variable"].sum()) == 40
    assert "scfair_hvg_clusters" in ad.obs
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "score"
    assert meta["selection"] == "cluster_vs_rest_specificity"
    assert meta["n_clusters_used"] >= 1
    assert meta["n_clusters_used"] == len(meta["cluster_weights"])
    assert np.isfinite(ad.var["scfair_score"]).all()


def test_hvg_hybrid_default_and_overlap_with_global(adata_for_hvg):
    ad_g = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_g, n_top_genes=40, balance_method="none")
    g_global = set(ad_g.var_names[ad_g.var["highly_variable"]])

    ad_h = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_h,
        n_top_genes=40,
        balance_method="hybrid",
        blend_global=0.95,
        balance_power=0.5,
        min_cluster_size=20,
        random_state=0,
    )
    g_h = set(ad_h.var_names[ad_h.var["highly_variable"]])
    meta = ad_h.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "hybrid"
    assert meta["blend_global"] == 0.95
    # pool re-rank stays close to global HVG set
    assert len(g_h & g_global) >= int(0.5 * 40)


def test_hvg_reweight_mode(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=40,
        balance_method="reweight",
        balance_power=0.5,
        min_cluster_size=20,
        flavor="seurat_v3",
        random_state=0,
    )
    assert int(ad.var["highly_variable"].sum()) == 40
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "reweight"
    assert meta["selection"] == "cell_reweighted_global_hvg"
    assert meta["n_clusters_used"] >= 1
    assert "scfair_hvg_clusters" in ad.obs


def test_hvg_inverse_size_alias_sets_beta_neg1(adata_for_hvg):
    """``inverse_size`` → score with true inverse-size weights (β=-1)."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=25,
        balance_method="inverse_size",
        min_cluster_size=20,
        random_state=0,
    )
    assert ad.uns[UNS_KEY]["hvg"]["balance_method"] == "score"
    assert ad.uns[UNS_KEY]["hvg"]["balance_power"] == -1.0


def test_balance_power_allows_negative(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="score",
        balance_power=-0.5,
        min_cluster_size=20,
        random_state=0,
        diagnose=False,
    )
    assert ad.uns[UNS_KEY]["hvg"]["balance_power"] == -0.5


def test_balance_power_rejects_out_of_range(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match=r"balance_power.*\[-2, 2\]"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="score", balance_power=-3.0)


@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger", "seurat_v3_paper"])
def test_hvg_flavor_routing(adata_for_hvg, flavor):
    """Each supported flavor selects k genes and leaves no scratch layer behind.

    The flavor itself is scanpy's; what is scFair's is the routing between
    _COUNTS_FLAVORS and _LOG_FLAVORS, which is what this checks.
    """
    ad = adata_for_hvg.copy()
    before = set(ad.layers)
    scf.pp.highly_variable_genes(ad, n_top_genes=20, flavor=flavor, balance_method="none")
    assert int(ad.var["highly_variable"].sum()) == 20
    # internal scratch must be cleaned up regardless of flavor
    assert "_scfair_log" not in ad.layers
    assert set(ad.layers) - before <= {"counts"}


def test_hvg_score_with_seurat_flavor(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=30,
        flavor="seurat",
        balance_method="score",
        balance_power=0.5,
        min_cluster_size=20,
        random_state=0,
    )
    assert int(ad.var["highly_variable"].sum()) == 30
    assert ad.uns[UNS_KEY]["hvg"]["flavor"] == "seurat"


def test_hvg_filter_ribo(adata_for_hvg):
    ad = adata_for_hvg.copy()
    # rename a few genes to look ribosomal
    new_names = list(ad.var_names)
    new_names[0] = "RPL19"
    new_names[1] = "RPS6"
    ad.var_names = new_names
    # force them highly variable via markers first without filter
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, balance_method="none", marker_genes=["RPL19", "RPS6"]
    )
    assert bool(ad.var.loc["RPL19", "highly_variable"])

    ad2 = adata_for_hvg.copy()
    ad2.var_names = new_names
    scf.pp.highly_variable_genes(ad2, n_top_genes=20, balance_method="none", filter_ribo=True)
    assert not bool(ad2.var.loc["RPL19", "highly_variable"])
    assert not bool(ad2.var.loc["RPS6", "highly_variable"])


def test_hvg_marker_genes_forced(adata_for_hvg):
    ad = adata_for_hvg.copy()
    markers = ["g50", "g51"]
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="none", marker_genes=markers)
    for g in markers:
        assert bool(ad.var.loc[g, "highly_variable"])


def test_hvg_inplace_false_returns_dataframe(adata_for_hvg):
    ad = adata_for_hvg.copy()
    out = scf.pp.highly_variable_genes(ad, n_top_genes=15, balance_method="none", inplace=False)
    assert out is not None
    assert "highly_variable" in out.columns
    assert "scfair_score" in out.columns
    assert out["highly_variable"].sum() == 15


def test_hvg_subset_true(adata_for_hvg):
    ad = adata_for_hvg.copy()
    n_full = ad.n_vars
    # subset=True auto-enables store_raw so full gene universe is recoverable.
    with pytest.warns(UserWarning, match="subset=True requires a raw_snapshot"):
        scf.pp.highly_variable_genes(ad, n_top_genes=25, balance_method="none", subset=True)
    assert ad.n_vars == 25
    assert len(ad.uns[UNS_KEY]["raw_snapshot"]["var_names"]) == n_full


def test_hvg_from_existing_counts_layer(adata_for_hvg):
    ad = adata_for_hvg.copy()
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    scf.pp.highly_variable_genes(ad, n_top_genes=20, layer="counts", balance_method="none")
    assert ad.var["highly_variable"].sum() == 20


def test_hvg_invalid_balance_method(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="balance_method"):
        scf.pp.highly_variable_genes(ad, balance_method="not_a_method")


def test_hvg_invalid_flavor(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="flavor"):
        scf.pp.highly_variable_genes(ad, flavor="not_a_flavor")


def test_public_api_surface(adata_for_hvg):
    import scfair as scf
    import scfair.pp as pp

    assert "highly_variable_genes" in scf.__all__
    assert "highly_variable_genes" in pp.__all__
    assert not hasattr(scf, "store_raw_counts")
    # the top-level alias must be the same callable, and must work
    assert scf.highly_variable_genes is pp.highly_variable_genes
    ad = adata_for_hvg.copy()
    scf.highly_variable_genes(ad, n_top_genes=10, balance_method="none")
    assert int(ad.var["highly_variable"].sum()) == 10


# ---------------------------------------------------------------------------
# Regression tests: each covers a defect the rest of the suite would have
# missed.
# ---------------------------------------------------------------------------


def _hvg_selected(ad) -> set[str]:
    return set(ad.var_names[ad.var["highly_variable"]].astype(str))


def _run_hvg_kwargs(ad, **kw):
    """Call highly_variable_genes; return selected gene set."""
    from scfair.pp import HVGOptions

    # Default research knobs small enough for the fixture (do not drop
    # an explicitly passed options=).
    if "options" not in kw and "append_budget" not in kw:
        kw["options"] = HVGOptions(append_budget=5)
    kw.setdefault("diagnose", False)
    kw.setdefault("balance_method", "append")
    scf.pp.highly_variable_genes(ad, **kw)
    return _hvg_selected(ad)


def test_same_object_second_call_matches_clean_object(adata_for_hvg):
    """Core anti-pollution contract (G1/G2/G3 pattern).

    Real workflows re-use one AnnData to compare layer / flavor / k. Suite tests
    that always ``adata.copy()`` never see leftover ``layers['counts']``,
    snapshot, or flavor score columns. Rule: after call_a then call_b on the
    *same* object, the result of call_b must equal call_b alone on a clean copy.
    """
    from scfair.pp import HVGOptions

    rng = np.random.default_rng(42)
    alt = rng.poisson(6.0, size=adata_for_hvg.X.shape).astype(np.float32)

    scenarios = [
        # (name, prepare_shared, kwargs_a, kwargs_b)
        (
            "layer_alt_then_default",
            lambda ad: ad.layers.__setitem__("alt", alt.copy()),
            dict(
                n_top_genes=25,
                layer="alt",
                options=HVGOptions(append_budget=5),
            ),
            dict(n_top_genes=25, options=HVGOptions(append_budget=5)),
        ),
        (
            "flavor_seurat_v3_then_seurat",
            None,
            dict(
                n_top_genes=25,
                flavor="seurat_v3",
                options=HVGOptions(append_budget=5),
            ),
            dict(
                n_top_genes=25,
                flavor="seurat",
                options=HVGOptions(append_budget=5),
            ),
        ),
        (
            "k_20_then_k_35",
            None,
            dict(n_top_genes=20, options=HVGOptions(append_budget=5)),
            dict(n_top_genes=35, options=HVGOptions(append_budget=5)),
        ),
        (
            "append_then_none",
            None,
            dict(
                n_top_genes=25,
                balance_method="append",
                options=HVGOptions(append_budget=8),
            ),
            dict(n_top_genes=25, balance_method="none"),
        ),
        (
            "none_then_append",
            None,
            dict(n_top_genes=25, balance_method="none"),
            dict(
                n_top_genes=25,
                balance_method="append",
                options=HVGOptions(append_budget=5),
            ),
        ),
    ]

    for name, prepare, kw_a, kw_b in scenarios:
        dirty = adata_for_hvg.copy()
        clean = adata_for_hvg.copy()
        if prepare is not None:
            prepare(dirty)
            prepare(clean)
        _run_hvg_kwargs(dirty, **kw_a)
        sel_dirty_b = _run_hvg_kwargs(dirty, **kw_b)
        sel_clean_b = _run_hvg_kwargs(clean, **kw_b)
        assert sel_dirty_b == sel_clean_b, (
            f"{name}: second call on reused object ≠ clean object "
            f"(only_dirty={sel_dirty_b - sel_clean_b}, "
            f"only_clean={sel_clean_b - sel_dirty_b})"
        )


def test_layer_does_not_hijack_later_default_calls(adata_for_hvg):
    """G1: layer='alt' must not make subsequent layer=None calls reuse alt."""
    from scfair.pp import HVGOptions

    rng = np.random.default_rng(11)
    alt = rng.poisson(6.0, size=adata_for_hvg.X.shape).astype(np.float32)

    ad_alt = adata_for_hvg.copy()
    ad_alt.layers["alt"] = alt.copy()
    sel_alt = _run_hvg_kwargs(
        ad_alt, n_top_genes=25, layer="alt", options=HVGOptions(append_budget=5)
    )
    if "counts" in ad_alt.layers:
        assert not np.array_equal(np.asarray(ad_alt.layers["counts"]), np.asarray(alt))

    dirty = adata_for_hvg.copy()
    dirty.layers["alt"] = alt.copy()
    _run_hvg_kwargs(dirty, n_top_genes=25, layer="alt", options=HVGOptions(append_budget=5))
    sel_after = _run_hvg_kwargs(dirty, n_top_genes=25, options=HVGOptions(append_budget=5))
    sel_fresh = _run_hvg_kwargs(
        adata_for_hvg.copy(), n_top_genes=25, options=HVGOptions(append_budget=5)
    )
    assert sel_after == sel_fresh
    assert sel_after != sel_alt


def test_x_swap_after_first_call_uses_new_matrix(adata_for_hvg):
    """G2: after .X swap, selection follows .X; user counts layer not clobbered."""
    from scfair.pp import HVGOptions
    from scfair.pp._raw_counts import INTERNAL_COUNTS_LAYER

    rng = np.random.default_rng(12)
    ad = adata_for_hvg.copy()
    sel_old = _run_hvg_kwargs(ad, n_top_genes=20, options=HVGOptions(append_budget=5))
    # First call may have materialised counts from X1.
    counts_user = rng.poisson(4.0, size=ad.X.shape).astype(np.float32)
    ad.layers["counts"] = counts_user.copy()
    X2 = rng.poisson(9.0, size=ad.X.shape).astype(np.float32)
    ad.X = X2
    with pytest.warns(UserWarning, match="left unchanged|internal layer"):
        sel_new = _run_hvg_kwargs(ad, n_top_genes=20, options=HVGOptions(append_budget=5))
    # User matrix intact; ephemeral layer cleaned up.
    assert np.array_equal(np.asarray(ad.layers["counts"]), np.asarray(counts_user))
    assert INTERNAL_COUNTS_LAYER not in ad.layers

    ad_ref = adata_for_hvg.copy()
    ad_ref.X = X2.copy()
    sel_ref = _run_hvg_kwargs(ad_ref, n_top_genes=20, options=HVGOptions(append_budget=5))
    assert sel_new == sel_ref
    assert sel_new != sel_old


def test_mode_with_fixed_int_k_warns(adata_for_hvg):
    """E3: mode= only steers auto k; fixed int k should warn once."""
    with pytest.warns(UserWarning, match="mode=.*no effect|n_top_genes is a fixed int"):
        scf.pp.highly_variable_genes(
            adata_for_hvg.copy(),
            n_top_genes=20,
            mode="fine",
            balance_method="none",
            diagnose=False,
        )


def test_flavor_rerun_does_not_reuse_prior_score_columns(adata_for_hvg):
    """G3: second flavor on same object equals clean run of that flavor."""
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    sel_v3 = _run_hvg_kwargs(
        ad, n_top_genes=25, flavor="seurat_v3", options=HVGOptions(append_budget=5)
    )
    assert "variances_norm" in ad.var.columns

    sel_seurat = _run_hvg_kwargs(
        ad, n_top_genes=25, flavor="seurat", options=HVGOptions(append_budget=5)
    )
    assert "dispersions_norm" in ad.var.columns or "dispersions" in ad.var.columns

    sel_fresh = _run_hvg_kwargs(
        adata_for_hvg.copy(),
        n_top_genes=25,
        flavor="seurat",
        options=HVGOptions(append_budget=5),
    )
    assert sel_seurat == sel_fresh
    assert sel_seurat != sel_v3 or "variances_norm" not in ad.var.columns


def test_hybrid_degenerates_to_global_when_no_clusters(adata_for_hvg):
    """H1: n_clusters_used=0 must not be logged as a successful hybrid run."""
    ad = adata_for_hvg.copy()
    ad_none = adata_for_hvg.copy()
    # Force every Leiden cluster below min size → specificity has no signal.
    with pytest.warns(UserWarning, match="n_clusters_used=0|degenerated|≈ scanpy"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=20,
            balance_method="hybrid",
            min_cluster_size=100_000,
            diagnose=False,
            random_state=0,
        )
    scf.pp.highly_variable_genes(
        ad_none,
        n_top_genes=20,
        balance_method="none",
        diagnose=False,
        random_state=0,
    )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["n_clusters_used"] == 0
    assert meta["selection"] == "hybrid_degenerated_to_global"
    assert meta.get("degenerated_to_global") is True
    assert "hybrid_global_anchor" not in str(meta["selection"])
    assert set(ad.var_names[ad.var["highly_variable"]]) == set(
        ad_none.var_names[ad_none.var["highly_variable"]]
    )


def test_highly_variable_rank_is_nan_outside_selection(adata_for_hvg):
    """scanpy convention: only selected genes have finite highly_variable_rank."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="none", diagnose=False)
    rank = ad.var["highly_variable_rank"]
    hv = ad.var["highly_variable"].to_numpy()
    n_hv = int(hv.sum())
    assert n_hv == 20
    assert int(rank.notna().sum()) == n_hv
    assert bool(np.all(np.isnan(rank.to_numpy()[~hv])))
    assert bool(np.all(np.isfinite(rank.to_numpy()[hv])))
    # Downstream scanpy-style filters must work (inf used to pass every gene).
    filtered = ad.var.dropna(subset=["highly_variable_rank"])
    assert len(filtered) == n_hv
    # Selected ranks cast cleanly after dropna.
    _ = filtered["highly_variable_rank"].astype(int)


def test_append_rank_matches_final_selection_size(adata_for_hvg):
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=5),
        diagnose=False,
    )
    n_sel = int(ad.var["highly_variable"].sum())
    assert n_sel == 25
    assert int(ad.var["highly_variable_rank"].notna().sum()) == n_sel


def test_hybrid_pool_is_the_true_top_2k():
    """The candidate pool must be ordered by all-gene scores, not by rank.

    scanpy leaves highly_variable_rank as NaN outside the top-n_top, which
    _gene_rank_series fills with +inf. Ordering by that column put the true ranks
    k+1..2k *outside* the pool and arbitrary index-ordered genes inside it, so a
    highly specific gene at true rank k+1 could never be promoted.
    """
    from scfair.pp._highly_variable_genes import _hybrid_anchor_select

    n = 20
    names = [f"g{i}" for i in range(n)]
    # descending variability: g0 highest ... g19 lowest
    global_scores = pd.Series(np.linspace(1.0, 0.0, n), index=names)
    # mimic scanpy / public scFair column: only the top-5 carry a finite rank
    rank = pd.Series(np.nan, index=names)
    rank.iloc[:5] = np.arange(1, 6)
    # a gene just outside the top-5 (true rank 6) is the most specific one
    spec = pd.Series(0.0, index=names)
    spec["g5"] = 1.0

    selected, _, _ = _hybrid_anchor_select(
        global_rank=rank,
        global_scores=global_scores,
        spec_scores=spec,
        n_top_genes=5,
        blend_global=0.5,
        pool_factor=2.0,
    )
    # with a 2x pool it is eligible, and at alpha=0.5 its specificity wins a slot
    assert "g5" in selected


def test_blend_global_zero_stays_inside_the_pool(adata_for_hvg):
    """alpha=0 must mean "specificity-only *within the pool*", not genome-wide.

    A gate that treats alpha=0 as "hybrid disabled" would silently return
    pure `score` while metadata still said hybrid.
    """
    k = 10  # keep 2*k well below n_vars so the pool is a real restriction
    ad_h = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_h,
        n_top_genes=k,
        balance_method="hybrid",
        blend_global=0.0,
        min_cluster_size=20,
        random_state=0,
    )
    # the invariant that defines "hybrid": whatever alpha is, the selection can
    # only contain genes from the global top-2k candidate pool
    ref = adata_for_hvg.copy()
    sc.pp.highly_variable_genes(ref, n_top_genes=ref.n_vars - 1, flavor="seurat_v3")
    pool = set(ref.var["variances_norm"].astype(float).sort_values(ascending=False).index[: 2 * k])
    assert set(_hvg_names(ad_h)) <= pool


def test_mito_ribo_refill_picks_by_variability(adata_for_hvg):
    """Refill after filtering must use variability, not var_names order."""
    ad = adata_for_hvg.copy()
    ad.var_names = [f"RPL{i}" if i < 6 else f"g{i}" for i in range(ad.n_vars)]
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=30,
        balance_method="none",
        filter_ribo=True,
        min_cluster_size=20,
        random_state=0,
    )
    sel = _hvg_names(ad)
    assert not any(s.startswith("RPL") for s in sel)
    vn = ad.var["variances_norm"].astype(float)
    eligible = vn[[not i.startswith("RPL") for i in vn.index]]
    # the selection must be the top-|sel| eligible genes by variability
    assert set(sel) == set(eligible.sort_values(ascending=False).index[: len(sel)])


def test_no_scratch_layer_leaks_for_log_flavors(adata_for_hvg):
    """Log-flavor runs must not leave scratch layers behind in adata.layers."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, flavor="seurat", balance_method="hybrid", min_cluster_size=20
    )
    assert "_scfair_log" not in ad.layers


def test_reweight_excluded_cells_do_not_dominate_sampling():
    """Cells outside a valid cluster must not outweigh cells inside a big one."""
    from scfair.pp._highly_variable_genes import _cell_weights

    labels = pd.Series(["big"] * 1000 + ["tiny"] * 5 + ["mid"] * 200)
    valid = pd.Series({"big": 1000, "mid": 200})
    w = _cell_weights(labels, valid, balance_power=0.5)

    w_big, w_excluded = w[0], w[1000]
    assert w_excluded <= 2.0 * w_big, f"excluded cell weight {w_excluded} vs big-cluster {w_big}"
    # and the size-balancing intent still holds between valid clusters
    assert w[1005] > w_big  # smaller valid cluster keeps more mass per cell
    # the excluded fragment must stay a rounding error in total mass
    assert w[1000:1005].sum() < 0.05


def test_reweight_never_reports_a_missing_placeholder_cluster(adata_for_hvg):
    """'__missing__' is an alignment placeholder and must never become a cluster."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=30, balance_method="reweight", min_cluster_size=20, random_state=0
    )
    assert "__missing__" not in (ad.uns[UNS_KEY]["hvg"].get("cluster_weights") or {})


def test_inplace_false_does_not_touch_the_input(adata_for_hvg):
    """inplace=False must return a DataFrame without mutating the caller's object,
    since var['highly_variable'] would silently change a later sc.pp.pca."""
    ad = adata_for_hvg.copy()
    before = (
        set(ad.layers),
        set(ad.var.columns),
        set(ad.uns),
        set(ad.obs.columns),
    )
    out = scf.pp.highly_variable_genes(
        ad, n_top_genes=20, balance_method="hybrid", min_cluster_size=20, inplace=False
    )
    assert isinstance(out, pd.DataFrame)
    assert (
        set(ad.layers),
        set(ad.var.columns),
        set(ad.uns),
        set(ad.obs.columns),
    ) == before


def test_duplicate_var_names_rejected_with_actionable_message(adata_for_hvg):
    """10x gene symbols are not unique and scanpy tolerates them; scFair aligns by
    name, so it must say so instead of raising from inside pandas."""
    ad = adata_for_hvg.copy()
    names = list(ad.var_names)
    names[3] = names[0]
    ad.var_names = names
    with pytest.raises(ValueError, match="var_names_make_unique"):
        scf.pp.highly_variable_genes(ad, n_top_genes=20)


@pytest.mark.parametrize("lo,hi,msg", [(0, 100, ">= 1"), (10, 0, ">= 1")])
def test_auto_bounds_reject_nonpositive(adata_for_hvg, lo, hi, msg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match=msg):
        scf.pp.highly_variable_genes(ad, n_top_genes="auto", n_top_min=lo, n_top_max=hi)


def test_auto_bounds_reject_min_gt_max(adata_for_hvg):
    """Contradictory [n_top_min, n_top_max] must raise, not silently clamp."""
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="n_top_min=.*> n_top_max"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes="auto",
            n_top_min=300,
            n_top_max=100,
            auto_n_method="elbow",
            balance_method="none",
            diagnose=False,
        )


def test_unknown_strategy_fails_before_any_clustering(adata_for_hvg):
    """Validation must fail before any clustering runs, not after."""
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="Unknown n_top_genes"):
        scf.pp.highly_variable_genes(ad, n_top_genes="elbowww")
    # nothing should have been computed on the object yet
    assert "scfair_hvg_clusters" not in ad.obs


def test_numeric_n_top_genes_from_configs(adata_for_hvg):
    """YAML/JSON hand over 2000.0 or "2000"; treat integral values as ints."""
    for value in (20.0, "20", np.int64(20)):
        ad = adata_for_hvg.copy()
        scf.pp.highly_variable_genes(
            ad, n_top_genes=value, balance_method="none", min_cluster_size=20
        )
        assert int(ad.var["highly_variable"].sum()) == 20


def test_global_score_skips_the_flavor_pass(adata_for_hvg):
    """global_score must replace the flavor pass entirely, not just override its
    output."""
    ad = adata_for_hvg.copy()
    gs = pd.Series(np.linspace(1.0, 0.0, ad.n_vars), index=ad.var_names.astype(str))
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="none", global_score=gs)
    assert ad.uns[UNS_KEY]["hvg"]["global_score"] == "injected"
    # the seurat_v3 pass would have written these; it must not have run
    assert "variances_norm" not in ad.var.columns
    # and the anchor really drove the selection
    assert _hvg_names(ad) == list(gs.sort_values(ascending=False).index[:20])


def test_scale_clustering_default_is_true(adata_for_hvg):
    """Default scale_clustering=True (zero-centre intermediate PCA)."""
    from scfair.pp import HVGOptions

    assert HVGOptions().scale_clustering is True
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0
    )
    assert ad.uns[UNS_KEY]["hvg"]["scale_clustering"] is True


def test_scale_clustering_false_still_runs(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        random_state=0,
        scale_clustering=False,
    )
    assert ad.uns[UNS_KEY]["hvg"]["scale_clustering"] is False
    assert int(ad.var["highly_variable"].sum()) == 40


def test_logfc_space_default_is_bit_identical(adata_for_hvg):
    kw = dict(n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0)
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, **kw)
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(b, logfc_space="log1p", **kw)
    assert _hvg_names(a) == _hvg_names(b)
    assert a.uns[UNS_KEY]["hvg"]["logfc_space"] == "log1p"


@pytest.mark.parametrize("space", ["linear", "linear_regularised"])
def test_logfc_space_alternatives_run_and_are_recorded(adata_for_hvg, space):
    a = adata_for_hvg.copy()
    ctx = (
        pytest.warns(UserWarning, match="benchmark")
        if space == "linear"
        else contextlib.nullcontext()
    )
    with ctx:
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=40,
            balance_method="score",
            logfc_space=space,
            min_cluster_size=20,
            random_state=0,
        )
    assert int(a.var["highly_variable"].sum()) == 40
    assert a.uns[UNS_KEY]["hvg"]["logfc_space"] == space


def test_logfc_space_invalid(adata_for_hvg):
    with pytest.raises(ValueError, match="logfc_space"):
        scf.pp.highly_variable_genes(adata_for_hvg.copy(), n_top_genes=10, logfc_space="lin")


# ---------------------------------------------------------------------------
# post-hybrid allocation: cap / coverage / none
# ---------------------------------------------------------------------------


def test_allocation_method_none_matches_default(adata_for_hvg):
    """allocation_method='none' matches the default product path."""
    ad_a = adata_for_hvg.copy()
    ad_b = adata_for_hvg.copy()
    if "counts" not in ad_a.layers:
        ad_a.layers["counts"] = ad_a.X.copy()
        ad_b.layers["counts"] = ad_b.X.copy()
    kw = dict(
        n_top_genes=20,
        balance_method="hybrid",
        flavor="seurat_v3",
        layer="counts",
        min_cluster_size=20,
        diagnose=False,
        random_state=0,
    )
    scf.pp.highly_variable_genes(ad_a, **kw)
    scf.pp.highly_variable_genes(ad_b, allocation_method="none", **kw)
    assert (ad_a.var["highly_variable"].to_numpy() == ad_b.var["highly_variable"].to_numpy()).all()
    assert ad_a.uns[UNS_KEY]["hvg"]["allocation_method"] == "none"
    assert ad_a.uns[UNS_KEY]["hvg"]["cap_allocation"] is False


def test_allocation_method_default_is_none(adata_for_hvg):
    """Post-hoc cap/coverage stay opt-in; default is hybrid blend only."""
    kw = dict(
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        random_state=0,
        diagnose=False,
    )
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, **kw)
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(b, allocation_method="none", **kw)
    assert _hvg_names(a) == _hvg_names(b)
    assert a.uns[UNS_KEY]["hvg"]["allocation_method"] == "none"
    assert a.uns[UNS_KEY]["hvg"]["cap_allocation"] is False


def test_allocation_method_coverage_runs(adata_for_hvg):
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        allocation_method="coverage",
        random_state=0,
        diagnose=False,
    )
    assert int(a.var["highly_variable"].sum()) == 40
    assert a.uns[UNS_KEY]["hvg"]["allocation_method"] == "coverage"
    assert a.uns[UNS_KEY]["hvg"]["cap_allocation"] is False


def test_adaptive_starved_budget_scales_with_starvation():
    """No starved → 0; few starved → small; many → up to hard cap."""
    from scfair.pp._highly_variable_genes import _adaptive_starved_budget

    b0, m0 = _adaptive_starved_budget(
        n_top=2000,
        n_units=10,
        n_starved=0,
        total_need=100,
        max_frac=0.10,
    )
    assert b0 == 0
    assert m0["budget_mode"] == "zero"

    # 1/10 starved: prevalence = 1/5 = 0.2 → soft = 0.2 * 200 = 40
    # but at least min(n_starved, hard, need) = 1
    b1, m1 = _adaptive_starved_budget(
        n_top=2000,
        n_units=10,
        n_starved=1,
        total_need=80,
        max_frac=0.10,
    )
    assert m1["hard_cap"] == 200
    assert b1 == min(80, 200, m1["soft_cap"])
    assert b1 <= 200
    assert b1 < 200  # not full ceiling for a single starved unit

    # half units starved → full soft = hard
    b_half, m_half = _adaptive_starved_budget(
        n_top=2000,
        n_units=10,
        n_starved=5,
        total_need=500,
        max_frac=0.10,
    )
    assert m_half["hard_cap"] == 200
    assert m_half["soft_cap"] == 200
    assert b_half == 200  # capped by hard, not total_need

    # need smaller than soft → spend only need
    b_need, _ = _adaptive_starved_budget(
        n_top=2000,
        n_units=10,
        n_starved=5,
        total_need=30,
        max_frac=0.10,
    )
    assert b_need == 30


def test_allocation_method_starved_topup_runs(adata_for_hvg):
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        allocation_method="starved_topup",
        random_state=0,
        diagnose=False,
    )
    assert int(a.var["highly_variable"].sum()) == 40
    assert a.uns[UNS_KEY]["hvg"]["allocation_method"] == "starved_topup"
    assert a.uns[UNS_KEY]["hvg"]["cap_allocation"] is False


def test_allocation_method_starved_alias(adata_for_hvg):
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        allocation_method="starved",
        random_state=0,
        diagnose=False,
    )
    assert a.uns[UNS_KEY]["hvg"]["allocation_method"] == "starved_topup"


def test_cap_allocation_removed(adata_for_hvg):
    """allocation_method='cap' / cap_allocation raise (removed 0.2.0)."""
    a = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="removed"):
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=40,
            balance_method="hybrid",
            min_cluster_size=20,
            resolution=0.5,
            allocation_method="cap",
            random_state=0,
            diagnose=False,
        )
    with pytest.raises(ValueError, match="removed"):
        scf.pp.highly_variable_genes(
            a,
            n_top_genes=40,
            balance_method="hybrid",
            min_cluster_size=20,
            resolution=0.5,
            cap_allocation=True,
            random_state=0,
            diagnose=False,
        )


def test_starved_topup_stays_in_hybrid_pool(adata_for_hvg):
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=20,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        allocation_method="starved_topup",
        random_state=0,
        diagnose=False,
    )
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        b,
        n_top_genes=40,
        balance_method="none",
        random_state=0,
        diagnose=False,
    )
    if "highly_variable_rank" in b.var.columns:
        rank = b.var["highly_variable_rank"].astype(float)
        pool = set(rank.nsmallest(40).index.astype(str))
    else:
        pool = set(b.var_names[b.var["highly_variable"]].astype(str))
        # none@40 is not exactly 2×20 hybrid pool; rebuild via variances if needed
        if len(pool) < 40 and "variances_norm" in b.var.columns:
            pool = set(b.var["variances_norm"].astype(float).nlargest(40).index.astype(str))
    sel = set(a.var_names[a.var["highly_variable"]].astype(str))
    # hybrid pool is top 40 global; starved top-up must stay inside
    assert sel <= pool or len(sel - pool) == 0


def test_allocation_method_invalid(adata_for_hvg):
    with pytest.raises(ValueError, match="allocation_method"):
        scf.pp.highly_variable_genes(
            adata_for_hvg.copy(), n_top_genes=10, allocation_method="quota"
        )


def test_spec_on_legitimate_units_runs(adata_for_hvg):
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        spec_on_legitimate_units=True,
        allocation_method="none",
        random_state=0,
        diagnose=False,
    )
    assert int(a.var["highly_variable"].sum()) == 40
    assert a.uns[UNS_KEY]["hvg"]["spec_on_legitimate_units"] is True
    part = a.uns[UNS_KEY]["hvg"]["clustering"].get("spec_partition")
    assert part in ("legitimate_units", "leiden_fallback", "leiden")


def test_coverage_stays_in_hybrid_pool(adata_for_hvg):
    """Coverage top-up must not change the requested selection size."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=20,
        balance_method="hybrid",
        min_cluster_size=20,
        resolution=0.5,
        allocation_method="coverage",
        random_state=0,
        diagnose=False,
    )
    selected = set(a.var_names[a.var["highly_variable"]].astype(str))
    assert len(selected) == 20


# ---------------------------------------------------------------------------
# intermediate-clustering diagnostics (uns['scfair']['hvg']['clustering'])
# ---------------------------------------------------------------------------


def _clustering_diag(a):
    return a.uns[UNS_KEY]["hvg"]["clustering"]


def test_clustering_diagnostics_reported(adata_for_hvg):
    """The intermediate clustering is reported, not silently discarded."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a, n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0
    )
    d = _clustering_diag(a)
    assert d["n_genes_clustered"] == 40
    assert d["n_cells"] == a.n_obs
    assert d["resolution"] == 0.5
    assert d["min_cluster_size"] == 20
    assert d["n_passes"] == 1
    # the reported partition must be the one that was actually used
    assert d["n_clusters_total"] == a.obs["scfair_hvg_clusters"].nunique()
    assert sum(d["cluster_sizes"].values()) == a.n_obs
    assert len(d["pca_variance_ratio"]) == d["n_pcs_used"]
    # kept (passed min_cluster_size) is an upper bound on scored: a kept cluster
    # covering every cell has no "rest" and yields no specificity score
    assert d["n_clusters_kept"] >= a.uns[UNS_KEY]["hvg"]["n_clusters_used"]


def test_clustering_diagnostics_report_effective_not_requested_settings(adata_for_hvg):
    """n_pcs/n_neighbors are clamped internally; the clamped values are what ran.

    Reporting only the request would be misleading exactly when it matters --
    on inputs small enough for the clamp to bite.
    """
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=12,
        balance_method="hybrid",
        min_cluster_size=20,
        n_pcs=50,
        n_neighbors=500,
        random_state=0,
    )
    d = _clustering_diag(a)
    assert d["n_pcs_requested"] == 50
    assert d["n_pcs_used"] == 11  # min(50, n_genes_clustered - 1, n_obs - 1)
    assert d["n_neighbors_requested"] == 500
    assert d["n_neighbors_used"] == a.n_obs - 1


def test_clustering_diagnostics_record_dropped_clusters(adata_for_hvg):
    """Clusters below min_cluster_size are excluded from scoring; say which."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=65,  # the 60-cell group cannot pass
        resolution=0.5,
        random_state=0,
    )
    d = _clustering_diag(a)
    dropped = set(d["clusters_dropped"])
    assert dropped == {c for c, n in d["cluster_sizes"].items() if n < 65}
    assert d["n_clusters_kept"] == d["n_clusters_total"] - len(dropped)


def test_auto_structure_single_hybrid_pass(adata_for_hvg):
    """Default structure auto: structure k then one hybrid (no double intermediate)."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",  # auto_n_method defaults to structure
        n_top_min=10,
        n_top_max=60,
        balance_method="hybrid",
        min_cluster_size=20,
        random_state=0,
    )
    d = _clustering_diag(a)
    assert d["n_passes"] == 1
    auto = a.uns[UNS_KEY]["hvg"]["auto_n"]
    assert auto["strategy"] == "structure"
    assert auto.get("structure_hybrid_fast") is True


def test_auto_ensemble_reuses_intermediate_on_realign(adata_for_hvg):
    """Ensemble auto still realigns, but reuses PCA/Leiden/specificity."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",
        auto_n_method="ensemble",
        n_top_min=10,
        n_top_max=60,
        balance_method="hybrid",
        min_cluster_size=20,
        random_state=0,
        resolution=0.5,  # fixed so protocol matches across passes
    )
    d = _clustering_diag(a)
    # Physical clustering once; realign reuses.
    assert d["n_passes"] == 1
    assert d.get("intermediate_reused") is True
    auto = a.uns[UNS_KEY]["hvg"]["auto_n"]
    assert auto.get("pool_realign") == "hybrid_2xk"
    assert auto.get("intermediate_reused") is True


def test_no_balance_method_reports_no_clustering(adata_for_hvg):
    """balance_method='none' runs no intermediate clustering; claim nothing."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, n_top_genes=40, balance_method="none")
    assert a.uns[UNS_KEY]["hvg"]["clustering"] is None
    assert "scfair_hvg_clusters" not in a.obs


# ---------------------------------------------------------------------------
# cluster_genes (explicit clustering pool; enables iterated selection)
# ---------------------------------------------------------------------------


def test_cluster_genes_clusters_on_the_given_list(adata_for_hvg):
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a, n_top_genes=40, balance_method="hybrid", min_cluster_size=20, random_state=0
    )
    round1 = _hvg_names(a)

    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        b,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=20,
        random_state=0,
        cluster_genes=round1,
    )
    d = _clustering_diag(b)
    assert d["n_genes_clustered"] == len(round1)
    assert b.uns[UNS_KEY]["hvg"]["cluster_pool_source"] == "cluster_genes"
    # candidate pool stays global, so round 2 may return a different set
    assert int(b.var["highly_variable"].sum()) == 40


def test_cluster_genes_equivalent_to_cluster_pool_on_same_genes(adata_for_hvg):
    """The two knobs are the same mechanism; on the same gene set they agree."""
    probe = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(probe, n_top_genes=70, balance_method="none")
    pool_genes = _hvg_names(probe)

    kw = dict(
        n_top_genes=30,
        balance_method="hybrid",
        min_cluster_size=20,
        n_pcs=10,
        random_state=0,
    )
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, cluster_pool=70, **kw)
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(b, cluster_genes=pool_genes, **kw)
    assert _hvg_names(a) == _hvg_names(b)


def test_cluster_genes_rejects_conflicting_and_unmatched_input(adata_for_hvg):
    with pytest.raises(ValueError, match="not both"):
        scf.pp.highly_variable_genes(
            adata_for_hvg.copy(), n_top_genes=40, cluster_pool=70, cluster_genes=["g0"]
        )
    with pytest.raises(ValueError, match="cluster_genes matched only"):
        scf.pp.highly_variable_genes(
            adata_for_hvg.copy(), n_top_genes=40, cluster_genes=["nope", "nah"]
        )


# ---------------------------------------------------------------------------
# consensus_resolutions
# ---------------------------------------------------------------------------


def test_consensus_none_matches_default(adata_for_hvg):
    """Default stays bit-identical: the ladder must be opt-in."""
    kw = dict(
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=10,
        resolution=1.0,
        random_state=0,
    )
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, **kw)
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(b, consensus_resolutions=None, **kw)
    assert _hvg_names(a) == _hvg_names(b)


def test_consensus_single_rung_matches_default(adata_for_hvg):
    """A one-rung ladder at the primary resolution is the degenerate case."""
    kw = dict(
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=10,
        resolution=1.0,
        random_state=0,
    )
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, **kw)
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(b, consensus_resolutions=[1.0], **kw)
    assert _hvg_names(a) == _hvg_names(b)


def test_consensus_keeps_primary_resolution_as_the_reported_partition(adata_for_hvg):
    """obs and the diagnostics must describe `resolution`, not the ladder's head.

    Otherwise obs could carry a partition from a resolution the caller never
    asked for, and if that rung yielded one cluster the whole call would
    silently degenerate to global HVG.
    """
    kw = dict(
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=10,
        resolution=1.0,
        random_state=0,
    )
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, **kw)
    b = adata_for_hvg.copy()
    # ladder deliberately led by a coarser rung, and omitting the primary
    scf.pp.highly_variable_genes(b, consensus_resolutions=[0.2, 2.0], **kw)

    assert b.obs["scfair_hvg_clusters"].nunique() == a.obs["scfair_hvg_clusters"].nunique()
    diag = b.uns[UNS_KEY]["hvg"]["clustering"]
    assert diag["resolution"] == 1.0
    # the primary is inserted at the head, so the ladder actually run is 3 long
    assert diag["consensus_resolutions"][0] == 1.0
    assert len(diag["consensus_n_clusters"]) == 3


def test_consensus_is_recorded_and_scores_change(adata_for_hvg):
    """The ladder must reach the specificity scores, not just the metadata."""
    kw = dict(
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=10,
        resolution=1.0,
        random_state=0,
        # the ladder only moves the specificity term, which carries weight
        # (1 - blend_global); at the 0.95 default it cannot express itself,
        # so this test uses a setting where it can
        blend_global=0.3,
    )
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, **kw)
    b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(b, consensus_resolutions=[0.5, 1.0, 2.0], **kw)

    assert b.uns[UNS_KEY]["hvg"]["consensus_resolutions"] == [0.5, 1.0, 2.0]
    assert a.uns[UNS_KEY]["hvg"]["consensus_resolutions"] is None
    assert not np.allclose(a.var["scfair_score"].to_numpy(), b.var["scfair_score"].to_numpy())


def test_flavor_used_recorded_on_success(adata_for_hvg):
    """Successful seurat_v3 path records flavor_used == requested."""
    ad = adata_for_hvg.copy()
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="none",
        flavor="seurat_v3",
        layer="counts",
        diagnose=False,
    )
    h = ad.uns[UNS_KEY]["hvg"]
    assert h.get("flavor_used") == "seurat_v3"
    assert h.get("flavor_requested") == "seurat_v3"
    assert h.get("fallback_reason") is None


def test_seurat_v3_fallback_warns_and_records(monkeypatch, adata_for_hvg):
    """When seurat_v3 fails, warn + record seurat fallback (strict=False)."""
    import scfair.pp._highly_variable_genes as hvg_mod

    ad = adata_for_hvg.copy()
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    real_once = hvg_mod._run_hvg_once
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if kwargs.get("flavor") in ("seurat_v3", "seurat_v3_paper") and calls["n"] == 1:
            raise RuntimeError("simulated seurat_v3 failure")
        return real_once(*args, **kwargs)

    monkeypatch.setattr(hvg_mod, "_run_hvg_once", boom)
    with pytest.warns(UserWarning, match="falling back to flavor='seurat'"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=15,
            balance_method="none",
            flavor="seurat_v3",
            layer="counts",
            diagnose=False,
            strict=False,
        )
    h = ad.uns[UNS_KEY]["hvg"]
    assert h["flavor_used"] == "seurat"
    assert h["flavor_requested"] == "seurat_v3"
    assert h["fallback_reason"] is not None
    assert "RuntimeError" in str(h["fallback_reason"])


def test_seurat_v3_strict_raises(monkeypatch, adata_for_hvg):
    import scfair.pp._highly_variable_genes as hvg_mod

    ad = adata_for_hvg.copy()
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    def boom(*args, **kwargs):
        if kwargs.get("flavor") in ("seurat_v3", "seurat_v3_paper"):
            raise RuntimeError("simulated seurat_v3 failure")
        raise AssertionError("should not reach other flavors under strict")

    monkeypatch.setattr(hvg_mod, "_run_hvg_once", boom)
    with pytest.raises(RuntimeError, match="strict=True"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=15,
            balance_method="none",
            flavor="seurat_v3",
            layer="counts",
            diagnose=False,
            strict=True,
        )


def test_n_top_genes_1_does_not_unpack_crash(adata_for_hvg):
    """Early cluster abort must return a 4-tuple (cluster_labels, valid, weight_map, X_pca)."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=1,
        balance_method="hybrid",
        diagnose=False,
        random_state=0,
    )
    assert int(ad.var["highly_variable"].sum()) == 1
    assert "scfair_version" in ad.uns[UNS_KEY]["hvg"]


def test_negative_counts_raise_clear_error(adata_for_hvg):
    ad = adata_for_hvg.copy()
    ad.X = ad.X - 1.0
    with pytest.raises(ValueError, match="negative"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="none", diagnose=False)


def test_nan_counts_raise_clear_error(adata_for_hvg):
    ad = adata_for_hvg.copy()
    X = np.asarray(ad.X, dtype=float).copy()
    X[0, 0] = np.nan
    ad.X = X
    with pytest.raises(ValueError, match="non-finite"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="none", diagnose=False)


def test_scfair_version_recorded(adata_for_hvg):
    from scfair import __version__

    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="none", diagnose=False)
    assert ad.uns[UNS_KEY]["hvg"]["scfair_version"] == __version__


def test_hybrid_scfair_score_nan_outside_pool(adata_for_hvg):
    """scfair_score must match the pool-normalized blend used for selection."""
    ad = adata_for_hvg.copy()
    k = 15
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=k,
        balance_method="hybrid",
        min_cluster_size=20,
        diagnose=False,
        random_state=0,
    )
    score = ad.var["scfair_score"]
    sel = set(ad.var_names[ad.var["highly_variable"]].astype(str))
    # Selected genes must have finite scores; outside the 2k pool → NaN.
    assert score.loc[list(sel)].notna().all()
    n_nan = int(score.isna().sum())
    assert n_nan >= 1 or ad.n_vars <= 2 * k
    # Top-k by scfair_score among finite entries should match selected set.
    finite = score.dropna().sort_values(ascending=False)
    top_k = set(finite.index[:k].astype(str))
    assert top_k == sel


def test_cluster_pool_and_genes_mutex_even_for_none(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="either cluster_pool or cluster_genes"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=10,
            balance_method="none",
            cluster_pool=20,
            cluster_genes=["g0"],
            diagnose=False,
        )


def test_default_n_top_genes_is_2000():
    """Product default is fixed 2000 + append, not structure auto / hybrid."""
    import inspect

    sig = inspect.signature(scf.pp.highly_variable_genes)
    assert sig.parameters["n_top_genes"].default == 2000
    assert sig.parameters["balance_method"].default == "append"


def test_duplicate_obs_names_rejected_at_entry(adata_for_hvg):
    ad = adata_for_hvg.copy()
    names = list(ad.obs_names)
    names[1] = names[0]
    ad.obs_names = names
    with pytest.raises(ValueError, match="obs_names"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="hybrid", diagnose=False)


def test_does_not_clobber_scanpy_uns_hvg(adata_for_hvg):
    ad = adata_for_hvg.copy()
    ad.uns["hvg"] = {"flavor": "seurat_v3", "user_key": 42}
    scf.pp.highly_variable_genes(
        ad, n_top_genes=15, balance_method="none", flavor="seurat_v3", diagnose=False
    )
    assert ad.uns["hvg"]["user_key"] == 42
    # flavor reflects the method that actually ran
    assert ad.uns["hvg"]["flavor"] == ad.uns[UNS_KEY]["hvg"]["flavor_used"]


def test_near_identity_k_warns(adata_for_hvg):
    ad = adata_for_hvg.copy()
    # Select ≥80% of genes → near-identity warning
    k = max(10, int(0.85 * ad.n_vars))
    with pytest.warns(UserWarning, match="nearly identity|80%|% of the matrix"):
        scf.pp.highly_variable_genes(ad, n_top_genes=k, balance_method="none", diagnose=False)


def test_structure_auto_n_records_k_source(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes="auto",
        n_top_min=20,
        n_top_max=60,
        balance_method="none",
        diagnose=False,
        random_state=0,
    )
    auto = ad.uns[UNS_KEY]["hvg"]["auto_n"]
    assert auto is not None
    assert auto.get("strategy") == "structure" or "structure" in str(auto.get("strategy"))
    # k_source must be present when structure path ran (not buried only in nested dict)
    if auto.get("strategy") == "structure":
        assert auto.get("k_source") is not None
        assert "depth" not in auto  # dead None placeholders removed
        assert "ensemble" not in auto


def test_starved_topup_skips_when_few_units():
    """n_units<3 is skipped_structure_too_coarse, not checked_no_starved."""
    import numpy as np
    import pandas as pd

    from scfair.pp._highly_variable_genes import _starved_topup_allocate

    sel = [f"g{i}" for i in range(30)]
    score = pd.Series(np.linspace(1.0, 0.0, 30), index=sel)
    ranks = {"0": sel[:15], "1": sel[10:]}
    with pytest.warns(UserWarning, match="skipped"):
        out, diag = _starved_topup_allocate(sel, ranks, score, 30)
    assert out == sel
    assert diag["allocation_status"] == "skipped_structure_too_coarse"
    assert diag["n_units"] == 2
    assert diag["n_starved"] == 0


def test_allocation_method_not_duplicated_on_clustering(adata_for_hvg):
    """allocation_method lives only on top-level hvg meta (request)."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=30,
        balance_method="hybrid",
        allocation_method="starved_topup",
        min_cluster_size=20,
        resolution=0.5,
        diagnose=True,
        random_state=0,
    )
    h = ad.uns[UNS_KEY]["hvg"]
    assert h["allocation_method"] == "starved_topup"
    cl = h.get("clustering") or {}
    assert "allocation_method" not in cl


def test_all_zero_counts_layer_raises_clear_error(adata_for_hvg):
    ad = adata_for_hvg.copy()
    ad.layers["counts"] = np.zeros_like(ad.X)
    # Explicit layer= pins the zero matrix. (Default layer=None would detect
    # fingerprint mismatch vs integer .X and refresh counts — G2.)
    with pytest.raises(ValueError, match="all zeros|degenerate|empty"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=10, balance_method="none", layer="counts", diagnose=False
        )


def test_stale_zero_counts_refreshed_from_integer_X(adata_for_hvg):
    """Zero decoy in layers['counts'] must not stick when .X is real counts."""
    ad = adata_for_hvg.copy()
    ad.layers["counts"] = np.zeros_like(ad.X)
    with pytest.warns(UserWarning, match="differs from the current integer-like"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="none", diagnose=False)
    assert int(ad.var["highly_variable"].sum()) == 10


def test_failed_call_rolls_back_partial_var_mask(adata_for_hvg):
    """A raised error must not leave a misleading highly_variable mask."""
    ad = adata_for_hvg.copy()
    assert "highly_variable" not in ad.var.columns
    with pytest.raises(ValueError, match="cluster_genes"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=15,
            balance_method="hybrid",
            cluster_genes=["not_a_real_gene_xyz"],
            diagnose=False,
        )
    assert "highly_variable" not in ad.var.columns
    assert ad.uns.get("scfair", {}).get("hvg_failed", {}).get("failed") is True


def test_options_dict_raises_typeerror(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(TypeError, match="HVGOptions"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=20, options={"cluster_pool": 40}, diagnose=False
        )


def test_options_conflicts_with_legacy_kwarg(adata_for_hvg):
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="Conflicting research knobs"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=20,
            cluster_pool=40,
            options=HVGOptions(cluster_pool=70),
            diagnose=False,
        )


def test_n_top_max_below_default_min_raises(adata_for_hvg):
    """Lowering only n_top_max below default min=500 is a hard error."""
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="n_top_min=.*> n_top_max"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes="auto",
            options=HVGOptions(n_top_max=30, n_top_min=500, auto_n_method="elbow"),
            balance_method="none",
            diagnose=False,
        )


def test_n_top_bounds_consistent_works(adata_for_hvg):
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes="auto",
        options=HVGOptions(n_top_min=10, n_top_max=30, auto_n_method="elbow"),
        balance_method="none",
        diagnose=False,
    )
    n = int(ad.uns[UNS_KEY]["hvg"]["n_top_genes_used"])
    assert 1 <= n <= 30


def test_strict_must_be_bool(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(TypeError, match="strict must be bool"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=10, balance_method="none", strict="yes", diagnose=False
        )


def test_options_recorded_in_uns(adata_for_hvg):
    """All resolved HVGOptions fields land in uns for reproducibility."""
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    ad.obs["batch"] = ["a"] * (ad.n_obs // 2) + ["b"] * (ad.n_obs - ad.n_obs // 2)
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=15,
        balance_method="none",
        flavor="seurat",
        diagnose=False,
        options=HVGOptions(batch_key="batch", n_bins=40, min_mean=0.05),
    )
    h = ad.uns[UNS_KEY]["hvg"]
    assert h["batch_key"] == "batch"
    assert "options" in h
    assert h["options"]["batch_key"] == "batch"
    assert h["options"]["n_bins"] == 40
    assert h["options"]["min_mean"] == 0.05
    assert h["options"]["n_top_min"] == 500
    assert h["n_top_min"] == 500
    assert h["n_top_max"] == 5000


def test_hvg_base_plus_append_helper():
    from scfair.pp._highly_variable_genes import _hvg_base_plus_append

    scores = pd.Series(
        {f"g{i}": float(100 - i) for i in range(20)},
        dtype=float,
    )
    sel, meta = _hvg_base_plus_append(scores, n_base=5, n_append=3)
    assert sel == ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7"]
    assert meta["n_base"] == 5
    assert meta["n_append_used"] == 3
    assert meta["n_final"] == 8
    assert meta["append_source"] == "secondary_global_hvg"

    sel0, meta0 = _hvg_base_plus_append(scores, n_base=5, n_append=0)
    assert sel0 == ["g0", "g1", "g2", "g3", "g4"]
    assert meta0["n_append_used"] == 0

    # Cap at available genes
    sel_all, meta_all = _hvg_base_plus_append(scores, n_base=18, n_append=10)
    assert len(sel_all) == 20
    assert meta_all["n_append_used"] == 2


def test_append_default_freezes_base_and_extends(adata_for_hvg):
    """Default append: top-k global base is frozen; +m secondary; no clusters."""
    from scfair.pp import HVGOptions

    ad_none = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_none, n_top_genes=20, balance_method="none", diagnose=False)
    base_genes = list(ad_none.var_names[ad_none.var["highly_variable"]])
    assert len(base_genes) == 20

    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=5),
        diagnose=False,
    )
    selected = list(ad.var_names[ad.var["highly_variable"]])
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "append"
    assert meta["selection"] == "hvg_base_plus_secondary_append"
    assert meta["append_budget"] == 5
    assert meta["append"]["n_base"] == 20
    assert meta["append"]["n_append_used"] == 5
    assert meta["n_highly_variable_final"] == 25
    assert len(selected) == 25
    # Frozen base: global top-20 must all be present (order may differ by rank ties)
    assert set(base_genes).issubset(set(selected))
    # No intermediate clustering
    assert "scfair_hvg_clusters" not in ad.obs
    assert meta["n_clusters_used"] == 0
    assert meta["resolution"] is None


def test_append_is_package_default(adata_for_hvg):
    """Bare call (no balance_method) uses append with default budget 200."""
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    # Fixture has 80 genes; override budget so we don't fill the matrix.
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, options=HVGOptions(append_budget=8), diagnose=False
    )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "append"
    assert meta["append_budget"] == 8
    assert int(ad.var["highly_variable"].sum()) == 28
    assert meta["append"]["n_base"] == 20
    assert meta["append"]["n_append_used"] == 8
    # Signature default remains append (options only overrides budget).
    import inspect

    assert (
        inspect.signature(scf.pp.highly_variable_genes).parameters["balance_method"].default
        == "append"
    )


def test_append_budget_zero_matches_none(adata_for_hvg):
    from scfair.pp import HVGOptions

    ad_a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad_a, n_top_genes=25, balance_method="none", diagnose=False)
    ad_b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_b,
        n_top_genes=25,
        balance_method="append",
        options=HVGOptions(append_budget=0),
        diagnose=False,
    )
    assert set(ad_a.var_names[ad_a.var["highly_variable"]]) == set(
        ad_b.var_names[ad_b.var["highly_variable"]]
    )
    assert int(ad_b.var["highly_variable"].sum()) == 25


def test_hybrid_rerank_alias_still_clusters(adata_for_hvg):
    """Legacy hybrid path still runs intermediate clustering."""
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=30,
        balance_method="hybrid_rerank",
        min_cluster_size=20,
        diagnose=False,
        random_state=0,
    )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "hybrid"
    assert "scfair_hvg_clusters" in ad.obs
    assert int(ad.var["highly_variable"].sum()) == 30


def test_append_does_not_flag_insufficient_clusters(adata_for_hvg):
    """Diagnosis must not claim degenerate populations for the no-cluster path."""
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=5),
        diagnose=True,
    )
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert "insufficient_clusters" not in (diag.get("flags") or [])
    assert not any("Fewer than 2 populations" in t for t in (diag.get("tips") or []))


def test_tiny_n_vars_seurat_v3_does_not_segfault():
    """n_vars ≤ loess min must not C-segfault; fall back to seurat (strict=False)."""
    import anndata as anndata_mod

    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(300, 3)).astype(np.float32)
    ad = anndata_mod.AnnData(X=X)
    ad.obs_names = [f"c{i}" for i in range(300)]
    ad.var_names = [f"g{i}" for i in range(3)]
    # Must complete without process death. Fallback → seurat; may select all 3.
    with pytest.warns(UserWarning, match="loess|n_vars|seurat"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=1, balance_method="none", flavor="seurat_v3", diagnose=False
        )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["flavor_used"] == "seurat"
    assert "n_vars_too_small_for_loess" in str(meta.get("fallback_reason") or "")


def test_tiny_n_vars_strict_raises():
    import anndata as anndata_mod

    rng = np.random.default_rng(1)
    X = rng.poisson(1.0, size=(100, 3)).astype(np.float32)
    ad = anndata_mod.AnnData(X=X)
    ad.obs_names = [f"c{i}" for i in range(100)]
    ad.var_names = [f"g{i}" for i in range(3)]
    with pytest.raises(ValueError, match="loess|n_vars|strict"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=1,
            balance_method="none",
            flavor="seurat_v3",
            strict=True,
            diagnose=False,
        )


def test_append_budget_capped_to_n_top(adata_for_hvg):
    """Absolute append_budget must not exceed n_top (panel blow-up guard)."""
    from scfair.pp import HVGOptions

    ad = adata_for_hvg.copy()  # 80 genes
    with pytest.warns(UserWarning, match="append_budget"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=20,
            balance_method="append",
            options=HVGOptions(append_budget=200),
            diagnose=False,
        )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["append_budget"] == 20  # capped to n_top
    assert meta["append_budget_requested"] == 200
    assert meta["append_budget_capped"] is True
    assert int(ad.var["highly_variable"].sum()) == 40  # 20+20


def test_non_integer_counts_warns_and_records(adata_for_hvg):
    """log1p-normalized .X snapshotted as counts must warn + record in uns."""
    ad = adata_for_hvg.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    with pytest.warns(UserWarning, match="integer counts|raw integer"):
        scf.pp.highly_variable_genes(ad, n_top_genes=15, balance_method="none", diagnose=False)
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["counts_integer_like"] is False
    assert meta["counts_warning"] == "non_integer_counts"
