"""Tests for imbalance / suitability diagnostics."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pytest

import scfair as scf
from scfair._utils import UNS_KEY
from scfair.pp._diagnosis import (
    check_config,
    cluster_size_metrics,
    diagnose_from_labels,
    diagnose_hvg_run,
    recommend_cluster_resolution,
)


@pytest.fixture
def adata_for_hvg():
    """Synthetic counts with three groups (mirrors test_highly_variable_genes)."""
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


def test_cluster_size_metrics_strong_imbalance():
    # Cao-like: one large majority + long tail
    m = cluster_size_metrics({"maj": 3000, "a": 200, "b": 100, "c": 50, "d": 30})
    assert m["imbalance"] == "strong"
    assert m["max_frac"] > 0.6
    assert m["n_rare_clusters"] >= 1
    assert m["max_min_ratio"] > 15


def test_cluster_size_metrics_balanced():
    m = cluster_size_metrics({"a": 100, "b": 100, "c": 95, "d": 105})
    assert m["imbalance"] == "balanced"
    assert m["max_frac"] < 0.45
    assert m["max_min_ratio"] < 5
    assert m["shannon_evenness"] > 0.85


def test_cluster_size_metrics_degenerate():
    m = cluster_size_metrics({"only": 500})
    assert m["imbalance"] == "degenerate"
    m2 = cluster_size_metrics(None)
    assert m2["imbalance"] == "unknown"


def test_check_config_warns_low_blend_global_for_deprivation():
    """Lowering blend_global worsens, not helps, equal-share starvation."""
    ok = check_config(
        balance_method="hybrid",
        blend_global=0.95,
        log=False,
    )
    assert "blend_global_low_vs_deprivation" not in ok["flags"]

    low = check_config(
        balance_method="hybrid",
        blend_global=0.85,
        log=False,
    )
    assert "blend_global_low_vs_deprivation" in low["flags"]
    assert any("deprivation" in t.lower() or "starv" in t.lower() for t in low["tips"])
    assert any("0.95" in t for t in low["tips"])

    # non-hybrid: no blend tip
    none = check_config(
        balance_method="none",
        blend_global=0.5,
        log=False,
    )
    assert "blend_global_low_vs_deprivation" not in none["flags"]


def test_diagnose_from_labels_describes_imbalance_without_forecasting():
    """Imbalance is reported; it must not be turned into a recommendation.

    Size imbalance does not correlate with measured benefit, and the
    (weak, non-significant) correlation flips sign depending on how the
    margin is measured. Recommending `hybrid` from imbalance would assert
    a relationship the data doesn't support.
    """
    labels = ["T"] * 800 + ["B"] * 100 + ["Mono"] * 80 + ["DC"] * 20
    d = diagnose_from_labels(labels)
    assert d["imbalance"] == "strong"
    assert d["recommendation"] == "keep_current"
    assert d["benefit_evidence"] == "not_predictable"
    assert d["known_no_gain_regime"] is False
    assert any("does not say" in t.lower() or "does not predict" in t.lower() for t in d["tips"])


def test_diagnose_from_labels_balanced_is_also_not_a_forecast():
    """Balanced data is likewise not evidence that scFair will underperform."""
    labels = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50)
    d = diagnose_from_labels(labels)
    assert d["imbalance"] == "balanced"
    assert d["recommendation"] == "keep_current"
    assert d["known_no_gain_regime"] is False


def test_diagnose_from_labels_single_population_is_decidable():
    """The one case labels alone *can* settle: no "rest" to contrast against."""
    d = diagnose_from_labels(["only"] * 300)
    assert d["recommendation"] == "use_scanpy_or_none"
    assert d["known_no_gain_regime"] is True


def test_diagnose_hvg_run_insufficient_clusters():
    d = diagnose_hvg_run(
        balance_method="hybrid",
        n_top_genes_used=2000,
        clustering={
            "n_clusters_kept": 1,
            "cluster_sizes": {"0": 1000},
            "clusters_dropped": [],
        },
        n_clusters_used=1,
        log=False,
    )
    assert d["benefit_evidence"] == "none"
    assert d["known_no_gain_regime"] is True
    assert d["recommendation"] == "use_scanpy_or_none"
    assert "insufficient_clusters" in d["flags"]


def test_diagnose_hvg_run_k_ge_3000():
    d = diagnose_hvg_run(
        balance_method="hybrid",
        n_top_genes_used=3000,
        clustering={
            "n_clusters_kept": 5,
            "cluster_sizes": {"0": 500, "1": 200, "2": 100, "3": 50, "4": 30},
            "clusters_dropped": [],
        },
        log=False,
    )
    assert "k_ge_3000" in d["flags"]
    assert d["recommendation"] == "use_scanpy_or_none"


def _cao_like_diagnosis(**overrides):
    kw = dict(
        balance_method="hybrid",
        n_top_genes_used=2000,
        resolution=0.5,
        neighbor_contrast=0.0,
        clustering={
            "n_clusters_kept": 5,
            "n_clusters_total": 5,
            "cluster_sizes": {"0": 3000, "1": 200, "2": 100, "3": 50, "4": 40},
            "clusters_dropped": [],
        },
        log=False,
    )
    kw.update(overrides)
    return diagnose_hvg_run(**kw)


def test_strong_imbalance_is_described_not_graded():
    """A strongly imbalanced multi-type atlas still gets no benefit grade.

    The most imbalanced datasets in evaluation are not reliably the ones
    that gain the most, and some large multi-type atlases lose to plain
    HVG outright -- a grade here would be wrong as well as unsupported.
    """
    d = _cao_like_diagnosis()
    assert d["imbalance"] == "strong"
    assert d["benefit_evidence"] == "not_predictable"
    assert d["recommendation"] == "keep_current"
    assert d["known_no_gain_regime"] is False
    # strong imbalance still gets a short user-facing tip
    assert any("imbalance" in t.lower() or "largest" in t.lower() for t in d["tips"])
    # ... and no tip promises a gain
    joined = " ".join(d["tips"]).lower()
    assert "largest gains" not in joined
    assert "most useful" not in joined
    assert "seurat_v4" not in joined


def test_rare_tail_advice_is_conditional_not_a_recommendation():
    """A size tail cannot tell us the rare type is *adjacent* to a common one.

    The explanatory tip text was dropped 2026-08-01 (too verbose for routine
    output) -- the flag stays for programmatic use, but no prose about it
    should appear (and it must not leak into the machine-readable field
    disguised as a firm recommendation).
    """
    d = _cao_like_diagnosis()
    assert "rare_tail_no_neighbor_contrast" in d["flags"]
    assert not any("neighbor_contrast=1.0" in t for t in d["tips"])
    assert d["recommendation"] != "use_adjacent_rare_config"


def test_config_conflict_is_flagged_because_it_was_measured():
    """neighbor_contrast + low resolution measured worse than either alone."""
    d = _cao_like_diagnosis(neighbor_contrast=1.0, resolution=0.5)
    assert "nc_low_resolution" in d["flags"]
    assert d["benefit_evidence"] == "config_conflict"
    assert d["recommendation"] == "check_config"


def test_hvg_writes_diagnosis(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=40, balance_method="hybrid", min_cluster_size=20, diagnose=True
    )
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert "imbalance" in diag
    assert "recommendation" in diag
    assert "tips" in diag
    assert "metrics" in diag
    assert diag["balance_method"] == "hybrid"


def test_hvg_diagnose_false_skips(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=40, balance_method="hybrid", min_cluster_size=20, diagnose=False
    )
    assert "diagnosis" not in ad.uns[UNS_KEY]["hvg"]


def test_hvg_diagnosis_logs_on_none_path(adata_for_hvg, caplog):
    ad = adata_for_hvg.copy()
    with caplog.at_level(logging.INFO, logger="scfair.pp._diagnosis"):
        scf.pp.highly_variable_genes(ad, n_top_genes=40, balance_method="none", diagnose=True)
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert diag["recommendation"] == "use_scanpy_or_none"
    assert any("diagnosis" in r.message for r in caplog.records)


def test_none_path_does_not_call_the_data_degenerate(adata_for_hvg):
    """balance_method='none' runs no clustering, so it has no finding about populations.

    Reporting "fewer than 2 populations in intermediate_clusters" there is a
    category error -- and it was emitted at WARNING, telling a user who
    deliberately chose the scanpy path that their data is degenerate.
    """
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=40, balance_method="none")
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert diag["source"] == "config_only"
    assert diag["imbalance"] == "unknown"
    assert not any("Fewer than 2 populations" in t for t in diag["tips"])


def test_ordinary_call_does_not_warn(adata_for_hvg, caplog):
    """ "We cannot predict the margin" is not a warning about the user's data.

    Warning on every ordinary call trains people to ignore the channel.
    """
    ad = adata_for_hvg.copy()
    with caplog.at_level(logging.INFO, logger="scfair.pp._diagnosis"):
        # resolution=1.0 so the fixture actually splits; at the default 0.5 it
        # collapses to a single cluster, which is a genuine no-gain regime and
        # *should* warn
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=40,
            balance_method="hybrid",
            min_cluster_size=10,
            resolution=1.0,
        )
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert diag["benefit_evidence"] == "not_predictable"
    assert diag["n_clusters_kept"] >= 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_diagnose_from_labels_public_export():
    assert callable(scf.diagnose_from_labels)
    assert callable(scf.pp.diagnose_from_labels)


def test_diagnosis_points_at_the_bigger_lever(adata_for_hvg):
    """`not_predictable` is still tracked as a structured field.

    The explanatory "no measured basis" / "resolution matters more" tip text
    was dropped 2026-08-01 (too verbose for routine output); `benefit_evidence`
    still carries the finding programmatically.
    """
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="hybrid",
        min_cluster_size=10,
        resolution=1.0,
    )
    diag = a.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert diag["benefit_evidence"] == "not_predictable"
    assert not any("13x" in t for t in diag["tips"])  # retracted claim, never reintroduced


def test_no_gain_regime_does_not_get_the_tuning_advice(adata_for_hvg):
    """When we already know the call gains nothing, lead with that, not advice."""
    a = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(a, n_top_genes=40, balance_method="none")
    tips = a.uns[UNS_KEY]["hvg"]["diagnosis"]["tips"]
    assert not any("Sweep it" in t for t in tips)


def test_structure_auto_tips_short_k_user_friendly():
    from scfair.pp._diagnosis import _structure_auto_tips

    tips, flags = _structure_auto_tips(
        k=500,
        structure_meta={
            "rule_branch": "short_hard_vm0.70_nd12",
            "k_source": "unanimous_seed_vote",
            "rule_explain": {
                "rule_branch": "short_hard_vm0.70_nd12",
                "n_density_pops": 14,
                "n_leiden": 19,
                "ratio": 1.36,
                "valley_median": 0.71,
            },
        },
    )
    assert "structure_short_k" in flags
    assert len(tips) == 1
    blob = tips[0].lower()
    assert "500" in blob
    assert "n_top_genes=2000" in tips[0]
    # no internal jargon dumps
    assert "nd=" not in blob
    assert "valley" not in blob
    assert "v7" not in blob
    assert "branch" not in blob


def test_structure_auto_tips_v7_band_floor():
    from scfair.pp._diagnosis import _structure_auto_tips

    tips, flags = _structure_auto_tips(
        k=2000,
        structure_meta={
            "rule_branch": "v7_fine_atlas_band",
            "rule_explain": {
                "rule_branch": "v7_fine_atlas_band",
                "n_density_pops": 17,
                "n_leiden": 16,
                "ratio": 0.94,
                "valley_median": 0.80,
                "v7_band_eligible": True,
            },
        },
    )
    assert "structure_v7_band_floor" in flags
    assert "downstream_fine_resolution" in flags
    assert len(tips) == 1
    assert "1.5" in tips[0]
    assert "2000" in tips[0]
    assert not any("seurat_v4" in t.lower() for t in tips)
    assert not any("nd=" in t for t in tips)
    assert not any("append_budget" in t for t in tips)


def test_structure_auto_tips_soft_buffer_short():
    """Soft-buffer is normal product behaviour — flag only, no user tip."""
    from scfair.pp._diagnosis import _structure_auto_tips

    tips, flags = _structure_auto_tips(
        k=1000,
        structure_meta={
            "rule_branch": "short_hard_vm0.70_nd12+k_buffer:500→1000",
            "k_buffer_raw": 500,
            "rule_explain": {
                "rule_branch": "short_hard_vm0.70_nd12+k_buffer:500→1000",
                "k_buffer_raw": 500,
            },
        },
    )
    assert "structure_k_buffer" in flags
    # No tip about 500→1000 / soft-buffer — confuses users; done line shows k.
    assert tips == []


def test_resolve_hvg_mode_auto():
    from scfair.pp._diagnosis import resolve_hvg_mode

    assert resolve_hvg_mode(mode="auto", n_types=31)["mode"] == "fine"
    assert resolve_hvg_mode(mode="auto", n_types=31)["cluster_resolution"] == 1.5
    assert resolve_hvg_mode(mode="auto", n_types=31)["allow_short_soft_buffer"] is False
    assert resolve_hvg_mode(mode="auto", n_types=8)["mode"] == "balanced"
    assert (
        resolve_hvg_mode(mode="auto", rule_branch="short_hard_vm0.80_nd6+k_buffer:500→1000")["mode"]
        == "compact"
    )
    # High density-pop count alone must NOT force fine (would re-floor 1000→2000).
    assert (
        resolve_hvg_mode(
            mode="auto",
            n_density_pops=17,
            rule_branch="short_hard_vm0.70_nd12+k_buffer:500→1000",
        )["mode"]
        == "compact"
    )
    assert (
        resolve_hvg_mode(mode="auto", n_density_pops=17, n_obs=20_000)["mode"] != "fine"
        or resolve_hvg_mode(mode="auto", n_density_pops=17, n_obs=20_000)["mode"] == "balanced"
    )
    # bare nd without short → balanced, not fine
    assert resolve_hvg_mode(mode="auto", n_density_pops=17)["mode"] == "balanced"
    assert resolve_hvg_mode(mode="fine")["mode"] == "fine"


def test_recommend_cluster_resolution_tiers():
    from scfair.pp._diagnosis import resolve_cluster_resolution

    coarse = recommend_cluster_resolution(n_types=8)
    assert coarse["tier"] == "coarse"
    assert coarse["resolution"] == 0.8

    fine = recommend_cluster_resolution(n_types=31)
    assert fine["tier"] == "fine"
    assert fine["resolution"] == 1.5
    assert 1.5 in fine["resolution_sweep"]

    fine_st = recommend_cluster_resolution(rule_branch="v7_fine_atlas_band")
    assert fine_st["tier"] == "fine"

    fine_nd = recommend_cluster_resolution(n_density_pops=18)
    assert fine_nd["tier"] == "fine"

    # auto switch
    r = resolve_cluster_resolution(resolution="auto", n_types=31)
    assert r["auto"] is True and r["resolution"] == 1.5 and r["tier"] == "fine"
    r2 = resolve_cluster_resolution(resolution="auto", n_types=8)
    assert r2["resolution"] == 0.8 and r2["tier"] == "coarse"
    r3 = resolve_cluster_resolution(resolution=2.0)
    assert r3["tier"] == "manual" and r3["resolution"] == 2.0 and r3["auto"] is False


def test_diagnose_from_labels_fine_downstream():
    # 20 equal types → fine tier tip
    labels = [f"t{i}" for i in range(20) for _ in range(10)]
    d = diagnose_from_labels(labels)
    assert d["downstream_clustering"]["tier"] == "fine"
    assert d["downstream_clustering"]["resolution"] == 1.5
    assert any("1.5" in t for t in d["tips"])


def test_diagnose_hvg_run_structure_meta_short():
    # append: no intermediate clustering required to surface the short-k tip
    diag = diagnose_hvg_run(
        balance_method="append",
        n_top_genes_used=500,
        n_top_is_auto=True,
        auto_n_strategy="structure",
        structure_meta={
            "rule_branch": "short_hard_vm0.80_nd6",
            "features": {"n_density_pops": 8, "n_leiden": 10, "valley_median": 0.85},
            "rule_explain": {
                "rule_branch": "short_hard_vm0.80_nd6",
                "n_density_pops": 8,
                "n_leiden": 10,
                "ratio": 1.25,
                "valley_median": 0.85,
                "v7_band_miss": ["nd_in_band", "ratio_ok"],
            },
        },
        log=False,
    )
    assert "structure_short_k" in diag["flags"]
    assert any("2000" in t for t in diag["tips"])
    assert len(diag["tips"]) <= 2


def test_diagnosis_coarse_partition_and_fallback_not_keep_current():
    """Collapsed intermediate partition must not recommend keep_current."""
    from scfair.pp._diagnosis import diagnose_hvg_run

    diag = diagnose_hvg_run(
        balance_method="hybrid",
        n_top_genes_used=200,
        clustering={
            "n_clusters_total": 2,
            "n_clusters_kept": 2,
            "cluster_sizes": {"0": 1000, "1": 1000},
            "clusters_dropped": [],
            "resolution": 0.5,
            "resolution_source": "fallback",
            "granularity": {"reason": "no_embedding"},
            "min_cluster_frac": 0.5,
            "starved_topup_allocation_status": "skipped_structure_too_coarse",
            "starved_topup_n_units": 2,
        },
        n_clusters_used=2,
        log=False,
    )
    assert "coarse_partition" in diag["flags"]
    assert "resolution_fallback" in diag["flags"]
    assert "allocation_skipped_coarse" in diag["flags"]
    assert diag["recommendation"] == "raise_resolution"
    assert diag["benefit_evidence"] == "structure_unreliable"
    assert diag["imbalance_source"] == "intermediate_clusters_unreliable"
    blob = " ".join(diag["tips"]).lower()
    assert "too coarse" in blob or "fallback" in blob or "only 2" in blob
    assert len(diag["tips"]) <= 2
