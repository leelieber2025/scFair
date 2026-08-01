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
    assert any("does not predict" in t.lower() for t in d["tips"])


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
    # the imbalance is still reported, with its numbers
    assert any("imbalance: strong" in t for t in d["tips"])
    # ... and no tip promises a gain
    joined = " ".join(d["tips"]).lower()
    assert "largest gains" not in joined
    assert "most useful" not in joined


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
