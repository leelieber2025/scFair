"""Tests for automatic n_top_genes selection."""

from __future__ import annotations

import numpy as np
import pytest

from scfair.pp._auto_n import (
    depth_aware_auto_knobs,
    effective_k_ceiling,
    effective_k_floor,
    estimate_n_top_structure,
    explain_structure_rule,
    resolve_n_top_genes,
    select_n_top_coverage,
    select_n_top_cumfrac,
    select_n_top_elbow,
    select_n_top_ensemble,
    select_n_top_ensemble_detail,
    select_n_top_from_structure,
    select_n_top_knee,
)


def _fake_scores(n: int = 5000, decay: float = 0.001) -> np.ndarray:
    # smoothly decreasing positive scores
    x = np.arange(n, dtype=float)
    return np.exp(-decay * x) * 10.0 + 0.01


@pytest.mark.parametrize("fn", [select_n_top_elbow, select_n_top_knee])
def test_shape_strategies_respect_bounds(fn):
    """Both shape strategies must clamp into [k_min, k_max]."""
    assert 500 <= fn(_fake_scores(), k_min=500, k_max=5000) <= 5000


def test_cumfrac_monotonic():
    s = _fake_scores()
    k7 = select_n_top_cumfrac(s, frac=0.7, k_min=100, k_max=5000)
    k9 = select_n_top_cumfrac(s, frac=0.9, k_min=100, k_max=5000)
    assert k7 <= k9


def test_structure_auto_n_duo_like_long():
    """Few density cores + shallow valleys → long shortlist."""
    k = select_n_top_from_structure(
        valley_median=0.03,
        frac_shallow=1.0,
        n_density_pops=3,
        mean_stability=0.85,
        min_stability=0.5,
    )
    assert k >= 3000


def test_structure_long_branch_is_continuous_in_stability():
    """LONG no longer jumps 1000 genes at a hard ms=0.8 threshold."""
    base = dict(
        valley_median=0.03,
        frac_shallow=1.0,
        n_density_pops=3,
        min_stability=0.5,
        k_max=5000,
    )
    k_lo = select_n_top_from_structure(mean_stability=0.55, **base)
    k_mid = select_n_top_from_structure(mean_stability=0.75, **base)
    k_hi = select_n_top_from_structure(mean_stability=0.95, **base)
    assert k_lo <= k_mid <= k_hi
    # Continuous span should not be a single cliff of 1000 between mid and high.
    assert (k_hi - k_lo) >= 500
    assert k_hi <= 5000


def test_structure_k_max_reaches_5000():
    """Documented n_top_max=5000 must not be silently capped at 4000."""
    k = select_n_top_from_structure(
        valley_median=0.03,
        frac_shallow=1.0,
        n_density_pops=3,
        mean_stability=0.95,
        min_stability=0.5,
        k_max=5000,
        n_genes=20_000,
    )
    assert k > 4000
    assert k <= 5000


def test_structure_auto_n_pancreas_like_short():
    """Many deep density cores → SHORT rule, soft buffer lifts 500→1000."""
    k = select_n_top_from_structure(
        valley_median=0.81,
        frac_shallow=0.1,
        n_density_pops=14,
        mean_stability=0.73,
        min_stability=0.0,
        # n_obs omitted / small: anti-SHORT does not fire; buffer still lifts
        n_obs=3_000,
    )
    assert k == 1000


def test_structure_false_short_floors_after_buffer():
    """Zheng-like short_hard + large n + low conf + nd≤8 → floor 2000.

    Soft buffer alone would leave k=1000 and formerly skipped residual
    anti-SHORT (which only checked k≤500). False-SHORT must fire after buffer.
    Covers both the original nd=6 calibration and retest nd=7 Zheng-20k.
    """
    from scfair.pp._auto_n import explain_structure_rule

    for nd in (6, 7, 8):
        d = explain_structure_rule(
            valley_median=0.83,
            frac_shallow=0.1,
            n_density_pops=nd,
            mean_stability=0.7,
            min_stability=0.1,
            n_obs=20_000,
            n_leiden=8,
            density_confidence="low",
        )
        assert d["short_blocked"] is True, nd
        assert d["n_top"] == 2000, nd
        assert d["short_block_reason"] == "false_short_nd_low", nd
        assert "antishort:false_short_nd_low" in d["rule_branch"], nd
        assert "k_buffer:500→1000" in d["rule_branch"], nd

    k = select_n_top_from_structure(
        valley_median=0.83,
        frac_shallow=0.1,
        n_density_pops=7,  # GOLD-15 retest Zheng landed here
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=20_000,
        n_leiden=8,
        density_confidence="low",
    )
    assert k == 2000


def test_structure_soft_buffer_when_not_false_short():
    """short_hard with nd just above false-SHORT max still soft-buffers 500→1000."""
    from scfair.pp._auto_n import explain_structure_rule

    d = explain_structure_rule(
        valley_median=0.83,
        frac_shallow=0.1,
        n_density_pops=9,  # FALSE_SHORT_ND_MAX=8 → not false SHORT
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=20_000,
        n_leiden=10,
        density_confidence="low",
    )
    assert d["short_blocked"] is False
    assert d["n_top"] == 1000
    assert d["k_buffer_raw"] == 500
    assert "k_buffer:500→1000" in d["rule_branch"]
    assert "antishort:false_short" not in d["rule_branch"]


def test_structure_true_short_skips_buffer_with_n_types():
    """Multi-core short_hard + n_types≥5: keep k=500 (no soft buffer)."""
    from scfair.pp._auto_n import explain_structure_rule

    # SLN-like: high nd, multi-type labels
    d = explain_structure_rule(
        valley_median=0.92,
        frac_shallow=0.1,
        n_density_pops=12,
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=15_000,
        n_leiden=14,
        density_confidence="low",
        n_types=9,
    )
    assert d["n_top"] == 500
    assert d["no_buffer"] is True
    assert "no_buffer:nd12_ntypes9" in d["rule_branch"]
    assert d["k_buffer_raw"] is None
    # Without n_types: soft buffer remains (protect 2-way boards)
    d2 = explain_structure_rule(
        valley_median=0.92,
        frac_shallow=0.1,
        n_density_pops=12,
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=15_000,
        n_leiden=14,
        density_confidence="low",
    )
    assert d2["n_top"] == 1000
    assert "k_buffer:500→1000" in d2["rule_branch"]
    # n_types=2 (TM brain-like): still buffer
    d3 = explain_structure_rule(
        valley_median=0.79,
        frac_shallow=0.1,
        n_density_pops=12,
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=10_000,
        n_leiden=14,
        density_confidence="low",
        n_types=2,
    )
    assert d3["n_top"] == 1000
    assert "k_buffer" in d3["rule_branch"]


def test_structure_anti_short_residual_and_false_short_post_combine():
    """Post-combine floor: false_short on buffered 1000; residual on raw 500."""
    from scfair.pp._auto_n import _apply_short_floor_if_needed

    # Buffered false-SHORT (k=1000, nd=7 Zheng-like, large n, low conf) → 2000
    k1, src1, tag1 = _apply_short_floor_if_needed(
        k=1000,
        k_source="unanimous_seed_vote",
        n_obs=20_000,
        n_density_pops=7,
        density_confidence="low",
        density_depth_sensitivity=3,
        k_min=500,
        k_max=5000,
        n_genes=20_000,
    )
    assert k1 == 2000
    assert tag1 is not None and "false_short" in tag1

    # Raw 500 under false-SHORT geometry → same 2000 floor (via false_short path)
    k2, src2, tag2 = _apply_short_floor_if_needed(
        k=500,
        k_source="unanimous_seed_vote",
        n_obs=20_000,
        n_density_pops=6,
        density_confidence="low",
        density_depth_sensitivity=3,
        k_min=500,
        k_max=5000,
        n_genes=20_000,
    )
    assert k2 == 2000
    assert tag2 is not None and "false_short" in tag2

    # nd=9 is above FALSE_SHORT_ND_MAX=8 → no post-combine floor
    k2b, _, tag2b = _apply_short_floor_if_needed(
        k=1000,
        k_source="unanimous_seed_vote",
        n_obs=20_000,
        n_density_pops=9,
        density_confidence="low",
        density_depth_sensitivity=3,
        k_min=500,
        k_max=5000,
        n_genes=20_000,
    )
    assert k2b == 1000
    assert tag2b is None

    # High-nd raw 500: conf alone must not floor (true multi-core SHORT)
    k3, src3, tag3 = _apply_short_floor_if_needed(
        k=500,
        k_source="unanimous_seed_vote",
        n_obs=15_000,
        n_density_pops=12,
        density_confidence="low",
        density_depth_sensitivity=4,
        k_min=500,
        k_max=5000,
        n_genes=20_000,
    )
    assert k3 == 500
    assert tag3 is None


def test_structure_auto_n_adt_like_mid():
    """Mid rule 1500 is soft-buffered to 2000."""
    k = select_n_top_from_structure(
        valley_median=0.77,
        frac_shallow=0.05,
        n_density_pops=10,
        mean_stability=0.69,
        min_stability=0.12,
    )
    assert k == 2000


def test_structure_mid_unstable_bumps_to_2000():
    """Mid band + low pair-stability already at 2000; buffer is a no-op."""
    k = select_n_top_from_structure(
        valley_median=0.71,
        frac_shallow=0.21,
        n_density_pops=7,
        mean_stability=0.52,
        min_stability=-0.10,
    )
    assert k == 2000
    # stable mid: rule 1500 → buffer 2000
    assert (
        select_n_top_from_structure(
            valley_median=0.77,
            frac_shallow=0.05,
            n_density_pops=10,
            mean_stability=0.69,
            min_stability=0.12,
        )
        == 2000
    )


def test_vote_structure_k_mode_and_tie():
    from scfair.pp._auto_n import _vote_structure_k

    assert _vote_structure_k([500, 500, 2000]) == 500
    # tie 500/2000 → median
    assert _vote_structure_k([500, 2000]) in (500, 2000, 1250)
    assert _vote_structure_k([1000, 1000, 1000]) == 1000


def test_combine_structure_k_prefers_aggregate_not_fragile_vote():
    """Mode vote all-500 must not override density-surplus aggregate → 2000."""
    from scfair.pp._auto_n import _combine_structure_k

    # unanimous SHORT seeds, large n, aggregate mid/long
    k, src = _combine_structure_k(
        k_from_agg=2000,
        k_vote=500,
        per_seed_k=[500, 500, 500],
        n_obs=20_000,
    )
    assert k == 2000
    assert src == "anti_short_veto_large_n"

    # small n + unanimous short → allow short (combine itself does not buffer;
    # buffer is applied in explain / post-combine floor helpers)
    k2, src2 = _combine_structure_k(
        k_from_agg=2000,
        k_vote=500,
        per_seed_k=[500, 500, 500],
        n_obs=2_000,
    )
    assert k2 == 500
    assert src2 == "unanimous_seed_vote"

    # disagreeing seeds → aggregate primary
    k3, src3 = _combine_structure_k(
        k_from_agg=1500,
        k_vote=500,
        per_seed_k=[500, 1500, 1500],
        n_obs=6_000,
    )
    assert k3 == 1500
    assert src3 == "aggregated_features"


def test_structure_auto_n_default_fallback():
    k = select_n_top_from_structure(
        valley_median=0.4,
        frac_shallow=0.4,
        n_density_pops=5,
    )
    assert k == 2000


def test_structure_v5_large_atlas_floor():
    """Many deep cores + large n_obs must not SHORT-hard."""
    kwargs = dict(
        valley_median=0.80,
        frac_shallow=0.01,
        n_density_pops=18,
        mean_stability=0.64,
        min_stability=0.04,
        n_obs=20_000,
    )
    # v4 short rule + soft buffer → 1000
    assert select_n_top_from_structure(**kwargs, version="v4") == 1000
    assert select_n_top_from_structure(**kwargs, version="v5") == 2000


def test_structure_v5_sln_like_still_short():
    """Large n_obs but only ~12 density pops: short rule + buffer → 1000."""
    k = select_n_top_from_structure(
        valley_median=0.92,
        frac_shallow=0.05,
        n_density_pops=12,
        mean_stability=0.58,
        min_stability=-0.03,
        n_obs=15_820,
        version="v5",
    )
    assert k == 1000


def test_structure_v5_without_n_obs_matches_v4():
    """Atlas guard requires n_obs; omit → v5 behaves like v4 (both buffered)."""
    kwargs = dict(
        valley_median=0.80,
        frac_shallow=0.01,
        n_density_pops=18,
        mean_stability=0.64,
        min_stability=0.04,
    )
    assert select_n_top_from_structure(**kwargs, version="v5") == 1000
    assert select_n_top_from_structure(**kwargs, version="v4") == 1000


def test_structure_v6_seurat_density_surplus_floor():
    """Density finds more cores than Leiden on large data → floor."""
    kwargs = dict(
        valley_median=0.80,
        frac_shallow=0.01,
        n_density_pops=18,
        n_leiden=15,  # leid/dens < 1
        mean_stability=0.64,
        min_stability=0.04,
        n_obs=20_000,
    )
    assert select_n_top_from_structure(**kwargs, version="v4") == 1000
    assert select_n_top_from_structure(**kwargs, version="v6") == 2000
    assert select_n_top_from_structure(**kwargs, version="v7") == 2000


def test_structure_v6_lung_like_still_short():
    """Leiden ≥ density on large multi-core → SHORT rule + soft buffer → 1000."""
    kwargs = dict(
        valley_median=0.79,
        frac_shallow=0.12,
        n_density_pops=24,
        n_leiden=28,  # leid/dens > 1; nd above v7 band
        mean_stability=0.71,
        min_stability=-0.01,
        n_obs=15_000,
    )
    # v5 over-guards to ~100*n_pops floor (≥2000), not short
    assert select_n_top_from_structure(**kwargs, version="v5") >= 2000
    assert select_n_top_from_structure(**kwargs, version="v6") == 1000
    assert select_n_top_from_structure(**kwargs, version="v7") == 1000
    assert select_n_top_from_structure(**kwargs, version="v4") == 1000


def test_structure_v7_seurat_near_parity_still_floors():
    """v7: near-parity band floors even when ratio slightly > 1 (v6 would SHORT)."""
    kwargs = dict(
        valley_median=0.793,
        frac_shallow=0.02,
        n_density_pops=17,
        n_leiden=19,  # ratio ≈ 1.12 — fails v6 surplus, hits v7 band
        mean_stability=0.64,
        min_stability=0.02,
        n_obs=20_000,
    )
    assert select_n_top_from_structure(**kwargs, version="v6") == 1000
    assert select_n_top_from_structure(**kwargs, version="v7") == 2000
    # default version is v7
    assert select_n_top_from_structure(**kwargs) == 2000


def test_structure_v7_sln_like_still_short():
    """Large n but ratio too high for v7 band → SHORT."""
    kwargs = dict(
        valley_median=0.927,
        frac_shallow=0.05,
        n_density_pops=12,
        n_leiden=15,  # ratio 1.25 > 1.12
        mean_stability=0.58,
        min_stability=0.1,
        n_obs=15_820,
    )
    assert select_n_top_from_structure(**kwargs, version="v7") == 1000


def test_structure_soft_1000_buffers_to_1500():
    """soft_1000_vm rule pick is buffered one rung to 1500."""
    from scfair.pp._auto_n import explain_structure_rule

    d = explain_structure_rule(
        valley_median=0.66,
        frac_shallow=0.3,
        n_density_pops=5,
        mean_stability=0.6,
        min_stability=0.3,
        n_obs=4_000,
    )
    assert d["k_buffer_raw"] == 1000
    assert d["n_top"] == 1500
    assert "soft_1000" in d["rule_branch"]
    assert "k_buffer" in d["rule_branch"]


def test_resolve_structure_without_adata_falls_back():
    """structure needs adata; without it resolve falls back (still returns a k)."""
    s = _fake_scores()
    order = [f"g{i}" for i in range(s.size)]
    k, meta = resolve_n_top_genes(
        "structure",
        s,
        order,
        method=None,
        adata=None,
        k_min=500,
        k_max=5000,
    )
    assert 500 <= k <= 5000
    # fallback path is ensemble
    assert meta["strategy"] == "ensemble"


def test_structure_stability_n_boot_default_lighter_than_cap_merge():
    """Structure pair-stability uses n_boot=5; cap-merge keeps 15."""
    from scfair.pp._auto_n import STRUCTURE_STABILITY_N_BOOT
    from scfair.pp._highly_variable_genes import _MERGE_N_BOOT

    assert STRUCTURE_STABILITY_N_BOOT == 5
    assert _MERGE_N_BOOT == 15
    assert STRUCTURE_STABILITY_N_BOOT < _MERGE_N_BOOT


def test_product_structure_uses_n_seeds_3_not_library_default():
    """Shipped auto must not inherit estimate_n_top_structure's n_seeds=1 default."""
    from scfair.pp._auto_n import PRODUCT_STRUCTURE_N_SEEDS

    assert PRODUCT_STRUCTURE_N_SEEDS == 3

    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_obs, n_vars = 200, 400
    X = sp.csr_matrix(rng.poisson(1.2, size=(n_obs, n_vars)).astype(np.float32))
    a = ad.AnnData(X)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"g{i}" for i in range(n_vars)]
    a.obs_names = [f"c{i}" for i in range(n_obs)]

    # resolve_n_top_genes("structure") is the non-fast-path product entry
    scores = np.arange(n_vars, 0, -1, dtype=float)
    order = [f"g{i}" for i in range(n_vars)]
    k, meta = resolve_n_top_genes(
        "structure",
        scores,
        order,
        adata=a,
        k_min=50,
        k_max=300,
        random_state=0,
    )
    assert 50 <= k <= 300
    struct = meta.get("structure") or {}
    assert struct.get("n_seeds") == PRODUCT_STRUCTURE_N_SEEDS
    assert len(struct.get("per_seed_k") or []) == PRODUCT_STRUCTURE_N_SEEDS


def test_estimate_n_top_structure_records_stability_n_boot():
    """Detail records which n_boot was used for structure stability."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_obs, n_vars = 200, 400
    X = sp.csr_matrix(rng.poisson(1.2, size=(n_obs, n_vars)).astype(np.float32))
    a = ad.AnnData(X)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"g{i}" for i in range(n_vars)]
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    k, detail = estimate_n_top_structure(a, random_state=0, k_min=100, k_max=4000)
    assert 1 <= k <= min(4000, a.n_vars)
    assert detail.get("stability_n_boot") == 5
    feats = detail.get("features") or {}
    if "stability_n_boot" in feats:
        assert feats["stability_n_boot"] == 5


def test_estimate_n_top_structure_smoke(adata_counts_sparse):
    """End-to-end structure on a tiny count matrix returns a clipped k."""
    import scipy.sparse as sp

    # upscale fixture so PCA / Leiden / density have room to run
    rng = np.random.default_rng(0)
    n_obs, n_vars = 200, 400
    X = sp.csr_matrix(rng.poisson(1.2, size=(n_obs, n_vars)).astype(np.float32))
    import anndata as ad

    a = ad.AnnData(X)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"g{i}" for i in range(n_vars)]
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    k, detail = estimate_n_top_structure(a, random_state=0, k_min=100, k_max=4000)
    # k is clipped to n_vars when the matrix is smaller than k_min
    assert 1 <= k <= min(4000, a.n_vars)
    assert detail["strategy"] == "structure"
    assert "features" in detail
    assert detail["features"]["n_obs"] == n_obs


def test_structure_k_source_tags_n_vars_clamp():
    """When k is bound by n_vars, k_source must not look purely data-driven."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    n_obs, n_vars = 150, 80
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32))
    a = ad.AnnData(X)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"g{i}" for i in range(n_vars)]
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    # Default-like min 50 with small gene set → final k often == n_vars.
    k, detail = estimate_n_top_structure(a, random_state=0, k_min=50, k_max=5000, n_genes=n_vars)
    assert k == n_vars or k <= n_vars
    if k >= n_vars:
        assert "clamped_n_vars" in str(detail.get("k_source") or "")


def test_fine_mode_floor_respects_k_max():
    """fine_mode_floor must not push k past user k_max / n_top_max."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(2)
    n_obs, n_vars = 200, 3000
    X = sp.csr_matrix(rng.poisson(0.8, size=(n_obs, n_vars)).astype(np.float32))
    a = ad.AnnData(X)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"g{i}" for i in range(n_vars)]
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    k, detail = estimate_n_top_structure(
        a,
        random_state=0,
        k_min=100,
        k_max=800,
        n_genes=n_vars,
        hvg_mode="fine",
    )
    assert k <= 800
    assert k <= n_vars


def test_coverage_requires_all_clusters():
    order = [f"g{i}" for i in range(100)]
    # cluster A needs gene at position 40; B at 60
    ranks = {
        "A": [f"g{i}" for i in range(40, 50)],
        "B": [f"g{i}" for i in range(60, 70)],
    }
    k = select_n_top_coverage(order, ranks, min_per_cluster=5, k_min=10, k_max=100)
    # 5th gene of B is g64 → k >= 65
    assert k >= 65


def test_ensemble_median_generic_keys():
    # custom keys still median; floor on n_genes=5000 is 1000 but median=2000
    k = select_n_top_ensemble({"a": 500, "b": 2000, "c": 3000}, k_min=100, k_max=5000, n_genes=5000)
    assert k == 2000


def test_ensemble_v2_no_double_count_shape():
    """elbow+knee both 500 must not pull median to 500 when mass is high."""
    detail = select_n_top_ensemble_detail(
        {"elbow": 500, "knee": 500, "cumfrac": 3500, "coverage": 4000},
        k_min=500,
        k_max=5000,
        n_genes=10000,
    )
    # shape once + floor; cumfrac clipped to ceiling 2500; anchor 2000
    assert detail["version"] == "ensemble_v2.2"
    assert "shape" in detail["votes"]
    assert "elbow" not in detail["votes"]
    assert "anchor" in detail["votes"]
    assert detail["k"] >= 1000
    assert detail["k"] != 500
    assert detail["k"] <= detail["k_ceiling"]
    # votes {shape:1000, cumfrac:2500 (clipped), anchor:2000} → median 2000
    assert detail["k"] == 2000
    assert any("coverage_ignored" in n for n in detail["notes"])
    assert any("cumfrac_clipped" in n for n in detail["notes"])


def test_ensemble_v2_1_ceiling_blocks_high_k():
    """Even without anchor, final k must not exceed soft ceiling (~2500)."""
    k = select_n_top_ensemble(
        {"elbow": 2000, "knee": 2000, "cumfrac": 4000, "coverage": 5000},
        k_min=500,
        k_max=5000,
        n_genes=10000,
        anchor=None,
    )
    assert k <= 2500


def test_ensemble_v2_no_coverage_still_safe():
    """Without coverage, v1 median(500,500,3500)=500; v2 must stay ≥ floor."""
    k = select_n_top_ensemble(
        {"elbow": 500, "knee": 500, "cumfrac": 3500},
        k_min=500,
        k_max=5000,
        n_genes=10000,
    )
    assert k >= 1000
    # with anchor → 2000
    assert k == 2000


def test_ensemble_v2_coverage_soft_raise_only():
    """Modest coverage can raise k slightly; huge coverage cannot explode k."""
    # coverage just above median band
    k_soft = select_n_top_ensemble(
        {"elbow": 1500, "knee": 1500, "cumfrac": 1800, "coverage": 2000},
        k_min=500,
        k_max=5000,
        n_genes=10000,
        anchor=None,  # isolate coverage effect
    )
    # median(shape=1500, cumfrac=1800)=1650; cover 2000 <= 1.25*1650=2062 → raise to 2000
    assert k_soft == 2000

    k_huge = select_n_top_ensemble(
        {"elbow": 1500, "knee": 1500, "cumfrac": 1800, "coverage": 5000},
        k_min=500,
        k_max=5000,
        n_genes=10000,
        anchor=None,
    )
    # 5000 > 1.25*1650 → ignored; k stays median 1650
    assert k_huge == 1650


def test_effective_k_floor():
    assert effective_k_floor(10000, k_min=500) == 1000
    assert effective_k_floor(3000, k_min=500) == 600  # 0.2 * 3000
    assert effective_k_floor(80, k_min=20, soft_floor=1000) == 20


def test_effective_k_ceiling():
    assert effective_k_ceiling(10000, k_min=500, soft_ceiling=2500) == 2500
    assert effective_k_ceiling(1000, k_min=500, soft_ceiling=2500) == 1000


def test_resolve_fixed():
    s = _fake_scores(1000)
    order = [f"g{i}" for i in range(1000)]
    k, meta = resolve_n_top_genes(800, s, order, k_min=100, k_max=900)
    assert k == 800
    assert meta["strategy"] == "fixed"


def test_resolve_auto_ensemble():
    s = _fake_scores(3000)
    order = [f"g{i}" for i in range(3000)]
    ranks = {"c0": order[:50], "c1": order[100:150]}
    k, meta = resolve_n_top_genes(
        "auto",
        s,
        order,
        method="ensemble",
        k_min=200,
        k_max=2500,
        cluster_gene_ranks=ranks,
        min_per_cluster=10,
    )
    assert 200 <= k <= 2500
    assert meta["strategy"] == "ensemble"
    assert "method_picks" in meta
    assert meta["ensemble"] is not None
    assert meta["ensemble"]["version"] == "ensemble_v2.2"
    # floor for 3000 genes with k_min=200 → max(200, min(1000,600))=600
    assert k >= meta["ensemble"]["k_floor"]
    assert k <= meta["ensemble"]["k_ceiling"]


def test_depth_aware_knobs_tiers():
    sh = depth_aware_auto_knobs(median_counts=245, median_genes=150, n_genes=10000)
    assert sh["depth_tier"] == "shallow"
    assert sh["anchor"] == 2000
    assert sh["soft_ceiling"] == 2000

    mid = depth_aware_auto_knobs(median_counts=2200, median_genes=820, n_genes=10000)
    assert mid["depth_tier"] == "medium"
    assert mid["anchor"] == 2000
    assert mid["soft_ceiling"] == 2500

    deep = depth_aware_auto_knobs(median_counts=288391, median_genes=6070, n_genes=10000)
    assert deep["depth_tier"] == "deep"
    assert deep["soft_ceiling"] <= 2000
    assert deep["anchor"] <= 1500


def test_hvg_auto_runs(adata_for_hvg=None):
    import anndata as ad

    import scfair as scf

    rng = np.random.default_rng(0)
    X = rng.poisson(1.5, size=(120, 80)).astype(float)
    # plant structure
    X[:40, 0] += 20
    X[40:80, 1] += 20
    X[80:, 2] += 20
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(120)]
    a.var_names = [f"g{i}" for i in range(80)]
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",
        n_top_min=20,
        n_top_max=60,
        auto_n_method="elbow",
        balance_method="none",
        flavor="seurat_v3",
    )
    n = int(a.var["highly_variable"].sum())
    assert 20 <= n <= 60
    assert a.uns["scfair"]["hvg"]["auto_n"] is not None
    assert a.uns["scfair"]["hvg"]["n_top_genes_used"] == n


def test_hvg_auto_structure_default():
    """Package default auto_n_method=structure picks k and hybrid-realigns."""
    import anndata as ad

    import scfair as scf

    rng = np.random.default_rng(1)
    X = rng.poisson(2.0, size=(200, 200)).astype(float)
    for i, start in enumerate((0, 50, 100, 150)):
        X[start : start + 50, i] += 25
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(200)]
    a.var_names = [f"g{i}" for i in range(200)]
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",  # default path; method defaults to structure
        n_top_min=30,
        n_top_max=120,
        balance_method="hybrid",
        flavor="seurat_v3",
        min_cluster_size=20,
    )
    n = int(a.uns["scfair"]["hvg"]["n_top_genes_used"])
    assert 30 <= n <= 120
    auto = a.uns["scfair"]["hvg"]["auto_n"]
    assert auto["strategy"] == "structure"
    assert auto.get("structure") is not None
    # hybrid auto should realign to 2×k pool after k pick
    assert auto.get("pool_realign") == "hybrid_2xk"


def test_hvg_auto_ensemble_opt_in():
    """Previous ensemble default still available via auto_n_method."""
    import anndata as ad

    import scfair as scf

    rng = np.random.default_rng(1)
    X = rng.poisson(2.0, size=(200, 200)).astype(float)
    for i, start in enumerate((0, 50, 100, 150)):
        X[start : start + 50, i] += 25
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(200)]
    a.var_names = [f"g{i}" for i in range(200)]
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",
        auto_n_method="ensemble",
        n_top_min=30,
        n_top_max=120,
        balance_method="hybrid",
        flavor="seurat_v3",
        min_cluster_size=20,
    )
    n = int(a.uns["scfair"]["hvg"]["n_top_genes_used"])
    assert 30 <= n <= 120
    ens = a.uns["scfair"]["hvg"]["auto_n"].get("ensemble")
    assert ens is not None
    assert ens["version"] == "ensemble_v2.2"
    depth = a.uns["scfair"]["hvg"]["auto_n"].get("depth")
    assert depth is not None
    assert "depth_tier" in depth


# ---------------------------------------------------------------------------
# rule_branch labels + holdout feature-level regressions
# ---------------------------------------------------------------------------


def test_structure_k_buffer_ladder():
    from scfair.pp._auto_n import apply_structure_k_buffer

    assert apply_structure_k_buffer(500) == (1000, 500)
    assert apply_structure_k_buffer(1000) == (1500, 1000)
    assert apply_structure_k_buffer(1500) == (2000, 1500)
    assert apply_structure_k_buffer(2000) == (2000, None)
    assert apply_structure_k_buffer(2750) == (2750, None)


def test_post_combine_does_not_double_buffer():
    """Regression: soft_1000 → 1500 in the rule must not be re-lifted to 2000.

    Real TM-spleen-like path showed
    ``soft_1000_vm+k_buffer:1000→1500+…+k_buffer:1500→2000`` when
    ``_apply_short_floor_if_needed`` re-applied the ladder after combine.
    """
    from scfair.pp._auto_n import _apply_short_floor_if_needed

    k, src, tag = _apply_short_floor_if_needed(
        k=1500,
        k_source="aggregated_features",
        n_obs=1_689,
        n_density_pops=8,
        density_confidence="high",
        density_depth_sensitivity=1,
        k_min=500,
        k_max=5000,
        n_genes=15_000,
    )
    assert k == 1500
    assert src == "aggregated_features"
    assert tag is None

    # Raw SHORT under false-SHORT geometry floors to 2000
    k2, src2, tag2 = _apply_short_floor_if_needed(
        k=500,
        k_source="unanimous_seed_vote",
        n_obs=20_000,
        n_density_pops=6,
        density_confidence="low",
        density_depth_sensitivity=3,
        k_min=500,
        k_max=5000,
        n_genes=20_000,
    )
    assert k2 == 2000
    assert tag2 is not None and "false_short" in tag2


def test_explain_structure_rule_seurat_v7_band_floor():
    """Holdout seurat-like: v7 fine-atlas band → ~2000; v6 SHORT → buffer 1000."""
    # ratio ≈ 1.12 fails v6 density-surplus (<1) but hits v7 band
    kwargs = dict(
        valley_median=0.793,
        frac_shallow=0.02,
        n_density_pops=17,
        n_leiden=19,
        mean_stability=0.64,
        min_stability=0.02,
        n_obs=20_000,
    )
    ex = explain_structure_rule(**kwargs, version="v7")
    assert ex["n_top"] == 2000
    assert ex["rule_branch"] == "v7_fine_atlas_band"
    assert ex["v7_band_eligible"] is True
    # v6 short rule 500 + soft buffer → 1000 (nd=17 so not large_n_few_density anti-SHORT)
    assert select_n_top_from_structure(**kwargs, version="v6") == 1000
    assert select_n_top_from_structure(**kwargs, version="v7") == 2000


def test_explain_structure_rule_lung_short():
    """Holdout lung-like: high nd / high ratio → SHORT rule; soft buffer → 1000."""
    kwargs = dict(
        valley_median=0.79,
        frac_shallow=0.12,
        n_density_pops=24,
        n_leiden=28,
        mean_stability=0.71,
        min_stability=-0.01,
        n_obs=15_000,
    )
    ex = explain_structure_rule(**kwargs, version="v7")
    assert ex["n_top"] == 1000
    assert "short_hard" in ex["rule_branch"]
    assert "k_buffer" in ex["rule_branch"]
    assert ex["k_buffer_raw"] == 500
    assert ex["v7_band_eligible"] is False
    assert "nd_in_band" in (ex.get("v7_band_miss") or [])


def test_explain_structure_rule_adt_mid():
    """Holdout ADT-like: mid density cores → rule 1500, buffer → 2000."""
    kwargs = dict(
        valley_median=0.72,
        frac_shallow=0.15,
        n_density_pops=9,
        n_leiden=10,
        mean_stability=0.70,
        min_stability=0.20,
        n_obs=8_000,
    )
    ex = explain_structure_rule(**kwargs, version="v7")
    assert ex["n_top"] == 2000
    assert "mid_1500" in ex["rule_branch"]
    assert "k_buffer" in ex["rule_branch"]
    assert ex["k_buffer_raw"] == 1500


def test_explain_structure_rule_matches_selector():
    """explain_structure_rule['n_top'] always equals select_n_top_from_structure."""
    cases = [
        dict(valley_median=0.9, frac_shallow=0.05, n_density_pops=8, n_leiden=10, n_obs=5_000),
        dict(valley_median=0.5, frac_shallow=0.3, n_density_pops=4, n_leiden=5, n_obs=3_000),
        dict(
            valley_median=0.82,
            frac_shallow=0.01,
            n_density_pops=15,
            n_leiden=14,
            n_obs=20_000,
            version="v7",
        ),
    ]
    for kw in cases:
        ver = kw.pop("version", "v7")
        assert explain_structure_rule(**kw, version=ver)["n_top"] == select_n_top_from_structure(
            **kw, version=ver
        )
