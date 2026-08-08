"""Tests for automatic n_top_genes selection."""

from __future__ import annotations

import numpy as np

from scfair.pp._auto_n import (
    APPEND_BUDGET_FLOOR,
    APPEND_BUDGET_HI,
    APPEND_BUDGET_OFFSET,
    APPEND_BUDGET_PER_NEED,
    estimate_n_top_structure,
    explain_structure_rule,
    plain_auto_n_message,
    product_append_budget,
    select_n_top_elbow,
    select_n_top_from_structure,
)


def _fake_scores(n: int = 5000, decay: float = 0.001) -> np.ndarray:
    # smoothly decreasing positive scores
    x = np.arange(n, dtype=float)
    return np.exp(-decay * x) * 10.0 + 0.01


def test_product_append_budget_tight_density():
    """Floor 200; raise only when n_need > OFFSET=12; cap at 300."""
    assert APPEND_BUDGET_FLOOR == 200
    assert APPEND_BUDGET_HI == 300
    assert APPEND_BUDGET_OFFSET == 12
    assert APPEND_BUDGET_PER_NEED == 12

    m0, info0 = product_append_budget(None)
    assert m0 == 200
    assert info0["n_need"] == 0
    assert info0["append_budget_raised"] is False
    assert info0["append_budget_rule"] == "tight_density_v1"

    m_low, info_low = product_append_budget(12)
    assert m_low == 200
    assert info_low["n_need"] == 12
    assert info_low["append_budget_raised"] is False

    # n_need=13 → extra = (13-12)*12 = 12 → m = 212
    m13, info13 = product_append_budget(13)
    assert m13 == 212
    assert info13["n_need"] == 13
    assert info13["append_budget_raised"] is True

    # n_need=20 → extra = 8*12 = 96 → m = 296
    m20, _ = product_append_budget(20)
    assert m20 == 296

    # n_need=21 → extra = 9*12 = 108 → m = min(300, 308) = 300
    m21, info21 = product_append_budget(21)
    assert m21 == 300
    assert info21["append_budget_raised"] is True

    # float n_density_pops is rounded
    m_f, info_f = product_append_budget(12.6)
    assert info_f["n_need"] == 13
    assert m_f == 212

    m_nan, _ = product_append_budget(float("nan"))
    assert m_nan == 200


def test_plain_auto_n_message_low_conf_and_sizes():
    """One-line user-facing text for common structure branches."""
    msg = plain_auto_n_message(k=2000, rule_branch="…low_conf_floor…")
    assert "2000" in msg
    assert "confidence" in msg.lower() or "safer" in msg.lower()

    msg_short = plain_auto_n_message(k=500, rule_branch="no_buffer:short")
    assert "500" in msg_short
    assert "short" in msg_short.lower()

    msg_block = plain_auto_n_message(k=1000, short_blocked=True)
    assert "1000" in msg_block
    assert "not trusted" in msg_block.lower() or "short" in msg_block.lower()

    msg_long = plain_auto_n_message(k=3500)
    assert "3500" in msg_long
    assert "long" in msg_long.lower()

    msg_mid = plain_auto_n_message(k=2000, rule_branch="density_mid")
    assert "2000" in msg_mid
    assert "override" in msg_mid.lower() or "structure" in msg_mid.lower()


def test_elbow_respects_bounds():
    """Elbow strategy must clamp into [k_min, k_max]."""
    k = select_n_top_elbow(_fake_scores(), k_min=500, k_max=5000)
    assert 500 <= k <= 5000


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


def test_structure_long_branch_caps_relative_to_n_genes():
    """LONG must not select ~100% of genes on small pools (n_vars < k_max)."""
    from scfair.pp._auto_n import explain_structure_rule

    for n_genes in (1200, 4000):
        d = explain_structure_rule(
            valley_median=0.03,
            frac_shallow=1.0,
            n_density_pops=3,
            mean_stability=0.95,
            min_stability=0.5,
            k_max=5000,
            n_genes=n_genes,
        )
        assert d["rule_branch"].startswith("long_shallow_few_cores")
        assert d["n_top"] <= int(0.5 * n_genes)
        assert d["n_top"] < n_genes


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
    """short_hard + large n + low conf + nd≤FALSE_SHORT_ND_MAX → floor 2000."""
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
        n_density_pops=7,
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=20_000,
        n_leiden=8,
        density_confidence="low",
    )
    assert k == 2000


def test_structure_soft_buffer_when_not_false_short():
    """nd just above false-SHORT max: soft buffer; low conf then floors to 2000."""
    from scfair.pp._auto_n import explain_structure_rule

    d_hi = explain_structure_rule(
        valley_median=0.83,
        frac_shallow=0.1,
        n_density_pops=9,
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=20_000,
        n_leiden=10,
        density_confidence="high",
    )
    assert d_hi["short_blocked"] is False
    assert d_hi["n_top"] == 1000
    assert d_hi["k_buffer_raw"] == 500
    assert "k_buffer:500→1000" in d_hi["rule_branch"]
    assert "low_conf_floor" not in d_hi["rule_branch"]

    d = explain_structure_rule(
        valley_median=0.83,
        frac_shallow=0.1,
        n_density_pops=9,
        mean_stability=0.7,
        min_stability=0.1,
        n_obs=20_000,
        n_leiden=10,
        density_confidence="low",
    )
    assert d["short_blocked"] is True
    assert d["n_top"] == 2000
    assert d["short_block_reason"] == "density_confidence_low"
    assert "k_buffer:500→1000" in d["rule_branch"]
    assert "low_conf_floor:1000→2000" in d["rule_branch"]
    assert "antishort:false_short" not in d["rule_branch"]


def test_structure_true_short_skips_buffer_with_n_types():
    """Multi-core short_hard + n_types≥5: keep k=500 (no soft buffer)."""
    from scfair.pp._auto_n import explain_structure_rule

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
    assert "low_conf_floor" not in d["rule_branch"]
    # Without n_types: soft buffer then low-conf floor → 2000
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
    assert d2["n_top"] == 2000
    assert "k_buffer:500→1000" in d2["rule_branch"]
    assert "low_conf_floor" in d2["rule_branch"]
    # n_types=2: still buffer, then low-conf floor → 2000
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
    assert d3["n_top"] == 2000
    assert "k_buffer" in d3["rule_branch"]
    assert "low_conf_floor" in d3["rule_branch"]


def test_structure_anti_short_residual_and_false_short_post_combine():
    """Post-combine floors: false_short and low_conf; high-nd 500 exempt."""
    from scfair.pp._auto_n import _apply_short_floor_if_needed

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

    k2b, src2b, tag2b = _apply_short_floor_if_needed(
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
    assert k2b == 2000
    assert tag2b is not None and "low_conf_floor" in tag2b
    assert "low_conf_floor" in src2b

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


def test_structure_auto_n_default_uses_nd_budget():
    """Fall-through scales with density cores instead of a hard 2000."""
    from scfair.pp._auto_n import ND_GENES_PER_CORE, explain_structure_rule

    d5 = explain_structure_rule(
        valley_median=0.4,
        frac_shallow=0.4,
        n_density_pops=5,
        density_confidence="high",
    )
    assert d5["rule_branch"].startswith("nd_budget")
    assert d5["n_top"] == _clip_expect(5 * ND_GENES_PER_CORE)

    d10 = explain_structure_rule(
        valley_median=0.4,
        frac_shallow=0.4,
        n_density_pops=10,
        density_confidence="high",
    )
    d20 = explain_structure_rule(
        valley_median=0.4,
        frac_shallow=0.4,
        n_density_pops=20,
        density_confidence="high",
    )
    # nd must move k (the old default_2000 discarded this signal)
    assert d5["n_top"] < d10["n_top"] <= d20["n_top"]
    assert d20["n_top"] >= 2000


def _clip_expect(k: int, k_min: int = 500, k_max: int = 5000, n_genes: int = 50_000) -> int:
    from scfair.pp._auto_n import _clip_k, apply_structure_k_buffer

    k0 = _clip_k(int(k), k_min, k_max, n_genes)
    k1, _ = apply_structure_k_buffer(k0)
    return _clip_k(int(k1), k_min, k_max, n_genes)


def test_structure_missing_nd_falls_back_to_2000():
    """Without a finite nd, classical 2000 remains the fall-through."""
    # n_density_pops is required by the signature as int — pass 0-ish via
    # explain with a non-finite path is hard; zero cores → nd_budget:nd0
    # uses the >=1 guard and falls to default_2000 only when nd is non-finite.
    # Use n_density_pops that is finite but check missing-nd via direct call
    # with NaN through explain's float path:
    from scfair.pp._auto_n import explain_structure_rule

    d = explain_structure_rule(
        valley_median=0.4,
        frac_shallow=0.4,
        n_density_pops=float("nan"),  # type: ignore[arg-type]
        density_confidence="high",
    )
    assert d["n_top"] == 2000
    assert d["rule_branch"].startswith("default_2000")


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


def test_hvg_auto_runs():
    """Product auto (structure) runs and records auto_message."""
    import anndata as ad

    import scfair as scf
    from scfair.pp import HVGOptions

    rng = np.random.default_rng(0)
    X = rng.poisson(1.2, size=(120, 80)).astype(np.float32)
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(80)]
    a.obs_names = [f"c{i}" for i in range(120)]
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",
        balance_method="append",
        options=HVGOptions(n_top_min=10, n_top_max=40, structure_n_seeds=1, append_budget=3),
        diagnose=False,
        progress=False,
    )
    h = a.uns["scfair"]["hvg"]
    assert h["auto_n"]["strategy"] == "structure"
    assert isinstance(h.get("auto_message"), str)
    assert int(a.var["highly_variable"].sum()) >= 10


def test_hvg_auto_structure_default():
    """Default auto path is structure + append."""
    import anndata as ad

    import scfair as scf
    from scfair.pp import HVGOptions

    rng = np.random.default_rng(1)
    X = rng.poisson(1.2, size=(120, 80)).astype(np.float32)
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(80)]
    a.obs_names = [f"c{i}" for i in range(120)]
    scf.pp.highly_variable_genes(
        a,
        n_top_genes="auto",
        options=HVGOptions(n_top_min=10, n_top_max=40, structure_n_seeds=1, append_budget=3),
        diagnose=False,
        progress=False,
    )
    assert a.uns["scfair"]["hvg"]["auto_n"]["strategy"] == "structure"
    assert a.uns["scfair"]["hvg"]["balance_method"] == "append"


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
