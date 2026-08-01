"""Tests for automatic n_top_genes selection."""

from __future__ import annotations

import pytest
import numpy as np

from scfair.pp._auto_n import (
    depth_aware_auto_knobs,
    effective_k_ceiling,
    effective_k_floor,
    estimate_n_top_structure,
    select_n_top_coverage,
    select_n_top_cumfrac,
    select_n_top_elbow,
    select_n_top_ensemble,
    select_n_top_ensemble_detail,
    select_n_top_from_structure,
    select_n_top_knee,
    resolve_n_top_genes,
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


def test_structure_auto_n_pancreas_like_short():
    """Many deep density cores → short shortlist."""
    k = select_n_top_from_structure(
        valley_median=0.81,
        frac_shallow=0.1,
        n_density_pops=14,
        mean_stability=0.73,
        min_stability=0.0,
    )
    assert k == 500


def test_structure_auto_n_adt_like_mid():
    k = select_n_top_from_structure(
        valley_median=0.77,
        frac_shallow=0.05,
        n_density_pops=10,
        mean_stability=0.69,
        min_stability=0.12,
    )
    assert k == 1500


def test_structure_mid_unstable_bumps_to_2000():
    """Mid band + low pair-stability bumps one rung up (1500→2000)."""
    k = select_n_top_from_structure(
        valley_median=0.71,
        frac_shallow=0.21,
        n_density_pops=7,
        mean_stability=0.52,
        min_stability=-0.10,
    )
    assert k == 2000
    # stable mid still 1500
    assert (
        select_n_top_from_structure(
            valley_median=0.77,
            frac_shallow=0.05,
            n_density_pops=10,
            mean_stability=0.69,
            min_stability=0.12,
        )
        == 1500
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

    # small n + unanimous short → allow short
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
    assert select_n_top_from_structure(**kwargs, version="v4") == 500
    assert select_n_top_from_structure(**kwargs, version="v5") == 2000


def test_structure_v5_sln_like_still_short():
    """Large n_obs but only ~12 density pops still shorts."""
    k = select_n_top_from_structure(
        valley_median=0.92,
        frac_shallow=0.05,
        n_density_pops=12,
        mean_stability=0.58,
        min_stability=-0.03,
        n_obs=15_820,
        version="v5",
    )
    assert k == 500


def test_structure_v5_without_n_obs_matches_v4():
    """Atlas guard requires n_obs; omit → v5 behaves like v4."""
    kwargs = dict(
        valley_median=0.80,
        frac_shallow=0.01,
        n_density_pops=18,
        mean_stability=0.64,
        min_stability=0.04,
    )
    assert select_n_top_from_structure(**kwargs, version="v5") == 500
    assert select_n_top_from_structure(**kwargs, version="v4") == 500


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
    assert select_n_top_from_structure(**kwargs, version="v4") == 500
    assert select_n_top_from_structure(**kwargs, version="v6") == 2000
    assert select_n_top_from_structure(**kwargs, version="v7") == 2000


def test_structure_v6_lung_like_still_short():
    """Leiden ≥ density on large multi-core → still SHORT."""
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
    assert select_n_top_from_structure(**kwargs, version="v6") == 500
    assert select_n_top_from_structure(**kwargs, version="v7") == 500
    assert select_n_top_from_structure(**kwargs, version="v4") == 500


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
    assert select_n_top_from_structure(**kwargs, version="v6") == 500
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
    assert select_n_top_from_structure(**kwargs, version="v7") == 500


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
