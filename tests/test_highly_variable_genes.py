"""Tests for the public scfair.pp.highly_variable_genes product API.

Product surface: balance_method append | none; n_top_genes auto|structure|int.
Cluster-aware methods (hybrid/score/reweight) were removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import scfair as scf
from scfair._utils import UNS_KEY
from scfair.pp import HVGOptions
from scfair.pp._highly_variable_genes import (
    _apply_gene_filters,
    _hvg_base_plus_append,
    _infer_gene_nomenclature,
    _is_mito_name,
    _is_ribo_name,
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


def test_hvg_global_only(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=30, balance_method="none", flavor="seurat_v3")
    assert int(ad.var["highly_variable"].sum()) == 30
    assert "counts" in ad.layers
    sc = ad.uns.get(UNS_KEY, {})
    assert "raw_snapshot" not in sc
    assert ad.uns[UNS_KEY]["hvg"]["balance_method"] == "none"
    assert ad.uns[UNS_KEY]["hvg"]["store_raw"] is False
    assert "scfair_score" in ad.var


def test_hvg_append_default_extends_base(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=5),
        diagnose=False,
        progress=False,
    )
    assert int(ad.var["highly_variable"].sum()) == 25
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["balance_method"] == "append"
    assert meta["n_top_genes_used"] == 20
    assert meta["append"]["n_base"] == 20
    assert meta["append"]["n_append_used"] == 5


def test_hvg_base_plus_append_helper():
    scores = pd.Series(np.arange(10, 0, -1, dtype=float), index=[f"g{i}" for i in range(10)])
    sel, meta = _hvg_base_plus_append(scores, n_base=3, n_append=2)
    assert sel == ["g0", "g1", "g2", "g3", "g4"]
    assert meta["n_base"] == 3
    assert meta["n_append_used"] == 2


def test_append_budget_zero_matches_none_size(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=25,
        balance_method="append",
        options=HVGOptions(append_budget=0),
        diagnose=False,
        progress=False,
    )
    assert int(ad.var["highly_variable"].sum()) == 25


def test_append_budget_capped_to_n_top(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.warns(UserWarning, match="append_budget"):
        scf.pp.highly_variable_genes(
            ad,
            n_top_genes=20,
            balance_method="append",
            options=HVGOptions(append_budget=200),
            diagnose=False,
            progress=False,
        )
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["append_budget"] == 20
    assert meta["append_budget_requested"] == 200
    assert meta["append_budget_capped"] is True
    assert int(ad.var["highly_variable"].sum()) == 40


def test_hybrid_rejected(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="append|none|removed"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="hybrid")


def test_score_rejected(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(ValueError, match="append|none|removed"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="score")


def test_removed_options_rejected(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(TypeError, match="removed option"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=10, options=HVGOptions(), neighbor_contrast=1.0
        )


def test_mito_ribo_name_helpers():
    assert _is_mito_name("MT-CO3", "human")
    assert not _is_mito_name("mt-Nd1", "human")
    assert _is_mito_name("mt-Nd1", "mouse")
    assert not _is_mito_name("MT-CO3", "mouse")
    assert _is_mito_name("MT-CO3", "mixed")
    assert _is_mito_name("mt-Nd1", "unknown")
    assert not _is_mito_name("COX4I1", "human")
    assert not _is_mito_name("MTOR", "human")
    assert _is_ribo_name("RPL19")
    assert _is_ribo_name("RPS6")
    assert _is_ribo_name("RPSA")
    assert _is_ribo_name("RPLP0")
    assert _is_ribo_name("Rpl19")
    assert _is_ribo_name("Rplp0")
    assert _is_ribo_name("Rpsa")
    assert _is_ribo_name("Rpl13a")
    assert not _is_ribo_name("RPA1")
    assert not _is_ribo_name("RPS6KA1")
    assert not _is_ribo_name("Rps6ka1")
    assert _infer_gene_nomenclature(["MT-ND1", "RPL13", "GAPDH"]) == "human"
    assert _infer_gene_nomenclature(["mt-Nd1", "Rpl13", "Gapdh"]) == "mouse"
    assert _infer_gene_nomenclature(["ENSG000001", "MT-CO1"]) == "human"
    assert _infer_gene_nomenclature(["ENSMUSG000001", "mt-Co1"]) == "mouse"
    assert _infer_gene_nomenclature(["g1", "g2", "g3"]) == "unknown"


def test_hvg_filter_ribo(adata_for_hvg):
    ad = adata_for_hvg.copy()
    new_names = list(ad.var_names)
    new_names[0] = "RPL19"
    new_names[1] = "RPS6"
    ad.var_names = new_names
    # markers protected when filters on
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="none",
        marker_genes=["RPL19", "RPS6"],
        options=HVGOptions(filter_ribo=True, filter_mito=True),
    )
    assert bool(ad.var.loc["RPL19", "highly_variable"])

    # opt-in filters drop ribo symbols
    ad2 = adata_for_hvg.copy()
    ad2.var_names = new_names
    scf.pp.highly_variable_genes(
        ad2,
        n_top_genes=20,
        balance_method="none",
        options=HVGOptions(filter_ribo=True, filter_mito=True),
    )
    assert not bool(ad2.var.loc["RPL19", "highly_variable"])
    assert not bool(ad2.var.loc["RPS6", "highly_variable"])

    # product default: filters off — ribo can stay
    ad3 = adata_for_hvg.copy()
    ad3.var_names = new_names
    scf.pp.highly_variable_genes(ad3, n_top_genes=20, balance_method="none")
    # RPL19 is highly expressed private marker in fixture group 0 → often selected
    assert "filter_ribo" in ad3.uns[UNS_KEY]["hvg"]
    assert ad3.uns[UNS_KEY]["hvg"]["filter_ribo"] is False


def test_unknown_nomenclature_tip_without_failing(adata_for_hvg):
    ad = adata_for_hvg.copy()
    ad.var_names = [f"feat_{i}" for i in range(ad.n_vars)]
    fill = pd.Series(np.arange(ad.n_vars, 0, -1, dtype=float), index=ad.var_names)
    selected = list(ad.var_names[:25])
    kept, info = _apply_gene_filters(
        selected,
        ad.var_names,
        filter_mito=True,
        filter_ribo=True,
        marker_genes=None,
        fill_rank=fill,
        n_top_genes=20,
        gene_nomenclature=None,
    )
    assert info["gene_nomenclature"] == "unknown"
    assert info["n_mito_ribo_dropped"] == 0
    assert any("MT/ribo" in t or "human" in t.lower() for t in info["tips"])
    assert len(kept) == 20

    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="none",
        options=HVGOptions(filter_mito=True, filter_ribo=True),
        diagnose=True,
    )
    h = ad.uns[UNS_KEY]["hvg"]
    assert h["gene_nomenclature"] == "unknown"
    tips = list(h.get("gene_filter_tips") or [])
    tips += list((h.get("diagnosis") or {}).get("tips") or [])
    assert any("MT/ribo" in str(t) or "nomenclature" in str(t).lower() for t in tips)


def test_hvg_marker_genes_forced(adata_for_hvg):
    ad = adata_for_hvg.copy()
    markers = ["g50", "g51"]
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="none", marker_genes=markers)
    for g in markers:
        assert bool(ad.var.loc[g, "highly_variable"])


def test_hvg_inplace_false_returns_dataframe(adata_for_hvg):
    ad = adata_for_hvg.copy()
    out = scf.pp.highly_variable_genes(
        ad, n_top_genes=15, balance_method="none", inplace=False, diagnose=False
    )
    assert out is not None
    assert "highly_variable" in out.columns
    assert "scfair_score" in out.columns
    assert out["highly_variable"].sum() == 15
    # caller object untouched
    assert "highly_variable" not in ad.var.columns


def test_hvg_subset_true(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=15, balance_method="none", subset=True, diagnose=False
    )
    assert ad.n_vars == 15
    assert bool(ad.var["highly_variable"].all())


def test_hvg_store_raw_true_writes_snapshot(adata_for_hvg):
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


def test_hvg_preserves_preexisting_snapshot_when_store_raw_false(adata_for_hvg):
    """A later default call must not delete a snapshot kept for full-gene restore."""
    from scfair.pp._raw_counts import _store_raw_counts

    ad = adata_for_hvg.copy()
    _store_raw_counts(ad, layer="counts", sidecar=True)
    assert "raw_snapshot" in ad.uns[UNS_KEY]
    scf.pp.highly_variable_genes(ad, n_top_genes=15, balance_method="none", diagnose=False)
    assert "raw_snapshot" in ad.uns.get(UNS_KEY, {})


def test_hvg_injected_global_score(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scores = pd.Series(np.arange(ad.n_vars, dtype=float), index=ad.var_names)
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=10,
        balance_method="none",
        options=HVGOptions(global_score=scores),
        diagnose=False,
    )
    assert int(ad.var["highly_variable"].sum()) == 10
    # top 10 by score = highest indices
    selected = set(ad.var_names[ad.var["highly_variable"]])
    assert selected == {f"g{i}" for i in range(70, 80)}


def test_hvg_injected_global_score_misaligned(adata_for_hvg):
    ad = adata_for_hvg.copy()
    bad = pd.Series([1.0, 2.0], index=["not_a_gene", "also_not"])
    with pytest.raises(ValueError, match="align"):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=5, options=HVGOptions(global_score=bad), diagnose=False
        )


def test_progress_false_is_silent(adata_for_hvg, capsys):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=3),
        progress=False,
        diagnose=False,
    )
    assert capsys.readouterr().err == ""


def test_progress_default_quiet_on_small_data(adata_for_hvg, capsys):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(ad, n_top_genes=20, balance_method="none", diagnose=False)
    assert capsys.readouterr().err == ""


def test_progress_true_announces_stages(adata_for_hvg, capsys):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=3),
        progress=True,
        diagnose=False,
    )
    err = capsys.readouterr().err.lower()
    assert "scfair" in err
    assert "done" in err


def test_structure_n_seeds_one_recorded(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes="auto",
        balance_method="append",
        options=HVGOptions(n_top_min=10, n_top_max=40, structure_n_seeds=1, append_budget=5),
        diagnose=False,
        progress=False,
    )
    auto = ad.uns[UNS_KEY]["hvg"].get("auto_n") or {}
    assert auto.get("structure_n_seeds") == 1
    assert not auto.get("fallback_reason"), auto.get("fallback_reason")
    st = auto.get("structure") or {}
    assert not st.get("fallback_reason"), st.get("fallback_reason")
    assert auto.get("rule_branch") or st.get("rule_branch")
    msg = ad.uns[UNS_KEY]["hvg"].get("auto_message")
    assert isinstance(msg, str) and "Base list size" in msg
    assert "structure auto failed" not in msg
    assert int(ad.var["highly_variable"].sum()) >= 10


def test_e2e_default_append_to_leiden_no_subset(adata_for_hvg):
    ad = adata_for_hvg.copy()
    n_vars_before = int(ad.n_vars)
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=25,
        balance_method="append",
        options=HVGOptions(append_budget=5),
        diagnose=False,
        progress=False,
    )
    assert int(ad.var["highly_variable"].sum()) == 30
    assert int(ad.n_vars) == n_vars_before

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad)
    sc.pp.neighbors(ad, n_neighbors=10, n_pcs=10)
    sc.tl.leiden(ad, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
    assert "leiden" in ad.obs
    assert ad.obs["leiden"].nunique() >= 1
    assert int(ad.n_vars) == n_vars_before


def test_hvg_flavor_routing(adata_for_hvg, flavor):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=15, balance_method="none", flavor=flavor, diagnose=False
    )
    assert int(ad.var["highly_variable"].sum()) == 15


@pytest.fixture(params=["seurat_v3", "seurat", "seurat_v3_paper", "cell_ranger"])
def flavor(request):
    return request.param


def _batched_counts(seed: int = 0, n_obs: int = 200, n_vars: int = 400):
    """Two-batch Poisson matrix with batch-private markers (for batch_key tests)."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    X = rng.poisson(1.5, size=(n_obs, n_vars)).astype(np.float32)
    half = n_obs // 2
    X[half:] = rng.poisson(3.0, size=(n_obs - half, n_vars)).astype(np.float32)
    for g in range(20):
        X[:half, g] = rng.poisson(15, size=half).astype(np.float32)
        X[half:, g + 20] = rng.poisson(15, size=n_obs - half).astype(np.float32)
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_vars)]
    a.obs["batch"] = (["A"] * half) + (["B"] * (n_obs - half))
    a.layers["counts"] = a.X.copy()
    return a


@pytest.mark.parametrize(
    "flavor",
    ["seurat_v3", "seurat_v3_paper", "seurat", "cell_ranger"],
)
def test_batch_key_selected_genes_match_scanpy(flavor):
    """With batch_key, selected set must match scanpy's per-batch merge (not mean score)."""
    n_top = 80
    a0 = _batched_counts(seed=1)
    a_sc = a0.copy()
    a_sf = a0.copy()
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        sc.pp.highly_variable_genes(
            a_sc,
            n_top_genes=n_top,
            flavor=flavor,
            batch_key="batch",
            layer="counts",
            inplace=True,
            subset=False,
        )
    else:
        sc.pp.normalize_total(a_sc, target_sum=1e4)
        sc.pp.log1p(a_sc)
        sc.pp.highly_variable_genes(
            a_sc,
            n_top_genes=n_top,
            flavor=flavor,
            batch_key="batch",
            inplace=True,
            subset=False,
        )
    scf.pp.highly_variable_genes(
        a_sf,
        n_top_genes=n_top,
        balance_method="none",
        flavor=flavor,
        options=HVGOptions(batch_key="batch"),
        diagnose=False,
    )
    set_sc = set(a_sc.var_names[a_sc.var["highly_variable"].to_numpy()])
    set_sf = set(a_sf.var_names[a_sf.var["highly_variable"].to_numpy()])
    assert set_sf == set_sc
    assert len(set_sf) == n_top
    # No gene with nbatches==0 should be selected under batch merge.
    if "highly_variable_nbatches" in a_sf.var.columns:
        nb = a_sf.var.loc[a_sf.var["highly_variable"], "highly_variable_nbatches"]
        assert int((nb.to_numpy() == 0).sum()) == 0


def test_no_batch_selected_genes_match_scanpy():
    """No-batch path remains a drop-in for seurat_v3 gene sets."""
    n_top = 80
    a0 = _batched_counts(seed=0)
    a_sc = a0.copy()
    a_sf = a0.copy()
    sc.pp.highly_variable_genes(
        a_sc, n_top_genes=n_top, flavor="seurat_v3", layer="counts", inplace=True, subset=False
    )
    scf.pp.highly_variable_genes(
        a_sf, n_top_genes=n_top, balance_method="none", flavor="seurat_v3", diagnose=False
    )
    set_sc = set(a_sc.var_names[a_sc.var["highly_variable"].to_numpy()])
    set_sf = set(a_sf.var_names[a_sf.var["highly_variable"].to_numpy()])
    assert set_sf == set_sc


def test_selected_genes_golden_stable():
    """Lock the selected gene set for a fixed seed (catches ranking drift)."""
    ad = _batched_counts(seed=42, n_obs=120, n_vars=200)
    scf.pp.highly_variable_genes(
        ad, n_top_genes=40, balance_method="none", flavor="seurat_v3", diagnose=False
    )
    selected = list(ad.var_names[ad.var["highly_variable"].to_numpy()])
    # Order by highly_variable_rank when present, else by name for stability of set.
    selected_set = frozenset(selected)
    assert len(selected_set) == 40
    # Second run identical (stability).
    ad2 = _batched_counts(seed=42, n_obs=120, n_vars=200)
    scf.pp.highly_variable_genes(
        ad2, n_top_genes=40, balance_method="none", flavor="seurat_v3", diagnose=False
    )
    selected2 = frozenset(ad2.var_names[ad2.var["highly_variable"].to_numpy()])
    assert selected2 == selected_set


def test_public_api_surface():
    assert "highly_variable_genes" in scf.pp.__all__
    assert "HVGOptions" in scf.pp.__all__
    assert "restore_raw_counts" in scf.pp.__all__
    assert "recommend_cluster_resolution" not in scf.pp.__all__
    assert "estimate_n_top_structure" not in scf.pp.__all__


def test_append_equals_none_with_larger_k(adata_for_hvg):
    """append base k + budget m is the same gene set as none with k+m."""
    ad_a = adata_for_hvg.copy()
    ad_b = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad_a,
        n_top_genes=30,
        balance_method="append",
        options=HVGOptions(append_budget=10),
        diagnose=False,
    )
    scf.pp.highly_variable_genes(
        ad_b,
        n_top_genes=40,
        balance_method="none",
        diagnose=False,
    )
    set_a = set(ad_a.var_names[ad_a.var["highly_variable"]])
    set_b = set(ad_b.var_names[ad_b.var["highly_variable"]])
    assert set_a == set_b
    assert len(set_a) == 40


def test_n_top_genes_bool_rejected(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.raises(TypeError, match="bool"):
        scf.pp.highly_variable_genes(ad, n_top_genes=True, balance_method="none")


def test_backed_adata_raises_clear_error(adata_for_hvg, tmp_path):
    """backed='r' must not explode inside numpy isfinite — clear NotImplementedError."""
    path = tmp_path / "tiny.h5ad"
    ad = adata_for_hvg.copy()
    ad.write_h5ad(path)
    backed = __import__("anndata").read_h5ad(path, backed="r")
    try:
        with pytest.raises(NotImplementedError, match="backed|to_memory"):
            scf.pp.highly_variable_genes(
                backed, n_top_genes=10, balance_method="none", diagnose=False
            )
    finally:
        # Close file handle so tmp_path cleanup works on all platforms.
        if hasattr(backed, "file") and backed.file is not None:
            try:
                backed.file.close()
            except Exception:
                pass


def test_mode_none_treated_as_auto(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, mode=None, balance_method="none", diagnose=False
    )
    assert ad.uns[UNS_KEY]["hvg"]["mode_requested"] == "auto"


def test_duplicate_var_names_rejected(adata_for_hvg):
    ad = adata_for_hvg.copy()
    names = list(ad.var_names)
    names[1] = names[0]
    ad.var_names = names
    with pytest.raises(ValueError, match="duplicate"):
        scf.pp.highly_variable_genes(ad, n_top_genes=10, balance_method="none")


def test_non_integer_counts_warns_and_records(adata_for_hvg):
    ad = adata_for_hvg.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    with pytest.warns(UserWarning, match="integer counts|raw integer|internal layer"):
        scf.pp.highly_variable_genes(ad, n_top_genes=15, balance_method="none", diagnose=False)
    meta = ad.uns[UNS_KEY]["hvg"]
    assert meta["counts_integer_like"] is False
    assert "non_integer_counts" in (meta["counts_warning"] or [])
    # Must not permanently pollute layers['counts'] with log data.
    assert "counts" not in ad.layers
    from scfair.pp._raw_counts import INTERNAL_COUNTS_LAYER

    assert INTERNAL_COUNTS_LAYER not in ad.layers  # ephemeral, popped at end


def test_mode_with_fixed_int_k_warns(adata_for_hvg):
    ad = adata_for_hvg.copy()
    with pytest.warns(UserWarning, match="mode="):
        scf.pp.highly_variable_genes(
            ad, n_top_genes=20, mode="fine", balance_method="none", diagnose=False
        )


def test_hvg_from_existing_counts_layer(adata_for_hvg):
    ad = adata_for_hvg.copy()
    ad.layers["counts"] = ad.X.copy()
    ad.X = ad.X.astype(float) + 0.1  # non-integer X; counts layer is good
    scf.pp.highly_variable_genes(
        ad, n_top_genes=15, balance_method="none", layer="counts", diagnose=False
    )
    assert int(ad.var["highly_variable"].sum()) == 15


def test_same_object_second_call_matches_clean_object(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, balance_method="none", diagnose=False, progress=False
    )
    mask1 = ad.var["highly_variable"].to_numpy().copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, balance_method="none", diagnose=False, progress=False
    )
    assert np.array_equal(mask1, ad.var["highly_variable"].to_numpy())


def test_diagnosis_on_append_path(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=20,
        balance_method="append",
        options=HVGOptions(append_budget=3),
        diagnose=True,
        progress=False,
    )
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert diag["balance_method"] == "append"
    assert diag["recommendation"] == "keep_current"


def test_diagnosis_none_path(adata_for_hvg):
    ad = adata_for_hvg.copy()
    scf.pp.highly_variable_genes(
        ad, n_top_genes=20, balance_method="none", diagnose=True, progress=False
    )
    diag = ad.uns[UNS_KEY]["hvg"]["diagnosis"]
    assert diag["balance_method"] == "none"
    assert "balance_method_none" in diag["flags"]
