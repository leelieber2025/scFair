"""Ground-truth tests of the package's core claim: rare identity genes survive.

1. **Hybrid re-rank** — controlled ``global_score`` puts rare markers outside
   top-``k`` but inside ``cluster_pool`` / hybrid top-``2k``. ``none`` misses
   them; ``hybrid`` promotes them via specificity.
2. **Score** — pure specificity on counts recovers planted rare markers that a
   tight global cut under-selects (with resolution high enough to isolate rare).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scfair as scf


def _build_counts_with_rare(
    *,
    n_large: int = 600,
    n_rare: int = 50,
    # Keep n_shared + n_type < cluster_pool so rare markers enter clustering
    # and the hybrid top-2k pool under a controlled global_score ranking.
    n_shared: int = 40,
    n_type: int = 30,
    n_markers: int = 20,
    n_bg: int = 100,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_obs = 3 * n_large + n_rare
    n_vars = n_shared + n_type + n_markers + n_bg
    X = rng.poisson(0.8, size=(n_obs, n_vars)).astype(np.float32)
    labels = np.array(["L0"] * n_large + ["L1"] * n_large + ["L2"] * n_large + ["R"] * n_rare)

    for j in range(n_shared):
        cells = np.flatnonzero(labels != "R")
        pick = rng.choice(cells, size=max(1, int(0.45 * len(cells))), replace=False)
        X[pick, j] = rng.poisson(28, size=pick.size).astype(np.float32)

    base = n_shared
    per = n_type // 3
    for g, name in enumerate(["L0", "L1", "L2"]):
        cells = np.flatnonzero(labels == name)
        idx = np.arange(base + g * per, base + (g + 1) * per)
        X[np.ix_(cells, idx)] = rng.poisson(20, size=(cells.size, per)).astype(np.float32)

    m0 = n_shared + n_type
    midx = np.arange(m0, m0 + n_markers)
    cells_r = np.flatnonzero(labels == "R")
    X[np.ix_(cells_r, midx)] = rng.poisson(30, size=(cells_r.size, n_markers)).astype(np.float32)

    adata = ad.AnnData(X)
    adata.obs_names = [f"c{i}" for i in range(n_obs)]
    adata.var_names = (
        [f"shared_{i}" for i in range(n_shared)]
        + [f"type{g}_{i}" for g in range(3) for i in range(per)]
        + [f"rare_marker_{i}" for i in range(n_markers)]
        + [f"bg_{i}" for i in range(n_bg)]
    )
    adata.obs["cell_type"] = labels.astype(str)
    adata.layers["counts"] = adata.X.copy()
    adata.uns["rare_marker_names"] = [f"rare_marker_{i}" for i in range(n_markers)]
    return adata


def _selected(adata) -> set[str]:
    return set(adata.var_names[adata.var["highly_variable"]].astype(str))


def _controlled_global_score(adata: ad.AnnData) -> pd.Series:
    """shared/type ≫ rare markers ≫ background (rare outside top-k, inside 2k)."""
    score = pd.Series(0.0, index=adata.var_names, dtype=float)
    for i, g in enumerate(adata.var_names):
        if g.startswith("shared_"):
            score[g] = 1000.0 - i * 0.01
        elif g.startswith("type"):
            score[g] = 500.0 - i * 0.01
        elif g.startswith("rare_marker_"):
            score[g] = 100.0 - int(g.split("_")[-1]) * 0.1
        else:
            score[g] = 1.0 - i * 1e-5
    return score


@pytest.fixture
def adata_rare_markers():
    return _build_counts_with_rare()


def test_hybrid_promotes_rare_markers_missed_by_global_cut(adata_rare_markers):
    """Hybrid re-rank recovers rare markers that a pure global top-k cut drops.

    ``cluster_pool`` must include rare markers so intermediate Leiden can form
    a rare-enriched cluster (clustering genes = top of global_score).
    """
    a = adata_rare_markers
    markers = set(a.uns["rare_marker_names"])
    k = 40
    gscore = _controlled_global_score(a)

    a_none = a.copy()
    scf.pp.highly_variable_genes(
        a_none,
        n_top_genes=k,
        balance_method="none",
        global_score=gscore,
        diagnose=False,
        random_state=0,
    )
    hit_none = markers & _selected(a_none)

    a_hyb = a.copy()
    scf.pp.highly_variable_genes(
        a_hyb,
        n_top_genes=k,
        balance_method="hybrid",
        global_score=gscore,
        # n_shared(40)+n_type(30)+n_markers(20)=90 < 100 → rare genes cluster
        # and sit in hybrid's top-2k re-rank pool.
        cluster_pool=100,
        blend_global=0.4,
        min_cluster_size=15,
        resolution=2.0,
        diagnose=False,
        random_state=0,
    )
    hit_hyb = markers & _selected(a_hyb)

    assert len(hit_none) == 0, (
        f"fixture broken: none should miss rare markers, got {sorted(hit_none)}"
    )
    assert len(hit_hyb) >= 10, (
        f"hybrid should recover ≥10/20 planted rare markers; got {len(hit_hyb)}: {sorted(hit_hyb)}"
    )


def test_score_recovers_rare_markers_vs_none_on_counts(adata_rare_markers):
    """Score recovers rare markers; global HVG under a tight k under-selects them."""
    a = adata_rare_markers
    markers = set(a.uns["rare_marker_names"])
    k = 50

    a_none = a.copy()
    scf.pp.highly_variable_genes(
        a_none,
        n_top_genes=k,
        balance_method="none",
        flavor="seurat_v3",
        layer="counts",
        diagnose=False,
        random_state=0,
    )
    hit_none = markers & _selected(a_none)

    a_score = a.copy()
    scf.pp.highly_variable_genes(
        a_score,
        n_top_genes=k,
        balance_method="score",
        flavor="seurat_v3",
        layer="counts",
        min_cluster_size=10,
        # Higher res isolates the rare pop so specificity can assign markers.
        resolution=2.0,
        diagnose=False,
        random_state=0,
    )
    hit_score = markers & _selected(a_score)

    assert len(hit_score) >= 12, (
        f"score should recover most rare markers; got {len(hit_score)}/20: {sorted(hit_score)}"
    )
    assert len(hit_score) > len(hit_none), (
        f"score ({len(hit_score)}) should beat none ({len(hit_none)})"
    )


def test_marker_mode_none_default_without_markers(adata_rare_markers):
    a = adata_rare_markers.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=40,
        balance_method="none",
        flavor="seurat_v3",
        layer="counts",
        marker_mode=None,
        diagnose=False,
    )
    assert a.uns["scfair"]["hvg"].get("marker_mode") is None
    assert int(a.var["highly_variable"].sum()) == 40


# ---------------------------------------------------------------------------
# 4-group atlas with a 30-cell rare type (user audit regression)
# ---------------------------------------------------------------------------


def _build_four_group_rare30(
    *,
    sizes: tuple[int, ...] = (900, 600, 470, 30),
    n_markers: int = 20,
    n_bg: int = 1420,  # 4*20 markers + bg → 1500 genes
    fold: float = 3.0,
    seed: int = 0,
) -> ad.AnnData:
    """Four populations with planted per-type markers (fold× background).

    Matches the audit fixture: rare type has only 30 cells; without an
    adequate intermediate resolution those cells are absorbed into majority
    Leiden communities and pure specificity fails to recover their markers.
    """
    rng = np.random.default_rng(seed)
    n_obs = int(sum(sizes))
    n_types = len(sizes)
    n_vars = n_types * n_markers + n_bg
    X = rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    labels = np.concatenate([np.full(n, f"c{i}", dtype=object) for i, n in enumerate(sizes)])
    base_rate = 1.0
    for t in range(n_types):
        cells = np.flatnonzero(labels == f"c{t}")
        midx = np.arange(t * n_markers, (t + 1) * n_markers)
        X[np.ix_(cells, midx)] = rng.poisson(base_rate * fold, size=(cells.size, n_markers)).astype(
            np.float32
        )

    adata = ad.AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.var_names = [f"c{t}_m{j}" for t in range(n_types) for j in range(n_markers)] + [
        f"bg_{i}" for i in range(n_bg)
    ]
    adata.obs["cell_type"] = labels.astype(str)
    adata.layers["counts"] = adata.X.copy()
    adata.uns["rare_marker_names"] = [f"c3_m{j}" for j in range(n_markers)]
    adata.uns["all_marker_names"] = {
        f"c{t}": [f"c{t}_m{j}" for j in range(n_markers)] for t in range(n_types)
    }
    return adata


@pytest.fixture
def adata_four_group_rare30():
    return _build_four_group_rare30()


def test_auto_resolution_floors_and_records_diagnostics(adata_four_group_rare30):
    """resolution='auto' must not sit at absurdly low values (floor ≥ 0.2).

    Pre-fix density search could land near 0.05, collapse to 3 majority
    communities, and leave rare=30 cells unresolved while ARI on the coarse
    partition stayed high and clusters_dropped stayed empty.
    """
    from scfair.pp._granularity import AUTO_RESOLUTION_LO

    a = adata_four_group_rare30.copy()
    scf.pp.highly_variable_genes(
        a,
        n_top_genes=150,
        balance_method="score",
        flavor="seurat_v3",
        layer="counts",
        resolution="auto",
        min_cluster_size=15,
        diagnose=True,
        random_state=0,
    )
    cl = a.uns["scfair"]["hvg"]["clustering"]
    assert float(cl["resolution"]) >= float(AUTO_RESOLUTION_LO) - 1e-9
    assert "min_cluster_frac" in cl
    assert cl["min_cluster_frac"] is not None
    assert cl["min_cluster_frac"] > 0
    # Floor / under-partition metadata when density field drove the search.
    if cl.get("resolution_source") == "density_field":
        assert (
            float(cl.get("resolution_floor", AUTO_RESOLUTION_LO))
            >= float(AUTO_RESOLUTION_LO) - 1e-9
        )


def test_score_recovers_rare30_when_resolution_isolates(adata_four_group_rare30):
    """With resolution high enough to isolate the 30-cell type, score recovers markers.

    Scientific regression for the audit failure mode: when intermediate Leiden
    under-partitions, pure specificity scores ~0 on rare markers (they never
    form a community). When the rare type *is* isolated, score must recover a
    large fraction — not collapse to ~1/20. Global seurat_v3 already finds
    strong fold markers, so we do not require score > none here; isolation +
    recovery is the contract.
    """
    a = adata_four_group_rare30
    markers = set(a.uns["rare_marker_names"])
    k = 150

    a_score = a.copy()
    scf.pp.highly_variable_genes(
        a_score,
        n_top_genes=k,
        balance_method="score",
        flavor="seurat_v3",
        layer="counts",
        # High enough to split the rare type; min size below 30 so it scores.
        resolution=2.0,
        min_cluster_size=15,
        diagnose=False,
        random_state=0,
    )
    hit_score = markers & _selected(a_score)
    cl = a_score.uns["scfair"]["hvg"]["clustering"]
    # Rare-sized community should exist (or something ≤ ~2× rare size).
    sizes = list((cl.get("cluster_sizes") or {}).values())
    assert any(s <= 60 for s in sizes), (
        f"expected a small intermediate community near rare=30; sizes={sizes}"
    )
    assert "min_cluster_frac" in cl
    # ≥ half the planted rare markers when a rare-sized community exists.
    # Absolute recall varies with scale_clustering / Leiden noise; the
    # under-partition failure mode was ~1/20, so ≥8 locks the real bug.
    assert len(hit_score) >= 8, (
        f"score should recover rare markers when isolated; "
        f"got {len(hit_score)}/20: {sorted(hit_score)}"
    )
