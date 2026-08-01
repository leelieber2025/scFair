"""Private raw-count helpers (not public API; used by highly_variable_genes)."""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp

from scfair._utils import UNS_KEY, _is_integer_counts_like, resolve_aligned_raw_counts
from scfair.pp._raw_counts import (
    _prepare_counts_layer,
    _restore_raw_counts,
    _store_raw_counts,
)


def test_store_and_restore_private(adata_counts):
    ad = adata_counts.copy()
    X_raw = np.asarray(ad.X, dtype=float).copy()
    _store_raw_counts(ad, layer="counts")

    assert "counts" in ad.layers
    assert "raw_gene_list" in ad.uns.get(UNS_KEY, {})
    assert ad.uns[UNS_KEY]["raw_snapshot"]["backend"] == "inline"

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    restored = _restore_raw_counts(ad, layer="counts", inplace=False)
    assert np.allclose(np.asarray(restored.X, dtype=float), X_raw)
    assert "log1p" not in restored.uns


def test_restore_realigns_permuted_gene_order(adata_counts):
    ad = adata_counts.copy()
    X_raw = np.asarray(ad.X, dtype=float).copy()
    _store_raw_counts(ad, layer="counts")
    perm = list(range(ad.n_vars))
    perm.reverse()
    ad_perm = ad[:, perm].copy()
    sc.pp.normalize_total(ad_perm, target_sum=1e4)

    restored = _restore_raw_counts(ad_perm, layer="counts", inplace=False)
    assert np.allclose(np.asarray(restored.X, dtype=float), X_raw[:, perm])


def test_restore_legacy_path_rejects_permuted_order(adata_counts):
    ad = adata_counts.copy()
    _store_raw_counts(ad, layer="counts")
    perm = list(range(ad.n_vars))
    perm.reverse()
    ad_perm = ad[:, perm].copy()
    sc.pp.normalize_total(ad_perm, target_sum=1e4)

    with pytest.raises(ValueError, match="raw_gene_list order differs"):
        _restore_raw_counts(ad_perm, layer="counts", inplace=False, prefer_snapshot=False)


def test_full_genes_after_subset(adata_counts):
    ad = adata_counts.copy()
    _store_raw_counts(ad, layer="counts")
    n_full = ad.n_vars
    hvg = ad[:, :8].copy()
    sc.pp.normalize_total(hvg, target_sum=1e4)
    sc.pp.log1p(hvg)

    full = _restore_raw_counts(hvg, full_genes=True)
    assert full.n_vars == n_full
    assert full.n_obs == hvg.n_obs
    assert list(full.var_names) == list(ad.var_names)


def test_full_genes_inplace_rejected(adata_counts):
    ad = adata_counts.copy()
    _store_raw_counts(ad)
    with pytest.raises(ValueError, match="cannot be done inplace"):
        _restore_raw_counts(ad, full_genes=True, inplace=True)


def test_duplicate_cell_names_rejected():
    X = sp.csr_matrix(np.random.default_rng(1).poisson(0.5, size=(6, 5)).astype("f4"))
    import anndata as anndata_mod

    ad = anndata_mod.AnnData(X=X)
    ad.obs_names = ["c1", "c1", "c2", "c3", "c3", "c4"]
    ad.var_names = [f"g{i}" for i in range(5)]
    _store_raw_counts(ad)
    with pytest.raises(ValueError, match="unique cell names"):
        _restore_raw_counts(ad)


def test_restore_refuses_log_normalized_raw(adata_counts):
    ad = adata_counts.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.raw = ad.copy()
    ad.layers.pop("counts", None)
    if UNS_KEY in ad.uns:
        ad.uns[UNS_KEY].pop("raw_snapshot", None)
    with pytest.raises(ValueError, match="integer counts|No usable raw"):
        _restore_raw_counts(ad, layer="counts")


def test_auto_mode_recovers_from_raw(adata_counts):
    ad = adata_counts.copy()
    ad.raw = ad.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    assert not _is_integer_counts_like(ad.X)

    _store_raw_counts(ad, mode="auto")
    assert "counts" in ad.layers
    assert _is_integer_counts_like(ad.layers["counts"])


def test_sparse_roundtrip(adata_counts_sparse):
    ad = adata_counts_sparse.copy()
    _store_raw_counts(ad)
    assert sp.issparse(ad.layers["counts"])
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    restored = _restore_raw_counts(ad)
    assert np.allclose(
        np.asarray(restored.X.todense() if sp.issparse(restored.X) else restored.X),
        np.asarray(ad.layers["counts"].todense()),
    )


def test_ondisk_sidecar(adata_counts, tmp_path):
    ad = adata_counts.copy()
    X_raw = np.asarray(ad.X, dtype=float).copy()
    path = tmp_path / "raw_snapshot.h5ad"
    _store_raw_counts(ad, sidecar="ondisk", snapshot_path=str(path))
    assert path.exists()
    assert ad.uns[UNS_KEY]["raw_snapshot"]["backend"] == "ondisk"

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    restored = _restore_raw_counts(ad)
    assert np.allclose(np.asarray(restored.X, dtype=float), X_raw)


def test_prepare_counts_layer(adata_counts):
    ad = adata_counts.copy()
    name = _prepare_counts_layer(ad)
    assert name == "counts"
    assert "counts" in ad.layers
    assert "raw_snapshot" in ad.uns[UNS_KEY]


def test_resolve_aligned_after_subset(adata_counts):
    ad = adata_counts.copy()
    _store_raw_counts(ad)
    sub = ad[:, :5].copy()
    mat = resolve_aligned_raw_counts(sub, layer="counts", require_integer=True)
    assert mat is not None
    assert mat.shape == (sub.n_obs, sub.n_vars)


def test_public_api_does_not_export_store_restore():
    import scfair as scf
    import scfair.pp as pp

    assert not hasattr(scf, "store_raw_counts")
    assert not hasattr(scf, "restore_raw_counts")
    assert not hasattr(pp, "store_raw_counts")
    assert not hasattr(pp, "restore_raw_counts")
    assert "store_raw_counts" not in scf.__all__
    assert "highly_variable_genes" in scf.__all__
    assert "highly_variable_genes" in pp.__all__


def test_explicit_layer_refreshes_a_stale_snapshot():
    """An explicit layer= must overwrite an existing snapshot."""
    import anndata as ad
    import numpy as np

    from scfair._utils import UNS_KEY
    from scfair.pp._raw_counts import _prepare_counts_layer

    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(30, 12)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"c{i}" for i in range(30)]
    adata.var_names = [f"g{i}" for i in range(12)]
    adata.layers["a"] = X.copy()
    adata.layers["b"] = (X * 3).astype(np.float32)

    used = _prepare_counts_layer(adata, layer="a")
    assert used == "a"
    snap_a = np.asarray(adata.uns[UNS_KEY]["raw_snapshot"]["X"])

    used = _prepare_counts_layer(adata, layer="b")
    assert used == "b", "an explicitly requested layer must be the one used"
    snap_b = np.asarray(adata.uns[UNS_KEY]["raw_snapshot"]["X"])

    assert not np.array_equal(snap_a, snap_b), "snapshot was not refreshed"
    assert np.array_equal(snap_b, np.asarray(adata.layers["b"]))


def test_explicit_layer_wins_over_an_existing_counts_layer():
    """An explicitly named layer must be used even when a 'counts' layer already exists."""
    import anndata as ad
    import numpy as np

    from scfair.pp._raw_counts import _prepare_counts_layer

    rng = np.random.default_rng(1)
    X = rng.poisson(2.0, size=(25, 10)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"c{i}" for i in range(25)]
    adata.var_names = [f"g{i}" for i in range(10)]
    adata.layers["raw"] = X.copy()
    adata.layers["counts"] = np.zeros_like(X)  # decoy

    assert _prepare_counts_layer(adata, layer="raw") == "raw"
