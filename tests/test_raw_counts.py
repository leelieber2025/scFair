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


def test_is_integer_counts_like_uses_atol_only():
    """rtol relative to magnitude would accept 100000.5; use atol only."""
    # Half-count at large magnitude must NOT be integer-like.
    big = np.array([[100000.5, 2.0], [3.0, 4.0]], dtype=float)
    assert not _is_integer_counts_like(big)
    # True integer floats still pass.
    ok = np.array([[100000.0, 2.0], [3.0, 4.0]], dtype=float)
    assert _is_integer_counts_like(ok)
    # Small fractional noise within atol=1e-6 still ok.
    noisy = np.array([[1.0 + 1e-7, 2.0]], dtype=float)
    assert _is_integer_counts_like(noisy)


def test_is_integer_counts_like_catches_trailing_fractional_sparse():
    """Isolated non-integer at the end of a large sparse .data must not pass.

    Full check covers typical sizes; above the threshold, pinned endpoints
    still catch a fractional last entry.
    """
    n = 200_000
    data = np.ones(n, dtype=np.float64)
    data[-1] = 0.5  # single fractional at the end
    indptr = np.arange(0, n + 1, dtype=np.int32)
    #  n cells × 1 gene sparse CSR with n nonzeros
    indices = np.zeros(n, dtype=np.int32)
    X = sp.csr_matrix((data, indices, indptr), shape=(n, 1))
    assert not _is_integer_counts_like(X)
    # Pure integers still pass at this size.
    data2 = np.ones(n, dtype=np.float64)
    X2 = sp.csr_matrix((data2, indices, indptr), shape=(n, 1))
    assert _is_integer_counts_like(X2)


def test_is_integer_counts_like_pins_endpoints_when_subsampled():
    """Above full_check_upto, first/mid/last pins catch local pollution."""
    n = 50_000
    data = np.ones(n, dtype=np.float64)
    data[-1] = 0.5
    indptr = np.arange(0, n + 1, dtype=np.int32)
    indices = np.zeros(n, dtype=np.int32)
    X = sp.csr_matrix((data, indices, indptr), shape=(n, 1))
    # Force subsample path with a tiny full_check threshold.
    assert not _is_integer_counts_like(X, full_check_upto=1000, max_check=500)
    data[0] = -1.0
    data[-1] = 1.0
    Xneg = sp.csr_matrix((data, indices, indptr), shape=(n, 1))
    assert not _is_integer_counts_like(Xneg, full_check_upto=1000, max_check=500)


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
    # Default: counts layer only, no uns raw_snapshot (avoids 3× h5ad bloat).
    name = _prepare_counts_layer(ad)
    assert name == "counts"
    assert "counts" in ad.layers
    assert "raw_snapshot" not in ad.uns.get(UNS_KEY, {})


def test_prepare_counts_layer_store_raw(adata_counts):
    ad = adata_counts.copy()
    name = _prepare_counts_layer(ad, store_raw=True)
    assert name == "counts"
    assert "raw_snapshot" in ad.uns[UNS_KEY]
    assert ad.uns[UNS_KEY]["raw_snapshot"]["backend"] == "inline"


def test_resolve_aligned_after_subset(adata_counts):
    ad = adata_counts.copy()
    _store_raw_counts(ad)
    sub = ad[:, :5].copy()
    mat = resolve_aligned_raw_counts(sub, layer="counts", require_integer=True)
    assert mat is not None
    assert mat.shape == (sub.n_obs, sub.n_vars)


def test_resolve_aligned_from_raw_after_gene_subset(adata_counts):
    """adata.raw with more genes than current object is the normal scanpy case."""
    ad = adata_counts.copy()
    X_full = np.asarray(ad.X, dtype=float).copy()
    ad.raw = ad.copy()
    # Subset genes; drop counts layer and snapshot so only .raw remains.
    sub = ad[:, :5].copy()
    sub.layers.pop("counts", None)
    if UNS_KEY in sub.uns:
        sub.uns[UNS_KEY].pop("raw_snapshot", None)
        sub.uns[UNS_KEY].pop("raw_gene_list", None)
    mat = resolve_aligned_raw_counts(sub, layer="counts", require_integer=True)
    assert mat is not None
    assert mat.shape == (sub.n_obs, 5)
    assert np.allclose(np.asarray(mat, dtype=float), X_full[:, :5])


def test_public_api_exports_restore_not_store():
    import scfair as scf
    import scfair.pp as pp

    # store remains private (via HVGOptions.store_raw); restore is public.
    assert not hasattr(scf, "store_raw_counts")
    assert not hasattr(pp, "store_raw_counts")
    assert "store_raw_counts" not in scf.__all__
    assert hasattr(scf, "restore_raw_counts")
    assert hasattr(pp, "restore_raw_counts")
    assert "restore_raw_counts" in scf.__all__
    assert "restore_raw_counts" in pp.__all__
    assert "highly_variable_genes" in scf.__all__
    assert "highly_variable_genes" in pp.__all__
    # Implementation details not exported
    assert "recommend_cluster_resolution" not in pp.__all__
    assert "estimate_n_top_structure" not in pp.__all__


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

    used = _prepare_counts_layer(adata, layer="a", store_raw=True)
    assert used == "a"
    snap_a = np.asarray(adata.uns[UNS_KEY]["raw_snapshot"]["X"])

    used = _prepare_counts_layer(adata, layer="b", store_raw=True)
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
    # Must not overwrite or replace the existing decoy counts layer.
    assert np.array_equal(adata.layers["counts"], np.zeros_like(X))


def test_matrix_fingerprint_uses_column_sums():
    """Same global sum but different column distribution must not collide."""
    from scfair.pp._raw_counts import _matrix_fingerprint

    a = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    b = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
    assert a.sum() == b.sum()
    assert a.shape == b.shape
    assert int(np.count_nonzero(a)) == int(np.count_nonzero(b))
    assert _matrix_fingerprint(a) != _matrix_fingerprint(b)
    assert _matrix_fingerprint(a) == _matrix_fingerprint(a.copy())


def test_matrix_fingerprint_ignores_storage_format():
    """Dense vs CSR of the same counts must fingerprint equal (no false mismatch)."""
    import scipy.sparse as sp

    from scfair.pp._raw_counts import (
        INTERNAL_COUNTS_LAYER,
        _matrix_fingerprint,
        _prepare_counts_layer,
    )

    rng = np.random.default_rng(7)
    dense = rng.poisson(2.0, size=(50, 30)).astype(np.float32)
    sparse = sp.csr_matrix(dense)
    assert _matrix_fingerprint(dense) == _matrix_fingerprint(sparse)

    # End-to-end: mixed format with identical content must keep layers['counts'].
    import anndata as ad

    adata = ad.AnnData(X=sparse.copy())
    adata.obs_names = [f"c{i}" for i in range(50)]
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.layers["counts"] = dense.copy()  # same values, dense storage
    used = _prepare_counts_layer(adata)
    assert used == "counts"
    assert INTERNAL_COUNTS_LAYER not in adata.layers


def test_explicit_layer_does_not_materialize_into_counts():
    """layer='alt' must not permanently hijack later default (layer=None) calls."""
    import anndata as ad
    import numpy as np

    from scfair.pp._raw_counts import _prepare_counts_layer

    rng = np.random.default_rng(2)
    X = rng.poisson(1.0, size=(40, 15)).astype(np.float32)
    alt = rng.poisson(5.0, size=(40, 15)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"c{i}" for i in range(40)]
    adata.var_names = [f"g{i}" for i in range(15)]
    adata.layers["alt"] = alt

    assert _prepare_counts_layer(adata, layer="alt") == "alt"
    assert "counts" not in adata.layers

    used = _prepare_counts_layer(adata)  # default
    assert used == "counts"
    # Default path materialises from .X, not from the prior layer='alt' call.
    assert np.array_equal(np.asarray(adata.layers["counts"]), np.asarray(X))


def test_stale_counts_uses_internal_layer_preserves_user_counts():
    """Fingerprint mismatch: use .X via _scfair_counts; never clobber user layer."""
    import anndata as ad
    import numpy as np
    import pytest

    from scfair.pp._raw_counts import INTERNAL_COUNTS_LAYER, _prepare_counts_layer

    rng = np.random.default_rng(3)
    X1 = rng.poisson(1.0, size=(40, 15)).astype(np.float32)
    X2 = rng.poisson(8.0, size=(40, 15)).astype(np.float32)
    counts_b = rng.poisson(3.0, size=(40, 15)).astype(np.float32)
    adata = ad.AnnData(X=X1)
    adata.obs_names = [f"c{i}" for i in range(40)]
    adata.var_names = [f"g{i}" for i in range(15)]

    assert _prepare_counts_layer(adata) == "counts"
    # User replaces .X and also keeps a distinct intentional counts matrix.
    adata.X = X2
    adata.layers["counts"] = counts_b.copy()
    with pytest.warns(UserWarning, match="left unchanged|internal layer"):
        used = _prepare_counts_layer(adata)
    assert used == INTERNAL_COUNTS_LAYER
    # User layer must be byte-identical to what they put there.
    assert np.array_equal(np.asarray(adata.layers["counts"]), np.asarray(counts_b))
    assert np.array_equal(np.asarray(adata.layers[INTERNAL_COUNTS_LAYER]), np.asarray(X2))


def test_intentional_dual_integer_counts_not_destroyed():
    """Both .X and layers['counts'] integer but different — preserve the layer."""
    import anndata as ad
    import numpy as np
    import pytest

    from scfair.pp._raw_counts import INTERNAL_COUNTS_LAYER, _prepare_counts_layer

    rng = np.random.default_rng(5)
    counts_a = rng.poisson(1.0, size=(30, 12)).astype(np.float32)
    counts_b = rng.poisson(7.0, size=(30, 12)).astype(np.float32)
    adata = ad.AnnData(X=counts_a.copy())
    adata.obs_names = [f"c{i}" for i in range(30)]
    adata.var_names = [f"g{i}" for i in range(12)]
    adata.layers["counts"] = counts_b.copy()

    with pytest.warns(UserWarning, match="left unchanged"):
        used = _prepare_counts_layer(adata)
    assert used == INTERNAL_COUNTS_LAYER
    assert np.array_equal(np.asarray(adata.layers["counts"]), np.asarray(counts_b))
    # Escape hatch still works.
    assert _prepare_counts_layer(adata, layer="counts") == "counts"
    assert np.array_equal(np.asarray(adata.layers["counts"]), np.asarray(counts_b))


def test_log_normalized_X_keeps_counts_layer():
    """Usual scanpy pattern: raw in counts, log1p in .X — do not overwrite counts."""
    import anndata as ad
    import numpy as np
    import scanpy as sc

    from scfair.pp._raw_counts import _prepare_counts_layer

    rng = np.random.default_rng(4)
    X = rng.poisson(2.0, size=(40, 15)).astype(np.float32)
    adata = ad.AnnData(X=X.copy())
    adata.obs_names = [f"c{i}" for i in range(40)]
    adata.var_names = [f"g{i}" for i in range(15)]
    adata.layers["counts"] = X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    assert _prepare_counts_layer(adata) == "counts"
    assert np.array_equal(np.asarray(adata.layers["counts"]), np.asarray(X))


def test_log_X_without_counts_uses_internal_layer():
    """Log .X and no counts layer must not invent layers['counts'] from log data."""
    import anndata as ad
    import numpy as np
    import pytest
    import scanpy as sc

    from scfair.pp._raw_counts import INTERNAL_COUNTS_LAYER, _prepare_counts_layer

    rng = np.random.default_rng(6)
    X = rng.poisson(2.0, size=(40, 15)).astype(np.float32)
    adata = ad.AnnData(X=X.copy())
    adata.obs_names = [f"c{i}" for i in range(40)]
    adata.var_names = [f"g{i}" for i in range(15)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    with pytest.warns(UserWarning, match="internal layer|not writing"):
        used = _prepare_counts_layer(adata)
    assert used == INTERNAL_COUNTS_LAYER
    assert "counts" not in adata.layers
    assert INTERNAL_COUNTS_LAYER in adata.layers
