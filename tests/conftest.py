"""Shared fixtures for scFair tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def adata_counts():
    """Small AnnData with integer Poisson counts (dense)."""
    rng = np.random.default_rng(0)
    X = rng.poisson(1.5, size=(40, 20)).astype(np.float32)
    obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    var_names = [f"gene_{i}" for i in range(X.shape[1])]
    adata = ad.AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obs["condition"] = ["A" if i % 2 == 0 else "B" for i in range(adata.n_obs)]
    return adata


@pytest.fixture
def adata_counts_sparse(adata_counts):
    """Same as adata_counts but with a CSR matrix."""
    ad_sp = adata_counts.copy()
    ad_sp.X = sp.csr_matrix(ad_sp.X)
    return ad_sp
