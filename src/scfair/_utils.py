"""Internal helpers for raw-count storage and alignment."""

from __future__ import annotations

import logging
import os
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse

logger = logging.getLogger(__name__)

# Namespace key for package metadata in adata.uns
UNS_KEY = "scfair"


def _is_integer_counts_like(X: Any, max_check: int = 100_000, atol: float = 1e-6) -> bool:
    """Return True if the matrix looks like non-negative integer counts.

    Tolerant of float storage of integer values (common after aggregation).
    Large matrices are subsampled deterministically (seed 0) for performance.
    """
    if sparse.issparse(X):
        data = X.data
        if data.size == 0:
            return True
        if not np.all(np.isfinite(data)):
            return False
        vals = data
    else:
        arr = np.asarray(X)
        if not np.all(np.isfinite(arr)):
            return False
        vals = arr.ravel()

    if vals.size == 0:
        return True

    if vals.size > max_check:
        rng = np.random.default_rng(0)
        stride = max(1, vals.size // (max_check // 2))
        stride_vals = vals[::stride]
        n_random = max_check - stride_vals.size
        if n_random > 0:
            random_vals = rng.choice(vals, size=min(n_random, vals.size), replace=False)
            vals = np.concatenate([stride_vals, random_vals])
        else:
            vals = stride_vals[:max_check]

    rounded = np.round(vals)
    return bool(np.all(vals >= 0) and np.allclose(vals, rounded, atol=atol, rtol=1e-5))


def _clear_log_preprocess_metadata(adata: ad.AnnData) -> None:
    """Drop scanpy log-transform markers after restoring raw counts into .X."""
    adata.uns.pop("log1p", None)


def _get_raw_snapshot(adata: ad.AnnData) -> dict[str, Any] | None:
    """Return the label-indexed raw snapshot from ``uns['scfair']``, or None."""
    meta = adata.uns.get(UNS_KEY)
    if not isinstance(meta, dict):
        return None
    snap = meta.get("raw_snapshot")
    if not isinstance(snap, dict):
        return None
    has_inline = "X" in snap
    is_ondisk = snap.get("backend") == "ondisk" and snap.get("path")
    if not (has_inline or is_ondisk):
        return None
    return snap


def _load_snapshot_matrix(snap: dict[str, Any]) -> Any:
    """Return the full count matrix for a snapshot, loading from disk if needed."""
    if snap.get("backend") == "ondisk":
        path = snap.get("path")
        if not path or not os.path.exists(path):
            raise ValueError(
                f"On-disk raw_snapshot file not found at {path!r}. It may have been moved "
                "or deleted; re-run store_raw_counts(sidecar='ondisk', snapshot_path=...)."
            )
        snap_ad = ad.read_h5ad(path)
        X = snap_ad.X
    else:
        X = snap["X"]

    if sparse.issparse(X):
        return X.tocsr()
    return X


def _slice_snapshot_matrix(mat: Any, row_idx: np.ndarray, col_idx: np.ndarray) -> Any:
    """Slice a snapshot matrix by row/column index arrays, preserving format."""
    if sparse.issparse(mat):
        return mat[row_idx][:, col_idx]
    return np.asarray(mat)[np.ix_(row_idx, col_idx)]


def _snapshot_alignment_indices(
    adata: ad.AnnData, snap: dict[str, Any], *, full_genes: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(row_idx, col_idx, var_names_out)`` aligning a snapshot to ``adata``."""
    snap_obs = pd.Index(np.asarray(snap["obs_names"]).astype(str))
    snap_var = pd.Index(np.asarray(snap["var_names"]).astype(str))

    if not snap_obs.is_unique or not adata.obs_names.is_unique:
        raise ValueError(
            "raw_snapshot restore requires unique cell names on both the snapshot and "
            "the current object. Call adata.obs_names_make_unique() before "
            "store_raw_counts() (and keep names stable) to enable label-aligned restore."
        )
    if not full_genes and (not snap_var.is_unique or not adata.var_names.is_unique):
        raise ValueError(
            "raw_snapshot restore requires unique gene names. Call "
            "adata.var_names_make_unique() before store_raw_counts()."
        )

    row_idx = snap_obs.get_indexer(adata.obs_names.astype(str))
    if (row_idx < 0).any():
        n_missing = int((row_idx < 0).sum())
        raise ValueError(
            f"raw_snapshot is missing {n_missing} of {adata.n_obs} current cells "
            "(obs_names not found). The snapshot may originate from a different object."
        )

    if full_genes:
        col_idx = np.arange(snap_var.size)
        var_names_out = snap_var.to_numpy()
    else:
        col_idx = snap_var.get_indexer(adata.var_names.astype(str))
        if (col_idx < 0).any():
            n_missing = int((col_idx < 0).sum())
            raise ValueError(
                f"raw_snapshot is missing {n_missing} of {adata.n_vars} current genes "
                "(var_names not found). Re-run store_raw_counts() on the current object."
            )
        var_names_out = adata.var_names.to_numpy()

    return row_idx, col_idx, var_names_out


def _align_snapshot_counts(
    adata: ad.AnnData,
    *,
    full_genes: bool = False,
    require_integer: bool = False,
) -> tuple[Any, np.ndarray] | None:
    """Align the ``uns`` raw count snapshot to ``adata`` by label.

    Returns ``(X, var_names)`` or None when no usable snapshot exists.
    """
    snap = _get_raw_snapshot(adata)
    if snap is None:
        return None
    if require_integer and not bool(snap.get("is_integer", False)):
        return None
    row_idx, col_idx, var_names_out = _snapshot_alignment_indices(
        adata, snap, full_genes=full_genes
    )
    X = _load_snapshot_matrix(snap)
    return _slice_snapshot_matrix(X, row_idx, col_idx), var_names_out


def resolve_aligned_raw_counts(
    adata: ad.AnnData,
    *,
    layer: str = "counts",
    require_integer: bool = True,
) -> Any | None:
    """Return a count matrix aligned to current cells/genes, or None if unsafe.

    Prefers the label-indexed uns snapshot (survives HVG/cell subset), then
    ``layers[layer]``, then integer-looking ``adata.raw``.
    """
    try:
        snap_result = _align_snapshot_counts(
            adata, full_genes=False, require_integer=require_integer
        )
    except ValueError as exc:
        logger.warning(
            "raw_snapshot present but could not be aligned (%s); falling back to "
            "layers/.raw for count resolution.",
            exc,
        )
        snap_result = None
    if snap_result is not None:
        snap_mat, _ = snap_result
        logger.debug("Using label-aligned raw_snapshot from uns.")
        return snap_mat

    candidates: list[tuple[str, Any]] = []
    if layer in adata.layers:
        candidates.append((f"layers['{layer}']", adata.layers[layer]))
    raw = getattr(adata, "raw", None)
    if (
        raw is not None
        and raw.shape[1] == adata.n_vars
        and hasattr(raw, "var_names")
        and np.array_equal(raw.var_names, adata.var_names)
    ):
        candidates.append(("adata.raw", raw.X))

    for source_name, mat in candidates:
        n_cols = mat.shape[1] if hasattr(mat, "shape") else 0
        if n_cols != adata.n_vars:
            logger.warning(
                "Counts from %s have %d columns but adata has %d genes; skipping.",
                source_name,
                n_cols,
                adata.n_vars,
            )
            continue
        if require_integer and not _is_integer_counts_like(mat):
            logger.warning(
                "Counts from %s do not look like raw integer counts; skipping.",
                source_name,
            )
            continue

        raw_gene_list = adata.uns.get(UNS_KEY, {}).get("raw_gene_list")
        if raw_gene_list is not None:
            stored = np.asarray(raw_gene_list)
            current = adata.var_names.to_numpy()
            if len(stored) == adata.n_vars and not np.array_equal(stored, current):
                logger.warning(
                    "Counts from %s match n_vars but stored raw_gene_list order differs from "
                    "adata.var_names. Refusing misaligned counts. "
                    "Re-run store_raw_counts() on the current object.",
                    source_name,
                )
                continue
            if len(stored) != adata.n_vars:
                logger.info(
                    "raw_gene_list (%d genes) differs from current n_vars (%d). "
                    "Using %s aligned to current genes.",
                    len(stored),
                    adata.n_vars,
                    source_name,
                )
        return mat

    return None
