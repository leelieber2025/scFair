"""Private raw-count snapshot helpers used by ``highly_variable_genes``.

Not part of the public API. Users should only call
:func:`scfair.pp.highly_variable_genes`.

Design (package-private):

- Axis-aligned copy in ``adata.layers['counts']`` only when the source is
  integer-like (or recoverable from ``adata.raw``). Non-integer / log ``.X``
  is staged on :data:`INTERNAL_COUNTS_LAYER` and popped after the HVG call —
  never left as a permanent fake ``layers['counts']``.
- Optional label-indexed sidecar in ``adata.uns['scfair']['raw_snapshot']``
  only when ``store_raw=True`` / ``"ondisk"``. A later default call does
  **not** delete a snapshot the user already stored.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .._utils import (
    UNS_KEY,
    _align_snapshot_counts,
    _clear_log_preprocess_metadata,
    _get_raw_snapshot,
    _is_integer_counts_like,
    resolve_aligned_raw_counts,
)
from .._version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Ephemeral counts matrix for one HVG call. Never a user-facing layer name:
# written when default path must use .X without clobbering layers['counts'].
# Caller (highly_variable_genes) pops it after the run, like "_scfair_log".
INTERNAL_COUNTS_LAYER = "_scfair_counts"


def _snapshot_matches_matrix(snap: dict[str, Any], adata: Any, mat: Any) -> bool:
    """True when an existing snapshot is the same cells/genes/counts as ``mat``."""
    try:
        snap_obs = np.asarray(snap.get("obs_names"))
        snap_var = np.asarray(snap.get("var_names"))
        if snap_obs.shape[0] != adata.n_obs or snap_var.shape[0] != adata.n_vars:
            return False
        if not np.array_equal(snap_obs.astype(str), adata.obs_names.to_numpy().astype(str)):
            return False
        if not np.array_equal(snap_var.astype(str), adata.var_names.to_numpy().astype(str)):
            return False
        if snap.get("backend") == "ondisk":
            path = snap.get("path")
            if not path or not os.path.exists(path):
                return False
            return _ondisk_file_matches(
                path,
                adata.obs_names.to_numpy().astype(str),
                adata.var_names.to_numpy().astype(str),
                mat,
            )
        if "X" not in snap:
            return False
        return _matrix_fingerprint(snap["X"]) == _matrix_fingerprint(mat)
    except Exception:
        return False


def _ondisk_file_matches(
    path: str,
    obs_names: np.ndarray,
    var_names: np.ndarray,
    mat: Any,
) -> bool:
    """True when the h5ad at ``path`` is the same cells/genes/counts as ``mat``."""
    try:
        snap_ad = ad.read_h5ad(path)
    except Exception:
        return False
    if snap_ad.n_obs != len(obs_names) or snap_ad.n_vars != len(var_names):
        return False
    if not np.array_equal(np.asarray(snap_ad.obs_names).astype(str), np.asarray(obs_names)):
        return False
    if not np.array_equal(np.asarray(snap_ad.var_names).astype(str), np.asarray(var_names)):
        return False
    return _matrix_fingerprint(snap_ad.X) == _matrix_fingerprint(mat)


def _record_raw_counts_metadata(adata: Any) -> None:
    """Write scfair metadata (raw_gene_list)."""
    if UNS_KEY not in adata.uns:
        adata.uns[UNS_KEY] = {}
    prev_list = adata.uns[UNS_KEY].get("raw_gene_list")
    n_genes = int(adata.n_vars)
    if prev_list is not None and len(prev_list) != n_genes:
        logger.debug(
            "Updating raw_gene_list (%d → %d genes) after subset / re-store.",
            len(prev_list),
            n_genes,
        )
    if (
        prev_list is not None
        and len(prev_list) > n_genes
        and "raw_gene_list_full" not in adata.uns[UNS_KEY]
    ):
        adata.uns[UNS_KEY]["raw_gene_list_full"] = list(prev_list)
    adata.uns[UNS_KEY]["raw_gene_list"] = list(adata.var_names)
    adata.uns[UNS_KEY]["store_raw_counts_n_genes"] = n_genes


def _store_raw_snapshot(
    adata: Any,
    mat: Any,
    *,
    overwrite: bool = False,
    ondisk: bool = False,
    snapshot_path: str | None = None,
) -> None:
    """Write a label-indexed raw count snapshot into ``uns['scfair']``."""
    if UNS_KEY not in adata.uns:
        adata.uns[UNS_KEY] = {}
    existing = adata.uns[UNS_KEY].get("raw_snapshot")
    if existing is not None and not overwrite:
        if _snapshot_matches_matrix(existing, adata, mat):
            logger.debug("raw_snapshot already exists and matches; skipping.")
            return
        logger.debug("raw_snapshot exists but does not match current counts; refreshing.")
    if mat.shape[0] != adata.n_obs or mat.shape[1] != adata.n_vars:
        raise ValueError(
            f"Cannot store raw_snapshot: matrix shape {mat.shape} does not match "
            f"adata shape ({adata.n_obs}, {adata.n_vars})."
        )

    stored = mat.copy()
    obs_names = adata.obs_names.to_numpy().astype(str)
    var_names = adata.var_names.to_numpy().astype(str)
    is_integer = bool(_is_integer_counts_like(stored))

    if ondisk:
        if not snapshot_path:
            raise ValueError("sidecar='ondisk' requires snapshot_path=<file.h5ad>.")
        path = os.path.abspath(snapshot_path)
        reuse = False
        if os.path.exists(path) and not overwrite:
            reuse = _ondisk_file_matches(path, obs_names, var_names, stored)
            if reuse:
                logger.debug("On-disk snapshot %s already exists and matches; reusing.", path)
        if not reuse:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            snap_ad = ad.AnnData(
                X=stored,
                obs=pd.DataFrame(index=pd.Index(obs_names)),
                var=pd.DataFrame(index=pd.Index(var_names)),
            )
            snap_ad.write_h5ad(path)
        adata.uns[UNS_KEY]["raw_snapshot"] = {
            "backend": "ondisk",
            "path": path,
            "obs_names": obs_names,
            "var_names": var_names,
            "is_integer": is_integer,
            "version": __version__,
        }
        return

    adata.uns[UNS_KEY]["raw_snapshot"] = {
        "backend": "inline",
        "X": stored,
        "obs_names": obs_names,
        "var_names": var_names,
        "is_integer": is_integer,
        "version": __version__,
    }
    logger.debug(
        "Stored raw snapshot (%d cells x %d genes) in uns['%s']['raw_snapshot'].",
        adata.n_obs,
        adata.n_vars,
        UNS_KEY,
    )


def _sidecar_mode(sidecar: bool | str) -> tuple[bool, bool]:
    if isinstance(sidecar, str):
        if sidecar == "ondisk":
            return True, True
        raise ValueError(f"Unknown sidecar mode {sidecar!r}; use True, False, or 'ondisk'.")
    return bool(sidecar), False


def _store_raw_counts(
    adata: Any,
    layer: str = "counts",
    overwrite: bool = False,
    sidecar: bool | str = True,
    snapshot_path: str | None = None,
    mode: str = "force",
) -> None:
    """Private: snapshot raw counts into ``layers[layer]`` + optional uns sidecar."""
    if mode not in ("force", "auto"):
        raise ValueError(f"Unknown mode {mode!r}; use 'force' or 'auto'.")

    enabled, ondisk = _sidecar_mode(sidecar)

    def _finalize() -> None:
        if enabled and layer in adata.layers:
            _store_raw_snapshot(
                adata,
                adata.layers[layer],
                overwrite=overwrite,
                ondisk=ondisk,
                snapshot_path=snapshot_path,
            )
            _record_raw_counts_metadata(adata)

    if mode == "auto":
        if (
            layer in adata.layers
            and not overwrite
            and _is_integer_counts_like(adata.layers[layer])
            and adata.layers[layer].shape[1] == adata.n_vars
        ):
            _finalize()
            return
        if not _is_integer_counts_like(adata.X):
            raw = getattr(adata, "raw", None)
            if raw is not None and _is_integer_counts_like(raw.X) and raw.shape[1] == adata.n_vars:
                if hasattr(raw, "var_names") and np.array_equal(raw.var_names, adata.var_names):
                    adata.layers[layer] = raw.X.copy()
                    logger.debug("Recovered raw counts from adata.raw into layers['%s'].", layer)
                    _finalize()
                    return
            # fall through — may still store non-integer with a warning below

    if layer in adata.layers and not overwrite:
        if _is_integer_counts_like(adata.layers[layer]):
            _finalize()
            return
        logger.warning(
            "Existing layer '%s' does not look like raw counts; pass overwrite=True to replace it.",
            layer,
        )
        _finalize()
        return

    if not _is_integer_counts_like(adata.X):
        logger.warning(
            "adata.X does not look like raw integer counts. "
            "scfair.pp.highly_variable_genes expects raw counts in .X or layers['%s'] "
            "(call before normalize/log1p when possible).",
            layer,
        )
    mat = adata.X.copy()
    if mat.shape[1] != adata.n_vars:
        raise ValueError(
            f"Cannot store raw counts: matrix has {mat.shape[1]} columns "
            f"but adata has {adata.n_vars} genes."
        )
    adata.layers[layer] = mat
    _finalize()


def _restore_full_genes_from_snapshot(
    adata: Any, *, layer: str = "counts", prefer_snapshot: bool = True
) -> Any:
    """Build a new AnnData with the full pre-subset gene universe.

    Prefers ``uns['scfair']['raw_snapshot']`` (survives ``subset=True``) when
    it is a gene superset. A snapshot that is not larger than the current
    gene axis does not hide a bigger ``adata.raw``.
    """
    aligned = None
    if prefer_snapshot:
        try:
            aligned = _align_snapshot_counts(adata, full_genes=True)
        except ValueError as exc:
            if "unique" in str(exc).lower():
                raise
            aligned = None
    snap_n = None
    if aligned is not None:
        snap_n = int(len(aligned[1]))
    raw_attr = getattr(adata, "raw", None)
    raw_is_superset = (
        raw_attr is not None
        and raw_attr.X is not None
        and hasattr(raw_attr, "var_names")
        and int(getattr(raw_attr, "n_vars", raw_attr.X.shape[1])) > int(adata.n_vars)
        and int(getattr(raw_attr, "n_obs", raw_attr.X.shape[0])) == int(adata.n_obs)
        and _is_integer_counts_like(raw_attr.X)
    )
    if (
        aligned is not None
        and raw_is_superset
        and snap_n is not None
        and snap_n <= int(adata.n_vars)
        and int(raw_attr.n_vars) > snap_n
    ):
        aligned = None
    if aligned is None:
        raw_attr = getattr(adata, "raw", None)
        if raw_is_superset:
            X_full = raw_attr.X.copy() if hasattr(raw_attr.X, "copy") else raw_attr.X
            var_names = np.asarray(raw_attr.var_names)
        elif layer in getattr(adata, "layers", {}):
            X_full = adata.layers[layer].copy()
            var_names = adata.var_names.to_numpy()
        elif _is_integer_counts_like(adata.X):
            X_full = adata.X.copy()
            var_names = adata.var_names.to_numpy()
        else:
            raise ValueError(
                f"Full-gene restore requires adata.uns['{UNS_KEY}']['raw_snapshot'], "
                "integer adata.raw with the full gene axis, or integer counts in "
                f"layers[{layer!r}] / .X. "
                "Pass options=HVGOptions(store_raw=True) before subset=True, "
                "or keep raw counts in layers['counts'] / adata.raw."
            )
        var_df = pd.DataFrame(index=pd.Index(var_names, name=adata.var_names.name))
        new = ad.AnnData(X=X_full, obs=adata.obs.copy(), var=var_df)
        if layer not in new.layers:
            new.layers[layer] = X_full.copy() if hasattr(X_full, "copy") else X_full
        for key in adata.obsm:
            new.obsm[key] = adata.obsm[key].copy()
        _clear_log_preprocess_metadata(new)
        return new
    X_full, var_names = aligned
    var_df = pd.DataFrame(index=pd.Index(var_names, name=adata.var_names.name))
    new = ad.AnnData(X=X_full, obs=adata.obs.copy(), var=var_df)
    if layer not in new.layers:
        new.layers[layer] = X_full.copy() if hasattr(X_full, "copy") else X_full
    for key in adata.obsm:
        new.obsm[key] = adata.obsm[key].copy()

    new_uns = dict(adata.uns)
    meta = new_uns.get(UNS_KEY)
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["raw_gene_list"] = list(var_names)
        meta["store_raw_counts_n_genes"] = int(len(var_names))
        new_uns[UNS_KEY] = meta
    new.uns = new_uns

    _clear_log_preprocess_metadata(new)
    return new


def restore_raw_counts(
    adata: Any,
    *,
    layer: str = "counts",
    inplace: bool = False,
    full_genes: bool = False,
    prefer_snapshot: bool = True,
) -> Any | None:
    """Restore raw counts into ``.X`` from a snapshot or counts layer.

    Pair with ``options=HVGOptions(store_raw=True)`` (or ``"ondisk"``) when you
    need a full-gene universe after ``subset=True``, or to put integer counts
    back into ``.X`` after log normalization.

    Parameters
    ----------
    adata
        AnnData that may hold ``uns['scfair']['raw_snapshot']`` and/or
        ``layers[layer]``.
    layer
        Counts layer name when no snapshot is used (default ``"counts"``).
    inplace
        Write into ``adata.X`` when True (not allowed with ``full_genes=True``).
    full_genes
        If True, return a **new** AnnData with the full gene axis from a
        ``store_raw=True`` snapshot, or from ``adata.raw`` when that is a
        gene superset.
    prefer_snapshot
        Prefer ``uns['scfair']['raw_snapshot']`` over ``layers[layer]`` when both
        exist.

    Returns
    -------
    AnnData or None
        ``None`` when ``inplace=True``; otherwise a copy (or a new full-gene
        object when ``full_genes=True``).
    """
    return _restore_raw_counts(
        adata,
        layer=layer,
        inplace=inplace,
        full_genes=full_genes,
        prefer_snapshot=prefer_snapshot,
    )


def _restore_raw_counts(
    adata: Any,
    layer: str = "counts",
    inplace: bool = False,
    full_genes: bool = False,
    prefer_snapshot: bool = True,
) -> Any | None:
    """Internal restore implementation (also used by tests)."""
    if full_genes:
        if inplace:
            raise ValueError("full_genes=True changes the gene axis and cannot be done inplace.")
        return _restore_full_genes_from_snapshot(
            adata, layer=layer, prefer_snapshot=prefer_snapshot
        )

    if prefer_snapshot and _get_raw_snapshot(adata) is not None:
        try:
            aligned = _align_snapshot_counts(adata, full_genes=False)
        except ValueError as exc:
            # Duplicate names make label alignment unsafe — do not fall back.
            if "unique" in str(exc).lower():
                raise
            aligned = None
        if aligned is not None:
            snap_mat, _ = aligned
            target = adata if inplace else adata.copy()
            target.X = snap_mat.copy() if hasattr(snap_mat, "copy") else snap_mat
            if _is_integer_counts_like(snap_mat):
                _clear_log_preprocess_metadata(target)
            return None if inplace else target

    if layer in adata.layers and getattr(adata.layers[layer], "shape", (0, 0))[1] == adata.n_vars:
        raw = adata.layers[layer].copy()
        raw_gene_list = adata.uns.get(UNS_KEY, {}).get("raw_gene_list")
        if (
            raw_gene_list is not None
            and len(raw_gene_list) == adata.n_vars
            and not np.array_equal(np.asarray(raw_gene_list), adata.var_names.to_numpy())
        ):
            raise ValueError(
                f"Stored counts in layers['{layer}'] match n_vars but raw_gene_list "
                "order differs from current adata.var_names."
            )
    else:
        aligned = resolve_aligned_raw_counts(adata, layer=layer, require_integer=True)
        if aligned is None:
            raise ValueError(
                f"No usable raw counts found in layer '{layer}', uns snapshot, or adata.raw."
            )
        raw = aligned.copy() if hasattr(aligned, "copy") else aligned

    target = adata if inplace else adata.copy()
    target.X = raw
    if _is_integer_counts_like(raw):
        _clear_log_preprocess_metadata(target)
    return None if inplace else target


def _matrix_fingerprint(X: Any) -> tuple[Any, ...]:
    """Cheap content identity: shape + nnz + total + column-sum hash.

    Used to detect a stale ``layers['counts']`` after the user replaced ``.X``
    with a new integer count matrix without updating the layer. Column-sum hash
    (adler32 of float64 col sums) catches permutations / redistributions that
    keep the global sum unchanged.

    Storage format (dense vs sparse) is **not** part of the fingerprint: the
    same counts in CSR vs ndarray must compare equal so a format-only change
    does not bypass the user's counts layer.
    """
    import zlib

    import scipy.sparse as sparse

    if X is None:
        return ("none",)
    if sparse.issparse(X):
        # count_nonzero, not .nnz: explicit zeros / COO duplicates must not
        # make the same counts look different from a dense copy.
        nnz = int(np.count_nonzero(X.data)) if X.data.size else 0
        total = float(X.sum())
        col_sums = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
        shape = tuple(X.shape)
    else:
        arr = np.asarray(X)
        shape = tuple(arr.shape)
        if arr.size == 0:
            return (shape, 0, 0.0, 0)
        flat = arr.ravel()
        nnz = int(np.count_nonzero(flat))
        total = float(flat.sum())
        col_sums = (
            np.asarray(arr.sum(axis=0), dtype=np.float64).ravel()
            if arr.ndim == 2
            else np.asarray([total], dtype=np.float64)
        )
    # Stable across processes (unlike Python's salted hash()).
    col_hash = int(zlib.adler32(np.ascontiguousarray(col_sums).tobytes())) & 0xFFFFFFFF
    return (shape, nnz, total, col_hash)


def _prepare_counts_layer(
    adata: Any,
    layer: str | None = None,
    *,
    counts_layer: str = "counts",
    store_raw: bool | str = False,
    snapshot_path: str | None = None,
) -> str:
    """Ensure a usable counts layer exists; optionally write uns raw_snapshot.

    Parameters
    ----------
    layer
        Explicit layer to use as counts for **this call only**. Never copied
        into ``layers[counts_layer]`` (that used to permanently hijack later
        default calls). If None, prefer existing integer ``counts`` layer when
        it still matches integer-like ``.X``, else refresh/copy from ``.X``.
    counts_layer
        Default name when creating a new counts layer.
    store_raw
        ``False`` (default): only ensure ``layers[counts_layer]`` — no second
        full matrix in ``uns`` (avoids ~3× h5ad bloat). ``True``: inline
        snapshot; ``"ondisk"``: write h5ad sidecar at ``snapshot_path``.
    snapshot_path
        Required when ``store_raw="ondisk"``.
    """
    import warnings

    # Resolve sidecar mode for _store_raw_counts / _store_raw_snapshot.
    if store_raw is False or store_raw is None:
        sidecar: bool | str = False
    elif store_raw is True:
        sidecar = True
    elif isinstance(store_raw, str) and store_raw == "ondisk":
        sidecar = "ondisk"
    else:
        raise ValueError(f"store_raw must be False, True, or 'ondisk', got {store_raw!r}.")

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"layer={layer!r} not found in adata.layers.")
        if not _is_integer_counts_like(adata.layers[layer]):
            logger.warning(
                "layer=%r does not look like integer counts; HVG results may be unreliable.",
                layer,
            )
        if sidecar:
            # Snapshot the matrix *used this call* for subset/full-gene restore.
            # Do NOT also materialise it into layers['counts'] — that permanently
            # redirects later default (layer=None) calls away from .X.
            ondisk = sidecar == "ondisk"
            _store_raw_snapshot(
                adata,
                adata.layers[layer],
                overwrite=True,
                ondisk=ondisk,
                snapshot_path=snapshot_path,
            )
            if UNS_KEY not in adata.uns:
                adata.uns[UNS_KEY] = {}
            # Record which layer fed the snapshot so default path can tell a
            # one-off layer= snapshot apart from the canonical counts source.
            meta = adata.uns[UNS_KEY]
            if isinstance(meta.get("raw_snapshot"), dict):
                meta["raw_snapshot"] = dict(meta["raw_snapshot"])
                meta["raw_snapshot"]["source_layer"] = str(layer)
            _record_raw_counts_metadata(adata)
        # Use the explicit layer for this call only — never copy into
        # layers['counts'] (G1: permanent hijack of subsequent default calls).
        return layer

    # Prefer existing integer counts layer *if it still matches .X* when .X
    # also looks like raw counts. Standard scanpy pattern (.X log-normalized,
    # counts layer raw) keeps the layer: .X is not integer-like.
    if counts_layer in adata.layers and _is_integer_counts_like(adata.layers[counts_layer]):
        x_int = _is_integer_counts_like(adata.X)
        if x_int and _matrix_fingerprint(adata.layers[counts_layer]) != _matrix_fingerprint(
            adata.X
        ):
            # Do NOT overwrite the user layer (spliced/unspliced, dual assay,
            # ambient-corrected vs raw). Stage .X on an internal key instead.
            warnings.warn(
                f"layers[{counts_layer!r}] differs from the current integer-like "
                f".X (shape/nnz/column-sum fingerprint mismatch). Using .X for "
                f"this call via internal layer {INTERNAL_COUNTS_LAYER!r}; "
                f"layers[{counts_layer!r}] is left unchanged. "
                f"To use the layer instead, pass layer={counts_layer!r}. "
                "If .X is log-normalized, keep raw counts only in the layer "
                "(usual scanpy workflow) — that path still uses the layer.",
                UserWarning,
                stacklevel=3,
            )
            adata.layers[INTERNAL_COUNTS_LAYER] = adata.X.copy()
            if sidecar:
                ondisk = sidecar == "ondisk"
                _store_raw_snapshot(
                    adata,
                    adata.layers[INTERNAL_COUNTS_LAYER],
                    overwrite=True,
                    ondisk=ondisk,
                    snapshot_path=snapshot_path,
                )
                if UNS_KEY not in adata.uns:
                    adata.uns[UNS_KEY] = {}
                snap = adata.uns[UNS_KEY].get("raw_snapshot")
                if isinstance(snap, dict):
                    adata.uns[UNS_KEY]["raw_snapshot"] = dict(snap)
                    adata.uns[UNS_KEY]["raw_snapshot"]["source_layer"] = INTERNAL_COUNTS_LAYER
                _record_raw_counts_metadata(adata)
            return INTERNAL_COUNTS_LAYER
        _store_raw_counts(
            adata,
            layer=counts_layer,
            mode="auto",
            overwrite=False,
            sidecar=sidecar,
            snapshot_path=snapshot_path,
        )
        return counts_layer

    # Stale non-integer layer (log/CPM leftover) with integer .X: do not use
    # the layer for HVG and do not overwrite it. Stage .X internally.
    if counts_layer in adata.layers and _is_integer_counts_like(adata.X):
        warnings.warn(
            f"layers[{counts_layer!r}] does not look like raw integer counts, "
            f"but .X does. Using .X for this call via internal layer "
            f"{INTERNAL_COUNTS_LAYER!r}; layers[{counts_layer!r}] is left "
            "unchanged. Pass layer= to force a specific matrix.",
            UserWarning,
            stacklevel=3,
        )
        adata.layers[INTERNAL_COUNTS_LAYER] = adata.X.copy()
        if sidecar:
            ondisk = sidecar == "ondisk"
            _store_raw_snapshot(
                adata,
                adata.layers[INTERNAL_COUNTS_LAYER],
                overwrite=True,
                ondisk=ondisk,
                snapshot_path=snapshot_path,
            )
            if UNS_KEY not in adata.uns:
                adata.uns[UNS_KEY] = {}
            snap = adata.uns[UNS_KEY].get("raw_snapshot")
            if isinstance(snap, dict):
                adata.uns[UNS_KEY]["raw_snapshot"] = dict(snap)
                adata.uns[UNS_KEY]["raw_snapshot"]["source_layer"] = INTERNAL_COUNTS_LAYER
            _record_raw_counts_metadata(adata)
        return INTERNAL_COUNTS_LAYER

    # No usable counts layer. Prefer real integer counts in layers[counts_layer]
    # (.X integer, or recoverable from adata.raw). Never permanently write
    # log-normalized / non-integer .X as layers['counts'] — scvi-tools and
    # scanpy treat that name as raw UMI counts.
    x_int = _is_integer_counts_like(adata.X)
    raw_attr = getattr(adata, "raw", None)
    raw_ok = (
        raw_attr is not None
        and _is_integer_counts_like(raw_attr.X)
        and raw_attr.shape[1] == adata.n_vars
        and hasattr(raw_attr, "var_names")
        and np.array_equal(raw_attr.var_names, adata.var_names)
    )
    if x_int or raw_ok:
        _store_raw_counts(
            adata,
            layer=counts_layer,
            mode="auto",
            overwrite=False,
            sidecar=sidecar,
            snapshot_path=snapshot_path,
        )
        if counts_layer not in adata.layers:
            raise ValueError(
                "Could not prepare a counts layer for highly_variable_genes. "
                "Provide raw integer counts in adata.X or adata.layers['counts'], "
                "or pass layer= explicitly."
            )
        return counts_layer

    # Non-integer .X, no raw recovery: stage on the internal key only (same
    # policy as fingerprint-mismatch). Caller pops INTERNAL_COUNTS_LAYER.
    warnings.warn(
        f"No integer counts layer found and .X does not look like raw counts. "
        f"Using .X for this call via internal layer {INTERNAL_COUNTS_LAYER!r}; "
        f"not writing layers[{counts_layer!r}] (downstream tools treat that "
        f"name as raw UMI counts). Pass layer= or keep raw counts in "
        f"layers[{counts_layer!r}] / .X before normalize/log1p.",
        UserWarning,
        stacklevel=3,
    )
    adata.layers[INTERNAL_COUNTS_LAYER] = adata.X.copy()
    if sidecar:
        ondisk = sidecar == "ondisk"
        _store_raw_snapshot(
            adata,
            adata.layers[INTERNAL_COUNTS_LAYER],
            overwrite=True,
            ondisk=ondisk,
            snapshot_path=snapshot_path,
        )
        if UNS_KEY not in adata.uns:
            adata.uns[UNS_KEY] = {}
        snap = adata.uns[UNS_KEY].get("raw_snapshot")
        if isinstance(snap, dict):
            adata.uns[UNS_KEY]["raw_snapshot"] = dict(snap)
            adata.uns[UNS_KEY]["raw_snapshot"]["source_layer"] = INTERNAL_COUNTS_LAYER
        _record_raw_counts_metadata(adata)
    return INTERNAL_COUNTS_LAYER
