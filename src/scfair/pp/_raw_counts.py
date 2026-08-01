"""Private raw-count snapshot helpers used by ``highly_variable_genes``.

Not part of the public API. Users should only call
:func:`scfair.pp.highly_variable_genes`, which stores / restores counts
internally when the fair HVG pipeline subsets genes and needs the full
universe again.

Design (package-private):

- Axis-aligned copy in ``adata.layers[layer]`` (default ``"counts"``).
- Label-indexed sidecar in ``adata.uns['scfair']['raw_snapshot']`` that
  survives HVG gene subsetting and cell subsetting.
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
)
from .._version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _record_raw_counts_metadata(adata: Any, *, overwrite: bool = False) -> None:
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
    if adata.uns[UNS_KEY].get("raw_snapshot") is not None and not overwrite:
        logger.debug("raw_snapshot already exists; skipping.")
        return
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
        if os.path.exists(path) and not overwrite:
            logger.debug("On-disk snapshot %s already exists; reusing.", path)
        else:
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
        _record_raw_counts_metadata(adata, overwrite=overwrite)

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


def _restore_full_genes_from_snapshot(adata: Any) -> Any:
    """Build a new AnnData with the full pre-subset gene universe from the uns snapshot."""
    aligned = _align_snapshot_counts(adata, full_genes=True)
    if aligned is None:
        raise ValueError(
            f"Full-gene restore requires adata.uns['{UNS_KEY}']['raw_snapshot']. "
            "This should have been created automatically by highly_variable_genes."
        )
    X_full, var_names = aligned
    var_df = pd.DataFrame(index=pd.Index(var_names, name=adata.var_names.name))
    new = ad.AnnData(X=X_full, obs=adata.obs.copy(), var=var_df)
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


def _restore_raw_counts(
    adata: Any,
    layer: str = "counts",
    inplace: bool = False,
    full_genes: bool = False,
    prefer_snapshot: bool = True,
) -> Any | None:
    """Private: restore raw counts into ``.X`` (current genes or full snapshot)."""
    if full_genes:
        if inplace:
            raise ValueError("full_genes=True changes the gene axis and cannot be done inplace.")
        return _restore_full_genes_from_snapshot(adata)

    if prefer_snapshot and _get_raw_snapshot(adata) is not None:
        aligned = _align_snapshot_counts(adata, full_genes=False)
        if aligned is not None:
            snap_mat, _ = aligned
            target = adata if inplace else adata.copy()
            target.X = snap_mat.copy() if hasattr(snap_mat, "copy") else snap_mat
            _clear_log_preprocess_metadata(target)
            return None if inplace else target

    raw_attr = getattr(adata, "raw", None)
    if layer in adata.layers:
        raw = adata.layers[layer].copy()
        source = f"layers['{layer}']"
    elif raw_attr is not None and _is_integer_counts_like(raw_attr.X):
        raw = raw_attr.X.copy()
        source = "adata.raw"
    else:
        detail = ""
        if raw_attr is not None:
            detail = " adata.raw is present but does not look like integer counts and was not used."
        raise ValueError(
            f"No usable raw counts found in layer '{layer}' or the uns snapshot.{detail}"
        )

    if raw.shape[1] != adata.n_vars:
        raise ValueError(
            f"Stored raw counts have {raw.shape[1]} genes, but current adata has "
            f"{adata.n_vars} genes."
        )

    if source == "adata.raw" and hasattr(adata.raw, "var_names"):
        if not np.array_equal(adata.raw.var_names, adata.var_names):
            raise ValueError("adata.raw gene names/order differ from current adata.var_names.")
    elif source.startswith("layers"):
        raw_gene_list = adata.uns.get(UNS_KEY, {}).get("raw_gene_list")
        if (
            raw_gene_list is not None
            and len(raw_gene_list) == adata.n_vars
            and not np.array_equal(np.asarray(raw_gene_list), adata.var_names.to_numpy())
        ):
            raise ValueError(
                f"Stored counts in {source} match n_vars but raw_gene_list order differs "
                "from current adata.var_names."
            )

    target = adata if inplace else adata.copy()
    target.X = raw
    _clear_log_preprocess_metadata(target)
    return None if inplace else target


def _prepare_counts_layer(
    adata: Any,
    layer: str | None = None,
    *,
    counts_layer: str = "counts",
) -> str:
    """Ensure a usable counts layer + snapshot exist; return the layer name.

    Parameters
    ----------
    layer
        Explicit layer to use as counts. If None, prefer existing integer
        ``counts`` layer, else snapshot current ``.X`` into ``counts_layer``.
    counts_layer
        Default name when creating a new counts layer.
    """
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"layer={layer!r} not found in adata.layers.")
        if not _is_integer_counts_like(adata.layers[layer]):
            logger.warning(
                "layer=%r does not look like integer counts; HVG results may be unreliable.",
                layer,
            )
        # Refresh sidecar from this layer so full-gene restore works after subset.
        # overwrite=True is required: _store_raw_snapshot returns early if a
        # snapshot already exists, which would otherwise leave a stale matrix
        # in place on an object that already carries one (e.g. after subsetting).
        _store_raw_snapshot(adata, adata.layers[layer], overwrite=True)
        _record_raw_counts_metadata(adata)
        # An explicitly requested layer is always the layer actually used --
        # never silently substituted for an unrelated "counts" layer that
        # happens to exist.
        if layer != counts_layer and counts_layer not in adata.layers:
            adata.layers[counts_layer] = adata.layers[layer].copy()
        return layer

    # Prefer existing integer counts layer.
    if counts_layer in adata.layers and _is_integer_counts_like(adata.layers[counts_layer]):
        _store_raw_counts(adata, layer=counts_layer, mode="auto", overwrite=False)
        return counts_layer

    # Snapshot .X (or recover via auto mode).
    _store_raw_counts(adata, layer=counts_layer, mode="auto", overwrite=False)
    if counts_layer not in adata.layers:
        raise ValueError(
            "Could not prepare a counts layer for highly_variable_genes. "
            "Provide raw integer counts in adata.X or adata.layers['counts'], "
            "or pass layer= explicitly."
        )
    return counts_layer
