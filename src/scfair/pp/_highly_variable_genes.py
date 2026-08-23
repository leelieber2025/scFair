"""Highly variable gene selection: structure-aware auto-``n`` + cutoff append.

Public entry point: :func:`highly_variable_genes`.

1. **How many genes (``n``)?** Default ``n_top_genes="auto"`` estimates a base
   size ``k`` from multi-seed density structure. Pass a fixed int when the
   protocol already locks ``n``.
2. **Hard top-``k`` cutoff.** Global HVG ranking is dominated by large
   populations; genes useful for smaller types often sit just below top-``k``.
   Default ``balance_method="append"`` keeps the global top-``k`` and adds a
   short same-rank tail (``append_budget``). The set is ``top-(k+m)``.
   Pass ``balance_method="none"`` (or ``append_budget=0``) for pure top-``k``.
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse

from .._utils import UNS_KEY, _is_integer_counts_like
from ._auto_n import plain_auto_n_message
from ._diagnosis import check_config, diagnose_hvg_run, resolve_hvg_mode
from ._options import HVGOptions, resolve_hvg_options
from ._raw_counts import (
    INTERNAL_COUNTS_LAYER,
    _prepare_counts_layer,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_COUNTS_FLAVORS = frozenset({"seurat_v3", "seurat_v3_paper", "pearson_residuals"})
_LOG_FLAVORS = frozenset({"seurat", "cell_ranger"})
_ALL_FLAVORS = _COUNTS_FLAVORS | _LOG_FLAVORS

_BALANCE_ALIASES = {
    None: "append",
    "append": "append",
    "none": "none",
}

BalanceMethod = Literal["append", "none"]

# Auto strategies accepted by the product entry point.
_AUTO_STRATEGIES = frozenset({"auto", "structure"})

# Recoverable failures for HVG paths (not KeyboardInterrupt / SystemExit).
_RECOVERABLE_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    ModuleNotFoundError,
    ValueError,
    TypeError,
    RuntimeError,
    ArithmeticError,
    MemoryError,
    FloatingPointError,
    np.linalg.LinAlgError,
)

_MITO_HUMAN_RE = re.compile(r"^(MT-|MT\.)")
_MITO_MOUSE_RE = re.compile(r"^(mt-|mt\.)")
_RIBO_RE = re.compile(
    r"^RPLP?\d+[A-Z]?$|^RPS(?:A|\d+[A-Z]?)$",
    re.IGNORECASE,
)
_RIBO_MOUSE_SIGNAL_RE = re.compile(r"^Rp[ls]")
_RIBO_HUMAN_SIGNAL_RE = re.compile(r"^RP[LS][A-Z0-9]")

_HVG_PARTIAL_VAR_COLS = (
    "highly_variable",
    "highly_variable_rank",
    "means",
    "variances",
    "variances_norm",
    "dispersions",
    "dispersions_norm",
    "residual_variances",
    "highly_variable_nbatches",
    "highly_variable_intersection",
    "scfair_score",
)

_FLAVOR_SCORE_COLS: dict[str, tuple[str, ...]] = {
    "seurat_v3": ("variances_norm", "variances"),
    "seurat_v3_paper": ("variances_norm", "variances"),
    "pearson_residuals": ("residual_variances",),
    "seurat": ("dispersions_norm", "dispersions"),
    "cell_ranger": ("dispersions_norm", "dispersions"),
}

# seurat_v3 loess (skmisc) can SIGSEGV in C on tiny matrices. Not catchable.
_LOESS_MIN_OBS = 2


def _loess_unsafe(n_obs: int, n_vars: int, span: float = 0.3) -> str | None:
    """Return a reason if seurat_v3 loess is unsafe, else None."""
    span_f = float(span) if span and float(span) > 0 else 0.3
    min_genes = max(10, int(np.ceil(2.0 / span_f)))
    n_obs_i = int(n_obs)
    n_vars_i = int(n_vars)
    if n_obs_i < _LOESS_MIN_OBS:
        return f"n_obs={n_obs_i}<{_LOESS_MIN_OBS}"
    if n_vars_i < min_genes:
        return f"n_vars={n_vars_i}<{min_genes}"
    return None


_SCANPY_HVG_VAR_COLS = (
    "highly_variable",
    "highly_variable_rank",
    "means",
    "variances",
    "variances_norm",
    "dispersions",
    "dispersions_norm",
    "residual_variances",
    "highly_variable_nbatches",
    "highly_variable_intersection",
)


def highly_variable_genes(
    adata: Any,
    *,
    n_top_genes: int | str = "auto",
    flavor: str = "seurat_v3",
    layer: str | None = None,
    balance_method: str | None = "append",
    mode: str = "auto",
    marker_genes: Sequence[str] | None = None,
    marker_mode: str | None = None,
    diagnose: bool = True,
    strict: bool = False,
    random_state: int = 0,
    inplace: bool = True,
    subset: bool = False,
    progress: bool | None = None,
    options: HVGOptions | None = None,
    **legacy_kwargs: Any,
) -> pd.DataFrame | None:
    """Select highly variable genes (structure auto-``n`` + cutoff append).

    Parameters
    ----------
    adata
        AnnData containing the input expected by ``flavor``. The count-based
        flavors require raw integer counts in ``.X`` or a counts layer (or pass
        ``layer=``); ``seurat`` and ``cell_ranger`` expect log-transformed data.
    n_top_genes
        ``"auto"`` / ``"structure"`` (**default**): structure-aware base size
        from multi-seed density features. Pass a fixed int (e.g. ``2000``)
        when the protocol is locked. Booleans are rejected. Auto is multi-seed
        and slower than a fixed ``k``.
    flavor
        Scanpy HVG method for the global ranking:
        counts: ``seurat_v3``, ``seurat_v3_paper``, ``pearson_residuals``;
        log: ``seurat``, ``cell_ranger``.
    balance_method
        ``"append"`` (default): keep global top-``k``, then add the next
        ``append_budget`` genes from the **same** ranking (near-miss genes
        kept). The set equals ``top-(k+m)``. ``"none"``: top-``k`` only
        (scanpy-like fixed list size).
    mode
        ``"auto"`` / ``"compact"`` / ``"balanced"`` / ``"fine"`` — steers
        auto-``k`` floors and default append budget when not set explicitly.
        ``None`` is treated as ``"auto"``.
    marker_genes, marker_mode
        Optional forced markers. Default mode is ``"force"`` when markers
        are given, else ``"none"``.
    diagnose
        Write advisory ``uns['scfair']['hvg']['diagnosis']`` (does not change genes).
    strict
        If True, do not fall back when seurat_v3 or structure auto fails.
    progress
        Stage messages on stderr. ``None`` enables for large matrices and for
        auto-``n`` once ``n_obs >= 1000`` (so mid-size auto runs are not silent).
    options
        :class:`~scfair.pp.HVGOptions` for secondary knobs (bounds, filters,
        append_budget, store_raw, batch_key, …). With ``store_raw=True``,
        restore via :func:`~scfair.pp.restore_raw_counts`.
    """
    if getattr(adata, "isbacked", False):
        raise NotImplementedError(
            "backed AnnData is not supported by scfair.pp.highly_variable_genes "
            "(the pipeline needs repeated full-matrix access). "
            "Call adata = adata.to_memory() first, then re-run."
        )

    if legacy_kwargs:
        from ._options import _REMOVED_OPTION_NAMES

        removed = sorted(k for k in legacy_kwargs if k in _REMOVED_OPTION_NAMES)
        if removed:
            raise TypeError(f"removed option(s): {removed}. Use balance_method='append' or 'none'.")
        raise TypeError(
            f"highly_variable_genes() got unexpected keyword argument(s) "
            f"{sorted(legacy_kwargs)}. Secondary knobs use options=HVGOptions(...)."
        )
    opt = resolve_hvg_options(options)

    n_top_min = int(opt.n_top_min)
    n_top_max = int(opt.n_top_max)
    marker_extra = bool(opt.marker_extra)
    global_score = opt.global_score
    span = float(opt.span)
    n_bins = int(opt.n_bins)
    min_mean = float(opt.min_mean)
    max_mean = float(opt.max_mean)
    min_disp = float(opt.min_disp)
    max_disp = float(opt.max_disp)
    batch_key = opt.batch_key
    filter_mito = bool(opt.filter_mito)
    filter_ribo = bool(opt.filter_ribo)
    gene_nomenclature = opt.gene_nomenclature
    if gene_nomenclature is not None:
        gene_nomenclature = str(gene_nomenclature).lower().strip()
        if gene_nomenclature not in ("human", "mouse", "mixed", "unknown"):
            raise ValueError(
                f"gene_nomenclature must be None or one of "
                f"'human', 'mouse', 'mixed', 'unknown'; got {gene_nomenclature!r}."
            )
    user_append_budget = opt.append_budget
    if user_append_budget is not None:
        append_budget = int(user_append_budget)
        if append_budget < 0:
            raise ValueError(f"append_budget must be >= 0, got {append_budget}.")
    else:
        # Product floor; may rise after structure auto via density cores.
        from ._auto_n import APPEND_BUDGET_FLOOR

        append_budget = int(APPEND_BUDGET_FLOOR)
    label_key = opt.label_key
    store_raw = opt.store_raw
    snapshot_path = opt.snapshot_path
    if store_raw and not inplace:
        warnings.warn(
            "store_raw is ignored when inplace=False (the snapshot would be "
            "written on a discarded copy). Call with inplace=True.",
            UserWarning,
            stacklevel=2,
        )
        store_raw = False
    structure_n_seeds_opt = opt.structure_n_seeds
    if structure_n_seeds_opt is not None:
        structure_n_seeds_opt = int(structure_n_seeds_opt)
        if structure_n_seeds_opt < 1:
            raise ValueError(f"structure_n_seeds must be >= 1, got {structure_n_seeds_opt}.")

    mode_arg = "auto" if mode is None else mode
    mode_info = resolve_hvg_mode(
        adata,
        mode=str(mode_arg),
        label_key=label_key,
        n_obs=int(getattr(adata, "n_obs", 0) or 0) or None,
    )
    hvg_mode = str(mode_info["mode"])
    # Mode may suggest a floor budget; density-tight rule (auto path) overrides
    # when the user did not set append_budget explicitly.
    if user_append_budget is None and mode_info.get("append_budget") is not None:
        append_budget = int(mode_info["append_budget"])
    append_budget_requested = int(append_budget)
    append_budget_capped = False
    append_budget_info: dict[str, Any] | None = None

    if not isinstance(strict, bool):
        raise TypeError(f"strict must be bool, got {type(strict).__name__}={strict!r}.")
    if isinstance(n_top_genes, bool):
        raise TypeError(
            f"n_top_genes must be an int or 'auto'/'structure', not bool ({n_top_genes!r})."
        )
    if isinstance(n_top_genes, (int, np.integer)) and int(n_top_genes) < 1:
        raise ValueError("n_top_genes must be >= 1.")
    if flavor not in _ALL_FLAVORS:
        raise ValueError(f"Unknown flavor={flavor!r}. Supported: {sorted(_ALL_FLAVORS)}.")
    if balance_method not in _BALANCE_ALIASES:
        raise ValueError(
            f"Unknown balance_method={balance_method!r}. "
            "Use 'append' (default same-rank cutoff tail) or 'none' (top-k only)."
        )
    method = _BALANCE_ALIASES[balance_method]

    if marker_mode is None:
        marker_mode = "force" if marker_genes else "none"
    elif marker_mode not in ("force", "none"):
        raise ValueError("marker_mode must be None, 'force', or 'none'.")

    config_check = check_config(
        n_top_genes=n_top_genes,
        balance_method=method,
        log=diagnose,
    )

    dup = adata.var_names[adata.var_names.duplicated()]
    if len(dup):
        raise ValueError(
            f"adata.var_names has {len(dup)} duplicate entries "
            f"(e.g. {sorted(set(map(str, dup)))[:3]}). "
            "Call adata.var_names_make_unique() first."
        )
    dup_obs = adata.obs_names[adata.obs_names.duplicated()]
    if len(dup_obs):
        raise ValueError(
            f"adata.obs_names has {len(dup_obs)} duplicate entries "
            f"(e.g. {sorted(set(map(str, dup_obs)))[:3]}). "
            "Call adata.obs_names_make_unique() first."
        )
    if n_top_min < 1 or n_top_max < 1:
        raise ValueError(f"n_top_min={n_top_min} and n_top_max={n_top_max} must both be >= 1.")
    if n_top_min > n_top_max:
        raise ValueError(
            f"n_top_min={n_top_min} > n_top_max={n_top_max}. Pass a consistent interval."
        )

    if subset and not inplace:
        warnings.warn(
            "subset=True is ignored when inplace=False (the return value is a "
            "DataFrame, not AnnData). Use inplace=True to subset the object, "
            "or apply the returned highly_variable mask yourself.",
            UserWarning,
            stacklevel=2,
        )

    if not inplace:
        adata = adata.copy()

    _hvg_snap = _snapshot_adata_for_rollback(adata)
    effective_store_raw: bool | str = store_raw
    try:
        counts_layer = _prepare_counts_layer(
            adata,
            layer=layer,
            counts_layer="counts",
            store_raw=effective_store_raw,
            snapshot_path=snapshot_path,
        )
        counts_validate_info = _validate_counts_matrix(
            adata,
            counts_layer=counts_layer,
            flavor=flavor,
            span=span,
            strict=strict,
        )

        _prior_scanpy_hvg = (
            dict(adata.uns["hvg"]) if isinstance(adata.uns.get("hvg"), dict) else None
        )

        auto_meta: dict[str, Any] | None = None
        if not isinstance(n_top_genes, (int, np.integer)):
            try:
                as_float = float(n_top_genes)
            except (TypeError, ValueError):
                pass
            else:
                if np.isfinite(as_float) and float(as_float).is_integer():
                    n_top_genes = int(as_float)
        n_top_is_auto = not isinstance(n_top_genes, (int, np.integer))
        _mode_arg = str(mode or "auto").lower().strip()
        if (
            not n_top_is_auto
            and _mode_arg not in ("auto", "")
            and _mode_arg in ("compact", "balanced", "fine")
        ):
            warnings.warn(
                f"mode={mode!r} has no effect on gene selection when "
                f"n_top_genes is a fixed int ({int(n_top_genes)}). "
                "Pass n_top_genes='auto' to let mode influence k.",
                UserWarning,
                stacklevel=2,
            )
        if n_top_is_auto and str(n_top_genes).lower() not in _AUTO_STRATEGIES:
            raise ValueError(
                f"Unknown n_top_genes={n_top_genes!r}. Pass an int, 'auto', or 'structure'."
            )
        if n_top_is_auto:
            n_top_request = min(int(n_top_max), adata.n_vars)
        else:
            n_top_request = min(int(n_top_genes), adata.n_vars)

        if method == "append" and not n_top_is_auto:
            _cap = min(int(append_budget_requested), max(0, int(n_top_request)))
            if _cap != int(append_budget_requested):
                append_budget_capped = True
                warnings.warn(
                    f"append_budget={append_budget_requested} exceeds n_top_genes="
                    f"{n_top_request}; capping secondary append at {_cap} "
                    f"(final ≤ {n_top_request + _cap}, or n_vars). "
                    "Pass options=HVGOptions(append_budget=N) with N≤n_top, "
                    "or append_budget=0 to disable.",
                    UserWarning,
                    stacklevel=2,
                )
            append_budget = _cap

        hvg_params = dict(
            flavor=flavor,
            counts_layer=counts_layer,
            span=span,
            n_bins=n_bins,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            max_disp=max_disp,
        )

        show = (
            _progress_default(adata, n_top_is_auto=n_top_is_auto)
            if progress is None
            else bool(progress)
        )
        if n_top_is_auto:
            _progress(
                show,
                "n_top_genes='auto': estimating list size "
                "(several graph builds; use n_top_genes=2000 for a faster fixed list)...",
            )

        hvg_run_meta: dict[str, Any] = {
            "flavor_requested": str(flavor),
            "flavor_used": str(flavor),
            "fallback_reason": None,
        }
        if global_score is None:
            _progress(
                show,
                "global HVG pass (flavor=%s) over %d genes x %d cells...",
                flavor,
                adata.n_vars,
                adata.n_obs,
            )
            hvg_run_meta = _run_hvg(
                adata,
                n_top_genes=n_top_request,
                batch_key=batch_key,
                strict=strict,
                **hvg_params,
            )
        else:
            hvg_run_meta = {
                "flavor_requested": str(flavor),
                "flavor_used": "injected_global_score",
                "fallback_reason": None,
            }
            _progress(
                show,
                "using injected global_score as the anchor; skipping the %s pass.",
                flavor,
            )

        if global_score is not None:
            gs = pd.Series(global_score).reindex(adata.var_names)
            if gs.isna().all():
                raise ValueError(
                    "global_score does not align with adata.var_names (all NaN after reindex)."
                )
            global_scores = gs.fillna(float(np.nanmin(gs.to_numpy())))
            global_rank = global_scores.rank(ascending=False, method="average")
            top_mask = global_rank <= n_top_request
            adata.var["highly_variable"] = top_mask.to_numpy()
            adata.var["highly_variable_rank"] = global_rank.where(top_mask, np.nan).to_numpy()
        else:
            _flavor_for_scores = str(hvg_run_meta.get("flavor_used") or flavor)
            global_scores = _variability_raw_scores(
                adata,
                flavor=flavor,
                flavor_used=_flavor_for_scores,
                batch_key=batch_key,
            )
            global_rank = global_scores.rank(ascending=False, method="average")

        append_meta: dict[str, Any] | None = None
        n_top_base: int | None = None
        selected: list[str]
        selection_tag = "global"
        score_type: str | None = "global_variability"
        aggregated: pd.Series | None = global_scores

        if n_top_is_auto:
            from ._auto_n import PRODUCT_STRUCTURE_N_SEEDS, estimate_n_top_structure

            n_types_struct: int | None = None
            if label_key and label_key in getattr(adata, "obs", {}):
                from ._diagnosis import count_label_types

                n_types_struct = count_label_types(adata.obs[label_key])
            n_seeds_struct = (
                int(structure_n_seeds_opt)
                if structure_n_seeds_opt is not None
                else int(PRODUCT_STRUCTURE_N_SEEDS)
            )
            try:
                n_final, structure_detail = estimate_n_top_structure(
                    adata,
                    counts_layer=counts_layer,
                    random_state=random_state,
                    version="v7",
                    k_min=int(n_top_min),
                    k_max=min(int(n_top_max), int(adata.n_vars)),
                    n_genes=int(adata.n_vars),
                    n_seeds=n_seeds_struct,
                    progress=show,
                    hvg_mode=hvg_mode if hvg_mode != "balanced" else "auto",
                    n_types=n_types_struct,
                    label_key=label_key,
                )
                mode_info = resolve_hvg_mode(
                    adata,
                    mode=str(mode_arg),
                    label_key=label_key,
                    n_obs=int(adata.n_obs),
                    n_density_pops=(structure_detail.get("features") or {}).get("n_density_pops"),
                    rule_branch=structure_detail.get("rule_branch"),
                )
                hvg_mode = str(mode_info["mode"])
                # Product default: floor 200, raise with density cores (tight).
                # Explicit options.append_budget always wins.
                if user_append_budget is None and method == "append":
                    from ._auto_n import product_append_budget

                    feat = structure_detail.get("features") or {}
                    nd = feat.get("n_density_pops")
                    if nd is None and isinstance(structure_detail.get("rule_explain"), dict):
                        nd = structure_detail["rule_explain"].get("n_density_pops")
                    append_budget, append_budget_info = product_append_budget(nd)
                    append_budget_requested = int(append_budget)
                elif user_append_budget is None and mode_info.get("append_budget") is not None:
                    append_budget = int(mode_info["append_budget"])
                    append_budget_requested = int(append_budget)
                if hvg_mode == "fine" and int(n_final) < 2000 and int(n_final) < int(adata.n_vars):
                    n_final = min(2000, int(adata.n_vars), int(n_top_max))
                    structure_detail = dict(structure_detail)
                    structure_detail["hvg_mode"] = hvg_mode
                    structure_detail["n_top_selected"] = int(n_final)
                    rb = str(structure_detail.get("rule_branch") or "")
                    if "fine_mode_floor" not in rb:
                        structure_detail["rule_branch"] = f"{rb}+fine_mode_floor:{n_final}"
                    ks = str(structure_detail.get("k_source") or "structure")
                    if "fine_mode_floor" not in ks:
                        structure_detail["k_source"] = f"{ks}+fine_mode_floor:{n_final}"
            except _RECOVERABLE_ERRORS as exc:
                msg = (
                    f"structure auto_n failed ({type(exc).__name__}: {exc}); "
                    "falling back to n_top_genes=2000."
                )
                if strict:
                    raise RuntimeError(
                        "structure auto_n failed and strict=True; not falling back. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)
                n_final = min(2000, int(adata.n_vars), int(n_top_max))
                n_final = max(int(n_top_min), int(n_final))
                structure_detail = {
                    "fallback_reason": "structure_auto_n_failed→2000",
                    "structure_error": str(exc),
                    "structure_error_type": type(exc).__name__,
                    "rule_branch": f"fallback_2000:{type(exc).__name__}",
                    "k_source": f"fallback_2000:{type(exc).__name__}",
                }

            n_final = int(n_final)
            if structure_detail.get("fallback_reason"):
                _auto_msg = (
                    f"Base list size set to {int(n_final)} "
                    f"(structure auto failed: {structure_detail.get('structure_error_type')}; "
                    "using classical 2000). Pass n_top_genes=... to override."
                )
            else:
                _auto_msg = plain_auto_n_message(
                    k=int(n_final),
                    rule_branch=structure_detail.get("rule_branch"),
                    short_blocked=bool(structure_detail.get("short_blocked")),
                )
            if method == "append":
                _cap = min(int(append_budget_requested), max(0, int(n_final)))
                if _cap != int(append_budget_requested):
                    append_budget_capped = True
                    warnings.warn(
                        f"append_budget={append_budget_requested} exceeds auto "
                        f"n_top={n_final}; capping secondary append at {_cap}.",
                        UserWarning,
                        stacklevel=2,
                    )
                append_budget = _cap
                _progress(
                    show, "auto: base list k=%d, then +%d more genes...", n_final, append_budget
                )
                selected, append_meta = _hvg_base_plus_append(
                    global_scores, n_base=n_final, n_append=append_budget
                )
                selection_tag = "hvg_base_plus_secondary_append"
                score_type = "global_variability_plus_append"
                n_top_base = int(n_final)
                n_top_for_markers = int(len(selected))
            else:
                selected = _top_genes_from_scores(global_scores, n_final)
                selection_tag = "global"
                score_type = "global_variability"
                n_top_for_markers = int(n_final)

            auto_meta = {
                "strategy": "structure",
                "n_top_selected": int(n_final),
                "k_source": structure_detail.get("k_source"),
                "rule_branch": structure_detail.get("rule_branch"),
                "message": _auto_msg,
                "method_picks": {"structure": int(n_final)},
                "structure": structure_detail,
                "structure_n_seeds": n_seeds_struct,
                "pool_realign": ("append_secondary_hvg" if method == "append" else "none"),
                "n_top_after_realign": int(len(selected)),
                "append": append_meta if method == "append" else None,
                "append_budget_info": append_budget_info,
            }
            if structure_detail.get("fallback_reason"):
                auto_meta["fallback_reason"] = structure_detail["fallback_reason"]
            _progress(show, "%s", _auto_msg)
        else:
            # Fixed k
            if method == "append":
                selected, append_meta = _hvg_base_plus_append(
                    global_scores, n_base=n_top_request, n_append=append_budget
                )
                selection_tag = "hvg_base_plus_secondary_append"
                score_type = "global_variability_plus_append"
                n_top_base = int(n_top_request)
                n_top_for_markers = int(len(selected))
            else:
                selected = _top_genes_from_scores(global_scores, n_top_request)
                selection_tag = "global"
                score_type = "global_variability"
                n_top_for_markers = int(n_top_request)

        n_markers_already_selected = (
            None
            if marker_genes is None
            else int(len({str(g) for g in marker_genes} & {str(g) for g in selected}))
        )

        selected, gene_filter_info = _apply_gene_filters(
            selected,
            adata.var_names,
            filter_mito=filter_mito,
            filter_ribo=filter_ribo,
            marker_genes=marker_genes if marker_mode == "force" else None,
            fill_rank=global_scores.rank(ascending=False, method="first"),
            n_top_genes=n_top_for_markers,
            gene_nomenclature=gene_nomenclature,
        )
        if marker_mode == "force":
            selected = _merge_markers(
                selected,
                marker_genes,
                adata.var_names,
                n_top_for_markers,
                extra=marker_extra,
            )

        n_markers_present = (
            0 if marker_genes is None else int(sum(g in adata.var_names for g in marker_genes))
        )
        from dataclasses import asdict as _asdict

        from .._version import __version__ as _scfair_version

        options_resolved = {k: v for k, v in _asdict(opt).items() if k != "global_score"}
        options_resolved["global_score"] = "injected" if global_score is not None else None

        meta: dict[str, Any] = {
            "scfair_version": _scfair_version,
            "flavor": hvg_run_meta.get("flavor_used", flavor),
            "flavor_requested": hvg_run_meta.get("flavor_requested", flavor),
            "flavor_used": hvg_run_meta.get("flavor_used", flavor),
            "fallback_reason": hvg_run_meta.get("fallback_reason"),
            "balance_method": method,
            "mode": hvg_mode,
            "mode_requested": str(mode_arg),
            "n_top_genes_request": n_top_genes if not n_top_is_auto else str(n_top_genes),
            "n_top_genes_used": (
                int(n_top_base)
                if method == "append" and n_top_base is not None
                else n_top_for_markers
            ),
            "n_top_min": int(n_top_min),
            "n_top_max": int(n_top_max),
            "auto_n_method": "structure" if n_top_is_auto else None,  # auto is structure-only
            "auto_n": auto_meta,
            "auto_message": (
                (auto_meta or {}).get("message") if isinstance(auto_meta, dict) else None
            ),
            "append_budget": int(append_budget) if method == "append" else None,
            "append_budget_requested": (
                int(append_budget_requested) if method == "append" else None
            ),
            "append_budget_capped": bool(append_budget_capped) if method == "append" else None,
            "append_budget_info": append_budget_info if method == "append" else None,
            "append": append_meta if method == "append" else None,
            "selection": selection_tag,
            "score_type": score_type,
            "scfair_score_note": (
                "For append: scfair_score is global variability (same ranking as "
                "the base + secondary list). For none: global variability scores."
            ),
            "counts_layer": counts_layer,
            "filter_mito": filter_mito,
            "filter_ribo": filter_ribo,
            "gene_nomenclature": gene_filter_info.get("gene_nomenclature"),
            "gene_nomenclature_source": gene_filter_info.get("gene_nomenclature_source"),
            "n_mito_ribo_dropped": gene_filter_info.get("n_mito_ribo_dropped"),
            "gene_filter_tips": list(gene_filter_info.get("tips") or []) or None,
            "marker_mode": marker_mode if marker_genes else None,
            "marker_extra": marker_extra if marker_mode == "force" else None,
            "n_marker_genes": n_markers_present,
            "n_marker_genes_already_selected": n_markers_already_selected,
            "n_highly_variable_final": len(selected),
            "random_state": random_state,
            "batch_key": batch_key,
            "store_raw": bool(effective_store_raw) if effective_store_raw is not False else False,
            "global_score": "injected" if global_score is not None else None,
            "options": options_resolved,
            "counts_integer_like": counts_validate_info.get("counts_integer_like"),
            "counts_warning": counts_validate_info.get("counts_warning"),
            "label_key": label_key,
            "structure_n_seeds": (
                (auto_meta or {}).get("structure_n_seeds") if isinstance(auto_meta, dict) else None
            ),
        }
        # merge counts validate extras
        for k, v in counts_validate_info.items():
            if k not in meta and v is not None:
                meta[k] = v

        _k_for_identity = (
            int(n_top_for_markers) if isinstance(n_top_for_markers, (int, np.integer)) else None
        )
        if method == "append" and isinstance(append_meta, dict) and "n_base" in append_meta:
            _k_for_identity = int(append_meta["n_base"])
        if (
            _k_for_identity is not None
            and adata.n_vars >= 10
            and _k_for_identity >= max(10, int(0.8 * adata.n_vars))
        ):
            warnings.warn(
                f"n_top_genes resolved to {_k_for_identity} of {adata.n_vars} genes "
                f"({100.0 * _k_for_identity / adata.n_vars:.0f}% of the matrix). "
                "HVG selection is nearly identity. Pass a smaller integer n_top_genes "
                "or lower options.n_top_max.",
                UserWarning,
                stacklevel=2,
            )
        n_sel_final = int(len(selected))
        if (
            method == "append"
            and adata.n_vars >= 10
            and n_sel_final > max(10, int(0.5 * adata.n_vars))
        ):
            warnings.warn(
                f"append selection kept {n_sel_final}/{adata.n_vars} genes "
                f"({100.0 * n_sel_final / adata.n_vars:.0f}% of the matrix). "
                "On small gene sets use options=HVGOptions(append_budget=0) "
                "or a smaller budget relative to n_top_genes.",
                UserWarning,
                stacklevel=2,
            )

        if diagnose:
            auto_strat = None
            structure_meta = None
            if isinstance(auto_meta, dict):
                auto_strat = auto_meta.get("strategy")
                raw_st = auto_meta.get("structure")
                if isinstance(raw_st, dict):
                    structure_meta = raw_st
            _diag_k = (
                int(n_top_base)
                if method == "append" and n_top_base is not None
                else n_top_for_markers
            )
            meta["diagnosis"] = diagnose_hvg_run(
                balance_method=method,
                n_top_genes_used=_diag_k,
                n_top_is_auto=n_top_is_auto,
                auto_n_strategy=str(auto_strat) if auto_strat else None,
                structure_meta=structure_meta,
                config_check=config_check,
                log=True,
            )
            _filter_tips = [str(t) for t in (gene_filter_info.get("tips") or [])]
            if _filter_tips:
                tips_now = list(meta["diagnosis"].get("tips") or [])
                for t in _filter_tips:
                    if t not in tips_now:
                        tips_now.append(t)
                meta["diagnosis"]["tips"] = tips_now[:2]
        elif gene_filter_info.get("tips"):
            for t in list(gene_filter_info["tips"])[:2]:
                _progress(show, "tip: %s", t)

        if method == "none":
            _progress(show, "done: %d genes (global HVG, no clustering).", len(selected))
        else:
            n_base = (
                int(append_meta["n_base"])
                if isinstance(append_meta, dict) and "n_base" in append_meta
                else max(0, len(selected) - int(append_budget))
            )
            n_extra = (
                int(append_meta["n_append_used"])
                if isinstance(append_meta, dict) and "n_append_used" in append_meta
                else max(0, len(selected) - n_base)
            )
            _progress(
                show,
                "done: %d genes (%d base + %d append).",
                len(selected),
                n_base,
                n_extra,
            )

        if diagnose and isinstance(meta.get("diagnosis"), dict):
            for tip in meta["diagnosis"].get("tips") or []:
                _progress(show, "tip: %s", tip)

        result = _apply_selection(
            adata,
            selected=selected,
            aggregated_score=aggregated,
            global_scores=global_scores,
            meta=meta,
            prior_scanpy_hvg=_prior_scanpy_hvg,
        )

        adata.layers.pop("_scfair_log", None)
        adata.layers.pop(INTERNAL_COUNTS_LAYER, None)

        if subset and inplace:
            # Same private API scanpy uses; anndata has no public inplace subset.
            adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())

        if not inplace:
            return result
        return None

    except BaseException as exc:
        _rollback_adata_after_failure(adata, _hvg_snap, exc)
        raise


def _progress(on: bool, msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    logger.info(text)
    if on:
        print(f"scfair: {text}", file=sys.stderr, flush=True)


def _progress_default(adata: Any, *, n_top_is_auto: bool = False) -> bool:
    """Announce stages when the call is slow enough to look like a hang.

    Auto list-size estimation runs multiple graph builds and can take tens of
    seconds on ~1–2k cells × ~8k genes; surface progress from ``n_obs >= 1000``
    when ``n_top_genes`` is automatic so mid-size default runs are not silent.
    """
    n_obs = int(getattr(adata, "n_obs", 0) or 0)
    n_vars = int(getattr(adata, "n_vars", 0) or 0)
    if n_top_is_auto and n_obs >= 1_000:
        return True
    return bool(n_obs >= 10_000 or n_obs * n_vars >= 5e7)


def _snapshot_adata_for_rollback(adata: Any) -> dict[str, Any]:
    """Capture caller-visible state that a failed HVG call may leave dirty."""
    var_had = {c: adata.var[c].copy() for c in _HVG_PARTIAL_VAR_COLS if c in adata.var.columns}
    return {
        "var_had": var_had,
        "var_cols": set(adata.var.columns),
        "had_scfair_log": "_scfair_log" in getattr(adata, "layers", {}),
        "had_scfair_counts": INTERNAL_COUNTS_LAYER in getattr(adata, "layers", {}),
        "had_counts_layer": "counts" in getattr(adata, "layers", {}),
        "uns_scfair": (
            {k: v for k, v in adata.uns["scfair"].items()}
            if isinstance(adata.uns.get("scfair"), dict)
            else None
        ),
        "had_uns_hvg": "hvg" in adata.uns,
        "uns_hvg": dict(adata.uns["hvg"]) if isinstance(adata.uns.get("hvg"), dict) else None,
    }


def _rollback_adata_after_failure(adata: Any, snap: dict[str, Any], exc: BaseException) -> None:
    """Undo partial HVG annotations so a failed call does not leave a false mask."""
    # var columns: restore prior values or drop ones we introduced
    for c in _HVG_PARTIAL_VAR_COLS:
        if c in snap["var_had"]:
            adata.var[c] = snap["var_had"][c]
        elif c in adata.var.columns and c not in snap["var_cols"]:
            del adata.var[c]
    # internal log / counts layers
    if not snap["had_scfair_log"]:
        try:
            adata.layers.pop("_scfair_log", None)
        except Exception:
            pass
    if not snap.get("had_scfair_counts"):
        try:
            adata.layers.pop(INTERNAL_COUNTS_LAYER, None)
        except Exception:
            pass
    if not snap.get("had_counts_layer"):
        try:
            adata.layers.pop("counts", None)
        except Exception:
            pass
    # scfair uns: restore prior dict keys for hvg; always record failure
    prev = snap["uns_scfair"]
    if prev is None:
        if "scfair" in adata.uns and isinstance(adata.uns["scfair"], dict):
            # keep raw_snapshot if prepare wrote it; drop incomplete hvg
            store = adata.uns["scfair"]
            store.pop("hvg", None)
        else:
            adata.uns.setdefault("scfair", {})
    else:
        # restore shallow copy of prior scfair, then mark failure
        store = dict(prev)
        adata.uns["scfair"] = store
    fail_rec = {
        "failed": True,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    adata.uns.setdefault("scfair", {})
    adata.uns["scfair"]["hvg_failed"] = fail_rec
    # scanpy uns["hvg"]: restore if we had one
    if snap["uns_hvg"] is not None:
        adata.uns["hvg"] = dict(snap["uns_hvg"])
    elif not snap["had_uns_hvg"] and "hvg" in adata.uns:
        adata.uns.pop("hvg", None)


def _validate_counts_matrix(
    adata: Any,
    *,
    counts_layer: str,
    flavor: str | None = None,
    span: float = 0.3,
    strict: bool = False,
) -> dict[str, Any]:
    """Hard-fail on non-finite, negative, or degenerate counts used for HVG.

    Also pre-empts seurat_v3 loess segfaults on tiny gene sets (n_vars too small
    for span) and surfaces non-integer / log-normalized input via UserWarning
    (or raise when ``strict=True``).

    Returns a small diagnostic dict for ``uns['scfair']['hvg']`` (e.g.
    ``counts_integer_like``).
    """
    info: dict[str, Any] = {
        "counts_integer_like": None,
        "counts_warning": [],  # list of warning codes (may accumulate)
        "n_vars": int(getattr(adata, "n_vars", 0) or 0),
    }
    if counts_layer in getattr(adata, "layers", {}):
        X = adata.layers[counts_layer]
        where = f"layers[{counts_layer!r}]"
    else:
        X = adata.X
        where = ".X"
    if X is None:
        raise ValueError("counts matrix is missing (X and layers are empty).")

    # --- finite / non-negative checks on stored values only (no full densify) ---
    if sparse.issparse(X):
        data = X.data
        if data is None or data.size == 0:
            raise ValueError(
                f"counts matrix {where} is empty/all-zero (no stored non-zeros). "
                "scFair cannot select HVGs from a degenerate counts matrix. "
                "Check that layers['counts'] (or .X) holds real raw counts."
            )
        # Work on the CSR/CSC .data buffer — never materialise a dense copy.
        if not np.isfinite(data).all():
            n_bad = int(np.size(data) - np.isfinite(data).sum())
            raise ValueError(
                f"counts matrix contains {n_bad} non-finite value(s) (NaN/Inf). "
                "scFair requires finite raw counts (or a finite counts layer)."
            )
        if np.any(data < 0):
            n_neg = int(np.sum(data < 0))
            raise ValueError(
                f"counts matrix contains {n_neg} negative value(s). "
                "flavor='seurat_v3' and related count HVG methods require non-negative "
                "counts. Restore raw counts (e.g. layers['counts']) before calling."
            )
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        # getnnz avoids building a full boolean sparse matrix via (X != 0).
        try:
            col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
        except Exception:  # pragma: no cover — older scipy
            col_nnz = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        # Dense: check finiteness / negativity without an extra full ravel copy
        # when possible (views share memory with X).
        arr = np.asarray(X)
        if arr.size == 0:
            raise ValueError(f"counts matrix {where} is empty.")
        if arr.ndim != 2:
            raise ValueError(f"counts matrix {where} must be 2-dimensional, got ndim={arr.ndim}.")
        if not np.isfinite(arr).all():
            n_bad = int(arr.size - np.isfinite(arr).sum())
            raise ValueError(
                f"counts matrix contains {n_bad} non-finite value(s) (NaN/Inf). "
                "scFair requires finite raw counts (or a finite counts layer)."
            )
        if np.any(arr < 0):
            n_neg = int(np.sum(arr < 0))
            raise ValueError(
                f"counts matrix contains {n_neg} negative value(s). "
                "flavor='seurat_v3' and related count HVG methods require non-negative "
                "counts. Restore raw counts (e.g. layers['counts']) before calling."
            )
        if not np.any(arr):
            raise ValueError(
                f"counts matrix {where} is all zeros. "
                "scFair cannot select HVGs from a degenerate counts matrix. "
                "Check that layers['counts'] (or .X) holds real raw counts "
                "(a placeholder zeros matrix will crash later in PCA/ARPACK)."
            )
        row_sums = arr.sum(axis=1)
        col_nnz = np.count_nonzero(arr, axis=0)

    n_empty_cells = int(np.sum(row_sums <= 0))
    if n_empty_cells == int(X.shape[0]):
        raise ValueError(
            f"every cell has zero total counts in {where}. "
            "Filter empty barcodes or restore a real counts matrix."
        )
    if n_empty_cells > 0 and n_empty_cells >= max(1, int(0.5 * X.shape[0])):
        warnings.warn(
            f"{n_empty_cells}/{X.shape[0]} cells have zero total counts in {where}; "
            "PCA/neighbors may be unstable. Filter empty barcodes before HVG.",
            UserWarning,
            stacklevel=3,
        )
    n_dead_genes = int(np.sum(col_nnz == 0))
    n_live = int(X.shape[1]) - n_dead_genes
    if n_live < 2:
        raise ValueError(
            f"counts matrix {where} has fewer than 2 genes with any non-zero "
            f"entry ({n_live} live / {X.shape[1]} total). Cannot run HVG/PCA."
        )

    # Integer-like check (log-normalized .X snapshotted as "counts" is a
    # common footgun). Surface as UserWarning + record; strict → raise.
    is_int = bool(_is_integer_counts_like(X))
    info["counts_integer_like"] = is_int
    warn_codes: list[str] = []
    if not is_int:
        msg = (
            f"counts matrix {where} does not look like raw integer counts "
            "(values are non-integer or fractional). seurat_v3 / "
            "pearson_residuals expect UMI counts; log1p-normalized input "
            "yields unreliable HVGs. Restore raw counts before calling, "
            "or pass layer= pointing at a true counts matrix."
        )
        warn_codes.append("non_integer_counts")
        if strict:
            raise ValueError(msg + " (strict=True)")
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    # seurat_v3 loess (skmisc) can SIGSEGV in C on tiny n_obs / n_vars.
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        why = _loess_unsafe(int(X.shape[0]), int(X.shape[1]), span=span)
        if why:
            msg = (
                f"flavor={flavor!r} loess is unsafe on this matrix ({why}). "
                "Use more cells/genes, or flavor='seurat' / a non-loess method."
            )
            warn_codes.append("loess_unsafe")
            info["loess_unsafe"] = why
            if strict:
                raise ValueError(msg + " (strict=True)")
            warnings.warn(
                msg + " Will fall back to flavor='seurat' if HVG proceeds.",
                UserWarning,
                stacklevel=3,
            )

    info["counts_warning"] = warn_codes or None
    return info


def _run_hvg(
    adata: Any,
    *,
    n_top_genes: int,
    flavor: str,
    counts_layer: str,
    span: float,
    n_bins: int,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    max_disp: float,
    batch_key: str | None,
    strict: bool = False,
) -> dict[str, Any]:
    """Run HVG in place; optionally fall back seurat_v3* → seurat.

    Returns
    -------
    dict
        ``flavor_requested``, ``flavor_used``, ``fallback_reason`` (or None).
    """
    meta = {
        "flavor_requested": str(flavor),
        "flavor_used": str(flavor),
        "fallback_reason": None,
    }
    # Drop leftover score columns before any flavor run (G3).
    _clear_scanpy_hvg_var_columns(adata)
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        why = _loess_unsafe(int(adata.n_obs), int(adata.n_vars), span=span)
        if why:
            reason = f"flavor={flavor!r} loess is unsafe ({why}); falling back to flavor='seurat'."
            if strict:
                raise ValueError(f"flavor={flavor!r} loess is unsafe ({why}). (strict=True)")
            warnings.warn(reason, UserWarning, stacklevel=3)
            logger.warning(reason)
            _clear_scanpy_hvg_var_columns(adata)
            try:
                _run_hvg_once(
                    adata,
                    n_top_genes=n_top_genes,
                    flavor="seurat",
                    counts_layer=counts_layer,
                    span=span,
                    n_bins=n_bins,
                    min_mean=min_mean,
                    max_mean=max_mean,
                    min_disp=min_disp,
                    max_disp=max_disp,
                    batch_key=batch_key,
                )
            except _RECOVERABLE_ERRORS + (IndexError,) as exc:
                raise ValueError(
                    f"{reason} seurat fallback also failed ({type(exc).__name__}: {exc})."
                ) from exc
            meta["flavor_used"] = "seurat"
            meta["fallback_reason"] = f"loess_unsafe:{why}"
            return meta
    try:
        _run_hvg_once(
            adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            counts_layer=counts_layer,
            span=span,
            n_bins=n_bins,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            max_disp=max_disp,
            batch_key=batch_key,
        )
        return meta
    except _RECOVERABLE_ERRORS as exc:
        if flavor not in ("seurat_v3", "seurat_v3_paper"):
            raise
        reason = (
            f"flavor={flavor!r} failed ({type(exc).__name__}: {exc}); "
            "falling back to flavor='seurat' on log-normalized counts."
        )
        if strict:
            raise RuntimeError(
                f"{reason} (strict=True; set strict=False to allow fallback.)"
            ) from exc
        warnings.warn(reason, UserWarning, stacklevel=3)
        logger.warning(reason)
        _clear_scanpy_hvg_var_columns(adata)
        _run_hvg_once(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat",
            counts_layer=counts_layer,
            span=span,
            n_bins=n_bins,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            max_disp=max_disp,
            batch_key=batch_key,
        )
        meta["flavor_used"] = "seurat"
        meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return meta


def _clear_scanpy_hvg_var_columns(adata: Any) -> None:
    """Drop scanpy HVG score / mask columns left by a previous flavor run.

    Without this, a second call with a different flavor can leave e.g.
    ``variances_norm`` in place while writing ``residual_variances``;
    score readers then silently reuse the *previous* flavor's column.
    """
    drop = [c for c in _SCANPY_HVG_VAR_COLS if c in adata.var.columns]
    if drop:
        adata.var.drop(columns=drop, inplace=True, errors="ignore")


def _run_hvg_once(
    adata: Any,
    *,
    n_top_genes: int,
    flavor: str,
    counts_layer: str,
    span: float,
    n_bins: int,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    max_disp: float,
    batch_key: str | None,
) -> None:
    work_layer = _materialize_flavor_matrix(adata, flavor=flavor, counts_layer=counts_layer)

    if flavor == "pearson_residuals":
        try:
            from scanpy.experimental.pp import highly_variable_genes as hvg_exp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "flavor='pearson_residuals' requires scanpy.experimental.pp."
            ) from exc
        pr_kwargs: dict[str, Any] = {
            "flavor": "pearson_residuals",
            "n_top_genes": n_top_genes,
            "layer": work_layer,
            "inplace": True,
            "subset": False,
            "check_values": True,
        }
        if batch_key is not None:
            pr_kwargs["batch_key"] = batch_key
        hvg_exp(adata, **pr_kwargs)
        return

    kwargs: dict[str, Any] = {
        "n_top_genes": n_top_genes,
        "flavor": flavor,
        "inplace": True,
        "subset": False,
    }
    if flavor in _COUNTS_FLAVORS:
        kwargs["layer"] = work_layer
        kwargs["span"] = span
    else:
        if work_layer is not None:
            kwargs["layer"] = work_layer
        kwargs["n_bins"] = n_bins
        kwargs["min_mean"] = min_mean
        kwargs["max_mean"] = max_mean
        kwargs["min_disp"] = min_disp
        kwargs["max_disp"] = max_disp
    if batch_key is not None:
        kwargs["batch_key"] = batch_key

    sc.pp.highly_variable_genes(adata, **kwargs)


def _materialize_flavor_matrix(adata: Any, *, flavor: str, counts_layer: str) -> str | None:
    if flavor in _COUNTS_FLAVORS:
        if counts_layer not in adata.layers:
            raise ValueError(f"Counts layer {counts_layer!r} missing for flavor={flavor!r}.")
        return counts_layer

    import anndata as ad

    log_layer = "_scfair_log"

    def _log_from(X: Any) -> str:
        ad_tmp = ad.AnnData(X=X.copy() if hasattr(X, "copy") else X)
        sc.pp.normalize_total(ad_tmp, target_sum=1e4)
        sc.pp.log1p(ad_tmp)
        adata.layers[log_layer] = ad_tmp.X
        return log_layer

    if counts_layer in adata.layers:
        X = adata.layers[counts_layer]
        # Already-logged / non-integer staged matrices must not be logged again.
        # Scanpy seurat/cell_ranger expect log1p input and expm1 internally.
        if _is_integer_counts_like(X):
            return _log_from(X)
        return counts_layer

    if _is_integer_counts_like(adata.X):
        return _log_from(adata.X)

    logger.debug("Using existing .X for log-based flavor=%s.", flavor)
    return None


def _batch_merge_scores(
    adata: Any,
    *,
    flavor_used: str | None = None,
) -> pd.Series | None:
    """Descending scores that reproduce scanpy's per-batch HVG merge order.

    scanpy does **not** select by the mean of per-batch ``variances_norm`` /
    ``dispersions_norm``. Criteria (scanpy ≥1.10):

    * ``seurat_v3``: ``highly_variable_rank`` ASC, then ``nbatches`` DESC
    * ``seurat_v3_paper``: ``nbatches`` DESC, then ``rank`` ASC
    * ``seurat`` / ``cell_ranger``: ``nbatches`` DESC, then ``dispersions_norm`` DESC
      (no ``highly_variable_rank`` column is written)
    """
    var = getattr(adata, "var", None)
    if var is None or "highly_variable_nbatches" not in var.columns:
        return None

    nbatches = pd.to_numeric(var["highly_variable_nbatches"], errors="coerce")
    nbatches = nbatches.reindex(adata.var_names).fillna(0.0).astype(float)
    flavor = str(flavor_used or "")

    if "highly_variable_rank" in var.columns:
        rank = pd.to_numeric(var["highly_variable_rank"], errors="coerce")
        rank = rank.reindex(adata.var_names)
        # Missing rank (nbatches==0) sorts last in scanpy (na_position='last').
        rank_fill = float(np.nanmax(rank.to_numpy())) + 1.0 if rank.notna().any() else 1.0e9
        rank_f = rank.fillna(rank_fill)
        if flavor == "seurat_v3_paper":
            # nbatches DESC primary, rank ASC secondary.
            return nbatches * 1.0e9 - rank_f
        # seurat_v3 (and any other rank-writing batch path): rank ASC primary.
        return -rank_f + nbatches * 1.0e-6

    # seurat / cell_ranger batch merge: nbatches DESC, then mean score DESC.
    # Encode so any nbatches=k+1 beats any nbatches=k even when the secondary
    # score has mixed sign (max(|sec|)+1 is not a safe radix).
    for col in ("dispersions_norm", "variances_norm", "residual_variances"):
        if col in var.columns:
            sec = pd.to_numeric(var[col], errors="coerce")
            sec = sec.reindex(adata.var_names)
            sec_f = sec.fillna(float(np.nanmin(sec.to_numpy())) - 1.0 if sec.notna().any() else 0.0)
            sec_arr = sec_f.to_numpy()
            sec_min = float(np.nanmin(sec_arr)) if sec_f.notna().any() else 0.0
            sec_max = float(np.nanmax(sec_arr)) if sec_f.notna().any() else 0.0
            span = (sec_max - sec_min) + 1.0
            return nbatches * span + (sec_f - sec_min)
    return nbatches


def _variability_raw_scores(
    adata: Any,
    *,
    flavor: str | None = None,
    flavor_used: str | None = None,
    batch_key: str | None = None,
) -> pd.Series:
    """All-gene variability scores for ranking / append.

    When ``flavor`` / ``flavor_used`` is set, only that flavor's scanpy score
    column is read (no priority walk across leftover columns from a prior
    call). Prefer ``flavor_used`` when seurat_v3 fell back to seurat.

    **Batch mode.** With ``batch_key`` (or when scanpy wrote
    ``highly_variable_nbatches``), use :func:`_batch_merge_scores` so the
    selected set matches scanpy's per-batch merge — not the cross-batch mean
    of flavor score columns.
    """
    if batch_key is not None or "highly_variable_nbatches" in getattr(adata, "var", {}).columns:
        batch_scores = _batch_merge_scores(adata, flavor_used=flavor_used or flavor)
        if batch_scores is not None:
            return batch_scores

    use_flavor = flavor_used or flavor
    if use_flavor is not None:
        cols = _FLAVOR_SCORE_COLS.get(str(use_flavor))
        if cols is None:
            # Unknown / injected — fall through to generic probe below.
            cols = ()
        for col in cols:
            if col in adata.var.columns:
                s = pd.to_numeric(adata.var[col], errors="coerce")
                s = s.reindex(adata.var_names)
                fill = float(np.nanmin(s.to_numpy())) if s.notna().any() else 0.0
                return s.fillna(fill)
        if cols:
            raise ValueError(
                f"flavor={use_flavor!r} finished without writing score column(s) "
                f"{list(cols)} on adata.var. Cannot build global variability "
                "scores. Re-run highly_variable_genes or check the flavor path."
            )

    # Legacy / no-flavor probe (e.g. injected global_score path helpers).
    for col in (
        "variances_norm",
        "dispersions_norm",
        "residual_variances",
        "variances",
        "dispersions",
    ):
        if col in adata.var.columns:
            s = pd.to_numeric(adata.var[col], errors="coerce")
            s = s.reindex(adata.var_names)
            fill = float(np.nanmin(s.to_numpy())) if s.notna().any() else 0.0
            return s.fillna(fill)
    if "highly_variable_rank" in adata.var.columns:
        rank = pd.to_numeric(adata.var["highly_variable_rank"], errors="coerce")
        rank = rank.reindex(adata.var_names).fillna(np.inf)
        return -rank
    return pd.Series(0.0, index=adata.var_names)


def _top_genes_from_rank(rank: pd.Series, n_top: int) -> list[str]:
    return [str(g) for g in rank.sort_values(ascending=True, kind="stable").index[:n_top]]


def _top_genes_from_scores(scores: pd.Series, n_top: int) -> list[str]:
    return [str(g) for g in scores.sort_values(ascending=False, kind="stable").index[:n_top]]


def _hvg_base_plus_append(
    scores: pd.Series,
    *,
    n_base: int,
    n_append: int,
) -> tuple[list[str], dict[str, Any]]:
    """Freeze global top-``n_base``, append the next ``n_append`` by the same score.

    No intermediate clustering, no re-ranking inside a 2× pool. Base genes are
    never dropped for append slots (zero-sum displacement avoided).
    """
    n_base = max(0, int(n_base))
    n_append = max(0, int(n_append))
    order = [str(g) for g in scores.sort_values(ascending=False, kind="stable").index]
    if n_base <= 0:
        base: list[str] = []
    else:
        base = order[: min(n_base, len(order))]
    base_set = set(base)
    extra = [g for g in order[len(base) :] if g not in base_set][:n_append]
    selected = base + extra
    meta = {
        "n_base": len(base),
        "n_append_requested": n_append,
        "n_append_used": len(extra),
        "n_final": len(selected),
        "append_source": "secondary_global_hvg",
    }
    return selected, meta


def _infer_gene_nomenclature(var_names: Sequence[str]) -> str:
    """Infer gene-name convention from symbols / Ensembl ids.

    Returns
    -------
    ``"human"`` | ``"mouse"`` | ``"mixed"`` | ``"unknown"``

    Used to choose MT prefix rules and for diagnostics. Ribosomal structural
    patterns are shared (case-insensitive) once a species is chosen; mixed /
    unknown applies both mito conventions.
    """
    h = 0
    m = 0
    for raw in var_names:
        n = str(raw)
        if n.startswith("ENSG") and not n.startswith("ENSMUS"):
            h += 2
        if n.startswith("ENSMUSG") or n.startswith("ENSMUST"):
            m += 2
        if _MITO_HUMAN_RE.match(n):
            h += 3
        if _MITO_MOUSE_RE.match(n):
            m += 3
        if _RIBO_HUMAN_SIGNAL_RE.match(n) and n[:3] == n[:3].upper():
            h += 1
        if _RIBO_MOUSE_SIGNAL_RE.match(n):
            m += 1
    if h == 0 and m == 0:
        return "unknown"
    if h > 0 and m > 0:
        if h >= 3 * m:
            return "human"
        if m >= 3 * h:
            return "mouse"
        return "mixed"
    return "human" if h >= m else "mouse"


def _is_mito_name(name: str, nomenclature: str = "unknown") -> bool:
    """Mito symbol under inferred (or forced) nomenclature."""
    n = str(name)
    nom = str(nomenclature or "unknown").lower()
    if nom == "human":
        return bool(_MITO_HUMAN_RE.match(n))
    if nom == "mouse":
        return bool(_MITO_MOUSE_RE.match(n))
    # mixed / unknown: both conventions
    return bool(_MITO_HUMAN_RE.match(n) or _MITO_MOUSE_RE.match(n))


def _is_ribo_name(name: str, nomenclature: str = "unknown") -> bool:
    """Ribosomal structural protein; nomenclature reserved for API symmetry."""
    del nomenclature  # shared pattern; species already informed mito rules
    return bool(_RIBO_RE.match(str(name)))


def _apply_gene_filters(
    selected: list[str],
    var_names: pd.Index,
    *,
    filter_mito: bool,
    filter_ribo: bool,
    marker_genes: Sequence[str] | None,
    fill_rank: pd.Series,
    n_top_genes: int,
    gene_nomenclature: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Drop MT/ribo from selection (unless markers); refill from global rank.

    Returns
    -------
    selected, info
        Filtered gene list and a small diagnostic dict (nomenclature, n dropped).
    """
    nom = (
        str(gene_nomenclature).lower() if gene_nomenclature else _infer_gene_nomenclature(var_names)
    )
    if nom not in ("human", "mouse", "mixed", "unknown"):
        nom = _infer_gene_nomenclature(var_names)
    info: dict[str, Any] = {
        "gene_nomenclature": nom,
        "gene_nomenclature_source": "user" if gene_nomenclature else "auto",
        "n_mito_ribo_dropped": 0,
        "tips": [],
    }
    if not filter_mito and not filter_ribo:
        return selected[:n_top_genes], info

    # Tip when naming is not recognized (other species / Ensembl-only ids).
    if nom == "unknown" and (filter_mito or filter_ribo):
        info["tips"].append(
            "MT/ribo filters are on, but gene names do not look human (MT-/RPL) "
            "or mouse (mt-/Rpl). No symbols matched — filters skipped matching. "
            "Pass options=HVGOptions(gene_nomenclature='human'|'mouse') if names "
            "use another convention, or filter_mito=False / filter_ribo=False."
        )

    protect = set(map(str, marker_genes or ()))
    kept: list[str] = []
    dropped = 0
    for g in selected:
        gs = str(g)
        if gs in protect:
            kept.append(gs)
            continue
        if filter_mito and _is_mito_name(gs, nom):
            dropped += 1
            continue
        if filter_ribo and _is_ribo_name(gs, nom):
            dropped += 1
            continue
        kept.append(gs)

    if dropped:
        logger.info(
            "Filtered %d mito/ribo genes from HVG set (nomenclature=%s); "
            "refilling from global rank.",
            dropped,
            nom,
        )
    info["n_mito_ribo_dropped"] = int(dropped)
    if len(kept) >= n_top_genes:
        return kept[:n_top_genes], info

    need = n_top_genes - len(kept)
    have = set(kept)
    for g in _top_genes_from_rank(fill_rank, len(fill_rank)):
        gs = str(g)
        if gs in have:
            continue
        if gs not in protect:
            if filter_mito and _is_mito_name(gs, nom):
                continue
            if filter_ribo and _is_ribo_name(gs, nom):
                continue
        kept.append(gs)
        have.add(gs)
        need -= 1
        if need <= 0:
            break
    return kept[:n_top_genes], info


def _merge_markers(
    selected: list[str],
    marker_genes: Sequence[str] | None,
    var_names: pd.Index,
    n_top_genes: int,
    *,
    extra: bool = False,
) -> list[str]:
    """Force-include markers.

    Parameters
    ----------
    extra
        If False (default), total length is capped at ``n_top_genes`` (markers
        occupy slots). If True, markers are prepended and algorithm genes are
        kept up to ``n_top_genes``, so final size may exceed ``n_top_genes``.
    """
    if not marker_genes:
        return selected[:n_top_genes]
    var_set = set(map(str, var_names))
    markers = [str(g) for g in marker_genes if str(g) in var_set]
    missing = [str(g) for g in marker_genes if str(g) not in var_set]
    if missing:
        shown = missing[:10] + (["..."] if len(missing) > 10 else [])
        warnings.warn(
            f"marker_genes not in adata.var_names (ignored): {shown}",
            UserWarning,
            stacklevel=3,
        )
        logger.warning("marker_genes not in adata.var_names (ignored): %s", shown)
    if not markers:
        return selected[:n_top_genes]

    if extra:
        out: list[str] = []
        seen: set[str] = set()
        for g in markers:
            if g not in seen:
                seen.add(g)
                out.append(g)
        for g in selected[:n_top_genes]:
            gs = str(g)
            if gs not in seen:
                seen.add(gs)
                out.append(gs)
        return out

    out = []
    seen: set[str] = set()
    for g in markers + list(selected):
        if g not in seen:
            seen.add(g)
            out.append(g)
    if len(out) > n_top_genes:
        mark_set = set(markers)
        head = [g for g in out if g in mark_set]
        tail = [g for g in out if g not in mark_set]
        out = head + tail[: max(0, n_top_genes - len(head))]
    return out


def _apply_selection(
    adata: Any,
    *,
    selected: list[str],
    aggregated_score: pd.Series | None,
    global_scores: pd.Series,
    meta: dict[str, Any],
    prior_scanpy_hvg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    hv = adata.var_names.astype(str).isin([str(g) for g in selected])
    adata.var["highly_variable"] = hv

    # Match scanpy: finite rank only for selected genes (0-based in selection
    # order); non-HVG genes are NaN — not +inf. Downstream code uses
    # ``rank.notna()`` / ``dropna(subset=['highly_variable_rank'])``; inf made
    # those filters no-ops and broke ``astype(int)``.
    rank_map = {g: float(i) for i, g in enumerate(selected)}
    ranks = np.array(
        [rank_map.get(str(g), np.nan) for g in adata.var_names],
        dtype=float,
    )
    adata.var["highly_variable_rank"] = ranks

    if aggregated_score is not None:
        sc_score = aggregated_score.reindex(adata.var_names.astype(str))
        # if index was original var_names already
        if sc_score.isna().all():
            sc_score = aggregated_score.reindex(adata.var_names)
        adata.var["scfair_score"] = sc_score.to_numpy(dtype=float)
    else:
        gs = global_scores.reindex(adata.var_names).to_numpy(dtype=float)
        adata.var["scfair_score"] = gs

    # scFair owns ``uns["scfair"]``. Do not replace scanpy's ``uns["hvg"]`` with
    # a one-key stub. Prefer the caller's pre-call dict (scanpy's internal HVG
    # may have overwritten it mid-run); fall back to whatever is there now.
    flavor_used = meta.get("flavor_used", meta.get("flavor"))
    flavor_requested = meta.get("flavor_requested", meta.get("flavor"))
    base = prior_scanpy_hvg
    if base is None and isinstance(adata.uns.get("hvg"), dict):
        base = dict(adata.uns["hvg"])
    if base is not None:
        merged = dict(base)
        merged["flavor"] = flavor_used
        if flavor_requested is not None and flavor_requested != flavor_used:
            merged["flavor_requested"] = flavor_requested
            merged["scfair_flavor_note"] = (
                "flavor is the method that actually ran; flavor_requested may differ "
                "after seurat_v3 fallback"
            )
        adata.uns["hvg"] = merged
    # else: leave uns["hvg"] unset — full metadata is under uns["scfair"]["hvg"]

    if UNS_KEY not in adata.uns:
        adata.uns[UNS_KEY] = {}
    adata.uns[UNS_KEY]["hvg"] = {
        **meta,
        "n_highly_variable": int(hv.sum()),
        "selected_genes": list(selected),
    }

    cols = ["highly_variable", "highly_variable_rank", "scfair_score"]
    for extra in (
        "means",
        "variances",
        "variances_norm",
        "dispersions",
        "dispersions_norm",
        "residual_variances",
        "highly_variable_nbatches",
    ):
        if extra in adata.var.columns:
            cols.append(extra)
    return adata.var.loc[:, cols].copy()


# ---------------------------------------------------------------------------
# Helpers used by structure auto_n (pair stability on intermediate Leiden).
# Kept here so estimate_n_top_structure can import them.
# ---------------------------------------------------------------------------

_MERGE_N_BOOT = 15
_MERGE_FRAC = 0.8


def _mean_axis0(X: Any, row_mask: np.ndarray) -> np.ndarray:
    n = int(row_mask.sum())
    if n == 0:
        raise ValueError("Empty row mask for mean.")
    if sparse.issparse(X):
        return np.asarray(X[row_mask].mean(axis=0)).ravel()
    return np.asarray(X[row_mask], dtype=float).mean(axis=0)


def _nearest_cluster_map(X: Any, masks: dict[str, np.ndarray]) -> dict[str, str]:
    """Map each cluster to its closest other cluster by centroid correlation."""
    labels = list(masks)
    if len(labels) < 2:
        return {}
    centroids = np.vstack([_mean_axis0(X, masks[c]) for c in labels])
    cent = centroids - centroids.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(cent, axis=1)
    norms[norms == 0] = 1.0
    corr = (cent @ cent.T) / np.outer(norms, norms)
    np.fill_diagonal(corr, -np.inf)
    return {labels[i]: labels[int(np.argmax(corr[i]))] for i in range(len(labels))}


def _pair_bootstrap_stability(
    X_pca: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    n_boot: int = _MERGE_N_BOOT,
    frac: float = _MERGE_FRAC,
    random_state: int = 0,
) -> float:
    """Mean ARI of original A/B labels vs KMeans(k=2) on bootstrap subsamples."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    idx = np.where(mask_a | mask_b)[0]
    n = idx.size
    if n < 20:
        raise ValueError("pair bootstrap needs at least 20 cells in the union.")
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        sub = rng.choice(idx, size=max(int(frac * n), 4), replace=False)
        km = KMeans(n_clusters=2, n_init=5, random_state=int(rng.integers(1_000_000)))
        km.fit(X_pca[sub])
        y_sub = mask_a[sub].astype(int)
        scores.append(adjusted_rand_score(y_sub, km.labels_))
    return float(np.mean(scores))
