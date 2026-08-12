"""Advisory diagnostics: size-imbalance metrics and short tips.

These helpers do not change gene selection. They report cluster-size
imbalance from known labels or from a finished HVG call, plus tips when a
setting is unusual (for example a very long auto list).

They do not predict how much a call will gain over plain scanpy HVG.
Size imbalance alone is not a reliable forecast of that margin.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def count_label_types(labels: Any) -> int | None:
    """Distinct labels after dropping missing / empty / ``nan`` / ``none``."""
    try:
        s_raw = pd.Series(labels)
    except Exception:
        return None
    present = s_raw.notna()
    s = s_raw[present].astype(str)
    s = s[(s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none")]
    n = int(s.nunique())
    return n if n > 0 else None


# Downstream clustering protocol tiers (advisory; does not change HVG genes).
_FINE_N_TYPES = 15
_FINE_N_DENSITY_POPS = 15
_RES_COARSE = 0.8
_RES_FINE = 1.5
_HVG_MODES = frozenset({"auto", "balanced", "fine", "compact"})

_MAX_FRAC_STRONG = 0.60
_MAX_FRAC_BALANCED = 0.45
_RATIO_STRONG = 15.0
_RATIO_BALANCED = 5.0
_EVENNESS_BALANCED = 0.85
_EVENNESS_STRONG = 0.60
_RARE_FRAC = 0.05


def cluster_size_metrics(
    sizes: Mapping[Any, int] | pd.Series | Sequence[int] | None,
) -> dict[str, Any]:
    """Summarise cluster sizes into imbalance metrics.

    Parameters
    ----------
    sizes
        Mapping cluster-id → n_cells, a Series, or a plain list of sizes.
        Empty / None yields ``imbalance="unknown"``.

    Returns
    -------
    dict
        ``n_clusters``, ``n_cells``, ``sizes``, ``fractions``, ``max_frac``,
        ``min_frac``, ``max_min_ratio``, ``gini``, ``shannon_evenness``,
        ``n_rare_clusters`` (frac < 5%), ``imbalance``
        (``balanced`` | ``moderate`` | ``strong`` | ``unknown`` | ``degenerate``).
    """
    empty: dict[str, Any] = {
        "n_clusters": 0,
        "n_cells": 0,
        "sizes": {},
        "fractions": {},
        "max_frac": None,
        "min_frac": None,
        "max_min_ratio": None,
        "gini": None,
        "shannon_evenness": None,
        "n_rare_clusters": 0,
        "imbalance": "unknown",
    }
    if sizes is None:
        return empty

    if isinstance(sizes, pd.Series):
        size_map = {str(k): int(v) for k, v in sizes.items() if int(v) > 0}
    elif isinstance(sizes, Mapping):
        size_map = {str(k): int(v) for k, v in sizes.items() if int(v) > 0}
    else:
        size_map = {str(i): int(v) for i, v in enumerate(sizes) if int(v) > 0}

    if not size_map:
        return empty

    vals = np.asarray(list(size_map.values()), dtype=float)
    n = int(vals.size)
    total = float(vals.sum())
    if total <= 0 or n == 0:
        return empty

    fracs = vals / total
    frac_map = {k: float(vals[i] / total) for i, k in enumerate(size_map)}
    max_frac = float(fracs.max())
    min_frac = float(fracs.min())
    max_min = float(vals.max() / vals.min()) if vals.min() > 0 else float("inf")
    gini = _gini(vals)
    evenness = _shannon_evenness(fracs)
    n_rare = int(np.sum(fracs < _RARE_FRAC))

    if n < 2:
        imbalance = "degenerate"
    elif (
        max_frac >= _MAX_FRAC_STRONG
        or max_min >= _RATIO_STRONG
        or (evenness is not None and evenness < _EVENNESS_STRONG and n >= 3)
    ):
        imbalance = "strong"
    elif (
        max_frac <= _MAX_FRAC_BALANCED
        and max_min <= _RATIO_BALANCED
        and (evenness is None or evenness >= _EVENNESS_BALANCED)
    ):
        imbalance = "balanced"
    else:
        imbalance = "moderate"

    return {
        "n_clusters": n,
        "n_cells": int(total),
        "sizes": {k: int(v) for k, v in size_map.items()},
        "fractions": frac_map,
        "max_frac": max_frac,
        "min_frac": min_frac,
        "max_min_ratio": max_min,
        "gini": gini,
        "shannon_evenness": evenness,
        "n_rare_clusters": n_rare,
        "imbalance": imbalance,
    }


def check_config(
    *,
    n_top_genes: Any = None,
    balance_method: str | None = "append",
    log: bool = True,
) -> dict[str, Any]:
    """Parameter-only checks, resolvable before any data is touched.

    Returns ``{"flags": [...], "tips": [...]}``.
    """
    del n_top_genes
    method = str(balance_method or "none")
    flags: list[str] = []
    tips: list[str] = []

    if method == "none":
        flags.append("balance_method_none")
        tips.append("balance_method='none': global HVG only (same idea as scanpy).")
    elif method == "append":
        pass
    else:
        flags.append("unknown_balance_method")
        tips.append(
            f"balance_method={method!r} is not supported. Use 'append' (default) or 'none'."
        )

    if log:
        for tip in tips:
            logger.warning("config: %s", tip)
    return {"flags": flags, "tips": tips}


def recommend_cluster_resolution(
    *,
    n_types: int | None = None,
    n_density_pops: float | int | None = None,
    rule_branch: str | None = None,
) -> dict[str, Any]:
    """Suggest post-HVG Leiden resolution (advisory; does not change genes).

    Protocol tiers used in internal evaluation:

    - **coarse** (default): resolution ``0.8`` — Easy / ≤~12 types.
    - **fine**: resolution ``1.5`` (sweep optional ``[0.8, 1.5, 2.0]``) when
      labels or structure look multi-type/fine-grained. On seurat_v4 l2,
      product ``append`` @2000+200 lost at 0.8 but won at 1.5; raising
      ``append_budget`` to 500 did not beat m=200.

    Parameters
    ----------
    n_types
        Number of label classes if known (e.g. author ``celltype.l2``).
    n_density_pops
        Structure density-core count when available.
    rule_branch
        Structure auto ``rule_branch`` string (detects ``v7_fine_atlas``).
    """
    reasons: list[str] = []
    branch = str(rule_branch or "")
    if n_types is not None and int(n_types) >= _FINE_N_TYPES:
        reasons.append(f"n_types={int(n_types)}>={_FINE_N_TYPES}")
    if n_density_pops is not None:
        try:
            nd = float(n_density_pops)
            if np.isfinite(nd) and nd >= _FINE_N_DENSITY_POPS:
                reasons.append(f"n_density_pops={nd:.0f}>={_FINE_N_DENSITY_POPS}")
        except (TypeError, ValueError):
            pass
    if "fine_atlas" in branch or branch.startswith("v7_fine_atlas"):
        reasons.append("structure_fine_atlas_band")

    if reasons:
        return {
            "tier": "fine",
            "resolution": float(_RES_FINE),
            "resolution_sweep": [0.8, 1.5, 2.0],
            "primary_metric": "macro_f1",
            "reasons": reasons,
            # One short actionable line only — no dataset / metric jargon.
            "note": (
                f"After HVG, cluster at Leiden ≈ {_RES_FINE} if you expect "
                "many cell types (0.8 is often too coarse)."
            ),
        }
    return {
        "tier": "coarse",
        "resolution": float(_RES_COARSE),
        "resolution_sweep": [0.8],
        "primary_metric": "ARI",
        "reasons": [],
        "note": "",  # no tip for the common case
    }


def resolve_hvg_mode(
    adata: Any | None = None,
    *,
    mode: str = "auto",
    n_types: int | None = None,
    n_obs: int | None = None,
    n_density_pops: float | int | None = None,
    rule_branch: str | None = None,
    label_key: str | None = None,
) -> dict[str, Any]:
    """Choose product HVG operating mode: ``compact`` / ``balanced`` / ``fine``.

    ``mode="auto"`` (default):

    - **fine** (k floor ≥2000, res 1.5) only when **labels** say multi-type
      (``n_types≥15``) or structure is **v7 fine-atlas band**. High density
      pop count alone does **not** force fine — that was wrongly re-raising
      SHORT soft lists (500→1000) back to 2000 and undoing compact wins
      (base 1000 + append 200 = 1200).
    - **compact** — structure SHORT geometry: soft buffer 500→1000 kept,
      append 200, res 0.8 (v3 / pancreas-like).
    - **balanced** — default fixed-2000 product path.

    Explicit ``mode`` in {balanced, fine, compact} skips detection.
    """
    m = str(mode or "auto").lower().strip()
    if m not in _HVG_MODES:
        raise ValueError(f"mode must be one of {sorted(_HVG_MODES)}, got {mode!r}")

    # Fill from adata
    if adata is not None:
        if n_obs is None:
            n_obs = int(getattr(adata, "n_obs", 0) or 0) or None
        if n_types is None and label_key and label_key in getattr(adata, "obs", {}):
            n_types = count_label_types(adata.obs[label_key])
        if n_density_pops is None or rule_branch is None:
            try:
                h = (adata.uns.get("scfair") or {}).get("hvg") or {}
            except Exception:
                h = {}
            auto_n = h.get("auto_n") if isinstance(h, Mapping) else {}
            if not isinstance(auto_n, Mapping):
                auto_n = {}
            st = auto_n.get("structure") if isinstance(auto_n.get("structure"), Mapping) else {}
            rx = st.get("rule_explain") if isinstance(st.get("rule_explain"), Mapping) else st
            feat = st.get("features") if isinstance(st.get("features"), Mapping) else {}
            if rule_branch is None:
                rule_branch = (
                    auto_n.get("rule_branch")
                    or st.get("rule_branch")
                    or (rx or {}).get("rule_branch")
                )
            if n_density_pops is None:
                n_density_pops = (rx or {}).get("n_density_pops")
                if n_density_pops is None:
                    n_density_pops = feat.get("n_density_pops")

    reasons: list[str] = []
    branch = str(rule_branch or "")

    def _as_fine() -> dict[str, Any]:
        return {
            "mode": "fine",
            "n_top_floor": 2000,
            "append_budget": 200,
            "allow_short_soft_buffer": False,
            "cluster_resolution": float(_RES_FINE),
            "reasons": list(reasons),
            "auto": m == "auto",
            # Machine-readable notes stay short; not printed as product tips.
            "note": f"fine mode; after HVG cluster at Leiden ≈ {_RES_FINE}.",
        }

    def _as_compact() -> dict[str, Any]:
        # High density-pop count still suggests finer *downstream* clustering,
        # but gene-list stays short (soft 500→1000).
        res = float(_RES_COARSE)
        if n_density_pops is not None:
            try:
                if float(n_density_pops) >= _FINE_N_DENSITY_POPS:
                    res = float(_RES_FINE)
            except (TypeError, ValueError):
                pass
        return {
            "mode": "compact",
            "n_top_floor": None,
            "append_budget": 200,
            "allow_short_soft_buffer": True,
            "cluster_resolution": res,
            "reasons": list(reasons),
            "auto": m == "auto",
            "note": f"compact mode; after HVG cluster at Leiden ≈ {res}.",
        }

    def _as_balanced() -> dict[str, Any]:
        return {
            "mode": "balanced",
            "n_top_floor": 2000,
            "append_budget": 200,
            "allow_short_soft_buffer": True,
            "cluster_resolution": float(_RES_COARSE),
            "reasons": list(reasons),
            "auto": m == "auto",
            "note": f"balanced mode; after HVG cluster at Leiden ≈ {_RES_COARSE}.",
        }

    if m == "fine":
        reasons.append("user_mode_fine")
        return _as_fine()
    if m == "compact":
        reasons.append("user_mode_compact")
        return _as_compact()
    if m == "balanced":
        reasons.append("user_mode_balanced")
        return _as_balanced()

    # ---- auto detection (order matters) ----
    # 1) True fine: author multi-type labels or v7 fine-atlas gene budget.
    if n_types is not None and int(n_types) >= _FINE_N_TYPES:
        reasons.append(f"n_types={int(n_types)}>={_FINE_N_TYPES}")
        return _as_fine()
    if "fine_atlas" in branch or branch.startswith("v7_fine_atlas"):
        reasons.append("structure_fine_atlas_band")
        return _as_fine()

    # 2) Structure SHORT / soft 500→1000 → compact (keep 1000+append).
    # Do NOT promote to fine just because n_density_pops is large — that
    # re-floored k to 2000 and wiped measured compact wins (1200 genes).
    if (
        branch.startswith("short_")
        or "+k_buffer:500→1000" in branch
        or "short_hard" in branch
        or "short_unstable" in branch
    ):
        reasons.append("structure_short_geometry")
        return _as_compact()

    reasons.append("default_balanced")
    return _as_balanced()


def resolve_cluster_resolution(
    adata: Any | None = None,
    *,
    resolution: float | str = "auto",
    label_key: str | None = None,
    n_types: int | None = None,
    n_density_pops: float | int | None = None,
    rule_branch: str | None = None,
) -> dict[str, Any]:
    """Pick Leiden resolution: ``"auto"`` → coarse 0.8 / fine 1.5, or honor a float.

    **Automatic switch** (when ``resolution="auto"``):

    - fine (1.5) if ``n_types≥15``, density pops ≥15, or structure
      ``v7_fine_atlas`` branch (from ``adata.uns['scfair']['hvg']`` when
      present);
    - else coarse (0.8).

    Pass an explicit float to override. Does not modify ``adata`` or HVG genes.

    Parameters
    ----------
    adata
        Optional AnnData: used to read ``label_key`` counts and structure meta
        under ``uns['scfair']['hvg']``.
    resolution
        ``"auto"`` (default) or a Leiden resolution float.
    label_key
        ``obs`` column for type count when ``n_types`` is not given.
    """
    if not (
        resolution is None or (isinstance(resolution, str) and str(resolution).lower() == "auto")
    ):
        try:
            res_f = float(resolution)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resolution must be 'auto' or a float, got {resolution!r}") from exc
        return {
            "tier": "manual",
            "resolution": res_f,
            "resolution_sweep": [res_f],
            "primary_metric": "ARI",
            "reasons": ["user_override"],
            "note": f"User-fixed Leiden resolution={res_f}.",
            "auto": False,
        }

    # Fill gaps from adata
    if adata is not None:
        if n_types is None and label_key is not None and label_key in getattr(adata, "obs", {}):
            n_types = count_label_types(adata.obs[label_key])
        h = {}
        try:
            h = (adata.uns.get("scfair") or {}).get("hvg") or {}
        except Exception:
            h = {}
        auto_n = h.get("auto_n") if isinstance(h, Mapping) else None
        if not isinstance(auto_n, Mapping):
            auto_n = {}
        st = auto_n.get("structure") if isinstance(auto_n.get("structure"), Mapping) else {}
        rx = st.get("rule_explain") if isinstance(st.get("rule_explain"), Mapping) else st
        feat = st.get("features") if isinstance(st.get("features"), Mapping) else {}
        if rule_branch is None:
            rule_branch = (
                auto_n.get("rule_branch") or st.get("rule_branch") or (rx or {}).get("rule_branch")
            )
        if n_density_pops is None:
            n_density_pops = (rx or {}).get("n_density_pops")
            if n_density_pops is None:
                n_density_pops = feat.get("n_density_pops")

    rec = recommend_cluster_resolution(
        n_types=n_types,
        n_density_pops=n_density_pops,
        rule_branch=str(rule_branch) if rule_branch else None,
    )
    rec = dict(rec)
    rec["auto"] = True
    return rec


def diagnose_from_labels(
    labels: pd.Series | Sequence[Any] | np.ndarray,
    *,
    min_cluster_size: int = 1,
) -> dict[str, Any]:
    """Imbalance diagnosis from an ``obs`` label column (pre-call planning).

    Use this **before** choosing scFair vs scanpy when you already have coarse
    cell-type or sample annotations. Does not run HVG.

    Missing labels (``NaN`` / empty / the string ``"nan"``) are dropped before
    size metrics; the count is reported as ``n_labels_dropped``.
    """
    s_raw = pd.Series(labels)
    n_input = int(len(s_raw))
    present = s_raw.notna()
    s = s_raw[present].astype(str)
    s = s[(s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none")]
    n_dropped = n_input - int(len(s))
    counts = s.value_counts()
    if min_cluster_size > 1:
        counts = counts[counts >= int(min_cluster_size)]
    metrics = cluster_size_metrics(counts)
    tips = _imbalance_tips(metrics, source="user_labels")
    rec = _recommendation_from_imbalance(metrics)
    n_types = int(metrics.get("n_clusters") or 0)
    downstream = recommend_cluster_resolution(n_types=n_types if n_types > 0 else None)
    if n_dropped > 0:
        tips.append(
            f"Dropped {n_dropped}/{n_input} missing or empty labels before "
            "imbalance metrics; check annotation coverage."
        )
    if rec == "keep_current":
        tips.append(
            "Label imbalance alone does not say whether scFair will beat "
            "scanpy — compare both methods if the choice matters."
        )
    if downstream["tier"] == "fine" and downstream.get("note"):
        tips.append(str(downstream["note"]))
    return {
        "source": "user_labels",
        "metrics": metrics,
        "imbalance": metrics["imbalance"],
        "recommendation": rec,
        "benefit_evidence": "none" if rec == "use_scanpy_or_none" else "not_predictable",
        "known_no_gain_regime": rec == "use_scanpy_or_none",
        "downstream_clustering": downstream,
        "n_labels_input": n_input,
        "n_labels_dropped": n_dropped,
        "tips": tips,
    }


def diagnose_hvg_run(
    *,
    balance_method: str,
    n_top_genes_used: int | None,
    n_top_is_auto: bool = False,
    auto_n_strategy: str | None = None,
    structure_meta: Mapping[str, Any] | None = None,
    config_check: Mapping[str, Any] | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Build a diagnosis dict for one finished HVG call.

    ``config_check`` is the result of :func:`check_config`. Flags are folded
    in; tips are not re-logged. ``structure_meta`` is the structure auto_n
    detail dict when ``n_top_genes='auto'``.
    """
    method = str(balance_method or "none")
    metrics = cluster_size_metrics(None)
    k = int(n_top_genes_used) if n_top_genes_used is not None else None

    if config_check is None:
        pre = check_config(n_top_genes=n_top_genes_used, balance_method=method, log=False)
        flags: list[str] = list(pre["flags"])
        tips: list[str] = list(pre["tips"])
    else:
        flags = list(config_check.get("flags") or [])
        tips = []

    strat = str(auto_n_strategy or "").lower() if auto_n_strategy else ""
    if n_top_is_auto and strat in ("structure", "auto") and k is not None:
        stips, sflags = _structure_auto_tips(k=k, structure_meta=structure_meta)
        for f in sflags:
            if f not in flags:
                flags.append(f)
        tips.extend(stips)

    evidence, recommendation = _assess(method=method, flags=flags)

    if recommendation == "use_scanpy_or_none" and "use_scanpy" not in " ".join(tips).lower():
        tips.insert(
            0,
            "This setup usually matches plain scanpy HVG — balance_method='none' is enough.",
        )

    nd_for_res = None
    branch_for_res = None
    if structure_meta is not None:
        sm0 = dict(structure_meta)
        rx0 = sm0.get("rule_explain") if isinstance(sm0.get("rule_explain"), Mapping) else sm0
        feat0 = sm0.get("features") if isinstance(sm0.get("features"), Mapping) else {}
        nd_for_res = (rx0 or {}).get("n_density_pops")
        if nd_for_res is None:
            nd_for_res = feat0.get("n_density_pops")
        branch_for_res = sm0.get("rule_branch") or (rx0 or {}).get("rule_branch")
    downstream = recommend_cluster_resolution(
        n_density_pops=nd_for_res,
        rule_branch=str(branch_for_res) if branch_for_res else None,
    )
    if downstream["tier"] == "fine" and "downstream_fine_resolution" not in flags:
        flags.append("downstream_fine_resolution")
        note = str(downstream.get("note") or "").strip()
        if note and note not in tips and recommendation != "use_scanpy_or_none":
            tips.append(note)

    tips = _finalize_user_tips(tips, recommendation=recommendation, flags=flags)

    out: dict[str, Any] = {
        "source": "config_only",
        "metrics": metrics,
        "imbalance": metrics["imbalance"],
        "imbalance_source": "config_only",
        "n_clusters_kept": metrics["n_clusters"],
        "n_clusters_dropped": 0,
        "n_top_genes_used": k,
        "balance_method": method,
        "flags": flags,
        "benefit_evidence": evidence,
        "recommendation": recommendation,
        "known_no_gain_regime": evidence == "none",
        "downstream_clustering": downstream,
        "tips": tips,
        "thresholds": {
            "max_frac_strong": _MAX_FRAC_STRONG,
            "max_frac_balanced": _MAX_FRAC_BALANCED,
            "max_min_ratio_strong": _RATIO_STRONG,
            "max_min_ratio_balanced": _RATIO_BALANCED,
            "rare_frac": _RARE_FRAC,
        },
    }

    if log:
        _emit_diagnosis_log(out)

    return out


def _finalize_user_tips(
    tips: Sequence[str],
    *,
    recommendation: str,
    flags: Sequence[str],
    max_tips: int = 2,
) -> list[str]:
    """Deduplicate and cap user-facing tips (flags stay complete in diagnosis)."""
    out: list[str] = []
    seen: set[str] = set()
    # When the run already has a clear "stop / use scanpy" message, extra
    # structure / imbalance lines only add noise.
    hard_stop = recommendation == "use_scanpy_or_none" or "balance_method_none" in flags
    for raw in tips:
        t = str(raw).strip()
        if not t or t in seen:
            continue
        if hard_stop and (t.startswith("After HVG, cluster") or t.startswith("Auto selected")):
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tips:
            break
    return out


def _structure_auto_tips(
    *,
    k: int,
    structure_meta: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Short, user-facing tips for structure auto_n (flags stay detailed).

    Product UX: at most one short line. Soft-buffer (500→1000) and mid-k
    outcomes are normal — they only set flags; the done line already shows
    the final gene count. Research detail lives in ``adata.uns``, not tips.
    """
    tips: list[str] = []
    flags: list[str] = []
    sm = dict(structure_meta or {})
    rx = sm.get("rule_explain") if isinstance(sm.get("rule_explain"), Mapping) else sm
    branch = str(sm.get("rule_branch") or (rx or {}).get("rule_branch") or "")
    short_blocked = bool(sm.get("short_blocked") or (rx or {}).get("short_blocked"))
    k_buf_raw = (rx or {}).get("k_buffer_raw")
    if k_buf_raw is None and structure_meta is not None:
        k_buf_raw = structure_meta.get("k_buffer_raw")
    no_buffer = bool((rx or {}).get("no_buffer") or sm.get("no_buffer"))

    # Soft buffer / mid k: flag only (final count is already in the done line).
    if "k_buffer" in branch or (
        k_buf_raw is not None and int(k) != int(k_buf_raw) and int(k) < 2000
    ):
        flags.append("structure_k_buffer")
    elif int(k) <= 1500 and "short" not in branch and not short_blocked:
        flags.append("structure_mid_k")

    # User tips only when the choice is unusual or needs a next step.
    if "fine_mode_floor" in branch:
        flags.append("structure_fine_mode_floor")
        tips.append(f"Auto selected {int(k)} genes. Pass n_top_genes=... to override.")
    elif short_blocked or "antishort:" in branch or "anti_short" in branch:
        flags.append("structure_short_blocked")
        tips.append(f"Auto selected {int(k)} genes. Pass n_top_genes=... to override.")
    elif no_buffer or ("no_buffer:" in branch and int(k) <= 500) or int(k) <= 500:
        flags.append("structure_short_k")
        tips.append(
            f"Auto selected a short list ({int(k)} genes). "
            "Pass n_top_genes=2000 for a standard list."
        )
    elif int(k) >= 2000 and branch.startswith("v7_fine_atlas"):
        flags.append("structure_v7_band_floor")
        flags.append("downstream_fine_resolution")
        tips.append(f"Auto selected {int(k)} genes. After HVG, cluster at Leiden ≈ {_RES_FINE}.")
    # else: common ~1000/1500/2000 — silent (done line is enough)

    return tips, flags


def _assess(*, method: str, flags: Sequence[str]) -> tuple[str, str]:
    """Return (benefit_evidence, recommendation)."""
    del flags
    if method == "none":
        return "none", "use_scanpy_or_none"
    return "not_predictable", "keep_current"


def _imbalance_tips(metrics: Mapping[str, Any], *, source: str) -> list[str]:
    imb = metrics.get("imbalance")
    tips: list[str] = []
    # "unknown" / missing sizes: silent (common on append; not actionable).
    if imb in (None, "unknown"):
        return tips
    if imb == "degenerate":
        tips.append(
            "Too few intermediate groups for cluster-based rebalancing. "
            "Try a higher resolution, or use balance_method='none'/'append'."
        )
        return tips

    max_frac = metrics.get("max_frac")
    n_rare = metrics.get("n_rare_clusters")
    # One short line only when imbalance is strong (actionable-ish).
    if imb == "strong":
        tips.append(
            f"Strong size imbalance (largest group ≈ {_fmt_frac(max_frac)}"
            f"{f'; {n_rare} small groups' if n_rare else ''})."
        )
    # moderate / balanced: no tip (noise for most users)
    return tips


def _recommendation_from_imbalance(metrics: Mapping[str, Any]) -> str:
    """Only the degenerate case is decidable from labels alone.

    Fewer than two populations means cluster-vs-rest has no "rest" -- that is
    arithmetic, not a forecast. Every other tier used to map to a method
    recommendation, but imbalance does not predict which is right, so the
    user has to measure it.
    """
    imb = metrics.get("imbalance")
    n = int(metrics.get("n_clusters") or 0)
    if imb in ("degenerate", "unknown") or n < 2:
        return "use_scanpy_or_none"
    return "keep_current"


def _emit_diagnosis_log(diag: Mapping[str, Any]) -> None:
    evidence = diag.get("benefit_evidence")
    rec = diag.get("recommendation")
    metrics = diag.get("metrics") or {}
    logger.info(
        "diagnosis: imbalance=%s benefit_evidence=%s recommendation=%s "
        "(n_clusters=%s max_frac=%s max/min=%s flags=%s)",
        diag.get("imbalance"),
        evidence,
        rec,
        diag.get("n_clusters_kept"),
        _fmt_frac(metrics.get("max_frac")),
        _fmt_ratio(metrics.get("max_min_ratio")),
        list(diag.get("flags") or []),
    )
    # WARN only where there is a measured problem: a regime known to gain
    # nothing, or a config whose two settings cancel. "We cannot predict the
    # margin" is not a warning about the user's data, so it stays at INFO --
    # warning on every ordinary call would make the channel noisy.
    actionable = evidence in ("none", "config_conflict") or rec in (
        "use_scanpy_or_none",
        "check_config",
    )
    for tip in diag.get("tips") or []:
        if actionable:
            logger.warning("diagnosis: %s", tip)
        else:
            logger.info("diagnosis: %s", tip)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = equal, →1 unequal)."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = x.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def _shannon_evenness(fracs: np.ndarray) -> float | None:
    p = np.asarray(fracs, dtype=float)
    p = p[p > 0]
    n = p.size
    if n <= 1:
        return 1.0 if n == 1 else None
    entropy = float(-np.sum(p * np.log(p)))
    return float(entropy / np.log(n))


def _fmt_frac(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.2f}"


def _fmt_ratio(x: Any) -> str:
    if x is None:
        return "NA"
    v = float(x)
    if not np.isfinite(v):
        return "inf"
    return f"{v:.1f}"
