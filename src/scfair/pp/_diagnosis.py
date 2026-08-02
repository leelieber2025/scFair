"""Post-selection diagnostics: imbalance metrics + known no-gain regimes.

These helpers are **advisory**. They do not change gene selection. They answer
two questions, and the distinction between them is deliberate:

1. **Descriptive.** Do intermediate (or user-supplied) populations look
   size-imbalanced? This is a measurement of the data, reported as-is.
2. **Grounded advice.** Is this call in a regime known to yield nothing
   over plain variance-based HVG selection — inert, degenerate, or
   misconfigured?

What these helpers deliberately do **not** do is predict the size of the
gain from the imbalance metrics. See "Why there is no expected_benefit"
below.

Known no-gain / conflict regimes:

- ``k >= 3000``: the balanced re-ranking's advantage over plain HVG tends
  to vanish at that list length.
- ``n_clusters_kept < 2``: cluster-vs-rest has no "rest", every
  specificity score collapses → result ≈ global HVG. Structural, not
  statistical.
- ``neighbor_contrast > 0`` with ``resolution < 0.75``: the two settings
  target the same failure and cancel, measured worse than either alone.
- Adjacent rare boundaries: often need ``neighbor_contrast`` +
  ``resolution>=1`` + a smaller *k*, not bare default hybrid. Whether a
  rare population is *adjacent* to a common sibling cannot be read off
  sizes, so this is offered as a conditional tip, never as a
  recommendation.

Why there is no ``expected_benefit``
------------------------------------
An earlier revision graded calls ``high`` / ``moderate`` / ``low`` /
``none`` from the imbalance tier. That mapping does not hold up: on a
labeled evaluation panel, size imbalance does not correlate with measured
benefit, and the direction of the (weak, non-significant) correlation
flips depending on how the margin is measured — a sign that it is noise,
not a usable signal.

So imbalance is reported as a **description of the data**, and the only
benefit-related claims made here are the ones with a measurement behind
them.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._auto_n import select_n_top_elbow

logger = logging.getLogger(__name__)

# Downstream clustering protocol tiers (advisory; does not change HVG genes).
# Validated on seurat_v4 20k: append@2000+200 loses on l2 at Leiden 0.8 but
# wins at 1.5; deeper append_budget=500 did not beat m=200 there.
_FINE_N_TYPES = 15
_FINE_N_DENSITY_POPS = 15
_RES_COARSE = 0.8
_RES_FINE = 1.5
_HVG_MODES = frozenset({"auto", "balanced", "fine", "compact"})

# Operational cut-offs (documented, not magic biology). Tuned to be conservative:
# "strong" should fire on Cao-like majority + long tail; "balanced" on even blobs.
_MAX_FRAC_STRONG = 0.60
_MAX_FRAC_BALANCED = 0.45
_RATIO_STRONG = 15.0
_RATIO_BALANCED = 5.0
_EVENNESS_BALANCED = 0.85
_EVENNESS_STRONG = 0.60
_RARE_FRAC = 0.05  # cluster fraction treated as "tail / rare-ish"
_K_NO_GAIN = 3000
_PC_ELBOW_MIN_PCS = 4  # below this, an elbow read is not meaningful
_PC_ELBOW_LOW_FRAC = 0.7  # elbow at <70% of n_pcs_used -> "fewer would do"


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
    neighbor_contrast: float = 0.0,
    resolution: float | None = None,
    blend_global: float | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Parameter-only checks, resolvable **before** any data is touched.

    These are the findings a caller can act on while it is still cheap to act:
    the balanced methods spend ~90% of their runtime on the intermediate
    PCA -> neighbours -> Leiden, so a warning that arrives after it has run has
    already cost the user the thing it is warning about.

    Everything here is a function of the arguments alone. Findings that need the
    clustering (how many clusters survived, how skewed they are) necessarily
    come later, from :func:`diagnose_hvg_run`.

    Returns ``{"flags": [...], "tips": [...]}``. ``diagnose_hvg_run`` folds the
    same flags into its record but does not re-log them, so each finding is
    emitted once, at the earliest point it is known.
    """
    method = str(balance_method or "none")
    # Normalize aliases so hybrid_rerank / inverse_size match the right checks.
    _method_alias = {
        "hybrid_rerank": "hybrid",
        "rerank": "hybrid",
        "weighted_score": "score",
        "size_power": "score",
        "inverse_size": "score",
        "cell_reweight": "reweight",
    }
    method = _method_alias.get(method, method)
    flags: list[str] = []
    tips: list[str] = []

    if method == "none":
        flags.append("balance_method_none")
        tips.append(
            "balance_method='none' is the scanpy-like global HVG path; "
            "no cluster-fair reallocation is applied."
        )
    elif method == "append":
        # Product default: silent on success; no intermediate clustering cost.
        pass

    k = None
    if isinstance(n_top_genes, (int, np.integer)):
        k = int(n_top_genes)
    if k is not None and k >= _K_NO_GAIN and method in ("hybrid", "score", "reweight"):
        flags.append("k_ge_3000")
        tips.append(
            f"n_top_genes={k} >= 3000: measured advantage of scFair over scanpy "
            "highly_variable_genes often vanishes at this fixed size. "
            "Structure auto may still pick long k on Duo-like data; for a "
            "classical fixed list use n_top_genes=2000."
        )

    if (
        neighbor_contrast > 0
        # resolution="auto" is resolved from the density field after the
        # neighbour graph exists, so its value is not knowable here -- this
        # check stays a parameter-only check and simply skips that case.
        and isinstance(resolution, (int, float, np.number))
        and not isinstance(resolution, bool)
        and float(resolution) < 0.75
        and method in ("hybrid", "score")
    ):
        flags.append("nc_low_resolution")
        tips.append(
            f"neighbor_contrast={neighbor_contrast:.2f} with "
            f"resolution={float(resolution):.2f}: the two settings target the "
            "same failure and cancel — a low resolution merges the sibling "
            "populations the contrast needs. Measured worse than either alone. "
            "Use resolution>=1.0, or turn neighbor_contrast off."
        )

    # blend_global is a re-rank strength knob, not a fairness lever: lowering
    # it increases equal-share "starved" clusters on panels that show
    # deprivation, the opposite of the "more specificity fixes allocation"
    # intuition.
    if method == "hybrid" and blend_global is not None and float(blend_global) < 0.95:
        flags.append("blend_global_low_vs_deprivation")
        tips.append(
            f"blend_global={float(blend_global):.2f} < 0.95: measured equal-share "
            "deprivation (n_starved / min_share) gets *worse* as α falls — "
            "un-normalised specificity is dominated by strong clusters, so more "
            "weight on it amplifies them rather than rescuing starved ones. "
            "Keep blend_global near 0.95; do not lower it to 'fix "
            "fairness'. Allocation fairness belongs in post-hoc "
            "allocation_method research ('starved_topup' / 'coverage'), "
            "off by default; 'cap' was removed."
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
            "note": (
                "Fine multi-type setting: cluster / evaluate at Leiden "
                f"resolution≈{_RES_FINE} (not only 0.8). Prefer macro-F1 "
                "alongside ARI. Product append keeps append_budget=200 "
                "(deeper budgets were not better on seurat_v4 l2)."
            ),
        }
    return {
        "tier": "coarse",
        "resolution": float(_RES_COARSE),
        "resolution_sweep": [0.8],
        "primary_metric": "ARI",
        "reasons": [],
        "note": (
            f"Coarse / default setting: Leiden resolution≈{_RES_COARSE} is "
            "enough for matched-k ARI vs scanpy HVG."
        ),
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
            try:
                n_types = int(pd.Series(adata.obs[label_key]).astype(str).nunique())
            except Exception:
                n_types = None
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
            "note": (
                "fine mode: k floor ≥2000 (no SHORT→1000 list), append_budget=200, "
                f"downstream Leiden≈{_RES_FINE}."
            ),
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
            "note": (
                "compact mode: structure short soft-buffer (500→1000) allowed, "
                f"append_budget=200, downstream Leiden≈{res}."
            ),
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
            "note": (
                "balanced mode: product default append @k≈2000 + budget 200, "
                f"downstream Leiden≈{_RES_COARSE}."
            ),
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
            try:
                n_types = int(pd.Series(adata.obs[label_key]).astype(str).nunique())
            except Exception:
                n_types = None
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
    """
    s = pd.Series(labels).astype(str)
    s = s[s.notna() & (s != "") & (s != "nan")]
    counts = s.value_counts()
    if min_cluster_size > 1:
        counts = counts[counts >= int(min_cluster_size)]
    metrics = cluster_size_metrics(counts)
    tips = _imbalance_tips(metrics, source="user_labels")
    rec = _recommendation_from_imbalance(metrics)
    n_types = int(metrics.get("n_clusters") or 0)
    downstream = recommend_cluster_resolution(n_types=n_types if n_types > 0 else None)
    if rec == "keep_current":
        tips.append(
            "Imbalance does not predict whether scFair will beat scanpy here. "
            "If the choice matters, run both at matched k and "
            "compare over a range of clustering resolutions, not one."
        )
    if downstream["tier"] == "fine":
        tips.append(downstream["note"])
    return {
        "source": "user_labels",
        "metrics": metrics,
        "imbalance": metrics["imbalance"],
        "recommendation": rec,
        "benefit_evidence": "none" if rec == "use_scanpy_or_none" else "not_predictable",
        "known_no_gain_regime": rec == "use_scanpy_or_none",
        "downstream_clustering": downstream,
        "tips": tips,
    }


def diagnose_hvg_run(
    *,
    balance_method: str,
    n_top_genes_used: int | None,
    n_top_is_auto: bool = False,
    auto_n_strategy: str | None = None,
    structure_meta: Mapping[str, Any] | None = None,
    resolution: float | None = None,
    neighbor_contrast: float = 0.0,
    min_cluster_size: int | None = None,
    clustering: Mapping[str, Any] | None = None,
    n_clusters_used: int | None = None,
    config_check: Mapping[str, Any] | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Build a full diagnosis dict for one finished (or planned) HVG call.

    Prefer intermediate-cluster sizes from ``clustering`` when present.

    ``config_check`` is the result of :func:`check_config`, already emitted by
    the caller before the run. Its flags are folded into the record so the
    diagnosis is complete, but its tips are **not** re-logged — one finding,
    one message, at the earliest point it could be known. Pass ``None`` and the
    parameter-only checks are recomputed and logged here instead.

    ``structure_meta`` is the structure auto_n detail dict (from
    ``estimate_n_top_structure`` / ``auto_n['structure']``) when
    ``n_top_genes='auto'`` used the structure path. Used for short-k and
    fine-atlas band tips.
    """
    method = str(balance_method or "none")

    sizes = None
    if clustering:
        raw_sizes = clustering.get("cluster_sizes")
        if isinstance(raw_sizes, Mapping) and raw_sizes:
            # Prefer kept clusters (those that score specificity).
            dropped = {str(x) for x in (clustering.get("clusters_dropped") or [])}
            kept = {k: int(v) for k, v in raw_sizes.items() if str(k) not in dropped}
            sizes = kept if kept else raw_sizes

    # append freezes global HVG + secondary ranks; no intermediate clustering.
    uses_clusters = method not in ("none", "append")

    metrics = cluster_size_metrics(sizes)
    if uses_clusters and n_clusters_used is not None and metrics["n_clusters"] == 0:
        # Clustering diag missing sizes but we know how many scored. Guarded on
        # method: with balance_method='none'/'append' no intermediate clustering
        # runs at all, so "fewer than 2 populations" would be a category error
        # -- there is no finding about the data, only about the configuration.
        if int(n_clusters_used) < 2:
            metrics = {**metrics, "n_clusters": int(n_clusters_used), "imbalance": "degenerate"}

    n_kept = metrics["n_clusters"]
    if clustering and clustering.get("n_clusters_kept") is not None:
        n_kept = int(clustering["n_clusters_kept"])
    elif n_clusters_used is not None:
        n_kept = int(n_clusters_used)

    n_dropped = 0
    if clustering and clustering.get("clusters_dropped") is not None:
        n_dropped = len(clustering["clusters_dropped"])
    elif clustering and clustering.get("n_clusters_total") is not None:
        n_dropped = max(0, int(clustering["n_clusters_total"]) - n_kept)

    k = int(n_top_genes_used) if n_top_genes_used is not None else None

    # Parameter-only findings. When the caller already ran check_config (the
    # normal path) they were logged before the expensive step; carry the flags
    # but leave the tips out of `tips` so nothing is said twice.
    if config_check is None:
        pre = check_config(
            n_top_genes=n_top_genes_used,
            balance_method=method,
            neighbor_contrast=neighbor_contrast,
            resolution=resolution,
            log=False,
        )
        flags: list[str] = list(pre["flags"])
        tips: list[str] = list(pre["tips"])
    else:
        flags = list(config_check.get("flags") or [])
        tips = []

    # --- post-run only: findings that needed the clustering to exist ---
    n_total_clusters = None
    if clustering and clustering.get("n_clusters_total") is not None:
        n_total_clusters = int(clustering["n_clusters_total"])

    if n_kept < 2 and uses_clusters:
        flags.append("insufficient_clusters")
        tips.append(
            f"Only {n_kept} intermediate cluster(s) pass min_cluster_size"
            f"{f'={min_cluster_size}' if min_cluster_size is not None else ''}. "
            "Cluster-vs-rest specificity has no signal → result ≈ scanpy HVG. "
            "Lower min_cluster_size, raise resolution, or use balance_method='none'."
        )

    # Two (or fewer) intermediate communities: every specificity / allocation
    # mechanism is already degraded — equal-share "fairness" on 2 blobs is a
    # no-op that looks balanced. Treat as a hard structural warning even when
    # both pass min_cluster_size.
    if (
        uses_clusters
        and n_total_clusters is not None
        and int(n_total_clusters) <= 2
        and "insufficient_clusters" not in flags
    ):
        flags.append("coarse_partition")
        tips.append(
            f"Intermediate Leiden resolved only {int(n_total_clusters)} community(ies). "
            "Cluster-vs-rest specificity and post-hoc allocation (starved_topup / "
            "coverage) cannot recover rare types that were never split out. "
            "Pass a higher float resolution (e.g. 1.0–2.0) or use balance_method='none'."
        )

    # `auto` resolves k only during the run, so a k>=3000 that came from auto
    # could not have been caught by the pre-flight check.
    strat = str(auto_n_strategy or "").lower() if auto_n_strategy else ""
    if (
        k is not None
        and k >= _K_NO_GAIN
        and method in ("hybrid", "score", "reweight")
        and "k_ge_3000" not in flags
    ):
        flags.append("k_ge_3000")
        if n_top_is_auto and strat in ("structure", "auto"):
            # structure intentionally picks a long (3-4k) list for certain
            # data shapes; the k>=3000 warning would otherwise read as a
            # false positive there.
            tips.append(
                f"structure auto resolved n_top={k} >= 3000 (often Duo-like long "
                "branch). Fixed-size hybrid often plateaus by 3000; "
                "this long pick is intentional for residual structure. For a "
                "classical protocol use n_top_genes=2000."
            )
        else:
            tips.append(
                f"n_top_genes resolved to {k} >= 3000: measured advantage of scFair "
                "over scanpy HVG often vanishes at this fixed size. Consider "
                "bounding n_top_max, or passing n_top_genes=2000."
            )

    # Structure auto: explain short/mid k and v7 band (protocol vs geometry).
    if n_top_is_auto and strat in ("structure", "auto") and k is not None:
        stips, sflags = _structure_auto_tips(k=k, structure_meta=structure_meta)
        for f in sflags:
            if f not in flags:
                flags.append(f)
        tips.extend(stips)

    if n_dropped > 0 and method != "none":
        flags.append("clusters_dropped")
        size_detail = ""
        if clustering:
            all_sizes = clustering.get("cluster_sizes") or {}
            dropped_ids = clustering.get("clusters_dropped") or []
            dropped_sizes = {
                str(c): int(all_sizes[str(c)]) for c in dropped_ids if str(c) in all_sizes
            }
            if dropped_sizes:
                mcs = (
                    f" (min_cluster_size={min_cluster_size})"
                    if min_cluster_size is not None
                    else ""
                )
                size_detail = f" Sizes: {dropped_sizes}{mcs}."
        tips.append(
            f"{n_dropped} intermediate cluster(s) fell below min_cluster_size and "
            "did not contribute to specificity (often the rare tail the balancing "
            f"is meant to protect).{size_detail} Consider lowering min_cluster_size or "
            "raising resolution if rare recovery matters."
        )

    # --- descriptive: what the intermediate populations look like ---
    # Imbalance is measured on the *intermediate partition*, not on true
    # cell-type sizes. When that partition is coarse (≤2 blobs), failed
    # structure looks "balanced" (max/min≈1) and would otherwise recommend
    # keep_current — the reverse of the right advice.
    imbalance_source = "intermediate_clusters"
    if method != "none":
        if n_total_clusters is not None and int(n_total_clusters) <= 2:
            imbalance_source = "intermediate_clusters_unreliable"
            tips.append(
                f"imbalance metrics (n={n_total_clusters}, "
                f"max/min={_fmt_ratio(metrics.get('max_min_ratio'))}) describe the "
                "collapsed intermediate partition, not the data. Coarse Leiden "
                "partitions often look balanced while rare types are absorbed. "
                "Use scfair.pp.diagnose_from_labels(obs['cell_type']) for true "
                "size imbalance, or raise resolution."
            )
        else:
            tips.extend(_imbalance_tips(metrics, source="intermediate_clusters"))

    if clustering:
        var_ratio = clustering.get("pca_variance_ratio")
        n_pcs_used = clustering.get("n_pcs_used")
        if var_ratio and n_pcs_used is not None:
            pc_tip = _pc_elbow_tip(var_ratio, int(n_pcs_used))
            if pc_tip:
                flags.append("pc_elbow")
                tips.append(pc_tip)

        if clustering.get("resolution_source") == "fallback":
            if "resolution_fallback" not in flags:
                flags.append("resolution_fallback")
            res_used = clustering.get("resolution")
            tips.append(
                f"resolution='auto' fell back to {res_used} "
                f"(density field unusable: "
                f"{(clustering.get('granularity') or {}).get('reason', 'unknown')}). "
                "If results look under-partitioned, pass an explicit float "
                "resolution (e.g. 1.0–2.0)."
            )

        if clustering.get("resolution_source") == "density_field":
            target = clustering.get("n_populations_target")
            achieved = clustering.get("n_clusters_achieved")
            calls = clustering.get("n_leiden_calls")
            if target is not None and achieved is not None and int(target) != int(achieved):
                flags.append("density_target_unreached")
                call_note = (
                    f" ({calls} Leiden probes, close to the search cap)"
                    if isinstance(calls, int) and calls >= 10
                    else f" ({calls} Leiden probes)"
                    if calls is not None
                    else ""
                )
                tips.append(
                    f"resolution='auto' targeted {target} population(s) from the "
                    f"density field but Leiden's resolution search landed on "
                    f"{achieved}{call_note} — Leiden's cluster count is a step "
                    "function of resolution and can skip the exact target. Not "
                    "necessarily wrong, but the two counts disagree."
                )

        # Under-partition / unresolved-rare signal: clusters_dropped only
        # catches communities that *formed* then fell under min_cluster_size.
        # When Leiden never splits a rare type out of a majority blob, the
        # drop list stays empty while specificity still cannot score it.
        if clustering.get("under_partition_warning"):
            if "under_partition" not in flags:
                flags.append("under_partition")
            tips.append(
                "Intermediate Leiden under-partitioned relative to the density "
                "field target (rare types may be absorbed into majority "
                "communities). clusters_dropped will not list them. Pass a "
                "higher float resolution (e.g. 0.5–1.0) if rare recovery matters."
            )
        min_frac = clustering.get("min_cluster_frac")
        n_total = clustering.get("n_clusters_total")
        if (
            method != "none"
            and min_frac is not None
            and n_total is not None
            and int(n_total) >= 3
            and float(min_frac) >= 0.08
            and int(n_total) <= 6
        ):
            # Coarse partition whose smallest blob is still large (≥8% of
            # cells): possible that a rarer type exists but was not resolved.
            if "possible_unresolved_rare" not in flags:
                flags.append("possible_unresolved_rare")
            tips.append(
                f"Smallest intermediate community is {100.0 * float(min_frac):.1f}% of "
                f"cells across only {int(n_total)} communities. If a rarer type "
                "exists in the data, it may be unresolved (absorbed, not "
                "dropped). Check clustering.min_cluster_frac / raise resolution."
            )

        # Allocation layer may report "n_starved=0" because structure only
        # produced 2 units — that is not "checked and fair".
        for prefix in ("starved_topup", "coverage"):
            status = clustering.get(f"{prefix}_allocation_status")
            if status == "skipped_structure_too_coarse":
                if "allocation_skipped_coarse" not in flags:
                    flags.append("allocation_skipped_coarse")
                n_u = clustering.get(f"{prefix}_n_units")
                tips.append(
                    f"allocation_method='{prefix}' was requested but skipped: "
                    f"structure resolved only {n_u} unit(s) (need ≥3). "
                    "n_starved=0 here means 'nothing to check', not 'fair'."
                )

    imbalance = metrics["imbalance"]
    has_rare_tail = int(metrics.get("n_rare_clusters") or 0) > 0

    if (
        method in ("hybrid", "score")
        and neighbor_contrast <= 0
        and has_rare_tail
        and imbalance in ("moderate", "strong")
        and n_kept >= 3
    ):
        # Structured signal kept for programmatic use; tip text dropped
        # (2026-08-01 user feedback: too verbose for routine output). See
        # git history for the full explanation if this needs restoring.
        flags.append("rare_tail_no_neighbor_contrast")

    if n_top_is_auto and method in ("hybrid", "score", "reweight"):
        # Same as above -- flag kept, tip text dropped for brevity.
        flags.append("auto_n_double_cluster")

    evidence, recommendation = _assess(method=method, flags=flags)

    if recommendation == "use_scanpy_or_none" and "use_scanpy" not in " ".join(tips).lower():
        tips.insert(
            0,
            "This call is in a regime measured to give nothing over scanpy HVG — "
            "scanpy highly_variable_genes (or balance_method='none') is enough.",
        )
    # `not_predictable` carries no separate tip text; `evidence` /
    # `recommendation` below expose it programmatically.

    # Downstream clustering protocol (P0 coarse / P1 fine) — advisory only.
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
        if downstream["note"] not in tips:
            tips.append(downstream["note"])

    out: dict[str, Any] = {
        "source": imbalance_source if sizes else "config_only",
        "metrics": metrics,
        "imbalance": imbalance,
        "imbalance_source": imbalance_source if sizes else "config_only",
        "n_clusters_kept": n_kept,
        "n_clusters_dropped": n_dropped,
        "n_top_genes_used": k,
        "balance_method": method,
        "flags": flags,
        # Deliberately not an "expected_benefit" grade -- see module docstring.
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
            "k_no_gain": _K_NO_GAIN,
        },
    }

    if log:
        _emit_diagnosis_log(out)

    return out


def _structure_auto_tips(
    *,
    k: int,
    structure_meta: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Tips for structure auto_n k: short list, mid list, v7 band miss."""
    tips: list[str] = []
    flags: list[str] = []
    sm = dict(structure_meta or {})
    # Prefer nested rule_explain; fall back to top-level keys.
    rx = sm.get("rule_explain") if isinstance(sm.get("rule_explain"), Mapping) else sm
    branch = str(sm.get("rule_branch") or (rx or {}).get("rule_branch") or "")
    feat = sm.get("features") if isinstance(sm.get("features"), Mapping) else {}
    nd = (rx or {}).get("n_density_pops")
    if nd is None:
        nd = feat.get("n_density_pops")
    nl = (rx or {}).get("n_leiden")
    if nl is None:
        nl = feat.get("n_leiden")
    vm = (rx or {}).get("valley_median")
    if vm is None:
        vm = feat.get("valley_median")
    ratio = (rx or {}).get("ratio")
    if ratio is None and nl is not None and nd is not None:
        try:
            nd_f, nl_f = float(nd), float(nl)
            ratio = (nl_f / nd_f) if nd_f > 0 else None
        except (TypeError, ValueError):
            ratio = None
    v7_miss = (rx or {}).get("v7_band_miss")
    k_source = sm.get("k_source")
    short_blocked = bool(sm.get("short_blocked") or (rx or {}).get("short_blocked"))
    short_why = sm.get("short_block_reason") or (rx or {}).get("short_block_reason")
    short_raw = sm.get("short_k_raw")
    if short_raw is None:
        short_raw = (rx or {}).get("short_k_raw")
    dens_conf = (rx or {}).get("density_confidence") or feat.get("density_confidence")

    def _fmt_num(x: Any, ndigits: int = 3) -> str:
        try:
            return f"{float(x):.{ndigits}f}"
        except (TypeError, ValueError):
            return str(x)

    parts = []
    if nd is not None:
        parts.append(f"nd={nd}")
    if nl is not None:
        parts.append(f"nl={nl}")
    if ratio is not None:
        parts.append(f"nl/nd={_fmt_num(ratio)}")
    if vm is not None:
        parts.append(f"vm={_fmt_num(vm)}")
    if dens_conf is not None:
        parts.append(f"density_conf={dens_conf}")
    if branch:
        parts.append(f"branch={branch}")
    if k_source:
        parts.append(f"k_source={k_source}")
    feat_s = ", ".join(parts) if parts else "structure features unavailable"

    # Soft buffer and/or fine-mode k floor (keep messages accurate).
    k_buf_raw = (rx or {}).get("k_buffer_raw")
    if k_buf_raw is None and structure_meta is not None:
        k_buf_raw = structure_meta.get("k_buffer_raw")
    if "fine_mode_floor" in branch:
        flags.append("structure_fine_mode_floor")
        tips.append(
            f"structure auto fine-mode k floor → n_top={k} ({feat_s}). "
            "Multi-type / fine-atlas mode keeps list length ≥2000 (does not "
            "keep SHORT soft lists at 1000). Use mode='compact' to allow "
            "base≈1000 + append."
        )
    elif "k_buffer" in branch or (k_buf_raw is not None and int(k) != int(k_buf_raw)):
        flags.append("structure_k_buffer")
        # Prefer ladder target in branch (500→1000) over final k when a
        # later floor changed the number.
        shown = str(k)
        if "k_buffer:" in branch:
            try:
                shown = branch.split("k_buffer:", 1)[1].split("+", 1)[0]
            except Exception:
                shown = f"{int(k_buf_raw)}→{k}" if k_buf_raw is not None else str(k)
        elif k_buf_raw is not None:
            shown = f"{int(k_buf_raw)}→{k}"
        tips.append(
            f"structure auto soft-buffered n_top={shown} ({feat_s}). "
            "Classical discrete picks lift one rung (500→1000, 1000→1500, "
            "1500→2000). Pass a fixed n_top_genes if you want a classical 2000."
        )

    # Residual anti-SHORT floor (only when post-buffer k was still ≤500).
    if short_blocked or "anti_short" in branch:
        flags.append("structure_short_blocked")
        raw_note = f" (rule raw k={short_raw})" if short_raw is not None else ""
        why_note = f" reason={short_why}" if short_why else ""
        tips.append(
            f"structure auto residual SHORT floor{raw_note} → n_top={k} "
            f"({feat_s}).{why_note} Soft buffer did not lift above 500 and "
            "density was untrusted, so k was floored to ~2000. Pass "
            "n_top_genes=2000 for a classical protocol."
        )
    elif int(k) <= 500:
        flags.append("structure_short_k")
        tips.append(
            f"structure auto chose n_top={k} ({feat_s}). Density geometry favours a "
            "short HVG list (many cores / deep valleys), not a classical ~2000 "
            "global HVG. This is intentional on compact multi-core data; if the "
            "dataset is a large multi-type atlas and results look under-selected, "
            "pass n_top_genes=2000 explicitly."
        )
        if v7_miss:
            flags.append("structure_v7_band_miss")
            tips.append(
                "v7 fine-atlas floor (would raise k toward 2000 on seurat-like "
                f"atlases) did not apply; missing condition(s): {list(v7_miss)}. "
                "Typical misses: valley_median < 0.78, nl/nd > 1.12, or "
                "n_density_pops outside [12, 20]."
            )
    elif int(k) <= 1500 and "short" not in branch:
        flags.append("structure_mid_k")
        tips.append(
            f"structure auto chose n_top={k} ({feat_s}). Mid list for "
            "CITE-like / intermediate structure. For a classical fixed "
            "protocol use n_top_genes=2000."
        )
    elif int(k) >= 2000 and branch.startswith("v7_fine_atlas"):
        flags.append("structure_v7_band_floor")
        tips.append(
            f"structure auto chose n_top={k} via v7 fine-atlas band ({feat_s}): "
            "large n_obs, mid density-pop count, deep valleys, Leiden≈density. "
            "This path prevents seurat-like short-k collapse."
        )
        # Pair gene-list floor with fine downstream protocol (measured seurat_v4).
        flags.append("downstream_fine_resolution")
        tips.append(
            "Fine-atlas structure: for multi-type / l2-style labels, cluster at "
            f"Leiden resolution≈{_RES_FINE} (0.8 under-partitions). Product "
            "append_budget=200 is enough — deeper append did not beat m=200 on "
            "seurat_v4 l2 at res=1.5."
        )

    return tips, flags


def _assess(*, method: str, flags: Sequence[str]) -> tuple[str, str]:
    """Return (benefit_evidence, recommendation).

    Every branch here is a regime with a measurement or a structural argument
    behind it. Imbalance is deliberately **not** an input -- it does not
    correlate with measured benefit, and grading calls from it would invent
    a number the data doesn't support.
    """
    flagset = set(flags)

    # measured or structural: nothing to gain here
    if method == "none":
        return "none", "use_scanpy_or_none"
    if method == "append":
        # Product path: global base + secondary ranks; no cluster re-rank claim.
        return "not_predictable", "keep_current"
    if "insufficient_clusters" in flagset:
        return "none", "use_scanpy_or_none"
    if "k_ge_3000" in flagset:
        return "none", "use_scanpy_or_none"

    # Structure layer failed before scoring/allocation can help. Do not emit
    # keep_current just because the collapsed partition looks balanced.
    if flagset & {
        "coarse_partition",
        "resolution_fallback",
        "under_partition",
        "allocation_skipped_coarse",
    }:
        return "structure_unreliable", "raise_resolution"

    # the two settings target the same failure and cancel out
    if "nc_low_resolution" in flagset:
        return "config_conflict", "check_config"

    # Everything else. The honest answer is that we cannot forecast the margin
    # from anything observable without labels -- not that the margin is small.
    return "not_predictable", "keep_current"


def _imbalance_tips(metrics: Mapping[str, Any], *, source: str) -> list[str]:
    imb = metrics.get("imbalance")
    tips: list[str] = []
    if imb == "unknown":
        tips.append(
            f"No cluster sizes available from {source}; cannot judge imbalance. "
            "Pass obs labels to scfair.pp.diagnose_from_labels(...) or run a "
            "balanced method so intermediate clusters are recorded."
        )
        return tips
    if imb == "degenerate":
        tips.append(f"Fewer than 2 populations in {source} — fair reallocation cannot help.")
        return tips

    max_frac = metrics.get("max_frac")
    ratio = metrics.get("max_min_ratio")
    n_rare = metrics.get("n_rare_clusters")
    n = metrics.get("n_clusters")
    # Descriptive only. Earlier revisions attached an expected gain to each
    # tier; that mapping does not hold -- the most imbalanced datasets in
    # evaluation are not reliably the ones that gain the most, and one of
    # the largest multi-type atlases loses to plain HVG outright. State the
    # shape of the data; let the user run the comparison.
    if imb == "strong":
        tips.append(
            f"imbalance: strong "
            f"(n={n}, max_frac={_fmt_frac(max_frac)}, max/min={_fmt_ratio(ratio)}, "
            f"{n_rare} rare<5%)."
        )
    elif imb == "moderate":
        tips.append(
            f"imbalance: moderate "
            f"(n={n}, max_frac={_fmt_frac(max_frac)}, max/min={_fmt_ratio(ratio)})."
        )
    elif imb == "balanced":
        tips.append(
            f"imbalance: balanced "
            f"(n={n}, max_frac={_fmt_frac(max_frac)}, max/min={_fmt_ratio(ratio)})."
        )
    return tips


def _pc_flatten_point(
    var_ratio: np.ndarray, *, tail_frac: float = 0.3, floor_mult: float = 1.5
) -> int:
    """First PC index (1-based) whose variance is within ``floor_mult`` of the
    tail's noise floor (median of the last ``tail_frac`` of the curve).

    A different question from the elbow: the elbow is where the curve bends
    *sharpest*, which real PCA spectra hit early (dominated by the first
    point or two); this is where the curve gets *close to flat*, which is
    later and moves a lot with how close "close" means -- shown directly by
    calling this at two thresholds rather than picking one (see
    ``_pc_elbow_tip``).
    """
    n = var_ratio.size
    tail_n = max(3, int(n * tail_frac))
    floor = float(np.median(var_ratio[-tail_n:]))
    thresh = floor * floor_mult
    idx = int(np.argmax(var_ratio <= thresh))
    return idx + 1


def _pc_elbow_tip(pca_variance_ratio: Sequence[float], n_pcs_used: int) -> str | None:
    """Read off where the PCA scree curve bends/flattens (descriptive, not a claim).

    Reuses :func:`scfair.pp._auto_n.select_n_top_elbow` -- same perpendicular-
    distance-from-chord elbow already used for the gene HVG curve, applied
    here to ``pca_variance_ratio`` instead. No new computation: this array is
    already carried in ``diag_out`` (``_highly_variable_genes.py``) from the
    PCA scFair's intermediate clustering already ran.

    Reports **three** numbers (elbow + two flatten thresholds) instead of
    one -- on real data the spread between them is large and
    threshold-dependent, and no single number predicts how many PCs a
    specific clustering task needs. A curve of *overall* variance cannot
    see *class-discriminative* structure that lives in individually
    low-variance PCs (the same blindness elbow/knee methods have on a
    gene-score curve). Showing the spread says that honestly; picking one
    number would not.
    """
    arr = np.asarray(pca_variance_ratio, dtype=float)
    n = arr.size
    if n < _PC_ELBOW_MIN_PCS or n_pcs_used < _PC_ELBOW_MIN_PCS:
        return None
    elbow = select_n_top_elbow(arr, k_min=2, k_max=n)
    flat_loose = _pc_flatten_point(arr, floor_mult=1.2)
    flat_strict = _pc_flatten_point(arr, floor_mult=1.05)
    if elbow <= int(_PC_ELBOW_LOW_FRAC * n_pcs_used) or flat_strict < n_pcs_used:
        return (
            f"PC variance: elbow={elbow}, ~flat by PC {flat_loose}-{flat_strict} "
            f"(of {n_pcs_used}). A range, not a recommendation."
        )
    if elbow >= n_pcs_used:
        return (
            f"PC variance hasn't flattened by PC {n_pcs_used} (last used); "
            "consider raising n_pcs if you have the budget."
        )
    return None


def _recommendation_from_imbalance(metrics: Mapping[str, Any]) -> str:
    """Only the degenerate case is decidable from labels alone.

    Fewer than two populations means cluster-vs-rest has no "rest" -- that is
    arithmetic, not a forecast. Every other tier used to map to ``use_hybrid``
    or ``use_scanpy_or_none``, but imbalance does not predict which is right,
    so the honest answer is that the user has to measure it.
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
    # warning on every ordinary call would train people to ignore the channel.
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
