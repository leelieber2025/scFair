"""Fair / balanced highly variable gene selection.

Public entry point: :func:`highly_variable_genes`.

Product default and balanced strategies:

**append** (default) — freeze global HVG top-``k``, then append the next ``m``
  genes from the **same** global ranking (ranks ``k+1 … k+m``). No intermediate
  clustering and no zero-sum re-rank inside a 2× pool. Final size is
  ``k + append_budget`` (capped at ``n_vars``).

**hybrid** (opt-in) — keep most of the global HVG ranking, swap in specificity
  1. Global HVG (``flavor``) → Leiden clusters.
  2. Cluster-vs-rest specificity scores (same as ``score``).
  3. Re-rank the top ``2 × n_top`` **global** candidates by
     ``blend_global · global + (1-blend_global) · specificity`` and take top
     ``n_top``. Keeps the scanpy subspace (NK / structure) while promoting
     identity genes that are also globally variable.

**score** — pure cluster-vs-rest specificity
  ``S_g = Σ_c w_c · logFC⁺_{g,c}`` with ``w_c ∝ n_c^{β}`` (no global anchor).

**reweight** — cell-reweighted global HVG
  Resample cells so cluster mass ``∝ n_c^{β}``, then one global ``flavor`` pass.

``balance_method='none'`` is a single global HVG pass (scanpy-like).

Users never call store/restore helpers.
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse

from .._utils import UNS_KEY, _is_integer_counts_like
from ._auto_n import resolve_n_top_genes
from ._diagnosis import check_config, diagnose_hvg_run, resolve_hvg_mode
from ._granularity import DEFAULT_RESOLUTION_FALLBACK, resolution_from_density_field
from ._options import HVG_OPTION_FIELD_NAMES, HVGOptions, resolve_hvg_options
from ._raw_counts import (
    INTERNAL_COUNTS_LAYER,
    _prepare_counts_layer,
    _restore_raw_counts,
    _store_raw_counts,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_COUNTS_FLAVORS = frozenset({"seurat_v3", "seurat_v3_paper", "pearson_residuals"})
_LOG_FLAVORS = frozenset({"seurat", "cell_ranger"})
_ALL_FLAVORS = _COUNTS_FLAVORS | _LOG_FLAVORS

_BALANCE_ALIASES = {
    None: "none",
    "none": "none",
    # Product default: freeze global top-k, append next m from the same ranking
    # (no intermediate clustering / no pool re-rank).
    "append": "append",
    # Legacy in-pool re-rank (2×k hybrid blend); opt-in research path.
    "hybrid": "hybrid",
    "hybrid_rerank": "hybrid",
    "rerank": "hybrid",
    "score": "score",
    "weighted_score": "score",
    "size_power": "score",
    "inverse_size": "score",  # score + default β=0
    "reweight": "reweight",
    "cell_reweight": "reweight",
}

BalanceMethod = Literal[
    "append",
    "hybrid",
    "hybrid_rerank",
    "rerank",
    "score",
    "weighted_score",
    "size_power",
    "inverse_size",
    "reweight",
    "cell_reweight",
    "none",
]

# Post-hybrid gene reallocation (product default is "none").
# - "coverage": legitimate units + own-marker coverage floor (experimental)
# - "starved_topup": legitimate units + equal-share starvation top-up
# "cap" was removed in 0.2.0.
_ALLOCATION_METHODS = frozenset({"coverage", "none", "starved_topup"})
_ALLOCATION_ALIASES = {
    "starved": "starved_topup",
    "topup": "starved_topup",
    "starved_top_up": "starved_topup",
}

# Recoverable failures for HVG / graph / scoring paths (intentionally broad
# but not bare Exception — excludes KeyboardInterrupt / SystemExit).
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


# Stage announcements. The balanced methods run PCA + neighbours + Leiden
# internally, so a default hybrid@2000 call on a mid-size dataset blocks for
# tens of seconds (`n_top_genes="auto"` structure path does ~4 PCA/neighbours
# builds: 3 seeds + hybrid). Messages are emitted *before* each slow stage,
# not after, so a waiting user can tell the call is alive.
def _progress(on: bool, msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    logger.info(text)
    if on:
        print(f"scfair: {text}", file=sys.stderr, flush=True)


def _progress_default(adata: Any) -> bool:
    """Announce stages only when the call is slow enough to look like a hang."""
    return bool(adata.n_obs >= 10_000 or adata.n_obs * adata.n_vars >= 5e7)


_AUTO_STRATEGIES = frozenset(
    {
        "auto",
        "ensemble",
        "elbow",
        "knee",
        "cumfrac",
        "coverage",
        "silhouette",
        "structure",
    }
)

_MITO_RE = re.compile(r"^(MT-|MT\.|mt-|mt\.)")
# Ribosomal structural proteins only. Do not match RPS6KA*/RPS6KB* kinases
# (MAPK / mTOR pathway members that share the RPS6 prefix).
# Matches RPL13, RPLP0, RPS6, RPSA, Rpl19 — not RPS6KA1 / RPS6KB1.
_RIBO_RE = re.compile(r"^(?:RPL|Rpl)P?\d+[A-Za-z]?$|^(?:RPS|Rps)(?:A|\d+[A-Za-z]?)$")


# Columns scanpy / scFair may write during a partial HVG run.
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

# Flavor → primary score column(s) written by scanpy (preferred first).
# Used so a re-run with a different flavor cannot silently reuse a leftover
# column from a previous call (G3 state pollution).
_FLAVOR_SCORE_COLS: dict[str, tuple[str, ...]] = {
    "seurat_v3": ("variances_norm", "variances"),
    "seurat_v3_paper": ("variances_norm", "variances"),
    "pearson_residuals": ("residual_variances",),
    "seurat": ("dispersions_norm", "dispersions"),
    "cell_ranger": ("dispersions_norm", "dispersions"),
}

# Scanpy HVG columns cleared before each flavor run so leftovers cannot win
# a priority walk in _variability_raw_scores.
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


def _snapshot_adata_for_rollback(adata: Any) -> dict[str, Any]:
    """Capture caller-visible state that a failed HVG call may leave dirty."""
    var_had = {c: adata.var[c].copy() for c in _HVG_PARTIAL_VAR_COLS if c in adata.var.columns}
    return {
        "var_had": var_had,
        "var_cols": set(adata.var.columns),
        "had_clusters": "scfair_hvg_clusters" in adata.obs.columns,
        "clusters": (
            adata.obs["scfair_hvg_clusters"].copy()
            if "scfair_hvg_clusters" in adata.obs.columns
            else None
        ),
        "had_scfair_log": "_scfair_log" in getattr(adata, "layers", {}),
        "had_scfair_counts": INTERNAL_COUNTS_LAYER in getattr(adata, "layers", {}),
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
    # intermediate cluster labels
    if not snap["had_clusters"] and "scfair_hvg_clusters" in adata.obs.columns:
        del adata.obs["scfair_hvg_clusters"]
    elif snap["clusters"] is not None:
        adata.obs["scfair_hvg_clusters"] = snap["clusters"]
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
        # only drop if we created a stub; if scanpy wrote mid-run, leave
        # a note rather than guessing — prefer restore-from-snap path above
        pass


def highly_variable_genes(
    adata: Any,
    *,
    # --- product surface (keep short) ---
    n_top_genes: int | str = 2000,
    flavor: str = "seurat_v3",
    layer: str | None = None,
    resolution: float | str = "auto",
    min_cluster_size: int = 30,
    balance_method: str | None = "append",
    mode: str = "auto",
    blend_global: float = 0.95,
    marker_genes: Sequence[str] | None = None,
    marker_mode: str | None = None,
    diagnose: bool = True,
    strict: bool = False,
    random_state: int = 0,
    inplace: bool = True,
    subset: bool = False,
    progress: bool | None = None,
    # --- experimental / secondary ---
    options: HVGOptions | None = None,
    **legacy_kwargs: Any,
) -> pd.DataFrame | None:
    """Select highly variable genes with optional fair / balanced strategies.

    Foolproof usage — call once; raw-count snapshot and intermediate restore
    are handled internally.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``.X`` or ``layers['counts']`` (or pass
        ``layer=``).
    diagnose
        If True (default), write ``adata.uns['scfair']['hvg']['diagnosis']``
        with intermediate-cluster **imbalance** metrics (descriptive) and
        warnings for regimes known to gain nothing over scanpy HVG:
        ``k>=3000``, fewer than two scoring clusters (structural), and
        ``neighbor_contrast`` with ``resolution<0.75``. Does **not**
        change the selected genes. Set False to silence the extra log lines.
        Pre-call planning with known labels:
        :func:`scfair.pp.diagnose_from_labels`.
    n_top_genes
        Number of HVGs to select, or an automatic strategy:

        - int: fixed count (**default ``2000``** — classical community size)
        - ``"auto"``: uses ``auto_n_method`` (default **structure v7**) to
          pick k, then hybrid re-selects in global top-``2×k``. Costs ~4×
          PCA→neighbours (3 structure seeds + 1 hybrid pass); prefer fixed
          ``2000`` for papers / reproducible protocols.
        - ``"structure"``: density-valley rule **v7** directly
          (``select_n_top_from_structure``) — short/mid/long by valley
          geometry + fine-atlas band guard
        - ``"ensemble"``: ensemble v2.2 (shape + cumfrac + depth-aware
          anchor; usually ≈2000)
        - ``"elbow"``, ``"knee"``, ``"cumfrac"``, ``"coverage"``,
          ``"silhouette"``: single strategy

        Automatic selection ranks a pool of size ``n_top_max`` then cuts
        to chosen k; markers with ``marker_extra=True`` are still appended
        outside this count. A resolved k ≥ 80% of ``n_vars`` emits a warning
        (near-identity HVG selection).
    n_top_min, n_top_max
        Bounds for automatic ``n_top_genes`` (default 500–5000). Both ends
        are respected by structure and ensemble paths (structure ``k_max``
        matches this ceiling).
    auto_n_method
        When ``n_top_genes='auto'``, which strategy to use (default
        ``"structure"``). Use ``"ensemble"`` for the previous depth-aware
        ~2000 anchor. Ignored for integer ``n_top_genes``.
    flavor
        Scanpy HVG method for the global pass and for ``reweight``:

        - counts: ``seurat_v3``, ``seurat_v3_paper``, ``pearson_residuals``
        - log: ``seurat``, ``cell_ranger``
    layer
        Explicit counts layer; default auto-prepares ``layers['counts']``.
    resolution, min_cluster_size
        Leiden settings for intermediate populations. Clusters smaller than
        ``min_cluster_size`` are excluded from weight / specificity mass
        (their cells still keep neutral weight in ``reweight``).

        The intermediate clustering runs
        ``normalize_total -> log1p -> scale(max_value=10) -> PCA -> neighbours ->
        Leiden`` by default (``scale_clustering=True``). Zero-centring prevents
        highly expressed genes from dominating intermediate PCA and collapsing
        multi-type structure (measured ARI gains of ~0.3 on imbalanced synthetic
        atlases). Pass ``options=HVGOptions(scale_clustering=False)`` to restore
        the pre-0.2 unscaled path.

        ``resolution`` defaults to **``"auto"``** (density-field target cluster
        count for intermediate Leiden). Pass a float (e.g. ``0.5`` or ``1.0``)
        for a fixed Leiden resolution.
    marker_genes
        Optional gene symbols used as prior knowledge (must match ``var_names``).
        ``None`` by default, and scFair ships **no** built-in marker lists — see
        the warning below before supplying one.

        .. warning::

           **Injecting markers is not free, even in the "free" mode.** Forcing in
           markers the algorithm did not already select can tilt the PCA
           subspace of the intermediate clustering (extra lineage genes pull the
           subspace toward them), which can hurt the purity of *unrelated*
           clusters — even **with ``marker_extra=True``**, i.e. even though no
           algorithm-selected gene was displaced. The damage is a *subspace*
           effect, not slot displacement, so ``marker_extra`` does not protect
           you from it.

           Check ``adata.uns['scfair']['hvg']['n_marker_genes_already_selected']``
           after a run with ``marker_mode="none"``: if it is close to
           ``n_marker_genes``, injection is redundant. Legitimate reason to inject
           anyway: a panel that must appear in the output for downstream reasons,
           independent of clustering quality.
    marker_mode
        How to use ``marker_genes``. Default **``None``** resolves to
        ``"force"`` when ``marker_genes`` is non-empty, else ``"none"``
        (so the bare default is not "force" with no markers):

        - ``"force"``: always include present markers in the final set
        - ``"none"``: ignore ``marker_genes``
    strict
        If True, do **not** silently fall back when ``flavor="seurat_v3"``
        fails or structure auto_n raises — re-raise instead. Default False
        keeps research-friendly recovery but always emits
        :func:`warnings.warn` and records ``flavor_used`` /
        ``fallback_reason`` in ``uns``.

        A third mode, ``"boost"`` (add a score bonus instead of hard-including),
        was **removed before the first release**. It added the bonus to a score
        already min-max normalised to ``[0, 1]``, so the default bonus of 1.0 put
        every marker above the highest attainable score of any non-marker gene —
        making it a hard include by arithmetic, identical to ``"force"`` with
        ``marker_extra=False``.
    marker_extra
        Only for ``marker_mode="force"``: if True (**default**), markers are
        appended on top of ``n_top_genes`` and do **not** displace
        algorithm-selected genes (final size may exceed ``n_top_genes``).
        If False, markers occupy slots within ``n_top_genes``.

        Note that this bounds only the *displacement* cost, not the subspace cost
        described under ``marker_genes``.
    balance_method
        - ``"append"`` (**default**): freeze the global HVG top-``n_top``
          (scanpy-like base) and **append** the next ``append_budget`` genes
          from the same global ranking (ranks ``n_top+1 … n_top+m``). Does
          **not** re-rank inside a 2× pool and does **not** require intermediate
          clustering — avoids zero-sum displacement of majority-type genes and
          ADT-style rest-contrast poisoning. Final size is
          ``n_top + append_budget`` (capped at ``n_vars``). ``append_budget=0``
          matches ``none``.
        - ``"hybrid"`` / ``"hybrid_rerank"`` / ``"rerank"``: legacy path —
          global top ``2×n_top`` pool re-ranked by
          ``blend_global * norm(global) + (1-blend_global) * norm(specificity)``.
          Opt-in research; not the product default.
        - ``"score"``: pure specificity ``S_g = Σ_c w_c · logFC^{+}_{g,c}``.
          **Not suitable for rare-type recovery** when intermediate Leiden
          fails to isolate the rare population. Prefer ``append`` or ``none``.
        - ``"reweight"``: cell-resample so cluster mass ``∝ n_c^{β}``, then
          one global HVG pass (``flavor``).
        - ``"none"``: single global HVG (scanpy-like), size exactly ``n_top``.
        - Aliases: ``weighted_score`` / ``size_power`` → score with default
          ``β=0.5``; ``inverse_size`` → score with default ``β=-1``;
          ``cell_reweight`` → reweight.
    balance_power
        ``β`` for cluster mass ``w_c ∝ n_c^{β}`` (normalized), range
        ``[-2, 2]``. Default ``0.5`` (mild large-cluster weight). ``β=0`` is
        equal weight; ``β<0`` up-weights small clusters. Alias
        ``inverse_size`` defaults to ``β=-1`` when this is omitted.
    blend_global
        For ``hybrid`` only: weight on the global score when re-ranking a
        global HVG candidate pool
        ``S = blend_global · norm(global) + (1-blend_global) · norm(spec)``
        (default ``0.95`` — mostly global, light specificity boost).
        ``1`` ≈ global-only; lower values emphasize cluster-vs-rest more.

        **Not a fairness / anti-deprivation knob.** Specificity is an
        un-normalised cross-cluster field (weighted sum + max logFC); lowering
        it *increases* equal-share starved clusters and lowers min_share on
        panels that show deprivation (see the package repo script
        ``examples/blend_global_deprivation.py``:
        https://github.com/leelieber2025/scFair/blob/master/examples/blend_global_deprivation.py
        ). Keep the default; use post-hoc ``allocation_method`` research arms
        for allocation fairness, not a smaller ``blend_global``.
    neighbor_contrast
        For ``hybrid`` / ``score``. Fraction of the peak-specificity term taken
        from a **nearest-neighbour** contrast instead of cluster-vs-rest
        (``0.0`` = current behaviour). Cluster-vs-rest cannot see the boundary
        between two adjacent populations: for a rare subset the "rest" is
        dominated by distant lineages, so genes separating it from its closest
        neighbour earn no credit and the boundary can be lost downstream. At
        ``1.0`` the peak term scores every cluster against its nearest
        neighbour by centroid correlation.
    combine
        How ``hybrid`` merges the global and specificity signals.

        - ``"blend"`` (default): ``blend_global·norm(global) +
          (1-blend_global)·norm(spec)``.
        - ``"best_rank"``: each gene takes its **better** of the two ranks,
          ties broken by the other rank. Structurally this can only promote a
          gene relative to its global rank — it can never demote one, which
          protects rare populations from being pushed out by the global
          ranking. It is the combination rule Zhao et al. (*Genome Biology*
          2025) found best for merging HVG methods. ``blend_global`` is
          ignored.
    cluster_pool
        Size of the gene pool the **intermediate clustering** runs on, decoupled
        from ``n_top_genes``. ``None`` (**default**) keeps the historical
        behaviour: cluster on the top-``n_top_genes`` global mask, so the
        clustering space shrinks as you ask for fewer genes.

        ``n_top_genes="auto"`` does this implicitly — it clusters on its
        ``n_top_max`` pool rather than the final selected count. Setting
        ``cluster_pool`` makes that behaviour explicit and available to
        fixed-``k`` calls. **Experimental.**
    cluster_genes
        Explicit gene list for the intermediate clustering, the general form of
        ``cluster_pool`` (which just takes the global top-N). Mutually exclusive
        with it. Names absent from ``var_names`` are dropped with a warning.

        Exists to make **iterated selection** measurable: pass a previous call's
        selection back in and the second round re-clusters on it, re-scores
        specificity against the new partition, and re-selects from *all* genes.
        Note that the candidate pool must stay global for this to do anything —
        re-selecting ``k`` genes out of a ``k``-gene pool is the identity.
        **Experimental, for benchmarking.**
    cap_allocation, cap_ceiling, cap_merge_threshold
        For ``hybrid`` only. **``cap_allocation`` is deprecated** when
        ``True`` (equal-share *ceiling* trim): high ARI variance, not the
        product path; emits ``DeprecationWarning``. Prefer
        ``allocation_method="none"`` or research ``"starved_topup"`` /
        ``"coverage"``. ``cap_merge_threshold`` is still used by coverage
        / starved_topup as the unit-stability merge threshold.
    allocation_method
        Post-hybrid allocation policy for ``balance_method="hybrid"``:

        - ``None`` (**default**): derive from ``cap_allocation`` (False →
          **``"none"``**).
        - ``"none"``: **product path** — hybrid blend only; no post-hoc
          reallocation.
        - ``"starved_topup"`` (**research**): legitimate units + top-up
          units below 50% equal-share own-gene count. Budget is
          **adaptive**: ``min(need, hard_cap, soft_cap)`` with hard ceiling
          default **10%** of ``n_top`` and soft cap scaling with how many
          units are starved (0 when none). Funded without trimming
          pure-type marker tails. Aliases: ``"starved"``, ``"topup"``.
        - ``"coverage"``: experimental units + own-marker coverage floor
          (different trigger than equal-share starvation).
        - ``"cap"``: **deprecated** equal-share ceiling + backfill.

        Prefer ``allocation_method=`` when comparing policies. Do **not**
        lower ``blend_global`` for fairness — it is not an allocation-fairness
        knob (see ``blend_global`` above).
    spec_on_legitimate_units
        **Experimental** (default ``False``). After intermediate Leiden, merge
        nearest-neighbour pairs that fail stability or pairwise DE into
        *legitimate units*, and compute cluster-vs-rest specificity on **those
        units** instead of raw Leiden labels. Gene selection still uses the
        hybrid 2× pool and ``blend_global``; post-hoc allocation stays off
        unless requested. Research arm: does a cleaner structure object
        improve HVG utility without equal-share reallocation?
        Diagnostics: ``clustering["spec_partition"]``, ``spec_units_merges``.
    progress
        Print stage messages to stderr so a long call is not mistaken for a hang.
        ``None`` (**default**) enables them only when the dataset is large enough
        for the call to visibly block (>=10k cells, or >=5e7 cells x genes);
        ``True``/``False`` force it either way.

        Worth knowing why this exists: the balanced methods run a full
        PCA -> neighbours -> Leiden internally. Fixed-k hybrid does it
        **once**. ``n_top_genes="auto"`` with structure does it about
        **four** times (3 multi-seed structure feature graphs + 1 hybrid
        pass) — not twice. On 6.9k cells x 16.7k genes that is roughly 4s
        for ``balance_method="none"``, ~16s for hybrid@2000, and ~30s+ for
        auto. It scales with cells x genes.

        Messages also always go to the ``scfair.pp._highly_variable_genes`` logger
        at INFO, independent of this setting, for callers that configure logging.
    global_score
        Optional externally computed global variability score, indexed by
        ``var_names`` (higher = more variable). Replaces the ``flavor`` pass as
        the global anchor, so any baseline can be used underneath the balanced
        methods. Mainly for benchmarking — see the repo script
        https://github.com/leelieber2025/scFair/blob/master/examples/hvg_baselines.py
        .
    filter_mito, filter_ribo
        If True, exclude mitochondrial / ribosomal gene symbols from the
        selected HVG set (markers are never filtered). Matching is name-based
        (``MT-`` / ``mt-``, ``RPL*`` / ``RPS*``).
    n_pcs, n_neighbors, random_state
        Intermediate clustering / resampling.
    span, n_bins, min_mean, max_mean, min_disp, max_disp
        Forwarded to scanpy HVG where applicable.
    batch_key
        Optional batch key for global / reweighted HVG only.
    inplace, subset
        Write-back behaviour.

    Returns
    -------
    None or DataFrame

    Notes
    -----
    The balanced methods run a full PCA -> neighbours -> Leiden internally. That
    intermediate clustering is reported rather than discarded:

    - ``adata.obs['scfair_hvg_clusters']`` — the partition itself.
    - ``adata.uns['scfair']['hvg']['clustering']`` — the settings that produced
      it and its shape: ``n_pcs_used`` / ``n_neighbors_used`` (the *clamped*
      values, which can differ from the requested ``n_pcs`` / ``n_neighbors`` on
      small inputs), ``resolution``, ``n_genes_clustered``,
      ``pca_variance_ratio``, ``n_clusters_total`` vs ``n_clusters_kept`` with
      ``cluster_sizes`` and ``clusters_dropped`` (below ``min_cluster_size``),
      and ``n_passes`` (physical intermediate builds; structure auto is 1;
      ensemble auto realign reuses the graph so stays 1 when protocol matches).

      ``n_clusters_kept`` counts clusters passing ``min_cluster_size``; the
      sibling key ``n_clusters_used`` counts those that actually produced a
      specificity score, and is smaller when a kept cluster spans every cell.

    Treat it as a diagnostic, not a substitute for your own clustering: it runs
    on the selection pool, before the final gene set exists, and at
    ``resolution=0.5`` — deliberately finer-grained than a typical downstream
    analysis. ``clusters_dropped`` is the entry worth checking; dropped clusters
    are usually the rare populations the balancing is supposed to protect.

    When ``diagnose=True`` (default), ``adata.uns['scfair']['hvg']['diagnosis']``
    also records imbalance metrics on those intermediate clusters
    (``max_frac``, ``max_min_ratio``, ``imbalance`` tier), a
    ``recommendation`` (``use_scanpy_or_none`` / ``check_config`` /
    ``keep_current``), ``benefit_evidence``, and human-readable ``tips``.
    Advisory only; it does not change the gene set. For planning from known
    cell labels without running HVG, see :func:`scfair.pp.diagnose_from_labels`.

    Note what it does **not** claim: the imbalance metrics are a description of
    your data, not a forecast of how much scFair will help. That mapping was
    tested and found unreliable — no consistent correlation, and what
    correlation exists can flip sign depending on the evaluation protocol.
    Only ``benefit_evidence="none"`` is a measured statement (``k>=3000``,
    fewer than two scoring clusters, or ``balance_method="none"``).

    Experimental knobs (``cluster_pool``, ``allocation_method``, graph
    sizes, …) live on :class:`~scfair.pp.HVGOptions` and are passed as
    ``options=...``. Former top-level names still work with a
    :class:`DeprecationWarning` for one release cycle. ``cap_allocation`` /
    ``allocation_method='cap'`` were **removed** in 0.2.0.
    """
    # --- resolve options bag (legacy kwargs still accepted with warning) ---
    if legacy_kwargs:
        unknown = set(legacy_kwargs) - HVG_OPTION_FIELD_NAMES
        if unknown:
            raise TypeError(
                f"highly_variable_genes() got unexpected keyword argument(s) "
                f"{sorted(unknown)}. Product knobs are the short signature; "
                f"research knobs use options=HVGOptions(...)."
            )
    # Resolve first so options/legacy conflicts raise without a misleading deprecation.
    opt = resolve_hvg_options(options, legacy_kwargs)
    if legacy_kwargs and options is None:
        warnings.warn(
            "Passing research/secondary knobs as top-level keyword arguments is "
            f"deprecated ({sorted(legacy_kwargs)}); use options=HVGOptions(...). "
            "See scfair.pp.HVGOptions.",
            DeprecationWarning,
            stacklevel=2,
        )
    n_top_min = int(opt.n_top_min)
    n_top_max = int(opt.n_top_max)
    auto_n_method = str(opt.auto_n_method)
    marker_extra = bool(opt.marker_extra)
    balance_power = opt.balance_power
    neighbor_contrast = float(opt.neighbor_contrast)
    combine = str(opt.combine)
    cluster_pool = opt.cluster_pool
    cluster_genes = opt.cluster_genes
    consensus_resolutions = opt.consensus_resolutions
    allocation_method = opt.allocation_method
    # renamed from cap_merge_threshold; used by coverage / starved_topup only
    cap_merge_threshold = opt.unit_merge_threshold
    # Internal only: private helpers still accept a ceiling kwarg for the
    # removed "cap" path (never selected). Keep a harmless default.
    cap_ceiling = 1.0
    spec_on_legitimate_units = bool(opt.spec_on_legitimate_units)
    scale_clustering = bool(opt.scale_clustering)
    logfc_space = str(opt.logfc_space)
    global_score = opt.global_score
    n_pcs = int(opt.n_pcs)
    n_neighbors = int(opt.n_neighbors)
    span = float(opt.span)
    n_bins = int(opt.n_bins)
    min_mean = float(opt.min_mean)
    max_mean = float(opt.max_mean)
    min_disp = float(opt.min_disp)
    max_disp = float(opt.max_disp)
    batch_key = opt.batch_key
    filter_mito = bool(opt.filter_mito)
    filter_ribo = bool(opt.filter_ribo)
    # append_budget: None → mode default (200); explicit int is never overridden.
    user_append_budget = opt.append_budget
    if user_append_budget is not None:
        append_budget = int(user_append_budget)
        if append_budget < 0:
            raise ValueError(f"append_budget must be >= 0, got {append_budget}.")
    else:
        append_budget = 200
    label_key = opt.label_key
    store_raw = opt.store_raw
    snapshot_path = opt.snapshot_path

    # Product operating mode (auto → compact / balanced / fine from data).
    mode_info = resolve_hvg_mode(
        adata,
        mode=str(mode),
        label_key=label_key,
        n_obs=int(getattr(adata, "n_obs", 0) or 0) or None,
    )
    hvg_mode = str(mode_info["mode"])
    # Mode fills budget only when user did not set append_budget explicitly.
    if user_append_budget is None and mode_info.get("append_budget") is not None:
        append_budget = int(mode_info["append_budget"])
    # Absolute append_budget blows up small panels (n_top=50 + 200 → 250 of 300).
    # Cap secondary budget at the frozen base size so final ≤ 2×k (still capped
    # at n_vars later). Track requested vs effective for uns / warnings.
    append_budget_requested = int(append_budget)
    append_budget_capped = False

    if not isinstance(strict, bool):
        raise TypeError(
            f"strict must be bool, got {type(strict).__name__}={strict!r}. "
            "Use strict=True or strict=False (string 'yes' is not accepted)."
        )
    if isinstance(n_top_genes, (int, np.integer)) and int(n_top_genes) < 1:
        raise ValueError("n_top_genes must be >= 1.")
    if flavor not in _ALL_FLAVORS:
        raise ValueError(f"Unknown flavor={flavor!r}. Supported: {sorted(_ALL_FLAVORS)}.")
    if balance_method not in _BALANCE_ALIASES:
        raise ValueError(
            f"Unknown balance_method={balance_method!r}. "
            f"Use one of {sorted(set(_BALANCE_ALIASES) - {None})}."
        )
    method = _BALANCE_ALIASES[balance_method]
    beta = _resolve_balance_power(balance_method, balance_power)
    if not 0.0 <= float(blend_global) <= 1.0:
        raise ValueError("blend_global must be in [0, 1].")
    blend_global = float(blend_global)
    if not 0.0 <= float(neighbor_contrast) <= 1.0:
        raise ValueError("neighbor_contrast must be in [0, 1].")
    neighbor_contrast = float(neighbor_contrast)
    if logfc_space not in _LOGFC_SPACES:
        raise ValueError(f"logfc_space must be one of {list(_LOGFC_SPACES)}, got {logfc_space!r}.")
    if logfc_space == "linear":
        warnings.warn(
            "logfc_space='linear' uses pseudocount 1e-9 on normalize_total(1e4) "
            "scale (essentially unregularised; near-zero denominators can explode). "
            "Prefer the default 'log1p' or 'linear_regularised' (pseudo=1.0). "
            "'linear' is retained for benchmark parity with scanpy rank_genes_groups "
            "only.",
            UserWarning,
            stacklevel=2,
        )
    if cluster_pool is not None:
        cluster_pool = int(cluster_pool)
        if cluster_pool < 2:
            raise ValueError("cluster_pool must be >= 2 (or None to disable).")
    # Parameter-only checks, emitted here rather than after the run: the
    # intermediate clustering is ~90% of the runtime, and a warning that arrives
    # once it is finished has already cost the caller the thing it warns about.
    # diagnose_hvg_run() folds these same flags into the record without
    # re-logging, so each finding is surfaced once, as early as it is knowable.
    config_check = check_config(
        n_top_genes=n_top_genes,
        balance_method=balance_method,
        neighbor_contrast=neighbor_contrast,
        resolution=resolution,
        blend_global=blend_global if method == "hybrid" else None,
        log=diagnose,
    )
    if combine not in ("blend", "best_rank"):
        raise ValueError("combine must be 'blend' or 'best_rank'.")
    # Resolve marker_mode after marker_genes is known so default None is clear.
    if marker_mode is None:
        marker_mode = "force" if marker_genes else "none"
    elif marker_mode not in ("force", "none"):
        raise ValueError(
            "marker_mode must be None, 'force', or 'none'. ('boost' was removed "
            "before the first release: it was arithmetically equivalent to "
            "marker_mode='force' with marker_extra=False.)"
        )
    # Post-hybrid allocation (cap removed in 0.2.0).
    if allocation_method is None:
        allocation_method = "none"
    else:
        allocation_method = str(allocation_method).lower()
        allocation_method = _ALLOCATION_ALIASES.get(allocation_method, allocation_method)
    if allocation_method == "cap":
        raise ValueError(
            "allocation_method='cap' was removed in 0.2.0. Use 'none' (default) "
            "or research 'starved_topup' / 'coverage' via HVGOptions."
        )
    if allocation_method not in _ALLOCATION_METHODS:
        raise ValueError(
            f"allocation_method must be one of {sorted(_ALLOCATION_METHODS)}, "
            f"got {allocation_method!r}."
        )

    # Duplicate gene names break every score/rank alignment in this module
    # (pandas cannot reindex on a duplicated axis) and would otherwise surface as
    # an opaque error deep in the call stack. 10x matrices read with gene symbols
    # routinely contain duplicates, and scanpy tolerates them, so say plainly what
    # to do rather than fail cryptically.
    dup = adata.var_names[adata.var_names.duplicated()]
    if len(dup):
        raise ValueError(
            f"adata.var_names has {len(dup)} duplicate entries "
            f"(e.g. {sorted(set(map(str, dup)))[:3]}). scFair aligns scores by gene "
            "name, so names must be unique. Call adata.var_names_make_unique() "
            "first."
        )
    # Duplicate cell names fail later inside raw_snapshot restore (reweight and
    # some restore paths). Catch them here so hybrid/score and reweight share
    # the same early, actionable error.
    dup_obs = adata.obs_names[adata.obs_names.duplicated()]
    if len(dup_obs):
        raise ValueError(
            f"adata.obs_names has {len(dup_obs)} duplicate entries "
            f"(e.g. {sorted(set(map(str, dup_obs)))[:3]}). scFair restores raw "
            "counts by cell name, so barcodes must be unique. Call "
            "adata.obs_names_make_unique() first."
        )
    if n_top_min < 1 or n_top_max < 1:
        raise ValueError(f"n_top_min={n_top_min} and n_top_max={n_top_max} must both be >= 1.")
    if n_top_min > n_top_max:
        # Contradictory bounds must not silently clamp (default min=500 with a
        # lowered max was a common footgun that hid the real interval).
        raise ValueError(
            f"n_top_min={n_top_min} > n_top_max={n_top_max}. Pass a consistent "
            "interval, e.g. options=HVGOptions(n_top_min=100, n_top_max=300) "
            "(default min is 500 — lower both ends when shrinking the range)."
        )
    # inplace=False must leave the caller's object untouched, as in scanpy. Every
    # stage below annotates `adata` (counts layer, var columns, uns, obs clusters),
    # and one of those columns — var["highly_variable"] — silently changes what a
    # later sc.pp.pca does, so leaking them is not cosmetic.
    if not inplace:
        adata = adata.copy()

    _hvg_snap = _snapshot_adata_for_rollback(adata)
    # When subset=True, full-gene recovery after the call needs a snapshot.
    # Auto-enable store_raw unless the user already asked for ondisk / True.
    effective_store_raw: bool | str = store_raw
    if subset and not store_raw:
        effective_store_raw = True
        warnings.warn(
            "subset=True requires a raw_snapshot to recover the full gene universe "
            "after the call; enabling store_raw=True for this run. Pass "
            "options=HVGOptions(store_raw=True) or store_raw='ondisk' to silence, "
            "or subset=False to avoid the extra matrix copy.",
            UserWarning,
            stacklevel=2,
        )
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

        # Preserve a pre-existing scanpy ``uns["hvg"]`` dict: scanpy's own HVG call
        # may overwrite it mid-run; we re-merge at the end instead of replacing with
        # a one-key stub that also lied about flavor after seurat_v3 fallback.
        _prior_scanpy_hvg = (
            dict(adata.uns["hvg"]) if isinstance(adata.uns.get("hvg"), dict) else None
        )

        auto_meta: dict[str, Any] | None = None
        # Configs loaded from YAML/JSON hand over 2000.0 or "2000"; treat anything
        # exactly equal to an integer as that integer instead of a strategy name.
        if not isinstance(n_top_genes, (int, np.integer)):
            try:
                as_float = float(n_top_genes)
            except (TypeError, ValueError):
                pass
            else:
                if np.isfinite(as_float) and float(as_float).is_integer():
                    n_top_genes = int(as_float)
        n_top_is_auto = not isinstance(n_top_genes, (int, np.integer))
        # E3: explicit mode with fixed int k does not change the gene list
        # (mode only steers auto k / soft floors / append budget defaults).
        _mode_arg = str(mode or "auto").lower().strip()
        if (
            not n_top_is_auto
            and _mode_arg not in ("auto", "")
            and _mode_arg in ("compact", "balanced", "fine")
        ):
            warnings.warn(
                f"mode={mode!r} has no effect on gene selection when "
                f"n_top_genes is a fixed int ({int(n_top_genes)}). "
                "mode only influences n_top_genes='auto' (k floor / soft "
                "buffer / product append defaults). Pass n_top_genes='auto' "
                "to let mode choose k, or ignore this warning if you only "
                "wanted mode recorded in uns.",
                UserWarning,
                stacklevel=2,
            )
        if n_top_is_auto and str(n_top_genes) not in _AUTO_STRATEGIES:
            # Validate here, not inside resolve_n_top_genes: everything between this
            # point and there runs a full PCA + neighbours + Leiden, so a mistyped
            # value used to cost tens of seconds before reporting a type error.
            raise ValueError(
                f"Unknown n_top_genes={n_top_genes!r}. Pass an int, or one of "
                f"{sorted(_AUTO_STRATEGIES)}."
            )
        if n_top_is_auto:
            # Rank a large pool, then cut to auto-selected k.
            n_pool = min(int(n_top_max), adata.n_vars)
            n_top_request = n_pool
        else:
            n_top_request = min(int(n_top_genes), adata.n_vars)

        # Cap append_budget ≤ n_base so small panels don't become "select almost
        # everything" (absolute m=200 on n_top=50 → 250 genes).
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

        show = _progress_default(adata) if progress is None else bool(progress)
        # Effective auto strategy (auto → auto_n_method). Used to skip a redundant
        # hybrid@n_top_max pass when structure alone decides k.
        auto_strategy_eff: str | None = None
        if n_top_is_auto:
            auto_strategy_eff = (
                str(auto_n_method).lower()
                if str(n_top_genes).lower() == "auto"
                else str(n_top_genes).lower()
            )
        structure_hybrid_fast = (
            n_top_is_auto and method in ("hybrid", "append") and auto_strategy_eff == "structure"
        )
        if n_top_is_auto and method in ("hybrid", "score") and not structure_hybrid_fast:
            _progress(
                show,
                "n_top_genes='auto' with balance_method=%r: intermediate graph is "
                "shared across select + realign when the protocol matches; pass an "
                "int to skip auto entirely.",
                method,
            )
        elif structure_hybrid_fast:
            _progress(
                show,
                "n_top_genes='auto' (structure): estimate k from structure features, "
                "then %s at that k.",
                "append secondary HVG" if method == "append" else "one hybrid pass",
            )

        # --- Pass 1: global HVG (seeds clustering features for balanced modes) ---
        # Skipped when an anchor is injected: global_score fully overwrites
        # highly_variable / highly_variable_rank and supplies global_scores, so running
        # the flavor pass first only to discard it was pure waste — and worst in the
        # very case the parameter exists for (repeated baseline benchmarking).
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
            # An injected anchor replaces the flavor pass entirely: rank, scores and
            # the highly_variable mask that seeds the intermediate clustering all
            # come from it, so the balanced methods sit on top of *that* baseline.
            gs = pd.Series(global_score).reindex(adata.var_names)
            if gs.isna().all():
                raise ValueError(
                    "global_score does not align with adata.var_names (all NaN after reindex)."
                )
            global_scores = gs.fillna(float(np.nanmin(gs.to_numpy())))
            global_rank = global_scores.rank(ascending=False, method="average")
            top_mask = global_rank <= n_top_request
            adata.var["highly_variable"] = top_mask.to_numpy()
            # scanpy convention: finite rank only for selected genes; rest NaN.
            adata.var["highly_variable_rank"] = global_rank.where(top_mask, np.nan).to_numpy()
        else:
            # Use the flavor that actually ran (may differ after seurat_v3→seurat
            # fallback). Explicit flavor avoids reading leftover score columns
            # from a previous call on the same object (G3).
            _flavor_for_scores = str(hvg_run_meta.get("flavor_used") or flavor)
            global_rank = _gene_rank_series(adata, flavor=_flavor_for_scores)
            global_scores = _variability_raw_scores(
                adata, flavor=_flavor_for_scores, flavor_used=_flavor_for_scores
            )

        # cluster_pool: mask must come from all-gene variability scores, not from
        # global_rank (that is only finite for the top-n_top_request genes).
        # Mutual exclusion is checked even for balance_method='none' so conflicts
        # are never silently ignored.
        if cluster_genes is not None and cluster_pool is not None:
            raise ValueError("Pass either cluster_pool or cluster_genes, not both.")
        cluster_mask: np.ndarray | None = None
        cluster_pool_source = "highly_variable"
        if cluster_genes is not None and method != "none":
            wanted = {str(g) for g in cluster_genes}
            cluster_mask = adata.var_names.astype(str).isin(wanted)
            n_found = int(cluster_mask.sum())
            if n_found < 2:
                raise ValueError(
                    f"cluster_genes matched only {n_found} of {len(wanted)} names in "
                    "var_names; cannot cluster."
                )
            if n_found < len(wanted):
                logger.warning(
                    "cluster_genes: %d of %d names not in var_names; clustering on %d.",
                    len(wanted) - n_found,
                    len(wanted),
                    n_found,
                )
            cluster_pool_source = "cluster_genes"
        elif cluster_pool is not None and method != "none":
            n_cp = max(2, min(int(cluster_pool), adata.n_vars))
            keep = global_scores.sort_values(ascending=False).index[:n_cp]
            cluster_mask = adata.var_names.isin(keep)
            cluster_pool_source = "cluster_pool"

        clustering_diag: dict[str, Any] = {}
        cluster_labels: pd.Series | None = None
        n_clusters_used = 0
        cluster_weights: dict[str, float] = {}
        aggregated: pd.Series | None = None
        selection_tag = "global"
        score_type: str | None = None
        cluster_gene_ranks: dict[str, list[str]] | None = None
        # Shared across auto select + realign: PCA/Leiden/specificity reuse.
        intermediate_cache: dict[str, Any] = {}
        append_meta: dict[str, Any] | None = None
        # Base k for append reporting (structure/auto/fixed request). Filters use
        # n_top_for_markers (= base+append); meta n_top_genes_used reports base.
        n_top_base: int | None = None

        # --- Structure + product path (append or legacy hybrid) ---
        # Resolve k first; append freezes base-k + secondary HVG; hybrid re-ranks.
        if structure_hybrid_fast:
            from ._auto_n import PRODUCT_STRUCTURE_N_SEEDS, estimate_n_top_structure

            try:
                n_final, structure_detail = estimate_n_top_structure(
                    adata,
                    counts_layer=counts_layer,
                    random_state=random_state,
                    version="v7",
                    k_min=int(n_top_min),
                    k_max=min(int(n_top_max), int(adata.n_vars)),
                    n_genes=int(adata.n_vars),
                    # Multi-seed for stability; library default alone is n_seeds=1.
                    n_seeds=PRODUCT_STRUCTURE_N_SEEDS,
                    # Per-seed "n/N (%)" progress + a please-wait note (this loop is
                    # the single longest-running, most opaque step in the auto path).
                    progress=show,
                    hvg_mode=hvg_mode if hvg_mode != "balanced" else "auto",
                )
                # Re-resolve mode with structure features (may promote to fine/compact).
                mode_info = resolve_hvg_mode(
                    adata,
                    mode=str(mode),
                    label_key=label_key,
                    n_obs=int(adata.n_obs),
                    n_density_pops=(structure_detail.get("features") or {}).get("n_density_pops"),
                    rule_branch=structure_detail.get("rule_branch"),
                )
                hvg_mode = str(mode_info["mode"])
                # Mode may change append_budget (today all modes use 200; keep
                # in sync so a future fine-specific budget is not silently stale).
                if user_append_budget is None and mode_info.get("append_budget") is not None:
                    append_budget = int(mode_info["append_budget"])
                    append_budget_requested = int(append_budget)
                if hvg_mode == "fine" and int(n_final) < 2000 and int(n_final) < int(adata.n_vars):
                    # Respect user n_top_max (do not force 2000 past their ceiling).
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
                    "falling back to ensemble path for n_top selection."
                )
                if strict:
                    raise RuntimeError(
                        "structure auto_n failed and strict=True; not falling back to ensemble. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)
                structure_hybrid_fast = False
                auto_meta = {
                    "structure_error": str(exc),
                    "structure_error_type": type(exc).__name__,
                    "fallback_reason": "structure_auto_n_failed→ensemble",
                    "strategy": "ensemble",  # may be overwritten by ensemble path
                }
            else:
                n_final = int(n_final)
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
                        show,
                        "structure auto_n → base k=%d; append +%d secondary HVG...",
                        n_final,
                        append_budget,
                    )
                    selected, append_meta = _hvg_base_plus_append(
                        global_scores,
                        n_base=n_final,
                        n_append=append_budget,
                    )
                    aggregated = global_scores
                    selection_tag = "hvg_base_plus_secondary_append"
                    score_type = "global_variability_plus_append"
                    auto_meta = {
                        "strategy": "structure",
                        "n_top_selected": n_final,
                        "k_source": structure_detail.get("k_source"),
                        "rule_branch": structure_detail.get("rule_branch"),
                        "method_picks": {"structure": n_final},
                        "structure": structure_detail,
                        "pool_realign": "append_secondary_hvg",
                        "n_top_after_realign": int(len(selected)),
                        "structure_hybrid_fast": True,
                        "append": append_meta,
                    }
                    n_top_base = int(n_final)
                    # Filters/markers keep the full base+append list.
                    n_top_for_markers = int(len(selected))
                else:
                    _progress(
                        show,
                        "structure auto_n → k=%d; running hybrid once at that k...",
                        n_final,
                    )
                    selected, cluster_labels, n_clusters_used, cluster_weights, aggregated = (
                        _score_weighted_select(
                            adata,
                            n_top_genes=n_final,
                            resolution=resolution,
                            min_cluster_size=min_cluster_size,
                            n_pcs=n_pcs,
                            n_neighbors=n_neighbors,
                            random_state=random_state,
                            balance_power=beta,
                            global_rank=global_rank,
                            global_scores=global_scores,
                            blend_global=blend_global,
                            cluster_mask=cluster_mask,
                            progress=show,
                            scale_clustering=scale_clustering,
                            logfc_space=logfc_space,
                            hybrid=True,
                            neighbor_contrast=neighbor_contrast,
                            combine=combine,
                            diag_out=clustering_diag,
                            consensus_resolutions=consensus_resolutions,
                            allocation_method=allocation_method,
                            cap_ceiling=cap_ceiling,
                            cap_merge_threshold=cap_merge_threshold,
                            spec_on_legitimate_units=spec_on_legitimate_units,
                            graph_cache=intermediate_cache,
                            **hvg_params,
                        )
                    )
                    selection_tag = "hybrid_global_anchor_specificity"
                    score_type = f"hybrid_blend_global={blend_global}"
                    auto_meta = {
                        "strategy": "structure",
                        "n_top_selected": n_final,
                        "k_source": structure_detail.get("k_source"),
                        "rule_branch": structure_detail.get("rule_branch"),
                        "method_picks": {"structure": n_final},
                        "structure": structure_detail,
                        "pool_realign": "hybrid_2xk",
                        "n_top_after_realign": int(len(selected)),
                        "structure_hybrid_fast": True,
                    }
                    n_top_for_markers = n_final

        if not structure_hybrid_fast:
            if method == "none":
                selected = _top_genes_from_rank(global_rank, n_top_request)
                aggregated = global_scores
            elif method == "append":
                # Fixed k: freeze base + append immediately.
                # Auto (non-structure): rank the full pool, cut k later, then
                # re-apply append so final = k + m (not truncated to k).
                aggregated = global_scores
                selection_tag = "hvg_base_plus_secondary_append"
                score_type = "global_variability_plus_append"
                if n_top_is_auto:
                    selected = [
                        str(g)
                        for g in global_scores.sort_values(ascending=False, kind="stable").index
                    ]
                else:
                    selected, append_meta = _hvg_base_plus_append(
                        global_scores,
                        n_base=n_top_request,
                        n_append=append_budget,
                    )
            elif method in ("score", "hybrid"):
                selected, cluster_labels, n_clusters_used, cluster_weights, aggregated = (
                    _score_weighted_select(
                        adata,
                        n_top_genes=n_top_request,
                        resolution=resolution,
                        min_cluster_size=min_cluster_size,
                        n_pcs=n_pcs,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                        balance_power=beta,
                        global_rank=global_rank,
                        global_scores=global_scores,
                        blend_global=blend_global if method == "hybrid" else 0.0,
                        cluster_mask=cluster_mask,
                        progress=show,
                        scale_clustering=scale_clustering,
                        logfc_space=logfc_space,
                        hybrid=(method == "hybrid"),
                        neighbor_contrast=neighbor_contrast,
                        combine=combine,
                        diag_out=clustering_diag,
                        consensus_resolutions=consensus_resolutions,
                        allocation_method=allocation_method,
                        cap_ceiling=cap_ceiling,
                        cap_merge_threshold=cap_merge_threshold,
                        spec_on_legitimate_units=spec_on_legitimate_units,
                        graph_cache=intermediate_cache if n_top_is_auto else None,
                        **hvg_params,
                    )
                )
                if method == "hybrid":
                    selection_tag = "hybrid_global_anchor_specificity"
                    score_type = f"hybrid_blend_global={blend_global}"
                else:
                    selection_tag = "cluster_vs_rest_specificity"
                    score_type = "weighted_one_sided_logfc"
                # per-cluster ranks for coverage auto-n
                if n_top_is_auto and cluster_labels is not None:
                    cluster_gene_ranks = _build_cluster_gene_ranks(
                        adata,
                        cluster_labels=cluster_labels,
                        counts_layer=counts_layer,
                        min_cluster_size=min_cluster_size,
                        logfc_space=logfc_space,
                    )
            else:  # reweight
                selected, cluster_labels, n_clusters_used, cluster_weights, aggregated = (
                    _cell_reweight_select(
                        adata,
                        n_top_genes=n_top_request,
                        resolution=resolution,
                        min_cluster_size=min_cluster_size,
                        cluster_mask=cluster_mask,
                        progress=show,
                        scale_clustering=scale_clustering,
                        logfc_space=logfc_space,
                        n_pcs=n_pcs,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                        balance_power=beta,
                        global_rank=global_rank,
                        batch_key=batch_key,
                        diag_out=clustering_diag,
                        **hvg_params,
                    )
                )
                selection_tag = "cell_reweighted_global_hvg"
                score_type = f"reweighted_{flavor}"

            # --- Auto n_top: cut ranked list ---
            if n_top_is_auto:
                # Order by aggregated score when available, else selection order
                if aggregated is not None:
                    gene_order = list(
                        aggregated.reindex(selected)
                        .fillna(-np.inf)
                        .sort_values(ascending=False)
                        .index.astype(str)
                    )
                    # append any selected missing from aggregated
                    seen = set(gene_order)
                    for g in selected:
                        gs = str(g)
                        if gs not in seen:
                            gene_order.append(gs)
                    scores_desc = aggregated.reindex(gene_order).fillna(0.0).to_numpy(dtype=float)
                else:
                    gene_order = [str(g) for g in selected]
                    scores_desc = np.arange(len(gene_order), 0, -1, dtype=float)

                # Map auto aliases
                n_req: int | str = n_top_genes  # type: ignore[assignment]
                if str(n_top_genes).lower() == "auto":
                    n_req = auto_n_method

                # Silhouette is expensive and on PBMC alone picks k too small (hurts DC).
                # Only run it when explicitly requested — not for default ensemble.
                n_final, auto_meta = resolve_n_top_genes(
                    n_req,
                    scores_desc,
                    gene_order,
                    k_min=int(n_top_min),
                    k_max=min(int(n_top_max), len(gene_order)),
                    # adata: silhouette strategy + depth-aware ensemble knobs
                    adata=adata,
                    counts_layer=counts_layer,
                    cluster_gene_ranks=cluster_gene_ranks,
                    random_state=random_state,
                    depth_aware=True,
                )
                # Align gene set with fixed hybrid@k: re-rank inside global top 2×k
                # pool (k-sweep: auto on a 5k pool then cut ≠ hybrid@k). Skip for
                # score/reweight/none and for pure geometric strategies without hybrid.
                # Intermediate graph/specificity reused via intermediate_cache when
                # the clustering protocol is unchanged.
                if method == "hybrid" and n_final >= 1:
                    selected, cluster_labels, n_clusters_used, cluster_weights, aggregated = (
                        _score_weighted_select(
                            adata,
                            n_top_genes=n_final,
                            resolution=resolution,
                            min_cluster_size=min_cluster_size,
                            n_pcs=n_pcs,
                            n_neighbors=n_neighbors,
                            random_state=random_state,
                            balance_power=beta,
                            global_rank=global_rank,
                            global_scores=global_scores,
                            blend_global=blend_global,
                            cluster_mask=cluster_mask,
                            progress=show,
                            scale_clustering=scale_clustering,
                            logfc_space=logfc_space,
                            hybrid=True,
                            neighbor_contrast=neighbor_contrast,
                            combine=combine,
                            diag_out=clustering_diag,
                            consensus_resolutions=consensus_resolutions,
                            allocation_method=allocation_method,
                            cap_ceiling=cap_ceiling,
                            cap_merge_threshold=cap_merge_threshold,
                            spec_on_legitimate_units=spec_on_legitimate_units,
                            graph_cache=intermediate_cache,
                            **hvg_params,
                        )
                    )
                    if auto_meta is not None:
                        auto_meta = dict(auto_meta)
                        auto_meta["pool_realign"] = "hybrid_2xk"
                        auto_meta["n_top_after_realign"] = int(len(selected))
                        if clustering_diag.get("intermediate_reused"):
                            auto_meta["intermediate_reused"] = True
                            auto_meta["intermediate_reuse"] = clustering_diag.get(
                                "intermediate_reuse"
                            )
                    n_top_for_markers = n_final
                elif method == "append" and n_final >= 1:
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
                    selected, append_meta = _hvg_base_plus_append(
                        global_scores,
                        n_base=n_final,
                        n_append=append_budget,
                    )
                    if auto_meta is not None:
                        auto_meta = dict(auto_meta)
                        auto_meta["pool_realign"] = "append_secondary_hvg"
                        auto_meta["n_top_after_realign"] = int(len(selected))
                        auto_meta["append"] = append_meta
                    n_top_base = int(n_final)
                    # Keep full base+append for filters/markers (do not clip to k).
                    n_top_for_markers = int(len(selected))
                else:
                    selected = gene_order[:n_final]
                    n_top_for_markers = n_final
            else:
                if method == "append":
                    # selected already base+append; never truncate to n_top alone.
                    n_top_base = int(n_top_request)
                    n_top_for_markers = int(len(selected))
                else:
                    n_top_for_markers = n_top_request
                    selected = selected[:n_top_for_markers]

        # Diagnostic, captured before any marker handling touches the selection: how
        # many of the supplied markers the algorithm chose on its own. This is the
        # number that says whether injection is worth doing at all -- forcing in
        # markers the algorithm already rejected is what can hurt cluster purity
        # (see the marker_genes docstring warning). A high count means injection
        # is redundant.
        n_markers_already_selected = (
            None
            if marker_genes is None
            else int(len({str(g) for g in marker_genes} & {str(g) for g in selected}))
        )

        selected = _apply_gene_filters(
            selected,
            adata.var_names,
            filter_mito=filter_mito,
            filter_ribo=filter_ribo,
            marker_genes=marker_genes if marker_mode == "force" else None,
            # A true all-gene rank. global_rank comes from scanpy's
            # highly_variable_rank (NaN outside the top-n_top; internal helpers
            # treat those as +inf), so refilling from it would pick replacement
            # genes in var_names order rather than by variability — the same
            # defect fixed in _hybrid_anchor_select.
            fill_rank=global_scores.rank(ascending=False, method="first"),
            n_top_genes=n_top_for_markers,
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
        # Full resolved research/secondary bag for reproducibility (batch_key,
        # seurat mean/disp bounds, n_top_min/max, …). global_score is not
        # serialisable as a Series — record presence only.
        from dataclasses import asdict as _asdict

        from .._version import __version__ as _scfair_version

        _opt_rec = _asdict(opt)
        if _opt_rec.get("global_score") is not None:
            _opt_rec["global_score"] = "injected"
        if _opt_rec.get("cluster_genes") is not None:
            _opt_rec["cluster_genes"] = [str(g) for g in _opt_rec["cluster_genes"]]
        if _opt_rec.get("consensus_resolutions") is not None:
            _opt_rec["consensus_resolutions"] = [
                float(r) for r in _opt_rec["consensus_resolutions"]
            ]
        if _opt_rec.get("max_disp") is not None and not np.isfinite(float(_opt_rec["max_disp"])):
            _opt_rec["max_disp"] = "inf"

        meta: dict[str, Any] = {
            "scfair_version": _scfair_version,
            "flavor": hvg_run_meta.get("flavor_used", flavor),
            "flavor_requested": hvg_run_meta.get("flavor_requested", flavor),
            "flavor_used": hvg_run_meta.get("flavor_used", flavor),
            "fallback_reason": hvg_run_meta.get("fallback_reason"),
            "strict": bool(strict),
            "options": _opt_rec,
            "n_top_min": int(n_top_min),
            "n_top_max": int(n_top_max),
            "auto_n_method": str(auto_n_method) if n_top_is_auto else None,
            "batch_key": batch_key,
            "n_top_genes": n_top_genes,
            "balance_method": method,
            "mode": hvg_mode,
            "mode_info": {
                k: mode_info.get(k)
                for k in (
                    "mode",
                    "n_top_floor",
                    "append_budget",
                    "allow_short_soft_buffer",
                    "cluster_resolution",
                    "reasons",
                    "auto",
                    "note",
                )
                if k in mode_info
            },
            "downstream_clustering": {
                "tier": "fine" if hvg_mode == "fine" else "coarse",
                "resolution": float(mode_info.get("cluster_resolution") or 0.8),
                "auto": True,
                "from_hvg_mode": hvg_mode,
            },
            "balance_power": beta if method not in ("none", "append") else None,
            "blend_global": blend_global if method == "hybrid" else None,
            "neighbor_contrast": neighbor_contrast if method in ("hybrid", "score") else None,
            "combine": combine if method == "hybrid" else None,
            "append_budget": int(append_budget) if method == "append" else None,
            "append_budget_requested": (
                int(append_budget_requested) if method == "append" else None
            ),
            "append_budget_capped": bool(append_budget_capped) if method == "append" else None,
            "append": append_meta if method == "append" else None,
            "store_raw": effective_store_raw,
            "counts_integer_like": counts_validate_info.get("counts_integer_like"),
            "counts_warning": counts_validate_info.get("counts_warning"),
            "cluster_pool": cluster_pool if method not in ("none", "append") else None,
            "cluster_pool_source": (
                cluster_pool_source if method not in ("none", "append") else None
            ),
            "consensus_resolutions": (
                list(consensus_resolutions)
                if consensus_resolutions and method in ("hybrid", "score")
                else None
            ),
            "scale_clustering": (
                bool(scale_clustering) if method not in ("none", "append") else None
            ),
            "logfc_space": logfc_space if method in ("hybrid", "score") else None,
            # Requested allocation arm (hybrid only). Outcome status lives under
            # clustering["{arm}_allocation_status"] — never a second
            # allocation_method key there (request vs used was ambiguous).
            "allocation_method": allocation_method if method == "hybrid" else None,
            "allocation_status": (
                clustering_diag.get(f"{allocation_method}_allocation_status")
                if method == "hybrid"
                and allocation_method in ("coverage", "starved_topup")
                and clustering_diag
                else None
            ),
            "spec_on_legitimate_units": (
                bool(spec_on_legitimate_units) if method in ("hybrid", "score") else None
            ),
            "cap_allocation": False,  # removed 0.2.0; always off
            "cap_ceiling": None,
            "unit_merge_threshold": (
                cap_merge_threshold
                if method == "hybrid" and allocation_method in ("coverage", "starved_topup")
                else None
            ),
            # alias for older readers / scripts
            "cap_merge_threshold": (
                cap_merge_threshold
                if method == "hybrid" and allocation_method in ("coverage", "starved_topup")
                else None
            ),
            "global_score": "injected" if global_score is not None else None,
            # What the caller asked for, and what actually ran. With
            # resolution="auto" these differ, and the second is the one a reader
            # needs to reproduce the partition.
            "resolution": resolution if method not in ("none", "append") else None,
            "resolution_used": (
                clustering_diag.get("resolution") if method not in ("none", "append") else None
            ),
            "min_cluster_size": (min_cluster_size if method not in ("none", "append") else None),
            "n_clusters_used": n_clusters_used,
            "cluster_weights": cluster_weights,
            "n_pcs": n_pcs if method not in ("none", "append") else None,
            "n_neighbors": n_neighbors if method not in ("none", "append") else None,
            # Full record of the intermediate clustering, which the balanced
            # methods run and then discard. Exposed so a caller can reuse or
            # audit it instead of re-deriving PCA/neighbour/resolution settings by
            # hand -- see the `clustering` entry in the docstring.
            "clustering": clustering_diag or None,
            "counts_layer": counts_layer,
            "filter_mito": filter_mito,
            "filter_ribo": filter_ribo,
            "marker_mode": marker_mode if marker_genes else None,
            "marker_extra": marker_extra if marker_mode == "force" else None,
            "n_marker_genes": n_markers_present,
            "n_marker_genes_already_selected": n_markers_already_selected,
            "n_highly_variable_final": len(selected),
            "n_top_genes_request": n_top_genes if not n_top_is_auto else str(n_top_genes),
            # For append: report frozen base k (not base+budget). Final mask
            # size is n_highly_variable_final / append.n_final.
            "n_top_genes_used": (
                int(n_top_base)
                if method == "append" and n_top_base is not None
                else n_top_for_markers
            ),
            "auto_n": auto_meta,
            "random_state": random_state,
            "selection": selection_tag,
            "score_type": score_type,
            "scfair_score_note": (
                "For append: scfair_score is global variability (same ranking as "
                "the base + secondary list). For hybrid: scfair_score is the "
                "pool-normalized blend used for selection; genes outside the "
                "global top-(2×k) pool are NaN."
            ),
        }
        # Hybrid/score with zero usable clusters is pure global HVG. Rewrite
        # selection so method-comparison scripts do not count it as a valid
        # hybrid observation; always surface via warnings (not only diagnose).
        if method in ("hybrid", "score") and int(n_clusters_used) == 0:
            if method == "hybrid":
                selection_tag = "hybrid_degenerated_to_global"
                score_type = "global_variability_fallback"
            else:
                selection_tag = "score_degenerated_to_global"
                score_type = "global_variability_fallback"
            meta["selection"] = selection_tag
            meta["score_type"] = score_type
            meta["degenerated_to_global"] = True
            warnings.warn(
                f"balance_method={method!r} found no usable intermediate clusters "
                f"(n_clusters_used=0; often min_cluster_size too high). "
                "Cluster-vs-rest specificity has no signal → result ≈ scanpy "
                f"global HVG. Recorded selection={selection_tag!r} so this is "
                "not counted as a successful hybrid/score run. Lower "
                "min_cluster_size, or use balance_method='none'/'append'.",
                UserWarning,
                stacklevel=2,
            )
        # Auto (or large fixed k) selecting nearly all genes is a no-op HVG step.
        # Threshold 80%: structure rules that clip to n_vars on small matrices often
        # land at 100% with no signal that the rule failed to discriminate.
        # For append, judge on the frozen base k (not base+m which can fill a
        # small test matrix without implying identity base selection).
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
                "HVG selection is nearly identity — structure/auto rules may not have "
                "discriminated on this matrix size. Pass a smaller integer n_top_genes "
                "(default 2000) or lower n_top_max / options.n_top_max.",
                UserWarning,
                stacklevel=2,
            )
        # Final mask (base+append) covering >50% of genes on panels / CITE / spatial
        # is effectively a soft no-op filter — surface explicitly.
        n_sel_final = int(len(selected))
        if (
            method == "append"
            and adata.n_vars >= 10
            and n_sel_final > max(10, int(0.5 * adata.n_vars))
        ):
            warnings.warn(
                f"append selection kept {n_sel_final}/{adata.n_vars} genes "
                f"({100.0 * n_sel_final / adata.n_vars:.0f}% of the matrix). "
                "On small gene sets (panels, CITE-seq, spatial) absolute "
                "append_budget can dominate; use options=HVGOptions(append_budget=0) "
                "or a smaller budget relative to n_top_genes.",
                UserWarning,
                stacklevel=2,
            )
        if diagnose:
            auto_strat = None
            structure_meta = None
            if isinstance(auto_meta, dict):
                auto_strat = auto_meta.get("strategy")
                # Nested detail from estimate_n_top_structure (rule_branch, features).
                raw_st = auto_meta.get("structure")
                if isinstance(raw_st, dict):
                    structure_meta = raw_st
            # Diagnose on base k for append (not base+budget), else tips claim
            # "floored n_top=2200" when the resolved base was 2000 / 1000.
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
                # the resolved number, not the request: with resolution="auto" the
                # nc_low_resolution guard would otherwise never fire, since "auto"
                # is not comparable to 0.75 at the parameter-only stage.
                resolution=(
                    clustering_diag.get("resolution", resolution) if method != "none" else None
                ),
                neighbor_contrast=neighbor_contrast if method in ("hybrid", "score") else 0.0,
                min_cluster_size=min_cluster_size if method != "none" else None,
                clustering=clustering_diag or None,
                n_clusters_used=n_clusters_used,
                config_check=config_check,
                log=True,
            )
        # Closing line on the same channel as the stage announcements. Without it
        # `progress` reports three stages *starting* and never reports the call
        # finishing or what it produced — so a user watching stderr cannot tell a
        # completed run from one still inside the slow step. The equivalent summary
        # existed only on the logger, i.e. invisible to anyone who has not
        # configured logging.
        if method == "none":
            _progress(show, "done: %d genes (global HVG, no clustering).", len(selected))
        elif method == "append":
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
                "done: %d genes (global HVG base %d + %d secondary append; no clustering).",
                len(selected),
                n_base,
                n_extra,
            )
        else:
            res_used = clustering_diag.get("resolution")
            src = clustering_diag.get("resolution_source")
            kept = clustering_diag.get("n_clusters_kept")
            # `n_clusters_used` (produced a specificity score) is lower than
            # `n_clusters_kept` (passed min_cluster_size) whenever a kept cluster
            # spans every cell, so there is no "rest" to contrast it against. Show
            # both when they disagree — the gap is the thing worth noticing.
            scored = (
                f"{n_clusters_used} of {kept}"
                if isinstance(kept, int) and kept != n_clusters_used
                else f"{n_clusters_used}"
            )
            _progress(
                show,
                "done: %d genes, scored against %s intermediate clusters at resolution %s%s.",
                len(selected),
                scored,
                f"{res_used:.3g}" if isinstance(res_used, float) else res_used,
                {
                    "density_field": " (chosen from the density field)",
                    "fallback": " (density field unusable; fell back)",
                }.get(str(src), ""),
            )

        # Tips were computed above (diagnose_hvg_run) but only reached the caller
        # via adata.uns and Python's logging module -- invisible on this same
        # stderr channel unless logging happens to be configured. Show them here
        # too, right after the call reports done, so `progress=True` alone is
        # enough to see them.
        if diagnose and isinstance(meta.get("diagnosis"), dict):
            for tip in meta["diagnosis"].get("tips") or []:
                _progress(show, "tip: %s", tip)

        result = _apply_selection(
            adata,
            selected=selected,
            aggregated_score=aggregated,
            global_scores=global_scores,
            cluster_labels=cluster_labels,
            meta=meta,
            prior_scanpy_hvg=_prior_scanpy_hvg,
        )

        # Ephemeral internals: drop rather than leave matrices the size of .X
        # on the caller's object (they would also be written by write_h5ad()).
        adata.layers.pop("_scfair_log", None)
        adata.layers.pop(INTERNAL_COUNTS_LAYER, None)

        if subset:
            adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())

        if inplace:
            return None
        return result

    except Exception as _hvg_exc:
        _rollback_adata_after_failure(adata, _hvg_snap, _hvg_exc)
        raise


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
        "counts_warning": None,
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
            return info
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
            "PCA/neighbours may be unstable. Filter empty barcodes before HVG.",
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
    if not is_int:
        msg = (
            f"counts matrix {where} does not look like raw integer counts "
            "(values are non-integer or fractional). seurat_v3 / "
            "pearson_residuals expect UMI counts; log1p-normalized input "
            "yields unreliable HVGs. Restore raw counts before calling, "
            "or pass layer= pointing at a true counts matrix."
        )
        info["counts_warning"] = "non_integer_counts"
        if strict:
            raise ValueError(msg + " (strict=True)")
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    # seurat_v3 loess (skmisc) segfaults in C when n_vars is too small for
    # span — not a catchable Python exception. Record here; _run_hvg falls
    # back to seurat when strict=False, or raises when strict=True.
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        span_f = float(span) if span and float(span) > 0 else 0.3
        min_genes = max(10, int(np.ceil(2.0 / span_f)))
        n_vars = int(X.shape[1])
        if n_vars < min_genes:
            msg = (
                f"flavor={flavor!r} (loess span={span_f}) needs at least "
                f"{min_genes} genes to fit safely; got n_vars={n_vars}. "
                "Use more genes, or flavor='seurat' / a non-loess method."
            )
            info["counts_warning"] = "n_vars_too_small_for_loess"
            info["loess_min_genes"] = min_genes
            if strict:
                raise ValueError(msg + " (strict=True)")
            # Non-strict: _run_hvg will fall back to seurat before calling loess.
            warnings.warn(
                msg + " Will fall back to flavor='seurat' if HVG proceeds.",
                UserWarning,
                stacklevel=3,
            )

    return info


def _resolve_balance_power(balance_method: str | None, balance_power: float | None) -> float:
    """Size weight exponent: ``w_c ∝ n_c^β`` (normalized).

    ``β > 0`` up-weights large clusters; ``β = 0`` is equal weight; ``β < 0``
    up-weights small clusters (true inverse-size). Allowed range ``[-2, 2]``.
    """
    if balance_power is not None:
        p = float(balance_power)
        if p < -2.0 or p > 2.0:
            raise ValueError(
                f"balance_power (β) must be in [-2, 2], got {p}. "
                "β>0 up-weights large clusters; β=0 equal weight; β<0 up-weights small."
            )
        return p
    if balance_method == "inverse_size":
        # True inverse size-weighting (small clusters get higher mass).
        return -1.0
    return 0.5


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
    # Pre-empt loess C segfault: n_vars too small for span is not catchable.
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        span_f = float(span) if span and float(span) > 0 else 0.3
        min_genes = max(10, int(np.ceil(2.0 / span_f)))
        if int(adata.n_vars) < min_genes:
            reason = (
                f"flavor={flavor!r} (loess span={span_f}) needs ≥{min_genes} genes; "
                f"got n_vars={adata.n_vars}; falling back to flavor='seurat'."
            )
            if strict:
                raise ValueError(
                    f"flavor={flavor!r} (loess span={span_f}) needs ≥{min_genes} genes; "
                    f"got n_vars={adata.n_vars}. (strict=True)"
                )
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
            meta["fallback_reason"] = f"n_vars_too_small_for_loess:{adata.n_vars}<{min_genes}"
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
        hvg_exp(
            adata,
            flavor="pearson_residuals",
            n_top_genes=n_top_genes,
            layer=work_layer,
            inplace=True,
            subset=False,
            check_values=True,
        )
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

    log_layer = "_scfair_log"
    if counts_layer in adata.layers:
        ad_tmp = adata.copy()
        ad_tmp.X = ad_tmp.layers[counts_layer].copy()
        ad_tmp.uns.pop("log1p", None)
        sc.pp.normalize_total(ad_tmp, target_sum=1e4)
        sc.pp.log1p(ad_tmp)
        adata.layers[log_layer] = ad_tmp.X.copy()
        return log_layer

    if _is_integer_counts_like(adata.X):
        ad_tmp = adata.copy()
        ad_tmp.uns.pop("log1p", None)
        sc.pp.normalize_total(ad_tmp, target_sum=1e4)
        sc.pp.log1p(ad_tmp)
        adata.layers[log_layer] = ad_tmp.X.copy()
        return log_layer

    logger.debug("Using existing .X for log-based flavor=%s.", flavor)
    return None


def _variability_raw_scores(
    adata: Any,
    *,
    flavor: str | None = None,
    flavor_used: str | None = None,
) -> pd.Series:
    """All-gene variability scores for ranking / append / hybrid pool.

    When ``flavor`` / ``flavor_used`` is set, only that flavor's scanpy score
    column is read (no priority walk across leftover columns from a prior
    call). Prefer ``flavor_used`` when seurat_v3 fell back to seurat.
    """
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


def _unit_rank_scores(raw: pd.Series) -> pd.Series:
    r = raw.rank(ascending=True, method="average", na_option="bottom")
    denom = max(len(r) - 1, 1)
    return (r - 1.0) / denom


def _gene_rank_series(adata: Any, *, flavor: str | None = None) -> pd.Series:
    if "highly_variable_rank" in adata.var.columns:
        rank = pd.to_numeric(adata.var["highly_variable_rank"], errors="coerce")
    else:
        scores = _variability_raw_scores(adata, flavor=flavor)
        rank = scores.rank(ascending=False, method="average")
    return rank.reindex(adata.var_names).fillna(np.inf)


def _top_genes_from_rank(rank: pd.Series, n_top: int) -> list[str]:
    return list(rank.sort_values(ascending=True).index[:n_top])


def _top_genes_from_scores(scores: pd.Series, n_top: int) -> list[str]:
    return list(scores.sort_values(ascending=False).index[:n_top])


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


def _cluster_size_weights(sizes: pd.Series, power: float) -> pd.Series:
    """``w_c ∝ n_c^{power}``, normalized to sum 1.

    ``power > 0``: large clusters weigh more; ``0``: equal; ``< 0``: small
    clusters weigh more (inverse size).
    """
    n = sizes.to_numpy(dtype=float)
    n = np.maximum(n, 1.0)  # guard for empty / zero sizes
    if power == 0:
        w = np.ones_like(n, dtype=float)
    else:
        w = np.power(n, float(power))
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        w = np.ones_like(n, dtype=float)
        s = float(w.sum())
    w = w / s
    return pd.Series(w, index=sizes.index)


def _is_mito_name(name: str) -> bool:
    return bool(_MITO_RE.match(str(name)))


def _is_ribo_name(name: str) -> bool:
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
) -> list[str]:
    """Drop MT/ribo from selection (unless markers); refill from global rank."""
    if not filter_mito and not filter_ribo:
        return selected[:n_top_genes]

    protect = set(map(str, marker_genes or ()))
    kept: list[str] = []
    dropped = 0
    for g in selected:
        gs = str(g)
        if gs in protect:
            kept.append(gs)
            continue
        if filter_mito and _is_mito_name(gs):
            dropped += 1
            continue
        if filter_ribo and _is_ribo_name(gs):
            dropped += 1
            continue
        kept.append(gs)

    if dropped:
        logger.info(
            "Filtered %d mito/ribo genes from HVG set; refilling from global rank.",
            dropped,
        )
    if len(kept) >= n_top_genes:
        return kept[:n_top_genes]

    need = n_top_genes - len(kept)
    have = set(kept)
    for g in _top_genes_from_rank(fill_rank, len(fill_rank)):
        gs = str(g)
        if gs in have:
            continue
        if gs not in protect:
            if filter_mito and _is_mito_name(gs):
                continue
            if filter_ribo and _is_ribo_name(gs):
                continue
        kept.append(gs)
        have.add(gs)
        need -= 1
        if need <= 0:
            break
    return kept[:n_top_genes]


# ---------------------------------------------------------------------------
# Intermediate clustering
# ---------------------------------------------------------------------------


def _hvg_mask_signature(hvg_mask: np.ndarray) -> tuple[int, int, int]:
    """Compact fingerprint of a boolean gene mask (n_true, first, last true idx)."""
    m = np.asarray(hvg_mask, dtype=bool).ravel()
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return (0, -1, -1)
    return (int(idx.size), int(idx[0]), int(idx[-1]))


def _intermediate_protocol(
    *,
    hvg_mask: np.ndarray,
    counts_layer: str,
    resolution: float | str,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    scale_clustering: bool,
    resolutions: Sequence[float] | None,
    min_cluster_size: int,
    balance_power: float,
    logfc_space: str,
    neighbor_contrast: float,
    combine: str,
    hybrid: bool,
    blend_global: float,
    allocation_method: str,
    cap_ceiling: float,
    cap_merge_threshold: float | None,
    spec_on_legitimate_units: bool,
) -> dict[str, Any]:
    """Protocol fingerprint for intermediate graph / Leiden / specificity reuse."""
    res_ladder = None
    if resolutions is not None:
        res_ladder = tuple(float(r) for r in resolutions)
    return {
        "mask_sig": _hvg_mask_signature(hvg_mask),
        "counts_layer": str(counts_layer),
        "resolution": (str(resolution) if isinstance(resolution, str) else float(resolution)),
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "random_state": int(random_state),
        "scale_clustering": bool(scale_clustering),
        "resolutions": res_ladder,
        "min_cluster_size": int(min_cluster_size),
        "balance_power": float(balance_power),
        "logfc_space": str(logfc_space),
        "neighbor_contrast": float(neighbor_contrast),
        "combine": str(combine),
        "hybrid": bool(hybrid),
        # blend / allocation affect only the final cut, not the graph/specificity
        # scores — still recorded for diagnostics.
        "blend_global": float(blend_global),
        "allocation_method": str(allocation_method),
        "cap_ceiling": float(cap_ceiling),
        "cap_merge_threshold": (
            None if cap_merge_threshold is None else float(cap_merge_threshold)
        ),
        "spec_on_legitimate_units": bool(spec_on_legitimate_units),
    }


def _graph_protocol_key(proto: Mapping[str, Any]) -> tuple[Any, ...]:
    """Keys that determine PCA + neighbours (+ Leiden resolution ladder)."""
    return (
        proto["mask_sig"],
        proto["counts_layer"],
        proto["resolution"],
        proto["n_pcs"],
        proto["n_neighbors"],
        proto["random_state"],
        proto["scale_clustering"],
        proto["resolutions"],
    )


def _score_protocol_key(proto: Mapping[str, Any]) -> tuple[Any, ...]:
    """Keys that determine cluster-vs-rest specificity scores."""
    return _graph_protocol_key(proto) + (
        proto["min_cluster_size"],
        proto["balance_power"],
        proto["logfc_space"],
        proto["neighbor_contrast"],
        proto["combine"],
        proto["spec_on_legitimate_units"],
        proto["cap_merge_threshold"],  # units / merge threshold for spec partition
    )


def _cluster_on_hvgs(
    adata: Any,
    *,
    hvg_mask: np.ndarray,
    counts_layer: str,
    resolution: float | str,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    scale_clustering: bool = True,
    diag_out: dict[str, Any] | None = None,
    resolutions: Sequence[float] | None = None,
    progress: bool = False,
) -> tuple[list[pd.Series], np.ndarray] | tuple[None, None]:
    ad_hvg = adata[:, hvg_mask].copy()
    if counts_layer in ad_hvg.layers:
        ad_hvg.X = ad_hvg.layers[counts_layer].copy()
    else:
        restored = _restore_raw_counts(ad_hvg, layer=counts_layer, inplace=False)
        ad_hvg.X = restored.X

    # `ad_hvg` IS the gene space we want to cluster on. It inherits
    # var["highly_variable"] from the parent, and sc.pp.pca defaults to masking by
    # that column (mask_var default is _empty = "use highly_variable if present"),
    # which would silently shrink the clustering space to the parent's top-k. That
    # is a no-op when the subset equals the mask, but wrong whenever it does not —
    # e.g. cluster_pool, where it both shrank the space and desynchronised
    # n_pcs_use from the matrix scanpy actually decomposed. Dropping the column is
    # version-proof (no mask_var / use_highly_variable branch needed).
    ad_hvg.var = ad_hvg.var.drop(
        columns=[c for c in ("highly_variable", "highly_variable_rank") if c in ad_hvg.var]
    )

    ad_hvg.uns.pop("log1p", None)
    sc.pp.normalize_total(ad_hvg, target_sum=1e4)
    sc.pp.log1p(ad_hvg)

    if scale_clustering:
        # Default True: zero-centre before PCA (standard scanpy). Without this,
        # high-expression genes dominate intermediate structure and multi-type
        # partitions often collapse to 2 communities.
        sc.pp.scale(ad_hvg, max_value=10)

    n_pcs_use = min(n_pcs, ad_hvg.n_vars - 1, ad_hvg.n_obs - 1)
    if n_pcs_use < 2:
        logger.warning("Not enough PCs for clustering; skipping balanced path.")
        return None, None

    _progress(progress, "  step 1/3: PCA (%d comps)...", n_pcs_use)
    sc.pp.pca(ad_hvg, n_comps=n_pcs_use, random_state=random_state)
    n_neighbors_use = min(n_neighbors, ad_hvg.n_obs - 1)
    _progress(progress, "  step 2/3: neighbour graph (n_neighbors=%d)...", n_neighbors_use)
    sc.pp.neighbors(
        ad_hvg,
        n_neighbors=n_neighbors_use,
        n_pcs=n_pcs_use,
        random_state=random_state,
    )
    _progress(
        progress,
        "  step 3/3: Leiden clustering%s...",
        " (resolution='auto' first searches for a matching resolution)"
        if resolution == "auto"
        else "",
    )
    # `resolution="auto"`: derive the target count from the density field of a 3D
    # embedding, then find the resolution that produces it. Done here, on the
    # graph that was just built, because the PCA and the neighbour search are
    # ~90% of the runtime and both the embedding and every Leiden probe reuse
    # them -- resolving it any higher up would mean building a second graph.
    granularity_diag: dict[str, Any] = {}
    if isinstance(resolution, str):
        if resolution != "auto":
            raise ValueError(f"resolution must be a number or 'auto', got {resolution!r}.")
        resolution, granularity_diag = resolution_from_density_field(
            ad_hvg,
            fallback=DEFAULT_RESOLUTION_FALLBACK,
            random_state=random_state,
        )

    if diag_out is not None:
        # Every number here is one the caller cannot recover afterwards: n_pcs_use
        # and n_neighbors_use are clamped copies of the request, and the PCA is
        # discarded with `ad_hvg`. Users re-doing PCA -> neighbours -> Leiden
        # downstream were otherwise guessing at settings we already resolved.
        var_ratio = np.asarray(ad_hvg.uns.get("pca", {}).get("variance_ratio", []), dtype=float)
        diag_out.update(
            {
                "n_passes": int(diag_out.get("n_passes", 0)) + 1,
                "n_genes_clustered": int(ad_hvg.n_vars),
                "n_cells": int(ad_hvg.n_obs),
                "n_pcs_requested": int(n_pcs),
                "n_pcs_used": int(n_pcs_use),
                "n_neighbors_requested": int(n_neighbors),
                "n_neighbors_used": int(n_neighbors_use),
                "resolution": float(resolution),
                **granularity_diag,
                "leiden_flavor": "igraph",
                "leiden_n_iterations": 2,
                "scale_clustering": bool(scale_clustering),
                "normalization": "normalize_total(1e4) -> log1p"
                + (" -> scale(max_value=10)" if scale_clustering else ""),
                "random_state": int(random_state),
                "pca_variance_ratio": var_ratio.tolist(),
                "pca_variance_ratio_total": float(var_ratio.sum()) if var_ratio.size else None,
            }
        )
    # One graph, one Leiden per resolution. The PCA and the neighbour search are
    # the expensive parts and they are shared, so a 3-resolution ladder costs
    # far less than 3x -- the same trick the evaluation sweeps use.
    # The primary resolution always leads the ladder, and is added if the caller
    # left it out. `partitions[0]` is what lands in obs and what the diagnostics
    # describe, so that slot has to mean the same thing with or without a ladder.
    if resolutions is None:
        res_list = [float(resolution)]
    else:
        res_list = [float(resolution)] + [
            float(r) for r in resolutions if not np.isclose(float(r), float(resolution))
        ]
    out: list[pd.Series] = []
    for res in res_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sc.tl.leiden(
                ad_hvg,
                resolution=res,
                random_state=random_state,
                key_added="scfair_hvg_clusters",
                flavor="igraph",
                n_iterations=2,
            )
        labels = ad_hvg.obs["scfair_hvg_clusters"].astype(str).copy()
        labels.index = ad_hvg.obs_names
        out.append(labels.reindex(adata.obs_names))
    if diag_out is not None and len(res_list) > 1:
        diag_out["consensus_resolutions"] = res_list
        diag_out["consensus_n_clusters"] = [int(s.nunique()) for s in out]
    # X_pca is shared across the whole resolution ladder (one graph, one PCA;
    # see the comment above) -- indexed by ad_hvg.obs_names, which is
    # adata.obs_names in its original order (subsetting genes doesn't
    # reorder cells), so it aligns directly with any Series in `out`.
    X_pca = np.asarray(ad_hvg.obsm["X_pca"], dtype=float)
    return out, X_pca


def _prepare_clusters(
    adata: Any,
    *,
    counts_layer: str,
    resolution: float | str,
    min_cluster_size: int,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    balance_power: float,
    global_rank: pd.Series,
    n_top_genes: int,
    cluster_mask: np.ndarray | None = None,
    progress: bool = False,
    scale_clustering: bool = True,
    logfc_space: str = "log1p",
    diag_out: dict[str, Any] | None = None,
    resolutions: Sequence[float] | None = None,
    extra_out: list[tuple[pd.Series, pd.Series]] | None = None,
    graph_cache: dict[str, Any] | None = None,
) -> (
    tuple[pd.Series, pd.Series, dict[str, float], np.ndarray]
    | tuple[pd.Series, None, dict, None]
    | tuple[None, None, dict, None]
):
    """Return (labels, valid_sizes, weight_map, X_pca) or (None, None, {}, None).

    ``X_pca`` is the PCA embedding the intermediate clustering itself ran
    on (shared across any resolution ladder), returned so callers that need
    it (e.g. ``cap_allocation``'s cluster-legitimacy merge) don't pay for a
    second PCA -- it is ~90% of this function's runtime.

    ``cluster_pool`` decouples the gene space the *intermediate clustering* runs
    on from the number of genes being selected. ``None`` (default) keeps the
    historical behaviour: cluster on the current ``highly_variable`` mask, whose
    size therefore tracks ``n_top_genes``. ``auto`` does this decoupling
    implicitly by clustering its ``n_top_max`` pool.

    ``graph_cache`` (optional mutable dict): when a prior call stored
    partitions under a matching graph protocol, Leiden + PCA are reused
    (auto realign). On a miss, this call fills the cache.
    """
    if cluster_mask is not None:
        hvg_mask = np.asarray(cluster_mask, dtype=bool)
    else:
        hvg_mask = adata.var["highly_variable"].to_numpy()
    if int(hvg_mask.sum()) < 2:
        logger.warning("Global HVG selected <2 genes; cannot cluster.")
        return None, None, {}, None

    graph_proto = {
        "mask_sig": _hvg_mask_signature(hvg_mask),
        "counts_layer": str(counts_layer),
        "resolution": (str(resolution) if isinstance(resolution, str) else float(resolution)),
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "random_state": int(random_state),
        "scale_clustering": bool(scale_clustering),
        "resolutions": (None if resolutions is None else tuple(float(r) for r in resolutions)),
    }
    graph_key = (
        graph_proto["mask_sig"],
        graph_proto["counts_layer"],
        graph_proto["resolution"],
        graph_proto["n_pcs"],
        graph_proto["n_neighbors"],
        graph_proto["random_state"],
        graph_proto["scale_clustering"],
        graph_proto["resolutions"],
    )

    reused = False
    if (
        graph_cache is not None
        and graph_cache.get("graph_key") == graph_key
        and graph_cache.get("partitions") is not None
        and graph_cache.get("X_pca") is not None
    ):
        partitions = list(graph_cache["partitions"])
        X_pca = np.asarray(graph_cache["X_pca"], dtype=float)
        reused = True
        if diag_out is not None:
            # Preserve prior clustering numbers; mark reuse so n_passes is honest.
            diag_out["intermediate_reused"] = True
            diag_out["intermediate_reuse"] = "leiden_pca"
            # Do not increment n_passes — physical clustering did not re-run.
            for k in (
                "n_genes_clustered",
                "n_cells",
                "n_pcs_requested",
                "n_pcs_used",
                "n_neighbors_requested",
                "n_neighbors_used",
                "resolution",
                "leiden_flavor",
                "leiden_n_iterations",
                "scale_clustering",
                "normalization",
                "random_state",
                "pca_variance_ratio",
                "pca_variance_ratio_total",
                "consensus_resolutions",
                "consensus_n_clusters",
            ):
                if k in graph_cache.get("diag_snapshot", {}) and k not in diag_out:
                    diag_out[k] = graph_cache["diag_snapshot"][k]
        _progress(
            progress,
            "reusing intermediate clustering (PCA + Leiden) from prior pass "
            "(%d genes x %d cells)...",
            int(hvg_mask.sum()),
            adata.n_obs,
        )
    else:
        _progress(
            progress,
            "intermediate clustering (the slow step): PCA -> neighbours -> "
            "Leiden(resolution=%s) on %d genes x %d cells... please wait.",
            # "auto" is still unresolved at this point -- it is derived from the
            # density field once the neighbour graph exists, inside _cluster_on_hvgs.
            f"{resolution:.2f}" if not isinstance(resolution, str) else resolution,
            int(hvg_mask.sum()),
            adata.n_obs,
        )
        partitions, X_pca = _cluster_on_hvgs(
            adata,
            hvg_mask=hvg_mask,
            scale_clustering=scale_clustering,
            counts_layer=counts_layer,
            resolution=resolution,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            random_state=random_state,
            diag_out=diag_out,
            resolutions=resolutions,
            progress=progress,
        )
        if partitions is None:
            return None, None, {}, None
        if graph_cache is not None:
            snap = {}
            if diag_out is not None:
                for k in (
                    "n_genes_clustered",
                    "n_cells",
                    "n_pcs_requested",
                    "n_pcs_used",
                    "n_neighbors_requested",
                    "n_neighbors_used",
                    "resolution",
                    "leiden_flavor",
                    "leiden_n_iterations",
                    "scale_clustering",
                    "normalization",
                    "random_state",
                    "pca_variance_ratio",
                    "pca_variance_ratio_total",
                    "consensus_resolutions",
                    "consensus_n_clusters",
                    "n_populations_density",
                    "resolution_source",
                ):
                    if k in diag_out:
                        snap[k] = diag_out[k]
            graph_cache.clear()
            graph_cache.update(
                {
                    "graph_key": graph_key,
                    "graph_proto": graph_proto,
                    "partitions": list(partitions),
                    "X_pca": np.asarray(X_pca, dtype=float),
                    "diag_snapshot": snap,
                }
            )

    # `cluster_labels` stays the partition at the primary `resolution` -- it is
    # what lands in obs and what the diagnostics describe, so the single-
    # resolution path is unchanged. Extra ladder partitions ride alongside.
    cluster_labels = partitions[0]
    extra = partitions[1:]

    sizes = cluster_labels.value_counts(dropna=True)
    valid = sizes[sizes >= min_cluster_size]
    if diag_out is not None:
        # `n_clusters_kept` counts communities passing min_cluster_size; the
        # top-level `n_clusters_used` counts those that produced a specificity
        # score, and is lower whenever a kept cluster spans every cell (no "rest"
        # to contrast against). Keeping both names distinct because the gap
        # between them is itself the signal.
        #
        # The totals matter for the same reason: a run with 15 Leiden communities
        # of which 6 fall under min_cluster_size scores on 9, and the dropped ones
        # are exactly the rare populations the balancing is meant to protect.
        n_cells = int(sizes.sum()) if len(sizes) else 0
        min_size = int(sizes.min()) if len(sizes) else 0
        min_frac = float(min_size / n_cells) if n_cells > 0 else None
        diag_out.update(
            {
                "min_cluster_size": int(min_cluster_size),
                "n_clusters_total": int(len(sizes)),
                "n_clusters_kept": int(len(valid)),
                "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
                "clusters_dropped": [str(k) for k, v in sizes.items() if v < min_cluster_size],
                # Smallest community share in the intermediate partition. A
                # coarse partition (few large blobs) often has min_frac ≫ true
                # rare type frequency — rare cells were absorbed, not dropped.
                "min_cluster_frac": min_frac,
                "min_cluster_n": min_size if len(sizes) else None,
            }
        )
        if reused:
            diag_out["intermediate_reused"] = True
    if valid.empty:
        logger.warning(
            "No clusters with size >= min_cluster_size=%d.",
            min_cluster_size,
        )
        return cluster_labels, None, {}, None
    if len(valid) < 2:
        # One cluster means "cluster vs rest" has no rest: every specificity score
        # collapses and the balanced layer degenerates to plain global HVG. Say so,
        # or the caller believes they used scFair when they used scanpy.
        logger.warning(
            "Only %d intermediate cluster passes min_cluster_size=%d, so "
            "cluster-vs-rest specificity carries no signal and the result will be "
            "≈ plain global HVG. Lower min_cluster_size, raise resolution, or use "
            "balance_method='none' deliberately.",
            len(valid),
            min_cluster_size,
        )

    weights = _cluster_size_weights(valid, balance_power)
    weight_map = {str(k): float(v) for k, v in weights.items()}
    if extra_out is not None:
        # Prefer cached extra partitions when present
        if reused and graph_cache is not None and graph_cache.get("extra_partitions"):
            for item in graph_cache["extra_partitions"]:
                extra_out.append(item)
        else:
            for labels in extra:
                s = labels.value_counts(dropna=True)
                v = s[s >= min_cluster_size]
                if len(v) >= 2:
                    extra_out.append((labels, v))
            if graph_cache is not None and extra_out:
                graph_cache["extra_partitions"] = list(extra_out)
    return cluster_labels, valid, weight_map, X_pca


# ---------------------------------------------------------------------------
# score: cluster-vs-rest specificity
# ---------------------------------------------------------------------------


def _lognorm_matrix_from_counts(adata: Any, counts_layer: str) -> Any:
    ad_tmp = adata.copy()
    if counts_layer in ad_tmp.layers:
        ad_tmp.X = ad_tmp.layers[counts_layer].copy()
    ad_tmp.uns.pop("log1p", None)
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)
    X = ad_tmp.X
    if sparse.issparse(X):
        return X.tocsr()
    return np.asarray(X, dtype=float)


def _mean_axis0(X: Any, row_mask: np.ndarray) -> np.ndarray:
    n = int(row_mask.sum())
    if n == 0:
        raise ValueError("Empty row mask for mean.")
    if sparse.issparse(X):
        return np.asarray(X[row_mask].mean(axis=0)).ravel()
    return np.asarray(X[row_mask], dtype=float).mean(axis=0)


_LOGFC_SPACES = ("log1p", "linear", "linear_regularised")

# Pseudocounts per space. log1p keeps the shipped 1e-2.
# "linear" reproduces scanpy rank_genes_groups (1e-9) on normalize_total(1e4)
# scale — essentially unregularised; retained as **benchmark-only** (emits
# UserWarning at entry). Prefer "linear_regularised" (pseudo=1.0, one count
# per 10k) for any non-benchmark linear-space experiment.
_LOGFC_PSEUDO = {"log1p": 1e-2, "linear": 1e-9, "linear_regularised": 1.0}


def _logfc_inputs(X_log: Any, space: str) -> tuple[Any, float]:
    """Return (matrix, pseudo) for the requested fold-change space.

    ``X_log`` is always ``log1p(normalize_total(counts))``; the linear spaces
    back-transform it with ``expm1`` first so the mean is taken in linear space,
    which is the conventional definition of a fold change.
    """
    if space == "log1p":
        return X_log, _LOGFC_PSEUDO[space]
    X = X_log.copy()
    if sparse.issparse(X):
        X.data = np.expm1(X.data)
    else:
        X = np.expm1(X)
    return X, _LOGFC_PSEUDO[space]


def _cluster_vs_rest_logfc(
    X: Any,
    in_cluster: np.ndarray,
    *,
    pseudo: float = 1e-2,
    one_sided: bool = True,
    out_mask: np.ndarray | None = None,
) -> np.ndarray:
    """One-sided contrast of ``in_cluster`` against a reference set.

    ``out_mask=None`` uses the complement (classic cluster-vs-rest). Passing an
    explicit mask scores against a chosen reference — used by the nearest-
    neighbour contrast, where "rest" is a single adjacent cluster.

    .. note::

       **This is not a standard log fold change, despite the name.** ``X`` here is
       already ``log1p(normalize_total(counts))``, and the statistic computed is

           ``log2( (mean_i log1p(x) + pseudo) / (mean_o log1p(x) + pseudo) )``

       i.e. a log-ratio of **mean log-expression**. The conventional definition
       (e.g. scanpy's ``rank_genes_groups``) back-transforms with ``expm1`` first,
       averages in linear space, and only then takes the ratio; by Jensen's
       inequality the two are not equal. Consequences: the compression applied to
       highly- versus lowly-expressed genes differs from a true fold change, and
       ``pseudo=1e-2`` is a far weaker regulariser on a log1p scale (values ~0-8)
       than the same constant would be on linear counts. The choice is not
       cosmetic — it materially changes which genes rank highest.

       It is nonetheless **not** obviously a defect to be corrected. The genes the
       conventional statistic favours tend to have near-zero expression in
       almost every cell, because in linear space ``mu_out`` can approach zero and
       a small linear epsilon barely regularises it, so the ratio explodes on
       ultra-sparse genes. On the log1p scale ``mu_out`` has a floor, which damps
       exactly that blow-up. The deviation may therefore be better behaved for
       feature selection than the definition it deviates from.

       Kept as-is: there is no evidence the conventional definition is
       preferable, and a fair test would need a third arm (linear space with a
       moderate pseudocount) to separate "log vs linear" from "pseudocount
       scale".
    """
    if out_mask is None:
        out_mask = ~in_cluster
    if not out_mask.any():
        return np.zeros(X.shape[1], dtype=float)
    mu_in = _mean_axis0(X, in_cluster)
    mu_out = _mean_axis0(X, out_mask)
    logfc = np.log2((mu_in + pseudo) / (mu_out + pseudo))
    if one_sided:
        logfc = np.maximum(logfc, 0.0)
    return logfc.astype(float, copy=False)


def _nearest_cluster_map(X: Any, masks: dict[str, np.ndarray]) -> dict[str, str]:
    """Map each cluster to its closest other cluster by centroid correlation.

    Closeness is measured on cluster centroids in log-normalized space. The
    nearest cluster is the population a fine boundary is most likely to be
    lost against (e.g. non-classical vs classical monocytes).
    """
    labels = list(masks)
    if len(labels) < 2:
        return {}
    centroids = np.vstack([_mean_axis0(X, masks[c]) for c in labels])
    # correlation distance is scale-free, so a small cluster is not pulled
    # toward whichever centroid happens to have the largest norm
    cent = centroids - centroids.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(cent, axis=1)
    norms[norms == 0] = 1.0
    corr = (cent @ cent.T) / np.outer(norms, norms)
    np.fill_diagonal(corr, -np.inf)
    return {labels[i]: labels[int(np.argmax(corr[i]))] for i in range(len(labels))}


# ---------------------------------------------------------------------------
# cap_allocation, part 2: merge cluster pairs that don't survive resampling,
# before cap_allocation computes equal share on them.
# ---------------------------------------------------------------------------
_MERGE_N_BOOT = 15
_MERGE_FRAC = 0.8
DEFAULT_CAP_MERGE_THRESHOLD = 0.5


def _pair_bootstrap_stability(
    X_pca: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    n_boot: int = _MERGE_N_BOOT,
    frac: float = _MERGE_FRAC,
    random_state: int = 0,
) -> float:
    """Mean ARI between the original A/B labels and a forced KMeans(k=2)
    split on an 80% cell subsample, repeated `n_boot` times, on the
    existing PCA embedding (no re-clustering from scratch).

    Deliberately not a significance test: testing "is A different from B"
    on the exact data Leiden used to produce A and B is circular (with
    thousands of genes, some difference is guaranteed). Testing whether
    the *same* split survives resampling asks a different, non-circular
    question -- the same logic this package already uses to price
    partition instability.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    idx = np.where(mask_a | mask_b)[0]
    n = idx.size
    if n < 20:
        return 1.0  # too few cells to test meaningfully; don't merge on noise
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        sub = rng.choice(idx, size=max(int(frac * n), 4), replace=False)
        km = KMeans(n_clusters=2, n_init=5, random_state=int(rng.integers(1_000_000)))
        km.fit(X_pca[sub])
        y_sub = mask_a[sub].astype(int)
        scores.append(adjusted_rand_score(y_sub, km.labels_))
    return float(np.mean(scores))


def _merge_unstable_clusters(
    X_pca: np.ndarray,
    cluster_labels: pd.Series,
    *,
    min_cluster_size: int,
    threshold: float = DEFAULT_CAP_MERGE_THRESHOLD,
    n_boot: int = _MERGE_N_BOOT,
    frac: float = _MERGE_FRAC,
    random_state: int = 0,
) -> tuple[pd.Series, list[tuple[str, str, float]]]:
    """Merge nearest-neighbour cluster pairs that don't survive cell
    resampling -- a single pass, never chained.

    All nearest-neighbour pairs are found and scored **once**, on the
    original (unmerged) clusters. Pairs scoring below `threshold` are
    merged greedily, worst first, but **a cluster already claimed by one
    merge cannot be tested or merged again in the same call** -- there is
    no second pass, no re-scoring, and a merged pair's combined cells are
    never re-offered as a candidate.

    This is deliberate, not a missing feature: a merged cluster is a worse
    test subject than an original one, since it already spans whatever its
    two components were. Chaining (score the *merged* cluster against its
    next neighbour, possibly merge again, repeat) turns "was this one
    population" into "can I find any 2-way split of an increasingly
    heterogeneous blob that fails to look like it" -- a question that gets
    easier to answer with a spuriously low score as the blob grows, not
    harder. A single non-chained pass cannot do that: at most
    `n_clusters // 2` merges are possible, structurally, because each
    cluster is spent after one.

    Only clusters passing `min_cluster_size` participate -- the ones
    `_build_cluster_gene_ranks` would drop anyway, and too small for a
    meaningful bootstrap. Returns the (possibly merged) labels and the
    list of merges actually made, for diagnostics.
    """
    labels = cluster_labels.astype(str).copy()
    sizes = labels.value_counts()
    active = [c for c in sizes.index if sizes[c] >= min_cluster_size]
    merges: list[tuple[str, str, float]] = []
    if len(active) < 2:
        return labels, merges

    masks = {c: (labels == c).to_numpy() for c in active}
    nn = _nearest_cluster_map(X_pca, masks)
    if not nn:
        return labels, merges

    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[float, str, str]] = []
    for c, nbr in nn.items():
        key = (c, nbr) if c < nbr else (nbr, c)
        if key in seen:
            continue
        seen.add(key)
        score = _pair_bootstrap_stability(
            X_pca,
            masks[c],
            masks[nbr],
            n_boot=n_boot,
            frac=frac,
            random_state=random_state,
        )
        candidates.append((score, c, nbr))
    candidates.sort(key=lambda x: x[0])  # worst (most unstable) first

    claimed: set[str] = set()
    rename: dict[str, str] = {}
    for score, a, b in candidates:
        if score >= threshold or a in claimed or b in claimed:
            continue
        claimed.add(a)
        claimed.add(b)
        merged_name = f"{a}+{b}"
        rename[a] = merged_name
        rename[b] = merged_name
        merges.append((a, b, score))

    if rename:
        labels = labels.replace(rename)
    return labels, merges


def _build_cluster_gene_ranks(
    adata: Any,
    *,
    cluster_labels: pd.Series,
    counts_layer: str,
    min_cluster_size: int,
    logfc_space: str = "log1p",
) -> dict[str, list[str]]:
    """Per-cluster gene order by the one-sided contrast (for coverage auto-n)."""
    labels = cluster_labels.reindex(adata.obs_names)
    sizes = labels.value_counts(dropna=True)
    valid = sizes[sizes >= min_cluster_size]
    if valid.empty:
        return {}
    if counts_layer not in adata.layers:
        return {}
    X_log, _pseudo = _logfc_inputs(_lognorm_matrix_from_counts(adata, counts_layer), logfc_space)
    out: dict[str, list[str]] = {}
    for cl in valid.index:
        mask = (labels == cl).to_numpy(dtype=bool)
        if mask.sum() < min_cluster_size or mask.sum() >= adata.n_obs:
            continue
        try:
            logfc = _cluster_vs_rest_logfc(X_log, mask, pseudo=_pseudo, one_sided=True)
            order = np.argsort(-logfc)
            out[str(cl)] = list(adata.var_names[order].astype(str))
        except _RECOVERABLE_ERRORS:
            continue
    return out


# ---------------------------------------------------------------------------
# Equal-share ceiling trim (formerly allocation_method='cap') — implementation
# kept private for research scripts; public API no longer exposes it.
# ---------------------------------------------------------------------------

# Tirosh et al. 2016 S/G2M gene sets (via Regev lab list, the same one
# scanpy's own cell-cycle tutorials ship). Symbols only -- a no-op on
# Ensembl-ID var_names, not a silent mismatch.
_CC_S_GENES = frozenset(
    """
MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1
MLF1IP HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3
MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1
CLSPN POLA1 CHAF1B BRIP1 E2F8
""".split()
)
_CC_G2M_GENES = frozenset(
    """
HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67
TMPO CENPF TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E
TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1
NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5
CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA
""".split()
)
_CELL_CYCLE_GENES = _CC_S_GENES | _CC_G2M_GENES
_CC_TOP_N = 20  # how far into a cluster's own logFC+ order to look
_CC_FRACTION = 0.3  # flag the cluster if >= this share of its top-N is cell-cycle


def _cell_cycle_flagged_clusters(ranks: Mapping[str, Sequence[str]]) -> set[str]:
    """Clusters whose own top marker genes are cell-cycle-dominated.

    An intermediate partition can split a dominant type into real, coherent
    sub-clusters that do not correspond to the population granularity a
    caller's ground truth cares about -- e.g. a proliferating substate with
    top genes like ``TOP2A, UBE2C, PBK, MKI67, CDC20, CENPF, BIRC5,
    AURKB...``. ``_cap_over_represented`` excludes flagged clusters from
    receiving backfill (their existing genes are left alone) so the
    mechanism does not actively promote more of a sub-structure the
    evaluation is unlikely to care about.

    Deliberately narrow: a standard reference list, not a general
    "is this a real population" classifier. It will not catch fragmentation
    that isn't cell-cycle-driven.
    """
    flagged: set[str] = set()
    for c, order_ in ranks.items():
        top = list(order_)[:_CC_TOP_N]
        if not top:
            continue
        frac = sum(1 for g in top if g in _CELL_CYCLE_GENES) / len(top)
        if frac >= _CC_FRACTION:
            flagged.add(str(c))
    return flagged


def _peak_cluster_map(
    genes: Iterable[str], own_pos: Mapping[str, Mapping[str, int]]
) -> dict[str, str]:
    """Attribute each gene to the cluster whose own logFC+ order ranks it best."""
    out: dict[str, str] = {}
    for g in genes:
        best_c: str | None = None
        best_p: int | None = None
        for c, pos in own_pos.items():
            p = pos.get(g)
            if p is not None and (best_p is None or p < best_p):
                best_p, best_c = p, c
        if best_c is not None:
            out[g] = best_c
    return out


def _cap_over_represented(
    selected: list[str],
    ranks: Mapping[str, Sequence[str]],
    scfair_score: pd.Series,
    n_top_genes: int,
    *,
    ceiling: float = 1.0,
    candidate_pool: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Trim clusters over ``ceiling * equal_share`` of their own genes in
    ``selected``; backfill the freed slots from the next-best genes by
    ``scfair_score`` (the full blended score, defined over every gene, not
    just the pool the original selection ranked).

    ``candidate_pool``, when given, restricts backfill to that gene set --
    for ``hybrid``, the same global top-``2*n_top`` pool the original
    selection was drawn from, so backfill cannot break hybrid's defining
    invariant (the selection never leaves that pool) even though
    ``scfair_score`` itself is defined genome-wide.

    No detection of "starved" clusters and no per-cluster target on the
    receiving side. A *targeted* top-up of under-represented clusters was
    tried and rejected: it actively hurts when the intermediate partition
    is finer than the labelled ground truth -- a real, coherent sub-cluster
    gets more of its own markers promoted, sharpening structure the
    evaluation does not carry and causing downstream over-splitting.
    Capping instead, with neutral backfill, scored better overall.

    **Removal is not universally safe, though** -- an over-ceiling
    cluster's "excess" genes are redundant on some datasets (removing them
    alone helps or is neutral) but a genuine, needed marker tail on others
    (removing them alone hurts more than the backfill side helps, e.g. a
    large, highly pure cluster with many real markers). "Over equal share"
    does not imply "redundant"; it can just mean "large and genuinely
    distinct."

    A gene ranking low in its *own* cluster's logFC+ order (why it may be
    removed) can still score well on ``scfair_score``, whose min-max
    normalisation runs over every gene rather than the smaller pool the
    original blend used -- so a capped cluster's own genes are excluded
    from backfill entirely, not just deprioritised, or the cap is silently
    undone by its own top genes flowing straight back in.

    Returns ``(selected_or_updated, diag)`` where ``diag`` is empty when
    nothing was over ceiling.
    """
    n_clusters = len(ranks)
    if n_clusters < 2:
        return selected, {}
    equal_share = n_top_genes / n_clusters
    own_pos = {c: {g: i for i, g in enumerate(order_)} for c, order_ in ranks.items()}
    cc_flagged = _cell_cycle_flagged_clusters(ranks)

    peak = _peak_cluster_map(selected, own_pos)
    count: dict[str, int] = {c: 0 for c in ranks}
    for g, c in peak.items():
        count[c] += 1

    current = set(selected)
    removed: list[str] = []
    trimmed: set[str] = set()
    excluded = set(cc_flagged)
    ceiling_n = int(round(ceiling * equal_share))
    for c in ranks:
        over = count.get(c, 0) - ceiling_n
        if over <= 0:
            continue
        trimmed.add(c)
        excluded.add(c)
        own_genes = sorted(
            (g for g in selected if peak.get(g) == c and g in current),
            key=lambda g: -own_pos[c].get(g, -1),  # deepest in own list first
        )
        for g in own_genes[:over]:
            current.discard(g)
            removed.append(g)

    if not removed:
        return selected, {
            "n_over": 0,
            "n_cc_flagged": len(cc_flagged),
            "n_added": 0,
            "n_removed": 0,
            "n_shortfall": 0,
        }

    score_order = [str(g) for g in scfair_score.sort_values(ascending=False).index]
    if candidate_pool is not None:
        pool_set = {str(g) for g in candidate_pool}
        score_order = [g for g in score_order if g in pool_set]
    added: list[str] = []
    need = len(removed)
    for g in score_order:
        if need == 0:
            break
        if g in current:
            continue
        best_c, best_p = None, None
        for c, pos in own_pos.items():
            p = pos.get(g)
            if p is not None and (best_p is None or p < best_p):
                best_p, best_c = p, c
        if best_c in excluded:
            continue
        current.add(g)
        added.append(g)
        need -= 1

    if need > 0:
        # No safe candidate for every freed slot (e.g. n_top_genes close to
        # n_vars leaves nothing outside `selected` to draw from, or every
        # remaining candidate belongs to a capped/cell-cycle cluster).
        # Restore rather than silently return fewer than len(selected)
        # genes -- capping must never shrink the selection.
        for g in removed[-need:]:
            current.add(g)
        removed = removed[: len(removed) - need]
        need = 0

    final = [g for g in selected if g in current] + added
    diag = {
        "n_over": len(trimmed),
        "n_cc_flagged": len(cc_flagged),
        "n_added": len(added),
        "n_removed": len(removed),
        "n_shortfall": need,
    }
    return final, diag


# ---------------------------------------------------------------------------
# coverage allocation (experimental alternative to cap):
#   1. legitimate units = merge NN pairs that fail stability OR pairwise DE
#   2. coverage-floor top-up for units whose own top-m markers are under-
#      represented in the hybrid selection; fund by dropping genes outside
#      every unit's top-m (never equal-share trim of large pure types)
# ---------------------------------------------------------------------------
DEFAULT_DE_SUPPORT_THRESHOLD = 0.25  # mean top-20 one-sided logFC, both dirs
DEFAULT_COVERAGE_OWN_M = 50
DEFAULT_COVERAGE_FLOOR = 0.40
DEFAULT_COVERAGE_TRIGGER = 0.50
DEFAULT_COVERAGE_BUDGET_FRAC = 0.10

# Starved top-up (equal-share deprivation)
DEFAULT_STARVED_TRIGGER = 0.50  # own peak count < trigger × equal_share → starved
DEFAULT_STARVED_TARGET = 0.50  # top up toward this × equal_share
# Hard ceiling on swaps as a fraction of n_top. Actual budget is *adaptive*:
# min(total_need, hard_cap, soft_cap) — 0 when no starved units; grows with
# how many units are starved; never above this ceiling (default 10%).
DEFAULT_STARVED_BUDGET_FRAC = 0.10
DEFAULT_STARVED_OWN_M = 50  # own-marker depth for candidates + protection


def _adaptive_starved_budget(
    *,
    n_top: int,
    n_units: int,
    n_starved: int,
    total_need: int,
    max_frac: float = DEFAULT_STARVED_BUDGET_FRAC,
) -> tuple[int, dict[str, Any]]:
    """Dataset-adaptive swap budget for starved top-up.

    A uniform fraction of ``n_top`` is the wrong shape: some datasets have no
    starved units (should spend 0) while multi-type atlases can exhaust a
    small fixed cap with real residual need. Policy:

    1. ``total_need`` — sum of genes starved units still want (demand).
    2. ``hard_cap = max_frac * n_top`` — safety ceiling (default **10%**).
    3. ``soft_cap`` scales with starvation *prevalence*:
       ``prevalence = n_starved / (0.5 * n_units)`` clipped to [0, 1]
       (full soft allowance when ≥ half of units are starved).
       ``soft_cap = hard_cap * prevalence``, then at least
       ``min(n_starved, hard_cap, total_need)`` so a single starved unit
       still gets a try.
    4. ``budget = min(total_need, hard_cap, soft_cap)``.
    """
    meta: dict[str, Any] = {
        "budget_mode": "adaptive",
        "total_need": int(max(0, total_need)),
        "max_frac": float(max_frac),
        "hard_cap": 0,
        "soft_cap": 0,
        "prevalence": 0.0,
    }
    if n_starved <= 0 or total_need <= 0 or n_top < 1 or n_units < 1:
        meta["budget_mode"] = "zero"
        return 0, meta
    hard_cap = max(0, int(round(float(max_frac) * int(n_top))))
    # Full soft cap when starved count reaches half the units
    prevalence = float(n_starved) / max(0.5 * float(n_units), 1.0)
    prevalence = float(min(1.0, max(0.0, prevalence)))
    soft_cap = max(0, int(round(hard_cap * prevalence)))
    soft_cap = max(soft_cap, min(int(n_starved), hard_cap, int(total_need)))
    budget = int(min(int(total_need), hard_cap, soft_cap))
    meta.update(
        {
            "hard_cap": hard_cap,
            "soft_cap": soft_cap,
            "prevalence": round(prevalence, 3),
        }
    )
    return budget, meta


def _pair_de_support(
    X_log: Any,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    top_n: int = 20,
    pseudo: float = 1e-2,
) -> float:
    """Gene-separability of a cluster pair: mean of each side's top-``n``
    one-sided logFC against the other, then averaged.

    Higher → the split needs independent gene support. Near zero → the two
    blocks are transcriptionally near-identical on this contrast and should
    not be independent allocation units.
    """
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if n_a < 5 or n_b < 5:
        return 0.0
    lfc_ab = _cluster_vs_rest_logfc(X_log, mask_a, pseudo=pseudo, one_sided=True, out_mask=mask_b)
    lfc_ba = _cluster_vs_rest_logfc(X_log, mask_b, pseudo=pseudo, one_sided=True, out_mask=mask_a)

    def _top_mean(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        k = min(top_n, int(x.size))
        # partial sort: the k largest
        part = np.partition(x, -k)[-k:]
        return float(np.mean(part))

    return 0.5 * (_top_mean(lfc_ab) + _top_mean(lfc_ba))


def _build_legitimate_units(
    X_pca: np.ndarray | None,
    X_log: Any,
    cluster_labels: pd.Series,
    *,
    min_cluster_size: int,
    stability_threshold: float | None = DEFAULT_CAP_MERGE_THRESHOLD,
    de_threshold: float | None = DEFAULT_DE_SUPPORT_THRESHOLD,
    n_boot: int = _MERGE_N_BOOT,
    frac: float = _MERGE_FRAC,
    random_state: int = 0,
    de_pseudo: float = 1e-2,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Merge nearest-neighbour pairs that are *not* legitimate allocation units.

    A split is kept only when it passes every criterion that is enabled
    (threshold not ``None``):

    - **stability** (needs ``X_pca``): subsample bootstrap ARI ≥ threshold
    - **DE support**: pairwise top-logFC mean ≥ threshold

    Single non-chained pass (same reason as :func:`_merge_unstable_clusters`).
    Returns relabelled series and a list of merge records for diagnostics.
    """
    labels = cluster_labels.astype(str).copy()
    sizes = labels.value_counts()
    active = [c for c in sizes.index if sizes[c] >= min_cluster_size]
    merges: list[dict[str, Any]] = []
    if len(active) < 2:
        return labels, merges
    if stability_threshold is None and de_threshold is None:
        return labels, merges

    masks = {c: (labels == c).to_numpy() for c in active}
    # Centroid graph on the same space used for DE when PCA is missing.
    nn_space: Any = X_pca if X_pca is not None else X_log
    try:
        nn = _nearest_cluster_map(nn_space, masks)
    except _RECOVERABLE_ERRORS:
        return labels, merges
    if not nn:
        return labels, merges

    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[float, str, str, float, float]] = []
    for c, nbr in nn.items():
        key = (c, nbr) if c < nbr else (nbr, c)
        if key in seen:
            continue
        seen.add(key)
        stab = 1.0
        if stability_threshold is not None and X_pca is not None:
            stab = _pair_bootstrap_stability(
                X_pca,
                masks[c],
                masks[nbr],
                n_boot=n_boot,
                frac=frac,
                random_state=random_state,
            )
        de = 1.0
        if de_threshold is not None:
            de = _pair_de_support(
                X_log,
                masks[c],
                masks[nbr],
                pseudo=de_pseudo,
            )
        # Sort key: worst first. Use min of normalised "pass margins" so pairs
        # that fail hard on either criterion rise to the front.
        stab_margin = (
            stab - float(stability_threshold)
            if stability_threshold is not None and X_pca is not None
            else 0.0
        )
        de_margin = de - float(de_threshold) if de_threshold is not None else 0.0
        # fail if any enabled criterion fails
        fails = False
        if stability_threshold is not None and X_pca is not None and stab < stability_threshold:
            fails = True
        if de_threshold is not None and de < de_threshold:
            fails = True
        if fails:
            candidates.append((min(stab_margin, de_margin), c, nbr, stab, de))

    candidates.sort(key=lambda x: x[0])
    claimed: set[str] = set()
    rename: dict[str, str] = {}
    for _margin, a, b, stab, de in candidates:
        if a in claimed or b in claimed:
            continue
        claimed.add(a)
        claimed.add(b)
        merged_name = f"{a}+{b}"
        rename[a] = merged_name
        rename[b] = merged_name
        merges.append(
            {
                "a": a,
                "b": b,
                "stability": round(float(stab), 3),
                "de_support": round(float(de), 3),
            }
        )
    if rename:
        labels = labels.replace(rename)
    return labels, merges


def _coverage_floor_allocate(
    selected: list[str],
    ranks: Mapping[str, Sequence[str]],
    scfair_score: pd.Series,
    n_top_genes: int,
    *,
    own_m: int = DEFAULT_COVERAGE_OWN_M,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    trigger: float = DEFAULT_COVERAGE_TRIGGER,
    budget_frac: float = DEFAULT_COVERAGE_BUDGET_FRAC,
    candidate_pool: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Top-up legitimate units whose own marker axes are under-covered.

    For each unit ``c`` let ``own = ranks[c][:own_m]`` and
    ``coverage = |own ∩ selected| / own_m``. If ``coverage < trigger``,
    request enough of ``own`` (in order, restricted to ``candidate_pool``)
    to reach ``coverage_floor * own_m``. Total adds are capped at
    ``budget_frac * n_top_genes``.

    Funding: drop genes in ``selected`` that are **not** in any unit's
    ``own[:own_m]``, lowest ``scfair_score`` first. This deliberately does
    **not** trim a large pure type's genuine marker tail (the failure mode
    of equal-share cap on a large, highly pure cluster). Never changes
    selection size: if the swap cannot be funded, fewer genes are added.
    """
    empty = {
        "n_units": 0,
        "n_starved": 0,
        "n_added": 0,
        "n_removed": 0,
        "budget": 0,
        "allocation_status": "not_run",
    }
    if not ranks or n_top_genes < 1 or not selected:
        return selected, empty

    m = max(1, int(own_m))
    floor_n = max(1, int(round(float(coverage_floor) * m)))
    budget = max(0, int(round(float(budget_frac) * int(n_top_genes))))
    n_units = len(ranks)
    # With ≤2 units, equal-share / coverage floors cannot see rare types that
    # the structure layer already merged away — n_starved=0 would look "fair".
    if n_units < 3:
        msg = (
            f"allocation_method='coverage' skipped: structure resolved only "
            f"{n_units} unit(s) (need ≥3). n_starved=0 means nothing to check, "
            "not that the selection is fair. Raise resolution or inspect "
            "clustering.n_clusters_total."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)
        return selected, {
            **empty,
            "n_units": n_units,
            "allocation_status": "skipped_structure_too_coarse",
            "skipped_reason": f"n_units={n_units}<3",
        }
    if budget == 0:
        return selected, {
            **empty,
            "n_units": n_units,
            "allocation_status": "budget_zero",
        }

    pool_set = {str(g) for g in candidate_pool} if candidate_pool is not None else None
    current = {str(g) for g in selected}
    protected: set[str] = set()
    unit_cov: dict[str, float] = {}
    # (unit, need, ordered candidates not yet in selection)
    requests: list[tuple[str, int, list[str]]] = []

    for c, order in ranks.items():
        own = [str(g) for g in list(order)[:m]]
        if not own:
            continue
        protected.update(own)
        hit = sum(1 for g in own if g in current)
        cov = hit / float(len(own))
        unit_cov[str(c)] = cov
        if cov >= float(trigger):
            continue
        need = max(0, floor_n - hit)
        if need <= 0:
            continue
        cands = [g for g in own if g not in current and (pool_set is None or g in pool_set)]
        if cands:
            requests.append((str(c), need, cands))

    if not requests:
        return selected, {
            "n_units": n_units,
            "n_starved": 0,
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "allocation_status": "checked_no_starved",
            "unit_coverage_before": {k: round(v, 3) for k, v in unit_cov.items()},
        }

    requests.sort(key=lambda t: unit_cov.get(t[0], 1.0))
    to_add: list[str] = []
    added_for: dict[str, int] = {}
    remaining = budget
    planned: set[str] = set()
    for c, need, cands in requests:
        if remaining <= 0:
            break
        took = 0
        for g in cands:
            if took >= need or remaining <= 0:
                break
            if g in planned or g in current:
                continue
            to_add.append(g)
            planned.add(g)
            took += 1
            remaining -= 1
        if took:
            added_for[c] = took

    if not to_add:
        return selected, {
            "n_units": n_units,
            "n_starved": len(requests),
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "allocation_status": "starved_but_no_candidates",
            "unit_coverage_before": {k: round(v, 3) for k, v in unit_cov.items()},
        }

    score = scfair_score.astype(float)
    removable = [str(g) for g in selected if str(g) not in protected]
    rem_scores = score.reindex(removable).fillna(-np.inf)
    removable = list(rem_scores.sort_values(ascending=True).index.astype(str))

    n_swap = min(len(to_add), len(removable))
    removed = removable[:n_swap]
    added = to_add[:n_swap]
    if n_swap == 0:
        return selected, {
            "n_units": n_units,
            "n_starved": len(requests),
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "allocation_status": "starved_but_unfunded",
            "unit_coverage_before": {k: round(v, 3) for k, v in unit_cov.items()},
            "note": "no unprotected genes to fund top-up",
        }

    drop = set(removed)
    final = [g for g in selected if str(g) not in drop] + added
    assert len(final) == len(selected), (
        f"coverage allocation changed selection size {len(selected)} -> {len(final)}"
    )

    return final, {
        "n_units": n_units,
        "n_starved": len(requests),
        "n_added": len(added),
        "n_removed": len(removed),
        "added_for": added_for,
        "budget": budget,
        "allocation_status": "applied" if added else "checked_no_change",
        "unit_coverage_before": {k: round(v, 3) for k, v in unit_cov.items()},
    }


def _starved_topup_allocate(
    selected: list[str],
    ranks: Mapping[str, Sequence[str]],
    scfair_score: pd.Series,
    n_top_genes: int,
    *,
    trigger_frac: float = DEFAULT_STARVED_TRIGGER,
    target_frac: float = DEFAULT_STARVED_TARGET,
    budget_frac: float = DEFAULT_STARVED_BUDGET_FRAC,
    own_m: int = DEFAULT_STARVED_OWN_M,
    candidate_pool: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Top-up units below equal-share own-gene count.

    For each unit, attribute selected genes by peak position in that unit's
    one-sided logFC+ order (same peak map as cap). Equal share is
    ``n_top / n_units``. A unit is **starved** when its peak count is
    ``< trigger_frac * equal_share`` (default 0.5). Request own markers
    (in rank order, restricted to ``candidate_pool``) until
    ``target_frac * equal_share``.

    **Adaptive budget:** not a uniform 5%/10% of the list for every
    dataset. ``budget_frac`` is the **hard ceiling** (default 10% of
    ``n_top``). Actual swaps = ``min(total_need, hard_cap, soft_cap)`` where
    soft_cap grows with how many units are starved (0 when none). See
    :func:`_adaptive_starved_budget`.

    Funding: drop genes not in any unit's ``own[:own_m]``, lowest
    ``scfair_score`` first — same design as coverage (does **not** trim a
    large pure type's genuine marker tail). Selection size is preserved.

    Does **not** trim over-represented units (that was deprecated cap).
    """
    empty: dict[str, Any] = {
        "n_units": 0,
        "n_starved": 0,
        "n_added": 0,
        "n_removed": 0,
        "budget": 0,
        "equal_share": 0.0,
        "allocation_status": "not_run",
    }
    if not ranks or n_top_genes < 1 or not selected:
        return selected, empty

    n_units = len(ranks)
    # ≤2 units: equal-share is near parity by construction even when true
    # rare types were absorbed into majority Leiden communities.
    if n_units < 3:
        msg = (
            f"allocation_method='starved_topup' skipped: structure resolved only "
            f"{n_units} unit(s) (need ≥3). n_starved=0 means nothing to check, "
            "not that the selection is fair. Raise resolution or inspect "
            "clustering.n_clusters_total."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)
        return selected, {
            **empty,
            "n_units": n_units,
            "allocation_status": "skipped_structure_too_coarse",
            "skipped_reason": f"n_units={n_units}<3",
        }

    equal_share = float(n_top_genes) / float(n_units)
    trigger_n = float(trigger_frac) * equal_share
    target_n = float(target_frac) * equal_share
    m = max(1, int(own_m))

    own_pos = {str(c): {str(g): i for i, g in enumerate(order_)} for c, order_ in ranks.items()}
    peak = _peak_cluster_map([str(g) for g in selected], own_pos)
    count: dict[str, int] = {str(c): 0 for c in ranks}
    for _g, c in peak.items():
        if c in count:
            count[c] += 1

    pool_set = {str(g) for g in candidate_pool} if candidate_pool is not None else None
    current = {str(g) for g in selected}
    protected: set[str] = set()
    for order in ranks.values():
        protected.update(str(g) for g in list(order)[:m])

    # (unit, need, cands, share_before)
    requests: list[tuple[str, int, list[str], float]] = []
    share_before: dict[str, float] = {}
    for c, order in ranks.items():
        cs = str(c)
        n_own = int(count.get(cs, 0))
        share = n_own / equal_share if equal_share > 0 else 0.0
        share_before[cs] = share
        if n_own >= trigger_n - 1e-9:
            continue
        need = max(0, int(np.ceil(target_n - n_own)))
        if need <= 0:
            continue
        cands = [
            str(g)
            for g in order
            if str(g) not in current and (pool_set is None or str(g) in pool_set)
        ]
        # prefer own-axis head for top-up quality
        cands = cands[: max(need * 3, m)]
        if cands:
            requests.append((cs, need, cands, share))

    total_need = int(sum(need for _c, need, _cands, _s in requests))
    budget, bud_meta = _adaptive_starved_budget(
        n_top=int(n_top_genes),
        n_units=n_units,
        n_starved=len(requests),
        total_need=total_need,
        max_frac=float(budget_frac),
    )

    if not requests:
        return selected, {
            "n_units": n_units,
            "n_starved": 0,
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "equal_share": round(equal_share, 3),
            "share_before": {k: round(v, 3) for k, v in share_before.items()},
            "allocation_status": "checked_no_starved",
            **bud_meta,
        }
    if budget <= 0:
        return selected, {
            "n_units": n_units,
            "n_starved": len(requests),
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "equal_share": round(equal_share, 3),
            "share_before": {k: round(v, 3) for k, v in share_before.items()},
            "allocation_status": "budget_zero",
            **bud_meta,
        }

    # worst share first
    requests.sort(key=lambda t: t[3])
    to_add: list[str] = []
    added_for: dict[str, int] = {}
    remaining = budget
    planned: set[str] = set()
    for cs, need, cands, _share in requests:
        if remaining <= 0:
            break
        took = 0
        for g in cands:
            if took >= need or remaining <= 0:
                break
            if g in planned or g in current:
                continue
            to_add.append(g)
            planned.add(g)
            took += 1
            remaining -= 1
        if took:
            added_for[cs] = took

    if not to_add:
        return selected, {
            "n_units": n_units,
            "n_starved": len(requests),
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "equal_share": round(equal_share, 3),
            "share_before": {k: round(v, 3) for k, v in share_before.items()},
            "allocation_status": "starved_but_no_candidates",
            "note": "no pool candidates for starved units",
            **bud_meta,
        }

    score = scfair_score.astype(float)
    removable = [str(g) for g in selected if str(g) not in protected]
    rem_scores = score.reindex(removable).fillna(-np.inf)
    removable = list(rem_scores.sort_values(ascending=True).index.astype(str))

    n_swap = min(len(to_add), len(removable))
    removed = removable[:n_swap]
    added = to_add[:n_swap]
    if n_swap == 0:
        return selected, {
            "n_units": n_units,
            "n_starved": len(requests),
            "n_added": 0,
            "n_removed": 0,
            "budget": budget,
            "equal_share": round(equal_share, 3),
            "share_before": {k: round(v, 3) for k, v in share_before.items()},
            "allocation_status": "starved_but_unfunded",
            "note": "no unprotected genes to fund starved top-up",
            **bud_meta,
        }

    drop = set(removed)
    final = [g for g in selected if str(g) not in drop] + added
    assert len(final) == len(selected), (
        f"starved_topup changed selection size {len(selected)} -> {len(final)}"
    )

    return final, {
        "n_units": n_units,
        "n_starved": len(requests),
        "n_added": len(added),
        "n_removed": len(removed),
        "added_for": added_for,
        "budget": budget,
        "equal_share": round(equal_share, 3),
        "trigger_frac": float(trigger_frac),
        "target_frac": float(target_frac),
        "budget_frac": float(budget_frac),
        "share_before": {k: round(v, 3) for k, v in share_before.items()},
        "allocation_status": "applied" if added else "checked_no_change",
        **bud_meta,
    }


def _minmax_norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = float(np.nanmin(s.to_numpy())), float(np.nanmax(s.to_numpy()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def _score_weighted_select(
    adata: Any,
    *,
    counts_layer: str,
    n_top_genes: int,
    flavor: str,
    resolution: float,
    min_cluster_size: int,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    balance_power: float,
    global_rank: pd.Series,
    global_scores: pd.Series | None = None,
    blend_global: float = 0.0,
    hybrid: bool = False,
    neighbor_contrast: float = 0.0,
    combine: str = "blend",
    cluster_mask: np.ndarray | None = None,
    progress: bool = False,
    scale_clustering: bool = True,
    logfc_space: str = "log1p",
    diag_out: dict[str, Any] | None = None,
    consensus_resolutions: Sequence[float] | None = None,
    allocation_method: str = "none",
    cap_ceiling: float = 1.0,
    cap_merge_threshold: float | None = 0.5,
    spec_on_legitimate_units: bool = False,
    graph_cache: dict[str, Any] | None = None,
    span: float,
    n_bins: int,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    max_disp: float,
) -> tuple[list[str], pd.Series | None, int, dict[str, float], pd.Series]:
    """Specificity scores; optional hybrid global-anchor selection.

    ``graph_cache``: optional mutable dict shared across auto's first select
    and hybrid realign. When the intermediate protocol matches, PCA + Leiden
    and (when score protocol matches) specificity scores are reused so the
    second pass only re-cuts the hybrid 2×k pool.
    """
    del flavor, span, n_bins, min_mean, max_mean, min_disp, max_disp

    if cluster_mask is not None:
        _mask_for_proto = np.asarray(cluster_mask, dtype=bool)
    else:
        _mask_for_proto = adata.var["highly_variable"].to_numpy()
    score_proto = _intermediate_protocol(
        hvg_mask=_mask_for_proto,
        counts_layer=counts_layer,
        resolution=resolution,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        scale_clustering=scale_clustering,
        resolutions=consensus_resolutions,
        min_cluster_size=min_cluster_size,
        balance_power=balance_power,
        logfc_space=logfc_space,
        neighbor_contrast=neighbor_contrast,
        combine=combine,
        hybrid=hybrid,
        blend_global=blend_global,
        allocation_method=str(allocation_method or "none"),
        cap_ceiling=cap_ceiling,
        cap_merge_threshold=cap_merge_threshold,
        spec_on_legitimate_units=spec_on_legitimate_units,
    )
    score_key = _score_protocol_key(score_proto)

    # Full specificity reuse: graph + scoring knobs match a prior pass.
    if (
        graph_cache is not None
        and graph_cache.get("score_key") == score_key
        and graph_cache.get("S_spec") is not None
        and graph_cache.get("cluster_labels") is not None
        and graph_cache.get("n_scored", 0) >= 1
    ):
        cluster_labels = graph_cache["cluster_labels"]
        weight_map = dict(graph_cache.get("weight_map") or {})
        n_scored = int(graph_cache["n_scored"])
        X_pca = (
            None
            if graph_cache.get("X_pca") is None
            else np.asarray(graph_cache["X_pca"], dtype=float)
        )
        S_spec = graph_cache["S_spec"].copy()
        if diag_out is not None:
            diag_out["intermediate_reused"] = True
            diag_out["intermediate_reuse"] = "specificity"
            if "n_passes" not in diag_out:
                diag_out["n_passes"] = int(graph_cache.get("diag_snapshot", {}).get("n_passes", 1))
            for k, v in (graph_cache.get("diag_snapshot") or {}).items():
                if k not in diag_out:
                    diag_out[k] = v
            diag_out.setdefault("spec_partition", graph_cache.get("spec_partition", "leiden"))
        _progress(
            progress,
            "reusing intermediate specificity scores (skip PCA/Leiden/logFC); "
            "re-cutting hybrid pool at n_top=%d...",
            int(n_top_genes),
        )
        # Jump to hybrid/score selection with cached S_spec
        if hybrid:
            selected, S_out, hybrid_pool = _hybrid_anchor_select(
                global_rank=global_rank,
                global_scores=global_scores if global_scores is not None else (-global_rank),
                spec_scores=S_spec,
                n_top_genes=n_top_genes,
                blend_global=blend_global,
                combine=combine,
                pool_factor=2.0,
            )
            # Allocation may still need ad_full / X_log — run only when requested.
            alloc_diag: dict[str, Any] = {}
            alloc_note = ""
            if allocation_method in ("coverage", "starved_topup") and n_scored >= 2:
                ad_full = _restore_raw_counts(adata, layer=counts_layer, full_genes=True)
                if counts_layer not in ad_full.layers:
                    ad_full.layers[counts_layer] = ad_full.X.copy()
                _store_raw_counts(ad_full, layer=counts_layer, mode="force", overwrite=True)
                X_log, _pseudo = _logfc_inputs(
                    _lognorm_matrix_from_counts(ad_full, counts_layer), logfc_space
                )
                if allocation_method == "coverage":
                    unit_labels, unit_merges = _build_legitimate_units(
                        X_pca,
                        X_log,
                        cluster_labels,
                        min_cluster_size=min_cluster_size,
                        stability_threshold=cap_merge_threshold,
                        de_threshold=DEFAULT_DE_SUPPORT_THRESHOLD,
                        random_state=random_state,
                        de_pseudo=float(_pseudo),
                    )
                    if unit_merges and diag_out is not None:
                        diag_out["coverage_merges"] = unit_merges
                    unit_ranks = _build_cluster_gene_ranks(
                        ad_full,
                        cluster_labels=unit_labels,
                        counts_layer=counts_layer,
                        min_cluster_size=min_cluster_size,
                        logfc_space=logfc_space,
                    )
                    if unit_ranks:
                        selected, alloc_diag = _coverage_floor_allocate(
                            selected,
                            unit_ranks,
                            S_out,
                            n_top_genes,
                            candidate_pool=hybrid_pool,
                        )
                    if diag_out is not None and alloc_diag:
                        # Keys are coverage_*; do not also set allocation_method
                        # here — that lives only on uns["scfair"]["hvg"] (request).
                        diag_out.update({f"coverage_{k}": v for k, v in alloc_diag.items()})
                    if alloc_diag.get("n_added"):
                        alloc_note = (
                            f"; coverage topped up {alloc_diag.get('n_starved', 0)} "
                            f"unit(s), swapped {alloc_diag['n_added']} gene(s)"
                        )
                else:  # starved_topup
                    unit_labels, unit_merges = _build_legitimate_units(
                        X_pca,
                        X_log,
                        cluster_labels,
                        min_cluster_size=min_cluster_size,
                        stability_threshold=cap_merge_threshold,
                        de_threshold=DEFAULT_DE_SUPPORT_THRESHOLD,
                        random_state=random_state,
                        de_pseudo=float(_pseudo),
                    )
                    if unit_merges and diag_out is not None:
                        diag_out["starved_topup_merges"] = unit_merges
                    unit_ranks = _build_cluster_gene_ranks(
                        ad_full,
                        cluster_labels=unit_labels,
                        counts_layer=counts_layer,
                        min_cluster_size=min_cluster_size,
                        logfc_space=logfc_space,
                    )
                    if unit_ranks:
                        selected, alloc_diag = _starved_topup_allocate(
                            selected,
                            unit_ranks,
                            S_out,
                            n_top_genes,
                            candidate_pool=hybrid_pool,
                        )
                    if diag_out is not None and alloc_diag:
                        diag_out.update({f"starved_topup_{k}": v for k, v in alloc_diag.items()})
                    if alloc_diag.get("n_added"):
                        alloc_note = (
                            f"; starved_topup filled {alloc_diag.get('n_starved', 0)} "
                            f"unit(s), swapped {alloc_diag['n_added']} gene(s)"
                        )
            logger.info(
                "hybrid (reused intermediate): re-rank global top-%d pool by "
                "%.2f·global + %.2f·specificity → %d genes%s.",
                int(round(n_top_genes * 2.0)),
                blend_global,
                1.0 - blend_global,
                len(selected),
                alloc_note,
            )
            return selected, cluster_labels, n_scored, weight_map, S_out

        S_use = S_spec.copy()
        selected = _top_genes_from_scores(S_use, n_top_genes)
        logger.info(
            "score (reused intermediate): selected %d genes from cached specificity.",
            len(selected),
        )
        return selected, cluster_labels, n_scored, weight_map, S_use

    extra_partitions: list[tuple[pd.Series, pd.Series]] | None = (
        [] if consensus_resolutions else None
    )

    cluster_labels, valid, weight_map, X_pca = _prepare_clusters(
        adata,
        counts_layer=counts_layer,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        balance_power=balance_power,
        global_rank=global_rank,
        n_top_genes=n_top_genes,
        cluster_mask=cluster_mask,
        progress=progress,
        scale_clustering=scale_clustering,
        diag_out=diag_out,
        resolutions=consensus_resolutions,
        extra_out=extra_partitions,
        graph_cache=graph_cache,
    )
    if cluster_labels is None or valid is None:
        scores = (-global_rank).reindex(adata.var_names).fillna(-np.inf)
        return _top_genes_from_rank(global_rank, n_top_genes), cluster_labels, 0, {}, scores

    weights = _cluster_size_weights(valid, balance_power)

    ad_full = _restore_raw_counts(adata, layer=counts_layer, full_genes=True)
    if counts_layer not in ad_full.layers:
        ad_full.layers[counts_layer] = ad_full.X.copy()
    _store_raw_counts(ad_full, layer=counts_layer, mode="force", overwrite=True)

    X_log, _pseudo = _logfc_inputs(_lognorm_matrix_from_counts(ad_full, counts_layer), logfc_space)
    gene_index = pd.Index(ad_full.var_names.astype(str))

    # Optional: score specificity on legitimate units (stable ∧ DE-supported
    # merges of nearest-neighbour Leiden pairs) instead of raw Leiden.
    # Selection still ends at hybrid blend; no post-hoc allocation here.
    if spec_on_legitimate_units:
        unit_labels, unit_merges = _build_legitimate_units(
            X_pca,
            X_log,
            cluster_labels,
            min_cluster_size=min_cluster_size,
            stability_threshold=(
                DEFAULT_CAP_MERGE_THRESHOLD
                if cap_merge_threshold is None
                else float(cap_merge_threshold)
            ),
            de_threshold=DEFAULT_DE_SUPPORT_THRESHOLD,
            random_state=random_state,
            de_pseudo=float(_pseudo),
        )
        sizes_u = unit_labels.value_counts(dropna=True)
        valid_u = sizes_u[sizes_u >= min_cluster_size]
        if len(valid_u) >= 2:
            if diag_out is not None:
                diag_out["spec_partition"] = "legitimate_units"
                diag_out["spec_units_merges"] = unit_merges
                diag_out["n_leiden_before_units"] = int(cluster_labels.nunique())
                diag_out["n_units_for_spec"] = int(len(valid_u))
            cluster_labels = unit_labels
            valid = valid_u
            weights = _cluster_size_weights(valid, balance_power)
            weight_map = {str(k): float(v) for k, v in weights.items()}
        elif diag_out is not None:
            diag_out["spec_partition"] = "leiden_fallback"
            diag_out["spec_units_merges"] = unit_merges
            diag_out["spec_units_note"] = "units collapsed to <2; scored on Leiden"

    if diag_out is not None and "spec_partition" not in diag_out:
        diag_out["spec_partition"] = "leiden"

    def _score_partition(labels: pd.Series, w_map) -> tuple[np.ndarray, np.ndarray, dict, int]:
        """Weighted cluster-vs-rest logFC for one partition."""
        lf = labels.reindex(ad_full.obs_names)
        S_ = np.zeros(ad_full.n_vars, dtype=float)
        S_max_ = np.zeros(ad_full.n_vars, dtype=float)
        masks_: dict[str, np.ndarray] = {}
        n_ = 0
        for cl, w in w_map.items():
            mask = (lf == cl).to_numpy(dtype=bool)
            n_cells = int(mask.sum())
            if n_cells < min_cluster_size or n_cells >= ad_full.n_obs:
                continue
            try:
                logfc = _cluster_vs_rest_logfc(X_log, mask, pseudo=_pseudo, one_sided=True)
                S_ += float(w) * logfc
                S_max_ = np.maximum(S_max_, logfc)
                masks_[str(cl)] = mask
                n_ += 1
            except _RECOVERABLE_ERRORS as exc:
                logger.warning("Cluster-vs-rest scoring failed for cluster %s (%s); skip.", cl, exc)
        return S_, S_max_, masks_, n_

    _progress(
        progress,
        "specificity scoring across %d intermediate clusters%s...",
        len(weights),
        f" (+{len(extra_partitions)} consensus partitions)" if extra_partitions else "",
    )
    # S: size-weighted sum over clusters. S_max: per-gene max logFC, so a tiny
    # but valid cluster can still surface its identity genes.
    S, S_max, masks, n_scored = _score_partition(cluster_labels, weights)

    # Nearest-neighbour contrast. Cluster-vs-*rest* cannot see a boundary
    # between two adjacent populations: for a rare subset the "rest" is
    # dominated by distant lineages, so every gene of the parent lineage
    # scores high and the genes that actually separate the subset from its
    # neighbour get no credit. Scoring each cluster against its closest
    # neighbour restores exactly those genes.
    S_nn = np.zeros(ad_full.n_vars, dtype=float)
    n_nn = 0
    if neighbor_contrast > 0 and len(masks) >= 2:
        try:
            nn_map = _nearest_cluster_map(X_log, masks)
            for cl, nn in nn_map.items():
                logfc_nn = _cluster_vs_rest_logfc(
                    X_log, masks[cl], one_sided=True, out_mask=masks[nn]
                )
                S_nn = np.maximum(S_nn, logfc_nn)
                n_nn += 1
        except _RECOVERABLE_ERRORS as exc:
            logger.warning("Nearest-neighbour contrast failed (%s); skipping.", exc)
            S_nn = np.zeros(ad_full.n_vars, dtype=float)
            n_nn = 0

    if n_scored == 0:
        scores = (-global_rank).reindex(adata.var_names).fillna(-np.inf)
        return (
            _top_genes_from_rank(global_rank, n_top_genes),
            cluster_labels,
            0,
            weight_map,
            scores,
        )

    # Combine weighted sum with max-logFC so tiny-but-valid clusters still surface
    # strong identity genes (e.g. PPBP) without equal-vote domination. The
    # max term is optionally blended toward the nearest-neighbour contrast:
    # both are "max over clusters of a one-sided logFC", so they share a
    # scale and the 0.7/0.3 split is unchanged.
    if n_nn > 0:
        S_peak = (1.0 - neighbor_contrast) * S_max + neighbor_contrast * S_nn
    else:
        S_peak = S_max
    S_comb = 0.7 * S + 0.3 * S_peak

    # Consensus across the resolution ladder. Each partition is scored in full
    # and then min-max normalised before averaging: partitions with more
    # clusters produce systematically larger weighted sums, so averaging raw
    # scores would silently let the finest resolution dominate.
    #
    # Motivation: a single Leiden partition is not very stable (it can move
    # substantially under a small gene perturbation) and can be fine-grained
    # yet misaligned with the populations being scored. Averaging over a
    # ladder cannot lock onto one wrong draw the way a single partition can.
    if extra_partitions:
        combined = [_minmax_norm(pd.Series(S_comb, index=gene_index))]
        for labels_i, valid_i in extra_partitions:
            w_i = _cluster_size_weights(valid_i, balance_power)
            S_i, S_max_i, _, n_i = _score_partition(labels_i, w_i)
            if n_i == 0:
                continue
            combined.append(_minmax_norm(pd.Series(0.7 * S_i + 0.3 * S_max_i, index=gene_index)))
        S_comb = pd.concat(combined, axis=1).mean(axis=1).to_numpy()
        logger.info(
            "consensus specificity: averaged over %d partitions (resolutions %s).",
            len(combined),
            list(consensus_resolutions or []),
        )

    S_spec = pd.Series(S_comb, index=gene_index)
    S_spec = S_spec.reindex(adata.var_names.astype(str)).fillna(0.0)
    S_spec.index = adata.var_names

    # Cache for auto realign: graph already stored by _prepare_clusters;
    # attach specificity so the second pass can skip logFC scoring too.
    if graph_cache is not None and n_scored >= 1:
        graph_cache["score_key"] = score_key
        graph_cache["score_proto"] = score_proto
        graph_cache["S_spec"] = S_spec.copy()
        graph_cache["cluster_labels"] = cluster_labels
        graph_cache["weight_map"] = dict(weight_map)
        graph_cache["n_scored"] = int(n_scored)
        graph_cache["X_pca"] = None if X_pca is None else np.asarray(X_pca, dtype=float)
        graph_cache["spec_partition"] = (
            None if diag_out is None else diag_out.get("spec_partition", "leiden")
        )
        if diag_out is not None:
            snap = dict(graph_cache.get("diag_snapshot") or {})
            for k in (
                "n_passes",
                "n_genes_clustered",
                "n_cells",
                "resolution",
                "n_pcs_used",
                "n_neighbors_used",
                "min_cluster_size",
                "n_clusters_total",
                "n_clusters_kept",
                "cluster_sizes",
                "clusters_dropped",
                "spec_partition",
            ):
                if k in diag_out:
                    snap[k] = diag_out[k]
            graph_cache["diag_snapshot"] = snap

    # `blend_global=0` still means "specificity-only ranking *within the global
    # pool*" — the pool restriction is what makes this method hybrid. Gating on
    # `blend_global > 0` used to drop out of the pool entirely and silently return
    # genome-wide pure `score` while metadata still said hybrid.
    if hybrid:
        selected, S_out, hybrid_pool = _hybrid_anchor_select(
            global_rank=global_rank,
            global_scores=global_scores if global_scores is not None else (-global_rank),
            spec_scores=S_spec,
            n_top_genes=n_top_genes,
            blend_global=blend_global,
            combine=combine,
            pool_factor=2.0,
        )

        # Post-hybrid allocation. Needs ≥2 scored clusters for units to form;
        # allocate functions themselves require ≥3 units or they skip with a
        # clear status (skipped_structure_too_coarse ≠ checked_no_starved).
        alloc_diag: dict[str, Any] = {}
        alloc_note = ""
        if allocation_method == "coverage" and n_scored >= 2:
            unit_labels, unit_merges = _build_legitimate_units(
                X_pca,
                X_log,
                cluster_labels,
                min_cluster_size=min_cluster_size,
                stability_threshold=cap_merge_threshold,
                de_threshold=DEFAULT_DE_SUPPORT_THRESHOLD,
                random_state=random_state,
                de_pseudo=float(_pseudo),
            )
            if unit_merges and diag_out is not None:
                diag_out["coverage_merges"] = unit_merges
            unit_ranks = _build_cluster_gene_ranks(
                ad_full,
                cluster_labels=unit_labels,
                counts_layer=counts_layer,
                min_cluster_size=min_cluster_size,
                logfc_space=logfc_space,
            )
            if unit_ranks:
                selected, alloc_diag = _coverage_floor_allocate(
                    selected,
                    unit_ranks,
                    S_out,
                    n_top_genes,
                    candidate_pool=hybrid_pool,
                )
            if diag_out is not None and alloc_diag:
                diag_out.update({f"coverage_{k}": v for k, v in alloc_diag.items()})
            if alloc_diag.get("n_added"):
                alloc_note = (
                    f"; coverage topped up {alloc_diag.get('n_starved', 0)} "
                    f"unit(s), swapped {alloc_diag['n_added']} gene(s)"
                )
        elif allocation_method == "starved_topup" and n_scored >= 2:
            unit_labels, unit_merges = _build_legitimate_units(
                X_pca,
                X_log,
                cluster_labels,
                min_cluster_size=min_cluster_size,
                stability_threshold=cap_merge_threshold,
                de_threshold=DEFAULT_DE_SUPPORT_THRESHOLD,
                random_state=random_state,
                de_pseudo=float(_pseudo),
            )
            if unit_merges and diag_out is not None:
                diag_out["starved_topup_merges"] = unit_merges
            unit_ranks = _build_cluster_gene_ranks(
                ad_full,
                cluster_labels=unit_labels,
                counts_layer=counts_layer,
                min_cluster_size=min_cluster_size,
                logfc_space=logfc_space,
            )
            if unit_ranks:
                selected, alloc_diag = _starved_topup_allocate(
                    selected,
                    unit_ranks,
                    S_out,
                    n_top_genes,
                    candidate_pool=hybrid_pool,
                )
            if diag_out is not None and alloc_diag:
                diag_out.update({f"starved_topup_{k}": v for k, v in alloc_diag.items()})
            if alloc_diag.get("n_added"):
                alloc_note = (
                    f"; starved_topup filled {alloc_diag.get('n_starved', 0)} "
                    f"unit(s), swapped {alloc_diag['n_added']} gene(s)"
                )

        logger.info(
            "hybrid: re-rank global top-%d pool by "
            "%.2f·global + %.2f·specificity (%d/%d clusters, β=%.3f) → %d genes"
            "%s.",
            int(round(n_top_genes * 2.0)),
            blend_global,
            1.0 - blend_global,
            n_scored,
            len(valid),
            balance_power,
            len(selected),
            alloc_note,
        )
        return selected, cluster_labels, n_scored, weight_map, S_out

    S_use = S_spec.copy()
    selected = _top_genes_from_scores(S_use, n_top_genes)
    logger.info(
        "score: %d/%d clusters (min_size=%d, β=%.3f); selected %d genes "
        "(0.7·Σ w logFC⁺ + 0.3·max logFC⁺).",
        n_scored,
        len(valid),
        min_cluster_size,
        balance_power,
        len(selected),
    )
    return selected, cluster_labels, n_scored, weight_map, S_use


def _best_rank_score(a: pd.Series, b: pd.Series) -> pd.Series:
    """Each gene takes its better of two ranks; ties broken by the other rank.

    Returned as a score (higher = better). Unlike a linear blend this can only
    promote a gene relative to its rank in either input — never demote it.
    """
    ra = a.rank(ascending=False, method="average")
    rb = b.rank(ascending=False, method="average")
    best = np.minimum(ra.to_numpy(), rb.to_numpy())
    other = np.maximum(ra.to_numpy(), rb.to_numpy())
    return pd.Series(-(best + 1e-6 * other), index=a.index)


def _hybrid_anchor_select(
    *,
    global_rank: pd.Series,
    global_scores: pd.Series,
    spec_scores: pd.Series,
    n_top_genes: int,
    blend_global: float,
    combine: str = "blend",
    pool_factor: float = 2.0,
) -> tuple[list[str], pd.Series, list[str]]:
    """Re-rank a *global HVG pool* by blending global + specificity scores.

    Strategy (designed to not abandon the scanpy subspace):

    1. Take the top ``pool_factor * n_top`` genes by **global** rank as a
       candidate pool (default 2× → only genes that are already globally
       variable can enter).
    2. Within that pool, score
       ``S = blend_global · norm(global) + (1-blend_global) · norm(spec)``.
    3. Select top ``n_top`` by ``S``.

    Marker genes are not handled here: ``marker_mode="force"`` merges them after
    selection, so this function stays purely score-driven.
    """
    n_pool = int(round(n_top_genes * max(pool_factor, 1.0)))
    n_pool = min(n_pool, len(global_scores))

    # Order by the all-gene variability score, NOT by ``global_rank``.
    # scanpy only assigns highly_variable_rank to the top ``n_top`` genes and
    # leaves the rest NaN, which _gene_rank_series fills with +inf. Sorting that
    # column therefore yields the true top-``n_top`` followed by inf-ranked genes
    # in *index order*, so the second half of a 2x pool was essentially
    # arbitrary rather than the true ranks k+1..2k. The genes that specificity
    # is supposed to be able to promote were mostly absent from the pool while
    # low-variance genes sat in it. ``global_scores`` is defined for every gene
    # (variances_norm etc.), so ordering by it gives the pool the docstring
    # claims. When global_scores was unavailable the caller passes
    # ``-global_rank``, which reduces to the previous ordering.
    global_ordered = list(
        global_scores.sort_values(ascending=False, kind="stable").index.astype(str)
    )
    pool = global_ordered[:n_pool]

    g_sc = global_scores.reindex(pool).fillna(0.0)
    s_sc = spec_scores.reindex(pool).fillna(0.0)
    g_sc.index = pool
    s_sc.index = pool
    if combine == "best_rank":
        S_pool = _best_rank_score(g_sc, s_sc)
    else:
        g_norm = _minmax_norm(g_sc)
        s_norm = _minmax_norm(s_sc)
        S_pool = blend_global * g_norm + (1.0 - blend_global) * s_norm
    selected = list(S_pool.sort_values(ascending=False).index[:n_top_genes])

    # Report the *same* scores used for selection: pool-normalized blend, NaN
    # outside the candidate pool. Re-normalizing on the full genome made
    # ``scfair_score`` disagree with ``highly_variable`` (different min-max).
    S_out = pd.Series(np.nan, index=spec_scores.index, dtype=float)
    S_out.loc[S_pool.index] = S_pool.to_numpy(dtype=float)
    return selected, S_out, pool


# ---------------------------------------------------------------------------
# reweight: cell-reweighted global HVG
# ---------------------------------------------------------------------------


def _cell_weights(
    labels: pd.Series,
    valid: pd.Series,
    *,
    balance_power: float,
) -> np.ndarray:
    """Per-cell weights so cluster total mass ∝ n^{β}: u_i ∝ n_c^{β-1}."""
    n_obs = len(labels)
    w = np.ones(n_obs, dtype=float)
    # map obs position
    # labels aligned to a contiguous index 0..n_obs-1
    lab = labels.to_numpy()
    for cl, n in valid.items():
        mask = lab == cl
        n_c = float(n)
        if balance_power == 1.0:
            w[mask] = 1.0
        else:
            w[mask] = n_c ** (balance_power - 1.0)
    # Cells outside any valid cluster must be *genuinely* neutral. Leaving them at
    # 1.0 was not neutral: valid per-cell weights are n_c**(β-1), which for β=0.5
    # and n_c=1000 is 0.032, so an excluded cell carried ~31x the resampling mass
    # of a cell in a large valid cluster — inverting the size-balancing this method
    # exists to do. Use the mean valid per-cell weight instead.
    valid_cells = np.isin(lab, np.asarray(list(valid.index), dtype=lab.dtype))
    if valid_cells.any() and (~valid_cells).any():
        w[~valid_cells] = float(w[valid_cells].mean())

    total = w.sum()
    if total <= 0:
        return np.full(n_obs, 1.0 / n_obs)
    return w / total


def _cell_reweight_select(
    adata: Any,
    *,
    counts_layer: str,
    n_top_genes: int,
    flavor: str,
    resolution: float,
    min_cluster_size: int,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    balance_power: float,
    global_rank: pd.Series,
    batch_key: str | None,
    cluster_mask: np.ndarray | None = None,
    progress: bool = False,
    scale_clustering: bool = True,
    logfc_space: str = "log1p",
    diag_out: dict[str, Any] | None = None,
    span: float,
    n_bins: int,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    max_disp: float,
) -> tuple[list[str], pd.Series | None, int, dict[str, float], pd.Series]:
    cluster_labels, valid, weight_map, _X_pca_unused = _prepare_clusters(
        adata,
        counts_layer=counts_layer,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        balance_power=balance_power,
        global_rank=global_rank,
        n_top_genes=n_top_genes,
        cluster_mask=cluster_mask,
        progress=progress,
        scale_clustering=scale_clustering,
        diag_out=diag_out,
    )
    if cluster_labels is None or valid is None:
        scores = (-global_rank).reindex(adata.var_names).fillna(-np.inf)
        return _top_genes_from_rank(global_rank, n_top_genes), cluster_labels, 0, {}, scores

    ad_full = _restore_raw_counts(adata, layer=counts_layer, full_genes=True)
    if counts_layer not in ad_full.layers:
        ad_full.layers[counts_layer] = ad_full.X.copy()
    # align labels to ad_full
    labels_full = cluster_labels.reindex(ad_full.obs_names)
    if labels_full.isna().any():
        # fill missing with a dummy excluded label
        labels_full = labels_full.fillna("__missing__")

    # recompute valid sizes on full object
    sizes = labels_full.value_counts()
    # "__missing__" is an alignment placeholder, not a population. Without this it
    # became a full-fledged cluster whenever enough labels failed to align, and
    # was then reweighted alongside real biology.
    sizes = sizes.drop(labels="__missing__", errors="ignore")
    valid = sizes[sizes >= min_cluster_size]
    if valid.empty:
        scores = (-global_rank).reindex(adata.var_names).fillna(-np.inf)
        return _top_genes_from_rank(global_rank, n_top_genes), cluster_labels, 0, {}, scores

    weight_map = {str(k): float(v) for k, v in _cluster_size_weights(valid, balance_power).items()}
    cell_w = _cell_weights(labels_full, valid, balance_power=balance_power)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(ad_full.n_obs, size=ad_full.n_obs, replace=True, p=cell_w)
    # Resampling with replacement duplicates barcodes. Assign unique names
    # before .copy() would still warn on the view; set immediately after copy.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Observation names are not unique",
            category=UserWarning,
        )
        ad_rs = ad_full[idx].copy()
    ad_rs.obs_names = pd.Index([f"scfair_rs_{i}" for i in range(ad_rs.n_obs)])
    ad_rs.obs_names_make_unique()
    # ensure counts layer present for flavor
    if counts_layer not in ad_rs.layers:
        ad_rs.layers[counts_layer] = ad_rs.X.copy()
    else:
        ad_rs.layers[counts_layer] = ad_rs.layers[counts_layer].copy()

    # Drop batch_key if it became unusable after resampling (rare)
    bk = batch_key
    if bk is not None and bk not in ad_rs.obs.columns:
        bk = None

    _run_hvg(
        ad_rs,
        n_top_genes=min(n_top_genes, ad_rs.n_vars),
        flavor=flavor,
        counts_layer=counts_layer,
        span=span,
        n_bins=n_bins,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        max_disp=max_disp,
        batch_key=bk,
    )

    scores_rs = _variability_raw_scores(ad_rs, flavor=flavor, flavor_used=flavor)
    # align to parent genes
    S_aligned = scores_rs.reindex(adata.var_names).fillna(
        float(scores_rs.min()) if len(scores_rs) else 0.0
    )
    selected = _top_genes_from_scores(S_aligned, n_top_genes)

    logger.info(
        "reweight: resampled %d cells (β=%.3f, %d valid clusters); "
        "global HVG flavor=%r → %d genes.",
        ad_rs.n_obs,
        balance_power,
        len(valid),
        flavor,
        len(selected),
    )
    return selected, cluster_labels, int(len(valid)), weight_map, S_aligned


# ---------------------------------------------------------------------------
# markers / write-back
# ---------------------------------------------------------------------------


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
        logger.warning(
            "marker_genes not in adata.var_names (ignored): %s",
            missing[:10] + (["..."] if len(missing) > 10 else []),
        )
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
    cluster_labels: pd.Series | None,
    meta: dict[str, Any],
    prior_scanpy_hvg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    hv = adata.var_names.astype(str).isin(selected)
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

    if cluster_labels is not None:
        adata.obs["scfair_hvg_clusters"] = cluster_labels.astype("category")

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
