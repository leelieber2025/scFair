"""Secondary knobs for :func:`~scfair.pp.highly_variable_genes`.

The public HVG signature stays short. Bounds, filters, and related options
live here and are passed as ``options=HVGOptions(...)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import pandas as pd

# Keys that used to be top-level kwargs on highly_variable_genes.
# Still accepted with DeprecationWarning; prefer HVGOptions.
# Removed research fields raise a clear TypeError via unknown-key handling.
HVG_OPTION_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "n_top_min",
        "n_top_max",
        "structure_n_seeds",
        "marker_extra",
        "global_score",
        "span",
        "n_bins",
        "min_mean",
        "max_mean",
        "min_disp",
        "max_disp",
        "batch_key",
        "filter_mito",
        "filter_ribo",
        "gene_nomenclature",
        "append_budget",
        "label_key",
        "store_raw",
        "snapshot_path",
    }
)

# Names removed with the cluster-aware balance methods / dead auto strategies.
_REMOVED_OPTION_NAMES: frozenset[str] = frozenset(
    {
        "auto_n_method",
        "balance_power",
        "neighbor_contrast",
        "combine",
        "cluster_pool",
        "cluster_genes",
        "consensus_resolutions",
        "allocation_method",
        "unit_merge_threshold",
        "spec_on_legitimate_units",
        "scale_clustering",
        "logfc_space",
        "n_pcs",
        "n_neighbors",
        "cap_allocation",
        "cap_ceiling",
        "cap_merge_threshold",
        "blend_global",
    }
)


@dataclass
class HVGOptions:
    """Secondary controls for HVG selection.

    Pass only fields you need to change::

        scf.pp.highly_variable_genes(
            adata,
            options=scf.pp.HVGOptions(append_budget=100, filter_mito=False),
        )
    """

    # auto bounds / structure seeds
    n_top_min: int = 500
    n_top_max: int = 5000
    # Structure auto multi-seed count. None → product default (3). Use 1 for
    # a faster exploratory pass (less stable k).
    structure_n_seeds: int | None = None

    # Optional obs key for auto mode / fine detection (n_types).
    label_key: str | None = None

    # markers
    marker_extra: bool = True

    # Extra genes beyond base ``n_top_genes`` / auto-``k`` from the **same**
    # global ranking (ranks ``k+1 … k+m``). The final set is mathematically
    # ``top-(k+m)`` — a list-length buffer, not population-aware reallocation.
    # None → product default floor 200; on ``n_top_genes="auto"`` may rise with
    # structure density cores. Explicit ``0``/``N`` is never overridden.
    append_budget: int | None = None

    # Opt-in: keep a full raw-count sidecar in ``uns['scfair']['raw_snapshot']``
    # after HVG. Restore with :func:`~scfair.pp.restore_raw_counts`. Default False.
    store_raw: bool | str = False
    snapshot_path: str | None = None

    global_score: pd.Series | None = None

    # scanpy HVG knobs (global ranking only)
    span: float = 0.3
    n_bins: int = 20
    min_mean: float = 0.0125
    max_mean: float = 3.0
    min_disp: float = 0.5
    max_disp: float = float("inf")
    # obs column for scanpy per-batch HVG merge. Selection then follows
    # scanpy's ``highly_variable_rank`` (nbatches + median rank), not the
    # mean of per-batch score columns.
    batch_key: str | None = None
    # Drop MT / ribosomal structural-protein symbols from the final HVG set
    # (refill from global rank). Markers are never filtered.
    # Default False: GOLD-style clustering metrics prefer keeping the global
    # rank as-is; set True for cleaner marker panels / QC-oriented lists.
    filter_mito: bool = False
    filter_ribo: bool = False
    # None → infer human/mouse/mixed/unknown from var_names.
    gene_nomenclature: str | None = None

    def merged(self, **overrides: Any) -> HVGOptions:
        """Return a copy with non-None overrides applied (``None`` skips)."""
        data = asdict(self)
        for k, v in overrides.items():
            if k not in data:
                raise TypeError(f"unknown HVGOptions field: {k!r}")
            if v is None:
                continue
            data[k] = v
        return HVGOptions(**data)


def resolve_hvg_options(
    options: HVGOptions | None,
    legacy_kwargs: Mapping[str, Any] | None = None,
) -> HVGOptions:
    """Merge explicit ``options`` with deprecated top-level kwargs.

    ``options`` must be ``None`` or an :class:`HVGOptions` instance (not a bare
    dict). ``options=`` and legacy top-level kwargs must not be mixed: if both
    are present, raise ``ValueError`` (dataclass fields have no "was set"
    sentinel, so per-field conflict detection is not possible).
    """
    if options is None:
        base = HVGOptions()
        options_provided = False
    elif isinstance(options, HVGOptions):
        base = options
        options_provided = True
    elif isinstance(options, Mapping):
        raise TypeError(
            "options must be an HVGOptions instance, not a dict/mapping. "
            "Use options=HVGOptions(append_budget=200) or "
            "options=HVGOptions(**your_dict)."
        )
    else:
        raise TypeError(
            f"options must be HVGOptions or None, got {type(options).__name__}. "
            "Use options=HVGOptions(...)."
        )

    if not legacy_kwargs:
        return base

    overrides = dict(legacy_kwargs)
    removed_hit = sorted(k for k in overrides if k in _REMOVED_OPTION_NAMES)
    if removed_hit:
        raise TypeError(
            f"removed option(s): {removed_hit}. Cluster-aware balance methods "
            "(hybrid/score/reweight), auto_n_method, and related knobs were "
            "deleted. Product auto is structure-only; use balance_method="
            "'append' (list buffer) or 'none' (top-k only)."
        )

    field_names = {f.name for f in fields(HVGOptions)}
    unknown = set(overrides) - field_names
    if unknown:
        raise TypeError(
            f"unknown highly_variable_genes option(s): {sorted(unknown)}. "
            f"Use HVGOptions fields or the short public signature."
        )

    if options_provided:
        conflict = sorted(k for k in overrides if k in field_names)
        if conflict:
            raise ValueError(
                "Do not mix options=HVGOptions(...) with deprecated top-level "
                f"kwargs. Put every secondary knob on the HVGOptions instance, "
                f"or pass only legacy kwargs (not both). Got legacy: {conflict}."
            )

    return base.merged(**overrides)
