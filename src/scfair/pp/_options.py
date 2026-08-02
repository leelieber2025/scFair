"""Optional / experimental knobs for :func:`~scfair.pp.highly_variable_genes`.

The public HVG signature stays short. Research and rarely-used parameters live
here and are passed as ``options=HVGOptions(...)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import pandas as pd

# Keys that used to be top-level kwargs on highly_variable_genes.
# Still accepted with DeprecationWarning; prefer HVGOptions.
HVG_OPTION_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "n_top_min",
        "n_top_max",
        "auto_n_method",
        "marker_extra",
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
        "global_score",
        "n_pcs",
        "n_neighbors",
        "span",
        "n_bins",
        "min_mean",
        "max_mean",
        "min_disp",
        "max_disp",
        "batch_key",
        "filter_mito",
        "filter_ribo",
        "append_budget",
        "label_key",
        "store_raw",
        "snapshot_path",
        # removed public names still intercepted for a clear error:
        "cap_allocation",
        "cap_ceiling",
        "cap_merge_threshold",
    }
)


@dataclass
class HVGOptions:
    """Experimental and secondary controls for HVG selection.

    Product defaults match the previous top-level defaults. Pass only fields
    you need to change::

        scf.pp.highly_variable_genes(
            adata,
            options=scf.pp.HVGOptions(cluster_pool=2000, neighbor_contrast=1.0),
        )
    """

    # auto bounds / strategy (also common; kept here to slim the main signature)
    n_top_min: int = 500
    n_top_max: int = 5000
    auto_n_method: str = "structure"

    # Optional obs key for auto mode / fine detection (n_types).
    label_key: str | None = None

    # markers
    marker_extra: bool = True

    # ranking / contrast
    balance_power: float | None = None
    neighbor_contrast: float = 0.0
    combine: str = "blend"

    # intermediate clustering gene set
    cluster_pool: int | None = None
    cluster_genes: Sequence[str] | None = None
    consensus_resolutions: Sequence[float] | None = None

    # post-hybrid allocation research arms (no "cap")
    allocation_method: str | None = None  # None → "none"
    # shared merge threshold for coverage / starved_topup unit stability
    unit_merge_threshold: float | None = 0.5
    spec_on_legitimate_units: bool = False

    # balance_method="append": extra genes beyond the frozen global top-k base
    # from the same global ranking (ranks k+1 … k+m). No intermediate clustering.
    # None → product mode chooses (200). Explicit 0/N is never overridden by mode.
    append_budget: int | None = None

    # Snapshot raw counts into uns['scfair']['raw_snapshot'].
    # False (default): only ensure layers['counts'] for HVG; no second full copy
    # in uns (was ~3× h5ad bloat). True = inline; "ondisk" needs snapshot_path.
    store_raw: bool | str = False
    snapshot_path: str | None = None

    scale_clustering: bool = True
    logfc_space: str = "log1p"

    global_score: pd.Series | None = None

    # scanpy HVG / graph knobs
    n_pcs: int = 30
    n_neighbors: int = 15
    span: float = 0.3
    n_bins: int = 20
    min_mean: float = 0.0125
    max_mean: float = 3.0
    min_disp: float = 0.5
    max_disp: float = float("inf")
    batch_key: str | None = None
    filter_mito: bool = False
    filter_ribo: bool = False

    def merged(self, **overrides: Any) -> HVGOptions:
        """Return a copy with non-None overrides applied."""
        data = asdict(self)
        for k, v in overrides.items():
            if k not in data:
                raise TypeError(f"unknown HVGOptions field: {k!r}")
            data[k] = v
        return HVGOptions(**data)


def resolve_hvg_options(
    options: HVGOptions | None,
    legacy_kwargs: Mapping[str, Any] | None = None,
) -> HVGOptions:
    """Merge explicit ``options`` with deprecated top-level kwargs.

    ``options`` must be ``None`` or an :class:`HVGOptions` instance (not a bare
    dict). When both ``options=`` and a legacy top-level research kwarg set the
    same field, raise ``ValueError`` — legacy must not silently override the
    new API during migration.
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
            "Use options=HVGOptions(cluster_pool=200) or "
            "options=HVGOptions(**your_dict)."
        )
    else:
        raise TypeError(
            f"options must be HVGOptions or None, got {type(options).__name__}. "
            "Use options=HVGOptions(...)."
        )

    if base.allocation_method is not None and str(base.allocation_method).lower() == "cap":
        raise ValueError(
            "allocation_method='cap' was removed in 0.2.0. Use 'none' (default) "
            "or research 'starved_topup' / 'coverage'."
        )
    if not legacy_kwargs:
        return base

    # Explicit removal of cap API
    if legacy_kwargs.get("cap_allocation") or legacy_kwargs.get("allocation_method") == "cap":
        raise ValueError(
            "cap_allocation / allocation_method='cap' was removed in 0.2.0. "
            "Use allocation_method='none' (default) or research "
            "options=HVGOptions(allocation_method='starved_topup'|'coverage')."
        )
    if "cap_ceiling" in legacy_kwargs:
        raise ValueError("cap_ceiling was removed with allocation_method='cap' in 0.2.0.")

    # Rename old threshold name
    overrides = dict(legacy_kwargs)
    if "cap_merge_threshold" in overrides:
        if "unit_merge_threshold" not in overrides:
            overrides["unit_merge_threshold"] = overrides.pop("cap_merge_threshold")
        else:
            overrides.pop("cap_merge_threshold")

    # Drop keys that are not fields (already handled)
    for dead in ("cap_allocation", "cap_ceiling"):
        overrides.pop(dead, None)

    field_names = {f.name for f in fields(HVGOptions)}
    unknown = set(overrides) - field_names
    if unknown:
        raise TypeError(
            f"unknown highly_variable_genes option(s): {sorted(unknown)}. "
            f"Use HVGOptions fields or the short public signature."
        )

    # Conflict: options=... plus the same research field as a top-level kwarg.
    # Legacy used to win silently; that makes migrations wrong without a clear error.
    if options_provided:
        conflict = sorted(k for k in overrides if k in field_names)
        if conflict:
            raise ValueError(
                "Conflicting research knobs: pass each field either via "
                f"options=HVGOptions(...) or as a deprecated top-level kwarg, "
                f"not both. Conflicting: {conflict}. Remove the top-level "
                "kwargs (recommended) or drop options= for those fields."
            )

    return base.merged(**overrides)
