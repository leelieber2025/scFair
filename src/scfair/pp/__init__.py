"""Preprocessing for scFair.

Public API
----------
- :func:`highly_variable_genes` — main user-facing entry point.
- :class:`HVGOptions` — secondary knobs (bounds, filters, store_raw, …).
- :func:`diagnose_from_labels` — pre-call imbalance tips from known labels.
- :func:`estimate_n_populations` — label-free density population count.
- :func:`restore_raw_counts` — restore counts after ``store_raw=True`` / subset.

Everything else under :mod:`scfair.pp` is an implementation detail.
"""

from ._diagnosis import diagnose_from_labels
from ._granularity import estimate_n_populations
from ._highly_variable_genes import highly_variable_genes
from ._options import HVGOptions
from ._raw_counts import restore_raw_counts

__all__ = [
    "highly_variable_genes",
    "HVGOptions",
    "diagnose_from_labels",
    "estimate_n_populations",
    "restore_raw_counts",
]
