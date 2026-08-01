"""Preprocessing for scFair.

Public API
----------
- :func:`highly_variable_genes` — main user-facing entry point.
- :func:`diagnose_from_labels` — pre-call imbalance / suitability tips from
  known cell labels (does not run HVG).
- :func:`estimate_n_populations` — how many populations the data's density
  field supports, read off a 3D embedding (advisory; does not run HVG).

Raw-count snapshot / restore helpers live in :mod:`scfair.pp._raw_counts` and
are **private**; they run automatically inside ``highly_variable_genes``.
Post-call diagnosis is written to ``adata.uns['scfair']['hvg']['diagnosis']``
when ``diagnose=True`` (default).
"""

from ._auto_n import estimate_n_top_structure, select_n_top_from_structure
from ._diagnosis import diagnose_from_labels
from ._granularity import estimate_n_populations
from ._highly_variable_genes import highly_variable_genes

__all__ = [
    "highly_variable_genes",
    "diagnose_from_labels",
    "estimate_n_populations",
    "estimate_n_top_structure",
    "select_n_top_from_structure",
]
