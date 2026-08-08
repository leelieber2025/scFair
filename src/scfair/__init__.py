"""scFair — structure-aware HVG list size and fairer handling of the HVG cutoff.

Main product value:

1. **How many genes (``n``)?** Nobody can predict a safe list length a priori.
   Default ``n_top_genes="auto"`` estimates a base size from density structure
   so unfamiliar users are less likely to ship a silent wrong ``n``. The
   multi-seed cost is intentional and worth it for that purpose. Pass a fixed
   int (e.g. ``2000``) only when a protocol is already locked.
2. **Unfair HVG allocation at the cutoff.** Global ranking is driven by large
   populations; markers for smaller types often sit just below a hard top-``k``
   cut. Default ``append`` freezes that backbone and adds a short **same-rank**
   tail (``top-(k+m)``) so near-miss genes are not discarded. This is a
   conservative response to cutoff unfairness — **not** cluster-conditional
   reallocation (``hybrid`` / ``score`` / ``reweight`` were removed). Use
   ``balance_method="none"`` for pure top-``k``.
3. **How many populations?** :func:`~scfair.pp.estimate_n_populations` reads
   a 3D density field so you need not sweep Leiden resolution (requires
   ``sc.pp.neighbors`` first). Reliable on well-separated data up to ~20
   populations; can under-count many tiny groups.

Public usage::

    import scfair as scf
    scf.pp.highly_variable_genes(adata)  # default: auto n + append
    # locked protocol / paper (skip structure auto):
    # scf.pp.highly_variable_genes(adata, n_top_genes=2000)
    # pure top-k (no append tail):
    # scf.pp.highly_variable_genes(adata, n_top_genes=2000, balance_method="none")
    # label-free population count (needs sc.pp.neighbors first):
    scf.pp.estimate_n_populations(adata)
    # optional pre-call planning from known labels:
    scf.pp.diagnose_from_labels(adata.obs["cell_type"])
"""

from . import pp
from ._version import __version__
from .pp import (
    diagnose_from_labels,
    estimate_n_populations,
    highly_variable_genes,
    restore_raw_counts,
)

__all__ = [
    "__version__",
    "pp",
    "highly_variable_genes",
    "diagnose_from_labels",
    "estimate_n_populations",
    "restore_raw_counts",
]
