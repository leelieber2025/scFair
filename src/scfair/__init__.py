"""scFair: structure-aware HVG list size and same-rank append at the cutoff.

Public API:

1. **How many genes (``n``)?** Default ``n_top_genes="auto"`` estimates a base
   size from multi-seed density structure. Pass a fixed int (e.g. ``2000``)
   when a protocol already locks the length. Auto is multi-seed and slower
   than a fixed ``k``.
2. **Hard top-``k`` cutoff.** Global ranking is driven by large populations;
   markers for smaller types often sit just below the cut. Default ``append``
   keeps the global top-``k`` and adds a short same-rank tail
   (``top-(k+m)``). Use ``balance_method="none"`` for pure top-``k``.
3. **How many populations?** :func:`~scfair.pp.estimate_n_populations` reads
   a 3D density field so you need not sweep Leiden resolution (requires
   ``sc.pp.neighbors`` first). Works best on well-separated data up to about
   20 populations; can under-count many tiny groups.

Usage::

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
    HVGOptions,
    diagnose_from_labels,
    estimate_n_populations,
    highly_variable_genes,
    restore_raw_counts,
)

__all__ = [
    "__version__",
    "pp",
    "highly_variable_genes",
    "HVGOptions",
    "diagnose_from_labels",
    "estimate_n_populations",
    "restore_raw_counts",
]
