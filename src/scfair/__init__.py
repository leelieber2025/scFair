"""scFair — fair highly variable gene selection for single-cell RNA-seq.

Addresses two everyday problems:

1. **How many genes (``n``)?** Default ``n_top_genes="auto"`` suggests a base
   size from the data — a safe default / prompt, not a claim of global
   optimality. Pass an int (e.g. 2000) for fixed protocols.
2. **Unfair HVG allocation.** Large populations dominate a plain top-``k``
   list. Default ``append`` freezes that backbone and adds near-miss genes.

Public usage::

    import scfair as scf
    scf.pp.highly_variable_genes(adata)  # default: auto n + append
    # fixed classical size (faster / paper protocol):
    # scf.pp.highly_variable_genes(adata, n_top_genes=2000)
    # optional pre-call planning from known labels:
    scf.pp.diagnose_from_labels(adata.obs["cell_type"])
    # optional, label-free: how many populations the data supports
    scf.pp.estimate_n_populations(adata)
"""

from . import pp
from ._version import __version__
from .pp import diagnose_from_labels, estimate_n_populations, highly_variable_genes

__all__ = [
    "__version__",
    "pp",
    "highly_variable_genes",
    "diagnose_from_labels",
    "estimate_n_populations",
]
