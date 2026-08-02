"""scFair — fair highly variable gene selection for single-cell RNA-seq.

Public usage::

    import scfair as scf
    scf.pp.highly_variable_genes(adata)  # default: n_top_genes=2000 + append
    # structure-aware k (opt-in; multi-seed, slower):
    # scf.pp.highly_variable_genes(adata, n_top_genes="auto")
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
