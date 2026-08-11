# API Reference

```python
import scfair as scf
```

Public surface: the names below, plus `scfair.__version__`. Everything else
in `scfair.pp` is an implementation detail used by `highly_variable_genes`
and is not listed here.

Signatures and parameter text are generated from the source docstrings at
build time. Defaults are `n_top_genes="auto"` (structure-aware base size) and
`balance_method="append"` (same-rank tail; set equals `top-(k+m)`, not
cluster-conditional reallocation). For usage narrative, see
{doc}`../user_guide/method` or {doc}`../quickstart`.

## Preprocessing (`scfair.pp`)

```{eval-rst}
.. currentmodule:: scfair.pp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   highly_variable_genes
   HVGOptions
   diagnose_from_labels
   estimate_n_populations
   restore_raw_counts
```

## Package

```{eval-rst}
.. currentmodule:: scfair

.. autosummary::
   :toctree: generated/
   :nosignatures:

   __version__
```
