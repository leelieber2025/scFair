# API Reference

```python
import scfair as scf
```

Public surface: the names below, plus `scfair.__version__`. Everything else
in `scfair.pp` is an implementation detail.

Signatures come from the source docstrings. Defaults are
`n_top_genes="auto"` and `balance_method="append"` (set equals
`top-(k+m)`). Usage: {doc}`../quickstart`, {doc}`../user_guide/method`.

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
