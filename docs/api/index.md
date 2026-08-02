# API Reference

```python
import scfair as scf
```

Public surface: the four names below, plus `scfair.__version__`. Everything
else in `scfair.pp` (resolution/mode helpers, the structure-based `k`
estimator, raw-count snapshotting, ...) is an internal implementation detail
used by `highly_variable_genes` and is not documented here. Narrative usage:
{doc}`../user_guide/index`.

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
```

## Package

```{eval-rst}
.. currentmodule:: scfair

.. autosummary::
   :toctree: generated/
   :nosignatures:

   __version__
```
