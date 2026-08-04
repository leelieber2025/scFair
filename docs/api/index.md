# API Reference

```python
import scfair as scf
```

Public surface: the four names below, plus `scfair.__version__`. Everything
else in `scfair.pp` is an implementation detail used by
`highly_variable_genes` and is not listed here.

**How to read this page.** Signatures and parameter text are generated from
the source docstrings at build time (not hand-copied). Product defaults
include `n_top_genes="auto"` and `balance_method="append"`. For *why* those
defaults exist (list size `n`, fairer HVG allocation), start with
{doc}`../user_guide/method` or {doc}`../quickstart` — this page is the
reference, not the tutorial.

Narrative usage: {doc}`../user_guide/index`.

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
