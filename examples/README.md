# Examples

Product usage is documented on Read the Docs:

- [Quickstart](https://scfair.readthedocs.io/en/latest/quickstart.html)
- [Tutorial: PBMC 10k](https://scfair.readthedocs.io/en/latest/tutorials/index.html)

Benchmark and research scripts that live under this directory in a full
developer checkout are **not** part of the public release or the PyPI
source distribution. They depend on local datasets (see the development
docs) and are not required to install or run scFair.

### Archived: fixed-`k` multi-method HVG rank (2026-08)

Conclusion + paths: `results/mix_rank_vs_seurat_CONCLUSION.md`.

| Script | Role |
|--------|------|
| `test5_mix_rank_vs_seurat.py` | Test-5: auto `k`, seurat_v3 vs mean-rank mix |
| `panel18_mix_rank_vs_seurat.py` | Same protocol on panel-18 |

**Verdict (do not re-open without new design):** ensemble rank at fixed auto `k` is not a robust default vs single seurat_v3.
