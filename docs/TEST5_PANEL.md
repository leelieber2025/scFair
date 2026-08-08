# Test-5 panel (deployment holdout, frozen product)

- **Date built:** 2026-08-05
- **Role:** small **holdout** for product defaults (`auto` + `append` vs `auto` + `none` vs HVG@2000).  
  **Do not retune** structure / append rules on this panel.
- **Base set (panel18 / GOLD+MIXED):** unchanged; still the historical / primary tables.
- **Excluded:** `~/files/**` (lab-private and working copies).

## Selection criteria

| Axis | Choice |
|---|---|
| Labels | **Hard:** Tabula Sapiens consortium **Cell Ontology** `cell_type`, restricted to `manually_annotated=True` |
| Chemistry | **10x Chromium 3′ v3 only** (drop Smart-seq2 and 10x 5′ v2) |
| Structure | Real multi-organ human atlas (not FACS re-mix toys) |
| Diversity | Kidney, liver, heart, mammary, bone marrow — none of these organs are the bulk of panel18 FACS-PBMC toys |
| Size | ≤20 000 cells; types with &lt;10 cells dropped; `min_cells=3` genes |

**Chemistry honesty (2026):** this is **v3**, not GEM-X v4. GEM-X public sets with *expert* (not auto) labels were scarce; hard labels were prioritized over newest chemistry. Document as **modern-vs-base (v2 era)** rather than “bleeding-edge 2026 GEM-X.”

## Files (`examples/data/`)

| file | n_obs | n_vars | n_types | manual | assay |
|---|---:|---:|---:|:---:|---|
| `ts_kidney_10x_v3.h5ad` | 9181 | 32833 | 9 | yes | 10x 3′ v3 |
| `ts_liver_10x_v3.h5ad` | 9844 | 33900 | 21 | yes | 10x 3′ v3 |
| `ts_heart_10x_v3.h5ad` | 17609 | 38478 | 14 | yes | 10x 3′ v3 |
| `ts_mammary_10x_v3.h5ad` | 20000 | 40377 | 15 | yes | 10x 3′ v3 |
| `ts_bone_marrow_10x_v3.h5ad` | 6549 | 35760 | 18 | yes | 10x 3′ v3 |

Machine inventory: `examples/data/test5_inventory.csv`  
Build meta (per-type counts): `examples/results/test5_build_meta.json`

Each object has:

- `obs["cell_type"]` — evaluation labels  
- `layers["counts"]` and `X` — integer UMI counts (from CELLxGENE `.raw`)  
- `uns["scfair_test5"]` — provenance (assay filter, manual filter, citation)

## Source

- **Tabula Sapiens** via [CZ CELLxGENE](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5)  
- Citation: The Tabula Sapiens Consortium, *Science* **376**, eabl4896 (2022)  
- Staging downloads (gitignored large raw): `examples/data/_staging/test5/*.raw.h5ad`

## Suggested evaluation protocol

Same as panel18 product retest:

1. `hvg2000` — scanpy `seurat_v3` top-2000  
2. `auto_none` — `n_top_genes="auto"`, `balance_method="none"`  
3. `auto_append` — `n_top_genes="auto"`, `balance_method="append"` (product default)  
4. 2 seeds; `resolve_cluster_resolution(auto)`; ARI vs `cell_type`

**Decision rule (pre-registered):** freeze base rules; report median ΔARI and win/loss on Test-5 only for default choice; do not open new knobs if one arm loses a single organ.

## Why not the earlier candidates

| Dropped idea | Reason |
|---|---|
| Zilionis lung | **inDrops**, not 10x |
| GSE141526 BCC | **10x v2** |
| bmcite GSE128639 | **10x v2** CITE |
| 10x GEM-X official melanoma | chemistry new, labels often **automated** (softer) |
