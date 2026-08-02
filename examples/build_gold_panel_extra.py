#!/usr/bin/env python
"""Convert newly-downloaded gold/near-gold datasets into scFair-ready h5ad.

Sourced 2026-08-01 while expanding the validation panel beyond PBMC:

  Villani et al. 2017 (GSE94820) -- human blood DC/monocyte subsets,
    Smart-seq2, FACS **sort-gate embedded directly in the cell barcode**
    (discovery set only; "deeper characterization" set skipped -- its
    columns are expression-binned within one gate, not a clean sort label).
    TRUE gold: label == FACS gate, not a clustering call.

  Haber et al. 2017 (GSE92332) -- mouse small intestine epithelium, 10x,
    EpCAM+ FACS gate (broad, gold) then published fine subtypes embedded
    in the cell barcode (Stem/TA/Enterocyte.*/Goblet/Paneth/Tuft/Endocrine).
    Fine labels are the paper's cluster+marker calls, not a second sort --
    same caveat class as the existing SHADOWED tier (§15 DEVELOPMENT_LOG).

  Tabula Muris FACS (figshare 5829687) -- Smart-seq2, 384-well, mouse.
    Checked `metadata_FACS.csv` per-plate `FACS.selection`:
      - Limb_Muscle: 4 plates with a real 4-way sort gate (CD31+ / CD45+ /
        CD31-CD45-Sca1+ / CD31-CD45-Sca1-VCAM1+) -- true gold, small n.
      - Brain_Myeloid (all "Microglia") vs Brain_Non-Myeloid (all "Neurons")
        -- true gold at the 2-way lineage split (combine the two files).
      - Lung / Kidney / Marrow / Spleen / Thymus: FACS.selection is
        "Multiple" or "Viable" for every plate -- the public
        `cell_ontology_class` there is a clustering call, not a sort
        label. Kept anyway (user: download now, decide later) but tagged
        SHADOWED in the manifest -- do not use as a "gold" claim.

Outputs (examples/data/):
  villani_dc_mono_gold.h5ad          (gold)
  haber_intestine_atlas.h5ad         (mixed: broad gold / fine SHADOWED)
  tm_limb_muscle_gold.h5ad           (gold)
  tm_brain_myeloid_vs_nonmyeloid_gold.h5ad  (gold)
  tm_lung_shadowed.h5ad              (SHADOWED)
  tm_kidney_shadowed.h5ad            (SHADOWED)
  tm_marrow_shadowed.h5ad            (SHADOWED)
  tm_spleen_shadowed.h5ad            (SHADOWED)
  tm_thymus_shadowed.h5ad            (SHADOWED)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "data" / "_staging" / "geo_gold"
OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)


def _finalize(a: ad.AnnData, min_genes: int = 200) -> ad.AnnData:
    a.X = sp.csr_matrix(a.X)
    a.layers["counts"] = a.X.copy()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=min_genes)
    sc.pp.filter_genes(a, min_cells=3)
    return a


def build_villani() -> None:
    print("villani: reading discovery set...", flush=True)
    df = pd.read_csv(STAGE / "villani_discovery.txt.gz", sep="\t", index_col=0)
    cell_type = [re.sub(r"_(P\d+_S\d+|S\d+)$", "", c) for c in df.columns]
    a = ad.AnnData(X=df.T.values.astype(np.float32))
    a.obs_names = df.columns.astype(str)
    a.var_names = df.index.astype(str)
    a.obs["cell_type"] = pd.Categorical(cell_type)
    a.obs["tier"] = "gold_facs_sort"
    a = _finalize(a, min_genes=200)
    print(f"  -> {a.shape}, types: {a.obs['cell_type'].value_counts().to_dict()}")
    a.write_h5ad(OUT / "villani_dc_mono_gold.h5ad")


def build_haber() -> None:
    print("haber: reading atlas UMI counts...", flush=True)
    df = pd.read_csv(STAGE / "haber_atlas.txt.gz", sep="\t", index_col=0)
    cell_type = [c.split("_", 2)[-1] for c in df.columns]
    a = ad.AnnData(X=df.T.values.astype(np.float32))
    a.obs_names = df.columns.astype(str)
    a.var_names = df.index.astype(str)
    a.obs["cell_type"] = pd.Categorical(cell_type)
    a.obs["tier"] = "mixed_broad_gold_fine_shadowed"
    a = _finalize(a, min_genes=200)
    print(f"  -> {a.shape}, {a.obs['cell_type'].nunique()} fine types")
    a.write_h5ad(OUT / "haber_intestine_atlas.h5ad")


def _read_tm_tissue(name: str) -> pd.DataFrame:
    path = STAGE / f"{name}-counts.csv"
    df = pd.read_csv(path, index_col=0)
    return df


def build_tm_limb_muscle() -> None:
    print("tabula muris: Limb_Muscle (true 4-way sort gate)...", flush=True)
    md = pd.read_csv(STAGE / "metadata_FACS.csv")
    gold_plates = set(
        md.loc[
            (md.tissue == "Limb_Muscle")
            & (md["FACS.selection"].astype(str).str.startswith("CD31")),
            "plate.barcode",
        ]
    )
    ann = pd.read_csv(STAGE / "annotations_FACS.csv")
    ann = ann[ann.tissue == "Limb_Muscle"].set_index("cell")
    df = _read_tm_tissue("Limb_Muscle")
    plate_of = {c: c.split(".")[1] if "." in c else "" for c in df.columns}
    keep = [c for c in df.columns if plate_of.get(c) in gold_plates]
    df = df[keep]
    labels = ann.reindex(df.columns)["cell_ontology_class"]
    a = ad.AnnData(X=df.T.values.astype(np.float32))
    a.obs_names = df.columns.astype(str)
    a.var_names = df.index.astype(str)
    a.obs["cell_type"] = pd.Categorical(labels.astype(str).values)
    a.obs["tier"] = "gold_facs_sort"
    a = _finalize(a, min_genes=200)
    print(f"  -> {a.shape}, types: {a.obs['cell_type'].value_counts().to_dict()}")
    a.write_h5ad(OUT / "tm_limb_muscle_gold.h5ad")


def build_tm_brain() -> None:
    print("tabula muris: Brain_Myeloid vs Brain_Non-Myeloid (2-way sort gate)...", flush=True)
    dfs = []
    for tissue, label in [
        ("Brain_Myeloid", "microglia_CD45_gate"),
        ("Brain_Non-Myeloid", "neuron_gate"),
    ]:
        df = _read_tm_tissue(tissue)
        sub = pd.DataFrame(index=df.index, columns=df.columns)
        sub[:] = df.values
        dfs.append((df, label))
    df_all = pd.concat([d for d, _ in dfs], axis=1)
    labels = pd.Series(
        [lbl for d, lbl in dfs for _ in d.columns],
        index=[c for d, _ in dfs for c in d.columns],
    )
    a = ad.AnnData(X=df_all.T.values.astype(np.float32))
    a.obs_names = df_all.columns.astype(str)
    a.var_names = df_all.index.astype(str)
    a.obs["cell_type"] = pd.Categorical(labels.reindex(a.obs_names).values)
    a.obs["tier"] = "gold_facs_sort"
    a = _finalize(a, min_genes=200)
    print(f"  -> {a.shape}, types: {a.obs['cell_type'].value_counts().to_dict()}")
    a.write_h5ad(OUT / "tm_brain_myeloid_vs_nonmyeloid_gold.h5ad")


def build_tm_shadowed(tissue: str, out_name: str) -> None:
    print(f"tabula muris: {tissue} (SHADOWED -- cluster-derived labels)...", flush=True)
    ann = pd.read_csv(STAGE / "annotations_FACS.csv")
    ann = ann[ann.tissue == tissue].set_index("cell")
    df = _read_tm_tissue(tissue)
    labels = ann.reindex(df.columns)["cell_ontology_class"]
    keep = labels.notna() & (labels != "unknown")
    df = df.loc[:, keep.values]
    labels = labels[keep]
    a = ad.AnnData(X=df.T.values.astype(np.float32))
    a.obs_names = df.columns.astype(str)
    a.var_names = df.index.astype(str)
    a.obs["cell_type"] = pd.Categorical(labels.astype(str).values)
    a.obs["tier"] = "SHADOWED"
    a = _finalize(a, min_genes=200)
    print(f"  -> {a.shape}, {a.obs['cell_type'].nunique()} types (cluster-derived)")
    a.write_h5ad(OUT / out_name)


if __name__ == "__main__":
    build_villani()
    build_haber()
    build_tm_limb_muscle()
    build_tm_brain()
    for tissue, out in [
        ("Lung", "tm_lung_shadowed.h5ad"),
        ("Kidney", "tm_kidney_shadowed.h5ad"),
        ("Marrow", "tm_marrow_shadowed.h5ad"),
        ("Spleen", "tm_spleen_shadowed.h5ad"),
        ("Thymus", "tm_thymus_shadowed.h5ad"),
    ]:
        build_tm_shadowed(tissue, out)
    print("done.")
