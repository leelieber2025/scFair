#!/usr/bin/env python
"""Does `resolution="auto"` survive contact with real data? Bug hunt, not benchmark.

One default call per dataset, chosen to span the situations that break things
rather than the ones that score well:

  size          2.4k -> 30.7k cells
  gene space    3.4k genes (paul15, where n_top_genes=2000 is 58% of the matrix)
  species       human and mouse
  tissue        PBMC, pancreas, lung-adjacent lymphoid, bone marrow
  labels        irrelevant here -- nothing is scored, only run

What is checked per dataset: the call completes, the density field produces a
count, the bisection reaches it (or records that it could not), and the
intermediate clustering does not collapse to <2 clusters -- the failure that
turns scFair silently into scanpy.

Outputs (examples/results/):
  umap3d_smoke.csv
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from adt_gold_benchmark import load_labeled as load_adt14  # noqa: E402
from adt_multi_validation import load_cite  # noqa: E402
from p3_public_validation import (  # noqa: E402
    load_paul15,
    load_pbmc3k_labeled,
    load_scib_pancreas_one_tech,
)

import scfair as scf  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "umap3d_smoke.csv"


def _plain(path: str, counts_from_X: bool = True):
    """Loader for sets that need no label construction -- this is a smoke test."""

    def load():
        a = ad.read_h5ad(DATA / path)
        a.obs_names_make_unique()
        a.var_names_make_unique()
        if "counts" not in a.layers and counts_from_X:
            a.layers["counts"] = a.X.copy()
        sc.pp.filter_genes(a, min_cells=3)
        return a

    return load


LOADERS = {
    # small / few genes
    "paul15": load_paul15,  # 2.7k cells, 3.4k genes
    "pancreas_smartseq2": lambda: load_scib_pancreas_one_tech("smartseq2"),
    "pbmc3k_louvain": load_pbmc3k_labeled,
    "crafted_base": _plain("crafted_base_3cellline_GSE136148.h5ad"),
    # sorted ground truth
    "duo4_pbmc": _plain("duo4_pbmc.h5ad"),
    "duo8_pbmc": _plain("duo8_pbmc.h5ad"),
    "duo4un_pbmc": _plain("duo4un_pbmc.h5ad"),
    # CITE-seq panel
    "pbmc5k_adt29": lambda: load_cite("pbmc_5k_v3"),
    "pbmc10k_adt14": load_adt14,
    "sln_208_mouse": lambda: load_cite("sln_208_mouse"),  # mouse
    "pbmc_seurat_v4_20k": lambda: load_cite("pbmc_seurat_v4_20k"),
    # largest, never benchmarked by this project
}
ORDER = list(LOADERS)


def run_one(name: str) -> dict:
    row: dict = {"dataset": name}
    t0 = time.time()
    a = LOADERS[name]()
    row.update(n_cells=int(a.n_obs), n_genes=int(a.n_vars), load_s=round(time.time() - t0, 1))
    print(f"  {a.n_obs} cells x {a.n_vars} genes  (load {row['load_s']}s)", flush=True)

    t1 = time.time()
    scf.pp.highly_variable_genes(a)  # pure defaults
    row["run_s"] = round(time.time() - t1, 1)

    h = a.uns["scfair"]["hvg"]
    c = h["clustering"]
    g = c.get("granularity") or {}
    row.update(
        n_selected=int(a.var["highly_variable"].sum()),
        n_populations=g.get("n_populations"),
        granularity_reason=g.get("reason"),
        bandwidth=g.get("bandwidth"),
        resolution_source=c.get("resolution_source"),
        target=c.get("n_populations_target"),
        achieved=c.get("n_clusters_achieved"),
        leiden_calls=c.get("n_leiden_calls"),
        resolution=round(float(c["resolution"]), 4),
        n_clusters_total=c["n_clusters_total"],
        n_clusters_kept=c["n_clusters_kept"],
        n_clusters_used=h["n_clusters_used"],
        n_passes=c["n_passes"],
    )

    problems = []
    if row["n_clusters_kept"] < 2:
        problems.append("COLLAPSED_TO_ONE_CLUSTER")
    if row["n_selected"] != min(2000, a.n_vars):
        problems.append(f"selected {row['n_selected']}")
    if row["target"] is not None and row["achieved"] != row["target"]:
        problems.append(f"target {row['target']} != achieved {row['achieved']}")
    row["problems"] = ";".join(problems)
    return row


def main(which=None) -> None:
    rows: list[dict] = []
    for name in which or ORDER:
        print(f"\n######## {name} ########", flush=True)
        try:
            row = run_one(name)
        except Exception as e:  # noqa: BLE001 - this is the hunt
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            row = {"dataset": name, "problems": f"EXCEPTION {type(e).__name__}: {e}"}
        rows.append(row)
        print(
            "  "
            + "  ".join(
                f"{k}={row[k]}"
                for k in (
                    "n_populations",
                    "resolution",
                    "achieved",
                    "n_clusters_kept",
                    "leiden_calls",
                    "run_s",
                )
                if k in row
            ),
            flush=True,
        )
        if row.get("problems"):
            print(f"  >>> {row['problems']}", flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    d = pd.DataFrame(rows)
    print("\n=== summary ===", flush=True)
    cols = [
        c
        for c in (
            "dataset",
            "n_cells",
            "n_populations",
            "resolution",
            "n_clusters_kept",
            "run_s",
            "problems",
        )
        if c in d
    ]
    print(d[cols].to_string(index=False), flush=True)
    bad = d[d["problems"].astype(str).str.len() > 0] if "problems" in d else d.iloc[:0]
    print(f"\n{len(d) - len(bad)}/{len(d)} clean", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
