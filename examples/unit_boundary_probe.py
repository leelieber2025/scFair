#!/usr/bin/env python
"""Research probe: legitimate units vs labelled types, and which boundaries
look *actionable* under default hybrid (allocation none).

Does **not** change gene selection. For each dataset:

1. Run hybrid @2000, resolution=0.5, allocation_method="none".
2. Build unit partition U = merge NN Leiden pairs that fail stability OR
   pairwise DE (same rules as experimental coverage units).
3. Score U and raw Leiden against ground-truth labels (ARI/NMI, fragment
   rate = labelled types split across ≥2 units with ≥min cells each).
4. For each NN pair in U, compute DE on full log space vs DE on selected
   genes only, plus how many high-DE genes remain in the hybrid 2× pool
   but outside the selection → ``actionable`` flag.

Outputs (examples/results/):
  unit_boundary_probe_units.csv     per-dataset unit vs label metrics
  unit_boundary_probe_bounds.csv    per-boundary actionable diagnostics

Usage:
  python examples/unit_boundary_probe.py
  python examples/unit_boundary_probe.py quick   # 4 datasets
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402
from scfair.pp._highly_variable_genes import (  # noqa: E402
    DEFAULT_CAP_MERGE_THRESHOLD,
    DEFAULT_DE_SUPPORT_THRESHOLD,
    _build_legitimate_units,
    _cluster_vs_rest_logfc,
    _logfc_inputs,
    _lognorm_matrix_from_counts,
    _nearest_cluster_map,
    _pair_bootstrap_stability,
    _pair_de_support,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
UNITS_CSV = OUT / "unit_boundary_probe_units.csv"
BOUNDS_CSV = OUT / "unit_boundary_probe_bounds.csv"

K = 2000
RESOLUTION = 0.5
MIN_CLUSTER_SIZE = 30
TOP_N = 20
# actionable: selected genes recover < this fraction of full-space DE support
GAP_FRAC = 0.5
# need at least this many unused high-DE genes still in the 2× pool
POOL_LEFT_MIN = 5

ORDER_FULL = [
    "paul15",
    "pbmc3k_louvain",
    "pancreas_smartseq2",
    "duo4_pbmc",
    "duo8_pbmc",
    "duo4un_pbmc",
    "pbmc5k_adt29",
    "pbmc10k_adt14",
    "sln_208_mouse",
    "pbmc_seurat_v4_20k",
]
ORDER_QUICK = ["pancreas_smartseq2", "duo4_pbmc", "duo8_pbmc", "pbmc3k_louvain"]


def _fragment_rate(y_true: pd.Series, y_part: pd.Series, min_cells: int = 20) -> float:
    """Fraction of labelled types that appear in ≥2 partition blocks (≥min_cells)."""
    n_frag = 0
    n_types = 0
    for lab, idx in y_true.groupby(y_true).groups.items():
        counts = y_part.loc[idx].value_counts()
        big = counts[counts >= min_cells]
        if big.sum() < min_cells:
            continue
        n_types += 1
        if len(big) >= 2:
            n_frag += 1
    return float(n_frag / n_types) if n_types else float("nan")


def _glue_rate(y_true: pd.Series, y_part: pd.Series, min_cells: int = 20) -> float:
    """Fraction of partition blocks whose majority label is <80% pure (among
    blocks with ≥min_cells). High = over-merging distinct types."""
    impure = 0
    n = 0
    for blk, idx in y_part.groupby(y_part).groups.items():
        if len(idx) < min_cells:
            continue
        n += 1
        top = y_true.loc[idx].value_counts(normalize=True).iloc[0]
        if top < 0.8:
            impure += 1
    return float(impure / n) if n else float("nan")


def _de_on_genes(X_log, mask_a, mask_b, gene_idx: np.ndarray | None, pseudo: float) -> float:
    """Pair DE support restricted to a gene subset (columns of X_log)."""
    if gene_idx is not None:
        # build a column-sliced view without copying the full matrix when dense
        if hasattr(X_log, "tocsc"):
            X = X_log.tocsc()[:, gene_idx]
        else:
            X = np.asarray(X_log)[:, gene_idx]
    else:
        X = X_log
    return _pair_de_support(X, mask_a, mask_b, top_n=TOP_N, pseudo=pseudo)


def probe_one(name: str, seed: int = 0) -> tuple[dict, list[dict]]:
    a = LOADERS[name]()
    ad = a.copy()
    scf.pp.highly_variable_genes(
        ad,
        n_top_genes=K,
        balance_method="hybrid",
        resolution=RESOLUTION,
        allocation_method="none",
        min_cluster_size=MIN_CLUSTER_SIZE,
        random_state=seed,
        diagnose=False,
    )
    labels = ad.obs["scfair_hvg_clusters"].astype(str)
    selected = list(ad.var_names[ad.var["highly_variable"]].astype(str))
    scores = ad.var["scfair_score"] if "scfair_score" in ad.var.columns else None

    # hybrid 2× pool ≈ top 2k by scfair global side; use score if present else HV rank
    if scores is not None:
        pool = list(scores.sort_values(ascending=False).index.astype(str)[: 2 * K])
    else:
        pool = selected  # fallback

    counts_layer = ad.uns["scfair"]["hvg"].get("counts_layer", "counts")
    if counts_layer not in ad.layers:
        # restore path: X may still be counts after scfair
        ad.layers[counts_layer] = ad.X.copy()
    X_log, pseudo = _logfc_inputs(_lognorm_matrix_from_counts(ad, counts_layer), "log1p")

    # PCA for stability (reuse intermediate if present — rebuild cheap on HVG)
    e = ad[:, ad.var["highly_variable"]].copy()
    if counts_layer in e.layers:
        e.X = e.layers[counts_layer].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    n_pcs = min(30, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    X_pca = np.asarray(e.obsm["X_pca"])

    unit_labels, merges = _build_legitimate_units(
        X_pca,
        X_log,
        labels,
        min_cluster_size=MIN_CLUSTER_SIZE,
        stability_threshold=DEFAULT_CAP_MERGE_THRESHOLD,
        de_threshold=DEFAULT_DE_SUPPORT_THRESHOLD,
        random_state=seed,
        de_pseudo=float(pseudo),
    )

    conf = (
        a.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in a.obs.columns
        else np.ones(a.n_obs, dtype=bool)
    )
    y_true = a.obs["cell_type"].astype(str)
    y_true = y_true[conf]
    lab_raw = labels.reindex(a.obs_names).astype(str)[conf]
    lab_u = unit_labels.reindex(a.obs_names).astype(str)[conf]

    unit_row = {
        "dataset": name,
        "seed": seed,
        "n_leiden": int(labels.nunique()),
        "n_units": int(unit_labels.nunique()),
        "n_merges": len(merges),
        "n_labels": int(y_true.nunique()),
        "ARI_leiden_vs_label": float(adjusted_rand_score(y_true, lab_raw)),
        "ARI_units_vs_label": float(adjusted_rand_score(y_true, lab_u)),
        "NMI_leiden_vs_label": float(normalized_mutual_info_score(y_true, lab_raw)),
        "NMI_units_vs_label": float(normalized_mutual_info_score(y_true, lab_u)),
        "frag_leiden": _fragment_rate(y_true, lab_raw),
        "frag_units": _fragment_rate(y_true, lab_u),
        "glue_leiden": _glue_rate(y_true, lab_raw),
        "glue_units": _glue_rate(y_true, lab_u),
    }

    # gene indices
    var_index = {str(g): i for i, g in enumerate(ad.var_names.astype(str))}
    sel_idx = np.array([var_index[g] for g in selected if g in var_index], dtype=int)
    pool_set = set(pool)
    sel_set = set(selected)

    # NN pairs on units
    sizes = unit_labels.value_counts()
    active = [c for c in sizes.index if sizes[c] >= MIN_CLUSTER_SIZE]
    masks = {c: (unit_labels == c).to_numpy() for c in active}
    bound_rows: list[dict] = []
    if len(active) >= 2:
        try:
            nn = _nearest_cluster_map(X_pca, masks)
        except Exception:
            nn = {}
        seen: set[tuple[str, str]] = set()
        for c, nbr in nn.items():
            key = (c, nbr) if c < nbr else (nbr, c)
            if key in seen:
                continue
            seen.add(key)
            ma, mb = masks[c], masks[nbr]
            de_full = _pair_de_support(X_log, ma, mb, top_n=TOP_N, pseudo=float(pseudo))
            de_sel = _de_on_genes(X_log, ma, mb, sel_idx, float(pseudo))
            gap = de_full - de_sel
            gap_frac = gap / de_full if de_full > 1e-9 else 0.0
            stab = _pair_bootstrap_stability(X_pca, ma, mb, random_state=seed)

            # unused pool genes with high A↔B logFC (either direction)
            lfc_ab = _cluster_vs_rest_logfc(
                X_log, ma, pseudo=float(pseudo), one_sided=True, out_mask=mb
            )
            lfc_ba = _cluster_vs_rest_logfc(
                X_log, mb, pseudo=float(pseudo), one_sided=True, out_mask=ma
            )
            lfc = np.maximum(lfc_ab, lfc_ba)
            order = np.argsort(-lfc)
            pool_left = 0
            for j in order[:100]:
                g = str(ad.var_names[j])
                if g in pool_set and g not in sel_set and lfc[j] > 0:
                    pool_left += 1
            actionable = (
                stab >= DEFAULT_CAP_MERGE_THRESHOLD
                and de_full >= DEFAULT_DE_SUPPORT_THRESHOLD
                and gap_frac >= GAP_FRAC
                and pool_left >= POOL_LEFT_MIN
            )
            bound_rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "a": c,
                    "b": nbr,
                    "n_a": int(ma.sum()),
                    "n_b": int(mb.sum()),
                    "stability": round(stab, 3),
                    "de_full": round(de_full, 4),
                    "de_selected": round(de_sel, 4),
                    "gap": round(gap, 4),
                    "gap_frac": round(gap_frac, 3),
                    "pool_left": pool_left,
                    "actionable": bool(actionable),
                }
            )

    unit_row["n_boundaries"] = len(bound_rows)
    unit_row["n_actionable"] = int(sum(r["actionable"] for r in bound_rows))
    return unit_row, bound_rows


def main(order: list[str]) -> None:
    unit_rows: list[dict] = []
    bound_rows: list[dict] = []
    for name in order:
        print(f"\n### {name}", flush=True)
        try:
            u, b = probe_one(name, seed=0)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
            continue
        unit_rows.append(u)
        bound_rows.extend(b)
        print(
            f"  leiden={u['n_leiden']} units={u['n_units']} merges={u['n_merges']}  "
            f"ARI L/U={u['ARI_leiden_vs_label']:.3f}/{u['ARI_units_vs_label']:.3f}  "
            f"frag L/U={u['frag_leiden']:.2f}/{u['frag_units']:.2f}  "
            f"actionable={u['n_actionable']}/{u['n_boundaries']}",
            flush=True,
        )
        pd.DataFrame(unit_rows).to_csv(UNITS_CSV, index=False)
        pd.DataFrame(bound_rows).to_csv(BOUNDS_CSV, index=False)

    print(f"\nwrote {UNITS_CSV}")
    print(f"wrote {BOUNDS_CSV}")
    if unit_rows:
        df = pd.DataFrame(unit_rows)
        print("\n=== units summary ===")
        print(
            df[
                [
                    "dataset",
                    "n_leiden",
                    "n_units",
                    "ARI_leiden_vs_label",
                    "ARI_units_vs_label",
                    "frag_leiden",
                    "frag_units",
                    "n_actionable",
                    "n_boundaries",
                ]
            ].to_string(index=False)
        )
        print(
            f"\nactionable boundaries total: "
            f"{df['n_actionable'].sum()} / {df['n_boundaries'].sum()}"
        )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    main(ORDER_QUICK if mode == "quick" else ORDER_FULL)
