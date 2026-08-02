#!/usr/bin/env python
"""ADT (CITE-seq protein) gold standard — a *non-circular* HVG benchmark.

Motivation
----------
Author cell-type labels in `p1`/`p2`/`p3` are produced by pipelines that
themselves run ``HVG(seurat_v3, n_top=2000) -> PCA -> Leiden -> annotate``.
Evaluating gene selection against those labels rewards methods that reproduce
the 2000-HVG subspace (see docs/DEVELOPMENT_LOG.md §5.7.5, "protocol shadow").
That bias makes the *auto vs fixed-k* comparison uninterpretable.

Here the labels come from a **different measurement modality**: 14 TotalSeq-B
antibodies measured on the same cells (10x PBMC 10k protein v3). Surface
protein is the classical immunology definition of a PBMC population and is
not a function of any RNA feature selection. RNA is never used to build the
labels.

Protocol
--------
Stage A (labels, protein only):
  CLR-normalize ADT across cells -> scale -> Leiden -> annotate each cluster
  from its **median CLR profile** via explicit marker rules (``ADT_RULES``).
  Clusters that match no rule become ``unassigned`` and are excluded from
  scoring. Cached to ``examples/data/pbmc_10k_adt_labeled.h5ad``.

Stage B (evaluation, RNA only):
  For each feature-selection method: select genes from raw counts, then
  log-normalize -> scale -> PCA -> Leiden(res=0.8) and score against the ADT
  labels. Repeated over ``SEEDS`` because §5.6 established that single-seed
  ARI differences of ~0.01 are inside seed noise.

Metrics
-------
ARI / NMI                 global agreement with the protein partition
per-population F1          best-matching Leiden cluster, per ADT population
rare_f1_mean               mean F1 over populations with prevalence < 2%
                           (Treg, non-classical monocyte) — the case
                           docs/DEVELOPMENT_LOG.md §5.6 lists as unsolved

Outputs: examples/results/adt_gold_*.csv / .json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scfair as scf

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "pbmc_10k_protein_v3.h5ad"
LABELED = DATA / "pbmc_10k_adt_labeled.h5ad"

METHODS = ["hvg", "scfair_hybrid", "scfair_auto", "scfair_score", "scfair_reweight"]
SEEDS = [0, 1, 2, 3, 4]
N_TOP = 2000
LEIDEN_RES = 0.8
RARE_MAX_FRAC = 0.02  # populations below this prevalence are "rare"

# --------------------------------------------------------------------------
# Stage A — protein-only labels
# --------------------------------------------------------------------------

# Rules are evaluated in order on a cluster's *median CLR* profile; first
# match wins. Thresholds are on CLR units, which are centred per protein
# across cells, so 0 is that protein's average cell and >1 is clearly
# positive. Audit them with the profile table printed by build_adt_labels().
ADT_RULES: list[tuple[str, str]] = [
    # exclude before typing: CD15-high is granulocyte/doublet debris in PBMC
    ("unassigned_CD15hi", "CD15 > 1.5"),
    ("B", "CD3 < 0 and CD19 > 1.5"),
    ("NK", "CD3 < 0 and CD56 > 1.0"),
    ("Mono_classical", "CD3 < 0 and CD14 > 1.0"),
    ("Mono_nonclassical", "CD3 < 0 and CD16 > 1.5 and CD14 > 0"),
    # Treg must precede the CD4 subsets: CD25+ CD127-low within CD4 T
    ("Treg", "CD3 > 0.5 and CD4 > 1.0 and CD25 > 1.5 and CD127 < 0.5"),
    ("CD4_naive", "CD3 > 0.5 and CD4 > 1.0 and CD45RA > CD45RO"),
    ("CD4_memory", "CD3 > 0.5 and CD4 > 1.0 and CD45RO >= CD45RA"),
    ("CD8_naive", "CD3 > 0.5 and CD8a > 1.0 and CD45RA > CD45RO"),
    ("CD8_memory", "CD3 > 0.5 and CD8a > 1.0 and CD45RO >= CD45RA"),
]


def clr_across_cells(P: np.ndarray) -> np.ndarray:
    """Centred log-ratio per protein (margin=2): log(x+1) minus column mean."""
    L = np.log1p(P.astype(float))
    return L - L.mean(axis=0, keepdims=True)


def annotate_cluster(profile: pd.Series) -> str:
    """Map one cluster's median CLR profile to a population via ADT_RULES."""
    env = {k: float(v) for k, v in profile.items()}
    for label, expr in ADT_RULES:
        try:
            if eval(expr, {"__builtins__": {}}, env):  # noqa: S307 - fixed rule table
                return "unassigned" if label.startswith("unassigned") else label
        except Exception:
            continue
    return "unassigned"


def build_adt_labels(*, resolution: float = 0.8, seed: int = 0, verbose: bool = True):
    """Build protein-derived labels. Touches ``obsm['protein_expression']`` only."""
    a = ad.read_h5ad(SOURCE)
    names = [str(n).replace("_TotalSeqB", "") for n in a.uns["protein_names"]]
    P = np.asarray(a.obsm["protein_expression"], dtype=float)
    C = clr_across_cells(P)

    adt = ad.AnnData(X=C.astype(np.float32))
    adt.obs_names = a.obs_names
    adt.var_names = names
    sc.pp.scale(adt, max_value=10)
    sc.pp.neighbors(adt, n_neighbors=20, use_rep="X", random_state=seed)
    sc.tl.leiden(
        adt,
        resolution=resolution,
        key_added="adt_cluster",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )

    clusters = adt.obs["adt_cluster"].astype(str)
    clr_df = pd.DataFrame(C, columns=names, index=a.obs_names)
    profiles = clr_df.groupby(clusters.values).median()
    mapping = {cl: annotate_cluster(profiles.loc[cl]) for cl in profiles.index}

    a.obs["adt_cluster"] = clusters.values
    a.obs["cell_type"] = clusters.map(mapping).values
    a.obs["adt_confident"] = a.obs["cell_type"] != "unassigned"

    if verbose:
        tbl = profiles.round(2).copy()
        tbl.insert(0, "label", [mapping[c] for c in profiles.index])
        tbl.insert(1, "n", clusters.value_counts().reindex(profiles.index).values)
        print("\n--- ADT cluster profiles (median CLR) -> rule assignment ---")
        print(tbl.to_string())

    # RNA side: raw counts only, standard gene filter
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)
    a.layers["counts"] = a.X.copy()
    a.obs_names_make_unique()
    a.var_names_make_unique()
    a.uns["label_source"] = (
        "CITE-seq ADT (14 TotalSeq-B) CLR -> Leiden -> marker-rule annotation; "
        "RNA not used for labels"
    )
    a.uns["dataset_source"] = "10x PBMC 10k protein v3 (pbmc_10k_protein_v3.h5ad)"
    return a


def load_labeled(*, rebuild: bool = False):
    if LABELED.exists() and not rebuild:
        a = ad.read_h5ad(LABELED)
        if "counts" not in a.layers:
            a.layers["counts"] = a.X.copy()
        return a
    a = build_adt_labels()
    # Source X is dense; ~920 MB uncompressed for X + counts. Compression only
    # affects storage, so cached labels stay bit-identical to §5.9's numbers.
    a.write_h5ad(LABELED, compression="gzip")
    return a


# --------------------------------------------------------------------------
# Stage B — RNA feature selection + scoring
# --------------------------------------------------------------------------


def select_genes(adata, method: str, seed: int = 0) -> tuple[list[str], dict]:
    a = adata.copy()
    meta: dict = {}
    n_top = min(N_TOP, a.n_vars - 1)
    common = dict(flavor="seurat_v3", layer="counts", marker_mode="none", random_state=seed)

    if method == "hvg":
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3", layer="counts")
    elif method == "scfair_hybrid":
        scf.pp.highly_variable_genes(
            a, n_top_genes=n_top, balance_method="hybrid", blend_global=0.95, **common
        )
    elif method == "scfair_auto":
        scf.pp.highly_variable_genes(
            a,
            n_top_genes="auto",
            n_top_min=500,
            n_top_max=min(5000, a.n_vars - 1),
            balance_method="hybrid",
            blend_global=0.95,
            **common,
        )
    elif method == "scfair_score":
        scf.pp.highly_variable_genes(a, n_top_genes=n_top, balance_method="score", **common)
    elif method == "scfair_reweight":
        scf.pp.highly_variable_genes(a, n_top_genes=n_top, balance_method="reweight", **common)
    else:
        raise ValueError(method)

    genes = a.var_names[a.var["highly_variable"]].astype(str).tolist()
    if method != "hvg":
        meta = dict(a.uns.get("scfair", {}).get("hvg", {}))
        meta.pop("selected_genes", None)
    return genes, meta


def per_population_f1(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Best-matching-cluster F1 for each true population.

    For population p, take the Leiden cluster maximizing F1(p, cluster). This
    is the standard 'can the pipeline isolate this population at all' measure
    and, unlike ARI, does not average a rare population away.
    """
    out: dict[str, float] = {}
    for pop in sorted(y_true.unique()):
        t = (y_true == pop).to_numpy()
        best = 0.0
        for cl in y_pred.unique():
            p = (y_pred == cl).to_numpy()
            tp = float(np.sum(t & p))
            if tp == 0:
                continue
            prec = tp / float(p.sum())
            rec = tp / float(t.sum())
            best = max(best, 2 * prec * rec / (prec + rec))
        out[pop] = best
    return out


def cluster_metrics(adata, genes: list[str], *, seed: int = 0) -> dict:
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    genes = [g for g in genes if g in a.var_names]
    if len(genes) < 10:
        return {"n_genes": len(genes), "ARI": np.nan, "NMI": np.nan}
    a = a[:, genes].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(40, a.n_vars - 1, a.n_obs - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), n_pcs=min(30, n_comps), random_state=seed)
    sc.tl.leiden(
        a,
        resolution=LEIDEN_RES,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        random_state=seed,
    )

    conf = a.obs["adt_confident"].to_numpy(dtype=bool)
    y_true = a.obs["cell_type"].astype(str)[conf]
    y_pred = a.obs["leiden"].astype(str)[conf]

    f1 = per_population_f1(y_true, y_pred)
    prev = y_true.value_counts(normalize=True)
    rare = [p for p in f1 if prev.get(p, 0.0) < RARE_MAX_FRAC]

    res = {
        "n_genes": len(genes),
        "n_leiden": int(y_pred.nunique()),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "macro_f1": float(np.mean(list(f1.values()))),
        "rare_f1_mean": float(np.mean([f1[p] for p in rare])) if rare else np.nan,
    }
    for pop, v in f1.items():
        res[f"f1_{pop}"] = float(v)
    return res


def main(rebuild: bool = False):
    adata = load_labeled(rebuild=rebuild)
    conf = adata.obs["adt_confident"].to_numpy(dtype=bool)
    # astype(str) first: 'unassigned' survives as an unused categorical level
    # after the confidence filter and would otherwise show up as an n=0
    # population and pollute the rare-population list.
    counts = adata.obs.loc[conf, "cell_type"].astype(str).value_counts()
    print(
        f"\n{adata.n_obs} cells x {adata.n_vars} genes | "
        f"{int(conf.sum())} confident ({100 * conf.mean():.1f}%) | "
        f"{counts.size} populations"
    )
    print("\n--- ADT gold-standard populations ---")
    print(pd.DataFrame({"n": counts, "pct": (100 * counts / counts.sum()).round(2)}).to_string())
    pops = pd.DataFrame({"n": counts, "frac": counts / counts.sum()})
    rare_pops = pops.index[pops["frac"] < RARE_MAX_FRAC].tolist()
    print(f"rare (<{RARE_MAX_FRAC:.0%}): {rare_pops}")

    rows = []
    for method in METHODS:
        for seed in SEEDS:
            print(f"  {method} seed={seed} ...", flush=True)
            try:
                genes, meta = select_genes(adata, method, seed=seed)
                res = cluster_metrics(adata, genes, seed=seed)
                res.update(
                    {
                        "method": method,
                        "seed": seed,
                        "n_top_used": meta.get("n_top_genes_used"),
                    }
                )
                rows.append(res)
                print(
                    f"    ARI={res['ARI']:.3f} NMI={res['NMI']:.3f} "
                    f"macroF1={res['macro_f1']:.3f} rareF1={res['rare_f1_mean']:.3f} "
                    f"n={res['n_genes']}",
                    flush=True,
                )
            except Exception as e:
                print(f"    FAIL {type(e).__name__}: {e}", flush=True)
                rows.append({"method": method, "seed": seed, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adt_gold_benchmark.csv", index=False)

    ok = df.dropna(subset=["ARI"]) if "ARI" in df.columns else df
    if ok.empty:
        print("no successful runs")
        return

    metric_cols = ["ARI", "NMI", "macro_f1", "rare_f1_mean"] + [
        c for c in ok.columns if c.startswith("f1_")
    ]
    summary = ok.groupby("method")[metric_cols].agg(["mean", "std"]).round(4)
    print("\n======== mean over seeds ========")
    print(
        ok.groupby("method")[["ARI", "NMI", "macro_f1", "rare_f1_mean", "n_genes"]]
        .mean()
        .round(3)
        .to_string()
    )
    print("\n======== std over seeds ========")
    print(
        ok.groupby("method")[["ARI", "NMI", "macro_f1", "rare_f1_mean"]].std().round(4).to_string()
    )
    print("\n======== per-population F1 (mean over seeds) ========")
    f1_cols = [c for c in ok.columns if c.startswith("f1_")]
    print(ok.groupby("method")[f1_cols].mean().round(3).T.to_string())
    summary.to_csv(OUT / "adt_gold_summary.csv")

    # Paired per-seed deltas vs the scanpy baseline — the seed is a shared
    # nuisance factor, so pairing is more sensitive than comparing means.
    base = ok[ok["method"] == "hvg"].set_index("seed")
    deltas = {}
    for method in ok["method"].unique():
        if method == "hvg":
            continue
        sub = ok[ok["method"] == method].set_index("seed")
        shared = base.index.intersection(sub.index)
        d = {}
        for m in ("ARI", "NMI", "macro_f1", "rare_f1_mean"):
            diff = (sub.loc[shared, m] - base.loc[shared, m]).astype(float)
            d[m] = {
                "mean_delta": float(diff.mean()),
                "std_delta": float(diff.std()),
                "wins": int((diff > 0).sum()),
                "n_seeds": int(diff.notna().sum()),
            }
        deltas[method] = d
    print("\n======== paired delta vs scanpy hvg@2000 (per seed) ========")
    for method, d in deltas.items():
        bits = " ".join(
            f"{m}={d[m]['mean_delta']:+.4f}(±{d[m]['std_delta']:.4f},{d[m]['wins']}/{d[m]['n_seeds']})"
            for m in ("ARI", "macro_f1", "rare_f1_mean")
        )
        print(f"  {method:18s} {bits}")

    with open(OUT / "adt_gold_benchmark.json", "w") as f:
        json.dump(
            {
                "label_source": adata.uns.get("label_source"),
                "dataset_source": adata.uns.get("dataset_source"),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_confident": int(conf.sum()),
                "populations": counts.to_dict(),
                "rare_populations": rare_pops,
                "leiden_resolution": LEIDEN_RES,
                "seeds": SEEDS,
                "paired_delta_vs_hvg": deltas,
                "records": df.replace({np.nan: None}).to_dict(orient="records"),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {OUT / 'adt_gold_benchmark.csv'}")
    print("DONE")


if __name__ == "__main__":
    import sys

    main(rebuild="--rebuild" in sys.argv)
