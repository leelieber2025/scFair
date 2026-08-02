#!/usr/bin/env python
"""Targeted top-up for clusters shortchanged by plain scanpy HVG, vs the
shipped hybrid default. Follow-up to DEVELOPMENT_LOG.md §5.29.

The question
------------
§5.29 found that plain scanpy top-``n_top`` (cluster-blind, ``flavor=
"seurat_v3"`` -- the actual pre-scFair baseline) leaves some intermediate
cluster with under half its equal share (``n_top / n_clusters``) of "own"
specificity genes in 6 of 10 labelled datasets, and that the starved cluster
is **not reliably the smallest one** -- in ``pbmc_seurat_v4_20k`` it is the
*second largest* of 17. A detector keyed on cluster cell-count size would
therefore miss the real deficits. This asks whether *directly* topping up
whichever cluster is starved -- not "small" clusters -- recovers anything,
without disturbing datasets that have no deficit.

The arms
--------
All arms are cut from the same partition (``scfair_hvg_clusters``) and the
same per-cluster one-sided ``logFC+`` order (``_build_cluster_gene_ranks``):

- ``rank`` -- the shipped default. ``scfair.pp.highly_variable_genes`` at
  ``n_top_genes=2000``, everything else at defaults.
- ``topup`` -- **superseded, kept only for the three-way comparison.**
  Starts from ``rank``'s selection, detects "starved" clusters against plain
  scanpy's own top-2000 (<``TRIGGER=0.5x`` equal share), and tops each up to
  ``TARGET=0.5x`` equal share from its own logFC+ order, funded by trimming
  clusters over ``DONOR_FLOOR=1.0x`` equal share. An ablation on
  ``pancreas_smartseq2`` (2 seeds) isolated the damage entirely to the
  *add* side: ``add_only`` ARI 0.429 vs ``rank`` 0.499, while ``remove_only``
  alone scored 0.525 -- **better than rank**. Root cause: the starved
  clusters there were not under-served populations but a fine intra-alpha
  substate (one is unmistakably a cell-cycle cluster: top genes ``TOP2A,
  UBE2C, PBK, MKI67, CDC20, CENPF, BIRC5, AURKB...``). Forcing its own
  markers into the selection sharpens exactly the sub-structure the
  labelled ground truth does not care about, so downstream Leiden
  over-splits the parent population. See DEVELOPMENT_LOG.md §5.29
  follow-up for the full diagnosis.
- ``cap`` -- the redesign the ablation motivated. No starved-cluster
  detection, no scanpy probe, no per-cluster target on the receiving side.
  Only trims: any cluster holding more than ``CEILING=1.0x`` equal share of
  its own peak-attributed genes in ``rank``'s selection is cut down to the
  ceiling, removing its own worst-ranked genes first (same "weak *for that
  cluster*" ordering topup's donor-side used, not the global blend). The
  freed slots are backfilled from the next-best genes by ``rank``'s own
  full blended score (``adata.var["scfair_score"]``, defined over every
  gene, not just the original top-2000) -- i.e. whatever the ordinary
  ranking would have picked next, unconstrained by cluster identity. This
  is ``remove_only`` from the ablation, but keeping ``k`` fixed by
  backfilling from the natural next-best candidates instead of just
  shrinking the selection.

Reading it
----------
Per-population is the headline, same as ``quota_allocation.py``:
``deprivation_topup_pops.csv`` is one row per (dataset, arm, seed,
resolution, population). Pooled ARI / macro_f1 / min_pop_f1 are background
in ``deprivation_topup.csv``. The 4 datasets with no starved/over-capped
cluster act as a sanity check: ``topup``/``cap`` should equal ``rank``
exactly there, per seed.

Scored over a resolution grid and summarised by the mean over resolution.
Resumable per (dataset, seed).

Outputs (examples/results/):
  deprivation_topup.csv       one row per (dataset, arm, seed, resolution)
  deprivation_topup_pops.csv  per-population F1, long format
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from umap3d_smoke import LOADERS  # noqa: E402

import scfair as scf  # noqa: E402
from scfair.pp._highly_variable_genes import _build_cluster_gene_ranks  # noqa: E402

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "deprivation_topup.csv"
POPS = OUT / "deprivation_topup_pops.csv"

K = 2000
TRIGGER = 0.5  # scanpy-share below this * equal_share -> flagged starved
TARGET = 0.5  # top up rank's own selection to this * equal_share
DONOR_FLOOR = 1.0  # never draw a donor cluster below this * equal_share
SEEDS = [0, 1, 2, 3, 4]
RES_GRID = [0.3, 0.5, 0.8, 1.2]
MIN_CLUSTER_SIZE = 30

ORDER = [
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


def scanpy_topk(a, k):
    """Plain scanpy top-k -- the pre-scFair baseline (§5.29's yardstick)."""
    p = a.copy()
    sc.pp.highly_variable_genes(
        p,
        n_top_genes=min(k, p.n_vars - 1),
        flavor="seurat_v3",
        layer="counts",
        span=0.3,
        inplace=True,
        subset=False,
    )
    return [str(g) for g in p.var_names[p.var["highly_variable"]]]


def peak_cluster_map(genes, ranks):
    """Attribute each gene to the cluster whose own logFC+ order ranks it best."""
    pos = {c: {g: i for i, g in enumerate(order_)} for c, order_ in ranks.items()}
    out = {}
    for g in genes:
        best_c, best_p = None, None
        for c, p_ in pos.items():
            p = p_.get(g)
            if p is not None and (best_p is None or p < best_p):
                best_p, best_c = p, c
        if best_c is not None:
            out[g] = best_c
    return out


def topup_select(genes_rank, ranks, k):
    """Top up `genes_rank` for clusters starved under a plain-scanpy probe.

    Order-preserving: untouched genes keep `genes_rank`'s own blended-score
    order (PCA(svd_solver="arpack") -> Leiden is sensitive to input column
    order, so a no-op call must return the *same list*, not just the same
    set -- see DEVELOPMENT_LOG.md §5.29 follow-up). Newly added genes are
    appended at the end in a fixed, deterministic order (grouped by starved
    cluster, each cluster's own logFC+ order) rather than via any
    set-iteration order.

    Returns (genes_topup, diag) where diag records what was flagged,
    added, removed, and any capped shortfall.
    """
    n_clusters = len(ranks)
    equal_share = k / n_clusters

    genes_scanpy = scanpy_topk_cache["genes"]
    peak_scanpy = peak_cluster_map(genes_scanpy, ranks)
    scanpy_count = {c: 0 for c in ranks}
    for g, c in peak_scanpy.items():
        scanpy_count[c] += 1
    starved = [c for c in ranks if scanpy_count[c] < TRIGGER * equal_share]

    peak_rank = peak_cluster_map(genes_rank, ranks)
    rank_count = {c: 0 for c in ranks}
    for g, c in peak_rank.items():
        rank_count[c] += 1

    current_set = set(genes_rank)
    # position within each cluster's OWN logFC+ order -- used to pick donor
    # genes that are weak *for their own cluster*, not genes that merely
    # score low on the 95%-global blend. A gene can rank near the bottom of
    # hybrid's global blend yet still be near the top of its own cluster's
    # logFC+ order (that's precisely the gene the specificity term exists to
    # keep); sorting donors by global rank was handing exactly those genes
    # back out. See DEVELOPMENT_LOG.md §5.29 follow-up (pancreas_smartseq2
    # diagnosis: donor genes removed at own-cluster logFC up to 5.6).
    own_pos = {c: {g: i for i, g in enumerate(order_)} for c, order_ in ranks.items()}

    added_genes: dict[str, list[str]] = {}
    removed_genes: set[str] = set()
    shortfall: dict[str, int] = {}

    for c in starved:
        target_n = int(round(TARGET * equal_share))
        need = max(0, target_n - rank_count.get(c, 0))
        if need == 0:
            continue
        candidates = [g for g in ranks[c] if g not in current_set][:need]
        if not candidates:
            shortfall[c] = need
            continue

        # donor pool: genes currently selected whose peak cluster is
        # >DONOR_FLOOR*equal_share in rank's own allocation, never drawing a
        # donor cluster below the floor. Sorted by the gene's position in
        # its OWN cluster's logFC+ order (worst-for-that-cluster first) --
        # not by the global blend, which anti-correlates with "safe to
        # remove" for exactly the genes specificity exists to protect.
        donor_budget = {
            dc: max(0, rank_count.get(dc, 0) - int(round(DONOR_FLOOR * equal_share)))
            for dc in ranks
            if dc != c
        }
        donor_pool = sorted(
            (g for g in genes_rank if g in current_set and peak_rank.get(g) in donor_budget),
            key=lambda g: -own_pos[peak_rank[g]].get(g, -1),  # deepest in own list first
        )

        removed_this: list[str] = []
        for g in donor_pool:
            if len(removed_this) >= len(candidates):
                break
            dc = peak_rank[g]
            if donor_budget[dc] <= 0:
                continue
            removed_this.append(g)
            donor_budget[dc] -= 1

        n_take = min(len(candidates), len(removed_this))
        add_this = candidates[:n_take]
        remove_this = removed_this[:n_take]

        for g in remove_this:
            current_set.discard(g)
        for g in add_this:
            current_set.add(g)
        # keep rank_count consistent in case a later starved cluster's
        # donor budget is computed against it
        rank_count[c] = rank_count.get(c, 0) + n_take

        added_genes[c] = add_this
        removed_genes.update(remove_this)
        if n_take < len(candidates):
            shortfall[c] = len(candidates) - n_take

    final = [g for g in genes_rank if g in current_set]
    for c in starved:
        final.extend(added_genes.get(c, []))

    diag = {
        "n_starved": len(starved),
        "starved": ",".join(starved),
        "n_added": sum(len(v) for v in added_genes.values()),
        "n_removed": len(removed_genes),
        "n_shortfall": sum(shortfall.values()),
    }
    return final, diag


scanpy_topk_cache: dict = {}

CEILING = 1.0  # cap: trim a cluster's own-gene count above this * equal_share

# Tirosh et al. 2016 S/G2M gene sets (via Regev lab list, the one scanpy's
# own cell-cycle tutorials ship) -- symbols only, so this is a no-op on
# Ensembl-ID datasets (e.g. duo8_pbmc) rather than a silent mismatch.
_CC_S_GENES = frozenset(
    """
MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1
MLF1IP HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3
MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1
CLSPN POLA1 CHAF1B BRIP1 E2F8
""".split()
)
_CC_G2M_GENES = frozenset(
    """
HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67
TMPO CENPF TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E
TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1
NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5
CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA
""".split()
)
CELL_CYCLE_GENES = _CC_S_GENES | _CC_G2M_GENES
CC_TOP_N = 20  # how far into a cluster's own order to look
CC_FRACTION = 0.3  # flag the cluster if >= this share of its top-N is cell-cycle


def cell_cycle_flagged_clusters(ranks):
    """Clusters whose own top marker genes are cell-cycle-dominated.

    Motivation: pancreas_smartseq2's intermediate partition splits its
    dominant type (alpha, 42% of cells) into 5 near-disjoint sub-clusters
    (top-30 own-gene Jaccard 0.00-0.07 between them -- not noise, real
    sub-structure). One of them is unmistakably a proliferating substate:
    top genes TOP2A, UBE2C, PBK, MKI67, CDC20, CENPF, BIRC5, AURKB... An
    add/remove ablation isolated cap's entire loss there to the *add* side
    (add_only ARI 0.429 vs rank 0.499; remove_only alone scored 0.525,
    *better* than rank) -- promoting this substate's own markers sharpens
    exactly the sub-structure the labelled ground truth does not carry,
    so downstream Leiden over-splits the parent population. See
    DEVELOPMENT_LOG.md §5.29 follow-up.

    This is deliberately narrow -- a well-precedented, standard reference
    list (the same one scanpy's own cell-cycle tutorials use), not a
    general "is this cluster a real population" classifier. No evidence
    exists yet for a broader guard; this is the one measured failure mode.
    """
    flagged = set()
    for c, order_ in ranks.items():
        top = order_[:CC_TOP_N]
        if not top:
            continue
        frac = sum(1 for g in top if g in CELL_CYCLE_GENES) / len(top)
        if frac >= CC_FRACTION:
            flagged.add(c)
    return flagged


def cap_select(genes_rank, ranks, scfair_score, k):
    """Trim over-represented clusters, backfill from the next-best global
    blended score. No detection of "starved" clusters, no per-cluster
    target on the receiving side -- see module docstring for why (the
    pancreas_smartseq2 ablation).

    `scfair_score` is `sel.var["scfair_score"]` -- rank's own full blended
    score, defined for every gene, not just the selected top-k, so backfill
    draws from the same ranking `rank` would have continued down.
    """
    n_clusters = len(ranks)
    equal_share = k / n_clusters
    own_pos = {c: {g: i for i, g in enumerate(order_)} for c, order_ in ranks.items()}
    cc_flagged = cell_cycle_flagged_clusters(ranks)

    peak_rank = peak_cluster_map(genes_rank, ranks)
    rank_count = {c: 0 for c in ranks}
    for g, c in peak_rank.items():
        rank_count[c] += 1

    current_set = set(genes_rank)
    removed: list[str] = []
    trimmed: set[str] = set()  # clusters actually cut down to the ceiling
    excluded_from_backfill: set[str] = set(cc_flagged)  # + trimmed, below
    ceiling_n = int(round(CEILING * equal_share))
    for c in ranks:
        over = rank_count.get(c, 0) - ceiling_n
        if over <= 0:
            continue
        trimmed.add(c)
        excluded_from_backfill.add(c)
        own_genes = sorted(
            (g for g in genes_rank if peak_rank.get(g) == c and g in current_set),
            key=lambda g: -own_pos[c].get(g, -1),  # deepest in own list first
        )
        for g in own_genes[:over]:
            current_set.discard(g)
            removed.append(g)
    n_over = len(trimmed)

    # Backfill from the next-best genes by the global blend -- but a
    # candidate's peak cluster (by the SAME own-cluster logFC order used to
    # cap) must not be one just capped, or the cap is silently undone: a
    # gene can be "deep in its own cluster's own logFC+ order" (why it was
    # removed) yet still score well on `scfair_score`, whose min-max
    # normalisation is over every gene, not the 4000-gene pool the original
    # blend used. Measured on duo8_pbmc: without this guard, 213/263
    # removed genes were backfilled right back in, cluster 2 ended at 720
    # own genes against a 500 ceiling, and ARI still dropped 0.626->0.560
    # for a cap that barely happened. See DEVELOPMENT_LOG.md §5.29 follow-up.
    score_order = [str(g) for g in scfair_score.sort_values(ascending=False).index]
    added: list[str] = []
    need = len(removed)
    for g in score_order:
        if need == 0:
            break
        if g in current_set:
            continue
        best_c, best_p = None, None
        for c, pos in own_pos.items():
            p = pos.get(g)
            if p is not None and (best_p is None or p < best_p):
                best_p, best_c = p, c
        if best_c in excluded_from_backfill:
            continue
        current_set.add(g)
        added.append(g)
        need -= 1

    final = [g for g in genes_rank if g in current_set] + added
    diag = {
        "n_starved": n_over,
        "starved": "",  # keep column shape aligned with topup
        "n_added": len(added),
        "n_removed": len(removed),
        "n_shortfall": need,
        "n_cc_flagged": len(cc_flagged),
    }
    return final, diag


def evaluate(a, genes, seed, rows, pop_rows, base):
    e = a.copy()
    e.X = e.layers["counts"].copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    e = e[:, [g for g in genes if g in e.var_names]].copy()
    sc.pp.scale(e, max_value=10)
    n_comps = min(40, e.n_vars - 1, e.n_obs - 1)
    sc.tl.pca(e, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(e, n_neighbors=15, n_pcs=min(30, n_comps), random_state=seed)

    conf = (
        e.obs["adt_confident"].to_numpy(dtype=bool)
        if "adt_confident" in e.obs.columns
        else np.ones(e.n_obs, dtype=bool)
    )
    y_true = e.obs["cell_type"].astype(str)[conf]
    prev = y_true.value_counts(normalize=True)

    for res in RES_GRID:
        sc.tl.leiden(
            e, resolution=res, key_added="L", flavor="igraph", n_iterations=2, random_state=seed
        )
        y_pred = e.obs["L"].astype(str)[conf]
        f1s = {}
        for pop in y_true.unique():
            t = (y_true == pop).to_numpy()
            best = 0.0
            for cl in y_pred.unique():
                p = (y_pred == cl).to_numpy()
                tp = float(np.sum(t & p))
                if tp:
                    pr, rc = tp / p.sum(), tp / t.sum()
                    best = max(best, 2 * pr * rc / (pr + rc))
            f1s[pop] = best
        for pop, f1 in f1s.items():
            pop_rows.append(
                {
                    "dataset": base["dataset"],
                    "arm": base["arm"],
                    "seed": base["seed"],
                    "resolution": res,
                    "population": pop,
                    "prevalence": float(prev.get(pop, np.nan)),
                    "f1": float(f1),
                }
            )
        rare = [p for p in f1s if prev.get(p, 0) < 0.02]
        rows.append(
            {
                **base,
                "resolution": res,
                "n_leiden": int(y_pred.nunique()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "macro_f1": float(np.mean(list(f1s.values()))),
                "min_pop_f1": float(f1s[str(prev.idxmin())]),
                "rare_f1_mean": (float(np.mean([f1s[p] for p in rare])) if rare else np.nan),
            }
        )


def main(which=None) -> None:
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    pop_rows = pd.read_csv(POPS).to_dict("records") if POPS.exists() else []
    seen_arms: dict[tuple, set] = {}
    for r in rows:
        seen_arms.setdefault((r["dataset"], r["seed"]), set()).add(r["arm"])
    done = {k for k, v in seen_arms.items() if {"rank", "topup", "cap"} <= v}
    print(f"resuming: {len(done)} (dataset, seed) blocks done", flush=True)

    for name in which or ORDER:
        if all((name, s) in done for s in SEEDS):
            print(f"### {name}: complete", flush=True)
            continue
        print(f"\n######## {name} ########", flush=True)
        try:
            a = LOADERS[name]()
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        if "cell_type" not in a.obs:
            print("  no cell_type; skipping", flush=True)
            continue
        print(
            f"  {a.n_obs} cells x {a.n_vars} genes, {a.obs['cell_type'].nunique()} labelled types",
            flush=True,
        )

        genes_scanpy = scanpy_topk(a, K)
        scanpy_topk_cache["genes"] = genes_scanpy

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.time()
            try:
                sel = a.copy()
                scf.pp.highly_variable_genes(
                    sel, n_top_genes=min(K, sel.n_vars - 1), random_state=seed
                )
                hv = sel.var["highly_variable"].to_numpy(dtype=bool)
                order = pd.to_numeric(sel.var["highly_variable_rank"], errors="coerce")[
                    hv
                ].sort_values()
                genes_rank = [str(g) for g in order.index]
                labels = sel.obs["scfair_hvg_clusters"]

                ranks = _build_cluster_gene_ranks(
                    sel,
                    cluster_labels=labels,
                    counts_layer="counts",
                    min_cluster_size=MIN_CLUSTER_SIZE,
                    logfc_space="log1p",
                )
                if not ranks:
                    print(f"  seed={seed}: no cluster ranks; skipping", flush=True)
                    continue
                genes_topup, diag = topup_select(genes_rank, ranks, K)
                genes_cap, diag_cap = cap_select(genes_rank, ranks, sel.var["scfair_score"], K)

                print(
                    f"  seed={seed}  {len(ranks)} clusters, "
                    f"topup: starved={diag['n_starved']} ({diag['starved']}), "
                    f"added={diag['n_added']} removed={diag['n_removed']} "
                    f"shortfall={diag['n_shortfall']}  |  "
                    f"cap: over={diag_cap['n_starved']}, cc_flagged={diag_cap['n_cc_flagged']}, "
                    f"added={diag_cap['n_added']} removed={diag_cap['n_removed']} "
                    f"shortfall={diag_cap['n_shortfall']}",
                    flush=True,
                )

                for arm, genes, d in (
                    ("rank", genes_rank, {"n_starved": 0}),
                    ("topup", genes_topup, diag),
                    ("cap", genes_cap, diag_cap),
                ):
                    base = {
                        "dataset": name,
                        "arm": arm,
                        "seed": seed,
                        "n_genes": len(genes),
                        "n_clusters": len(ranks),
                        "n_starved": d["n_starved"],
                    }
                    evaluate(a, genes, seed, rows, pop_rows, base)
                del sel
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL seed={seed}: {type(e).__name__}: {e}", flush=True)
                continue
            done.add((name, seed))
            pd.DataFrame(rows).to_csv(CSV, index=False)
            pd.DataFrame(pop_rows).to_csv(POPS, index=False)
            got = {
                arm: np.mean(
                    [
                        r["ARI"]
                        for r in rows
                        if r["dataset"] == name and r["seed"] == seed and r["arm"] == arm
                    ]
                )
                for arm in ("rank", "topup", "cap")
            }
            print(
                f"    ARI rank={got['rank']:.3f} topup={got['topup']:.3f} "
                f"cap={got['cap']:.3f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        del a

    print(f"\nwrote {CSV} ({len(rows)} rows) and {POPS} ({len(pop_rows)} rows)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
