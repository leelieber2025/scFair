"""Resolution sweep: is the duo* regression an artifact of fixed LEIDEN_RES=0.8?

Reuses the panel's gene sets, builds the kNN graph ONCE per (dataset, config,
seed), then sweeps Leiden resolution on that graph.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
sys.path.insert(0, "examples")
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sorted_gold_panel import load_dataset, per_population_f1, select_genes

OUT = Path("/tmp/claude-1000/-home-lieber-scFair/26e4d6a2-fbad-44e4-997f-93b3945f3c9f/scratchpad")
RES_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
K = 2000
SEEDS = range(10)
CONFIGS = ["hvg", "hybrid", "hybrid_cp5000"]
GCACHE = OUT / "sweep_genes.json"
RESULT = OUT / "res_sweep.csv"

genes_cache = json.load(open(GCACHE)) if GCACHE.exists() else {}
done = set()
if RESULT.exists():
    prev = pd.read_csv(RESULT)
    done = {(r.dataset, r.config, r.seed, round(r.resolution, 3)) for r in prev.itertuples()}
    rows = prev.to_dict("records")
else:
    rows = []

for ds in ["duo4_pbmc", "duo8_pbmc"]:
    a0, info = load_dataset(ds)
    for cfg in CONFIGS:
        for seed in SEEDS:
            key = f"{ds}|{cfg}|{seed}"
            if key not in genes_cache:
                t = time.time()
                g, _ = select_genes(a0, cfg, n_top=K, seed=seed)
                genes_cache[key] = g
                json.dump(genes_cache, open(GCACHE, "w"))
                print(f"  select {key}: {len(g)} genes ({time.time() - t:.0f}s)", flush=True)
            if all((ds, cfg, seed, round(r, 3)) in done for r in RES_GRID):
                continue
            genes = [g for g in genes_cache[key] if g in a0.var_names]
            a = a0.copy()
            a.X = a.layers["counts"].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            a = a[:, genes].copy()
            sc.pp.scale(a, max_value=10)
            sc.tl.pca(a, n_comps=40, svd_solver="arpack", random_state=seed)
            sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=seed)
            y_true = a.obs[info["label_col"]].astype(str)
            prev_counts = y_true.value_counts(normalize=True)
            smallest = prev_counts.idxmin()
            for res in RES_GRID:
                sc.tl.leiden(
                    a,
                    resolution=res,
                    key_added="L",
                    flavor="igraph",
                    n_iterations=2,
                    random_state=seed,
                )
                y_pred = a.obs["L"].astype(str)
                f1 = per_population_f1(y_true, y_pred)
                rows.append(
                    dict(
                        dataset=ds,
                        config=cfg,
                        seed=seed,
                        resolution=res,
                        n_leiden=int(y_pred.nunique()),
                        ARI=float(adjusted_rand_score(y_true, y_pred)),
                        NMI=float(normalized_mutual_info_score(y_true, y_pred)),
                        macro_f1=float(np.mean(list(f1.values()))),
                        min_pop_f1=float(f1[str(smallest)]),
                        min_pop=str(smallest),
                    )
                )
            pd.DataFrame(rows).to_csv(RESULT, index=False)
            print(f"{ds} {cfg} seed={seed} done", flush=True)
print("SWEEP DONE")
