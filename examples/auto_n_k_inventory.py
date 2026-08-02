#!/usr/bin/env python
"""Inventory: on each local h5ad, what k does structure auto_n (product) pick?

Uses the same multi-seed structure path as n_top_genes=\"auto\" (v7,
PRODUCT_STRUCTURE_N_SEEDS). Large matrices are stratified/random subsampled
to MAX_CELLS so the panel finishes in reasonable time; the subsample size
is recorded.

Output: examples/results/auto_n_k_inventory.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from scfair.pp._auto_n import (  # noqa: E402
    PRODUCT_STRUCTURE_N_SEEDS,
    estimate_n_top_structure,
)

warnings.filterwarnings("ignore")

DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "auto_n_k_inventory.csv"

MAX_CELLS = 20_000
SEED = 0


def _ensure_counts(a: ad.AnnData) -> ad.AnnData:
    if "counts" in a.layers:
        return a
    # Heuristic: use .X if non-negative
    X = a.X
    if X is None:
        raise ValueError("no X and no layers['counts']")
    a = a.copy()
    a.layers["counts"] = a.X.copy()
    return a


def _label_col(a: ad.AnnData) -> str | None:
    for c in ("cell_type", "celltype", "CellType", "cluster", "louvain", "leiden"):
        if c in a.obs.columns:
            return c
    return None


def _subsample(a: ad.AnnData, n_max: int, seed: int) -> tuple[ad.AnnData, str]:
    if a.n_obs <= n_max:
        return a, "full"
    rng = np.random.default_rng(seed)
    lab = _label_col(a)
    if lab is not None:
        # stratified
        idx = []
        labels = a.obs[lab].astype(str)
        groups = labels.groupby(labels, observed=False).indices
        # proportional allocation
        sizes = {g: len(ix) for g, ix in groups.items()}
        total = sum(sizes.values())
        take = {g: max(1, int(round(n_max * s / total))) for g, s in sizes.items()}
        # fix rounding
        while sum(take.values()) > n_max:
            g = max(take, key=take.get)
            if take[g] > 1:
                take[g] -= 1
            else:
                break
        while sum(take.values()) < n_max:
            g = max(sizes, key=lambda x: sizes[x] - take.get(x, 0))
            if take.get(g, 0) < sizes[g]:
                take[g] = take.get(g, 0) + 1
            else:
                break
        for g, ix in groups.items():
            n = min(take.get(g, 1), len(ix))
            pick = rng.choice(ix, size=n, replace=False)
            idx.append(pick)
        idx = np.concatenate(idx)
        if idx.size > n_max:
            idx = rng.choice(idx, size=n_max, replace=False)
        return a[np.sort(idx)].copy(), f"stratified_{n_max}"
    idx = rng.choice(a.n_obs, size=n_max, replace=False)
    return a[np.sort(idx)].copy(), f"random_{n_max}"


def run_one(path: Path) -> dict:
    t0 = time.time()
    row: dict = {
        "dataset": path.stem,
        "path": path.name,
        "status": "ok",
        "error": "",
        "n_obs_full": np.nan,
        "n_vars": np.nan,
        "n_obs_used": np.nan,
        "subsample": "",
        "n_top_selected": np.nan,
        "rule_branch": "",
        "k_source": "",
        "short_blocked": "",
        "short_block_reason": "",
        "short_k_raw": np.nan,
        "n_density_pops": np.nan,
        "n_leiden": np.nan,
        "valley_median": np.nan,
        "density_confidence": "",
        "density_depth_sensitivity": np.nan,
        "per_seed_k": "",
        "n_seeds": PRODUCT_STRUCTURE_N_SEEDS,
        "seconds": np.nan,
        "vs_2000": "",
    }
    try:
        a = ad.read_h5ad(path)
        row["n_obs_full"] = int(a.n_obs)
        row["n_vars"] = int(a.n_vars)
        a = _ensure_counts(a)
        a, how = _subsample(a, MAX_CELLS, SEED)
        row["subsample"] = how
        row["n_obs_used"] = int(a.n_obs)
        # drop empty genes if any (helps seurat_v3)
        if "counts" in a.layers:
            from scipy import sparse

            X = a.layers["counts"]
            if sparse.issparse(X):
                keep = np.asarray(X.sum(axis=0)).ravel() > 0
            else:
                keep = np.asarray(X).sum(axis=0) > 0
            if keep.sum() < a.n_vars:
                a = a[:, keep].copy()

        k, detail = estimate_n_top_structure(
            a,
            counts_layer="counts",
            random_state=SEED,
            version="v7",
            n_seeds=PRODUCT_STRUCTURE_N_SEEDS,
            n_genes=int(a.n_vars),
            progress=False,
        )
        feat = detail.get("features") or {}
        rx = detail.get("rule_explain") or {}
        row["n_top_selected"] = int(k)
        row["rule_branch"] = str(detail.get("rule_branch") or "")
        row["k_source"] = str(detail.get("k_source") or "")
        row["short_blocked"] = bool(detail.get("short_blocked") or rx.get("short_blocked"))
        row["short_block_reason"] = str(
            detail.get("short_block_reason") or rx.get("short_block_reason") or ""
        )
        raw = detail.get("short_k_raw")
        if raw is None:
            raw = rx.get("short_k_raw")
        row["short_k_raw"] = raw if raw is not None else np.nan
        row["n_density_pops"] = feat.get("n_density_pops", rx.get("n_density_pops"))
        row["n_leiden"] = feat.get("n_leiden", rx.get("n_leiden"))
        row["valley_median"] = feat.get("valley_median", rx.get("valley_median"))
        row["density_confidence"] = str(
            feat.get("density_confidence") or rx.get("density_confidence") or ""
        )
        row["density_depth_sensitivity"] = feat.get(
            "density_depth_sensitivity", rx.get("density_depth_sensitivity")
        )
        row["per_seed_k"] = str(detail.get("per_seed_k") or "")
        if int(k) == 2000:
            if row["short_blocked"]:
                row["vs_2000"] = "floor_2000_anti_short"
            elif "default_2000" in row["rule_branch"] or "2000" in row["rule_branch"]:
                row["vs_2000"] = "rule_chose_2000"
            else:
                row["vs_2000"] = "equals_2000"
        elif int(k) < 2000:
            row["vs_2000"] = f"below_2000({k})"
        else:
            row["vs_2000"] = f"above_2000({k})"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    paths = sorted(DATA.glob("*.h5ad"))
    print(f"Found {len(paths)} h5ad under {DATA}")
    print(f"structure n_seeds={PRODUCT_STRUCTURE_N_SEEDS}, max_cells={MAX_CELLS}")
    rows = []
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p.name} ...", flush=True)
        row = run_one(p)
        rows.append(row)
        print(
            f"  -> status={row['status']} k={row['n_top_selected']} "
            f"branch={row['rule_branch'][:60]!r} vs2000={row['vs_2000']} "
            f"t={row['seconds']}s err={row['error'][:80] if row['error'] else ''}",
            flush=True,
        )
        # incremental save
        pd.DataFrame(rows).to_csv(CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print(f"\nWrote {CSV}")
    ok = df[df.status == "ok"]
    if len(ok):
        print("\n=== summary (ok only) ===")
        print(
            ok[
                [
                    "dataset",
                    "n_obs_used",
                    "n_top_selected",
                    "vs_2000",
                    "rule_branch",
                    "short_blocked",
                ]
            ].to_string(index=False)
        )
        print("\nvs_2000 counts:")
        print(ok["vs_2000"].value_counts().to_string())
        print("\nk value counts:")
        print(ok["n_top_selected"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
