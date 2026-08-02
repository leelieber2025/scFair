#!/bin/bash
# §5.19 follow-ups, run sequentially to avoid CPU contention.
#   1. extend the resolution grid for the three peak-censored datasets
#   2. probe the pbmc_seurat_v4_20k deficit (partition quality + k-sweep)
set -u
cd /home/lieber/scFair
PY=/home/lieber/miniforge3/envs/python/bin/python
echo "===== [1/2] grid extension  $(date) ====="
$PY examples/cluster_pool_extend.py
echo "===== [2/2] deficit probe   $(date) ====="
$PY examples/deficit_probe.py
echo "===== ALL FOLLOWUPS DONE    $(date) ====="
