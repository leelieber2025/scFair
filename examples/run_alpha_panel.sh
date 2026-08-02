#!/bin/bash
# Waits for the consensus job to finish (CPU contention), then runs the alpha
# panel. Both live under the systemd user manager, so neither depends on an
# SSH session.
set -u
cd /home/lieber/scFair
while systemctl --user is-active --quiet scfair-consensus.service; do sleep 60; done
echo "===== consensus finished, starting alpha panel $(date) ====="
exec /home/lieber/miniforge3/envs/python/bin/python examples/alpha_panel.py
