#!/bin/bash
# Run from your LOCAL PC (Git Bash), VPN connected. Pulls results back.
#   bash slurm/fetch_results.sh
set -euo pipefail

USER_AT=adamg@slurm-login1.lnx.biu.ac.il
REMOTE=~/MscProj
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$LOCAL_ROOT/data/output" "$LOCAL_ROOT/logs"

echo "metrics..."
scp "$USER_AT:$REMOTE/benchmark/eval_results*.json" "$LOCAL_ROOT/benchmark/" || true
echo "augmented videos..."
scp "$USER_AT:$REMOTE/data/output/*_augmented.mp4" "$LOCAL_ROOT/data/output/" || true
echo "job logs..."
scp "$USER_AT:$REMOTE/logs/*.out" "$LOCAL_ROOT/logs/" || true

echo "done -> benchmark/eval_results*.json, data/output/, logs/"
