#!/bin/bash
# Run from your LOCAL PC (Git Bash), with the BIU VPN connected.
# Uploads the benchmark clips + tags (gitignored, so not in the repo clone).
#   bash slurm/sync_data.sh
set -euo pipefail

USER_AT=adamg@slurm-login1.lnx.biu.ac.il
REMOTE=~/MscProj
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "creating remote dirs..."
ssh "$USER_AT" "mkdir -p $REMOTE/data/input/benchmark $REMOTE/logs"

if command -v rsync >/dev/null 2>&1; then
    echo "rsync benchmark clips (resumable, skips unchanged)..."
    rsync -avz --partial --progress \
        "$LOCAL_ROOT/data/input/benchmark/" "$USER_AT:$REMOTE/data/input/benchmark/"
    rsync -avz "$LOCAL_ROOT/benchmark/tags.json" \
        "$LOCAL_ROOT/benchmark/suggestions.json" "$USER_AT:$REMOTE/benchmark/"
else
    echo "rsync not found -- falling back to a tarball over ssh..."
    tar -C "$LOCAL_ROOT" -czf - data/input/benchmark benchmark/tags.json \
        | ssh "$USER_AT" "tar -C $REMOTE -xzf -"
fi

echo
echo "remote inventory:"
ssh "$USER_AT" "cd $REMOTE && for d in data/input/benchmark/*/; do \
    printf '  %-22s %s\n' \"\$(basename \$d)\" \"\$(ls \$d 2>/dev/null | wc -l)\"; done"
echo "done. Next:  ssh $USER_AT  then  sbatch slurm/job_smoke.sh"
