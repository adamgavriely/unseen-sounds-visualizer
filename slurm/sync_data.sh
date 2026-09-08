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

# Upload the CODE too. The repo is private and GitHub no longer accepts password
# auth, so cloning on the cluster would mean putting a token on a shared machine.
# Pushing the working tree over ssh avoids that entirely and is what we already do
# for the clips.
echo "uploading code..."
if command -v rsync >/dev/null 2>&1; then
    rsync -az --delete         --include="*/"         --include="*.py" --include="*.sh" --include="*.json" --include="*.md"         --include="*.txt" --include="*.bat"         --exclude="*"         --exclude=".git/" --exclude="data/" --exclude="__pycache__/"         "$LOCAL_ROOT/" "$USER_AT:$REMOTE/"
else
    tar -C "$LOCAL_ROOT" --exclude=.git --exclude=data --exclude=__pycache__         -czf - src benchmark scripts slurm config.py main.py requirements.txt         | ssh "$USER_AT" "tar -C $REMOTE -xzf -"
fi

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
