#!/bin/bash
#SBATCH --job-name=ref_regate
#SBATCH --output=logs/ref_regate_%j.out
#SBATCH --error=logs/ref_regate_%j.err
#SBATCH --partition=L4-4h,L40s-4h,A100-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Re-verify and re-write the independent reference with the shipping gate (audio-decoy
# rank), keeping the cached listen and look steps, then re-judge. The first full pass
# used the mean+2sigma text bar; its cache and results are kept under *_strict so both
# can be reported (LIMITATIONS.md, verification gate). TAG picks the run (v2, v3).
#   sbatch slurm/job_ref_regate.sh             # v2
#   TAG=v3 sbatch slurm/job_ref_regate.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
TAG="${TAG:-v2}"
REF="benchmark/protocol_reference_indep_${TAG}.json"
RES="benchmark/protocol_results_${TAG}_indep.json"
# KEEP names the previous pass (strict: the 2-sigma text gate; list: the list-and-match
# visibility); DROP names the cache keys to recompute. Both default to the per-sound
# visibility rerun of 2026-09-14.
KEEP="${KEEP:-list}"
DROP="${DROP:-seen references}"
if [ ! -f "benchmark/protocol_reference_indep_${TAG}_${KEEP}.json" ] && grep -q '"references"' "$REF"; then
    cp "$REF" "benchmark/protocol_reference_indep_${TAG}_${KEEP}.json"
    [ -f "$RES" ] && mv "$RES" "benchmark/protocol_results_${TAG}_indep_${KEEP}.json"
    echo "kept the previous pass as *_${KEEP}"
fi
python - "$REF" $DROP <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
for k in sys.argv[2:]:
    d.pop(k, None)
json.dump(d, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("cache keys now:", list(d))
PY
RUN="$TAG"     # a prefix assignment sees the ones before it: TAG=.. DESC_TAG="$TAG" reads the NEW TAG
GEN=diffusion TAG="$RUN" PHASE=reference bash slurm/job_protocol.sh
# gate before the judge (review): the reference must agree with the human silence
# decision on 70% of clips, or the 20 GPU-minutes of judging are not spent
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
python scripts/ref_silence_check.py "$RUN" --bar "${BAR:-0.70}"
GEN=diffusion TAG="${RUN}_indep" DESC_TAG="$RUN" PHASE=judge INDEP=1 bash slurm/job_protocol.sh
echo "DONE -> $RES"
