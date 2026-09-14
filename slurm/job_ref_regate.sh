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
if [ ! -f "benchmark/protocol_reference_indep_${TAG}_strict.json" ] && grep -q '"references"' "$REF"; then
    cp "$REF" "benchmark/protocol_reference_indep_${TAG}_strict.json"
    [ -f "$RES" ] && mv "$RES" "benchmark/protocol_results_${TAG}_indep_strict.json"
    echo "kept the strict-gate pass as *_strict"
fi
python - "$REF" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
for k in ("verified", "references"):
    d.pop(k, None)
json.dump(d, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("cache keys now:", list(d))
PY
RUN="$TAG"     # a prefix assignment sees the ones before it: TAG=.. DESC_TAG="$TAG" reads the NEW TAG
GEN=diffusion TAG="$RUN" PHASE=reference bash slurm/job_protocol.sh
GEN=diffusion TAG="${RUN}_indep" DESC_TAG="$RUN" PHASE=judge INDEP=1 bash slurm/job_protocol.sh
echo "DONE -> $RES"
