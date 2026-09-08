#!/bin/bash
#SBATCH --job-name=judge2
#SBATCH --output=logs/judge2_%j.out
#SBATCH --error=logs/judge2_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# JUDGE RELIABILITY (docs/PLAN.md, Day 6).
#
# One 7B model deciding every score is the protocol's softest point: an examiner can
# ask whether the numbers describe the augmentations or the judge's idiosyncrasies.
# A SECOND, unrelated judge re-scores the SAME cached descriptions -- no rendering and
# no vision work at all, which is precisely why the run was split into two passes.
#
# Agreement between the two bounds how far any LLM-judged number here can be trusted,
# which is itself an answer to the proposal's "how should such systems be evaluated?".

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

JUDGE2="${JUDGE2:-Qwen/Qwen3-8B}"
echo "[judge2] second judge = $JUDGE2  (main judge stays in protocol_results.json)"

python - <<PY
import config, sys
config.DEVICE = "cuda"
sys.argv = ["run_protocol", "--phase", "judge", "--judge", "${JUDGE2}",
            "--tag", "judge2", "--desc-tag", ""]
from benchmark.run_protocol import main
main()
PY

python scripts/compare_runs.py protocol_results.json protocol_results_judge2.json
