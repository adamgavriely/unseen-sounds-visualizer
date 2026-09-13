#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --output=logs/timing_%j.out
#SBATCH --error=logs/timing_%j.err
#SBATCH --partition=L4-4h,L40s-4h,A100-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# Where does a sound start? Four timing methods on three clips (scripts/timing_compare.py).
# Method D needs PretrainedSED (github.com/fschmid56/PretrainedSED, MIT) in its own env,
# because its pins differ from ours; it is created on first run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
mkdir -p logs data/output/timing data/output/timing/psed

CLIPS="data/input/benchmark/unseen_ambient/ev_smoke_alarm_kitchen.mp4 data/input/benchmark/mixed/mv_tornado_scene.mp4 data/input/benchmark/unseen_ambient/mc_bridge_scene.mp4"

# ---- D: PretrainedSED in its own env
if ! conda env list | grep -q '^psed '; then
    echo "=== creating psed env ==="
    conda create -y -q -n psed python=3.10 >/dev/null
    conda activate psed
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -q "numpy<2"
    # sed_scores_eval is evaluation-only and its installer needs numpy at build time;
    # inference does not import it, so it is left out
    grep -v sed_scores_eval "$HOME/PretrainedSED/requirements.txt" > /tmp/psed_req.txt
    pip install -q -r /tmp/psed_req.txt
    conda deactivate
fi
conda activate psed
for c in $CLIPS; do
    stem=$(basename "$c" .mp4)
    ffmpeg -y -i "$c" -ac 1 -ar 16000 "data/output/timing/psed/$stem.wav" -loglevel error
    python scripts/psed_dump.py "data/output/timing/psed/$stem.wav" "data/output/timing/psed/$stem.npz"
done
conda deactivate

# ---- A, B, C and the comparison
conda activate msproj
python scripts/timing_compare.py $CLIPS --psed-dir data/output/timing/psed --device cuda
echo DONE
