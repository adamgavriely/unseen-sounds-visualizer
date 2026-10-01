#!/bin/bash
#SBATCH --job-name=heldouta4
#SBATCH --output=logs/heldouta4_%j.out
#SBATCH --error=logs/heldouta4_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#
# Round 42 HELDOUT-A4 (docs/prereg_round13_detector_push.md): EXPECT-A4 audio chain, no gate, on the 415 held-out clips. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/heldout_a4_screen.py wav
python benchmark/gold/heldout_a4_screen.py listen
python benchmark/gold/heldout_a4_screen.py cands
source "$HOME/venv_flap/bin/activate"
python benchmark/gold/heldout_a4_screen.py flap
deactivate
python benchmark/gold/heldout_a4_screen.py dasm
python benchmark/gold/heldout_a4_screen.py score
echo "DONE heldouta4"
