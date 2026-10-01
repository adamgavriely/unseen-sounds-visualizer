#!/bin/bash
#SBATCH --job-name=expecttest
#SBATCH --output=logs/expecttest_%j.out
#SBATCH --error=logs/expecttest_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#
# Round 40e TEST read (docs/prereg_round13_detector_push.md): EXPECT-A4 frozen, on merged TEST; gold read only in score. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/expect_test.py parts
python benchmark/gold/expect_test.py listen
python benchmark/gold/expect_test.py cands
source "$HOME/venv_flap/bin/activate"
python benchmark/gold/expect_test.py flap
deactivate
python benchmark/gold/expect_test.py dasm
python benchmark/gold/expect_test.py gate
python benchmark/gold/expect_test.py score
echo "DONE expecttest"
