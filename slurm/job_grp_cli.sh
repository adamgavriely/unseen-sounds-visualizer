#!/bin/bash
#SBATCH --job-name=grpcli
#SBATCH --output=logs/grpcli_%j.out
#SBATCH --error=logs/grpcli_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
# GROUP standalone step on the whole DEV (both parts, SHIP8 renders); answers must equal grp/group_answers_bench.json
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -f data/work/grp_cli_dev.json data/work/grp_cli_dev2.json
python -m src.stage6_visual_augmentation.group data/work/r13/SHIP8_proposed data/work/devcand/wav16 data/work/grp_cli_dev.json
python -m src.stage6_visual_augmentation.group data/work/r13dev2/SHIP8_proposed data/work/r13dev2/wav16 data/work/grp_cli_dev2.json
python - <<PY
import json
ref = json.load(open("benchmark/gold/grp/group_answers_bench.json"))
got = {**json.load(open("data/work/grp_cli_dev.json")), **json.load(open("data/work/grp_cli_dev2.json"))}
bad = [(c, k, v, ref.get(c, {}).get(k)) for c, d in got.items() for k, v in d.items() if ref.get(c, {}).get(k) not in (None, v)]
asked = sum(len(d) for d in got.values())
print("asked", asked, "mismatch", bad)
print("CLI", "PASS" if not bad and asked else "FAIL")
PY
