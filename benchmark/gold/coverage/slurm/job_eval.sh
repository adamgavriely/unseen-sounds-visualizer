#!/bin/bash
#SBATCH --job-name=s5eval
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/s5_eval_%j.out
# Step 5: fine-tuned ear on DEV + the 415 held-out, then tier 1 (results in ~/wt_slice/benchmark/gold/coverage/ear_tier1_dev.md).
set -euo pipefail
cd ~/MscProj_tg
L=~/open_data/ft/atstf_ft_s0/log.json
grep -q best_epoch $L || { echo "training not finished"; exit 1; }
E=$(python3 -c "import json;print(json.load(open('$L'))['best_epoch'])")
export PATH=~/miniconda3/envs/sota/bin:$PATH PYTHONUNBUFFERED=1
for s in dev heldout; do
  ~/venvs/psed2/bin/python benchmark/gold/coverage/ear_cache.py atstf_ft_s0 ATST-F ~/open_data/ft/atstf_ft_s0/epoch_$E.pt --set $s
done
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
cd ~/wt_slice
HELDOUT_JSON=~/wt_slice/audioset_heldout.json python benchmark/gold/coverage/ear_tier1.py \
  ~/MscProj_tg/scratch_cov/ears/atstf_orig ~/MscProj_tg/scratch_cov/ears/beatsS_orig ~/MscProj_tg/scratch_cov/ears/atstf_ft_s0
echo "best epoch $E"; echo S5_EVAL_DONE
