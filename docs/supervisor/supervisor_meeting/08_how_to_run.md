# How to run the system on one video

The pipeline needs a GPU with ~24 GB for the evaluated configuration (Qwen2.5-VL-7B and
FLUX.1); on CPU it runs but takes tens of minutes per clip. The models download from Hugging
Face on first use (~50 GB; FLUX.1 requires accepting its licence and a token in `HF_TOKEN`).

## On the university cluster (what the thesis numbers used)

```bash
ssh adamg@slurm-login1.lnx.biu.ac.il
cd MscProj
source ~/miniconda3/etc/profile.d/conda.sh && conda activate msproj
srun --partition=L4-12h --gres=gpu:1 --mem=64G --time=01:00:00 --pty bash   # a GPU shell
python main.py --input data/input/benchmark/mixed/ambient_citywalk_nyc_2627.mp4 \
               --device cuda --video-backend owlv2 --generator diffusion
```

Output: `data/output/<clip>_augmented.mp4` (the video with the side panel) and the
intermediate files under `data/work/<clip>/` (detections, gate decisions, phrases, pictures).

Add `--debug-panel` to print the phrase and every raw detection with the gate's verdict
under the panel (the `debug_*` demo videos were made this way).

## On a laptop (CPU, slow but works)

```bash
conda activate msproj
python main.py --input path/to/video.mp4 --device cpu --video-backend owlv2 --generator placeholder
```

`--generator placeholder` skips image generation (a labelled box instead of a FLUX picture),
which is the only stage that is impractical on CPU.

## What the flags mean

| flag | evaluated setting | note |
|---|---|---|
| `--video-backend` | `owlv2` | Stage 2 object finder; the config default is `siglip` (older) |
| `--generator` | `diffusion` (FLUX.1-schnell) | `placeholder` for a quick run, `retrieve` for stock photos |
| `--whisper-model` | `base` | speech is used as context only, never drawn |
| `--device` | `cuda` | `cpu` works |

Everything else (detector bar 0.35, three-vote visibility, 5-second stretches, dedup, panel
dwell 1.5 s) is fixed in `config.py` and is the configuration the thesis reports (v3).

## Re-creating the demo videos

```bash
sbatch slurm/job_demos_v3.sh      # ~12 minutes on an L4; clean + debug for the three Figure 2 clips
```

## Reproducing the headline numbers (no GPU)

```bash
python scripts/paired_stats.py v3 v3_grounded   # 3.17 vs 3.16, the scenario split, oracle 3.62
python scripts/oracle_gap.py v3_grounded         # distance to the perfect gate by scenario
python scripts/cost_metrics.py                   # panel-on time, wasted seconds
python scripts/cost_sensitivity.py v3_grounded   # judge-free hit/miss/redundant view
```
