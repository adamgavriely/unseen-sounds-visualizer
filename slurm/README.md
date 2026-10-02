# slurm — cluster jobs

**Final system.** `run_best.sh NAME /path/to/clips` runs stages 1–5 of the final system on a folder of clips
(default variant `SHIP8+MD3+WW5+SL`); the report (Section 10) lists the three follow-up scripts that draw and
compose the pictures. `job_trail_media.sh` renders the Decision Inspector videos. The other ~230 scripts are
development runs, one per experiment, kept so that every result in `docs/history/` can be traced to the job that
made it.

---

# Running this project on the BIU Slurm cluster

Login: `adamg` · VPN user: `gavriea2@biu.ac.il` · reset password via
<https://sspr.biu.ac.il/sspr/public/forgottenpassword?locale=en> (also affects Moodle/Inbar).

## Partitions we use (from *Slurm usage BIU v1.4*)

| Partition | GPU | Max time | Use for |
|---|---|---|---|
| `L4-4h` | L4 | 4 h | debugging, smoke tests (**always start here**) |
| `A100-4h` | A100 | 4 h | medium batches |
| `H200-12h` | H200 | 12 h | the full diffusion / VLM benchmark runs |

Limits: 4 jobs per user, max 2 GPUs per job, default 1 CPU + 16 G RAM unless asked.
`sbatch` jobs are **suspended and requeued** at the time limit — every script here
writes results incrementally so a requeued job resumes instead of restarting.

## One-time setup

```bash
# 0. connect the BIU VPN first, then from your PC:
ssh adamg@slurm-login1.lnx.biu.ac.il

# 1. get the code
git clone https://github.com/adamgavriely/MscFinalProject.git ~/MscProj
cd ~/MscProj

# 2. build the conda env (CPU login node; installs CUDA torch)
bash slurm/setup_env.sh
```

Then from your **local PC** (Git Bash), upload the benchmark clips (gitignored):

```bash
bash slurm/sync_data.sh
```

## Daily workflow

```bash
sbatch slurm/job_smoke.sh                 # 1. verify GPU + env      (L4, ~2 min)
sbatch slurm/job_eval_vlm.sh              # 2. VLM gate vs CLIP gate (A100, ~1-2 h)
sbatch slurm/job_diffusion.sh             # 3. SDXL augmentations    (H200, ~2-4 h)

squeue -u $USER                           # watch
tail -f logs/<jobname>_<jobid>.out        # follow output
scancel <job_id>                          # stop
```

Results land in `benchmark/eval_results*.json` and `data/work/`; pull them back with
`slurm/fetch_results.sh` from your PC.

## Why these jobs matter

`job_eval_vlm.sh` produces the **before/after table** (CLIP gate vs Qwen2.5-VL gate on the
same benchmark) — the headline result. `job_diffusion.sh` replaces retrieved Openverse
images with generated ones (v2-b), for the qualitative comparison.
