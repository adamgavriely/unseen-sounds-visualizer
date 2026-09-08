# GPU runbook — what to run, in order, and why

Goal for this session: a **minimum viable product** to show the advisor — one number
that decides whether the gate is salvageable, and a handful of watchable augmented
videos. Everything below is already committed; you only run commands.

Login: `adamg@slurm-login1.lnx.biu.ac.il` (BIU VPN must be connected first).

---

## Step 0 — one-time setup (~15 min, mostly waiting)

**On the cluster:**
```bash
ssh adamg@slurm-login1.lnx.biu.ac.il
git clone https://github.com/adamgavriely/MscFinalProject.git ~/MscProj
cd ~/MscProj
bash slurm/setup_env.sh          # conda env + CUDA torch + diffusers/OWLv2/Qwen
```

**Then on your PC (Git Bash, VPN on):**
```bash
cd /p/MscProj
bash slurm/sync_data.sh          # uploads the 274 benchmark clips + tags.json
```
The clips are gitignored, so this step is required — the cluster clone has code only.

---

## Step 1 — smoke test (~5 min queue + 2 min run)

```bash
sbatch slurm/job_smoke.sh
squeue -u $USER                  # watch it
cat logs/smoke_*.out             # read it when done
```

**Want to see:** GPU name, `torch cuda True`, the clip inventory, and
`SMOKE TEST PASSED`. If this fails, nothing else will work — send me the log.

---

## Step 2 — THE experiment: four gates, threshold-free (~2 h)

```bash
sbatch slurm/job_eval_vlm.sh
```

Runs the identical benchmark and identical human labels through four Stage-2
backends and prints one comparison table:

| backend | what it is |
|---|---|
| CLIP | the original baseline |
| SigLIP | independent per-concept scoring |
| **OWLv2** | open-vocabulary **detection** (localises the object) |
| **Qwen2.5-VL** | a full VLM that reasons over the scene |

Then it prints **AUROC** for each — the threshold-free measure of whether the
visibility score carries information at all.

**The number that matters.** On CPU, SigLIP scored **AUROC 0.647** (chance = 0.5).
- If OWLv2 or Qwen2.5-VL reach **≥ 0.75** → the gate is genuinely good, the ceiling
  was the model, and the thesis has a positive headline result.
- If all four sit around **0.65** → that is the finding: off-the-shelf vision-language
  models cannot answer *event visibility*, only *object presence*. A clean, defensible
  negative result, and it is already written up in `docs/project_notes.tex`
  §Results and Findings.

Either outcome is publishable. This is why it runs first.

---

## Step 3 — demo renders for the advisor (~1–2 h, can run at the same time)

```bash
sbatch slurm/job_diffusion.sh
```

Runs the full pipeline with the best gate + **SDXL** generation over 12 clips
(4 unseen / 4 mixed / 4 seen) and writes side-by-side augmented videos to
`data/output/`. The `seen_ambient` ones matter as much as the positives: they show
the system correctly staying silent, which *is* the contribution.

Fetch them to your PC when done:
```bash
bash slurm/fetch_results.sh      # run this on your PC
```

---

## Monitoring

```bash
squeue -u $USER                  # your jobs
tail -f logs/eval_vlm_<jobid>.out
scancel <jobid>                  # stop one
```

Jobs are suspended and requeued at the partition time limit, and every script writes
results incrementally — a requeued job resumes rather than restarting.

---

## What to bring to the advisor

1. **The comparison table + AUROC** from Step 2 — is the gate salvageable?
2. **2–3 augmented videos** from Step 3 — the prototype, actually working.
3. **`docs/project_notes.pdf` §Results and Findings** — the trivial-baseline ablation,
   the object-vs-event distinction, and the label-stability study (κ = 0.60).

Questions worth raising:
- The gate beats blind audio-to-image generation by +5.7 F1 but loses to "always stay
  silent" on accuracy. Is F1 against the blind baseline the right headline, given
  always-silent provides zero accessibility value?
- Benchmark frozen at 274 clips rather than 300: acceptable, given 15–20% usable yield
  measured across four independent sourcing strategies?
- Is "object presence ≠ event visibility" acceptable as the core contribution — a task
  formulation plus evidence that off-the-shelf models cannot satisfy it?
- Is a small DHH user study (10 participants × 10 clips) feasible, and what is the
  ethics/IRB timeline?
