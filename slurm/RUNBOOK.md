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

---

## The test battery — Steps T1–T6

All six run from `~/MscProj` on the cluster. T1–T3 are the thesis result; T4–T6 are
what makes it defensible. Every one writes incrementally, so a requeued job resumes.

### A scoring rule you should know about before reading any number

The pilot exposed a flaw that would have invalidated the comparison: when a system
produced **no augmentation at all**, the judge still handed it **2/4** out of charity.
That both rewarded the gated system for showing nothing and denied it credit on clips
where showing nothing is the *correct* behaviour — which is the gate's entire claim.

Empty augmentations are now scored deterministically in code, not by the judge
(`src/stage7_evaluation/protocol.py`, `judge()`):

| reference says | system showed nothing | score |
|---|---|---|
| `nothing beyond the picture` | correct silence | **4** |
| names a sound the viewer is missing | total miss | **0** |

Silence is never partially correct. The results table now also prints a **silent %**
and **right to be %** per system, so a system cannot post a respectable mean by
abstaining on the clips where abstaining happens to pay off without that being visible.

---

### T1 — re-score the pilot under the corrected rule (~10 min)

No rendering, no vision: the cached descriptions are unchanged, only the scoring rule
moved.

```bash
sbatch --export=PHASE=judge,RESCORE=1 slurm/job_protocol.sh
```

**Want to see:** the `proposed` records that read `no augmentation was shown` now score
0 or 4, never 2, and the three systems no longer sit on top of each other.

### T2 — balanced pilot, 12 clips × 3 systems (~40 min)

The first pilot silently measured one scenario: clips were taken alphabetically and
`mixed/` sorts first, which flatters the blind baseline because the gate suppresses
sounds on mixed clips by design. Sampling is now stratified round-robin across the
four human tags with a fixed seed.

```bash
sbatch --export=LIMIT=12,GEN=retrieve slurm/job_protocol.sh
```

**Want to see:** 3 clips from each of the four tags, scores that *vary*, and `why` text
that names the actual sound instead of praising generically. If all three systems tie,
the protocol is not measuring the systems — fix the prompt wording in
`src/stage7_evaluation/protocol.py` (the three prompts are module constants for exactly
this reason) before scaling up.

### T3 — the main experiment, 100 clips × 3 systems (~6–8 h)

```bash
sbatch --export=LIMIT=100 slurm/job_protocol.sh
```

100 clips gives roughly ±10% per mean, enough to separate three systems; 274 costs two
days for precision the argument does not need. This produces the table the thesis turns
on, per system and per proposal §5.1 scenario.

### T4 — judge reliability, second judge (~30 min, after T3)

One 7B model deciding every score is the protocol's softest point. A second, unrelated
judge re-scores the **same cached descriptions** — no rendering, no vision at all,
which is why the run is split into two passes:

```bash
sbatch slurm/job_judge2.sh
```

**Want to see:** Spearman ρ and quadratic κ between the judges. High agreement means
the scores are a property of the augmentations; low agreement bounds how far any
LLM-judged number in this area can be trusted, which is itself a finding.

### T5 — ablation: generate or retrieve? (~4 h, after T3)

The proposal names SDXL, but nobody has checked that *generating* an image beats
*retrieving* a stock one for the same label. If retrieval ties, that is worth stating:
it removes the GPU dependency from an accessibility tool meant to run on ordinary
hardware.

```bash
sbatch --export=LIMIT=100 slurm/job_ablation_gen.sh
```

### T6 — read the results (no GPU, on your PC)

```bash
bash slurm/fetch_results.sh
python scripts/compare_runs.py protocol_results.json protocol_results_judge2.json
python scripts/compare_runs.py protocol_results.json protocol_results_retrieve.json
```

`compare_runs.py` pairs on (clip, system), so a half-finished run cannot flatter itself,
and it prints the largest disagreements with their reference and description — those
records are the ones worth reading by hand.

---

## Order to run them

`T1` now (it is 10 minutes and tells you whether the fix worked) → `T2` → read the
records → `T3` overnight → `T4` and `T5` in parallel the next morning → `T6`.

Only T1 and T2 gate everything else. If T2 shows a judge that cannot discriminate,
stop and fix the prompt rather than spending eight hours on T3.

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
