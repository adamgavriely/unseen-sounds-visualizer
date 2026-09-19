# Night report, 19→20 September 2026 (Adam asleep; decisions with Fable panels, marked "(mine)")

Plain-English summary first, details below. Everything here is verified from logs and
result files; nothing is estimated.

## What happened, in order

1. **The v4ab row died and was restarted.** The final PSED-0.15 + Qwen3.8 row (job 30722392,
   3× L4) failed after 4 h at clip ~60: the three 24-GB cards were too small for the 27B
   model plus the picture generator (9 out-of-memory events), and at the same moment the
   home disk hit 100 % because a SAM-Audio load I started pulled a hidden 15-GB "judge"
   model. The disk part was my mistake. I freed 26 GB (re-downloadable caches only:
   Qwen2.5-VL, PixArt, the partial judge, an old Demucs cache) and resumed the row on an
   A100 (job 30723676; 60/100 clips were already stamped). Twin submissions on L4 and A100
   both started at once; I cancelled the L4 copy by hand.
2. **SAM-Audio is out.** Three Fables (engineer, project manager, examiner) agreed: it is the
   SOTA for open-vocabulary prompts, but for two fixed classes (speech, music) a stem model
   is stronger and has no 2023-era dependencies. Amendment 3 in `docs/prereg_detector_v5.md`:
   HTDemucs (`htdemucs_ft`, already installed on BIU) makes the views instead.
3. **V4 (cleaned-audio views) ran end to end on BIU** — no pod needed. 391 clips separated on
   one L4 in 25 min, PSED scored all four versions, selection on the calibration set.
   **Result: both pre-registered go/no-go checks fail → V4 is not run on slice B.**
   - Separation QC clean (lag 0, deterministic, no NaN; 0/17 gold sounds over-removed).
   - AUROC of the candidate score: original 0.762 → best cleaned 0.765 (needed +0.02).
   - At matched 2.6 false spans/min: 68.3 % vs 64.3 % of important sounds (+4), but the two
     held-out halves say +8 and 0 (needed ≥ +2 on both).
   - Correction made before selection: PSED at 0.15 alone already makes 5.4 false
     spans/min on the calibration set, so the base bar had to be on a grid (as fusion_v5
     did); control = PSED at its loosest bar under the target (0.28).
   - Thesis sentence written in the prereg. **The detector work is closed: thirteen
     pre-registered attempts, PSED at 0.15 stays.**
4. **Stage-2 swap check (SAM 3 vs OWLv2), declared then run** on the DCASE visibility set
   (same 258 events as the Qwen runs): SAM 3 agreement 66.8 %, OWLv2 69.2 %; AUROC 0.767 vs
   0.798. SAM 3 is more careful (off-screen recall 90.8 % vs 75.6 %) but misses 58 % of
   on-screen sources. **By the declared rule OWLv2 stays; the "2" swap is dropped.** Qwen3.8
   reasoning (69.8 %) remains the strongest visibility witness.
   (Engineering note: the first run was on CPU because `config.DEVICE` defaults to "cpu";
   fixed with a `--device`/auto-cuda switch, jobs resumed from saved records.)
5. **RunPod.** The "migrated" pod came up with an empty volume (pod volume disks are tied to
   one machine) and has since disappeared; the original pod `easy_sapphire_tick` (AP-IN-1,
   stopped) still holds the 280 GB of weights and envs. A network volume was created and
   deleted (US-TX-3 had no H100 free); EU-FR-1 does not allow network volumes. Nothing is
   deployed. Lessons written to memory (env vars live in `/proc/1/environ`; `runpodctl stop`
   needs `RUNPOD_API_KEY`).

## Numbers table (100-clip benchmark, rubric-enforced judge)

| row | gated | blind | caption | gated − blind [95 % CI] |
|---|---|---|---|---|
| v3 (BEATs, Qwen2.5-VL, FLUX) | 2.88 | 2.80 | 2.67 | +0.08 [−0.15, +0.31] |
| v4b (+ Qwen3.8-27B gate) | 2.85 | 2.75 | 2.66 | +0.10 [−0.13, +0.33] |
| v4ab (+ PSED 0.15) | running on the A100 (job 30723676) | | | |

## Decisions taken tonight (mine, with Fable)
- Resume v4ab on the A100 instead of waiting for a pod (same experiment, same code, resume from stamps).
- Replace SAM-Audio by HTDemucs (amendment 3) — unanimous panel.
- Base-bar correction in the V4 selection (declared as a correction before slice B).
- Run the stage-2 SAM 3 check on the free L40S (declared first).
- No pod deployed; no cluster deletions beyond re-downloadable caches.

## For the morning (with your annotations)
1. v4ab result → `rubric_enforce.py` → `v4_table.py` (I will do it when the job ends).
2. Per-sound scoring on your export: `python benchmark/gold/score_per_sound.py --annotations benchmark/gold/annotations/<name>.json --tag v4ab` (and slice B) → F1-strict, F1-phantom, weighted, CI, human-audible ceiling.
3. Gemma-4 second judge: needs an 80-GB card (A100/H200 queue) or the pod.
4. Decide v4c (Qwen-Image) — needs the pod or the 200 GB disk expansion.

**TLDR:** V4 fails its own go/no-go on calibration (cleaning does not help a frozen detector at
matched false alarms) → detector work closed at PSED 0.15. SAM 3 does not beat OWLv2 → stage-2
swap dropped. v4ab is being re-run on an A100 after the L4/disk failure. No pod used.

**Actions to decide (morning):** (1) export the annotations; (2) pod for Gemma judge + v4c, or wait for the BIU disk expansion.
