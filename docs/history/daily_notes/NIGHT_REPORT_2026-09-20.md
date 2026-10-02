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
   is stronger and has no 2023-era dependencies. Amendment 3 in `docs/history/preregistrations/prereg_detector_v5.md`:
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
| **v4ab (+ PSED 0.15)** | **2.60** | 2.46 | 2.46 | +0.14 [−0.13, +0.40] |

## 6. v4ab landed — and PSED is NOT adopted (pre-registered rule)

Gated v4ab 2.60 vs v4b 2.85; paired difference −0.25, 95 % CI [−0.53, −0.01] → by the
detector-arm rule BEATs stays. All three systems fell (blind 2.75 → 2.46, caption 2.66 → 2.46),
so the loss enters through the detector output, not the gate; the gate's own margin held
(+0.14 vs +0.10). The judge ran with Mistral (same as every row) because Gemma-4 is only
partly downloaded and filled the disk twice when the job tried to fetch it.

Why (read on the outputs afterwards — disclosed as post hoc): not "PSED fires more" (it fires
*fewer* spans than BEATs on the calibration set: 12.3 vs 15.0/min). Two integration defects:
(1) the "salient non-speech" label filter was written for BEATs' label names, so PSED's
AudioSet-Strong names leak through — "Breathing" ×17, "Video game sound" ×15, "Human voice"
×8, "Laughter" ×6 pictures in the blind system that should never exist; (2) pictures stay
up much longer (median span 10.0 s vs 4–5.7 s) because the hysteresis low bar (bar/2) and
no onset refinement let PSED spans run through the clip. A draft row **v4ab2** with two
detector-agnostic fixes (filter by ontology *branch*; span ends after 1 s below the bar +
a cap from the calibration set), applied to both detectors, is written in
`docs/history/preregistrations/prereg_v4.md` — **not run**: it waits for your approval (one product question: keep
baby cry / footsteps?) and for the per-sound F1 gate (run v4ab2 only if PSED's F1-strict on
your annotations ≥ BEATs' on the filtered classes).

Also queued: the adopted v4b configuration rendered on slice B (job 30724805, after the
v4ab one) so both rows get per-sound scores on both gold sets.

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

**TLDR:** v4ab (PSED 0.15) scores 2.60 vs v4b 2.85 → PSED not adopted by the rule; cause is two
integration bugs (label filter, span rule), fix drafted as v4ab2 and gated on your F1. V4 cleaning
fails its go/no-go → detector attempts closed. SAM 3 does not beat OWLv2 → stage-2 swap dropped.
No pod used.

**Actions to decide (morning):** (1) export the annotations → per-sound scores (both rows, both gold sets);
(2) v4ab2: approve the draft and choose Option A/B for human sounds (recommended B: keep baby cry, footsteps);
(3) Gemma judge + v4c: pod or wait for the BIU disk expansion.
