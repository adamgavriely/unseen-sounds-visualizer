# Gold re-run, 22 September 2026 — running log

Adam exported his per-sound annotations at 09:23 UTC (`benchmark/gold/annotations/gold_AG.json`,
139 usable clips) and asked for: per-stage tests (two Fables × two rounds each, problem statement
only), the models that work best on our data, then the full pipeline against the baselines with a
clear, significant win — everything documented; SOTA models tested against the older ones; every
GPU used; no waiting for his replies. Every decision below was taken with Fable consultation
(recorded here) or is marked "(mine)". Declared protocol: `docs/prereg_v4.md`, amendment 5.

## 1. The gold set (read once, before anything was scored)

| | clips | needed sounds ≥ 2 | visible/obvious ≥ 2 |
|---|---|---|---|
| all done, not bad | 139 | 110 (28 rated 3) in 64 clips | 139 |
| benchmark clips (headline population) | 109 | 79 | 90 |
| — of which old judge set (DEV) | 49 | 36 | 42 |
| — tagged after the freeze (TEST, benchmark part) | 60 | 43 | 48 |
| slice B (AudioSet-Strong, external check) | 30 | 31 | 49 |
| categories by his ticks | 33 unseen · 32 mixed · 44 seen · 30 no-ambient | | |

**Corrected 15:10** (advisor review): the first counts (103/54/49/36; 132 needed) were taken
over done clips *including* the 20 bad ones, and the scorer read only the *obvious* tick for
"needed" (22 visible-only rows were counted as needed — the direction that hides the gate's
effect). Both fixed before any TEST number was read; recorded in `docs/prereg_v4.md`.

- 25 free-text sound names had no ontology match → alias table `ALIASES` in
  `benchmark/gold/score_per_sound.py` (machinegun → Machine gun, tank shot → Artillery fire, keys
  jiggle → Keys jangling, golf swing → Whack, thwack, …), written before any score.
- 12 gold rows carry labels the depictable filter never draws ("Sound effect", "Generic impact
  sounds", "Drum", "Wind", "Mechanisms", top-level "Animal"); 4 of them are needed ≥ 2. They are
  neither hit nor miss (disclosed).
- The export also dumps the tool's UI state into the clip list (11 non-dict entries) — skipped.
- Earlier export (2026-09-20, 47 done clips) kept as `gold_AG_2026-09-20.json`; today's is a superset.

## 2. Protocol (Fables A + B, two rounds, converged; advisor review)

- **Headline** = the pre-registered configuration v4b4 (BEATs, bar 0.35, Qwen3.8-27B gate,
  majority of 3, depictable filter, 8-s cap, FLUX) rendered for real on all 139 clips, scored on
  the 103 benchmark clips: per-sound F1-strict, proposed vs blind_a2i, **paired** clip-bootstrap CI
  of ΔF1; SILENCE and audio_caption rows beside; breakdowns old-54 / new-49 / pre-screened, by
  category, slice B separate, all-139 pooled supplementary.
- **Knob selection** only on DEV-54 (the clips every earlier run already touched); TEST-85 opened
  once; the tuned variant is a secondary row, never the headline. Shared stages (detector, display
  bar, corroboration) are chosen by a gate-blind criterion (the BLIND system's own DEV F1 /
  detection F1 against all gold sounds); gate-rule changes (unanimous, "obvious" question) are
  treatment tuning → declared secondary rows.
- GATE is rendered, not derived from BLIND by masking (stretch cuts, scene-shaped prompts, panel
  packing differ). The render now logs the raw votes per sound per stretch (`gate_votes.json`) so
  the silence rule can be re-decided on CPU.
- PSED arm (v4ab4) rendered in parallel under the unchanged arm rule.

## 3. Jobs (BIU, 13:05–13:30)

`slurm/job_gold.sh` (new): one system per job, clips in shards (`data/input/gold139/{p1,p2,p3,b1,b2,all}`,
symlinks), work dirs share the tag. Composite mp4 now written to `data/output/<system>_<tag>/`
(two rows rendering the same clip collided on `data/output/<stem>_augmented.mp4`; harmless for
scoring — `augmentations.json` is written first — fixed for later jobs).

| row | system | shard | partition | job |
|---|---|---|---|---|
| v4b4 | proposed | p1 / p2 / p3 | A100-4h / A100-4h / generic-48G (2× RTX 6000) | 30967487 / 30967488 / 30967606 |
| v4b4 | blind_a2i | b1 / b2 | H200-4h / L4-4h | 30967490 / 30967538 |
| v4b4 | audio_caption | all | L4-4h | 30967540 |
| v4ab4 | proposed | p1 / p2 / p3 | A100-4h ×2 / generic-48G | 30967506 / 30967507 / 30967607 |
| v4ab4 | blind_a2i | b1 / b2 | H200-4h / L4-4h | 30967509 / 30967539 |
| v4ab4 | audio_caption | all | L4-4h | 30967541 |
| gate accuracy on gold sounds | OWLv2 / Qwen2.5-VL-7B / Qwen3.8-27B | — | L4-4h / L4-4h / H200-4h + B200-4h | 30967604 / 30967698 / 30967699 + 30967700 |
| v4b4 proposed, extra sub-shards (A100 pace ~6 min/clip would overrun 4 h) | p1c / p2c / p3a / p3b | H200 / H200 / B200 / H200 | 30967667 / 30967668 / 30967701 / 30967670 |

Disk: the home quota went from 200 to 400 GB at ~13:20 (223 GB free) — verified with `df`.
Downloaded on the login node (`hf download`; `huggingface-cli` no longer works): Qwen2.5-VL-7B-Instruct
(the v3 gate, for the SOTA-vs-old gate test), google/gemma-4-31B-it (the SOTA judge, was blocked
by disk) — both complete by 13:50. H200 allows 2 jobs per user, L4 4 GPUs per user.

## 4. Per-stage tests

### 4a. Scorer QA on the pre-fix v4b3 renders, DEV-54 only (49 clips with renders, 44 needed sounds)

| system | P | R | F1 [CI] | hits | miss | visible FA | cross | phantom |
|---|---|---|---|---|---|---|---|---|
| proposed | 0.23 | 0.27 | 0.25 [0.15, 0.35] | 12 | 32 | 8 | 20 | 12 |
| blind_a2i | 0.20 | 0.34 | 0.25 [0.15, 0.35] | 15 | 29 | 16 | 25 | 19 |
| audio_caption | 0.19 | 0.34 | 0.24 | 15 | 29 | 16 | 28 | 20 |
| silence | 0 | 0 | 0.00 | 0 | 44 | 0 | 0 | 0 |

ΔF1 proposed − blind = −0.002 [−0.066, +0.058] (paired). Anatomy (the story): the gate halves
the visible-source pictures (16 → 8) but loses 3 hits (an off-screen bird called "visible"); the
false-alarm pool is dominated by cross/phantom pictures both systems share (detector label
hallucinations — "Cooking" on a cow farm, "Cat"/"Steam" at a rail crossing, "Owl" in a pet shop —
and correct pictures starting 1.2–1.6 s after the gold onset, which the 1-s window counts as
FA + miss). Most misses: the detector never fired the family (hammer, clang, whistle, honk,
footsteps, door at 0.07).

### 4b. Levers to widen the margin honestly (Fables C + D, two rounds)

Consensus: (1) shared stages — detector, display bar, two-detector corroboration (BEATs ∧ PSED
same family, overlapping) — may be chosen only by the BLIND system's own DEV F1, never by ΔF1;
corroboration uses no visual input so it is detection, not a second gate, and it can only shrink
the gate's headroom; (2) gate-rule variants (unanimous; "majority + obvious", a fourth VLM
question that implements the annotator's "obvious" tick) are treatment tuning → declared
secondary rows, with the hits-lost count beside them; (3) no third full render before the two
pre-registered rows finish; the tuned stack is reported dry (planned draw list) with the caveat
that dry scoring cannot count picture-level failures; (4) the honest headline may be "no
significant pooled ΔF1, visible pictures halved, hits lost, precision gains on seen-only and
no-ambient clips" — the anatomy table goes beside the F1 row. Ceiling estimate on DEV: ΔF1 ≈
+0.04–0.06.

### 4c. Stage 4 — detector dry run (`benchmark/gold/detector_dry.py`) — DONE 16:10
BLIND shown set (detector → depictable filter → families → display timeline, no pictures), the
per-sound rules, corrected "needed". Verified exact: on 12 DEV clips the dry BEATs@0.35 set equals
the real v4b4 blind render picture for picture. DEV = 49 clips, 33 scorable needed sounds;
slice B = 32 clips / 29. (One PSED pass had printed pooled subsets before the DEV-only
restriction — disclosed above.)

| detector (bar) | DEV P | DEV R | DEV F1 | vis / cross / phantom | slice B F1 |
|---|---|---|---|---|---|
| **BEATs 0.35 (declared)** | 0.13 | 0.33 | **0.19** | 20 / 33 / 18 | 0.14 |
| BEATs 0.25 / 0.30 / 0.40 | | 0.42 / 0.36 / 0.33 | 0.18 / 0.19 / 0.21 | | 0.14 / 0.14 / 0.14 |
| PSED raw 0.15 (declared arm) | 0.17 | 0.52 | 0.25 | 25 / 41 / 20 | 0.24 |
| PSED raw 0.10 / 0.20 / 0.25 / 0.30 | | 0.52 / 0.42 / 0.42 / 0.42 | 0.21 / 0.24 / 0.26 / **0.31** | 0.30: 20 / 15 / 8 | 0.16 / 0.27 / 0.24 / 0.22 |
| PANNs CNN14 (v1) 0.25 / 0.30 / 0.35 / 0.40 | | 0.30 / 0.30 / 0.27 / 0.24 | 0.20 / 0.22 / 0.24 / 0.23 | 0.35: 11 / 15 / 7 | 0.21 / 0.19 / 0.15 / 0.13 |
| BEATs ∧ PSED corroboration 0.25–0.40 | | 0.27 / 0.27 / 0.24 / 0.24 | 0.17 / 0.19 / 0.18 / 0.19 | 0.35: 18 / 21 / 7 | 0.17 / 0.17 / 0.18 / 0.16 |

Reading: the SOTA frame-level detector (PSED) has the best recall of needed sounds (0.52 vs
0.33) and, at raw bar 0.30, the best DEV F1 (+0.12 over the declared BEATs 0.35 — larger than the
DEV bootstrap half-width ≈ 0.10). Corroboration by two detectors cuts phantoms (18 → 7) but costs
recall and does not raise F1 — dropped. PANNs (v1) is between.

**Cross-fit (`benchmark/gold/crossfit.py`, 5 folds stratified by category over all 139, config
picked on 4 folds by blind F1, scored out-of-fold):** picks per fold = psed@0.20, corr@0.40,
psed@0.30, psed@0.20, psed@0.20 — unstable; pooled out-of-fold F1 0.229 vs declared BEATs 0.228,
ΔF1 +0.001 [−0.068, +0.071] (seed 1: +0.003 [−0.055, +0.062]). **The DEV gain does not survive the
cross-fit → by the declared rule the detector stays BEATs 0.35 for the headline; PSED remains the
pre-registered arm (v4ab4, raw bar 0.15) and this table is the stage-4 sensitivity report.**
(Config chosen on all 139 would be psed@0.20, F1 0.256 — reported, not adopted.)

### 4d. Stage 5 — gate accuracy on the gold sounds (`benchmark/gold/gate_gold.py`)
Every gold sound (label, time) is put to the visibility check exactly as the pipeline asks it;
verdict vs the annotator's visible/obvious tick. Arms: Qwen3.8-27B (SOTA, v4), Qwen2.5-VL-7B
(v3), OWLv2 concept table (v2). Rules re-decided on CPU: majority, unanimous, majority+obvious,
obvious. _Results: pending._

### 4e. Stage 7 — judge (Mistral-7B rubric-enforced vs Gemma-4-31B) — after the renders, if time.

## 5. Full run — _pending_
