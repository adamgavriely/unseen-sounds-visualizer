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
| all done, not bad | 139 | 132 (38 rated 3) in 73 clips | 117 |
| benchmark clips (headline population) | 103 | | |
| — of which old judge set (DEV-54) | 54 | 44 (49 clips have renders) | |
| — tagged after the freeze (TEST, benchmark part) | 49 | | |
| slice B (AudioSet-Strong, external check) | 36 | | |
| categories by his ticks | 46 unseen · 35 mixed · 46 seen · 31 no-ambient | | |

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
| gate accuracy on gold sounds | OWLv2 / Qwen3.8-27B | — | L4-4h / A100-4h | 30967604 / 30967605 |

Disk: the home quota went from 200 to 400 GB at ~13:20 (223 GB free) — verified with `df`.
Downloads started on the login node: Qwen2.5-VL-7B-Instruct (the v3 gate, for the SOTA-vs-old
gate test), google/gemma-4-31B-it (the SOTA judge, was blocked by disk).

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

### 4c. Stage 4 — detector dry run (`benchmark/gold/detector_dry.py`)
BLIND shown set (detector → filter → families → display timeline, no pictures) for BEATs, PANNs
CNN14 (v1) and PretrainedSED at bars 0.25–0.40, scored per sound on DEV-54, slice B and all.
_Results: pending._

### 4d. Stage 5 — gate accuracy on the gold sounds (`benchmark/gold/gate_gold.py`)
Every gold sound (label, time) is put to the visibility check exactly as the pipeline asks it;
verdict vs the annotator's visible/obvious tick. Arms: Qwen3.8-27B (SOTA, v4), Qwen2.5-VL-7B
(v3), OWLv2 concept table (v2). Rules re-decided on CPU: majority, unanimous, majority+obvious,
obvious. _Results: pending._

### 4e. Stage 7 — judge (Mistral-7B rubric-enforced vs Gemma-4-31B) — after the renders, if time.

## 5. Full run — _pending_
