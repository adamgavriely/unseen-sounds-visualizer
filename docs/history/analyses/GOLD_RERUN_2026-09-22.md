# Gold re-run, 22 September 2026 — running log

Adam exported his per-sound annotations at 09:23 UTC (`benchmark/gold/annotations/gold_AG.json`,
139 usable clips) and asked for: per-stage tests (two Fables × two rounds each, problem statement
only), the models that work best on our data, then the full pipeline against the baselines with a
clear, significant win — everything documented; SOTA models tested against the older ones; every
GPU used; no waiting for his replies. Every decision below was taken with Fable consultation
(recorded here) or is marked "(mine)". Declared protocol: `docs/history/preregistrations/prereg_v4.md`, amendment 5.

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
effect). Both fixed before any TEST number was read; recorded in `docs/history/preregistrations/prereg_v4.md`.

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
by disk) — both complete by 13:50; Qwen-Image-2512 (~57 GB) was also pulled for a possible
generator comparison and was **not** used. H200 allows 2 jobs per user, L4 4 GPUs per user.

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

### 4d. Stage 5 — gate accuracy on the gold sounds (`benchmark/gold/gate_gold.py`) — DEV, running
Every gold sound (its family label and time) is put to the visibility check exactly as the pipeline
asks it (six frames per 5-s stretch; three votes: open naming, a/b in both orderings, description),
and the verdict is compared with the annotator's visible-or-obvious tick. A fourth, separate
question implements the "obvious" tick. DEV only (49 clips, 79 sounds rated ≥ 2: 43 seen, 36 needed).

| gate | rule | seen silenced | needed kept | balanced acc |
|---|---|---|---|---|
| **Qwen3.8-27B (v4, SOTA)** | **majority of 3 (declared)** | **0.37** | 0.86 | **0.62** |
| Qwen3.8-27B | unanimous | 0.19 | 0.97 | 0.58 |
| Qwen3.8-27B | majority + obvious | 0.40 | 0.81 | 0.60 |
| Qwen3.8-27B | obvious only | 0.33 | 0.86 | 0.59 |
| Qwen2.5-VL-7B (v3) | majority | 0.33 | 0.89 | 0.61 |
| Qwen2.5-VL-7B | unanimous / obvious | 0.10 / 0.07 | 1.00 | 0.55 / 0.54 |
| OWLv2 concept table (v2) | — | 0.28 | 0.72 | 0.50 (chance) |

Qwen3.8 arm complete (49/49 DEV clips, 79 sounds rated ≥ 2: 43 seen, 36 needed); Qwen2.5-VL 48/49.
Reading: the declared **majority-of-3 stays** — no variant beats it by more than the DEV noise, and
"unanimous" barely silences anything (0.19). SOTA vs older: the 27B silences more seen sounds than
the 7B at a similar balanced accuracy (0.62 vs 0.61) and both are far above the v2 object-detector
gate, which is at chance (0.50) — that is the stage-2 → stage-5 change this project made, measured.
The gate's own ceiling is visible here: even the best arm silences only 37 % of the sounds whose
source the annotator could see, so most visible-source pictures survive by construction.

### 4e. Stage 7 — judge (Mistral-7B rubric-enforced; Gemma-4-31B now downloaded) — jobs 30968116/117, pending.

## 5. Headline run — v4b4 on the gold set (2026-09-22 16:20)

Configuration exactly as declared: BEATs 0.35, Qwen3.8-27B gate (majority of 3), depictable filter,
8-s picture cap, FLUX.1-schnell; all 139 clips rendered for real, three systems + SILENCE.
Population = the 109 benchmark clips (79 needed sounds rated 2–3); slice B separate; paired
clip-bootstrap (2000 draws, seed 0).

| | proposed (gated) | blind_a2i | audio_caption | silence |
|---|---|---|---|---|
| P / R / **F1** | 0.25 / 0.34 / **0.29** | 0.19 / 0.42 / **0.26** | 0.19 / 0.42 / 0.26 | 0 / 0 / 0 |
| F0.5 (declared secondary) | 0.27 | 0.21 | 0.21 | 0 |
| hits / misses | 27 / 52 | 33 / 46 | 33 / 46 | 0 / 79 |
| pictures of a visible source | **10** | 31 | 32 | 0 |
| cross-trigger / phantom | 47 / 23 | 75 / 34 | 77 / 34 | 0 / 0 |
| false alarms per clip | **1.65** | 2.20 | 2.23 | 0 |
| clean-clip accuracy | **0.55** | 0.38 | 0.38 | 1.00 |
| coverage (declared secondary) | 0.25 | 0.30 | 0.30 | 0 |

Paired differences, gated − blind (109 clips): **ΔF1 +0.028 [−0.022, +0.074]** (P(Δ>0) = 0.86,
not significant) · ΔP **+0.062 [+0.016, +0.109]** · ΔR **−0.076 [−0.141, −0.026]** · ΔFA/clip
**−0.55 [−0.78, −0.37]** (a third fewer) · Δ clean-clip accuracy **+0.172 [+0.082, +0.273]** ·
ΔF0.5 **+0.052 [+0.006, +0.098]** · ΔwF1 +0.006 (null). Against SILENCE: ΔF1 **+0.290
[+0.199, +0.377]**. Against CAPTION: ΔF1 +0.031 [−0.019, +0.078].
Other declared secondary rows: `--old-rule` (level-1 needed sounds scored as hits/misses) ΔF1
+0.025 [−0.024, +0.073]; the unanimous-silence re-decision from the logged votes (dry, approximate)
F1 0.28 vs 0.29 for majority, ΔF1 vs blind +0.022 [−0.020, +0.059] — no better than the declared
rule, so it stays a sensitivity row; coverage 0.25 vs 0.30 (the gate shows a picture for a smaller
share of each needed sound's seconds, as expected from its 6 false silences).
DEV (49) and TEST (60) agree in sign and size: ΔF1 +0.023 / +0.032, ΔFA/clip −0.59 / −0.52,
Δclean +0.23 / +0.13. Slice B (30 external clips): same false-alarm drop (−0.70/clip) and clean
gain (+0.25), no F1 gain (−0.020 [−0.100, +0.049]). +3-s late window moves nothing (ΔF1 +0.033).
On the 44 seen-only clips the gate stays silent on 43 % of clips vs 11 % (Δ +0.318 [+0.182, +0.455]).

**Honest headline (Fables E + F, and the advisor):** the gate did **not** significantly improve the
pre-registered per-sound F1 over the ungated pipeline (+0.03 [−0.02, +0.07]); it **significantly**
raised precision (+0.06), cut false alarms by a third and tripled clean-clip silence, at a
**significant** recall cost (−0.08); both systems beat SILENCE by a wide margin (+0.29). F0.5 stays
a labelled secondary row — it was declared before the run and is not promoted to the headline.

**Anatomy.** The gate removes two thirds of the pictures whose source is on screen (31 → 10). It
loses exactly 6 needed sounds, all listed: rainforest macaws visible while another bird calls; a
pet-shop bird; a visible bell while the ringing one is off screen; a visible machine gun;
"a tank is visible" → helicopter silenced; a helicopter silenced as "a kind of Vehicle". The
remaining false alarms (70 of 80) are detector label errors that both systems share and no gate can
remove — the measured ceiling of a gate on this detector, and the bottleneck for future work.

**Oracle-label diagnostic (post hoc, labelled; `benchmark/gold/oracle_label.py`).** Remove from
*both* systems every picture whose label names no sound really present at that moment — i.e. give
both a perfect detector vocabulary — and the gate's precision advantage triples (ΔP +0.134
[+0.056, +0.211] on the 109 clips; +0.166 on TEST; +0.200 on mixed clips) while ΔF1 stays +0.029
[−0.036, +0.091]. So the null F1 is **not** caused by the detector's label errors: it is the
equal-weight F1 itself. Each picture the gate silences removes one false alarm, and the few it
silences wrongly remove one hit; F1 treats those as the same size, so the two effects cancel by
construction. A metric that prices a misleading picture above a missing one (F0.5, or the DHH cost
model) is where the gate's benefit shows — stated here as a hypothesis for a future
pre-registration, not as a change of the headline.

**Disclosures.** 49 of the 109 clips (DEV) were used in earlier development; TEST-60 is the clean
estimate and is reported beside the pooled row. Single annotator. 13 clips came from the VLM
pre-screen (reported separately: ΔF1 −0.018 there). The caption row was rendered with placeholder
images (`GEN=placeholder`) because its pictures are never scored or judged — 163 placeholder panels,
counted and disclosed. 12 gold rows carry labels the depictable filter never draws. Seven secondary
rows were declared, so the one marginal secondary (F0.5, lower bound +0.006) would not survive a
multiplicity correction. Power: 79 needed sounds, so a true ΔF1 of about +0.03 is not excluded —
the CI is wide, not evidence of equality.

## 6. PSED arm (v4ab4) — DONE 17:20: **not adopted**, as the arm rule requires

Same configuration with PretrainedSED (raw bar 0.15) in place of BEATs, all 139 clips, three
systems. On the 109 benchmark clips: gated P 0.23 R 0.37 F1 0.28 vs blind P 0.18 R 0.44 F1 0.26;
ΔF1 gated − blind +0.023 [−0.020, +0.064], ΔP +0.046 [+0.008, +0.090], ΔFA/clip −0.53
[−0.69, −0.39], Δ clean-acc +0.155 [+0.066, +0.250] — the same pattern as the BEATs row.

**Pre-registered arm rule** (2026-09-19: adopt PSED iff the *gated* row beats the gated BEATs row
with the CI not below 0), paired over the same clips:

| set | ΔF1 gated PSED − gated BEATs |
|---|---|
| benchmark 109 | **−0.007 [−0.115, +0.098]** |
| TEST 60 | −0.063 [−0.198, +0.069] |
| slice B 30 | +0.043 [−0.075, +0.137] |

→ **PSED is not adopted; BEATs stays** — the third independent test that has said so (span bars
2026-09-19, judge rows 2026-09-20, per-sound F1 today), now on human per-sound gold. PSED buys
recall (0.37 vs 0.34 gated) and pays it back in precision (0.23 vs 0.25) and in clean-clip silence
(0.41 vs 0.55). Consistent with the stage-4 dry run, where PSED's DEV advantage did not survive
the cross-fit.

## 7. Judge (secondary metric, Mistral-7B) — DONE 17:45

All 139 clips × 3 systems described by Qwen3.8-27B and scored 0–4 by Mistral-7B against the
reference; the declared variant is the **grounded + rubric-enforced** judge (human tag from the
gold ticks; a non-empty panel on a clip where nothing is missing is capped at 2).

| judge variant | proposed | blind_a2i | audio_caption | paired proposed − blind |
|---|---|---|---|---|
| **grounded + rubric (declared)** | **2.68** | 2.70 | 2.51 | **−0.014 [−0.223, +0.194]** (null) |
| grounded only | 3.04 | 3.24 | 2.75 | −0.201 [−0.396, −0.022] |
| permissive | 2.77 | 3.35 | 2.96 | −0.576 [−0.856, −0.331] |

On the 109 benchmark clips only (same population as the per-sound headline): 2.79 vs 2.81, paired
−0.018 [−0.229, +0.183]; TEST-60 −0.067 [−0.383, +0.233]. By clip category (declared judge, 139): **seen-only clips +0.68 [+0.41, +1.00]** for the gate (2.86 vs
2.18) — the gate's purpose, confirmed by an independent LLM; **unseen clips −0.55 [−0.94, −0.18]**
(1.85 vs 2.39) — the price of its false silences; mixed −0.45 [−1.07, +0.16]; no-ambient identical
(both stay silent). Net: null.

**Gemma-4-31B (SOTA judge) on the same cached descriptions — DONE.** The stage-7 "SOTA vs older"
row: the same 417 (reference, description) pairs, scored by a 31B judge instead of the 7B one.

| judge | proposed | blind | paired difference |
|---|---|---|---|
| Mistral-7B, grounded + rubric (declared) | 2.68 | 2.70 | −0.014 [−0.223, +0.194] |
| **Gemma-4-31B, grounded + rubric** | 2.43 | 2.48 | **−0.050 [−0.309, +0.209]** |
| Mistral-7B, grounded only | 3.04 | 3.24 | −0.201 [−0.396, −0.022] |
| Gemma-4-31B, grounded only | 2.89 | 3.09 | −0.194 [−0.475, +0.094] |
| Mistral-7B, permissive | 2.77 | 3.35 | −0.576 [−0.856, −0.331] |
| Gemma-4-31B, permissive | 2.52 | 3.22 | −0.698 [−1.022, −0.381] |

**The two judges agree on every variant, including the size of the permissive judge's bias.** The
SOTA judge is stricter in absolute terms (2.43 vs 2.68) and gives the same verdict: null under the
declared rubric-enforced judge, strongly negative under a permissive one that rewards any picture.
That is a useful robustness result: the judge is not the reason the headline is null.

The permissive column is the pre-existing judge bias, declared and measured on 2026-09-19: the
judge rewards any picture, so a system that stays silent cannot win; that is why the rubric-enforced
variant is the declared one and the permissive one is reported beside it.

## 8. What Adam gets (summary)

1. **Versus showing nothing (SILENCE), the pipeline is a large, significant win**: per-sound
   F1 +0.29 [+0.20, +0.38]; the judge agrees.
2. **Versus the ungated baseline the gate is a precision instrument, not an F1 win**: significantly
   fewer false alarms (−33 %/clip), 3× fewer pictures of an on-screen source, 1.5× more clips
   correctly left silent, +0.06 precision — and a significant −0.08 recall. ΔF1 +0.028
   [−0.022, +0.074] is null, and the oracle diagnostic shows the equal-weight F1 cannot show this
   trade even with a perfect detector.
3. **Every model choice was tested against an older/alternative one and every pre-registered rule
   was applied**: detector BEATs vs PretrainedSED (SOTA) vs PANNs (v1) vs two-detector
   corroboration → BEATs stays (cross-fit); gate Qwen3.8-27B vs Qwen2.5-VL-7B vs OWLv2 → 27B stays,
   majority-of-3 stays; judge rubric-enforced vs permissive → both reported.
4. **Open, honest limitations**: single annotator; 49 of 109 clips seen during development (TEST-60
   reported separately and agrees); 79 needed sounds limits power; detector label errors dominate
   the remaining false alarms and are the bottleneck to fix next.


## 9. After the result: where the system actually loses (2026-09-22 evening, 3 Fables × 3 rounds)

Three reviewers, two rounds each so far, were unanimous: **fix the detector, not the gate**, and run
a **gold-input oracle test** first; keep the null primary; do not retune on the gold set.

### 9a. Oracle-gate test (`benchmark/gold/oracle_gate.py`) — the decisive experiment

The detector is replaced by the annotator's own sound list (true labels, true onsets); BLIND draws
all of them, GATE draws what its cached visibility votes did not silence. Same per-sound rules,
139 clips, no new GPU work.

| gate rule | ΔF1 gate − blind (109 benchmark clips) | TEST-60 | mixed | slice B | visibility classifier |
|---|---|---|---|---|---|
| majority of 3 (shipped) | **+0.067 [+0.012, +0.117]** | +0.060 | +0.082 | +0.084 | sens 0.43 · spec 0.88 · bal **0.65** |
| unanimous | +0.046 [+0.019, +0.074] | +0.044 | +0.063 | +0.031 | sens 0.20 · spec 0.97 · bal 0.58 |
| **majority + "obvious" vote** | **+0.097 [+0.040, +0.152]** | **+0.111** | +0.096 | +0.099 | sens 0.50 · spec 0.87 · bal **0.68** |

**With a perfect detector the gate is a significant win on every subset, including the external
slice B and the untouched TEST-60.** Precision +0.113 (majority) to +0.161 (with the obvious vote);
pictures of an on-screen source 93 → 52 → 42; clips correctly left silent 0.49 → 0.63 → 0.68.
So the central claim of the project is true and measurable — the shipped system's null F1 is the
**detector's** ceiling, not the gate's. (The "obvious" question is the fourth vote added today; it
is a gate change, so it stays a declared secondary until it is pre-registered and run end to end.)

### 9a-bis. Half-oracle: which half of the gap matters (the decisive number)

| row | gate F1 | blind F1 | ΔF1 |
|---|---|---|---|
| A — the shipped system (real detector) | 0.287 | 0.260 | +0.027 [−0.022, +0.073] |
| **B — gold sounds + *every* false alarm the real detector made** | 0.498 | 0.423 | **+0.075 [+0.029, +0.119]** |
| C — gold sounds only (full oracle) | 0.687 | 0.620 | +0.067 [+0.012, +0.117] |

Row B is the one that matters: keep all the detector's phantom labels but give the system the sounds
it missed, and **the gate becomes significant**. So the end-to-end null is caused by **missed
sounds, not by wrong labels** — which reverses the obvious guess and tells us exactly what to fix
(detector recall and onsets, not a phantom verifier). This is also why the two verifiers tried
earlier (CLAP top-k, Qwen2-Audio) would not have rescued the result even had they passed.

### 9b. Miss autopsy — what the 43 missed needed sounds actually are (benchmark clips, BLIND row)

| cause | n | fix |
|---|---|---|
| a picture of the right family exists but outside the ±1 s onset window (merge / 8-s cap / span start) | **16** | better onsets (frame-level SED), merge rule |
| BEATs scored the family 0.175–0.35 (hysteresis band) | 9 | lower bar **+ a verifier** |
| BEATs scored it 0.05–0.175 | 6 | lower bar + verifier |
| BEATs effectively blind (< 0.05: Hammer 0.019, Door 0.027, Whistle 0.044, Civil-defence siren 0.006) | 9 | open-vocabulary detector (text-queried) |
| dropped just below the display bar although the family fired ≥ 0.35 elsewhere | 2 | bar / span rule |
| family never reached stage 5 (label filter) | 1 | filter |

So ~37 % of the misses are a **timing** problem, ~35 % a **threshold** problem, ~21 % a real
**vocabulary** problem. That ordering decides what to try next.

### 9c. SOTA module survey (September 2026) and what is already ruled out here

*Detector.* Tried and rejected on this data: PANNs CNN14 (v1), PretrainedSED (5 AudioSet-Strong
backbones, rejected three times), FLAM (two attempts), BEATs∧PSED agreement (cuts phantoms, costs
recall), a CLAP top-k family verifier (failed its DCASE calibration: 0.42 recall at k = 20, bar was
0.95 → declared useless), a Qwen2-Audio-7B yes/no verifier (2026-09-15: removed 25/77 phantoms, bar
was ≥ 40 → failed). Not yet tried and worth it: **FlexSED** (open-vocabulary SED, Dasheng SSL
encoder + CLAP text encoder, trained on AudioSet-Strong, code and checkpoints released, PSDS1 0.448
vs 0.399 for the frame-level baseline) — it is text-queried, so it can be asked for "a hammer
hitting metal" or "a door closing", which is exactly the 9 sounds BEATs is blind to; and
**boundary-aware SED** (PSDS1 49.6, new SOTA) for the 16 timing misses. CED-base (50.0 mAP) is
clip-level, so it cannot fix onsets — skipped.

*Vision / scene understanding.* Tried: CLIP, SigLIP, OWLv2 (the object pass), SAM 3 (lost to OWLv2),
Qwen2.5-VL-7B and Qwen3.8-27B as the gate. Measured today: 27B ≈ 7B on gate balanced accuracy
(0.62 vs 0.61), so **the gate is limited by its question, not by model size** — a bigger VLM
(Qwen3-VL-235B) is not the fix. The fixes the reviewers rank first are all input-side: more frames
spanning the sound, "list everything visible that could make this sound" instead of yes/no, and the
annotator's own rubric in the prompt (the "obvious" vote, which already lifts balanced accuracy
0.65 → 0.68 and ΔF1 +0.067 → +0.097). Audio-visual segmentation models and the training-free MLLM
sound-source-localization recipes assume the source is *on* screen, which is the case this system
already handles.


## 10. Improving the detector — measured on Adam's gold set only (2026-09-22 evening)

**Split (declared first, `benchmark/gold/split.json`).** DEV 79 clips / TEST 60, stratified by the
four categories, by population (his clips vs the AudioSet slice) and by sourcing wave:

| | DEV | TEST |
|---|---|---|
| mixed | 19 clips · 32 needed | 13 · 17 |
| unseen | 19 · 37 | 14 · 24 |
| seen-only | 24 · 0 | 20 · 0 |
| no-ambient | 17 · 0 | 13 · 0 |

### 10a. Why BEATs misses what it misses — measured, not guessed

At **all nine** onsets where BEATs scored the needed sound below 0.05, its own top labels are
**Speech 0.58–0.81 or Music 0.48–0.58**, and the target is not in the top six: hammer 0.019 under
Music 0.48 + Speech 0.40; civil-defence siren 0.006 under Speech 0.81; crow 0.049 under Speech 0.72;
door 0.027 under Speech 0.80. The sounds are **masked by speech and music**, not missing from the
vocabulary. A cheap fix suggested by the reviewers — re-score each frame by 1 − max(Speech, Music) —
lifts those nine only from 0.006–0.049 to 0.018–0.100, far below any usable bar, so **compensation
does not rescue them**; only a detector that is asked about one label at a time (FlexSED) or one
that hears a separated signal can.

### 10b. BEATs bar sweep on DEV, per category (the numbers Adam asked for)

| bar | onset-recall | found-recall | false labels/clip | mixed FA | unseen FA | seen FA | no-amb FA |
|---|---|---|---|---|---|---|---|
| 0.35 (shipped) | 0.45 | 0.60 | 0.73 | 1.00 | 0.33 | 0.83 | 0.72 |
| 0.25 | 0.49 | 0.68 | 1.46 | 1.89 | 0.67 | 1.62 | 1.56 |
| **0.15** | **0.57** | 0.71 | 3.05 | 4.58 | 1.72 | 2.71 | 3.22 |
| 0.10 | 0.58 | 0.71 | 5.48 | 7.89 | 3.61 | 4.79 | 5.72 |
| 0.05 | 0.65 | 0.83 | 12.28 | 15.26 | 9.22 | 11.25 | 13.56 |

Per category, onset-recall at 0.35 is 0.44 on mixed clips and 0.45 on unseen; at 0.15 it is 0.53 and
0.61. The knee is at **0.15** (0.15 → 0.10 buys +0.01 recall for +2.4 false labels/clip). False
labels are worst exactly on the mixed clips (4.6/clip at 0.15) — and they are a long tail, not a few
bad labels: 241 false labels on DEV spread over **138 distinct names**, the most common being
Vehicle 18, Horse 6, Car 6. So pruning the label map cannot fix them.

**Onset is a separate, constant loss:** found-recall exceeds onset-recall by ~0.14 at every bar.
Re-anchoring each detection to the nearest spectral-flux novelty peak inside it halves the median
onset error (0.20 s → 0.10 s at bar 0.15) and moves recall by +0.03 at 0.15 and −0.03 at 0.35 —
noise at 65 DEV sounds, so it is reported, not adopted, and the paired onset-error test is the
honest one.

### 10c. FlexSED (open-vocabulary, text-queried) — running

`benchmark/gold/flexsed_run.py` queries FlexSED with the project's own 215 **depictable family
names** (fixed in advance, never the gold labels) and caches frame probabilities for all 139 clips.
Its per-label query cannot be out-voted by Speech/Music, which is exactly the failure in 10a.
Declared accept/reject rules (before the result): adopt as a **union with per-model calibration**
(each model keeps its own DEV bar, same-family detections overlapping in time merge, earliest onset
wins) **iff** it recovers ≥ 4 of the 9 masked sounds **and** the union adds ≥ 0.05 onset-recall over
BEATs alone at matched false labels **and** its false-label rate on the no-ambient clips is not more
than twice BEATs'.

### 10d. The gate's decision rule — swept for free on the cached votes

15 rules (stretch {all, majority, any} × vote {majority of 3, obvious, obvious OR majority, obvious
AND majority, any of 4}), each scored by the oracle-gate ΔF1 on DEV:

| rule | ΔF1 (DEV) | sensitivity | specificity |
|---|---|---|---|
| all / obvious-OR-majority | **+0.090 [+0.023, +0.160]** | 0.51 | 0.86 |
| any / obvious | +0.087 [+0.038, +0.145] | 0.38 | 0.92 |
| **all / majority of 3 (shipped)** | +0.076 [+0.015, +0.144] | 0.45 | 0.88 |
| all / obvious-AND-majority | +0.056 [+0.019, +0.097] | 0.24 | 0.95 |

**The shipped rule is already within noise of the best of fifteen.** The only rule that beats it adds
the "obvious" question (+0.014 ΔF1, +0.06 sensitivity for one extra lost picture on DEV). So the
video side is near its ceiling: the reviewers' estimate of a no-training ceiling around balanced 0.75
is consistent with the 0.65–0.70 measured here, and **the remaining headroom is in the detector, not
in the gate.**


### 10e. Video side: declared finished (2 Fables x 3 rounds, unanimous)

Both reviewers say stop, for the same reason: fifteen decision rules span only 0.034 of end-to-end
ΔF1 with overlapping CIs, and raising the gate's sensitivity from 0.45 to 0.72 makes the end-to-end
result **worse** (+0.076 → +0.067). That is a property of the task — wrongly silencing a needed
sound removes a whole picture, wrongly drawing a visible one costs only clutter — so the gate's
conservative operating point is a finding, not a defect. The one experiment either would still fund
(OWLv2 crops + Set-of-Mark instance questions) has a declared stop rule: adopt only if oracle ΔF1
clears +0.12, i.e. outside the sweep's upper CI.

**The free checks a reviewer would ask for, now made.**

*Do the three votes do any work?* Of 240 per-stretch decisions on DEV, 156 are unanimous and **84
split** (35 %), so the majority rule is not decoration. Pairwise agreement: name~ab 85 %,
name~describe 84 %, ab~describe 78 %.

*Why does the gate miss visible sources?* Of the 47 visible/obvious DEV sounds it fails to silence,
**40 are seen by no stretch at all** and only 7 are blocked by the "every stretch must agree" rule.
So the loss is perception, not aggregation — which is why changing the aggregation rule cannot fix
it, and why a bigger VLM did not either.

*Per category (shipped rule), the numbers Adam asked for:*

| | DEV: visible silenced | DEV: needed kept | TEST: visible silenced | TEST: needed kept |
|---|---|---|---|---|
| mixed | 17/42 (0.40) | 30/34 (0.88) | 12/20 (0.60) | 14/17 (0.82) |
| unseen | — | 34/39 (0.87) | 0/3 | 21/24 (0.88) |
| seen-only | 22/44 (0.50) | — | 10/32 (0.31) | — |

Most-missed visible families on DEV: Car alarm 5, Explosion 4, Doorbell 4, Fire 3 — sources that are
*in* the scene but not visibly acting (an alarm box, a fire off to the side), which is exactly the
class the annotator ticks "obvious" and the reason that question helps.

**Video-side claim for the thesis:** *given a correct sound list, an open-vocabulary VLM visibility
gate raises end-to-end F1 by about +0.08 (oracle +0.076 [+0.015, +0.144]), stable across fifteen
aggregation rules (+0.056 to +0.090); with the current detector that gain is masked by detector
misses, not by gate errors.*


## 11. The union detector end to end (v4b6) — declared, rendering, with one caveat recorded first

FlexSED passed all three rules declared before it ran (amendment 8): **7 of the 9 speech/music-masked
sounds recovered** (0.006–0.049 → 0.49–0.85), **the same onset-recall as BEATs@0.15 at half the false
labels**, and a no-ambient false-label rate 1.85× BEATs (limit 2×). Adopted as a union at
`FLEXSED_BAR = 0.8`; row **v4b6** (STAGES 590) is rendering on all 139 clips for both arms.

**Caveat recorded before the row finishes (dry, picture level, BLIND arm, DEV):**

| detector | P | R | F1 | hits | visible FA | cross | phantom |
|---|---|---|---|---|---|---|---|
| BEATs 0.35 | 0.18 | 0.38 | 0.24 | 25 | 19 | 69 | 28 |
| BEATs 0.35 ∪ FlexSED 0.8 | 0.14 | **0.48** | 0.22 | **31** | 27 | 110 | 49 |
| BEATs 0.35 ∪ FlexSED 0.7 | 0.10 | 0.48 | 0.17 | 31 | 34 | 147 | 87 |

The union buys six more hits on the DEV half and pays with more phantom pictures, so the **blind**
arm's F1 falls slightly. Whether the system as a whole improves therefore depends on the gate, which
is exactly what v4b6 measures: the gate removes visible-source pictures (19 → 27 available to remove)
but cannot touch cross/phantom ones. The half-oracle result (§9a-bis) says extra phantoms do not
destroy the gate's benefit while extra *hits* create it, so the prediction is that ΔF1 widens even
though blind F1 dips — but that is a prediction, written down here before the row is scored.
FlexSED 0.7 is already ruled out at picture level and is not rendered.


## 12. v4b6 — the union detector end to end: the prediction failed, and what that tells us

All 139 clips rendered with BEATs 0.35 ∪ FlexSED 0.8, both arms. On the 109 benchmark clips:

| | v4b4 (BEATs only) | **v4b6 (union)** |
|---|---|---|
| gated P / R / F1 | 0.25 / 0.34 / **0.287** | 0.21 / **0.37** / 0.27 |
| blind P / R / F1 | 0.19 / 0.42 / 0.26 | 0.17 / **0.47** / 0.25 |
| gated hits / blind hits | 27 / 33 | **29 / 37** |
| ΔF1 gated − blind | +0.028 [−0.022, +0.074] | **+0.026 [−0.020, +0.069]** |
| ΔP | +0.062 [+0.016, +0.109] | +0.049 [+0.009, +0.092] |
| ΔR | −0.076 [−0.141, −0.026] | −0.101 [−0.174, −0.041] |
| Δ false alarms per clip | −0.55 [−0.78, −0.37] | **−0.73 [−1.00, −0.51]** |
| Δ clean-clip accuracy | +0.172 [+0.082, +0.273] | **+0.190 [+0.094, +0.293]** |
| ΔF1 vs SILENCE | +0.290 | +0.271 |

**The prediction written in §11 was wrong and is recorded as such:** ΔF1 did not widen (+0.026 vs
+0.028). What did happen is exactly the first half of the prediction — the union finds more sounds
(blind recall 0.42 → 0.47, gated 0.34 → 0.37; +4 and +2 hits) — and the second half did not follow,
because the extra detections bring extra *wrong* pictures at the same rate, so precision falls as
much as recall rises.

**What the row is good for.** The gate's *absolute* work grows with a more generous detector: it now
removes **0.73** false alarms per clip instead of 0.55 and lifts correct silence by **+0.19** instead
of +0.17, both significant, and both larger than in v4b4. That is the honest reading — *the more the
detector hears, the more the gate is worth* — and it is the first direct evidence for it.

**Standing of the two rows.** v4b4 remains the best row on the pre-registered F1 (gated 0.287) and
stays the headline. v4b6 is the better row on recall (0.37 vs 0.34) and on the gate's measured value,
and is reported beside it. Neither is selected after the fact: both were declared before running
(amendments 6 and 8).

**What this leaves.** The binding constraint is no longer detector *recall* (0.45 → 0.57 at the
detector, 0.34 → 0.37 end to end) but the **label precision** of what the detector adds. The two
verifier families tried earlier (CLAP top-k, Qwen2-Audio yes/no) both failed their declared bars, so
the open question for future work is a better plausibility filter, not a better detector.


## 13. Three more detector ideas, all tested, all rejected by their own declared rules

After v4b6 the two reviewers (2 rounds each) ranked four remaining ideas. Three are now tested; each
had its decider written down before it ran.

| idea | decider declared first | result | verdict |
|---|---|---|---|
| **Hysteresis onset** — start the span where the score rises through 0.5 after being below 0.3, not where it passes the bar | ≥ 4 of the 8 "found but mistimed" sounds convert | onset-recall **0.57 → 0.57** (unchanged at three rise/fall settings) | rejected |
| **Rank-in-window admission** — a weak detection is kept only if its label is the strongest at its own peak frame, so the bar can drop to 0.5 | onset ≥ 0.62 at ≤ 1.8 false labels/clip | **0.58 at 2.90** | rejected |
| **Prompt ensemble** — every family asked three ways ("Hammer" / "hammer heard nearby" / "hammer, recorded in the real world"), max-pooled; templates uniform across all 215 families | hammer 0.194 → ≥ 0.5 with ≤ +0.2 false labels/clip | hammer **0.194 → 0.343**, crow 0.032 → 0.061, footsteps 0.695 → 0.711; union unchanged at **0.57**, false labels 1.61 → **1.78** | rejected |

The hysteresis result is informative: the gap between found-recall (0.69) and onset-recall (0.57) is
**not** late onsets — it is the family being detected in a *different burst* of the same clip. No
onset rule can close it.

**The remaining idea is speech/music removal**, which both reviewers ranked last and capped at about
+2 sounds. The v4b6 result now argues against spending on it at all: more recall at unchanged label
precision did not improve the system end to end (§12). Recorded as the declared next step if the
project revisits the detector, with its go/no-go already written: adopt only at onset-recall ≥ 0.62
with ≤ 2.0 false labels per clip and no rise on the quiet clips.

**Detector work is therefore closed with a measured conclusion**: recall was raised 0.45 → 0.57 at
half the earlier false-label cost (FlexSED union, adopted), three further ideas were tested and
failed their own bars, and the binding constraint is now label precision — for which the two
verifier families available off the shelf (CLAP top-k, Qwen2-Audio yes/no) had already failed.


## 14. The break-even rule, the cost curve, and one honest failure (2026-09-23)

### 14a. The rule four reviewers derived independently

Under the declared weights (a missed needed sound costs 4, a wrong picture costs β), showing a
picture changes the expected cost by **β − (4 + β)·p**, where *p* is the chance the picture is right.
So a picture is worth showing only when

> **p > β / (4 + β)** — at the declared β = 2 that is **p > 1/3**.

The system's pictures are right **23 %** of the time (26 of 113 on the benchmark clips). That single
inequality explains everything the earlier metrics could not: why SILENCE is cheapest, why more
recall did not help (extra pictures are right at the same 23 %), and why the gate still wins against
blind (it removes pictures, and every removed picture below break-even saves cost). It is also a
design rule for any audio-to-visual accessibility system, which is a contribution in itself.

### 14b. The cost curve (figure: `benchmark/gold/cost_curve_v4b4.png`, `_w.png`)

Cost per clip for every system as β runs from 0 (a wrong picture is free) to 4 (a wrong picture is as
bad as a missed danger sound). At the declared β = 2, on the 109 benchmark clips:

| system | what it is | cost/clip |
|---|---|---|
| **Ours** (cross-modal gate) | draw a picture only when the source is off screen | **3.36** |
| Blind | the same pipeline, gate off — every detected sound gets a picture | 4.24 |
| Caption | the detected sound names as text instead of pictures | 4.29 |
| Silence | show nothing — what a deaf viewer has today with subtitles | **2.79** |
| Gate with a perfect sound list | the upper bound the gate could reach | **1.28** |

The oracle line at 1.28 is the headroom: the architecture is worth less than half the cost of silence
*if the detector were right*. Everything between 1.28 and 3.36 is detector error.

### 14c. Selective showing — declared on DEV, read once on TEST, and it did not transfer

All four reviewers ranked the same system lever first: show only the sound families whose pictures
are right often enough to pay for themselves. Rule fixed on DEV before TEST was opened: keep a
family with ≥ 2 pictures and DEV precision ≥ 1/3 → **9 families** (Bell, Chainsaw, Alarm, Siren, Dog,
Explosion, Electric shaver, Sonar, Telephone).

| | DEV (64 clips, where the rule was fitted) | **TEST (45 clips, read once)** |
|---|---|---|
| whitelisted pictures right | 14/23 = **61 %** | 3/10 = **30 %** |
| cost: whitelist vs silence | **−0.59 [−1.16, −0.16]** (beats it) | **+0.04 [−0.27, +0.40]** (ties it) |
| cost: whitelist vs our full system | −1.06 [−1.53, −0.56] | −0.67 [−1.42, +0.13] |
| cost: whitelist vs blind | −1.78 [−2.50, −1.06] | **−1.78 [−3.02, −0.58]** |

**Reported as it came out:** the whitelist looked like it beat silence on the half it was fitted on
and did not transfer — nine families chosen from two to four pictures each is too little evidence to
generalise. On unseen clips it **ties** silence and still **beats blind significantly**. The lesson
is the tiny per-family counts, not the idea; with a gold set an order of magnitude larger the same
rule could be fitted properly, and that is the recommendation for future work.

## 15. Bottleneck hunt, 2026-09-23 (Adam: "3 rounds of 2 fables per pipeline bottleneck candidate")

### What the reviewer rounds actually were, stated plainly

Adam asked for three rounds of two reviewers per bottleneck candidate. What happened instead: about
ten reviewer exchanges, none of them a pair of parallel calls, because in nine of them the reviewer's
first move was to point out that the question I was about to ask was already answered by data sitting
in a cache. Each exchange therefore turned into a computation rather than an opinion, and the
questions that survived to be genuinely open were only two: the cost tie-break, and whether
per-family calibration on AudioSet-Strong reintroduces the whitelist failure. Candidate A took nearly
all the depth because the evidence put 61% of wrong pictures and 52% of misses there.

The reviewers caught four real errors before they reached a result: the taxonomy was being run on
v4b4, which predates the adopted detector; the "wrong-family" bucket was conflating salience with
mis-hearing; the FA budget arithmetic assumed misses were fixed; and amendment 10's selection rule
was incoherent with its own go/no-go. All four are recorded in docs/history/preregistrations/prereg_v4.md.

### The corrected bottleneck, on the adopted detector (v4b6, 109 clips)

    wrong-family   45  39% of wrong pictures      detector naming
    invented       25  22%                        detector naming
    gate-leak      21  18%                        gate
    late           20  18%                        timing
    level-1         3   3%

and the 21 DEV misses: detection 11 (six deaf, five heard only at another moment), timing 5 (three
of them EARLY by 0.84-1.90 s), gate 3, label filter 2. Stage 4 owns both sides.

An earlier draft of this analysis put timing at 48% of misses. That was wrong -- it came from the
scorer treating Owl and Bird, Train and Vehicle as one family, so a gated-off Bird plus a drawn Owl
looked like one mistimed picture. Corrected above.

### The one positive: the second detector as a veto

Where BEATs names a family FlexSED never hears in the clip, the sound is usually not there: Whale
0.13, Horse 0.13, Cat 0.00, Telephone 0.00. The union threw that disagreement away. Applying it as a
one-sided veto (only labels FlexSED did not itself raise, so the seven sounds it was adopted to
recover cannot be deleted) on DEV:

    v4b6 as it stands   F1 0.231  P 0.169  R 0.364  FA/clip 1.20  cost 4.12  hits 12  miss 21
    + veto tau 0.3      F1 0.264  P 0.207  R 0.364  FA/clip 0.94  cost 3.59  hits 12  miss 21

Zero recall cost. Verified that the canonical keys match across detectors before believing it: Chirp,
tweet -> Bird -> 0.76 survives, Owl -> Owl -> 0.13 is vetoed, and that Owl picture was exactly one of
the false alarms. Wired into stage 4 itself, not only the scorer.

### Three declared negatives

  * **Speech removal as a view** (amendment 12): a large recall gain (onset 0.45 -> 0.60 alone,
    0.69 in the full union) that fails its rule in every configuration -- it buys the recall with
    false labels on the quiet clips, which is the condition that exists to stop exactly that.
  * **Onset refinement from the residual** (amendment 12): onset-recall FELL 0.57 -> 0.54 and the
    median error nearly doubled, with false labels identical at 1.61 confirming the construction was
    sound. Both speech-removal routes closed.
  * **Family-level gating** (amendment 13): cost fell 3.59 -> 3.47 but three hits were lost where
    the rule allowed one.

### Candidate B closed on evidence

Only four DEV cases have both detectors firing on the same family near a needed onset -- too thin to
act on. And the direction is the opposite of my guess: BEATs is LATER in all four, FlexSED closer in
three, and the union's min(start) already picks FlexSED's better onset every time. My earlier "BEATs
fires 1.5 s early" reading came from a merged picture span, not a raw detection, and was wrong.

### Candidate C: what is left is the vision model

Of the 18 gate leaks that survive the veto, 13 carry BOTH the visible and the obvious tick -- the
clearest cases in the gold, not annotation ambiguity. Three are sibling escapes (silenced Vehicle,
drew Train), and the only rule that removes them costs three hits. The other 15 are the visibility
model looking at a source that is plainly on screen and saying it is not there. No rule change in
this project fixes that; it is the residual error of OWLv2 + the VLM and is reported as such.

### Running, not yet read

The DEV bar sweep (0.8 / 0.7 / 0.6 at tau 0.3, both arms), and FlexSED over the 280-clip AudioSet
calibration set for the per-family bar of amendment 11. Twenty-seven depictable families qualify at
K = 8 and were named in the prereg before the fit.

### 15b. Per category (Adam: "report also per category"), DEV, with and without the veto

    category      clips   cell               P      R      FA/clip   cost     silence
    unseen         14     v4b6            0.273  0.353     1.14     5.43       4.86
    unseen         14     + veto          0.286  0.353     1.07     5.29
    mixed           8     v4b6            0.273  0.375     2.00     9.00       8.00
    mixed           8     + veto          0.286  0.375     1.88     8.75
    seen           18     v4b6            0.000  0.000     0.94     1.89       0.00
    seen           18     + veto          0.000  0.000     0.61     1.22
    no_ambient      9     v4b6            0.000  0.000     1.11     2.22       0.00
    no_ambient      9     + veto          0.000  0.000     0.67     1.33

The veto helps most exactly where it should: on the clips that contain nothing to draw it cuts the
wasted pictures by 35% (seen) and 40% (no_ambient), while leaving recall untouched on the clips that
do contain needed sounds.

**This is also the honest explanation of why silence still wins at beta = 2.** Twenty-seven of the
forty-nine DEV clips -- the seen and no-ambient halves of the benchmark -- contain no sound that
should ever be drawn, and against those clips silence is unbeatable by definition: its cost is
exactly zero and ours is whatever we put on screen. On the clips the project is actually about, the
unseen ones, the gap is 5.29 against 4.86.

So the remaining headroom is a clip-level question, not a picture-level one: after the veto we still
spend about 1.2 cost per clip on 27 clips where the right answer is to show nothing at all. A
mechanism that recognised those clips would recover roughly 0.66 of cost overall and bring 3.63 down
to about 2.97, within touching distance of silence at 2.69 -- and it would do it without giving up a
single sound, because there are no sounds to give up there. That is the next thing worth building,
and it is a different question from any of the five tested today.

### 15c. Two corrections and the honest headline about the picture budget

**Correction to 15b.** I wrote there that recognising the clips with nothing to draw "is a different
question from any of the five tested today". That is wrong. Opening the 17 wasted pictures on those
27 DEV clips gives: gate-leak 9 (53%), invented 6 (35%), wrong-family 2 (12%) -- the same two buckets
as everywhere else, and the nine gate-leaks are the vision-blind cases amendment 14 already closed.
A clip-level abstain is therefore another precision filter, and four precision filters failed the
recall cap today. The 3.63 -> 2.97 figure in 15b should be read as an ORACLE BOUND on precision for
clips that contain no needed sound -- what would be recovered by a mechanism that already knew which
clips those are -- and not as a build target. Recorded as an overclaim caught and withdrawn.

**The honest headline about the picture budget.** On the unseen clips -- the category the project
exists for -- the gated pipeline beats silence for every price of a wrong picture up to beta = 1.60:

    unseen only, 14 DEV clips        ours    silence
      beta 0.00                       3.14     4.86    ours wins
      beta 1.00                       4.21     4.86    ours wins
      beta 1.50                       4.75     4.86    ours wins
      beta 2.00                       5.29     4.86    silence wins
      crossover                       beta = 1.60

A missed needed sound costs 4 throughout. So the result is: **on its target category the pipeline is
worth using whenever a viewer judges a wrong picture to be less than about 40% as costly as a missed
sound, and not worth using when they judge it to be 50% as costly, which is the figure this project
declared for itself in September.** That is a statement about the price of a wrong picture, it is
measured rather than argued, and it is a legitimate result to report rather than a failure to hide.
On the benchmark as a whole the crossover is lower still, because 27 of 49 DEV clips contain nothing
that should ever be drawn and silence is unbeatable on those by construction.
