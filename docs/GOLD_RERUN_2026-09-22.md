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

**Gemma-4-31B (SOTA judge) on the same cached descriptions** — jobs 30968686/687, pending; new tags
(`v4b4_gemma*`) so the Mistral results are untouched. This is the stage-7 "SOTA vs older" row.

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
