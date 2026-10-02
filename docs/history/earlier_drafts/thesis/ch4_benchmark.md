# Chapter 4 — Benchmark and evaluation protocol (draft, 27 Sept 2026)

*This chapter describes pilot benchmark v1 for **off-screen sound visualisation**: the clips, the labels, the
splits, the per-sound metric, the modelled viewer cost and the statistics. Results are in Chapter 5. Clip and sound
counts were computed for this chapter from `benchmark/gold/annotations/gold_AG.json` (export of 22 Sept 2026,
09:23 UTC) with the scorer's own loader (`benchmark/gold/score_per_sound.py`); they agree with
`docs/GOLD_RERUN_2026-09-22.md` §1. Other sources are named beside each number. **[pending]** marks items not yet
available.*

## 4.1 Clips: sources, populations and categories

The task is to show a picture only for sounds that a viewer without hearing would miss. So the benchmark must label
**every sound**, not only every clip. Its unit is **one sound in one clip**. It has **139 clips** in two
populations.

- **Benchmark clips (109).** Short clips, most chosen by scene type rather than by a single sound: city walks,
  markets, streets, nature and weather, film scenes and trailers, where many sources are off screen. Of the 109,
  81 have an entry in the committed source lists (`benchmark/sources*.json`, `benchmark/manifest.json`): 50 from
  YouTube (12 of them film scenes), 13 AudioSet-Strong segments, 7 from UnAV-100, 7 from Wikimedia Commons and 4
  from the Internet Archive. The other 28 (mostly `ambient_*` and `movie_*` clips) have no entry. 70 of the 81 are
  marked "research use; not redistributed".
- **Slice B (30).** AudioSet-Strong *evaluation* clips (Hershey et al., 2021) with a consequential sound at least
  half covered by speech or music (at most 12 clips per family, seed 7; `benchmark/gold/audioset_slice.py`). Of the
  111 selected clips (`audioset_slice.json`), 30 were annotated. Slice B is always reported separately.

Each clip's **category** is derived from the per-sound ticks (the tool's `category()` rule, repeated in the
scorer). **Off screen:** every sound rated 2–3 is needed. **Mixed:** at least one needed sound rated 2–3 and at
least one visible or obvious sound rated 2–3. **On screen:** sounds are present, but none is needed. **Nothing to
draw:** no non-speech, non-music sound. Each clip also carried a *sourcing tag*, the reason it was collected. The
tag is intent; the scorer uses the derived category.

| | off screen | mixed | on screen | nothing to draw | AudioSet slice |
|---|---|---|---|---|---|
| sourcing tag (139) | 39 | 21 | 16 | 33 | 30 |
| derived category (139) | 33 | 32 | 44 | 30 | — |

## 4.2 Sounds

The export holds 298 sound rows; 295 have a label, a start and an end. The scorer drops 12 of these (speech,
music, or a label that is not a salient non-speech sound), so **283 sounds are scored**, with 99 distinct labels
(most frequent: Laughter 15, Bird 10, Water 10, Explosion 10). Of the 283:

- **120 are needed** (neither visible nor obvious): **110 rated 2 or 3** (28 rated 3), in 64 clips, and 10 rated 1;
- **163 are visible or obvious** (156 visible, 140 obvious, 133 both).

Importance over the 283: 34 rated 1, 207 rated 2, 42 rated 3. Twenty-five free-text names with no exact ontology
match are mapped by an alias table written before any score was read (for example "machinegun" → Machine gun;
`ALIASES` in `score_per_sound.py`). No scored label is unresolved.

## 4.3 Annotation protocol

**Tool and pre-fill.** One annotator (the author) labelled every clip in a browser tool
(`benchmark/gold/tool_template.html`). For benchmark clips the tool was pre-filled from the system under test
(candidate sounds, times, a visibility guess). For slice B, labels and times came from the human AudioSet-Strong
spans. The guideline told the annotator to delete what he did not hear, add what was missing and flip any wrong
tick. The possible anchoring is disclosed (`benchmark/gold/README.md`).

**Per sound:** a **label** (free text, mapped to the AudioSet ontology); **onset / offset** (about half a second is
enough); **visible**, meaning the source is *seen making the sound now* (a barking dog on screen is visible; an alarm
box on a wall is not visibly ringing; a source seen for only part of the sound is ticked only if seen for most of
it); **obvious**, meaning that with the sound off a viewer would still know the sound is happening now; and
**importance** 1–3.

**Needed.** A sound is *needed* when it is neither visible nor obvious. This extends a captioning practice
(Chapter 2 §2.1) to whole sounds: a sound needs a picture only if the video does not already show that it is
happening.

**Importance (rule of 22 Sept 2026,** declared before the gold re-run; `docs/metric_per_sound.md`). Importance is a
property of the sound, not of the screen: the annotator rates it **as if the screen were black**. **1** = the steady
noise of the place, with no start and no end (distant traffic, wind, rain). **2** = something happens that you can
say in one sentence (footsteps, a door, one bird call). **3** = danger or a key moment of the story (siren, alarm,
gunshot, glass, a baby crying, phone). When unsure, pick the lower level.

**Exhaustive labelling.** Every sound the annotator hears is added, including quiet ones, because a picture with no
gold sound behind it counts as a false alarm, with no excuse from the detector's scores.

## 4.4 Splits

| split | clips | off screen | mixed | on screen | nothing to draw | needed sounds (2–3) |
|---|---|---|---|---|---|---|
| DEV | 49 | 14 | 9 | 18 | 8 | 36 |
| TEST | 60 | 13 | 15 | 12 | 20 | 43 |
| slice B | 30 | 6 | 8 | 14 | 2 | 31 |

**DEV** is the gold ∩ the older judge set (`benchmark/gold/judge100.txt`): clips that earlier runs had already
touched. **TEST** is the rest of the benchmark clips (`test_bench`), tagged after the pipeline was frozen. These are
the scorer's subsets of amendment 5 (`subsets_of()`). All end-to-end choices were made on DEV. TEST is *held out,
with disclosed exposure*: ten exposures, each dated in Chapter 5 §5.15. TEST holds more clips with nothing to draw
than DEV (20 against 8).

**Split overlap.** A second definition exists: `benchmark/gold/split.json` (DEV 79 / TEST 60, seed 7, stratified by
category × population × sourcing wave), used by the stage-level detector bench. **35 of the scorer's 60 TEST clips
are in split.json's DEV-79, and 20 of the scorer's DEV-49 are split.json TEST.** FlexSED's bar 0.8 (amendment 8)
and the detector-level rejections (amendment 12) were selected on DEV-79; the vetoes, the onset rule and every
end-to-end DEV cell used the clean DEV-49 (`prereg_v4.md`, correction after amendment 21). On the clean DEV-49,
FlexSED's bar passes two of its three adoption rules and misses the third by 0.011 (`flexsed_recheck_dev49.json`).

## 4.5 Reliability

The only reliability number so far is **intra-rater**, and it is on an older, clip-level label of an earlier
274-clip set: the author re-labelled 60 clips blind to his first labels and agreed with himself on 78 %, κ 0.60
(`docs/EXECUTIVE_SUMMARY.md` §11; `docs/LEDGER_2026-09-26.md`, row F4; `docs/PLAN.md`). It does not measure this per-sound gold. A **second annotator** will label 30
DEV and TEST clips (no slice B; amendment 21), stratified by category, with the same fields. Declared reading
(`docs/FOR_SUPERVISOR.md` §3a): Cohen's κ on *needed* is the key number; κ ≥ 0.6 counts as substantial, κ < 0.4
would make the definition unreliable. Also reported: κ on visible and obvious, weighted κ on importance, median
onset difference. **Result: κ [pending].**

## 4.6 The per-sound metric

The rule was set on 19 Sept 2026 after a ten-reviewer debate (`docs/metric_per_sound.md`) and is implemented in
`score_per_sound.py`. One rule scores every system.

**Matching, time first, label second.**

1. **Onset window.** A picture matches a gold sound only if it *starts* within **[onset − 0.5 s, onset + 1.0 s]**:
   an onset-only, asymmetric collar in the style of event-based SED scoring (Mesaros et al., 2016). It is set for
   the viewer: a glass shattering shown 3 s late has lost its effect. The picture's end is ignored. Sensitivity
   rows use +0.5, +2 and +5 s.
2. **Family.** The picture must name the same AudioSet family (the class, a parent or a child). Top-level
   categories ("Sounds of things", "Animal") never match; depth-1 families such as Vehicle or Water do
   (`MIN_DEPTH = 1`).
3. **One-to-one.** Each gold sound takes at most one picture, the nearest in time. Extra pictures on a matched
   sound are *duplicates*: neither credited nor false alarms.
4. **"Dog at 45 s".** A picture is never judged alone: a dog picture at 45 s is first matched against dog-family
   gold sounds around 44–50 s.

**Classes.** Each needed sound is a **hit** or a **miss**. Each picture is a hit, a **visible-picture** (a real
sound that was visible or obvious), a **cross-trigger** (a real sound of another family), a **phantom** (no sound
there) or a duplicate.

**Numbers.** Counts are pooled over a split. Precision = hits / (hits + false alarms); recall = hits / needed
sounds; F1.

- **A visible picture is a false alarm.** The reviewers split 5 to 5; the author decided. The headline is
  **F1-strict**; F1 with phantoms and cross-triggers only is printed beside it.
- **Importance.** The headline counts needed sounds rated 2–3, unweighted. A needed sound rated 1 has no clear onset,
  so it is *don't care*. A picture of any visible or obvious sound, level 1 included, is a false alarm of weight 1.
- **Clean-clip accuracy:** the share of clips with no needed sound on which nothing was shown.
- **Side columns:** F0.5, F2, weighted F1, median lateness of hits, coverage of long sounds.

**Worked example.** Gold: a siren 12–18 s (needed, importance 3), a dog 30–32 s (obvious). System: SIREN 13–16,
DOG 30–32, HORN 40–42. SIREN is a hit (1 s late), DOG a visible-picture, HORN a phantom. Recall 1/1,
precision-strict 1/3, F1-strict 0.50. The rules block cheap strategies: showing everything fills the panel with
false alarms, showing nothing gives recall 0, and a vague label never matches.

## 4.7 The modelled viewer cost

F1 prices a missed sound and a wrong picture the same; a viewer may not. The **modelled viewer cost** per clip is

    cost = 4 × (needed sounds with no picture) + β × (wrong pictures)

(`docs/beta_specification.md` §1). A miss is fixed at 4, so only β, the price of a wrong picture, varies. The
weights (4 and 2) predate the gold, but the cost was promoted to a reported outcome after F1 came out null
(amendment 9): it is a **post-hoc** outcome, with β = 2 *assumed*. The cost implies a decision rule: show a picture
when its chance of being right exceeds **p\* = β / (4 + β)** (11 % at β = 0.5, 33 % at β = 2, 50 % at β = 4).

**Why β is not fixed.** No one has measured β for DHH viewers. The literature gives its direction, not its size:
users want speed for urgent sounds and accuracy for the rest (Jain et al., 2020), graphics can distract (Alonzo et
al., 2022), and deaf and hard-of-hearing viewers react differently (`beta_specification.md` §7–§9). So the cost is
reported as a **curve over β**, with its crossings against the blind arm and against showing nothing, and the claim
is an interval of β (Chapter 5 §5.5). A DHH study that would measure β is designed and not run
(`beta_specification.md` §5). The crossing values printed in `beta_specification.md` §2 and §6–§10 come from an
older pipeline and are superseded.

## 4.8 Statistics

All intervals are **paired clip bootstraps**: clips, not sounds, are resampled (sounds in a clip are not
independent), 2000 draws, seed 0, every system scored on the same resampled clips. p-values are two-sided, read
from the draws (resolution 1/2000). **Holm** correction applies inside each declared family: family 1 (ours −
blind) has seven rows, family 2 (against showing nothing) has three. The primary metric, ΔF1 ours − blind, is one
pre-registered test. Holm corrects for the number of rows; it does not repair a post-hoc entry.

## 4.9 AudioSet-Strong sets for the detector

Detector choices need more labelled sound than the gold holds. Two sets come from the AudioSet-Strong evaluation
split, where every sound is human-timed, so false spans per minute is a real false-alarm rate.

- **Calibration set, 280 clips.** A random sample of 320 (seed 11), disjoint from slice B; 40 were gone, 280 remain,
  about 47 minutes (`benchmark/gold/audioset_calib.json`; `prereg_v4.md`, calibration). They hold 1,264 non-speech,
  non-music events, 224 consequential, 31 of those masked by speech or music (`benchmark/detector_round_stage0.json`).
  They were used to compare detectors at a fixed false-span rate and as the *fit set* of the detector rounds
  (amendments 22–25); they are not an out-of-sample test.
- **Held-out set, 415 clips, unread.** Amendment 24 requested 500 (seed 23): 300 *complex* (speech or music covers
  at least half the clip and a non-speech event lies at least half under it) and 200 random. The panel had signed
  250 + 250; the author's direction towards complex scenes changed it. The set is disjoint from the 280, slice B and
  every gold clip. 415 were fetched (252 complex, 163 random), 85 were gone (`benchmark/gold/audioset_heldout.json`).
  No detector cell passed its fit-set rule, so the held-out set was **never scored**; its caches are kept unread.

## 4.10 What a release would contain, and what v2 needs

This is a **pilot benchmark**: 139 clips, one annotator. The protocol is the contribution; the numbers are
provisional. The evaluation panel's minimal release (`docs/panel3_topic3_rounds.md`, T3-c, T3-b):

1. clip IDs, timestamps and licence notes, no video (most clips are "research use; not redistributed", and 28
   sources must first be recorded);
2. the per-sound table (ontology label, onset, offset, visible, obvious, importance);
3. the guideline with frozen tick definitions, including "rate as if the screen were black";
4. the second annotator's agreement table;
5. the scorer (window, family match, visible picture = false alarm, bootstrap, Holm), `cost_curve.py` and a README
   that reproduces the thesis tables;
6. reference outputs: blind, silence, text tags, oracle gate;
7. the TEST exposure log;
8. a metric card (collar, β range) and a data card (category counts, licences), with a version tag and the task
   name.

**Before any release:** 28 benchmark clips have no source record yet (§4.1), and each must be traced to its source
and licence before the benchmark is released.

**v2 needs** at least 300 clips, at least two annotators, a β measured with DHH viewers, and a sealed held-out split
with a script that scores a submission folder (T3-a, T3-b).

## References

All works cited in this chapter are listed in the shared reference list, `docs/thesis/references.md`.
