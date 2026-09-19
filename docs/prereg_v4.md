# Pre-registration: v4 — the SOTA configuration, one swap at a time

*Committed 2026-09-18, before any v4 run. Decided with Adam after a Fable review and a live
check of the public leaderboards (arena.ai 7–13 Sept 2026, Artificial Analysis Sept 2026, HF
Open ASR). The supervisor's ask: the best models, a clear win over the baselines.*

## What v4 is

| stage | v3 (frozen, tag `v3-thesis-2026-09-17`) | v4 | why this one |
|---|---|---|---|
| 2 on-screen hints | OWLv2 (2023) | **SAM 3** (`facebook/sam3`, Nov 2025, text-prompted) | the recognised successor; context-only stage |
| 3 speech (context) | faster-whisper base | **Granite Speech 4.1-2B** (`ibm-granite/granite-speech-4.1-2b`, Apr 2026, WER 5.33, #1 Open ASR) | English assumed (6 languages; Whisper 99) |
| 4 sound detection | BEATs (2022), 527 classes | **FLAM** (Adobe, ICML 2025) with a descriptive query vocabulary and per-query calibration (§4) | the measured bottleneck; the only recognised frame-level open-vocabulary detector with released inference |
| 5 visibility + depiction + describer | Qwen2.5-VL-7B | **Qwen3.8-27B** (Aug 2026; Vision Arena open #4, 1244 ± 10) | already verified on an A100; 7B → 27B + thinking |
| 6 picture | FLUX.1-schnell (2024) | **Qwen-Image-2512** (Dec 2025, Apache-2.0; the only open model in the top 3 of both arenas) | quality; clean licence |
| 7 judge | Mistral-7B-Instruct (2024) | **Gemma-4-31B-it** (text mode; a different family from the Qwen describer and generator) | the describer and the generator are both Qwen in v4; the judge must not be |

Not in v4: MiniMax-H3 (video arm, "later", stays in the plan as the future section); DASM
(weights only, inference code unreleased — cited as a peer).

## Order and attribution

Four model swaps at once would make any gain unattributable. The swaps land in this order,
each with its own bar, on the frozen 100-clip test set and (when it exists) the gold set:

1. **+4 FLAM** → 20-clip check, then the full protocol (`v4a`)
2. **+5 Qwen3.8-27B** → 20-clip check, then full (`v4b`)
3. **+6 Qwen-Image-2512** → full (`v4c`)
4. **+3 Granite, +2 SAM 3** bundled last (`v4`) — context-only stages, expected Δ ≈ 0; run
   for freshness and reported as such

Each step is reported cumulatively (v3 → v4a → v4b → v4c → v4) and, at the end, as a
leave-one-out row per stage. The judge for every v4 row is Gemma-4-31B-it with Qwen3.8-27B
as describer; v3's panels are re-described and re-judged once under the same pair so the v3
row is comparable (the Mistral-judged v3 numbers stay in the thesis as the v3 chapter).

## The headline bar (what "significant improvement" means)

The claim is against the **baselines**, not against v3: under the same evaluation, the
gated system must beat the blind audio-to-image baseline. Pre-declared:

- **primary:** gated − blind judge score on the 100 test clips, paired bootstrap 95% CI
  excluding 0 and ≥ +0.25 on the 0–4 scale (v3: +0.01, a tie);
- **cost:** panel-on time and wasted seconds not worse than v3's ratio to blind (47% vs 60%);
- **gold set (when available):** per-sound "drawn / not drawn" accuracy of the gate vs the
  blind rule, and the human sentence as the reference.

If the primary bar fails after all swaps, the thesis says so with the numbers; the
cumulative table is the result either way.

## §4 FLAM, second attempt (the part that is a new experiment)

The first attempt (docs/prereg_flam.md) used the 527 bare AudioSet names as queries and
failed on precision (FP/min 693–2728 vs BEATs 5.2) while finding 70–85% of buried sounds.
Declared here, before the run:

- **Vocabulary:** a fixed list of ~85 descriptive queries ("a dog barking", "an emergency
  vehicle siren wailing"), each mapped to one AudioSet label so everything downstream
  (families, gate, dedup, depiction) is untouched. Written from the ontology and the
  pipeline's own families; five labels were added so that every *real* labelled detection
  on the dev split has a query (dev is the tuning split; disclosed). The list is
  `src/stage4_audio_event_detection/flam_queries.py` and does not change after the run.
- **Calibration (per query):** on DCASE 2025 task 3 **dev-train-tau** audio (600 five-second
  clips, one per mix and start time, never used for evaluation), the bar for query *q* is the
  lowest bar in {0.05, 0.10, …, 0.95} whose spans not overlapping a gold event of a matching
  class number ≤ 5.2 / |Q| per minute — the BEATs false-positive rate shared equally. A
  query that meets it at no bar gets 0.95. Scores are then rescaled piecewise-linearly so
  that the bar maps to `DISPLAY_THRESHOLD` (0.35) and 1 stays 1; the shipping span rule
  (reach the bar, extend through half of it, ≥ 0.5 s) runs unchanged on the rescaled scores.
- **Evaluation:** the same five bars as the first attempt, on **dev-test-tau** (the 255
  events) and the labelled dev detections:

| measure | BEATs | FLAM-v2 must reach |
|---|---|---|
| masked-event recall (under speech/music) | 9.5% | **≥ 24.5%** |
| clear-event recall | 32.6% | **≥ 32.6%** |
| false positives / min | 5.2 | **≤ 5.2** (a real test now, not by construction) |
| dev real detections still fired (of 23) | 23 | **≥ 21** |
| dev phantoms no longer fired (of 77) | 0 | **≥ 40** (a phantom whose label has no query counts as gone; disclosed) |

PASS iff all five. On pass, `AED_MODEL = "flam"` is v4a. On fail, attempt nine in the
table with the numbers, and v4 continues from stage 5 with BEATs.

## §5 Qwen3.8-27B — the visibility check already ran

docs/prereg_qwen38_visibility.md, thinking-off arm (job 30546271): on-screen recall
**56.3%** (bar ≥ 50% ✔), off-screen recall **82.6%** (bar ≥ 95% ✘) — the same trade the
32B made. The thinking arm (30546328) is still running. Whatever it says, the swap goes
ahead by decision (best models), the numbers are disclosed next to the 7B's, and no prompt
or vote rule is changed for it. The gold set's per-sound labels are the arbiter.

## Not done

No tuning on the test clips; no prompt changes after seeing numbers; the gate bar stays;
v3 stays in every table.

## §4 outcome (added 2026-09-18 evening, after the run) — FAILED as pre-registered

`benchmark/flam_v2_setting.json`, `benchmark/flam_calibration.json` (600 clips, 50 min;
median bar 0.53; 24 queries met the budget at no bar and were left at 0.95 as declared).

| measure | bar | FLAM-v2 |
|---|---|---|
| masked-event recall | ≥ 24.5% | **69.8%** ✔ |
| clear-event recall | ≥ 32.6% | **88.4%** ✔ |
| false positives / min | ≤ 5.2 | **61.6** ✘ |
| dev real detections kept | ≥ 21/23 | **15/23** ✘ |
| dev phantoms gone | ≥ 40/77 | **59/77** ✔ |

Diagnosis (exploratory, after the result): the false positives are impact and human
generics — Knock 5.9/min, Footsteps 4.6, Clapping 4.0, Doorbell 3.4, Cough 3.1, Bicycle
bell 3.1, Door 3.0, Bell 2.8, Rain 2.5 — and the calibration did not transfer from
dev-train to the busier, event-selected dev-test clips (with the 24 ceiling queries off the
rate would still be 34/min). Of the 8 lost dev reals, 4 had FLAM score ≈ 0 in the span
(Aircraft, Gunshot, Explosion, Zipper: FLAM has blind spots of its own), 3 sat under a
ceiling bar, 1 was marginal. BEATs' 5.2/min was never measured at matched recall, so the
bar is asymmetric; disclosed. Recorded as attempt nine. Two attempts say the same thing:
FLAM hears under masking what BEATs misses, with an unusable false-positive rate on
generic impacts — complementary, not a substitute.

## Added 2026-09-18 night (before any number): gold slice B and the detector table on it

Slice B = AudioSet-Strong *eval* clips with a consequential sound at least half covered by
speech or music (`benchmark/gold/audioset_slice.py`: fixed vocabulary of consequential
families, ≤ 12 clips per family, seed 7; clips gone from YouTube are listed). Human-timed
labels are the pre-fill; annotators add only visibility / "draw?" and the sentence. It is
reported separately from the 100-clip slice, never pooled, and is used for **detector**
measures only. `benchmark/audioset_detector_eval.py` gives one table for BEATs (v3),
PretrainedSED (v4 candidate) and FLAM-v2 (the failed attempt) on the same clips: recall of
masked consequential events, of all consequential events, of all events; false spans per
minute (every sound in these clips is labelled, so this is a real false-alarm rate); onset
error. No pass bar: it is the held-out check that the DCASE-chosen detector setting
transfers to real-world video. The PretrainedSED decision stays with docs/prereg_psed.md.

## Added 2026-09-19 00:40, after the PretrainedSED result and before any protocol score: the detector arm

PretrainedSED failed its eighth bar (docs/prereg_psed.md) and per that pre-registration
**BEATs stays the v4 detector**. Observed after the fact: that bar's population is BEATs'
own detections judged real by a human (23 items), so it measures agreement with BEATs
rather than recall of ground truth; on gold slice B (declared above, no pass bar, 111
held-out real-world clips, 236 masked consequential events) PSED was ahead of BEATs on
every measure (masked-consequential recall 62.7% vs 50.0%, all events 53.0% vs 37.9%, false
spans 2.6 vs 6.8 per minute, onset MAE 1.14 vs 1.48 s). Because this was seen after the
result it does not alter the v4 decision. Instead a second comparison is declared here,
with its rule fixed before any protocol score exists:

- rows **v4a** = PSED + Qwen2.5-VL-7B and **v4ab** = PSED + Qwen3.8-27B are run through the
  protocol next to v3_q38 (BEATs + 7B) and v4b (BEATs + Qwen3.8), same describer and judge;
- **PSED is adopted for v4 iff** the gated score of v4ab exceeds that of v4b (mean paired
  difference > 0) **and** the 95% paired bootstrap CI of that difference does not lie
  entirely below 0; an exact tie or a negative mean keeps BEATs;
- the dev-real bar is retired for future detector attempts in favour of slice B's
  exhaustive labels (a design correction, not a retroactive waiver).

## Judge v4 (rubric-enforced) — declared 2026-09-19 01:50, before any 100-clip v4 score

Observed on v3 (100 clips) and the v4b 20-clip check: on clips tagged seen / no-ambient
(grounded reference "nothing beyond the picture") the LLM judge gives a *redundant* picture
4 on most clips (blind 3.80 mean on the 10 no-due clips of the check), the same as correct
silence. The protocol already codes one side of the declared rubric (docs/plan_robustness.md,
2026-09-14: wrong silence costs 4, redundant picture ~1) — an empty panel on a no-due clip
is 4 in code — but not the other. This completes it, in code, not in the prompt:

> For clips whose grounded reference is "nothing beyond the picture": empty panel = 4
> (unchanged); non-empty panel = min(LLM score, 2), applied in code after the LLM call.
> Caption row: empty caption = 4; a caption naming any sound = min(score, 2). All other
> clips: unchanged. Applied identically to gated, blind, caption and the oracle, and
> retroactively to v3 (100 clips) and v4b (20 clips); the uncapped judge is reported as
> "permissive judge" beside it on every row. Primary score for v4 = rubric-enforced judge.
> The gate's dev sweep uses the same asymmetry (redundant picture −2, missed picture −4).
> No further metric change after this declaration.

Risk, stated: the cap is exactly the lever that separates gated from blind and it is
declared after numbers that favoured blind. Defence: the 09-14 written asymmetry, the
pre-existing one-sided code rule, both judges on every row, and the judge-free check —
the gold set's per-sound "draw?" accuracy — which must agree.

## Correction, 2026-09-19 morning: the PSED rows ran at the wrong bar

The v4a/v4ab rows of the night used PSED's scores against the pipeline default bar 0.35,
not PSED's own bar 0.20 (chosen on DCASE by the matched-false-alarm rule before slice B or
the test clips were scored). On slice B PSED at 0.35 recalls 44% of masked consequential
sounds (BEATs 50%) — the misses on "picture needed" clips came from this. The rows are kept
under the tag `v4ab_bar035` as a record and **discarded**; PSED's scores are now rescaled so
that 0.20 lands on 0.35 (`psed_infer.rescale`), and v4ab is re-run. Fable's review: keep
the DCASE-chosen 0.20 (rule declared first); the slice-B sweep (0.05–0.50) is reported as a
held-out check, not used for selection.

## Hardware note (2026-09-19)

Runs are placed on whatever GPUs are free (Adam's rule): a 27B model may be sharded across
3× L4 (24 GB) with `device_map="auto"` instead of one A100-80. Fable's review: same
experiment in design (same weights, bf16, code, prompts, greedy decoding); not bit-identical
across GPU types (different bf16 kernels → rare token flips on near-ties), no systematic
bias; reported CIs cover it. Disclosure for the thesis: "Inference ran on mixed NVIDIA
hardware (A100-80 / 3× L4 sharded); bf16 numerics differ slightly across GPU types, so
outputs are reproducible in distribution, not bit-for-bit." Each row's log records
torch/CUDA/GPU. The PSED row (v4ab) runs on 3× L4.

## Calibration on AudioSet-Strong (declared 2026-09-19 before any number on the calibration set)

Adam: "why DCASE and not AudioSet-Strong?" DCASE is synthetic indoor audio; it was kept only
because the threshold rule was written before slice B existed. From now on every detector's
bar is chosen on a **calibration set** = a random sample of ~320 AudioSet-Strong *eval* clips
(seed 11), disjoint from slice B's 150 ids, not filtered (ordinary YouTube audio), by the same
rule: the loosest bar in {0.05 ... 0.95} whose false spans per minute on the calibration clips
do not exceed BEATs' at its shipping bar 0.35 on the same clips. Slice B (111 clips, Adam's
annotations) stays the untouched check. DCASE remains only for the visibility check until
Adam's visibility annotations replace it. The PSED row (v4ab) is re-run at the new bar; the
DCASE-chosen 0.20 row is kept as a record. The five-model average is judged by the same rule
and the same slice-B pass rule (docs/prereg_psed_ensemble.md), with its bar from the
calibration set.

### Calibration result (2026-09-19, `benchmark/detector_calib.json`)

280 calibration clips (47 min). BEATs at 0.35 makes **6.39** false spans/min there (DCASE: 5.2),
so the matched bars are **PSED 0.15** (was 0.20 on DCASE) and **five-model average 0.10**.
Slice B, untouched, at those bars:

| detector | bar | masked-consequential recall | all events | false spans/min | onset MAE |
|---|---|---|---|---|---|
| BEATs | 0.35 | 50.0% | 37.9% | 6.76 | 1.48 s |
| PSED | 0.15 | 64.8% | 57.0% | 4.05 | 1.31 s |
| five-model average | 0.10 | 68.2% | 62.4% | 5.78 | 1.57 s |

Reading: at BEATs' own false-alarm rate the average finds 3.4 points more hidden sounds than
PSED, at 1.7 more false spans per minute — a trade along the curve, not a better detector
(slice B cannot resolve a 3-point difference). By its declared rule (≥ 65.7% AND ≤ 2.59/min)
the average **fails** on the false-alarm side; the ensemble attempt is closed. PSED at the
DCASE bar (0.20) stays the running v4ab row; the protocol is not re-run at 0.15 now — one
final detector configuration (docs/prereg_detector_v5.md) will be run once through the
protocol when the last detector attempt closes.

## Stage-2 swap check (declared 2026-09-20, before running): SAM 3 vs OWLv2 on the DCASE visibility set

SAM 3 was chosen as the stage-2 detector (v4 "2") but never measured. Dev-only check, same
258 DCASE 2025 Task 3 events, frames and seed as the VLM visibility runs
(`benchmark/eval_dcase_visibility.py`, seed 7); one concept phrase per DCASE class
(`benchmark/eval_dcase_visibility_det.py`, PHRASE), the same wording style as OWLv2's
DETECT_QUERY; best presence score over six frames; bars = each model's pipeline default
(SAM 3 0.5, OWLv2 0.20), nothing tuned. Reported: agreement, on-screen recall, off-screen
recall (the one that costs the viewer wrong pictures), AUROC of the raw score, per class.
Decision rule: SAM 3 replaces OWLv2 in stage 2 iff its agreement is higher AND its
off-screen recall is not lower by more than 2 points; otherwise OWLv2 stays and the "2"
swap is dropped. The DCASE label is geometric (in field of view), so both models are judged
against the same imperfect gold; only the comparison is read, not the absolute numbers.
