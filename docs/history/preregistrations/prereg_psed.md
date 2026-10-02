# Pre-registration: PretrainedSED BEATs-strong as the detector (v4 stage 4, attempt ten)

*Committed 2026-09-18 night, before the run. Decided with Adam after Fable's review of the
two FLAM attempts (both failed on precision; docs/history/preregistrations/prereg_v4.md §4).*

## What is swapped

Stage 4's BEATs (clip tagger, 2-s sliding window, occlusion onset refinement) →
**PretrainedSED `BEATs_strong_1`** (Schmid et al., CP-JKU, ICASSP 2025): the same BEATs
backbone fine-tuned frame by frame on AudioSet-Strong with distillation — PSDS1 46.5 vs
36.5 for plain BEATs, the recognised fixed-vocabulary state of the art with public weights
(the 2026 boundary-aware successor, 49.6, has none). Probabilities every ~40 ms for 447
AudioSet-Strong classes; 28 renamed classes are mapped to ontology names
(`psed_infer.STRONG_TO_ONTOLOGY`, fixed before the run); the hysteresis span rule and the
0.5-s minimum are unchanged; no occlusion step.

## The bar

Chosen on DCASE gold at BEATs' false-positive rate: the loosest bar in {0.10 … 0.95} with
FP/min ≤ 5.2 (the rule of the first FLAM attempt). Nothing is tuned on dev or test clips.

## Pass bars (revised for a fixed-vocabulary model, per Fable: its expected gain is timing
and phantoms, not buried sounds)

| measure | BEATs | PSED must reach |
|---|---|---|
| DCASE masked-event recall (under speech/music) | 9.5% | **≥ 9.5%** (no regression) |
| DCASE clear-event recall | 32.6% | **≥ 32.6%** |
| DCASE false positives / min | 5.2 | ≤ 5.2 (by construction of the bar) |
| DCASE onset: matched events (within 1.5 s, hysteresis rule) | 66 | **≥ 66** |
| DCASE onset: mean absolute error | 0.35 s | **≤ 0.35 s** |
| DCASE onset: within 0.5 s | 80% | **≥ 80%** |
| dev: labelled real detections still fired (of 23) | 23 | **≥ 21** |
| dev: labelled phantoms no longer fired (of 77) | 0 | **≥ 40** |

**PASS iff all.** On pass, `AED_MODEL = "psed"` is v4a and the protocol is re-run (20 clips,
then 100). On fail, attempt ten in the table with the numbers, BEATs stays.

## Not done

No per-class bars, no mapping changes after the numbers, no prompt or gate changes.

## Outcome (added 2026-09-18 night, after the run) — FAILED as pre-registered, on one bar

`benchmark/psed_setting.json`; bar chosen on DCASE at BEATs' false-positive rate: **0.20**.

| measure | bar | BEATs | PSED |
|---|---|---|---|
| masked-event recall | ≥ 9.5% | 9.5% | **34.1%** ✔ |
| clear-event recall | ≥ 32.6% | 32.6% | **44.2%** ✔ |
| false positives / min | ≤ 5.2 | 5.2 | **3.9** ✔ |
| onset: matched events | ≥ 66 | 66 | **107** ✔ |
| onset: mean abs. error | ≤ 0.35 s | 0.35 | **0.19 s** ✔ |
| onset: within 0.5 s | ≥ 80% | 80% | **91%** ✔ |
| dev phantoms gone | ≥ 40/77 | 0 | **70/77** ✔ |
| dev real detections kept | ≥ 21/23 | 23 | **7/23** ✘ |

Seven of eight bars cleared, by wide margins; the one it fails is the one that matters most
for a deaf viewer: 16 of the 23 labelled real sounds on the dev clips are not detected.
Diagnosis (scripts/psed_lost_reals.py): most are genuine misses, not a naming issue — the
model scores the labelled sound near zero where BEATs scored 0.4–0.7 (helicopter 0.16 under
music, rain 0.02 under music, zipper 0.00, horse 0.01, crack 0.00, explosion 0.02, aircraft
0.00/0.01); five sit between 0.10 and 0.20 (a looser bar would keep them but breaks the
false-positive bar on DCASE). Two "reals" it arguably re-labels correctly (BEATs' Goose/Honk
where PSED hears Gobble/Turkey). Reading: frame-level fine-tuning on AudioSet-Strong made the
model precise and well-timed but conservative on long ambient sounds under music and speech
in YouTube video — the opposite failure to FLAM's. Recorded as attempt ten; BEATs stays. The
held-out real-world table on gold slice B (benchmark/audioset_detector_eval.py, declared in
docs/history/preregistrations/prereg_v4.md) is run for all three detectors regardless.
