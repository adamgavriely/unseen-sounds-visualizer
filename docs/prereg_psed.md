# Pre-registration: PretrainedSED BEATs-strong as the detector (v4 stage 4, attempt ten)

*Committed 2026-09-18 night, before the run. Decided with Adam after Fable's review of the
two FLAM attempts (both failed on precision; docs/prereg_v4.md §4).*

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
