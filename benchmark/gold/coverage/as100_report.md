# Our detector vs state-of-the-art SED on 100 AudioSet-Strong clips

**Data.** 100 clips: the first 100 (sorted by name) of the 340 fresh AudioSet-Strong EVAL clips that have stage-4 output from 4 Oct.
Gold: `audioset_eval_strong.tsv`, with 920 events in total.
- **Ours:** stage-4 events of the frozen D' arm (`SHIP8+MD3+WW5+SL|proposed`), read from `~/MscProj_tg/data/work/r13freshf0*/stage4.json`.
- **SOTA:** the PretrainedSED strong models (Schmid et al.), all 5 of them, run on the same 16 kHz audio (`wav16`).

**Fair class set C (368 classes).** C is the 447 PretrainedSED classes that our stage 4 can name (BEATs 527 plus FlexSED, matched by MID), minus the Music and Speech ontology subtrees and the repo's never-drawn rule.
- Gold in C: 522 events across 100 classes.
- Ours in C: 118 of its 327 events. The other 209 are mostly Speech and Music.
- Every system is restricted to C.

## (a) PSDS1, SOTA only (their protocol: median filter 9, dtc = gtc = 0.7, alpha_st = 1)

| model | PSDS1 (126 gold classes) | PSDS1, no class-spread penalty | PSDS1 on C |
|---|---|---|---|
| ATST-F | **0.188** | **0.582** | 0.176 |
| ASIT | 0.173 | 0.568 | 0.165 |
| BEATs | 0.163 | 0.567 | 0.150 |
| fpasst | 0.153 | 0.550 | 0.155 |
| M2D | 0.144 | 0.542 | 0.140 |

The paper reports 0.41–0.47 on the full eval set. Our values are lower because PSDS1 subtracts the spread of scores between classes, and with 126 classes having about 4 events each, that spread is very large. Without the penalty, the scores (0.54–0.58) are at or above the paper's. The ranking (ATST-F first) matches the paper, so the models run correctly.

## (b) Segment-based (1 s) and event-based (onset 0.2 s, offset 0.2 s or 20 %) metrics, class set C, sed_eval

| system | seg P | seg R | seg F1 micro | seg F1 macro | evt P | evt R | evt F1 micro | evt F1 macro |
|---|---|---|---|---|---|---|---|---|
| **ours** | 0.34 | 0.12 | 0.182 | 0.132 | 0.02 | 0.00 | 0.006 | 0.006 |
| ATST-F @0.5 | 0.66 | 0.14 | 0.226 | 0.085 | 0.25 | 0.04 | 0.066 | 0.044 |
| BEATs @0.5 | 0.70 | 0.15 | 0.248 | 0.096 | 0.28 | 0.04 | 0.073 | 0.032 |
| fpasst / M2D / ASIT @0.5 | 0.62–0.67 | 0.12–0.14 | 0.21–0.23 | 0.06–0.08 | 0.20–0.31 | 0.03–0.04 | 0.05–0.07 | 0.02–0.04 |
| ATST-F best threshold* | 0.35 | 0.38 | 0.362 (th 0.15) | 0.271 | 0.32 | 0.17 | 0.218 (th 0.2) | 0.132 |
| BEATs best threshold* | 0.27 | 0.47 | 0.344 (th 0.1) | 0.315 | 0.23 | 0.15 | 0.177 (th 0.2) | 0.105 |

\* Optimistic: the threshold was chosen on these same clips. Extra metric, event F1 with onset only (no offset rule): ours 0.047; ATST-F 0.083 at 0.5 and 0.299 at best threshold.

## (c) Onset rule of our benchmark (same class, onset within −0.5 to +1.0 s, one-to-one; 522 gold events)

| system | hits | wrong | precision |
|---|---|---|---|
| **ours** | 30 | 88 | 0.25 |
| ATST-F @0.5 | 28 | 53 | 0.35 |
| BEATs @0.5 | 38 | 40 | 0.49 |
| fpasst / M2D / ASIT @0.5 | 28 / 25 / 23 | 44 / 49 / 48 | 0.32–0.39 |
| ATST-F best threshold* (0.15) | 152 | 249 | 0.38 |
| BEATs best threshold* (0.1) | 159 | 478 | 0.25 |

## Caveats

- Our stage 4 is tuned for off-screen sounds worth a picture, not for all 447 classes. Most of C's gold is texture and Foley classes it is not built for: Tap, Walk/footsteps, Mechanisms, Clapping, Tick, Breathing, Wind noise, Sound effect.
- Our events are merged (gap 2.5 s) and pass a veto chain, so their offsets do not follow the AudioSet segmentation. That is why ours scores near zero on event-F1 with the offset rule.
- The SOTA models were trained on AudioSet-Strong labels, the same label style as the gold. Ours often names a parent class (Vehicle, Animal), and exact-class matching counts that as wrong.
- With 100 clips, the results have large confidence intervals.

## Files

- Scripts: `benchmark/gold/coverage/as100_prep.py`, `as100_sota.py`, `as100_metrics.py`.
- Lists: `as100_lists.md` (the 100 clips and the class set C).
- Outputs on the cluster in `~/as100_eval/`: `metrics.json` (with threshold sweeps), `sota_<model>.npz`, `gold.json`, `ours.json`, `clips.txt`, `classes_C.txt`.
