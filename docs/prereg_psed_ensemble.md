# Pre-registration: the five-backbone PretrainedSED average (detector attempt eleven)

*Committed 2026-09-19, before the run. Adam's question ("combine a long-window and a
short-window model?") → Fable → the recipe the PretrainedSED paper itself uses: the same
frame-level training on five transformers (BEATs, ATST-F, fPaSST, M2D, ASiT) and their
average, PSDS1 47.1 against 46.5 for BEATs-strong alone. Disclosed: before this
declaration, four naive BEATs(weak)+PSED combinations (OR, MAX, MEAN, AND) were looked at
on slice B from cached scores — none is used here; the numbers are in
scripts/fusion_sweep.py's output (OR 64.8% / 7.9 per min, MAX 67.8% / 9.4, MEAN 54.2% / 1.6,
AND 29.7% / 0.3).*

## What is tested

The mean of the five backbones' frame probabilities (each fine-tuned on AudioSet-Strong by
the paper's pipeline; public checkpoints; the same 447 classes, name mapping and span rule
as the single model). The bar is chosen on DCASE gold by the same rule as before (loosest
bar with false spans/min ≤ 5.2), then everything else is measured at that bar.

## Pass rule (declared now; slice B numbers of the single model are the reference)

| measure | PSED-BEATs alone | the average must reach |
|---|---|---|
| slice B: recall of masked consequential sounds (236 events) | 62.7% | **≥ 65.7%** (+3 points) |
| slice B: false spans per minute | 2.59 | **≤ 2.59** |
| dev: labelled real detections kept (of 23) | 7 | **≥ 6** (lose at most one) |

PASS iff all three. On pass, the average becomes the v4 detector (`AED_MODEL = "psed_ens"`),
disclosed as attempt eleven; the v4ab row is re-run with it. On fail, the single model stays
and the numbers go in the table. No further detector attempt after this one; slice B has
now been looked at three times and is retired as a selection set.
