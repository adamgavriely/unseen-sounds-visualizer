# Pre-registration: Step 4, one calibrated keep-score from all saved answers (DEV only, CPU)

Written and committed 2026-10-07 before any model was fitted or any curve read. Adopt nothing; TEST not touched.

## Unit and truth

Unit = a DEV candidate burst (`verify_items_dev.json`, 1196 bursts: every BEATs / FlexSED / band / DASM candidate,
Speech / Music left out, same family joined across gaps <= 2.5 s). Training label (from gold, used only to fit):
**good** = the burst's start lies in the onset window (-0.5 .. +1.0 s) of a same-family gold sound that is needed
with importance 2-3 (it would be a hit); everything else is **not good**.

## Features (24)

Detector (cluster, from the cached frame scores, family = max over the ear's classes in the burst's family):
1-2 BEATs max / mean in the burst; 3-4 FlexSED max / mean; 5 DASM max in burst +- 0.5 s; 6-8 BEATs, FlexSED, DASM
family max over the whole clip; 9 BEATs Speech / Music max in the burst. Burst: 10 length; 11 number of member
candidates; 12-15 has a BEATs / FlexSED / band / DASM member. Listeners (saved runs): 16 Qwen3-Omni yes/no logit
(max over P1 items of the family overlapping the burst; missing -> empty); 17 Audio Flamingo Next open-list names it
(V4); 18 Audio Flamingo yes/no (YN0). Gate (frozen arm's records overlapping the burst; missing -> empty): 19 share of
stretches seen (majority); 20 share seen by the a/b rule (split -> majority). New Omni answers (Step 3/3b): 21 closed
choice P(own family) (siblings + none, mean of two orders); 22 P(none); 23 onset audio+video gate P(yes); 24 mean
"still heard" P(yes) over the burst's seconds.

## Model and validation

Gradient-boosted trees (sklearn HistGradientBoostingClassifier, max_depth 3, learning_rate 0.05, 200 iterations,
min_samples_leaf 20, missing values native), and L2 logistic regression (C 1.0, standardised, missing -> the
training fold's median) as the second model. Clip-grouped
10-fold CV (GroupKFold by clip); every burst's probability is out-of-fold.

## Two variants

- **(a) reorder-only**: the AB-m candidate's pictures (Step 2) stay as drawn; each picture gets the out-of-fold
  probability of the burst it came from (same family, overlapping; the max if several); a threshold removes the
  pictures below it. It can only remove pictures.
- **(b) replace the hand vetoes**: every burst with probability above the threshold becomes a picture (label = its
  family, start = burst start, end = burst end); the rule chain and the gate are not used except as features.
  Its representation baseline is reported too: the bursts the frozen system draws, rendered the same way.

## Report

Pictures scored with scorer v2. For each model and variant: the hits-vs-wrong curve over thresholds; hits at
wrong = 5 / 10 / 15 (most hits at or below that many wrong); onset cost and cost_cov at p > 1/3. Compared with
frozen 29 / 15 and the a/b candidate 32 / 16.
A **clear win** = at least 3 more hits than AB-m at the same or fewer wrong, or at least 3 fewer wrong at the same or
more hits. Otherwise the report says plainly that it does not beat the a/b candidate.
