# Working smarter: process rules and robustness plan (2026-09-13)

Written after two days in which the display bar went 0.30 → 0.35 → 0.40 → 0.35 and a
veto and a corroboration band went in and out, each move justified by the five clips on
the desk and undone by the next five. Adam: "every video we see we have to adjust to
that video instead of working in general."

## What practice says

**One operating point, chosen on a dev split, reported with its error rate.** DCASE scores
sound event detection threshold-independently (PSDS, `sed_scores_eval`) *and* at one
declared operating point; per-class optimum thresholds are computed on dev, never by eye.
- https://arxiv.org/html/2201.13148 (threshold-independent evaluation)
- https://arxiv.org/html/2406.08056v1 (DCASE 2024 Task 4)

**Error analysis is a catalogue, not a reaction.** Collect misclassified dev examples,
categorise, fix by frequency. Ad-hoc manual analysis produces biased conclusions because
a feature seen in a failure is often present in successes too.
- https://arxiv.org/pdf/2201.05017 (Towards Automated Error Analysis)
- https://arxiv.org/pdf/2212.08216 (Azimuth)

**Multiple-choice answers from LLMs/VLMs carry a letter prior.** Permutation-based
debiasing (PriDe) estimates and removes it; the vote-across-orderings we use is the crude
form.
- https://arxiv.org/abs/2309.03882 (Zheng et al., LLMs are not robust MC selectors)
- https://arxiv.org/pdf/2509.16805 (selection bias in vision-language models, 2025)

**Off-screen classification is a recognised task with public gold labels.** DCASE 2025
Task 3 (audiovisual track) adds onscreen/offscreen classification per event.
- https://arxiv.org/abs/2507.12042 (Shimada et al. 2025)
- https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content

**Separate before detecting.** Cinematic audio source separation (dialogue / music /
effects) is a defined problem with an open model; separation as a pre-processing step
improves SED.
- https://arxiv.org/abs/2407.07275 (DnR v3), https://github.com/kwatcharasupat/bandit-v2
- https://arxiv.org/pdf/2007.03932 (Improving SED using sound separation)

## Rules

1. Knobs (bar, veto, bounds) are set by the dev-split sweep, once. The test split is never
   looked at for tuning.
2. Every failure seen in a video goes into `docs/history/plans/failure_catalogue.md` with clip, stage,
   confidence. A design change needs >= 3 catalogued cases of one pattern, or a metric
   move on dev. One-offs go to Limitations.
3. Demos illustrate. They never trigger a change directly.
4. The error rate at the operating point is a result, and is reported.
5. Any change must be general (no clip-specific logic) and re-measured on dev before it
   ships.

## Upgrades, by value / cost

| | upgrade | why | cost |
|---|---|---|---|
| A | Evaluate the visibility gate on DCASE 2025 Task 3's onscreen/offscreen labels | gold for the core claim, without hand-tagging | 1 day |
| B | Run BEATs on the *effects* stem from Bandit-v2 (dialogue/music/effects separation) | the siren under the score, the phone alert: our recurring miss class | 1 day |
| C | Per-class thresholds (DCASE F1_MO on dev) + BEATs/PANNs probability ensemble | phantoms and misses cluster by class; one bar cannot fit Bird and Siren | 0.5 day |
| D | PriDe-style probability debiasing for the VLM questions | a confidence instead of a vote; fewer "cannot tell" splits | 0.5 day |
| E | Open-vocabulary second opinion (CLAP) where BEATs is unsure | sounds AudioSet has no class for | future work |

## This week

Freeze the architecture → launch v3 on the frozen config → A and B in parallel (they add
gold and a front end; they do not touch the pipeline's logic) → C from the dev sweep →
catalogue everything seen so far. D and E after.
