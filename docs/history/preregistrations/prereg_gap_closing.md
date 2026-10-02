# Pre-registration: "gap closing" — does the panel restore what a hearing viewer gets?

*Committed 2026-09-15, late night, before any code for it exists. Adam's idea (three
descriptions: with sound, muted, with our panel), shaped by a four-Fable panel after the
with-sound / without-sound REFERENCE failed (`docs/prereg_av_reference.md`). The test-set
headline does not change whatever happens here.*

## 1. What is proposed — a score, not a reference

No model is asked whether a sound's source is visible. Instead, the same audio-visual
model answers the same **four fixed questions** about a clip in three conditions:

- **D_sound** — frames + soundtrack (a hearing viewer);
- **D_mute** — the same frames, no sound (a deaf viewer with no help);
- **D_system** — the frames with a system's side panel, no sound (a deaf viewer with help);
  one run per system: ours (gated), blind, caption.

The questions (fixed now): (1) What is happening? (2) Is anything dangerous or urgent
happening, and what? (3) What is the mood of the scene? (4) Is anything happening that
you cannot see, and what? Fixed questions, not free lists, because tonight's pilot showed
free lists differ in *wording* on every clip; fixed questions put both runs on the same axes.

**Gap** = how different D_sound is from D_mute (what the sound adds).
**Score of a system** = how much closer D_system is to D_sound than D_mute was:
`gain = sim(D_system, D_sound) − sim(D_mute, D_sound)`, per question, averaged;
`sim` = MiniLM cosine on the answer text (cheap, fixed), with an LLM pairwise judge
("which of the two answers is closer to the with-sound answer?", both orders) as the
secondary reading if the primary passes.

What it measures: **information restored** to a muted viewer, weighted naturally toward
sounds that change the answers (a siren changes "what is happening" and "is anything
dangerous"; birds chirping barely do). What it does **not** measure: the cost of redundant
pictures — the gate's whole benefit. So it is reported as **one of two axes**: (1) gain,
(2) cost = seconds the panel is on. Blind is expected to win axis 1 and lose axis 2; the
gate's case is the trade-off, not a single number. This is stated in advance so the score
cannot be read as "blind beats gated".

## 2. Kill criterion (the sanity check that must pass first)

The score is meaningless unless the model actually hears: the gap must be larger on
clips that contain an ambient sound than on clips with none, and larger than the
model's own run-to-run wording noise.

Sample: **60 development clips** — the first 20 by sorted file name of each label
(`unseen_ambient`, `seen_ambient`, `no_ambient`) among the 156 dev clips with a video.
Three runs per clip, greedy decoding, model Qwen2.5-Omni-7B (the only audio-visual
model that loads here; the caveat that it shares the system's vision encoder is carried):

- D_sound (frames at slot phase 0.5, with audio);
- D_mute (same frames, muted);
- D_mute2 (frames at slot phase 0.25, muted) — the **noise floor**: how much the answers
  change when nothing but the frame timing changes.

Statistics, fixed now: `gap = 1 − cos(D_sound, D_mute)` averaged over the four questions;
`noise = 1 − cos(D_mute, D_mute2)` likewise.

**PASS iff** (a) AUROC of `gap` for the 40 ambient clips (unseen + seen) against the 20
no-ambient clips **≥ 0.75**, and (b) the median `gap` on ambient clips exceeds the 90th
percentile of `noise` over all 60 clips.

Reported alongside, **not** a criterion: AUROC of `gap` for the 20 unseen clips against
the 40 others (the visibility question — deliberately not the bar), the per-question
gaps, and the gap on siren/alarm clips.

## 3. What happens on pass / fail

- **Pass:** compute D_system for gated, blind and caption from the **existing 100 test
  renders** (no new renders, ~5 GPU-h) and report gain and panel-on time per system as a
  secondary analysis. The test set is then touched a second time, which LIMITATIONS will
  state. No re-runs; no prompt changes.
- **Fail:** reported as a negative pilot with the numbers; the score stays a proposal in the
  report's evaluation section with this kill criterion attached.

Either way the report gets a "proposed evaluation" section: the three-description design,
the two axes, the kill criterion, and the evidence from tonight's two pilots.

## 4. What will not be done

No prompt, question or threshold change after seeing the dev numbers; no tuning against
the human-grounded scores; the headline is unchanged.

## 5. Outcome (added 2026-09-16, 00:30, after the run)

Qwen2.5-Omni-7B, 60 dev clips, three runs each (`benchmark/gap_closing_pilot.json`).

| measure | result | bar |
|---|---|---|
| AUROC of gap, 40 ambient vs 20 no-ambient clips | **0.529** | ≥ 0.75 |
| median gap on ambient clips | 0.192 (no-ambient: 0.205) | > 90th pct of noise = 0.395 |
| median frame-timing noise | 0.147 | — |
| AUROC unseen vs rest (information only) | 0.475 | — |
| "danger" answer changed with sound | 6 of 60 clips | — |

**FAILED.** The change in the answers caused by the soundtrack is about the same size
as the change caused by shifting the frames a quarter-slot, and no larger on clips with an
ambient sound than on clips without one. Per clip there is real signal on some (kitchen
smoke alarm: gap 0.42 vs noise 0.06, the with-sound answer says "a smoke detector is going
off"), but it does not separate the groups.

Two things to disclose. (1) A flaw in the criterion as written: the no-ambient clips contain
speech or music, which legitimately changes the answers ("what is happening" becomes what
is being said), so "ambient vs no-ambient" is not a clean test of *hearing ambient sound*;
the noise-floor test (b) does not have this flaw and also failed. (2) The "danger" question
is almost always answered "No" in every condition, so its gap is zero by construction on
most clips. No prompt, model or criterion was changed after seeing the numbers; the quiz
version (§1) is not built. Reported as a negative pilot.
