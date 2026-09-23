# Scoring the final system with the judge, automatically — and checking the judge first

2026-09-23. Adam asked two things: *"we should of used a VLM judge, we said we wanted it to judge
us no?"* and, earlier, *"how we know the VLM judge do a good job?"* Those are one question, and the
order matters: a judge whose verdict nobody has checked produces a number nobody can defend.

So this has two halves, and **half B is the gate on half A**.

---

## A. What the judge run does

The machinery already exists (`slurm/job_gold_judge.sh`) and runs in three steps:

1. **describe** — a vision model looks at the rendered panel of each clip and writes what a viewer
   would learn from it. It never sees the labels.
2. **judge** — a language model compares that description with the *reference*: the sentence built
   from the annotator's own ticks saying what a deaf viewer needed to know here (or the words
   "nothing beyond the picture" when nothing was needed). It returns 1-5 and a reason.
3. **rubric cap** — a hard rule then caps the score, so the judge cannot give credit for a picture
   of a sound the annotator never marked. Without the cap the same clip scored 4 on a picture of a
   tambourine in a clip whose reference was "nothing beyond the picture".

Run, once both baseline renders finish:

    TAG=dev_symgen_v30 V4=590 CLIP_DIR=data/input/gold_dev sbatch slurm/job_gold_judge.sh

Output: one row per (clip, system) with `score`, `why`, `human_tag`, for all three arms — ours,
blind (draw every detected sound), and the caption-only control.

## B. The trust check, run on the same rows

Three questions, each with a number decided before the run.

**B1 — Does the judge agree with the annotator?** Each clip already has a viewer cost from the gold
labels (4 x missed + beta x wrong pictures). If the judge is doing its job, its score should fall as
that cost rises. Reported as Spearman rank correlation across the 49 DEV clips, with a bootstrap CI.
**Pass: rho <= -0.4 and the CI excludes 0.** Below that, the judge's opinion is close to noise and
half A is reported as "not informative" rather than as a result.

**B2 — Does it catch a picture we already know is wrong?** 33 pictures on the DEV clips are wrong by
the annotator's own ticks. Split the clips into those that carry at least one and those that carry
none, and compare the judge's mean score. **Pass: the clean clips score higher, and the gap's CI
excludes 0.** This is the direct test of whether the judge sees a false picture at all.

**B3 — Does it say the same thing twice?** Judge 20 clips a second time with a different seed and
report how often the score changes by more than one point. This is the ceiling: no agreement number
in B1 or B2 can be better than the judge's agreement with itself. **Report, do not gate.**

## What each outcome means

  * **B1 and B2 pass** — the judge is a usable second opinion. Half A's ranking of the three arms
    goes in the thesis as a secondary result, next to the per-sound metric, and if both point the
    same way that is real corroboration from a measurement that never saw the labels.
  * **Either fails** — that is itself the finding, and it is the reason the headline metric counts
    pictures against the annotator's ticks instead of asking a model. It is written up as a
    limitation of automatic judging, not hidden.

## Why the headline number still does not use the judge

The per-sound metric asks one mechanical question — did a picture of the right family appear within
[-0.5, +1.0] s of a needed sound — and every input to it is the annotator's. Nothing about it can
drift with a model version. The judge is a second, independent opinion on the same clips, and it is
reported as that.
