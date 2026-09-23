# Beta: what it is, what values it may take, and how it should be measured

Adam, 2026-09-23: *we don't choose 1 or 2 regarding beta — we discuss it, make a graph to show that
the significance depends on the DHH viewers, and say further study should be done. Define its limits,
define how it should be scored.*

## 1. Definition

    cost per clip = 4 x (needed sounds that got no picture) + beta x (pictures that were wrong)

A missed sound is fixed at 4 so that only one quantity varies. **Beta is the price of a wrong
picture, measured in the same units.** The ratio 4/beta is "how many wrong pictures is one missed
sound worth". Beta = 2 means one missed sound is as bad as two wrong pictures.

The decision rule follows from the cost function alone, with no free choice: a picture is worth
showing exactly when its probability of being right exceeds

    p* = beta / (4 + beta)          beta 0.5 -> 11%   beta 1 -> 20%   beta 2 -> 33%   beta 4 -> 50%

This is why beta is not a scoring detail. It sets how trigger-happy the system should be.

## 2. The admissible range, derived rather than chosen (DEV, 49 clips)

Two ends of the range are fixed by the structure of the problem, not by preference:

    beta < 0.35    the blind baseline overtakes the gated system.
                   A wrong picture is so cheap that a visibility gate costs more than it saves --
                   at this price one should simply draw every detected sound.

    beta > 4.30    even an ORACLE gate -- the same gate handed the annotator's own perfect sound
                   list -- costs a viewer more than showing nothing. Above this price no system of
                   this shape can win, however good its detector, so the question is moot.
                   (Checked: all 49 DEV clips have a complete oracle vote record, so this bound is
                   exact on DEV and not an artefact of missing data. Note precisely what "oracle"
                   means here: the gate is handed the annotator's perfect SOUND LIST, but it is
                   still OUR gate, with the same visibility model and the same eleven cases where it
                   looks at a source that is plainly on screen and says it is not there. So 4.30 is
                   the ceiling for "a perfect detector feeding this gate", not for a perfect system.
                   A perfect gate as well would push the bound higher, and that number is not
                   measured here.)

So **beta is only a meaningful question on [0.35, 4.30]**. Inside it, the two crossings that matter:

    beta = 0.95    the blind baseline meets silence
    beta = 1.37    OUR system meets silence (all clips), 95% interval [0.71, 2.19] -- an interval
                   that CONTAINS the rubric's asserted 2.0, so on this evidence neither "we beat
                   silence at beta = 2" nor "silence beats us at beta = 2" is established
    beta = 3.33    our system meets silence on the UNSEEN clips alone (95% interval [1.33, 10.0]
                   on 14 clips -- badly underpowered, and reported as such). For comparison the
                   blind baseline crosses at 3.11 on the same clips, so on this category the gate's
                   advantage is small.

                   CORRECTION: an earlier note in this project put this crossing at 5.00. That came
                   from applying the veto to v4b6's pictures in the scorer rather than from the
                   rendered configuration. The rendered number is 3.33 and supersedes it.

**Two different intervals, which must not be conflated:**

  * the **visibility gate pays for itself** for every beta above **0.35**;
  * the **whole system beats silence** for every beta below **1.37** (all clips);
  * between them, **[0.35, 1.37]**, both hold at once, and that band is where the project's claim
    lives. The rubric's asserted beta = 2 sits just outside it on the all-clips measure and inside
    it on the unseen-clips measure.

## 3. Cost at each price (DEV, 49 clips, selected configuration)

    beta        0.0   0.5   1.0   1.4   2.0   3.0   4.0   6.0   8.0
    ours       1.80  2.12  2.45  2.71  3.10  3.76  4.41  5.71  7.02
    blind      1.63  2.19  2.76  3.20  3.88  5.00  6.12  8.37 10.61
    oracle     0.33  0.60  0.88  1.10  1.43  1.98  2.53  3.63  4.73
    silence    2.69  2.69  2.69  2.69  2.69  2.69  2.69  2.69  2.69

## 4. How beta should be SCORED — two different meanings, both answered

**(a) How the SYSTEM is scored as a function of beta.** Not at a point. The report is the operating
curve above, with the four landmarks marked, for each configuration. The claim the project makes is
a claim about an interval, never about a single number: *whether this system helps depends on how a
deaf viewer prices a wrong picture, and we win below 1.37.*

**(b) How beta itself should be MEASURED from viewers.** Not asserted from a rubric. A study is
needed, it is not being run now, and its design is fixed below so that it cannot be shaped later by
the result it is meant to test.

## 5. The study, pre-registered now and NOT run

**Why not forced-choice pairs.** The obvious design -- "which is worse, missing this sound or seeing
this wrong picture?" -- fails twice. First, a deaf viewer cannot perceive a miss; the thing being
priced is invisible to them by definition, so the question has to describe it, and whoever writes the
description chooses the answer ("the ambulance behind you" guarantees one reply). Second, constructed
stimuli are exactly where the experimenter's own beta leaks in.

**The design that is pre-registered instead.** Use the pictures the system actually produced.

  * **Stimuli:** the rendered side-panel output for the 60 TEST clips -- real pictures, real timing,
    nothing constructed.
  * **Task:** one question per clip, wording fixed now: *"Did this panel help you, hurt you, or
    neither?"* Three points: helped / neither / hurt.
  * **Participants:** 5-10 deaf or hard-of-hearing viewers, one session each.
  * **Model, fixed now:** `helped ~ a x (missed needed sounds) + b x (wrong pictures)`, fitted by
    ordinal regression; beta_hat = 4 x b / a, reported with a 95% interval.
  * **Stratification, fixed now:** the same fit repeated within the annotator's `importance` rating
    (2 versus 3) and within clip category (unseen / mixed / seen / no-ambient). Reported whether or
    not the strata differ, because a beta that changes with context is itself the finding -- the
    all-clips crossing at 1.37 and the unseen-clips crossing at 5.00 already suggest it does.
  * **What counts as an answer:** the rubric's beta = 2 is the null. The study either places the
    measured beta below 1.37, in which case the system beats silence at a price set by viewers
    rather than by us, or it does not, in which case the honest finding is that on a benchmark where
    half the clips contain nothing to draw, silence is still cheaper at the viewers' own price.

**Order of operations:** wording, model and stratification are frozen above, before any viewer sees
a clip and before the number is compared with anything in section 2.

## 6. What this means for the thesis

The significance of this work is conditional, and saying so precisely is stronger than picking a
number. The sentence is: *the gated pipeline is cheaper for a deaf viewer than showing nothing
whenever a wrong picture costs less than about 1.4 missed-sound-units, and the visibility gate earns
its place whenever a wrong picture costs more than 0.35; where a real viewer sits in that band is an
empirical question this project defines and leaves open.*
