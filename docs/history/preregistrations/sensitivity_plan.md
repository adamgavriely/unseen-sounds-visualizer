# "Missing a picture is worse than showing a wrong one" — what that means, and the plan

Adam, 2026-09-23, before sleeping: *missing a picture is bad for us, I think more than presenting a
picture that is not needed. How does it affect the pipeline? Can we control the sensitivity easily?*

## 1. That preference already has a name in this project: it is beta

The viewer cost declared in amendment 9 is

    cost per clip = 4 x (needed sounds with no picture) + beta x (wrong pictures)

A missed sound always costs 4. **beta is exactly "how bad is a wrong picture", and the ratio 4/beta
is "how many wrong pictures is one missed sound worth".** Adam's instinct -- misses are worse than
the current setting implies -- means beta should be LOWER than 2, not higher. At beta = 2 a miss is
worth two wrong pictures; at beta = 1 it is worth four.

The decision rule follows directly, and four reviewers derived it independently in September: a
picture is worth showing when its chance of being right exceeds **beta / (4 + beta)**.

    beta = 2.0   show a picture when it is more than 33% likely to be right
    beta = 1.0   ... more than 20% likely
    beta = 0.5   ... more than 11% likely

So "misses matter more" does not just re-score the same system -- it changes how trigger-happy the
system should be, and it should be MORE trigger-happy, not less.

## 2. What it does to our results (DEV, 49 clips, measured tonight)

    cost per clip        b=0.0  b=0.5  b=1.0  b=1.5  b=2.0  b=3.0  b=4.0
    v4b6 (yesterday)      1.71   2.32   2.92   3.52   4.12   5.33   6.53
    veto only             1.71   2.19   2.67   3.15   3.63   4.59   5.55
    veto + PANNs          1.80   2.12   2.45   2.78   3.10   3.76   4.41
    per-family bars       1.47   2.27   3.06   3.86   4.65   6.24   7.84
    silence               2.69   2.69   2.69   2.69   2.69   2.69   2.69

    beta      best cell               beats silence?
    0.25      per-family bars         YES   1.87 < 2.69
    0.50      veto + PANNs            YES   2.12 < 2.69
    1.00      veto + PANNs            YES   2.45 < 2.69
    1.25      veto + PANNs            YES   2.61 < 2.69
    1.50      veto + PANNs            no    2.78 > 2.69
    2.00      veto + PANNs            no    3.10 > 2.69

**The pipeline beats silence on the whole benchmark for any viewer who prices a wrong picture below
about 1.4.** Above that, showing nothing is cheaper. The crossover is the result; which side of it a
viewer sits on is a question about viewers, not about the pipeline.

Note also that the best cell CHANGES with beta: below about 0.4 the high-recall per-family
configuration wins, above it the conservative one does. The system has two sensible settings, not
one.

## 3. The trap, and it is the important part of this note

**We cannot now declare beta = 1 because it makes us win.** Beta = 2 was fixed in September from this
project's own judging rubric, before any of tonight's results existed. Choosing a different beta
after seeing where the crossover falls is the same error as widening the timing window after seeing
which pictures were late -- and a reviewer will ask exactly when the judgment was formed.

What is legitimate is to **report the whole curve and say where the line falls**: the pipeline is
worth using for any viewer who prices a wrong picture under ~1.4 missed-sound-units; the project's
own September rubric priced it at 2, which is above that line. That sentence is honest, it is
measured, and it is a result rather than an excuse.

One more thing to say plainly, because someone will ask: the selected cell's precision is 0.256,
which is BELOW the beta = 2 break-even of 0.33. Each individual picture is, in expectation, a small
net loss at that price -- and the configuration still improves cost, because the realistic
alternative is not "show nothing", it is "show more and worse pictures", which is what every other
cell and the blind baseline do.

## 4. Can we control the sensitivity easily? Yes — three validated presets, no new fitting

Every configuration below is already rendered and scored on DEV. A single `SENSITIVITY` setting maps
to knobs that already exist (`FLEXSED_BAR`, `FLEXSED_FAMILY_BARS`, `FLEXSED_VETO`, `PANNS_VETO`):

    preset         config                              recall  wrong/clip  best when
    sensitive      per-family bars + veto              0.455    1.59       beta < ~0.4
    balanced       veto only                           0.364    0.96       -
    conservative   veto + PANNs veto  (SELECTED)       0.333    0.65       beta > ~0.4

This is a half-day of work, adds no parameter that was not already validated, and lets the thesis
report all three rather than defending one.

## 5. The plan — how to settle beta properly

**Measure it from deaf and hard-of-hearing viewers instead of asserting it.** This is the one piece
of new evidence that makes the choice defensible, and it is a contribution in its own right.

  * **Design.** Pairwise forced choice. Each item shows a viewer two failures from the same clip:
    one where a needed sound got no picture, one where a picture appeared for a sound that was not
    there. Question: "which of these is worse for you?" 20-30 pairs, 5-10 DHH participants.
  * **Analysis.** Beta is estimated from the choice rate; report the point estimate and a 95%
    interval. The September rubric's beta = 2 becomes the null hypothesis, not the answer.
  * **Order of operations, which matters more than the design.** The study is pre-registered --
    items, participants, analysis and stopping rule written down -- BEFORE a single response is
    collected, and before the number is compared with tonight's curve.
  * **What it buys.** If DHH viewers price a miss above about 3 wrong pictures (beta < 1.4), the
    pipeline beats silence on this benchmark at a beta that came from viewers rather than from us.
    If they do not, the honest finding is that at their own price silence is still cheaper on a
    benchmark where most clips contain nothing to draw -- which is worth reporting and is a sharper
    result than anything the rubric could give.

**What happens tonight is unaffected.** The TEST look uses the declared beta = 2 and the declared
go/no-go. Beta is orthogonal to which cell passes; nothing in this note leaks into that read.
