# Beta: what it is, what values it may take, and how it should be measured

> **Note 2026-09-27:** every crossing number in this file (1.37, 0.35, 4.30, 5.00) comes from the 23 Sept DEV pipeline and is
> superseded — see ch5 §5.5 (final: 0.39 / 2.56 TEST, 0.46 / 2.33 DEV). §5's TEST stimuli become DEV stimuli for any hearing pilot.

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

> **Superseded 2026-09-27 — see ch5 (final crossings 0.39/2.56 TEST, 0.46/2.33 DEV).** The numbers in this section
> (0.35, 1.37 [0.71, 2.19], 4.30) come from the 23 Sep DEV curve under `use_v4("59")` (8-s cap on, before amendment 21).
> Final: ours cheaper than blind above β = 0.39 (TEST, `test_final_v33`) / 0.46 (DEV, `dev_monocap_v31`); cheaper than
> silence below β = 2.56 (TEST) / 2.33 (DEV); the oracle line is not crossed within β ≤ 4, and 4.30 is not verified from a
> committed file (`docs/history/earlier_drafts/thesis/ch5_results.md` §5.5). The text below is kept as a record.

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

> **Superseded 2026-09-27 — see ch5 (final crossings 0.39/2.56 TEST, 0.46/2.33 DEV).** The sentence below uses the
> stale 1.4 and 0.35. Final wording (`docs/history/earlier_drafts/thesis/ch5_results.md` §5.5): on TEST the gated system is the cheapest of the
> three for β between 0.39 and 2.56 (DEV 0.46 to 2.33). The text below is kept as a record.

The significance of this work is conditional, and saying so precisely is stronger than picking a
number. The sentence is: *the gated pipeline is cheaper for a deaf viewer than showing nothing
whenever a wrong picture costs less than about 1.4 missed-sound-units, and the visibility gate earns
its place whenever a wrong picture costs more than 0.35; where a real viewer sits in that band is an
empirical question this project defines and leaves open.*

## 7. What the published DHH literature says beta is (2026-09-23)

The viewer study in section 5 cannot be run -- there is no DHH participant community available to
this project -- so beta was estimated from the literature instead, and the estimate is reported with
its direction of disagreement rather than its convenience.

**The strongest single number, and why it is NOT transferable.** The NER model (Romero-Fresco), the
viewer-centred standard for live caption quality, scores a *serious* error -- "misleading but
credible information" -- at -1.0 and a *standard* error -- "confusion and loss of information" -- at
-0.5. A misleading item is priced at twice an omission, which in our units would put beta near 8,
above the 4.30 ceiling at which no system of this shape can win at all. That transfer fails for a
structural reason: NER prices a SUBSTITUTION, where a wrong word replaces the right word in a
channel the viewer trusts completely and cannot check. Our wrong picture is an ADDITION -- it sits
beside correct subtitles, in a panel the viewer knows is machine-made, with the video present to
contradict it. A horse drawn over a street scene is a glance and a dismissal; a wrong caption is
believed. NER therefore bounds beta for an unverifiable channel, and ours is verifiable.

**What the sound-awareness literature actually gives is two regimes, not one number.** SoundWatch
(Jain et al., ASSETS 2020, 8 DHH participants) found users wanted minimum delay for urgent sounds
but "maximum accuracy for non-urgent sounds, to not be unnecessarily disturbed". The large
preference survey found users want filtering by importance, not every sound. Beyond Subtitles
(Adobe, ASSETS 2022, 11 DHH viewers) found viewers want the IMPORTANT non-speech sounds shown and
warns graphics are "evocative, but potentially ambiguous or distracting". Caption Royale (39 DHH)
found any visual enhancement is judged first on readability and MINIMAL DISTRACTION.

Read together: **for danger sounds a miss is the catastrophe and a false alarm is an annoyance;
for atmosphere a picture is a courtesy and a wrong one is pure distraction.** That is exactly the
axis the annotator already rated. The literature-motivated estimate is therefore

    sounds rated importance 3 (danger, consequence)   beta ~ 0.75   (range 0.5-1.0)
    sounds rated importance 2 (atmosphere, narrative) beta ~ 3.0    (range 2-4)

A single pooled beta is a mixture of the two weighted by the gold set's own ratio, which lands near
1.5-2.5. **The September rubric's asserted 2 is, as a pooled number, close to where the literature
puts it.**

## 8. The result under the literature's own weights — SUPERSEDED, see section 10

**The mapping used in this section is wrong and its two conclusions do not stand.** It assigned the
"ambient" beta of 3.0 to sounds rated importance 2, which in this gold set are machine guns,
explosions, chainsaws, alarms and helicopters -- not room tone. The literature's non-urgent regime
(SoundWatch's "speech, background noise") corresponds to the importance-1 tier, which the metric
already treats as don't-care and never scores. Kept below for the record; read section 10 instead.

A wrong picture carries no importance of its own, so the regime is set by the CLIP: a clip holding a
needed sound rated 3 is "urgent" (beta 0.75), every other clip is "ambient" (beta 3.0).

    DEV (49 clips: 8 urgent, 41 ambient)        flat beta=2   literature split
      ours (final cell)                            3.10           3.30
      blind                                        3.88           4.45
      silence                                      2.69           2.69
      silence - ours: -0.60, 95% CI [-1.35, +0.12]            not significant

    TEST (60 clips: 10 urgent, 50 ambient)      flat beta=2   literature split
      ours (final cell)                            2.53           2.67
      blind                                        3.27           3.72
      silence                                      2.87           2.87
      silence - ours: +0.19, 95% CI [-0.59, +1.01]            not significant

**This is worse for the project than the rubric's flat beta = 2, not better**, because most clips in
the gold set are ambient and the literature prices a wrong picture higher there. Adam's instinct --
that a missed picture hurts more than an unnecessary one -- is supported by the literature for the
sounds rated 3, which are the 28 that matter most, and is contradicted for the majority of sounds in
the set. Both halves of that are reported.

**One thing the split makes stronger, not weaker:** the gate's advantage over the blind baseline
GROWS under it (TEST: 2.67 against 3.72, a gap of 1.05, against 0.74 at flat beta = 2), because
blind draws roughly twice as many wrong pictures and the split prices them higher on the clips where
most of them fall.

## 9. The third axis: DHH is not one audience

The project's own literature review already records that hard-of-hearing viewers were significantly
more positive about enriched captions while several Deaf viewers disliked them (emotive captions,
11 DHH), and that Deaf participants cannot decode phonetic onomatopoeia at all. Beta is therefore
population-dependent as well as sound-dependent, and a single number would be a mixture over that
axis too. This is not a gap to apologise for; it is a third dimension of the same finding, and it is
the strongest argument in this project for reporting an operating curve rather than a point.

## 10. The corrected mapping (2026-09-23) — and why the first one made the project look worse

Adam asked why the literature-derived beta made our result worse. The answer is that it did not;
the mapping did, and the mapping was mine.

**The error.** SoundWatch's two regimes are *urgent* (users tolerate errors to get speed and recall)
and *non-urgent* (users want accuracy "to not be unnecessarily disturbed"). Its non-urgent examples
are speech and background noise. Section 8 mapped that non-urgent regime onto sounds the annotator
rated **importance 2**, and priced them at beta = 3.0. But importance 2 in this gold set is:

    Machine gun x5, Explosion x4, Bird x5, Vehicle x2, Hammer x2, Laughter x2, Footsteps x2,
    Door x2, Helicopter x2, Alarm x2, Chainsaw x2, Ice cream van x2, ...

Machine guns and explosions are not background noise. The literature's non-urgent tier corresponds
to **importance 1** -- Bird x3, Sheep, Vehicle, Chainsaw, the "noise of the place" -- which the
metric already treats as don't-care and never scores at all. Section 8 therefore priced consequential
events as if they were room tone, and that, not the literature, is what raised our cost.

**The corrected regime.** The distraction cost belongs to the PICTURE, not to the sound that was
missed (Beyond Subtitles, Caption Royale), so the regime keys on what the clip contains: a clip
holding any needed sound the annotator thought worth showing is one where misses dominate; a clip
holding none is one where any picture is pure disturbance. The literature fixes the DIRECTION of the
two betas but not their magnitude, so both are swept rather than chosen.

    TEST (60 clips: 28 with something to show, 32 with nothing)
    beta low/high     ours   blind   silence   ours cheaper than silence?
      0.75 / 2.5      2.22   2.71     2.87     YES
      0.75 / 3.0      2.26   2.86     2.87     YES
      0.75 / 4.0      2.35   3.16     2.87     YES
      1.00 / 2.5      2.29   2.85     2.87     YES
      1.00 / 3.0      2.33   3.00     2.87     YES
      1.00 / 4.0      2.42   3.30     2.87     YES
      1.50 / 2.5      2.43   3.13     2.87     YES
      1.50 / 3.0      2.48   3.28     2.87     YES
      1.50 / 4.0      2.56   3.58     2.87     YES

**Under every cell of the grid the system is cheaper to a viewer than showing nothing**, and the
result does not depend on picking a value inside it. The whole grid is reported for that reason.

**Significance, on the same TEST rows:**

    beta            vs silence                       vs blind
    0.75 / 3.0      +0.60  [-0.04, +1.32]            +0.60  [+0.08, +1.16]  significant
    1.00 / 3.0      +0.53  [-0.12, +1.25]            +0.67  [+0.12, +1.23]  significant
    1.50 / 3.0      +0.39  [-0.30, +1.12]            +0.81  [+0.25, +1.42]  significant
    1.00 / 2.5      +0.57  [-0.05, +1.30]            +0.56  [+0.07, +1.07]  significant
    1.00 / 4.0      +0.45  [-0.23, +1.17]            +0.88  [+0.23, +1.62]  significant

The advantage over the **blind baseline is significant in every cell**. The advantage over silence
is positive in every cell but its interval touches zero, so it is reported as a consistent direction
without a significance claim -- the same honest position as at the flat beta = 2.

**DEV disagrees, and that is reported too.** On DEV no cell of the grid puts us below silence
(2.79-3.49 against 2.69). DEV has been the harder half throughout -- our cost there is 3.10 against
TEST's 2.53 at the declared flat beta -- and nothing about the split changes that. The honest
summary is that the conclusion holds on the held-out half and not on the selection half, which is
the opposite of the direction that would worry a reviewer, but it is stated rather than smoothed.

**What Adam's instinct turns out to be worth.** For every sound this project actually scores, the
literature places beta BELOW the rubric's 2 -- because every scored sound has already passed the
annotator's filter of "a hearing viewer would notice and it matters". The instinct was right. The
sounds for which a wrong picture is dear are the ones the metric already ignores.
