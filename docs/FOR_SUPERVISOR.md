> **Superseded (stamped 2026-09-27).** This document describes an earlier system and earlier numbers. Current results: `docs/prereg_v4.md` amendment 21 (final TEST table), `docs/WEEK_PLAN_2026-09-26.md`, `docs/LEDGER_2026-09-26.md`, supervisor page `docs/supervisor_2026-09-26/index.html`. The text below is kept unchanged as a record.

# For the supervisor — where the project stands, and what needs a decision

Adam Gavriely, 2026-09-23. One page of results, then every question that needs you, then every
place a reviewer can push.

---

## 1. The headline

The pipeline shows a generated picture beside a video for ambient sounds whose source is off screen.
The final evaluation was taken **once**, on 60 clips never used for any choice, under rules written
down beforehand (`docs/prereg_v4.md`).

    TEST, 60 clips        precision   recall   wrong pictures/clip   cost to the viewer
    before this week        0.231      0.419        1.00                  3.67
    final system            0.421      0.372        0.37                  2.53
    blind baseline          0.278      0.465        0.87                  3.27
    show nothing               -          -         0.00                  2.87

**On the clips the project is about — where the sound's source is off screen — the system is
significantly cheaper for a viewer than showing nothing** (+3.08, 95% CI [+1.08, +5.38]), and its
pictures there are right **85%** of the time.

**Against the blind baseline** (draw every detected sound, ignore the video) the gate's precision
advantage is **significant for the first time**: +0.143, 95% CI [+0.034, +0.264].

Across all 60 clips the system is cheaper than silence on the point estimate (2.53 vs 2.87) but the
interval crosses zero, so **no claim is made there**.

## 2. What produced the improvement

Two "vetoes", both free of any new training or annotation:

  * where BEATs names a sound family that FlexSED never hears anywhere in the clip, drop it -- the
    sound is usually not there (Whale 0.13, Horse 0.13, Cat 0.00);
  * where FlexSED raises a family alone and PANNs does not support it, drop it -- PANNs scores
    median 0.42 where FlexSED is right and 0.026 where it is wrong.

Both are one-sided by construction, so a span both detectors raise is never touched.

## 3. Questions that need your decision

**(a) The second annotator, for inter-annotator agreement.** The gold set has one annotator. Every
reviewer consulted put this first. The design is written so it can be judged before anyone starts:

  * **25-30 clips**, drawn stratified across the four categories (unseen / mixed / seen /
    no-ambient) and both sourcing populations, so agreement is not measured only on easy clips;
  * the annotator labels **the same fields**: sound family, start, end, importance 1-3, and the two
    ticks `visible` and `obvious` that define whether a sound is "needed";
  * the number that matters is **Cohen's kappa on the `needed` decision**, because that is what the
    metric is built on -- not on the label, which is easier;
  * **declared before collection**: kappa >= 0.6 is reported as substantial agreement; below 0.4
    would mean the definition itself is unreliable and must be reported as a limitation of every
    number above.

  **What we need from you:** who, and whether the department has anyone appropriate. Adam has said
  this is not for right now; the design is here so it is ready when it is.

**(b) Beta -- the price of a wrong picture -- is left open on purpose.** The whole result depends on
how a deaf viewer weighs "a sound I never knew about" against "a picture of something that was not
there". The metric fixes a missed sound at 4 and calls the wrong-picture cost beta.

  * Derived limits: below **0.35** a visibility gate costs more than it saves; above **4.30** even a
    perfect detector feeding this gate loses to showing nothing. So beta is only a question on
    [0.35, 4.30].
  * The system beats silence below **beta = 1.37** (95% CI [0.71, 2.19]) on all clips.
  * The project's own September rubric asserted **2.0**, which lies inside that interval -- so on
    this evidence neither "we win at beta = 2" nor "we lose at beta = 2" is established.
  * The published DHH literature does not give one number. It gives a low beta for danger sounds
    (users tolerate false alarms to catch the siren) and a high one for ambience (users do not want
    to be disturbed). Pooled over our gold set it lands near the rubric's 2.

  **What we need from you:** whether reporting an operating CURVE with these limits is acceptable
  for the thesis, or whether you want a single declared operating point defended. A viewer study to
  measure beta was designed and **not run** -- there is no DHH participant community available here.
  The design is in `docs/beta_specification.md` if you want it attempted.

**(c) ComfyUI as a demo.** You suggested it as a GUI for the pipeline; that was the right reading and
an earlier assessment of it as an image generator answered the wrong question. It is built: the seven
stages load as nodes, it accepts any video, and the gate is a switch a committee can flip. The
server was killed by a memory limit on its first real run and the fix (more RAM) has not been
re-tested. Status in `docs/ComfyUI.md`.

  **What we need from you:** is this a defence artefact (~1 more day, no risk) or should it become
  the pipeline (~1 week, and it would have to reproduce the final numbers exactly before anything
  could be trusted)? Adam's and our recommendation is the former.

## 4. Everything a reviewer could push on, disclosed

  * **An accidental look at TEST.** A script printed the TEST rows for the gate experiment before
    the planned single look. No decision was taken on them and the operating point had already been
    chosen on DEV, but the single-look guarantee is broken for that one comparison and is described
    as "confirmatory" rather than clean. Full disclosure in `docs/prereg_v4.md`.
  * **TEST is composed slightly in our favour.** It carries 20 no-ambient clips to DEV's 9, and
    everything adopted is a detector change whose job is to stop the detector inventing sounds.
    This was written down *before* the TEST numbers were read, and results are reported per category
    so the effect is visible rather than taken on trust.
  * **One annotator.** See 3(a).
  * **Equal-weight F1 never becomes significant** (+0.047). This is structural and was established
    in September: F1 prices a picture of a sound that is not there exactly like a sound left
    undrawn, so it cannot see the trade the gate makes. The viewer-cost metric exists for that
    reason and its weights come from this project's own earlier judging rubric, not from the result.
  * **The gate's advantage over blind shrank** as the detector improved (1.03 -> 0.73 on cost),
    because the vetoes are a shared stage and the baseline gets them too. Reported against interest:
    the gate is worth most when the detector is worst.
  * **Five ideas were tested and rejected** by rules fixed before each run: speech removal as a
    detector view, speech removal for timing, family-level gating, the object detector as a second
    silencing vote, and an ensemble of four detectors. Each is recorded with its numbers.

## 5. What we now know about the video side, which is a result in itself

Three different mechanisms were tried to make the gate use the video better: a time-aligned OWLv2
vote, the same with SAM 3 (which scores more than double OWLv2 on open-vocabulary detection), and
a looser voting rule. **All three landed on exactly the same number: 2 wrong pictures removed per 1
real sound lost** -- precisely the break-even at the declared operating point.

That is the finding. A better vision model does not move the line, because the question the gate
needs answered is not the question these models answer. SAM 3 can say *"a crowd is on screen"*
correctly; the gate needs *"is the crowd we can hear the one on screen, or one around the corner."*
Presence is not source, and no current model is asked for source.

## 6. What is still open technically

  * **True audio-visual synchrony has not been tested** -- whether the image *changes at the instant
    the sound starts*, measured inside the detected object's region so camera motion does not
    confound it. It is the one video idea with an untried mechanism.
  * SAM 3 could replace OWLv2 as the stage-2 default. It changes no measured number, because that
    verdict is deliberately deferred to the VLM in the shipping configuration, so it is a
    demo-quality change to make after the write-up, not before.
