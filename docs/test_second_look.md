# The second deliberate look at TEST — fixed before it runs

2026-09-24. Adam approved it ("1 yes") and decided to remove the eight-second cap ("remove the 8
seconds limit"). This page is committed before the TEST run is submitted; nothing below may change
after the TEST numbers are seen.

**Disclosure.** TEST was looked at once deliberately on 2026-09-23 (the planned single look of the
adopted cell, `test_final_v30`), and once accidentally before that (disclosed in `docs/prereg_v4.md`).
This is the **second deliberate look**. The change being confirmed was chosen on DEV, where the five
early cases were found; 3 of its 4 DEV recoveries are those cases. The DEV gain is therefore the
selection estimate and is optimistic; the TEST number is the one to report.

## What runs (frozen)

| | base (already rendered) | new |
|---|---|---|
| tag | `test_final_v30` | `test_monocap_v31` |
| clips | the 60 TEST clips (`data/input/gold_test`) | same |
| detector / gate | V4 = 590, FLEXSED_BAR 0.8, FLEXSED_VETO 0.3, PANNS_VETO 0.05 | same |
| onset rule | off | **`ONSET_MONOTONE` on** |
| picture cap | `MAX_SPAN` 8 s | **off** (Adam's decision) |
| pictures | as shipped | as shipped (`PICTURE_V3` off: this look tests timing only) |

The two changes are confirmed together; with one look they cannot be separated on TEST. On DEV the
onset rule alone was significant (F1 +0.101 [+0.022, +0.196]); with the cap also removed it was not
(it lost one bell to the visibility vote) — so this TEST run checks the version with the weaker DEV
evidence, by Adam's choice, and the report will say so.

## What is reported (same code and statistics as DEV)

Paired over clips against the base, 2000 bootstrap draws, seed 0: F1, precision, recall, viewer cost
(4 × missed + 2 × wrong), needed sounds recovered and lost (listed by name), early starts, and the
displayed picture end measured on the rendered panel.

## The decision rule

  * **Keep** (both changes become the default) if on TEST at most **one** needed sound is lost against
    the base **and** the F1 point estimate is not below the base.
  * **Revert the onset rule** if the F1 change is negative with its 95% interval excluding zero, **or**
    more than one needed sound is lost.
  * **Anything in between**: reported as inconclusive; the onset rule stays as a bug fix proven by the
    trace (before it, the refinement moved 223 of 442 starts earlier, the worst by 8.98 s; with it, 0),
    and the gain is reported as DEV-only.

## Amendment — written while the run was in progress, before any TEST number was seen

Adam: *"Time limit is not a thing we should use — it depends on the sound. We should not manually
limit timing. The evidence significance should not be dependent on the cut. If so, it's manipulation."*

So the eight-second cap is **removed whatever TEST shows**, and the option above to restore it if it
turned out to be the cap that lost sounds is **deleted**: choosing to keep or drop a cap according to
which one makes the result significant is exactly the forking path he objects to. The decision rule now
concerns the onset rule only. If removing the cap costs sounds, that is reported as the cost of a
principled change, and the fix is a principled end rule that follows each sound (not a fixed number of
seconds), tested on DEV as its own change.

## Result (read 2026-09-24 ~21:00, after the reproduction check passed)

`test_final_v30` (base) vs `test_monocap_v31` (onset rule on, cap off), 60 TEST clips, paired clip
bootstrap (2000, seed 0), `arm_compare.py --subset test --boot`:

    F1 0.400 -> 0.381   dF1 -0.019 [-0.127, +0.082]   dP -0.042 [-0.146, +0.061]   dR +0.000
    dFA/clip +0.067 [-0.017, +0.150]   dcost +0.133 [-0.267, +0.600]   recovered 2, lost 2
    median end error (specs) -0.13 s -> -0.05 s

Every change, attributed by trace (mechanical, but done after the numbers were seen):

| sound | change | cause |
|---|---|---|
| Alarm at 7.0 s (chainsaw roof) | lost: start 6.95 -> 8.25 | onset rule: the clamp blocked a correct earlier move (the union anchor was late) |
| Bark at 16.3 s (pet parrot) | gained: start 15.58 -> 16.00 | onset rule |
| Chainsaw at 10.3 s (chainsaw roof) | lost: two pictures [0.06, 8.06] + [10.3, …] became one [0.06, 16.75] | cap (8.06 = start + 8.00); the picture is on screen through the whole sound, the start-only metric misses the second onset |
| Bell, importance 3 (bell_kazansky) | gained: silenced by the visibility vote before, shown now | cap (mirror of the DEV bell loss) |

The +4 wrong pictures (five clips gain a later same-family picture) cannot be split between the two
changes and are the run's cost. Hardware: base on A100, new on L40S with CPU offload; the gate differs
on 1 of 60 clips (the bell).

**Verdict under the rule and its amendment (panel P2): INCONCLUSIVE.** Onset-attributed: 1 lost, 1
gained — not "more than one lost", and the F1 interval contains zero, so no revert; the F1 point is
below the base, so no keep. The onset rule stays as a bug fix proven by the trace; its gain is reported
as **DEV, selected**. The cap stays off; its account is one chainsaw lost and one church bell gained.
Limitation recorded: the clamp can block a correct earlier move when the anchor is late. TEST is not
read again for this.

**Displayed ends on the rendered TEST panel** (`timing_audit.py --render`, one-picture clips): median
displayed end vs the sound's end +0.00 s, mean +1.25 s; **4 of 15 linger more than 2 s** (ice-cream
truck +11.9 s, thriller basement +7.8 s, pet parrot +5.5 s — measured against the first sound, a bark
follows —, live fire +3.2 s). DEV had 1 of 16. The base run's panels could not be measured (the audit
found no one-picture composites under its render folder), so there is no base panel figure; the base's
spec-level median end was -0.13 s. This is the cost side of removing the cap: with no cap, a picture ends
when the detector's span ends, and on these clips the detector's span outlasts the annotated sound. The
principled response is an end rule that follows the detector's own score (`AED_RELEASE`), tested on DEV
as its own change — not a return of a fixed number of seconds, and not tuned on TEST.
