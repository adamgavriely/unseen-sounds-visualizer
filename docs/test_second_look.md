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
  * **Revert** if the F1 change is negative with its 95% interval excluding zero, **or** more than one
    needed sound is lost. If it is the cap that loses them (checked by the per-sound list), the onset
    rule alone may be kept, and the cap restored — the rule is stated here so it is not chosen later.
  * **Anything in between**: reported as inconclusive; the onset rule stays as a bug fix proven by the
    trace (before it, the refinement moved 223 of 442 starts earlier, the worst by 8.98 s; with it, 0),
    and the gain is reported as DEV-only.
