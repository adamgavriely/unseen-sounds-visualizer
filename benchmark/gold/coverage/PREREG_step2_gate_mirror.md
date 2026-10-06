# Pre-registration: Step 2, "a/b decides" gate rule and a time-local mirror veto (DEV only)

Written and committed 2026-10-06, before any cell below was built or scored. Step 1 (hold) is parked; its code
stays in this folder.

## Why

- Gate: two of the three per-stretch questions (name, describe) test whether the object is in frame; only the a/b
  question tests whether it is making the sound. Adam marked the 5 DEV sounds the gate silences for this reason
  (rainforest bird, pet-shop bird, Miami bell, two farts) as real gate errors (5 Oct).
- Mirror veto (R13-2, shipped at b 0.7 / own < 0.4): it takes FlexSED's max of every query over the WHOLE span, so
  another family loud at one moment vetoes a span whose own sound happens at another moment.

## Rules (general, online per video)

**AB (gate)**: a stretch is "seen" iff the a/b answer is yes (both orders agree the thing makes the sound). A sound
is silenced iff every stretch is seen (unchanged). Split a/b answer ("cannot tell"):
- AB-s: split = not seen (show it; the project's rule that a missed sound costs more)
- AB-m: split = the current majority of the three votes

A restored sound is drawn exactly as the system without the gate draws it (its blind_a2i spec); a "kind of X" sound
that the family rule silenced follows its restored parent (as the gate replay of 5 Oct). Votes come from the arm's
own trail ("gate" records). No new model call.

**TL (mirror veto, time-local)**: for a BEATs-only span, a FlexSED frame is "mirrored" if, AT THAT FRAME, the top
query belongs to another family and scores >= 0.7, and the span's own family scores < 0.4. The span is dropped iff
at least half of the judged frames are mirrored:
- TL-span: judged frames = the span (start .. end)
- TL-onset: judged frames = the onset part (start .. start + 1.0 s), the part the hit rule looks at
Unchanged around it: the listener keep (F7, LISTENER_CONFIRMED_MIRROR), every later stage, the gate, the display.
TL arms run through the scoring harness (stage 4 + stage 5, DEV and dev2 parts); gate answers are reused from the
stored runs and memo, new stretches are asked live as in every earlier round.

## Cells (9)

| cell | mirror | gate |
|---|---|---|
| D' (frozen) | shipped | majority |
| AB-s, AB-m | shipped | AB |
| TL-span, TL-onset | TL | majority |
| TL-span+AB-s, TL-span+AB-m, TL-onset+AB-s, TL-onset+AB-m | TL | AB |

## Pass bar (merged DEV, 71 clips)

hits >= 31 and wrong <= 16 (frozen: 29 / 15). Reported for every cell: onset cost, cost_cov (scorer v2), hit cover,
wrong seconds, stale seconds, and the clips whose hits / wrong change.

## Selection

Among cells passing the bar: lowest onset cost; ties within 0.01 -> lower cost_cov. Nothing is adopted without
Adam. TEST is not scored for any cell.

## Files

`step2_arms.py` (cluster: registers the TL arms and the time-local veto, runs the harness), `dump_gate.py`
(gate votes + augmentations per arm), `step2_score.py` (local: AB re-decision, scorer v2, table) ->
`step2_dev.json`, `step2_dev.md`.
