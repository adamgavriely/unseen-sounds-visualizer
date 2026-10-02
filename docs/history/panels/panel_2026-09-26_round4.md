# Panel 2026-09-26 — round 4 (final): 5 of 5 SIGN

Plan: docs/history/panels/panel_2026-09-26_plan.md (v2). Non-blocking notes, adopted into the plan's execution:

- **P1** SIGN. Both [merge] choices checked against `score_per_sound.py` l.174–176 and `reason.py` l.396–412.
- **P2** SIGN. `paired_ci` is unpacked as a 4-tuple by `main()` (l.436): return the draws through a new helper,
  not a changed tuple; the Holm script asserts its recomputed d, lo, hi equal the printed row before any p is used.
- **P3** SIGN. The derived text arm would show every *planned* span (`require_image=False`, no `_assign_rows`)
  while pictures show only *placed* spans: restrict the derived tags to ours' placed spans so modality is the only
  difference, or label the row "same gate, planned spans".
- **P4** SIGN. Same edge as P3: print per clip that the derived arm's span list equals ours' placed spans; report
  any difference as a caveat.
- **P5** SIGN (D2 dissent stands). B.7's sensitivity row re-scores 15 TEST clips with the second person's ticks:
  list it in the read-history table as a gold-robustness re-score.
