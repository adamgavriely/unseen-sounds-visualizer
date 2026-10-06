# Pre-registration: Step 3c, start a picture at the earliest heard evidence just before it (DEV only, CPU)

Written and committed 2026-10-06 before any cell was scored. Adopt nothing; TEST not touched.

Note: this goes against the round-13 "monotone onset" rule (a later stage may never move a start earlier than its
anchor). It is tested because the hit rule forgives up to 0.5 s early and most remaining coverage loss is late starts.

## Rule (general, online per video)

For each picture on screen (label L, start a): walk back from a in 0.02 s steps, at most W s. The new start is the
earliest moment t such that the family of L has evidence at every step from t to a (no gap). Never earlier than the end
of the previous picture of the same label, nor 0. Ends and everything else unchanged. Family evidence as in Step 1
(`evidence_dev_all.json`: max over the ear's classes in L's family, nearest frame):

| evidence E | rule |
|---|---|
| F0.5 | FlexSED >= 0.5 |
| F0.3 | FlexSED >= 0.3 |
| FB | FlexSED >= 0.5 or BEATs >= 0.175 |
| ANY | FlexSED >= 0.3 or BEATs >= 0.10 or DASM >= 0.35 |

W in {0.5, 1.0, 1.5}: 12 cells, each applied to two bases: D' (frozen) and AB-m (Step 2 DEV candidate),
pictures from `step2_pics_dev.json`.

## Pass bar and selection (per base)

hits >= base hits, wrong <= base wrong, onset cost < base onset cost. Among passing cells: lowest onset cost; ties
within 0.01 -> lower cost_cov. Also reported: cost_cov, hit cover, clips whose hits / wrong change.
