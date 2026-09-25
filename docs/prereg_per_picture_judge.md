# Per-picture judge (PP-1) — declared before any number exists

2026-09-25. Adam: *"the AI judge should judge unneeded pictures too: on every picture it sees it should
give a score on how much information it adds, or if unnecessary deduct points."* Panel P2, P4, P5, two
rounds (`docs/panel_2026-09-25_judge_per_picture.md`). This page is committed before the measure is run.

## Order of events (disclosed)

Written **after** the per-clip direct judge's result was read (current DEV: ours vs blind +0.20
[-0.18, +0.61], vs caption +0.06 [-0.31, +0.39]; superseded 139-clip v4b4 row: vs blind +0.27 [+0.07,
+0.50]), and after V3's picture subjects were seen. It is therefore a **third, exploratory** measure. The
headline metric stays primary; the per-clip direct judge stays the registered secondary and its tie is
reported unchanged. Judge variants to date, all reported on every row, none dropped: Mistral
describe-then-judge, Gemma describe-then-judge, Gemma direct per clip, and this one.

**Decision-free:** no arm, configuration, prompt or picture rule may ever be chosen by this measure.

## What the judge sees and answers

Gemma-4-31B-it (the same weights as the direct judge), greedy decoding, **every picture shown** (no cap).
It sees the picture **alone** — no video, no frame, no reference, no sound name: the same blindness Adam
has when he rates. Prompt, verbatim:

> What is making a sound in this picture? Answer in a few words, or answer exactly: can't tell.

Captions: the primary reading is **mechanical** — the tag text itself is the answer (a judge misreading a
printed word would be judge noise charged to the caption arm). The same tag rendered as the text card the
viewer sees, read by the judge, is a check expected to agree near 100 %.

## Scoring — mechanical, copied from `score_per_sound.py`

1. Each answer is classified against every gold sound of the clip whose onset lies within [-0.5, +1.0] s
   of the picture's start, with the answer-sheet logic (`answer_sheet.sheet_for` on the gold label):
   correct (incl. narrower) / vague / wrong / can't tell / unclassified. When several gold sounds sit in
   the window, the picture takes its best class and is charged once; one picture may cover several
   same-family sounds. Answers the sheet cannot classify are counted as "unclassified" and reported, never
   silently decided. The sheet is extended to every DEV gold label **before** any judge answer is read.
2. Per clip, amendment-9 weights, not re-chosen: **cost = 4 x uncovered needed sounds + 2 x unnecessary
   pictures**, lower is better.
   * A needed sound (not visible, not obvious, importance >= 2) is **covered** by a **correct** picture of
     it in the window.
   * **Vague** (primary): the sound is *not* covered and the picture is *not* charged — cost as if no
     picture (4), matching the picture round, where vague is not counted as correct. Sensitivity rows,
     declared now: vague covers at half (2), and vague covers fully (0).
   * A **wrong** picture in a needed sound's window costs 2 **and** leaves the sound uncovered: 6 in total.
     **A false picture costs a deaf viewer more than no picture** (Jain et al., CHI 2019).
   * Every other picture costs 2: it matches a visible or obvious sound, matches nothing, is wrong, or is
     "can't tell". "Wrong" and "can't tell" are kept as separate columns.
   * Importance-1 needed sounds are don't-care: a picture of one costs 0 and covers nothing. A second correct
     (or vague) picture of an already-covered sound is a duplicate and costs 0. Gold rows the label filter
     blocks are neither covered nor missed.
   * A correct picture outside the window is logged as "correct, late"; its cost is unchanged.
3. Reported per arm: mean cost per clip, paired clip bootstrap (2000 draws, seed 0), per scenario category,
   next to **silence** (4 x needed, the floor) and the oracle; plus a table of the judge's calls per arm
   (correct / vague / wrong / can't tell / unclassified / unnecessary) and the picture count per arm.

## Trust checks — bars fixed now, before any ranking

* **B1** Spearman rho <= -0.4 against the annotator's viewer cost (expected to pass; partly built in).
* **B2** clips carrying a known-wrong picture cost more, CI excludes 0.
* **B3** 20 clips judged again with sampling on: at most 1 answer class changes.
* **B4** the judge's answers on the round-2 picture bench (never-annotated clips, not the gold renders it
  scores), classified with the committed sheet, against Adam's blind answers, the arm hidden: kappa >= 0.5,
  catches >= 70 % of his not-right, rejects <= 15 % of his right (the checker's frozen bars). Reported
  per generator. B4 runs only after Adam's ratings exist.
* **Failure:** if B2 or B4 fails, only the call table is published as a logged column. No ranking is
  read, and no prompt is redesigned on the same ratings.

## Where it runs

DEV 49 (`dev_monocap_v31`, all three arms at the new timing) and the superseded 139-clip `v4b4` row
(contains TEST clips already exposed on 22 Sept; instrument calibration only). TEST pictures are not judged
until the final reporting run. The 50 round-2 clips are Adam's calibration set and are never scored by this
judge as a result.

## Built-in asymmetries, stated before the run

* Charging unnecessary pictures favours the gate by construction.
* Text is legible by construction, which favours captions on recognisability.
* Vague half-credit (a sensitivity row) favours family-level pictures and captions.
* B4 calibrates to one rater.

**Written predictions** (so the result can be wrong on record): the gate beats blind on unnecessary
pictures; captions beat pictures on recognisability; V3's pictures are vague less often than today's.

## Amendment (Adam, 2026-09-25, before any per-picture number) — count the bad results, show a graph

The primary display of PP-1 is **counts, not a weighted sum**: per arm, a stacked bar of the bad results
over the clips — missed needed sounds, wrong pictures, pictures of a sound the annotator ticked *visible*
(source on screen), pictures of a sound ticked *obvious*, vague pictures, "can't tell", duplicates — beside
the count of good pictures (needed sounds covered). Counts need no weights; the weighted cost stays as the
secondary line. "How obvious" comes from the annotator's ticks (visible / obvious / importance), not from
the judge; a judge-rated obviousness grade may be logged as a column, never scored (the judge deciding
what was needed would redo the gate's job — panel round 1).
