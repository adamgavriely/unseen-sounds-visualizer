# Panel, 2026-09-25 — per-picture judging (Adam's proposal), round 1

Adam: *"maybe the AI judge should judge unneeded pictures too: on every picture it sees it should give a
score on how much information it adds to the image, if any, or if unnecessary deduct points."*

## The judge as it is now (`benchmark/gold/judge_direct.py`, committed, results already seen)

One score per clip, 0-4. Gemma-4-31B sees the pictures (at most 4) with the seconds each was on screen,
and a reference built only from the annotator's ticks: the needed sounds with their times, or "nothing
beyond the picture". Prompt: 4 = every needed sound shown recognisably, near the right time, nothing false
or extra; 2 = a needed sound missing/unrecognisable, **or an extra picture shows something the viewer did
not need**; if nothing was needed, 4 when nothing shown, at most 2 when something was (also capped in
code). Captions are shown as their text tags. The judge never sees the video.

Trust checks passed on 139 clips (B1 rho -0.72; B2 +1.11; B3 59/60 identical). Results already read:
current DEV (49 clips) ours 2.29, vs blind +0.20 [-0.18, +0.61], vs caption +0.06 [-0.31, +0.39];
older v4b4 139 clips ours vs blind +0.27 [+0.07, +0.50] *, vs caption -0.04. The main (annotator-based)
metric shows the gate significantly better than blind on precision and viewer cost, F1 a tie.

## The proposal, as I read it

Score **each picture**, not only the clip: for every picture shown, how much it adds to what the viewer
already has — positive if it gives needed information, zero if it adds nothing, **negative if it is
unnecessary** (the source is on screen, the sound is obvious, or it is wrong). The clip score is then
built from the per-picture scores (plus a penalty for needed sounds with no picture). This is close to
the headline metric's own logic (viewer cost = 4 x missed + 2 x wrong), but with the judge's eyes deciding
what each picture conveys.

Two ways to decide "unnecessary": (a) from the annotator's ticks (each gold sound is marked needed /
visible / obvious; a picture matching a non-needed sound is unnecessary) — the judge only says what the
picture shows and whether it matches a reference sound; (b) the judge also sees a video frame at the
picture's time and decides itself whether the picture adds anything to the image.

## Questions (at most eight lines each)

1. Is per-picture judging better than the current per-clip score? What does it measure that the current
   judge and the headline metric do not?
2. (a) annotator-ticks or (b) the judge sees a frame and decides? Which is honest and non-circular?
3. **Forking path:** the current judge's result (a tie) is already read. Changing the judge now, when the
   new one might favour us, looks like tuning the instrument. How must it be declared (primary /
   secondary, decided before any number, trust checks) so a supervisor accepts it?
4. The exact scoring rule (per picture scale, missed-sound penalty, how a clip score is formed) and the
   trust checks it must pass before any ranking.
5. Captions: how are text tags scored under the same rule, fairly?

---

# Round 2 — merged proposal (sign or amend)

**Agreed by all three in round 1**
* "Unnecessary" comes from the **annotator's ticks** (needed = not visible and not obvious), never from
  the judge looking at a video frame — that would redo the gate's own job with the same kind of model.
* The judge answers only **what each picture shows**; which gold sound it matches, whether it was needed,
  and the score are **mechanical, in code**.
* It is a **third, exploratory measure**, added after the per-clip judge's tie was read, at Adam's request.
  The per-clip judge stays the registered secondary and its tie is reported unchanged; both appear side by
  side on every row; the new rule is frozen in a committed page before any number; DEV and the superseded
  v4b4 row only — TEST only in the final reporting run; counted as the fourth judge variant.
* It adds exactly one new quantity: **recognisability** of each picture. Everything else is the viewer cost.
  Deducting unnecessary pictures favours the gate by construction — this is stated on the page.

**Proposed rule (PP-1)**
1. Judge (Gemma-4-31B, the same weights), per picture, **every picture** (no cap of 4): "What is making a
   sound in this picture? Answer in a few words, or: can't tell." — the same question Adam answers.
2. Code: the answer is classified against each gold sound of the clip whose onset lies within
   [−0.5, +1.0] s of the picture's start, with the answer-sheet logic (`answer_sheet.sheet_for` on the gold
   label): correct (incl. narrower) / vague / wrong / can't tell.
3. Per clip, same units as the viewer cost (amendment 9 weights, not re-chosen):
   **cost = 4 × uncovered needed sounds + 2 × unnecessary pictures**, lower is better.
   * a needed sound (importance ≥ 2) is *covered* by a **correct** picture of it in the window;
   * a **vague** picture of a needed sound covers it at half: 2 instead of 4 for that sound, no extra charge
     (so a vague picture is never worse than no picture);
   * every other picture — matches a visible/obvious sound, matches nothing, wrong, can't tell — costs 2;
     a second correct picture of an already-covered sound costs 0.
4. Captions: the tag is rendered as the text card the viewer sees and passes through the same judge and
   the same rule; the write-up says text is legible by construction, and reports recognisability split out.
5. Checks, bars fixed now, before any ranking: B1 rho ≤ −0.4 vs the annotator's viewer cost (expected to
   pass, reported as partly built in); B2 clean vs known-wrong clips, gap > 0, CI excludes 0; B3 repeat on
   20 clips, ≤ 1 change; **B4** the judge's answers vs Adam's blind round-2 answers, scored with the same
   sheet: kappa ≥ 0.5, catches ≥ 70 % of his not-right, rejects ≤ 15 % of his right (the checker's frozen
   bars). B4 can only run after Adam's ratings exist; until then no per-picture ranking is read.
6. Reported: mean cost per clip per arm, paired clip bootstrap (2000, seed 0), plus a per-arm table of the
   judge's calls (correct / vague / wrong / can't tell / unnecessary) so leniency toward one arm is visible.

## Questions (at most six lines each)
1. Sign PP-1 or amend each point (especially the vague rule in 3 and captions in 4).
2. Anything that must be in the declaration page that is not above?
