# Supervisor update — Visual Augmentation of Audio Semantics for Accessibility

*Adam Gavriely, 16 September 2026. Detail: `docs/EXECUTIVE_SUMMARY.pdf` (plain language) and
the thesis draft `docs/report/report.pdf` (19 pages).*

This is my first update and it comes late — I am sorry. The short version: the system from
the proposal is built, and I have evaluated it on a benchmark I made. It watches a video,
listens for ambient sounds, and shows a small picture beside the video only when the sound's
source is not visible on screen (the "gate"). The result is a **tie**: my system scores the
same as a much simpler one that shows a picture for every sound without looking at the
video — 3.17 vs 3.16 out of 4, where 4 means the panel conveys what a hearing viewer would
learn and 0 means it conveys nothing or the wrong thing. The reason is mostly the sound
detector, which misses sounds buried under speech or music, not the visibility decision. I
tried six ways around the detector and one around the visibility check, each with a
pass/fail target written down before the run; none passed, so the limit itself is the
finding. The thesis is written around that. I intend to submit on 18 September unless you
object, and I need four decisions from you.

**One-sentence claim:** a training-free chain of existing models can decide "heard but not
seen" well enough to remove 30% of unnecessary pictures, but with today's sound detectors it
does not beat "draw everything", and the reason is measured.

## Decisions I need by 17 September (my default if I hear nothing)

| # | Question | Options | My recommendation / default |
|---|---|---|---|
| 1 | Is this framing acceptable as the contribution: working pipeline + benchmark and protocol + the measured limits? If you prefer another framing I can rewrite the introduction and conclusion in a day; the numbers would not change. | findings framing / "system" framing | findings framing |
| 2 | Departures from the proposal (list below) — any objection? | accept / discuss | accept |
| 3 | One annotator is the weakest point. Can 2–3 lab members each spend **one hour** labelling 60 test clips ("is the sound's source visible?") on a web page I prepare? It would move submission to 19 Sept. | yes, names + ethics needed? / no | no if it risks the date |
| 4 | Helpfulness to deaf viewers is **untested** (the proposal specified automatic evaluation). State it plainly and leave a user study to future work? | yes / add a small check with hearing people (cannot happen before the 18th) | state it plainly |
| 5 | Report format: required structure, length, Hebrew abstract, submission channel? | — | article style, 19 pp |
| 6 | AI disclosure: I used an AI coding assistant for code and drafting; every design choice and every label is mine and logged in git. What wording does the department want? | — | one acknowledgement paragraph |

## Departures from the proposal

- Sound detector PANNs → BEATs; image generator SDXL → FLUX.1 (both measured improvements).
- 274 clips, not 300 (usable yield 15–20% of candidates); a development/test split added so
  settings are chosen on clips the score never sees.
- The proposal's evaluation reference was written by the system's own models; I found it
  circular (it shifts the comparison by about half a point) and replaced it with one
  corrected by my labels. Both scores are reported.
- Everything else as proposed: no training, minimum hand-written rules, automatic scoring.

## What exists

| | |
|---|---|
| System | 7 stages: frames → sound detector → visibility check by a vision-language model → picture → side panel. `python main.py <video>`. |
| Benchmark | 274 labelled clips, four scenarios (unseen / mixed / seen / no ambient); 100 test clips scored once for the headline. Re-labelling 60 clips blind, I agreed with my earlier labels 78% of the time; nobody else has labelled. |
| Outside check | DCASE 2025 (human on/off-screen labels): our visibility decision agrees 66.7% ± 6; it keeps 97% of off-screen sounds and silences only 35% of on-screen ones — it errs toward showing, which costs a redundant picture, not a missing one. |
| Documents | Thesis draft (19 pp), executive summary (14 pp), LIMITATIONS.md, failure catalogue, three demo videos; headline numbers reproducible from the repository without a GPU. |

## The findings

1. **Tie, and where each side loses.** 3.17 vs 3.16 (95% CI on the difference −0.15 to
   +0.18). A perfect gate would score 3.62 (not 4: the pictures themselves are imperfect);
   both systems sit ~0.45 below it, in opposite places — mine where a picture is due
   (withheld pictures), the baseline where none is due (redundant pictures). My system
   beats the text-caption baseline (+0.28 [+0.10, +0.46]), though a second judge moves that
   baseline by +0.61, so text rankings depend on the judge.
2. **The detector is the limit, measured.** Of 11 wrongly withheld test pictures, 7 are
   sounds that never reached the detector's threshold under speech or music; the other 4 are
   label ambiguity. About 80% of unnecessary pictures on the development clips are sounds
   the detector reports that are not there. Threshold and decision-rule changes trade the
   two along one curve; second-opinion models removed at most a third of the false sounds;
   separating the audio first made detection worse; a 4× larger vision model wrongly
   silenced off-screen sounds. All bars declared first; all reported.
3. **Automatic evaluation has a wall.** Three automatic references (independent models; a
   with-sound vs muted description; a "gap-closing" score) all failed at the same point:
   the two open audio-visual models tried (Idefics3, Qwen2.5-Omni, 7–8B) do not decide
   whether a sound's source is in view better than chance. The human-corrected reference is
   therefore the headline; the failures are documented with their pre-declared bars.
4. **Cost next to benefit.** The gate keeps the panel on 47% of a clip vs 60% for the
   baseline and removes 30% of panel time spent where nothing was due. A judge-free view
   prefers the gate once an unnecessary picture counts ≥ 40% of a missing one (exploratory;
   the crossover is poorly determined, CI 0.08–1.00).

## Next steps

17 Sept 18:00: all numbers frozen and checked across the three documents. 18 Sept: proofread
and submit. If you can read one thing before we talk: the executive summary, sections 1–3
and 9 (about ten minutes).
