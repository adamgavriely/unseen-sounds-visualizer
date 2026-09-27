# FREEZE — the final picture setup (GP-4 step 5), committed before any confirmation picture is drawn

Code at commit 3995769. Selected on picture-DEV by the AI assistant (Claude), blind to arm ("selected, assistant-screened" — a model-based
selection, not a calibrated instrument; assistant vs Adam kappa 0.46 on round 1;
`docs/panel_2026-09-25_scene_prompt.md`, assistant screening result).

**Final setup ("final")**
* Subject text: V3.1 — `PICTURE_V3` + `PICTURE_SCENE` + `PICTURE_SCENE_GUARD2` (the specific source; the scene may add
  a kind qualifier; list guards; no invented person; the drawn phrase must name the source's chain).
* Image prompt: subject + the rules tail ("the whole thing fully in frame, caught at the moment it makes the sound
  with its visible effect, one large subject filling the picture, plain white background").
* Group (b) templates (thunder, rain on a surface, car alarm, train horn, church bell, glass shatter/smash) and their
  negative words (`benchmark/gold/gen_screen.py` TEMPLATES, TEMPLATE_NEG); group (c) burst cards (WHOOSH, THUD,
  BANG, SMACK, WHACK), drawn once per word.
* Generator: Qwen-Image-2512, 1024 px, 50 steps, true CFG 4.0, negative prompt = negative_for(subject) + template
  negatives; blank guard redraws once. **Seed: seed_of(item) (offset 0), one seed.**

**Control ("today")**: the shipped subjects written by the frozen render `confirm_v32`, FLUX.1-schnell, seed_of(item),
guard off (picture bench arm A0).

**Confirmation** (GP-4 step 6, one sitting, Adam only): the 81 frozen sounds on the 50 frozen clips
(`benchmark/gold/pictures/confirm_set_frozen.json`), final vs today, glance fidelity (384 px, 1.5 s), 8 repeats, burst
cards excluded from the sitting. Order: draw and seal → mechanical rule-2 pass (checker OBJECT line vs subject) sealed →
Adam rates → by-eye rule-2 pass with his answers unopened → open and score. The answer sheet for the 81 sounds is
committed before his file is opened. Primary: correct incl. narrower, paired over sounds, 2000 draws seed 0, CI
excluding zero; zero false messages in the final arm; repeats ≥ 80 %; ties keep today. Claim: "the pictures show the
detector's source recognisably" (no gold on these clips).

Known risks written now: a tie between two emergency-vehicle kinds can still be drawn with one kind's livery;
the car-alarm template reads as "a car"; birds tend to be head close-ups.

**Control fairness check (after drawing, before any rating; advisor).** The shipped pipeline redraws a picture
once when it is uniform (`stage6._is_blank`, pixel std < 3); arm A0 was drawn with that guard off. On the 81
control pictures, 11 have little ink but only 1 (#12, "Rain hits") is blank by the shipped test; it was redrawn
once at seed + 1 (`scripts/redraw_control_blank.py`) and is still blank, which is what the shipped pipeline would
show. So the control is faithful to today's system; the page is unchanged. FINAL has no blank picture.

**Scoring commands, fixed before any answer exists:**

    python benchmark/gold/score_answers.py score --answers <Adam's answers/confirm export> \
        --key benchmark/gold/pictures/rate_confirm_SOUND_KEY.json \
        --sheet benchmark/gold/pictures/answer_sheet_confirm_v32.json [--overrides <decided with the arm key closed>]
    (by-eye rule-2 pass on FINAL, answers still unopened; the sealed checker file is opened now, sha256 checked)
    python benchmark/gold/score_answers.py unseal --scored <scored> \
        --arms data/work/rate_confirm_ARM_KEY_do_not_open_until_scored.json --pairs FINAL:A0 --look-arm FINAL

## Amendment (panel 3, topic 2) — rules for the sealed sitting, written before any FINAL picture or answer is seen

Signed by the three picture reviewers (`docs/panel3_topic2_rounds.md`, round 3). [NAME] / [NAMES] are filled in by Adam
before he sits; nothing else changes after that.

```
Amendment, 2026-09-27 (panel 3, topic 2), written before any FINAL picture or answer is seen. Code unchanged (3995769).
1. Rule 2. A FINAL picture is a false message if it shows as the maker an object that is not the source, its chain, the
   RESOLVE qualifier, or a declared word, or if it shows a person for a sound outside Human sounds. Declared words —
   general: pieces, splash, flash, smoke, sparks, dust, drops; per template: Thunder/Thunderstorm: lightning bolt, storm
   cloud; Rain/Rain on surface: drops, window pane; Car alarm: parked car, headlights, indicator lights; Train horn:
   locomotive; Church bell: bell, tower; Shatter/Smash: glass window, pieces. This list is closed now; nothing is added
   after a picture is seen.
2. The pass: by eye, by [NAME], on all 80 FINAL pictures in the sitting (1 of 81 is a burst card), arm known, Adam's
   file unopened; the GLM OBJECT line and an OWLv2 pass over each sound's forbidden nouns are flag lists only (a
   non-flag clears nothing); a per-picture yes/no with the object named is committed before Adam's file is opened.
3. Report-only rows beside the primary, never replacing it: clip-cluster bootstrap over the 50 clips; the split
   source != family (34) / source == family (47); wrong count per arm; template pictures separately.
4. Glance: each card is shown at 384 px for 1.5 s, then hidden and never re-shown. The page shows a card for 1.5 s after
   "Show" (as built 2026-09-25, docs/SUMMARY_2026-09-25_while_you_slept.md); its HTML sits with the sealed files and
   is committed after scoring; the report states whether this held for every card.
5. Disclosed: the confirm sheet counts "goat" correct for the 2 Bleat sources (Bleat sits under Sheep and Goat);
   round 2's Sheep source did not.
6. Additional raters, secondary: after Adam's answers are saved and before the arm key is opened, [N = 1-3: NAMES]
   hearing raters with no pipeline knowledge and no prior sight of any picture rate the same cards on the same page;
   reported as kappa with Adam and as a pooled rate; never a bar; Adam-only stays the primary.
7. Display: stage6 `_opacity` fades a shown picture to alpha 0.77 at bar 0.35 over near-black; the sitting rates
   full-opacity pictures. After the sitting is scored, alpha is set to 1 for any drawn picture (one switch, own
   commit); the frozen TEST rows are never re-rendered; demo videos may be re-rendered and the change is stated in ch3.
```

Report-only side job (touches nothing sealed): Qwen3-Omni-30B-A3B-Instruct on the 54 picture-DEV sounds (span ±1 s),
prompt A free, prompt B closed over the fired sub-labels + "unsure" → parent; B earns a future micro-sitting only if it
changes the drawn noun on ≥ 5/54 sounds with 0 changes to an unfired kind (majority of the three reviewers).
