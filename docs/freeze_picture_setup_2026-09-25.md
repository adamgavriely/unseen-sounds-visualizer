# FREEZE — the final picture setup (GP-4 step 5), committed before any confirmation picture is drawn

Code at commit 3995769. Selected on picture-DEV by the author, blind to arm ("selected, author-screened";
`docs/panel_2026-09-25_scene_prompt.md`, author screening result).

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
