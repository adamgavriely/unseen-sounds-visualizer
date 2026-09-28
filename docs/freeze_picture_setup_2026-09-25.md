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

### Report-only listener job — counts defined before it runs (2026-09-27)

Sounds: the 54 fresh-picture sounds (`data/work/picture_bench_fresh/specs.json`, sources and fired lists from
`subjects_V31G.json`, the frozen text), audio from `data/input/pic_fresh/<clip>.mp4`, span start − 1 s to end + 1 s (at most
7 s past the start). Model: Qwen3-Omni-30B-A3B-Instruct, text only, greedy.
**Population for prompt B, counted first:** sounds with at least one fired label that is neither the source nor one of the
source's ontology ancestors (a real alternative). If fewer than 5, prompt B cannot meet the bar and is reported moot.
**Prompt A** ("What is making this sound? Answer with a short noun phrase."): each answer is matched by words to the
AudioSet display names and classed, in this order, as **source** (names the chosen source or a descendant), **family** (names
an ancestor of the source only), **fired sibling** (names another fired label), **other ontology** (names another AudioSet
label), **off-ontology** (none). Raw answers kept.
**Prompt B** (choose one of the fired labels, or "unsure"): a **noun change** is a choice that is neither the source, one of its
ancestors, nor "unsure". Bar (report-only): prompt B is worth a future micro-sitting only if it makes ≥ 5 noun changes of 54
(changes to an unfired kind are impossible by construction and are reported as 0). Nothing enters the frozen setup.

**Result of the report-only listener job (2026-09-27; `benchmark/gold/pictures/listener_pictures.json`).** Parse fix, disclosed:
the model ran on past its answer, so prompt A is classed on the answer's first line (the counts are the same either way).
Prompt A (free "what is making this sound?"), 54 sounds: **source 9, family 8, fired sibling 2, other ontology 30,
off-ontology 5** — the free listener names a different maker on 35 of 54 (cluck → "goats bleating", meow → "human beatboxing",
shatter → "a sword", sheep → "a bee hive"). This confirms the panel's rule: free audio-LLM text must never reach the prompt.
Prompt B (closed choice over the fired labels): population 8; **6 noun changes** (≥ 5, so the written bar is met): church
bell → change ringing, pigeon → coo, telephone bell ringing → ringtone, railroad car → train, police car (siren) → siren,
emergency vehicle → siren. By eye, three of the six make the drawn noun vaguer (train, siren ×2), one is the same bird's
call, and two could help (change ringing; ringtone for the phone that was drawn as a desk bell). Reported as future work; no
micro-sitting before submission, and nothing changes in the frozen setup.

## Amendment, 2026-09-28 — the confirmation sitting is cancelled (Adam)

Adam's decision: the sealed confirmation sitting (GP-4 step 6) will not be held. The picture check-and-redraw loop
(`src/stage6_visual_augmentation/verify.py`, PICTURE_VERIFY on in `use_shipped()`; validated before use: 7/7 named bad
pictures caught, 33/33 deliberately wrong pictures caught, 3/68 good pictures rejected, 113/115 same answer on reshuffle)
replaces it. The sealed files stay unopened. The picture claim rests on the blind human glance test round 2
(14 → 26 of 54, +22 points) and on the loop's validation; the thesis states that the loop is a model check, not a
human recognition test, and that the final human confirmation was not run.

## Amendment, 2026-09-28 (after the sitting was cancelled) — the maker rule and VLM-written looks (Adam)

**Rule (Adam, 28 Sept).** If the sound is already an object (ambulance, bird, frog, car), draw it. If the sound is an
action (honk, chirp, knock, bang, laughter, applause, run, typing), the picture must show the OBJECT that makes it. If
several objects could make it, the video decides: the VLM picks one maker from the sound's own frames.

**How (config `PICTURE_MAKER`, `reason.with_maker`, table `reason.MAKERS`).** Only for the action labels in the table.
If the V3.1 subject already names one of the makers, it stays. Otherwise: one maker -> its fixed subject; several makers
-> one closed question to the same VLM on the same six frames as RESOLVE ("which of these ... or unsure"), answer
matched exactly against the list; unsure or no match -> the maker the subject names, else the default (first):

| label (source) | makers (default first) |
|---|---|
| Laughter, Giggle, Chuckle, Belly laugh, Snicker | a person |
| Baby laughter | a baby |
| Applause | an audience clapping their hands |
| Clapping | two hands |
| Run / Walk, footsteps | a person running / walking |
| Typing | computer keyboard (hands typing), typewriter |
| Honk (AudioSet: the goose's call) | goose, car |
| Toot / Vehicle horn | car, bus, truck, motorcycle |

Cause of the five action-only subjects in the 82 shipped pictures: every guard in `_depict_v31` that empties the phrase
falls back to the bare head word, and the head-word guard misses inflections and synonyms ("A man laughing" does not
name "laughter"). Burst cards (Whoosh, Thud, Bang, Smack, Whack) are unchanged: open question whether they need a maker.

**VLM-written looks (config `PICTURE_LOOK_VLM`, `verify.describe`).** Adam: the hand-written rewrite wording in the
checker's AMBIGUOUS table is "too specific". The clearer wording used by the redraw loop (from try 3, from try 1 for the
smoke detector) is written by the VLM, text only: "In one short sentence for an image generator, describe what a
typical <maker> looks like while it makes its <sound> sound ...". Guards (lists, no model judgement): it must name the
maker noun and pass `expand_guard`, `names_forbidden` and `other_branch_makers` (the maker's own words allowed);
otherwise the plain subject is used. Look-alike options written the same way (`PICTURE_LOOKALIKE_VLM`) are adopted only
if the checker validation still meets its four targets.

**Results (28 Sept).** Audit of the 82 shipped pictures: 70 subjects name a maker object, 5 name only the action
(applause, laughter x2, run, typing), 7 are a sound's own visible form (explosion x3, thunder x2, rain x2), 0 scene only.
With `PICTURE_MAKER` the 5 action-only subjects change (applause -> "an audience clapping their hands", laughter ->
"a person laughing" x2, run -> "a person running", typing -> "hands typing on a computer keyboard", the frames picked the
keyboard); all 5 pass the check on try 1 and are fine by eye. No other subject changes. **Adopted: `PICTURE_MAKER` on in
`use_shipped()`**; the 5 inspector pictures and their videos are redrawn (`data/work/shipped_vm_<tag>`); no score changes.
`PICTURE_LOOK_VLM`: **not adopted**. On the 12 redraws that reach the rewrite, it was worse by eye than the hand-written
wording: both DEV fire-alarm bells and the steam fell to word cards (the VLM's look was refused by the guard for
"hammer", and a plume of steam reads as a cloud), the crowd became a giant head in a crowd, the two smoke detectors look
like a fan or a speaker, the rattle a bowl of seeds. The hand-written rewrites stay. `PICTURE_LOOKALIKE_VLM`: **not
adopted** -- checker validation 3/7 known-bad caught (target 7/7), 29/33 wrong caught (33/33), 2/68 good rejected,
108/115 same on reshuffle; the hand-written look-alikes stay. Horn wordings on the sliceB Honk clip (2 seeds each):
"driver's hand pressing the horn on the steering wheel" 0/4 pass (an interior close-up of a hand), "a car seen from the
front with curved sound-wave lines coming out of its front grille" 4/4 pass and clear by eye, a goose (AudioSet's Honk)
2/2 pass; not adopted until Adam chooses (Honk = goose or car horn).
