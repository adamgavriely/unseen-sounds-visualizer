# Picture sense test (28 Sept 2026) -- plan, written before any picture is drawn

**Question.** Can a GENERIC method replace the hand-written per-word table (`verify.AMBIGUOUS`: intended, confusions,
rewrite, neg, rewrite_first) for tricky picture words, without losing pictures?

**Code.** `src/stage6_visual_augmentation/sense.py` (new; flag `config.PICTURE_SENSE`, default off), one flag-guarded
hook in `stage6._final_picture`, runner `scripts/picture_sense_test.py`, jobs `slurm/job_sense_prep.sh`,
`slurm/job_sense_arm.sh`. The AMBIGUOUS table, MAKERS and verify.py are not changed.

## The new method (arm C)

**(a) Slot form.** The pipeline VLM (Qwen3.8-27B), text only, greedy, gets the label (`spec.source`), its AudioSet
ontology path (`src/labels.py` ancestors, root first) and the label's OFFICIAL AudioSet description (vendored
`src/audioset_ontology.json`, the unchanged `ontology.json` of github.com/audioset/ontology). It fills four fixed slots:
OBJECT (what makes the sound), WHERE (where it is, close up, no scene), ACTION (what it does while sounding), CUE (one
visible sign of the sound). Drawing sentence, fixed template: `{a/an} {object} {where}, {action}, {cue}` (+ the shipped
rules tail). Guards (any failure -> the plain subject is drawn, as today):
1. all four slots present, each 1-8 words;
2. the OBJECT's head noun (its last word, plural stripped) is a word of the label's names, its ancestors' names or its
   official description;
3. no extra objects: no word that names another sound source, a person or a place (`reason.expand_guard`, with the
   object as subject and the description's words also allowed; "-ing" words such as ringing / cheering are the sound's
   action, not a source, unless they are a place word like "building"), no `labels.names_forbidden` word, no
   `labels.other_branch_makers` word;
4. no text words (text, letters, words, sign, label, logo, writing, caption, banner, number, numbers);
5. no people or hands unless the label is under "Human sounds".
Templates and word cards are not touched (C replaces only the V3.1 subject).

**(b) Mistake mining.** For each label, 4 pictures of the PLAIN subject (the one `_final_picture` would draw on try 1:
subject + rules tail + the screening negative), seeds `crc32("sense-mine|<label>|<k>")`, k = 0..3 (disjoint from the
test seeds). The VLM is asked openly: "What is the main thing in this picture? Answer in a few words."
**Judge (declared): the VLM, text only, one fixed closed question** -- "A picture was meant to show: <target>. (It
stands for the sound '<label>': <official description>) A viewer says the main thing in the picture is: '<answer>'. Is
that the intended thing, or a kind of it? Answer no if the viewer names a different main object, even a related one.
Answer yes or no." (target = the slot OBJECT if the slots passed, else the label's first name). Each "no" answer becomes
(i) negative-prompt words: its words minus the drawn sentence's words, the label's and ancestors' names, filler words and
a fixed stoplist (colours, materials, "view, image, picture, drawing, illustration, cartoon, icon, photo, close-up,
background, object, device, thing"), and "-ing" words ("a steaming kettle" -> kettle); (ii) a checker look-alike option "a <answer>", unless its content words are all
words of the target / label / drawn sentence. At most 4 look-alikes. Cached per (label, picture model) in
`data/work/sense_cache/Qwen-Image-2512.json` (with the LLM id; a different LLM recomputes the slots) and reused.
In production, C's own checker = intended "a <object>" (or the plain subject) + mined look-alikes + the generic options.

## Arms

| arm | drawing | checker options in THIS test |
|---|---|---|
| A | shipped: hand table (rewrite from try 3, rewrite_first for the smoke detector, table neg) | union (below) |
| B | `PICTURE_LOOK_VLM` + `PICTURE_LOOKALIKE_VLM` (VLM free-text look, VLM look-alikes) | union |
| C | `PICTURE_SENSE`: slot sentence + mined negatives from try 1; no table | union |

Same everything else: `use_shipped()`, Qwen-Image-2512 at 1024 px (stored at 512 px after checking), same seeds
(`seed_of(clip, label, start) + 1 + 1000 * try`), 5 tries, each refused try's wrong pick added to the negative
(`verify.feedback_negative`), "no text" from try 3 or after OCR text, then a word card.

**Checker (same for all arms).** One fixed question per sound: intended = arm A's intended option (`verify.options_for`,
the table's for the 7 known, the plain subject for the 25 unseen); confusions = the UNION of A's (table + generic), B's
(VLM look-alikes) and C's (mined) options, de-duplicated; an option whose content words are all words of the intended
option / label / plain subject is dropped (it would be a second right answer). Same shuffled order for every arm
(`verify._order_seed`). Up to 20 letters. So A in this test is not byte-identical to the shipped checker; the test
measures the DRAWING side. C's own table-free checker is also run once on every C final picture and reported.

**B on the 25 unseen is A by construction** (no table entry -> the B flags do nothing; same seed, greedy VLM). B is
drawn only on the 7 known; its unseen rows are A's rows, said so in the table.

## Sounds

**7 known bad cases** (one real spec each, the stored clip subject + `reason.with_maker` text-only, frozen):

| case | clip / index | label / source | shipped mistake |
|---|---|---|---|
| fire alarm | DEV `as_fire_alarm_kGKZ0YK4` / 0 | Alarm / Alarm | desk bell |
| crowd | DEV `mv_protest_scene_movie` / 0 | Crowd / Crowd | flock of crows |
| steam | DEV `b3_crossing_bells` / 0 | Steam / Steam | kettle |
| smoke detector | TEST `w8_helmetcam_chainsaw_roof_2b` / 1 | Alarm / Smoke detector, smoke alarm | dome camera |
| car horn | sliceB `97aoiaWwRVk_20000` / 0 | Vehicle / Vehicle horn, car horn, honking | megaphone / trumpet |
| rattle | sliceB `_U8kAFAm8tQ_30000` / 0 | Rattle (instrument) | ball of yarn / swirl |
| typing | TEST `ambient_nightlife_neon_97` / 0 | Typing / Typing | shouting man |

**25 unseen labels**, drawn before any picture: `random.Random(20260928).shuffle(sorted(families))` over the 215
families of `benchmark/gold/depictable_vocab.json`, in order, skipping CARDS, TEMPLATES, the 7 known sources (and
Vehicle), and any label matching an AMBIGUOUS sound name (all 215 pass the shipped `is_salient_nonspeech` filter).
Skipped on the way: Vehicle (known), Thunder (template).

1. Chorus effect 2. Tools 3. Miscellaneous sources 4. Shuffling cards 5. Hair dryer 6. Stomach rumble 7. Doorbell
8. Grunt 9. Power tool 10. Otoacoustic emission 11. Hoot 12. Dental drill, dentist's drill 13. Wobble 14. Drill
15. Bird 16. Whale vocalization 17. Microwave oven 18. Boat 19. Gull, seagull 20. Wood 21. Printer 22. Cat
23. Dial tone 24. Splash, splatter 25. Battle cry

Reserve, in order: Fire, Wildfire, Duck, Wail, moan, Heart murmur, Engine knocking, Chewing, mastication, Arrow, Cash
register, Snort (horse). **Swap rule:** an unseen label whose frozen plain subject matches an AMBIGUOUS entry (subject
patterns) is replaced by the next reserve label. Plain subject for an unseen label = `reason._depict_v31` with no frames
and no place (the pipeline's own text step; RESOLVE falls back to the head word) + `with_maker` (no frames -> default
maker), frozen in `items.json` by the prep job.

**Amendment 1 (before any picture was drawn).** The first prep job (31330333) showed, on the slot TEXT only, two
false refusals by guard 3: "inside" (a name of "Inside, small room") and "beeps" ("Beep, bleep") counted as other sound
sources. Fix: a word counts as another source only if some ontology label carrying it lies outside the two branches
that name acoustics or sound shapes, not things ("Channel, environment and background", "Source-ambiguous sounds").
The job was cancelled before mining; its slot cache was set aside (`sense_cache/Qwen-Image-2512.aborted.json`) and
everything re-ran from the start. No other guard was changed.

## Metrics (per arm; known 7 and unseen 25 apart)

- pass at try 1 (the checker passes the first picture);
- pass within 5 (a picture, not a word card);
- word-card rate;
- by eye: right object? yes / no for every final picture (a word card = "no picture": not correct, not a false message);
  wrong pictures (a false message) counted apart;
- C only: its own table-free checker on its final pictures.

**By-eye protocol.** Final pictures are copied to `blind/<random id>.png` with only the sound's label shown; the key
(`unblind.json`) is opened only after all verdicts are written to `verdicts.json`. Limits: word cards are recognisable,
and B = A on the unseen, so B's unseen pictures are duplicates. The verdict is Claude's and provisional; Adam's own look
at the contact sheet (`docs/picture_sense_test/index.html`) overrides it.

## Decision rule (fixed now)

**C replaces A** if and only if
1. on the 7 known cases, C's by-eye correct count >= A's, and
2. on the 25 unseen, C's by-eye correct count >= A's AND C's word cards <= A's.

Otherwise A stays. B is reported for context only. No re-runs, re-seeding or guard changes after the pictures are seen;
any change is a new, separately declared test.

## Results

(filled in after the run, below this line)

Run: jobs 31330354 (prep + mining), 31330355 (A, B), 31330356 (C), all COMPLETED. 32 sounds, 0 swaps. By-eye verdicts
were written blind (71 pictures, random ids, `verdicts.json` before `key.json`). Contact sheet:
`docs/picture_sense_test/index.html`.

| arm | sounds | pass at try 1 | pass within 5 | word cards | by eye right | by eye wrong |
|---|---|---|---|---|---|---|
| **A hand table (shipped)** | known 7 | 2 | 7 | 0 | **7** | 0 |
| B VLM free text | known 7 | 2 | 5 | 2 | 3 | 2 |
| C new (slots + mining) | known 7 | 2 | 3 | 4 | 3 | 0 |
| **A hand table (shipped)** | unseen 25 | 16 | 19 | 6 | **17** | 2 |
| B VLM free text (= A by construction) | unseen 25 | 16 | 19 | 6 | 17 | 2 |
| C new (slots + mining) | unseen 25 | 16 | 19 | 6 | 17 | 2 |

C's own table-free checker passed 2/3 of its known pictures and 17/19 of its unseen pictures.

**Decision (pre-set rule): C does NOT replace A.** Rule 1 fails (known: C 3 < A 7). Rule 2 holds (unseen: 17 = 17
right, 6 = 6 cards). A stays.

Where the arms differ:
- **Typing.** A: hands on a keyboard, passed at try 1. C: its slot sentence has no hands (the no-people guard), and
  mining added "hands" as a negative word. So C drew a keyboard alone five times. The fixed question expects "hands
  typing on a keyboard", so every try was refused and C ended with a word card. This is partly a checker mismatch:
  a keyboard alone is not a wrong picture.
- **Fire alarm.** A: a red alarm bell on a wall (try 3, the table rewrite). C: the slot object was just "alarm", so all 5
  tries drew a glowing alarm clock and it ended as a word card. B also ended as a word card.
- **Car horn.** A and B: a car with sound-wave lines (try 3). C: the slots were refused ("driver's finger") and
  mining found nothing, so C had no new wording and ended as a word card.
- **Unseen.** The only differences were Chorus effect (A card, C a correct effects unit), Stomach rumble (A a wrong
  picture, a face in a belly; C card) and Miscellaneous sources (A card; C a wrong picture, a ball of twigs). They
  even out.

Weak points of C seen here (for a possible later, separately declared test):
- the object slot can be too generic ("alarm");
- the no-hands guard fights human-action makers (typing);
- the mining judge marks the right thing as wrong, so it becomes a negative word: "phone" for Dial tone, "water" for
  Splash, "man" for Battle cry.
