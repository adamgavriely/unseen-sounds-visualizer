# PICTURE_V3 — what is tested and what counts as better (fixed before any V3 picture is drawn)

2026-09-24. Adam: pictures are drawn from the family tag instead of the specific sound; the scene must
be used; the prompt and probably the model need to be much better; use a stronger judge; ask what is
seen and whether it fits. Settled by a panel of five reviewers over two rounds
(`docs/picture_panel_brief.md`, `docs/picture_panel_round2.md`); round 3 reviews the built code and
this page before anything runs.

## The two arms (same sounds, same seed)

| arm | what it is |
|---|---|
| **today** | the shipped setup: family label + detail, v1 depiction, FLUX.1-schnell (render `fresh_v31`) |
| **N** | `PICTURE_V3`: specific source (`labels.choose_source`, REL_FLOOR 0.5, tie margin 0.8, both fixed a priori), v3 depiction (whole source caught making this sound, no place), Qwen-Image-2512 with the scenery negative prompt, blank guard only, **seed 0 is the picture shown** |

Sounds: every sound drawn by today's setup on the **50 DEV clips outside the 49-clip bench**
(`data/input/gold_fresh`) — clips the rules were not written from. TEST is not touched.

## Adam's rating

Pass 1 only, today: the picture **alone, no name under it**, and he types what he thinks is making a
sound (or "can't tell"). Both arms' pictures of every sound are mixed and shuffled under codes; ten
pictures appear twice, far apart. The key is sealed in a separate file.

Scoring, by the author, with the key sealed while scoring each answer against its sound:

  * **correct** — names the specific source the detector heard (or, where the source is the family,
    the family's thing);
  * **true but vague** — names only a broader kind (reported separately, not counted as correct);
  * **wrong** — names a sibling (fire engine for ambulance) or something of another kind;
  * **can't tell**.

## What counts as better

  1. **Correct rate**: N above today, paired over sounds, bootstrap 95% CI over sounds excluding zero.
  2. **No confirmed false message**: a *wrong* answer on an N picture where the picture itself shows a
     sibling or a thing the audio never established (fire, a bomb, a place) is looked at; one confirmed
     case vetoes adoption, whatever the correct rate.
  3. **Consistency**: Adam's own agreement on the ten repeats is reported; below 80% the round is called
     inconclusive regardless of the difference.

## Logged, not acting

The checker Adam asked for (GLM-4.6V-Flash, a model family used nowhere else, shown only the picture,
not told the sound) records what it thinks is drawn for every picture of both arms. It changes nothing:
it may not reject or redraw anything until its answers agree with Adam's on this round
(the calibration that decides that is written after, not before, his answers arrive).

## Not in this round (waiting)

The scene fields and RESOLVE step; the frames choosing between tied siblings; the checker acting; a
new describer for the judge; Z-Image; more seeds; pass 2 (name + frame + "anything false?").

## Amendment after round 3 — before any V3 picture is drawn

**The clips.** The "50 DEV clips outside the bench" turned out to be **TEST clips**: `split.json`
describes an older division, and every one of the 50 is in the gold TEST set. The render was cancelled
before anything from it was read, and its link folder removed. The picture test now uses **benchmark
clips that are not in the gold set at all** — never annotated, so neither DEV (where the rules were
written) nor TEST. Adam's pass-1 answers are scored against the detector's source, so no gold label is
needed. 50 clips, fixed seed: every unannotated `mixed` (13) and `unseen_ambient` (19) clip, plus 10
`seen_ambient` and 8 `no_ambient`.

**The code that runs** is HEAD after round 3, not 46ffa80. Changes since: the tie test sees every
sibling at the detector's bar, not only those above the floor (a 0.06 gap between two sirens decided
the picture); the source chain stops at the family or after three links; the depiction prompt again
says the thing is never the sky, the air or a body part alone, asks for the weather sign that appears
at the same instant as the sound, and asks for the word that says which kind of thing it is; "people"
is no longer in the negative prompt; weather subjects keep "sky" and "landscape"; the checker parser
reads markdown; the top candidates behind each chosen source are logged.

**Arms: three, not two.** `today` (shipped subjects, FLUX), `N` (V3 subjects, Qwen-Image, negative),
and **`N0` (shipped subjects, Qwen-Image, negative)** — so the gain can be split between the new
subjects and the new generator, which Adam asked about ("the prompt and probably the model").

**Scoring, made exact before the answers are opened.**
  * A per-sound answer sheet — for each sound: the source; accepted correct nouns and acts; vague
    nouns (the family / parent); wrong nouns (siblings) — is written and committed after the V3
    subjects exist and **before Adam's file is opened**. Anything not on the sheet is logged as
    "unclassified" and reported, never silently decided.
  * An answer **narrower** than the source counts as correct and is reported separately ("ambulance"
    for a drawn "emergency vehicle"); when the source is a tie's common parent, any of the tied
    children counts as correct.
  * The scorer sees the code, Adam's text and the sound (`*_SOUND_KEY_open_for_scoring.json`) and never
    the arm (`*_ARM_KEY_do_not_open_until_scored.json`, opened only after scoring).
  * The "today" and "N0" arms are scored against the same specific source they never tried to draw;
    that is Adam's complaint made measurable, and the report says so.
  * N's gain is also reported split by **whether the source differed from the family**, which is the
    only part of this round that measures specificity rather than the generator.

**The checker's calibration rule, frozen now.** The checker's OBJECT/SOUND answers are scored with the
same sheet, arm hidden, before Adam's file is opened. Over all rated pictures of all arms (repeats
excluded), it earns a vote only if it catches at least 70% of the pictures Adam did not get right,
wrongly rejects at most 15% of those he did, and reaches Cohen's kappa ≥ 0.5 with him. The prompt is
frozen; if it fails, it stays a logged column and any redesign is calibrated on a fresh set.

**Known limits of this round, stated in advance.** One source per family, chosen on its strongest burst
(a truck at 5 s and a horn at 20 s get one source). Depth ranks across branches, so a sound name can
beat the thing that made it (a horn over a bus) — logged per picture through the candidate list. The
scene does not act this round, so Adam's **car-door example cannot be fixed yet**: AudioSet has no
car-door sound, and only the scene could tell a car door from a house door. The rooster case can be.

## Amendment 2 — 2026-09-25, after round 2 of the night panel, before any N picture is seen or rated

Written after V3's *subjects* on the 50 clips were read (text only) and before any picture of any arm
was looked at or any rating received (`docs/panel_2026-09-25_brief.md`, `…_round2.md`).

* **Adam rates today / N0 / N, as declared.** N stays: dropping it after reading its subjects would be
  a fork, and N vs N0 is the prompt-versus-model split he asked for. **Primary: N vs today.** N0 vs
  today, N vs N0, the seed rows and the checker are secondary.
* **Seed 0 stays the picture shown.** Seeds 1 and 2 are drawn for every arm and go to the checker's
  logged column only (seed noise), never to Adam's cards and never to choose a picture.
* **V3.1 (`PICTURE_SCENE`) is not rated on these 50.** Its rules were written from these clips' V3
  subjects (eight named sounds: #10, 19, 20, 27, 29, 39, 41, 53). Its fresh set, reserved now: the
  **30 slice-B clips** (no picture rule has been written on them; rendered under the current config as
  `sliceB_v32`), topped up with unannotated `seen_ambient`/`no_ambient` clips if they yield too few
  sounds. On these 50 it is drawn only as checker-logged examples, labelled in-sample. N1's verdict
  also needs pass 2 (the source name + a frame: "does this say anything false?"), because pass 1 scores
  a narrower-but-wrong kind as correct.
* **V3.1, as signed:** the scene may add at most two qualifier words to the heard word, never a new noun
  and never a person; a list guard in `src/labels.py` (`names_forbidden`: siblings, unfired kinds, kinds
  inside a tie, and labels carrying the heard word) refuses anything the audio did not establish; no
  question to the model that wrote the answer is used as a check; a sound-name source with no qualifier
  is drawn as its visible effect, not a guessed maker.

**Amendment 2a (2026-09-24 ~21:15, before slice B is drawn).** V3.1's subjects on the 50 in-sample clips
(text only) still put a person into a sound with no human source ("A person whooshing a flag" for a
Whoosh, twice). A list check is added: outside the AudioSet "Human sounds" branch, a subject naming a
person is retried once with "Do not show any person" and otherwise falls back to the heard word. Known
gap, not fixed this round: in the pipeline path (`decide_subjects`) the raw firings are not on the spec,
so the guard runs with none allowed (stricter than the bench, where the fired labels are passed);
carrying them on the spec is required before V3.1 could ship.
