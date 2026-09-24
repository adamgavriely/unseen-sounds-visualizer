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
