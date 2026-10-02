# Picture quality — what is tested and what counts as better (fixed before any result)

2026-09-24, night. Adam: *"the image generation is not always very nice, it should be better. Maybe
try to get context from the scene? Or ask the judge what was drawn and see if it fits the sound."*
Settled with three reviewers over two rounds. Written before any of the arms below has produced a
picture.

## The reference

My own verdict on each of the 33 pictures of the final DEV render, committed before any evaluator ran
(`benchmark/gold/pictures/hand_labels_dev_symgen_v30.json`): 18 good, 3 the right sound but the wrong
kind for the scene, 2 blank, 5 wrong, 4 faint, 1 ambiguous and excluded. Binary — *would a viewer
glancing at it read the right sound?* — 24 yes, 8 no, over **32** pictures.

## The arms, measured in order, each against the one before it

So that a gain can be attributed, not only observed.

| arm | what changes | generator |
|---|---|---|
| A0 | the shipped pictures, regenerated with a fixed seed per picture | FLUX.1-schnell |
| A1 | + the blank/faint guard | FLUX.1-schnell |
| A2 | + new subjects: `DEPICT_V2` and `KIND_ALWAYS` | FLUX.1-schnell |
| A3 | the A2 subjects, drawn by Qwen-Image-2512 | Qwen-Image-2512 |

The subjects of A2 come from re-running only the depiction step of stage 5 on the same 33 sounds,
with the same frames; the gate is not re-run, so every arm draws the same 33 sounds.

**The guard** (A1 onwards, no model): the fraction of pixels that are not near-white. Below the bar,
redraw with seed + 1; still below, the picture is dropped and **counted as a failure**. At most two
redraws. The number of times the guard fired is reported per arm — an arm that needs it six times is
worse than one that needs it once at an equal final score.

## The evaluators, and how they earn a vote

Two, and they must be independent of each other and of everything in the pipeline:

  * **Idefics3-8B-Llama3**, forced choice. Shown the picture alone, asked which sound it shows: the
    target label, five decoys, or "cannot tell". The decoys are the target's siblings in the AudioSet
    ontology, drawn once with a fixed seed and identical across arms (hand-picked decoys are a lever).
  * **CLIP-L** (`openai/clip-vit-large-patch14`), the same target-plus-five-decoys question as a
    contrastive ranking. Fallback if it cannot be fetched: `clip-vit-base-patch32`. **Not**
    SigLIP-so400m — it is Idefics3's own vision tower, so the two would not be independent.

Why these: the Qwen3.8 VLM writes the subjects *and* describes the panels for the stage-7 judge, and
Gemma-4 is a judge row; neither may mark the pictures.

**Calibration first.** Each evaluator is run on the 32 current pictures and compared with the binary
hand verdict. An evaluator that agrees on **fewer than 26 of 32** does not vote.

## What counts as better (per arm, against the arm before it, 32 pictures, same seeds)

  1. **0 blank** after the guard;
  2. **net ≥ +4** pictures recovered, on **both** voting evaluators;
  3. **≤ 1** picture my hand verdict calls good that either evaluator says got worse, and that picture
     is looked at by eye before deciding;
  4. **no confirmed false message**: any picture where either evaluator picks a label from a
     *different top-level family* than the target is listed and looked at; one confirmed case —
     a picture that would tell a deaf viewer something that is not there — vetoes the arm whatever
     its net;
  5. any picture the two evaluators disagree on is also looked at by eye.

The flip table (fixed / broke / unchanged, per picture) is reported alongside the net, not instead of
it. If only one evaluator passes calibration, the arm is judged on that one plus the eye, and the
report says so.

## What this does not claim

The new subject rules were written from these 33 pictures. On DEV they are a smoke test; they are
claims until they are run on clips they were not written from.

## Deliberately not built tonight

A **verify-and-redraw loop** — draw, ask a model whether it fits, rewrite and redraw until it does.
All three reviewers rejected it as an optimiser: the pictures that survive are exactly the ones the
checker was lenient on, and the subjects drift towards what is easy to caption rather than what was
heard. Adam's idea is kept in the form that is safe: the check runs as a **logged verdict column**,
nothing is dropped on it, and it will be calibrated like any evaluator before it is allowed to act.
