# Detector — final (frozen 2 Oct 2026, 01:20 UTC)

**Frozen version: D'** = `ARMS["SHIP8+MD3+WW5+SL"]` = `config.use_shipped()` (parity check OK). Git tag `detector-frozen-2026-10-02`.

| set | hits / needed | wrong pictures (on screen / other sound / nothing) | cost* |
|---|---|---|---|
| merged DEV (71 clips) | **29 / 58** | **15** (6 / 7 / 2) | **2.056** |
| merged TEST (88 clips, read once, reported) | **24 / 65** | **24** (4 / 15 / 5) | **2.409** (B: 2.545, p 0.034) |

*cost = (4 x missed + 2 x wrong) / clips; lower is better. A hit = a picture of the right sound type starting 0.5 s before to 1 s after the sound.

## The pipeline in plain words (one video at a time)

1. **Hear candidates.** Two sound detectors listen to the whole clip: BEATs (a sound tagger) and FlexSED (finds sounds by name). Anything they hear strongly enough (and at least 0.3 s long) becomes a candidate.
2. **Second opinions for weak sounds.** Two "listener" models (Qwen3-Omni, Audio Flamingo) hear short cuts and say whether the sound is really there; a third model (DASM) must also agree.
3. **Filters ("witnesses").** A candidate needs backing: DASM clearly hears it, or both listeners name it, or one listener names it AND the vision model finds the sound credible for the scene (read as a probability, not as text). Other filters drop echoes, continuation pieces and sounds heard by one model only.
4. **On-screen check.** A vision model (Qwen3.8-27B) looks at the frames around each sound: if the thing making the sound is visible, no picture (the viewer can already see it).
5. **Clean the display.** Repeats of the same sound close together become one picture (2.5 s gap); Qwen3-Omni decides whether two pictures up to 8 s apart are one continuing sound; a picture is dropped when its event is visibly happening and the visible thing could sound like it.
6. **Draw.** A picture is generated for each remaining sound and shown beside the video while the sound plays.

Live use: `slurm/run_best.sh <name> <folder>` (or `src/pipeline.py`) runs all of this per video; tested end to end on unseen clips.

## What changed on 1 Oct (start of day -> frozen)

| change | DEV hits / wrong / cost |
|---|---|
| start (SHIP8) | 28 / 21 / 2.282 |
| repeat-merge gap 2.5 s (Adam) | 28 / 20 / 2.254 |
| smart grouping (Omni: same sound or new?) | 28 / 18 / 2.197 |
| shortest sound 0.3 s | 29 / 18 / 2.141 |
| visibly-happening check (Round 57) | 29 / 17 |
| listener cache completed (honest base) | 29 / 18 / 2.141 |
| witness rule + scene check (Rounds 53+60) | 29 / 14 / 2.028 |
| scene check read as probability, no cut-off answers (60L) = **D'** | **29 / 15 / 2.056** |

## Why the other ideas failed (all pre-registered; DEV or the held-out 415 decided, TEST never)

| idea | why it failed |
|---|---|
| hear more weak sounds (band runs, short twins, tagger ensemble, rooster rule, tighter listener cut) | the extra sounds are mostly wrong: a sound no listener names is right only 22 % of the time |
| stricter witnesses (Round 53 variants) | removes junk but also real sounds named by one listener (2 hits lost for 5 wrong removed) |
| smarter grouping limit (pause questions) | neither Omni nor the detectors can tell a 2-s pause from a dropout |
| "can you see the sound happening?" (sign / event questions) | the vision model answered by letter position or always "no"; read as probability it was just under the bar (0.647 vs 0.65) |
| Adam's look-alike reasoning (Round 61: no train here -> washing machine) | it works on the laundromat, but the same reasoning wrongly explains away real sounds (a crying sound "explained" by a visible parrot) |
| context +-5 s (Round 59) | Omni names what it SEES, not what it hears |
| multi-step "name it -> find it -> zoom -> is it making the sound now?" (Rounds 50, 50L, 66) | the zoom step works when a source is named, but the naming step and the zoomed answers are too noisy: on DEV it lost 3 real hits |
| bigger / other vision models (Gemma, model research) | every model moves along the same trade-off; the question (object present vs object making the sound) is the limit, not the model |
| bell rule (Round 64) | passed DEV (+1) but is effectively a bells-only rule -> not general, not shipped |

**The main lesson:** the remaining errors are where the models disagree at about chance level, and where "the object is on screen" is not "you can see the sound happening" (church tower vs ringing bell). That is the honest limit of today's open models; a larger vision model (235B) is future work.

Full record: `docs/history/preregistrations/prereg_round13_detector_push.md` (every pre-registration and result), panels in `docs/history/review/panel_2026-10-01/`, `panel2_2026-10-01/`, `vlm_panel_2026-10-01/`, supervisor log `docs/history/review/improvements_log_2026-09-30.md`, ledger of the frozen version `benchmark/gold/ledger_SHIP8_MD3_WW5.json` (D before 60L).
