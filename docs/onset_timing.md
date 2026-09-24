# The picture is out of step with its sound — what is wrong and what we will do

2026-09-24. Adam, watching the rendered videos on the error page: *"even on correct pictures the
sound comes 1-2 seconds early"*. This is the record of what was measured, what a panel of three
reviewers concluded over three rounds, and the rule we are going to implement.

---

## 1. Where the problem is not

A picture can be out of step in three places. All three were measured separately
(`benchmark/gold/timing_audit.py`).

| where | measurement | verdict |
|---|---|---|
| the rendered video | composite audio vs source audio, and composite left-half brightness vs source brightness: **+0.00 s at r = 1.00**, 8 of 8 clips | not the problem |
| the panel following our own plan | panel lights **0.04 s** from what the spec says, 25 clips | not the problem |
| the detector deciding when the sound starts | see below | **this is the problem** |

## 2. How wrong the timing is

23 drawn sounds on DEV match a gold sound of the same family. **13 of them have a gold onset at or
below 0.5 s and cannot be early.** Of the 10 that can:

  * **5 are early by more than 0.5 s** — Gunshot −2.51, Crowd −1.90, Siren −1.50, Bird −1.29,
    Laughter −0.84;
  * median over the 10 is **−0.47 s**, mean −0.80 s.

The end is worse, and it had never been measured. On the 17 clips that show exactly one picture,
measured on the **rendered panel**, the picture leaves **2.30 s before the sound stops** (median),
and **6 of the 17 windows are exactly 8.00 s long** — `config.MAX_SPAN` cutting them, not the
detector losing the sound. The subway picture ends at 8.0 s while the train runs to 15.8 s; the
botanic garden picture ends at 8.2 s while the birds run to 19.0 s.

## 3. What the panel concluded

Three reviewers, three rounds, seeing each other's answers from round 2. **All three changed their
ranking at least once**, and the attribution never converged: the accounts blamed the occlusion
onset refinement, the family merge, and the second detector's min-start rule respectively, and none
of them reconciles with the cached detector scores end to end. Two numbers the caches cannot
produce: the Siren's anchor of 2.60 s, and a Laughter span emitted at confidence 0.93 whose cache
peak is 0.42.

The best candidate mechanism, which fits every number and is now being tested, is that
`_refine_onsets_cam` refines **every** event including the ones FlexSED raised: `idx.get(e.label)`
finds a BEATs class for a FlexSED-origin span too, so a frame-level FlexSED start is treated as a
BEATs sliding-window stamp, `w0 = start - 1.5`, and the snap branch returns `w0` unchanged.
Siren 2.60 - 1.50 = 1.10 exactly; Applause 1.12 - 1.50 = -0.38, clamped by `max(0.0, t)` to the
0.00 the panel could not otherwise explain.

### What the instrumented run actually showed (job 30993350, five clips)

The log settles it, and no reviewer had it exactly right. Two steps move starts, and they compound:

    ly_applause   union   Applause            6.00 ->  1.12   -4.88
    ly_applause   union   Laughter            2.75 ->  0.00   -2.75
    mv_storm      union   Siren               3.00 ->  2.60   -0.40
    mv_storm      union   Civil defense siren 3.50 ->  2.60   -0.90
    all five      refine  49 spans moved: 30 earlier, 19 later, median -0.47, worst -6.00

  * **The FlexSED twin rule is a real mover.** It supplied the 2.60 s Siren anchor that no cache
    reproduced, and it pulled Applause from 6.00 to 1.12. Note it moved Applause **towards** the
    annotator (gold 1.90), so it is not simply wrong -- it is the better of the two detectors here.
  * **The occlusion refinement then takes another 1.5 s off.** Siren 2.60 - 1.50 = **1.10**, the
    number that was emitted. Applause 1.12 - 1.50 = -0.38, clamped to **0.00**, the number that was
    emitted. Both exactly.
  * Refinement is not uniformly early-biased -- 19 of 49 moves were later -- but its bad tail is
    severe, and it is the step that turns two acceptable onsets into two misses.

So the fix is the clamp, not the deletion of either step: **the union forms the anchor, and
refinement may sharpen inside it but never precede it.** On these five clips that alone should
return the Siren to 2.60 (annotator 2.60), the Bird to about 10.25 (annotator 10.40) and the Crowd
to 1.12 (annotator 1.90) -- two misses recovered and one large error halved.

**The pipeline records its own provenance** (`onset_trace.json`, one row per step per span:
extract, FlexSED raw, union, vetoes, occlusion refinement, MAX_SPAN cap, family merge, display
join). An intermediate start is not recoverable from outside; it is now recoverable from inside.

## 4. The rule all three signed

> A later stage may sharpen an onset within its own evidence window, extend an end, or merge spans,
> but it may **never produce a start earlier than the anchor it was given.**

Every mover we can see — the hysteresis anchor, the FlexSED min-start, the occlusion 10 % point, the
snap branch, the family-merge minimum — moves onsets only earlier. Nothing in the pipeline ever
moves one later, and the metric forgives late twice as much as early ([−0.5, +1.0] s).

Exceptions: two of three reviewers allow **none**; one allows the occlusion step to move a start
earlier by at most `STAMP_OFFSET` = 0.5 s and only for an abrupt sound (`ramp = False`), on the
grounds that the stamp itself sits 0.5 s before the window end. All three **reject** a fade-in
exception: the approaching helicopter the `ramp` flag was written for is already handled upstream by
extraction at 0.175, and the ramp path is what moved the Siren and the Bird out of the window.

## 5. The changes that follow from it

1. `beats_infer.occlusion_onset`: the `i <= 1` branch returns `None` (the window cannot localise the
   onset) instead of `window_start`, which is a hard 1.5 s jump backwards.
2. `_refine_onsets_cam`: do not refine a FlexSED-origin span with a BEATs class — it is already
   frame-level. Anchor the search window on the span's own evidence, and clamp the result so it
   cannot precede the anchor.
3. `labels.merge_by_label` and `stage6._display_spans`: joining bursts may extend an end, never move
   a start earlier than the member that supplies the picture.
4. `config.MAX_SPAN`: off for the proposed row, replaced by a release rule on the detector curve
   (`AED_RELEASE` swept over {0.175, 0.10, 0.05}), because the cap is cutting 6 of 17 pictures while
   the sound is still playing.

## 6. How it will be judged (fixed before the run)

**Start**, 49 DEV clips, hit = a picture starting within [−0.5, +1.0] s of a needed sound's onset:
at least 4 of the 5 early cases become hits; **0 of the current hits regress**; the 13 clip-start
sounds are untouched; median |error| over the 10 mid-clip sounds ≤ 0.25 s (from 0.47) and p10 ≥
−0.75 s (from −1.46).

**End**, the 17 single-picture clips on the rendered panel: median displayed-end error improves from
−2.30 s to −1.0 s or better; **zero** panels exactly 8.00 s long; at most 1 of 17 lingering more
than 2 s past the sound's end.

**Headline**: precision and F1 on all 49 DEV inside the paired clip-level bootstrap CI of the
current run — a moved start changes where the visibility check samples its frames, so this is not a
formality.

**Revert rather than keep** if a current hit is lost, if precision falls below that CI, if more than
3 of 17 panels linger past the sound, or if the instrumented log still shows any step emitting a
start earlier than its anchor — which would mean the rule was not actually implemented and any
improvement is luck.

**Compute**: the occlusion refinement and one stage-5 gate re-run need a GPU. Everything else
replays on CPU from `benchmark/gold/beats_fw/*.npz` and `data/work/flexsed_cache/*.npz`, which exist
for all 139 clips.
