# Pre-registration: Step 1, hold a picture while the sound is still heard (DEV only)

Written and committed 2026-10-06, before any cell of this sweep was built or scored.

## Problem

Stage 4 ends a span when the score drops below the start bar (AED_RELEASE None, AED_HYSTERESIS 1.0). A sustained
sound dips, so the picture leaves early. Frozen system (tag detector-frozen-2026-10-02) on merged DEV (71 clips),
scorer v2 (`score_coverage.py`, display = scoring harness, MAX_AFTER_END None):
29/58 hits, 15 wrong, onset cost 2.056, **cost_cov 2.509**, hit coverage 0.72, wrong 1.11 s/clip, stale 0.12 s/clip.

## The rule (general, online per video, no new model calls)

For every burst (`spans` entry) of every drawn picture: the start is not touched. From the burst's current end, walk
forward on a 0.02 s grid. A moment has **evidence** for the picture's label if any enabled ear is at or above its bar,
where an ear's score = max over its classes in the label's family (`score_per_sound.same_family`), read from the cached
frame scores (BEATs `j2_*_beats`, FlexSED `flexsed_cache`, DASM `devcand/dasm_cache` / `dasm_dev2`):

- FlexSED >= F (release bar)
- BEATs >= B (low bar)
- DASM >= 0.35 (D on) — the bar of the shipped two-witness rule

Gaps without evidence shorter than 1.0 s are bridged. The new end = last evidence moment of the chain + 0.5 s, and
only if the chain starts within 1.0 s of the current end. Ends never get shorter. Caps: the clip end; a burst never
runs into the next burst of the same label; and if the frozen display shows that next burst as a separate picture,
the end stays more than MERGE_GAP (2.5 s) before it (no new joins, so no picture start can change).

**Gate (G)**: strict = the tail may only run through time inside a stretch the gate judged "not seen" for that
family (the frozen arm's own gate records, `_trail` arm); lenient = the tail stops at the start of any stretch judged
"seen" for that family; time the gate never judged is allowed. Strict is the coordinator's rule; lenient is the
declared variant (the gate's stretches end where the original span ends, so strict may allow almost nothing).

Then the real display code (`_display_spans` + `_assign_rows`, the frozen arm's display flags) draws the pictures.

## Grid (36 cells)

| knob | values |
|---|---|
| F, FlexSED release bar | 0.3, 0.4, 0.5 |
| B, BEATs low bar | off, 0.10, 0.175 |
| D, DASM >= 0.35 | off, on |
| G, gate | strict, lenient |

Fixed: bridge < 1.0 s, tail <= last evidence + 0.5 s, starts unchanged.

## Pass bar (DEV, all four)

1. hit coverage >= 0.90
2. hits 29 and wrong 15 unchanged, and every clip's set of picture starts identical to the frozen system's
3. wrong-picture seconds <= 1.11 x 1.15 = 1.28 s/clip
4. stale seconds <= 0.12 + 0.5 = 0.62 s/clip

## Selection

Among cells passing all four bars: lowest DEV cost_cov; ties within 0.005 -> lower wrong + stale seconds. If no cell
passes bar 1, the report names the lowest-cost_cov cell among those passing bars 2-4 and says that bar 1 was not met;
adopting anything is Adam's call. TEST is not scored for any cell (only the frozen system's TEST baseline is in
`frozen_v2.json`).

## Files

`dump_evidence.py` (cluster: frame-score curves + gate stretches, DEV only) -> `evidence_dev.json`;
`hold_sweep.py build` (cluster: pictures per cell) -> `hold_pics_dev.json`; `hold_sweep.py score` (local) ->
`hold_sweep_dev.json`, `hold_sweep_dev.md`.
