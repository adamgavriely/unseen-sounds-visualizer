# Pre-registration: detector round 9 — local-contrast veto on BEATs-only spans

*Written 2026-09-28, after round 8 (`docs/prereg_round8_ideas.md`) and before any round-9 cost was computed. Detection
only, on the AudioSet-Strong fit set (the 280) and the held-out set (the 415). DEV, TEST and the "fresh" set are not
touched. Harness: `benchmark/detector_round9.py` (reuses round 8's stack, scoring and bootstrap), results in
`benchmark/detector_round9.json`. No model is run; only the saved BEATs / FlexSED caches are read.*

## Why
Round 8's I7 (drop a FlexSED-only span whose in-span score barely rises above its ±3 s surroundings) was the only
precision idea that went the right way on both sets (ΔC-overlap −0.064 on the 280, −0.039 [−0.087, +0.005] on the 415),
but it can only touch FlexSED-only spans. The audit (`docs/detector_audit_2026-09-28.md`) found that 80–84 % of the
phantom false spans are BEATs-only. Round 9 applies the same test to BEATs-only spans.

## Fixed parts
Baseline = the shipped stack, exactly round 8's gate 0 (`detector_round8.stack8` with no option = round 5's stack span for
span; 280: 3.036 / 3.850 / 51.3 % / 207 false spans; 415: 1.928 / 2.207 / 49.7 % / 228). Cost C, clip lists, the paired
clip bootstrap (2000 draws, seed 0), the one-sided p, the primary scoring filter (`LABEL_FILTER = "lists"`) and the
secondary "depictable" row are all as in round 8. Reported per cell: C-overlap, C-onset (ΔC with CI), recall, onset
recall, false/min, shown spans, heard-but-dropped events recovered, new false spans (gross / net), and false / true spans
removed.

## The cells (fixed here, no fitting)
**J1 — BEATs contrast veto.** A shown span of BEATs origin with no FlexSED twin (the audit's "BEATs only") is dropped if
the mean of its family's BEATs score inside the span minus the mean over the flank frames is < **0.1**. Family score =
per frame, the max over the BEATs columns whose canonical family equals the span's; inside = frames with t in
[start, end); flanks = frames in [start − 3, start) and [end, end + 3), clipped to the clip; a span with no flank frames
(it covers the whole clip) is kept, as in I7, and the number of such spans is reported. Applied after the baseline's
vetoes and display bar.
*Why 0.1:* I7's margin was 0.2 on FlexSED's scale, where the display bar is 0.8; BEATs' display bar is 0.35. Scaling
the margin by the ratio of the bars gives 0.2 × 0.35 / 0.8 = 0.0875, rounded to 0.1. This keeps the test's meaning:
the span must stand out from its surroundings by about one quarter of the detector's own display bar. The value is set
here without looking at any round-9 number.
**J2 — J1 + I7 together.** J1 on BEATs-only spans and round 8's I7 (margin 0.2, FlexSED scale) on FlexSED-only spans, both
after the baseline's vetoes. Spans with a BEATs and a FlexSED twin are never touched.

## Decision rules
- Picked on the 280 iff C-overlap < baseline AND C-onset < baseline. Each picked cell goes to the 415 alone (frozen); it
  **passes** iff the upper 95 % CI of ΔC-overlap < 0. Holm across the cells sent (family α = 0.025 one-sided) is also
  reported.
- If a cell passes the 415, it is reported and the round stops: the fresh set is NOT scored here (a later, separate step).

## Result (2026-09-28; `benchmark/detector_round9.json`, job 31330329 on the cluster)
Gate 0 passed on both sets (round 5's stack span for span, same numbers).

| set / cell | C-overlap (ΔC, 95 % CI) | C-onset (ΔC, 95 % CI) | recall | onset rec | false/min | shown | spans removed false / true | events lost | depictable ΔC-overlap |
|---|---|---|---|---|---|---|---|---|---|
| 280 baseline | 3.036 | 3.850 | 51.3 % | 25.9 % | 4.44 | 564 | — | — | — |
| 280 J1 | 2.979 (−0.057 [−0.121, +0.000]) | 3.893 (+0.043 [−0.071, +0.186]) | 50.9 % | 22.3 % | 4.22 | 514 | 10 / 40 | 1 | −0.057 |
| 280 **J2** | 2.914 (−0.121 [−0.207, −0.043]) | 3.829 (−0.021 [−0.150, +0.129]) | 50.9 % | 22.3 % | 4.03 | 497 | 19 / 48 | 1 | −0.121 |
| 415 baseline | 1.928 | 2.207 | 49.7 % | 32.7 % | 3.30 | 672 | — | — | — |
| 415 **J2** | **1.822 (−0.106 [−0.178, −0.039])** | 2.120 (−0.087 [−0.164, −0.010]) | 48.5 % | 30.4 % | 2.92 | 573 | 26 / 73 | 2 | −0.092 [−0.159, −0.029] |

J1 on the 280 raises C-onset, so it is not picked; **J2 is picked and passes on the 415** (upper CI −0.039 < 0; one-sided
p 0.0005; Holm with one cell: rejected). Strata on the 415: complex −0.151 [−0.238, −0.071] (n 252), random −0.037
[−0.135, +0.062] (n 163). J1 counts: 1111 / 1600 BEATs-only shown spans considered, 178 / 348 kept for lack of flanks,
172 / 228 dropped. Heard-but-dropped events recovered: 0 (J2 only removes spans).
**Caveat to read with the pass:** J2 removes more matched (true) spans than false ones (415: 73 true, 26 false). The
removed true spans sit on events that another span still hits or on non-consequential events (not split here), so only
2 consequential events are lost
(recall 49.7 → 48.5 %), but onset recall falls 32.7 → 30.4 % and shown spans fall 672 → 573. C does not charge for a
removed duplicate, so part of the gain is "fewer pictures" rather than "fewer wrong pictures".
As pre-registered, the round stops here: the fresh set is not scored.

## Amendment 1 (2026-09-28, written before any fresh-set number and before any DEV J2 number): the two confirmation checks and the ship rule
Decision (lead): no guard is added to J2 (a guard chosen after seeing the 415 would be post hoc). The removed true spans
are cost-neutral on the 415 (26 false × 2 / 415 ≈ −0.125; 2 lost needed events × 4 / 415 ≈ +0.019; total −0.106).

**A. Fresh set (`R.use_set("fresh")`, `docs/prereg_fresh_confirm_set.md`), scored once.** J2 exactly as registered above
(margins 0.1 / 0.2, ±3 s flanks, no-flank spans kept), no refit. Start only when the four caches (BEATs, PANNs, PE-A-Frame
in `benchmark/audioset_fresh_windows/`, FlexSED in `data/work/flexsed_fresh/`) each hold 422 `.npz` files; the clip list is
`detector_round2.usable()`. Gate: the round-8 stack with no option equals round 5's stack span for span on every fresh
clip (no expected numbers exist for this set). **Primary (one cell): J2 passes iff the upper 95 % CI of ΔC-overlap < 0**
(paired clip bootstrap, 2000 draws, seed 0). Secondary (reported, not decisive): C-onset (ΔC, CI), onset recall, recall,
false/min, shown spans, true / false spans removed, the "depictable" row, strata, and **coverage seconds**: for every
consequential salient gold event, the share of its seconds covered by the union of shown spans of the same family
(`E._same`); a missed event counts 0; mean over events, baseline vs J2.

**B. DEV check (DEV only, never TEST).** Renders: `data/work/protocol_{proposed,blind_a2i}_dev_monocap_v31/` (ours and the
pipeline without the gate), the DEV stems of `gold_AG.json` (`score_per_sound.subsets_of(gold)["dev"]`).
1. Frame scores: BEATs recomputed on each DEV clip's `audio.wav` with the shipped `infer_beats` (GPU); FlexSED from the
   pipeline's cache `data/work/flexsed_cache/<stem>.npz` (raw scores, as in the harness). **Gate D0 per clip:** the
   recomputed BEATs spans at 0.175 (`_extract_events`, as stage 4) equal the trace's `extract` step (same labels, starts
   and ends within 0.01 s), and the twin rule rebuilt from the trace's `extract` + `flexsed_raw` steps equals its `union`
   step. A clip that fails D0 keeps its pictures unchanged and is counted.
2. J2 is applied to the trace's `veto` step spans (the stage-4 output before onset refinement): origin = FlexSED-only if
   the span came from `flexsed_raw` without a BEATs twin, BEATs-only if it is an `extract` span with no FlexSED twin; the
   J1 test (BEATs, margin 0.1) and the I7 test (FlexSED, margin 0.2) exactly as above. `veto` and `refine` steps are paired
   by position (asserted same label and end).
3. A picture span (`spans` in `augmentations.json`, shown pictures only) is removed iff every refined stage-4 event that
   matches its label (same canonical family or ancestor/descendant) and overlaps it in time is dropped by J2; a span with
   no matching event is kept and counted; a picture with no span left is not shown. (A merged span whose events are only
   partly dropped is kept whole: shrinking is not simulated.)
4. Scored with the official `score_per_sound` (onset rule, default settings), before / after, for both systems: hits,
   wrong pictures by type (visible, cross-trigger, phantom), duplicates, F1, viewer cost, coverage.

**Ship rule.** J2 goes into `use_shipped()` only if (i) the fresh set passes (A, primary) AND (ii) on DEV, for ours
("proposed"), hits do not drop AND wrong pictures (visible + cross + phantom) drop. The pipeline-without-gate row is
reported, not decisive.

## Result B — DEV check (2026-09-28; `benchmark/gold/j2_dev_check.py`, `benchmark/gold/j2_dev_check.json`, job 31330369)
Gate D0 passed on all 49 DEV clips of both systems: recomputed BEATs spans equal the trace's `extract` step and the twin
rule (weak-twin mode "absorb") rebuilds its `union` step. J2 looked at 602 BEATs-only and 16 FlexSED-only stage-4 spans per
system and would drop 276 and 8 (most are below the display bar and never became pictures). Picture spans removed: 5 in
ours, 6 without the gate (4 Crowd spans in `mv_protest_scene_movie`, 1 Siren span in `mv_tornado_scene`, + 1 Water span
in `waves_herzliya` without the gate); no whole picture removed, no span unmapped.

| system | run | hits / 36 | wrong (visible / cross / phantom) | dup | F1 | viewer cost | coverage |
|---|---|---|---|---|---|---|---|
| ours (proposed) | scored | 14 | 24 (6 / 11 / 7) | 0 | 0.378 | 2.78 | 0.300 |
| ours (proposed) | J2 | **13** | **25** (6 / 12 / 7) | 0 | 0.351 | 2.90 | 0.262 |
| without gate (blind_a2i) | scored | 17 | 50 (17 / 21 / 12) | 3 | 0.330 | 3.59 | 0.363 |
| without gate (blind_a2i) | J2 | 16 | 50 (16 / 22 / 12) | 3 | 0.314 | 3.67 | 0.324 |

Why: in `mv_protest_scene_movie` the needed Crowd sound lasts the whole clip (0–20 s). Its BEATs spans do not stand out
from their surroundings (the crowd is there before and after each span), so J2 drops them; only a late Crowd piece at
18.9 s is left, which misses the onset and counts as a cross-trigger. So one hit is lost and one wrong picture is added.
The Siren picture in `mv_tornado_scene` is only shortened. **DEV rule: hits drop (14 → 13) and wrong pictures do not
drop (24 → 25), so the DEV part of the ship rule fails: J2 is not shipped, whatever the fresh set shows.** The fresh-set
check (A) is still scored once, as registered, when its caches are complete.
