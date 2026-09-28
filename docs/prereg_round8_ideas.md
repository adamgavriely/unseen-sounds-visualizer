# Pre-registration: detector round 8 — new ideas for sounds a detector hears but the stack drops

*Written 2026-09-28, before any cost of any round-8 cell was computed, before any round-8 cache existed, and before
`benchmark/detector_audit.json` was read. Detection only, on the AudioSet-Strong fit set (the 280, `R.use_set("calib")`)
and the held-out set (the 415, `"heldout"`). DEV, TEST and the new "fresh" set are not touched. Harness:
`benchmark/detector_round8.py`, results in `benchmark/detector_round8.json`, jobs `slurm/job_round8_*.sh`.*

## Why
The goal is sounds that some detector HEARS but the stack then DROPS. On DEV only 4 of 36 needed sounds are truly unheard;
most "never detected" sounds had FlexSED 0.42–0.92 while the FlexSED bar is 0.8. Every audio corroborator so far failed to
tell a true 0.5-level sound from a 0.5-level phantom (each rescued event came with about 2.4 false spans; break-even is
2.0). So the new cells use other information: time structure (I2, I7, I1), ontology (I4), per-clip normalisation (I3),
scale and band (I5, I8), and the video (I6).

## Fixed parts (as rounds 4–5)
- **Baseline = the shipped stack:** BEATs spans (AED 0.175, hysteresis 1.0, display 0.35) + FlexSED spans at 0.8 (low 0.8)
  + twin rule (same canonical family within 1 s → the BEATs span takes the earlier start) + FlexSED clip veto 0.3 (all
  spans) + BEATs self-veto on FlexSED-only spans (BEATs clip-max of the family ≥ b = 0.1218), PANNs off, min span 0.5 s.
  Built by `detector_round5.run(slot 0, AED 0.175, display 0.35, b = 0.1218)`.
- **Gate 0 (stop if it fails):** the baseline reproduces 280: C-overlap 3.036, C-onset 3.850, recall 51.3 %, 207 false
  spans (4.44/min), 564 shown; 415: 1.928 / 2.207 / 49.7 % / 228 false spans (3.30/min) / 672 shown; and the round-8 stack
  with every option off equals round 5's stack **span for span** on every clip of both sets.
- **Cost C** (amendment 24, `detector_round2.clip_cost`): per clip 4 × missed consequential events + 2 × false spans;
  C-overlap (primary) and C-onset (span start in [onset − 0.5, onset + 1.0] s). Clip list = `detector_round2.usable()`.
- **Bootstrap:** paired over clips, 2000 draws, seed 0 (the same draws for every cell); 95 % CI = 2.5 / 97.5 percentiles.
  One-sided p = share of draws whose mean ΔC ≥ 0 (Δ = cell − baseline).

## Reported per cell
C-overlap, C-onset (with ΔC and CI), recall (overlap), onset recall, false spans and false/min, shown spans, and the
**heard-but-dropped count**: among the baseline's missed consequential events (C-overlap) whose same-family FlexSED score
(`E._same` on FlexSED's labels) reaches 0.4–0.8 in [start − 1 s, end + 1 s] (the audit's flag ii), how many the cell hits
(recovered); next to it the **new false spans**, gross (cell false spans with no same-label baseline false span overlapping
them) and net (cell false − baseline false).

## The cells (all values fixed here; no grids except the stated one-parameter refits)
A cell changes only what is written; everything else is the baseline. "In-sample" = has a choice made on the 280, so only
its 415 number counts.

**I2 — repetition-conditioned FlexSED bar (6 cells: windows W ∈ {±5 s, ±10 s, whole clip} × bars β ∈ {0.5, 0.4};
margin 0.3 fixed).** Anchors = the spans of family f in the **baseline's final shown output** (any origin). FlexSED spans
of f are re-extracted at bar β (low β, min 0.5 s). A β-span is admitted iff (1) it overlaps [a.start − W, a.end + W] for
some anchor a of f (both directions: a weak start before the strong detection counts; "whole clip" = any anchor of f in
the clip), and (2) its peak minus the median of f's FlexSED score over all frames of the clip is ≥ 0.3. An admitted
β-span that contains a 0.8 span of f **replaces** that 0.8 span (so its weak start becomes the onset); an admitted β-span
with peak < 0.8 is added. Admitted spans then go through the unchanged twin rule (a BEATs twin takes the earlier start)
and both vetoes. Families with no anchor are unchanged. *Selection (Adam, 2026-09-28, written before any I2 cost):* per
bar β, among the three windows with C-overlap AND C-onset below the baseline on the 280, the one with the lowest
C-overlap (tie → lower C-onset → smaller window) goes to the 415; if no window is eligible, that bar sends nothing. So
at most 2 I2 cells enter the 415 and the Holm set. All 6 cells' 280 numbers are reported. In-sample (the window choice).

**I3 — per-clip normalised BEATs.** z = logit(p) − median over the 527 classes of logit(p) in the same frame (p clipped
to [1e-6, 1 − 1e-6] first; the cache is float16). The BEATs span source becomes q = sigmoid(z − d + logit(0.35)), so
q ≥ 0.35 ⇔ z ≥ d, and the AED low bar 0.175 sits at the same logit gap as shipped. q replaces BEATs only as a **span
source** (extraction, display bar, twin rule); the self-veto still reads the raw BEATs clip-max with b = 0.1218 (that bar
was fitted on raw p; a second refit is not allowed). The FlexSED clip veto is unchanged. **Refit of d:** grid 0.00, 0.05,
…, 20.00; d = the value whose full stack gives the false-span count on the 280 closest to the baseline's 207 (tie →
higher d); if the choice is on a grid edge, the grid is extended by 10 in that direction (disclosed). In-sample (d).

**I4 — ontology parent emission (conditional).** For each BEATs label P with ≥ 2 direct children among BEATs' 527 labels
(`src/labels._parents()`), parent score s_P(t) = 1 − Π_children (1 − p_c(t)). Parent spans are extracted from s_P at the
shipped bars (AED 0.175, display 0.35), labelled P, and emitted only where no child of P and not P itself has a raw BEATs
frame ≥ 0.35 inside the span. They are BEATs-origin spans (twin rule, FlexSED clip veto on canonical(P), default pass if
FlexSED has no such family). **Run only if** on the 280 the audit's false-span bin a (related family overlaps) is ≥ 20 %
of false spans OR its miss bin v (heard under a related name) is ≥ 20 % of misses; otherwise recorded "not run" with the
two numbers.

**I6 — VLM scene prior (wild card, no fitting).** Qwen/Qwen3.8-27B (`config.VLM_MODEL` of the shipped preset, thinking
off, greedy, `reason._ask`, max 1024 new tokens) sees 4 frames of the clip's mp4 (`data/input/audioset_calib|heldout`),
at t = duration × (i + 0.5) / 4, 640 px wide (`stage2 vlm._frames`), with this one prompt:
> These are 4 frames from one short video, in time order. Below is a numbered list of sound sources. Which of them could
> plausibly be heard in this place, on screen or off screen? Include every source that is plausible for this kind of
> place, not only what you can see. Answer with the numbers only, separated by commas. If none is plausible, answer: none.
followed by the 215 families of `benchmark/gold/depictable_vocab.json` as "1. Air brake" … "215. …" in file order.
Parse: every integer 1–215 in the answer; a per-clip "cut" flag is stored when the budget is used up. Cell: FlexSED bar
0.5 (low 0.5) for the listed families, 0.8 for the others; same twin rule and vetoes.

**I7 — FlexSED local-contrast veto (precision, conditional).** A FlexSED-only span is dropped if the mean of its family's
FlexSED score inside the span minus the mean over the flank frames ([start − 3, start) and (end, end + 3], clipped to the
clip) is < 0.2; a span with no flank frames is kept. Applied after the baseline's vetoes. **Run only if** on the 280 the
audit's phantom bin c is ≥ 33 % of false spans AND its FlexSED-only part is ≥ 10 % of all false spans (I7 can touch
only FlexSED-only spans); otherwise "not run" with the numbers.

**I1 — onset fixes (timing-only; three separate cells).**
- **I1-i:** in the twin rule, a BEATs span with ≥ 1 FlexSED twin takes the start of its earliest FlexSED twin (instead of
  the earlier of the two starts).
- **I1-ii:** a shown span whose family has a FlexSED column is split at a FlexSED dip: a run of frames with score < 0.4
  (bar/2) lasting ≥ 0.3 s, with a frame ≥ 0.8 in the span before it and a frame ≥ 0.8 in the span after it. The span
  becomes [start, dip start] and [dip end, end] (dip end = first frame after the run); parts shorter than 0.5 s are
  dropped; several dips give several parts. Same label and confidence.
- **I1-iii:** m = median signed onset error of the baseline on the 280 (span start − gold start; consequential salient
  events hit under C-overlap; the matched span = largest overlap, tie → earlier start). Every shown span's start becomes
  clip(start − m, 0, end − 0.25). m is written to the json before the cell's cost is computed. In-sample (m).

**I5 — multi-scale BEATs.** BEATs also on 1-s windows, hop 0.25, lead pad 0.75 s (reflect), stamped 0.25 s before the
window end (the same 25 % of the window as the shipped 0.5 s of 2 s); 1-s frames are matched to the 2-s cache times
(|Δt| < 1 ms; a 2-s frame with no match keeps p2). **1-s bar d1, set a priori:** BEATs-1s ALONE (AED d1/2, display d1,
salient spans) gives on the 280 the false-span count closest to BEATs-2s ALONE at 0.175/0.35 (grid 0.35, 0.36, …, 1.00;
tie → higher d1). Then p1' = min(1, p1 × 0.35 / d1) and the BEATs score becomes max(p2, p1') **everywhere** (spans, twin
rule, self-veto), the rest unchanged. In-sample (d1, fitted on false spans of the tagger alone, not on C).

**I8 — band-limited views.** Audio (16 kHz mono, as the caches) filtered with an 8th-order Butterworth (scipy
`sosfiltfilt`): low-pass < 300 Hz and high-pass > 4 kHz, saved as `<id>__lp.wav` / `<id>__hp.wav`. BEATs (same 2-s
windows; times asserted equal to the cache) and FlexSED (same runner, 215 families; frame counts asserted equal) on both.
BEATs score = max over the 3 views and FlexSED score = max over the 3 views, **everywhere** (spans, twin rule, both
vetoes), same bars.

## Decision rules
- **Recall/precision cells (I2 ×≤2, I3, I4, I5, I6, I7, I8):** picked on the 280 iff C-overlap < baseline AND C-onset <
  baseline. Each picked cell goes to the 415 alone, with everything frozen; it **passes** iff the upper 95 % CI of
  ΔC-overlap < 0. Holm across all cells sent to the 415 (one-sided p above, family α = 0.025, step-down) is also reported.
- **Timing cells (I1-i, I1-ii, I1-iii):** go to the 415 iff on the 280 C-onset < baseline and C-overlap ≤ baseline +
  0.05. On the 415: primary = upper 95 % CI of ΔC-onset < 0; guard = upper 95 % CI of ΔC-overlap < +0.05. Holm across
  the I1 cells sent (p on ΔC-onset) in a separate family.
- **Combination:** if ≥ 1 cell passes, the combination of the passing cells is written here as a dated amendment (order
  of application and how overlapping rules merge) before it is scored, and tested once on the 415. The fresh set is a
  later, separate step.
- The audit gates for I4 and I7 are read once `detector_audit.json` is complete, and recorded here with the numbers
  before those cells are scored.

## Amendment 1 (2026-09-28, written after the audit finished and before any round-8 cost or cache existed)
**Audit gates (the 280, `benchmark/detector_audit.json`, read once it was complete).**
- I4: false-span bin a (related family overlaps) = 59 / 207 = **28.5 %** ≥ 20 % → **I4 runs.** (Miss bin v = 0 / 109.)
- I7: phantom bin c = 137 / 207 = **66.2 %** ≥ 33 % (holds); its FlexSED-only part = 15 / 207 = **7.2 %**, below the
  10 % sub-condition I added above. The task's own rule was only "run if phantoms are a large share", which holds, and
  the lead asked to run it; so **I7 runs**, with this disclosed: it can touch at most the few FlexSED-only spans
  (15 phantoms on the 280), so a large effect is not expected.

**Changes asked by the lead (from the audit), fixed here before their costs:**
1. **Secondary scoring row.** The primary stays as pre-registered (`clip_cost` with `config.LABEL_FILTER = "lists"`, as
   rounds 2–7, so rounds stay comparable). Every cell and the baseline are ALSO scored with `LABEL_FILTER = "depictable"`
   (the shipped system's filter; about 20 % of the false spans are labels it never draws). Reported, not used for any
   decision.
2. **I9 — ontology-matched vetoes.** The FlexSED clip veto (0.3) and the BEATs self-veto (b = 0.1218) read the clip-max
   over every label of the other detector that matches the span's label by `E._same` (same label or canonical family,
   or ancestor/descendant in the AudioSet ontology), instead of the exact canonical name; a span with no matching label
   passes (as now). Nothing else changes. Fixed a priori, no fitting.
3. **I10 — speech/music guard on BEATs-only spans.** A shown BEATs-origin span with no FlexSED twin whose BEATs
   "Speech" or "Music" score reaches ≥ 0.3 inside the span (frames with t in [start, end]; the nearest frame if none,
   as the audit's `peak`) is kept only if FlexSED's max over its matching labels (`E._same`) inside the span is ≥ 0.5;
   if FlexSED has no matching label, it is kept only if the span's BEATs peak ≥ 0.5. Applied after the baseline's
   vetoes. Fixed a priori, no fitting.
4. I9 and I10 are recall/precision cells: same 280 pick rule, same 415 rule, and both are in the Holm set.
5. **Extra count per cell:** true spans removed = baseline shown spans that match a gold event of their family (not false)
   with no same-label cell span overlapping them; false spans removed = the same for the baseline's false spans. Reported
   next to the recovered heard-but-dropped events and the new false spans (gross, net).

## Amendment 2 (2026-09-28, before any I6 cost; only VLM answers existed)
Try 1 of the I6 prompt did not follow "numbers only": Qwen3.8-27B wrote a sentence and a bullet list with explanations,
and 6 of the first 10 answers (the 280) were cut at the 1024-token budget, so their lists were incomplete. The job was
stopped after 10 clips (answers kept in `benchmark/audioset_calib_windows/round8_vlm_try1_verbose.json`, not used). **Fix:**
one format line is appended after the numbered list: "Reply with ONE line that contains only the numbers, separated by
commas (for example: 3, 17, 102). Do not write any words, names or explanations." Everything else (model, frames, greedy,
thinking off, budget 1024, parse, the cell rule) is unchanged. The cut rate of the new answers is reported.

## Result (2026-09-28; `benchmark/detector_round8.json`)
**Nothing passes on the 415. No combination is scored. The shipped stack stays.**
Jobs: 31329927 views, 31329928 gate 0 (both sets), 31329930 BEATs 1-s + view caches, 31329931/31330003-5 FlexSED on the
views, 31330001/2 VLM (31329929 = try 1, stopped). Scoring from the caches (the same code) ran on a local copy of the
cluster caches; gate 0 passed there too (span for span, same numbers). Gate 0 on the cluster: both sets reproduce round 5
span for span; 280 3.036 / 3.850 / 51.3 % / 207 false (4.44/min) / 564 shown; 415 1.928 / 2.207 / 49.7 % / 228 false.
**Refits (the 280, written before the cells' costs):** I3 d = 9.15 (209 false spans vs 207); I5 d1 = 0.35 (1-s alone 294
false spans vs 299 for 2-s alone; this is the lower grid edge, which the prereg did not extend for I5 — disclosed); I1-iii
m = −1.32 s (baseline spans start a median 1.32 s early, 115 hits). VLM answers after amendment 2: median 7 families per
clip (0–165 on the 280, 0–215 on the 415), cut 5 / 280 and 4 / 415, empty 3 / 4. 1-s frames: at most 1 tail frame per
clip without a 2-s partner.

*The 280 (Δ = cell − baseline, clip bootstrap 95 % CI; HBD = heard-but-dropped events recovered, of 61; new false =
gross / net; removed = false / true spans; last column = C-overlap with the "depictable" filter, baseline 2.664):*

| cell | C-overlap (ΔC) | C-onset (ΔC) | recall % | onset rec % | false/min | HBD | new false | removed F / T | dep. C-ov |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.036 | 3.850 | 51.3 | 25.9 | 4.44 | — | — | — | 2.664 |
| I2 β 0.5 (±5 s = ±10 s = clip) | 3.050 (+0.014 [−0.079, +0.086]) | 3.879 (+0.029) | 52.2 | 26.3 | 4.56 | 0 | 8 / 6 | 1 / 0 | 2.679 |
| I2 β 0.4 (±5 s = ±10 s = clip) | 3.014 (−0.021 [−0.136, +0.064]) | 3.886 (+0.036) | 53.1 | 25.9 | 4.54 | 2 | 7 / 5 | 1 / 0 | 2.643 |
| I3 normalised BEATs | 3.393 (+0.357 [−0.150, +1.143]) | 3.836 (−0.014) | 40.6 | 26.8 | 4.48 | 0 | 65 / 2 | 65 / 66 | 3.086 |
| **I4 parent emission** | **2.921 (−0.114 [−0.364, +0.029])** | **3.807 (−0.043)** | 56.2 | 28.6 | 4.56 | 7 | 6 / 6 | 0 / 1 | 2.543 |
| I5 multi-scale BEATs | 3.250 (+0.214 [+0.050, +0.400]) | 4.107 (+0.257) | 53.6 | 26.8 | 5.29 | 3 | 53 / 40 | 3 / 6 | 2.757 |
| **I6 VLM scene prior** | **2.929 (−0.107 [−0.307, +0.050])** | **3.843 (−0.007)** | 55.4 | 26.8 | 4.50 | 5 | 11 / 3 | 4 / 1 | 2.557 |
| **I7 local-contrast veto** | **2.971 (−0.064 [−0.121, −0.014])** | **3.786 (−0.064)** | 51.3 | 25.9 | 4.24 | 0 | 0 / −9 | 9 / 8 | 2.600 |
| I8 band views | 6.807 (+3.771 [+3.250, +4.321]) | 7.636 (+3.786) | 50.9 | 25.0 | 15.71 | 0 | 531 / 526 | 2 / 1 | 5.943 |
| I9 ontology vetoes | 3.207 (+0.171 [−0.093, +0.393]) | 4.079 (+0.229) | 56.2 | 29.0 | 5.42 | 11 | 46 / 46 | 0 / 0 | 2.836 |
| I10 speech/music guard | 3.479 (+0.443 [−0.264, +1.507]) | 3.807 (−0.043) | 28.6 | 18.3 | 3.58 | 0 | 0 / −40 | 40 / 52 | 3.171 |
| I1-i FlexSED start | 3.050 (+0.014) | 3.879 (+0.029 [+0.000, +0.064]) | 51.3 | 25.4 | 4.48 | 0 | 4 / 2 | 2 / 2 | 2.679 |
| **I1-ii split at dips** | 3.036 (+0.000) | **3.793 (−0.057 [−0.171, +0.000])** | 51.3 | 27.7 | 4.44 | 0 | 0 / 0 | 0 / 0 | 2.664 |
| I1-iii shift −1.32 s | 3.571 (+0.536) | 4.286 (+0.436 [+0.264, +0.643]) | 39.3 | 17.0 | 4.89 | 0 | 21 / 21 | 0 / 0 | 3.200 |

I2: the three windows gave identical spans for each bar (on 10-s clips every admitted burst lay within ±5 s of an anchor),
and neither bar is eligible (C-onset rises), so no I2 cell goes to the 415. I2 recovers few heard-but-dropped events
(0 and 2 of 61) because most of them are in clips where the family is never shown at all, so there is no anchor.
I8: the views make BEATs hear insects/crickets in the high band and heartbeat in the low band (598 BEATs-origin false
spans, top Insect 86, Heart sounds 61, Cricket 41). **Picked (both C below baseline): I4, I6, I7; timing: I1-ii.**

*The 415 (frozen; pass = upper CI of ΔC-overlap < 0; I1-ii: upper CI of ΔC-onset < 0 and of ΔC-overlap < +0.05):*

| cell | C-overlap (ΔC) | C-onset (ΔC) | recall % | onset rec % | false/min | HBD (of 51) | new false | removed F / T | one-sided p | result |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.928 | 2.207 | 49.7 | 32.7 | 3.30 | — | — | — | — | — |
| I4 | 1.918 (−0.010 [−0.111, +0.072]) | 2.236 (+0.029) | 54.4 | 35.1 | 3.50 | 8 | 14 / 14 | 0 / 0 | 0.48 | fail |
| I6 | 1.995 (+0.067 [−0.048, +0.164]) | 2.284 (+0.077) | 52.6 | 35.1 | 3.64 | 5 | 28 / 24 | 0 / 2 | 0.90 | fail |
| I7 | 1.889 (−0.039 [−0.087, +0.005]) | 2.159 (−0.048 [−0.092, −0.010]) | 49.1 | 32.7 | 3.15 | 0 | 0 / −10 | 10 / 12 | 0.061 | fail (close) |
| I1-ii | 1.947 (+0.019 [+0.000, +0.048]) | 2.198 (−0.010 [−0.058, +0.039]) | 49.7 | 34.5 | 3.35 | 0 | 0 / +4 | 0 / 0 | 0.46 (onset) | fail |

Holm (α 0.025 one-sided): recall/precision family I7 p 0.061 (threshold 0.0083), I4 0.48, I6 0.90 — none rejected;
timing family I1-ii p 0.46 — not rejected. Strata (ΔC-overlap): I7 complex −0.056 [−0.127, −0.008], random −0.012
[−0.086, +0.061]; I4 complex +0.032, random −0.074; I6 complex +0.087 [+0.008, +0.167]. The "depictable" secondary
row gives the same picture (I7 −0.039, I4 −0.019, I6 +0.067, I1-ii +0.019).

**Reading.** (1) Recall can be raised: I4 (+4.7 / +4.9 points on the 280 / 415) and I6 (+4.1 / +2.9) recover real sounds
that are heard but dropped (7–8 and 5 of the 0.4–0.8 pool), but each brings more new false spans than events
(I4 on the 415: +8 events = −32 cost, +14 false spans = +28 cost, net −4 over 415 clips = −0.010; I6: +5 events,
+24 false spans), so C does not move beyond noise. (2) The only precision idea that holds in direction on both sets is
I7 (−0.064 on the 280, −0.039 on the 415, onset CI below 0), but it also drops 12 true spans for 10 false ones on the 415
and its C-overlap CI touches 0. (3) The 0.5-level phantom problem is not solved by time structure (I2), per-clip
normalisation (I3), scale (I5) or bands (I8); I9 and I10 as fixed a priori make things worse.
