# Pre-registration: detector round 10 — a discriminator for "heard but dropped" sounds (FlexSED 0.4–0.8)

*Written 2026-09-28, before any round-10 cache existed and before any round-10 cost was computed. Detection only, on the
AudioSet-Strong fit set (the 280, `R.use_set("calib")`) and the held-out set (the 415, `"heldout"`). DEV, the fresh set
and TEST are not touched by this part (DEV and fresh only through the conditional steps at the end; TEST never). Harness:
`benchmark/detector_round10.py` (reuses `detector_round8` for the stack, gate 0, scoring, bootstrap and Holm), results in
`benchmark/detector_round10.json`, jobs `slurm/job_round10_*.sh`.*

## Why
The audit (`docs/detector_audit_2026-09-28.md`) found that the largest group of missed needed sounds is sounds that FlexSED
scores 0.4–0.8, below its bar 0.8: 40 / 109 misses on the 280 and 36 / 86 on the 415 (flag ii without the order: 61 / 51
events, the "band group"). Every earlier way to let them in (a lower bar, corroboration by BEATs / PANNs / PSED / CLAP /
Qwen-Omni, repeats, a scene prior, parent emission) also let in about 2 or more false spans per rescued event. Break-even
under C is 2 false spans per rescued event (a miss costs 4, a false span 2). So round 10 does not add another detector
vote at the same level; it tests **discriminators**: checks that a real 0.5-level sound should pass and a 0.5-level
phantom should not.

## Fixed parts (as rounds 8–9)
- **Baseline = the shipped stack**, exactly round 8's gate 0: `detector_round8.stack8` with no option (= round 5's stack
  span for span). BEATs spans (AED 0.175, hysteresis 1.0, display 0.35) ∪ FlexSED spans at 0.8 (low 0.8) by the twin rule,
  FlexSED clip veto 0.3, BEATs self-veto b = 0.1218 on FlexSED-only spans, PANNs off, min span 0.5 s. Round 8's I7 and
  round 9's J1/J2 are **off** (not shipped).
- **Gate 0 (stop if it fails):** the baseline reproduces the 280: C-overlap 3.036, C-onset 3.850, recall 51.3 %, 207 false
  spans, 564 shown; the 415: 1.928 / 2.207 / 49.7 % / 228 / 672 (`detector_round8.gate_set`, span for span vs round 5).
- **Gate 0b:** round 10's own stack (`stack10`) with every option off equals `stack8` span for span on every clip; with
  I4 unfiltered it equals `stack8(i4)`, and with I6 unfiltered it equals `stack8(i6)` span for span on every 280 clip, and
  their 280 costs equal round 8's (I4 2.921 / 3.807, I6 2.929 / 3.843).
- **Cost C**, clip lists, the paired clip bootstrap (2000 draws, seed 0), the one-sided p (share of draws with mean ΔC ≥ 0),
  the primary filter (`LABEL_FILTER = "lists"`) and the secondary "depictable" row: all as in round 8.

## Band candidates (label-free; the object every R1–R5 cell filters)
Per clip, per FlexSED column (the 215 families of `benchmark/gold/depictable_vocab.json`): `_extract_events` at bar 0.4,
low 0.4, min duration 0.5 s. A span is a **band candidate** iff
1. its peak (confidence) is **< 0.8** (a 0.4-span that contains a ≥ 0.8 frame is the shipped span, only wider: not a candidate);
2. it is **FlexSED-only**: no BEATs span of the same canonical family (BEATs spans at AED 0.175, before any veto, starts not
   yet extended) lies within 1 s (the shipped twin rule's test `b.start − 1 ≤ e.end and e.start − 1 ≤ b.end`).
   **A candidate with a BEATs twin is discarded, not merged** — so no cell moves the onset of an existing span, and the
   R0–R5 cells only add spans;
3. it passes the shipped vetoes for FlexSED-only spans: the BEATs self-veto (BEATs clip-max of the family ≥ 0.1218; a
   family BEATs does not have passes). The FlexSED clip veto (0.3) and the display floor (0.35) pass automatically (peak ≥ 0.4).
Admitted candidates are added to the baseline's shown spans with their label, start, end and confidence. The list of
candidates per clip (`round10_cands.json` in each set's windows folder) is computed from the caches only, no gold read.

"In the span" = frames with t in [start, end]; if no frame lands inside, the frame nearest to the span's middle (as
`detector_round8.peak_in`). "Score of the family" = the column(s) whose canonical family equals the candidate's.

## The cells (values fixed here; no grids, no fitting)
- **R0 — reference, all band candidates admitted** (no discriminator). Not in the pick or Holm set. It shows what the band
  costs without a filter, and its recovered count is the **ceiling** of R1–R5 on the band group (a band event with no
  candidate span cannot be rescued by any R1–R5 cell). If any cell goes to the 415, R0 is scored there too, as a reference
  only (no test, no Holm, never shipped).
- **R1 — two open-vocabulary detectors agree.** Admit iff DASM (round-6 cache `<WIN>/dasm_cache`, same 215 queries, 50 fps)
  scores the same family **≥ 0.359375** at some frame with t in [start − 0.5, end + 0.5] (nearest frame if none).
  *Why 0.359375:* DASM's round-6 span bar g = 0.575 was matched to FlexSED's bar 0.8 on the 280 by span count; R1 asks DASM
  for the same relative level as the band's admit level 0.5 on FlexSED's scale: 0.575 × 0.5 / 0.8 = 0.359375.
  Check before use (round-6 c4): each set has a DASM cache for every clip and its labels equal FlexSED's.
- **R2 — query-paraphrase agreement.** FlexSED is asked each family in two more ways (below). Admit iff **both**
  paraphrases score **≥ 0.5** in the span.
- **R3 — perturbation stability (test-time augmentation).** FlexSED is re-run (the 215 family queries, its own template) on
  3 mild versions of the clip's audio: pitch +1 semitone, pitch −1 semitone, and a 0.95× time-stretch (below). Admit iff
  the family scores **≥ 0.5** in the span in **≥ 2 of the 3** versions (times mapped back to the original clip).
- **R4 = R1 AND R2.  R5 = R1 OR R2.**
- **R6 — round 8's I4 (ontology parent emission), confirmed** (added at the lead's request, from Adam, before any round-10
  number). I4's emitted parent spans (label P, a BEATs label; `detector_round8.parent_spans`, unchanged) are kept only if
  R1 OR R2 confirms them; an unconfirmed parent span is removed before the twin rule (it never exists). Because P is
  usually not one of the 215 families: R1 uses the DASM columns whose family matches P by `E._same` (same label or family,
  or ancestor/descendant in the AudioSet ontology), max over them, same bar 0.359375 and ±0.5 s; R2 passes iff **some**
  matching family has both paraphrases ≥ 0.5 in the span. No matching column → that test fails. Everything else is I4.
- **R7 — round 8's I6 (VLM scene prior, FlexSED bar 0.5 for the families Qwen3.8-27B listed), confirmed** (same origin).
  A "lowered-bar span" = an I6 FlexSED span of a listed family with peak < 0.8 (it exists only because of the lower bar).
  It is kept only if R1 OR R2 confirms it (same column rules as R1 / R2 above, the family's own column); an unconfirmed one
  is removed before the twin rule. I6 spans with peak ≥ 0.8 are untouched. Everything else is I6 (round-8 VLM answers,
  `round8_vlm.json`, unchanged).

### The paraphrases (R2)
Written once, by a fixed template, to **`benchmark/round10_paraphrases.json`** before any paraphrase score exists:
- the family's AudioSet entry = the first entry of `src/audioset_ontology.json` (file order) whose `canonical(name)` equals
  the family (this finds an entry for all 215; e.g. "Gunshot" → "Gunshot, gunfire");
- **P1 (definition)** = the first sentence of that entry's official `description` (text up to the first ". " that does
  not follow "e.g", "i.e" or "etc"; a final "." is added if missing), given to FlexSED's text encoder **as is**;
- **P2 (name form)** = the family name split at ", ", a leading "and " / "or " removed from each part, the parts joined
  with " or ", the first letter upper-case, + " can be heard in this recording." (e.g. "Baby cry or infant cry can be
  heard in this recording.");
- FlexSED's own `run_inference` wraps every query as "The sound of {x.capitalize()}"; P1 and P2 are full sentences, so they
  are encoded **without** that wrapper, by a copy of `api.run_inference` (lines 61–94 of `~/FlexSED/api.py`) that takes the
  text as given. Everything else is FlexSED's: CLAP text tower, 16-kHz audio (the same ffmpeg wav as the cache), 10-s
  chunks with the remainder zero-padded to 1 s (as `flexsed_run.py`), peak normalisation, sigmoid, 25 fps.
- **Checks before any paraphrase or view score is used (a failure stops the round):** (c1) the copy, given the 215
  families wrapped exactly as `run_inference` does, reproduces the cached FlexSED scores of the first 2 clips of each GPU
  shard (max abs diff < 5e-3; the cache is float16 and was made on another card class); (c2) query independence: one
  paraphrase alone vs inside its batch, on one clip, max abs diff < 1e-3.

### The perturbed audio (R3)
From the same 16-kHz mono audio as the caches (`ffmpeg` of the clip's mp4, as round 8's views), with librosa 0.11:
`pitch_shift(n_steps=+1)` (`__p1u`), `pitch_shift(n_steps=−1)` (`__p1d`), and `time_stretch(rate=1/0.95)` (`__s95`: the
duration becomes **0.95×**, so the clip stays within FlexSED's 10-s chunk and no padded tail is added — the artifact round 6
had to fix for DASM). Clipped to [−1, 1], written as 16-bit wavs. FlexSED frame k of a version has time k / 25 s; for
`__s95` it is mapped back to the original clip as t = (k / 25) / 0.95. Checks: the pitch versions have exactly the cached
frame count; `__s95` has 0.95 × that count ± 2 frames. Only clips that have ≥ 1 band candidate are perturbed and scored (R3
is only asked about band candidates). No gain change (FlexSED normalises the peak of each chunk anyway).

## Reported per cell
C-overlap, C-onset (ΔC with 95 % CI and one-sided p), recall, onset recall, false/min, shown spans; **heard-but-dropped
events rescued** (of the band group = the baseline's missed needed events with same-family FlexSED 0.4–0.8 within ±1 s;
round 8's `hbd_flags`) and **all rescued events**; **false spans added** (gross: cell false spans with no same-label
baseline false span overlapping; net); **true / false spans removed** (0 by construction for R0–R5); the **"depictable"**
row; and the discriminator's own counts: candidates (or parent / lowered-bar spans) considered, admitted, admitted true
(overlap a gold event of their family), admitted false. On the 415 also the complex / random strata.

## Decision rules
- **Picked on the 280** iff C-overlap < baseline AND C-onset < baseline (R1–R7; R0 is never picked).
- Each picked cell goes to the 415 alone, frozen; it **passes** iff the upper 95 % CI of ΔC-overlap < 0. **Holm** across
  all cells sent to the 415 (one-sided p, family α = 0.025, step-down) is reported with it; a cell counts as passed for the
  next steps only if it passes AND Holm rejects it.
- Before the 415 is scored, `docs/setup_audit_2026-09-28.md` (running) is read. **If it reports a timing / alignment bug,
  the 415 is not scored** and the round stops with a report.
- **No combination** of cells is scored in this round.

## Next steps, only for a cell that passes the 415 (conditional; details fixed now, finished as a dated amendment before computing)
1. **DEV check (DEV only, never TEST)**, like `benchmark/gold/j2_dev_check.py`, but for **added** spans: rebuild the cell's
   added stage-4 spans on each DEV clip from the pipeline's caches (BEATs recomputed with the shipped `infer_beats`, gate
   D0 as in j2's check; FlexSED from `data/work/flexsed_cache/<stem>.npz`; DASM / paraphrase / perturbation scores computed
   on the DEV clip's `audio.wav` with the same code). An added span only becomes a picture if the **stage-5 gate** would show
   it; the gate and picture logic are taken from the DEV check for added spans that another agent is writing
   (`benchmark/gold/dev_candidates_check.py`) — reused as is, its files not edited. Scored with the official
   `score_per_sound` (onset rule, defaults), before / after, for ours ("proposed"): hits, wrong pictures (visible / cross /
   phantom), duplicates, F1, viewer cost.
2. **Fresh set** (`R.use_set("fresh")`, 422 clips, `docs/prereg_fresh_confirm_set.md`), scored once, cell frozen, no refit;
   same gate (stack8 = round 5 span for span); primary: upper 95 % CI of ΔC-overlap < 0.
3. **Ship rule:** the cell goes into `use_shipped()` only if the fresh set passes AND on DEV hits do not drop AND DEV wrong
   pictures do not rise by more than 2 × the hits gained.

## What will not be done
No other bar, margin, window, paraphrase or perturbation; no per-family rule; no re-pick after the 415; no combination;
nothing on TEST; no edit of any running job's file.

## Clarification 1 (2026-09-28, after the caches were made, before any round-10 cost)
The cache check flagged 3 + 4 clips: their `__s95` version has 237 frames while the cached FlexSED count is 275. These are
the 10.007–10.01-s clips (round 6, clarification 2): FlexSED's cache holds 25 extra frames from the 10-ms remainder padded
to 1 s. The stretched version is 9.51 s long, so it has no padded tail. The check now compares `__s95` with 0.95 × the
frames that carry audio, **min(T, 250)**. Nothing else changes (for R3, a frame of `__s95` still maps to t = (k / 25) / 0.95).
Checks c1 (max diff 5e-5 to 2.5e-4 on 10 clips, all shards) and c2 (≤ 2e-7) passed.

## Result on the 280 (2026-09-28; `benchmark/detector_round10.json` → `fit`; jobs 31330543 prep, 31330544–48 GPU, 31330586 score)
Gate 0 passed (round 5's stack span for span; 3.036 / 3.850 / 51.3 % / 207 false / 564 shown). Gate 0b passed (`stack10` =
`stack8` span for span with no option, with I4 and with I6; I4 2.921 / 3.807 and I6 2.929 / 3.843 = round 8). Cache check:
0 missing, 0 misaligned on both sets; c1 max diff ≤ 2.5e-4, c2 ≤ 2e-7. Band candidates: 209 in 85 clips (the 415: 373 in
153 clips). Band group = 61 events.

Δ = cell − baseline, clip bootstrap 95 % CI. C-onset Δ equals C-overlap Δ for R0–R5 (the added spans hit or miss the same
events under both rules). "Rescued" = band-group events now hit (of 61) / all newly hit needed events. "Admitted" = spans the
filter let in, split into true (overlap a gold event of their family) and false.

| cell | C-overlap (ΔC, 95 % CI) | C-onset (ΔC) | recall | onset rec | false/min | rescued band / all | false spans added (gross / net) | true / false spans removed | admitted (true / false) of considered | depictable ΔC-overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.036 | 3.850 | 51.3 % | 25.9 % | 4.44 | — | — | — | — | (2.664) |
| R0 all candidates (reference) | 3.814 (+0.779 [+0.464, +1.064]) | 4.629 (+0.779) | 55.4 % | 29.9 % | 7.16 | 9 / 9 | 127 / 127 | 0 / 0 | 209 (82 / 127) of 209 | +0.779 |
| R1 DASM agrees | 3.021 (−0.014 [−0.221, +0.129]) | 3.836 (−0.014) | 54.5 % | 29.0 % | 4.69 | 7 / 7 | 12 / 12 | 0 / 0 | 32 (20 / 12) of 209 | −0.014 |
| R2 paraphrases agree | 3.029 (−0.007 [−0.043, +0.021]) | 3.843 (−0.007) | 51.8 % | 26.3 % | 4.46 | 1 / 1 | 1 / 1 | 0 / 0 | 8 (7 / 1) of 209 | −0.007 |
| R3 stable under perturbation | 3.207 (+0.171 [−0.064, +0.350]) | 4.021 (+0.171) | 54.0 % | 28.6 % | 5.21 | 6 / 6 | 36 / 36 | 0 / 0 | 71 (35 / 36) of 209 | +0.171 |
| R4 R1 AND R2 | 3.029 (−0.007 [−0.043, +0.021]) | 3.843 (−0.007) | 51.8 % | 26.3 % | 4.46 | 1 / 1 | 1 / 1 | 0 / 0 | 3 (2 / 1) of 209 | −0.007 |
| R5 R1 OR R2 | 3.021 (−0.014 [−0.221, +0.129]) | 3.836 (−0.014) | 54.5 % | 29.0 % | 4.69 | 7 / 7 | 12 / 12 | 0 / 0 | 37 (25 / 12) of 209 | −0.014 |
| R6 I4 confirmed | 2.914 (−0.121 [−0.364, +0.007]) | 3.800 (−0.050 [−0.121, +0.007]) | 55.8 % | 28.1 % | 4.50 | 7 / 10 | 3 / 3 | 1 / 0 | 40 (28 / 5 scored) of 105 parent spans | −0.121 |
| R7 I6 confirmed | 2.886 (−0.150 [−0.343, −0.007]) | 3.800 (−0.050 [−0.171, +0.043]) | 54.9 % | 26.3 % | 4.33 | 4 / 8 | 3 / −5 | 1 / 4 | 92 (76 / 16) of 172 lowered-bar spans | −0.150 |

For comparison, round 8 unfiltered on the 280: I4 −0.114 (6 new false, rescued 7 band), I6 −0.107 (11 new false, 5 band).

**Picked (both C below baseline): R1, R2, R4, R5, R6, R7.** R3 is not picked (C rises). Not yet sent to the 415 (see the note
below).

What the 280 says (descriptive):
- **The ceiling is low.** Only 9 of the 61 band-group events are reachable by a band candidate (R0 rescues 9). The other
  52 are lost to the candidate rules; possible reasons (not counted here): a BEATs span of the same family within 1 s
  (rule 2 discards the candidate), a FlexSED 0.4 span that also holds a ≥ 0.8 frame (so it is the shipped span, which
  misses the event by timing), or the self-veto.
- **R1 (DASM agreement) is a real discriminator:** it keeps 7 of the 9 reachable rescues and cuts the added false spans from
  127 to 12 (1.7 false spans per rescue, below break-even 2; unfiltered R0: 14 per rescue). But with only 7 rescues the net
  gain is tiny (−0.014).
- R2 (paraphrases) is too strict (8 of 209 admitted, 1 rescue). R3 (perturbation) barely filters (71 admitted, half false).
- R6 / R7 keep round 8's rescues with far fewer false spans than I4 / I6 alone (R7: net −5 false spans, 8 events rescued).

## Note 2026-09-28 — HOLD before the 415 (lead's instruction, written before any round-10 number on the 415)
The setup audit (`docs/setup_audit_2026-09-28.md`) found that the AudioSet cost C punishes any added span: silence beats
the shipped stack on the 415, each bark is counted as a separate needed event, and there is no time tolerance. A corrected
cost is being defined. **So the 415 is not scored for any cell, and DEV and fresh are not run.** The 280 step above is
reported exactly as registered. The picks (R1, R2, R4, R5, R6, R7) will be re-evaluated under the corrected cost before
anything is sent to the 415. All round-10 caches (DASM, paraphrase and perturbed-audio FlexSED scores, candidate lists)
exist for both sets, so re-scoring needs no GPU.
