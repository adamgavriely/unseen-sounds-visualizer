# Pre-registration: detector round 6 — DASM (Detect Any Sound Model) next to or in place of FlexSED

*Written 2026-09-28, before any DASM frame score was computed and before any cost. Detection only. TEST and DEV clips are
not touched. Harness: `benchmark/detector_round6.py`, results in `benchmark/detector_round6.json`.*

## The model
DASM (Cai et al., "Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries", ACM MM 2025;
arXiv 2507.16343; code github.com/cai525/Transformer4SED, weights huggingface.co/CPF2/detect_any_sound, MIT). An
open-vocabulary frame-level detector (HTS-AT backbone from MGA-CLAP + CNN + conformer decoder), queried by text
embeddings from the MGA-CLAP text encoder (github.com/Ming-er/MGA-CLAP, MIT; checkpoint `model.pt` from the Google Drive
link in its official README; text tower `bert-base-uncased`). Checkpoint used: `text_query/as_full_text_query_best_model.pt`
(trained on the full AudioSet-Strong class set), config `text_query/config.yaml`, backbone `htast_mga.pt`. Official
weights only.

## Frame scores (the cache)
- **Queries = exactly FlexSED's.** The 215 family names of `benchmark/gold/depictable_vocab.json`, in file order (the
  same list `benchmark/gold/flexsed_run.py` queried, without `--prompts`). Each model wraps them in its own official
  template: FlexSED `"The sound of " + x.capitalize()` (FlexSED `api.py` line 65); DASM `"sound of " + x.lower()`
  (official notebook `recipes/audioset_strong/detect_any_sound/detect_any_sound.ipynb`). The 215 wrapped strings and
  their embeddings are saved once to `data/work/dasm_text_queries.pt` (keys `vocab`, `queries`, `embeds [215, 1024]`),
  and the scoring step reads that file, so the claim can be audited.
- Text embedding: `clap.encode_text` → `clap.msc(word_embeds, codebook, mask)` → L2 normalise (official notebook),
  **one query at a time** (as FlexSED's `run_inference` does), so the padding of a batch cannot couple the queries.
- Query set given to DASM: the model's 407 base queries (`text_query.pt`) followed by the 215 custom queries, with the
  notebook's attention mask (a custom query sees the base queries and itself only, so custom queries do not affect each
  other). Output = the official `strong` output at `temp_w = 0.5` (the config's test value): `sigmoid(x / 0.5) * at_out`.
  No median filter (the FlexSED cache has none), no division by `at_out`.
- Audio: ffmpeg → mono 32 kHz; the official `waveform_modification` (pad or cut to 10 s, pad mask). All 695 clips are
  10.0 s; a longer clip would be scored in 10-s pieces and joined (rule written, not expected to trigger).
- Stored like the FlexSED cache: `fw [215, T] float16` (T = 500 frames per 10 s, 50 fps: hop 320 × 2 at 32 kHz; frames at
  or after the pad mask are cut), `labels` = the 215 family names in vocab order, `fps = 50`; extra array `at_out [215]`
  (descriptive only; the cells use the frame clip-max). Folder: `R.E.WIN / "dasm_cache"` for each set (the 280:
  `benchmark/audioset_calib_windows/dasm_cache`; the 415: `benchmark/audioset_heldout_windows/dasm_cache`).
- **Checks (a failure stops the round):** (c1) the DASM state dict and the HTS-AT backbone load with `strict=True`; the
  MGA-CLAP checkpoint (loaded `strict=False` as in the notebook) has no missing key under `text_encoder.`, `word_proj.`
  or `codebook`; (c2) batch independence: on one clip, one query scored alone equals its score inside the 215 batch
  (max abs diff < 1e-4); (c3) plausibility on the repo's 3 example wavs with the notebook's query list (printed, not a
  gate); (c4) every clip of both sets has a cache with 500 frames and the 215 labels equal to the FlexSED cache's labels.

- *Clarification to c4 (written after the caches existed, before any cost was computed):* 39 of the 280 and 49 of the
  415 clips have audio shorter than 10 s (the video is short; `duration` in the set json says 10.0). DASM gives fewer than
  500 frames for them, exactly 2 × FlexSED's frame count for the same clip (50 vs 25 fps). So c4 checks "≤ 500 frames and
  within ±2 of 2 × FlexSED's frames (FlexSED's capped at 10 s)", not "500 frames".
- *Clarification to the 10-s rule (same time, before any cost):* 28 clips (14 + 14) are 10.007–10.01 s long. The
  "10-s pieces" rule then added a 10-ms piece whose single zero-padded frame scored ~0.17 for 150–200 families, above
  their clip-max in the real audio — it would have inflated DASM's clip-max (the veto input). Fixed to the official
  behaviour: a clip up to 11 s is cut to 10 s (`waveform_modification` cuts at 10 s); only a longer clip (none here)
  is scored in pieces. Those 28 caches were deleted and re-scored. (FlexSED's cache has the same kind of tail: 25 frames
  from the 10-ms remainder padded to 1 s; it is part of the shipped baseline and left as is.)

## Training-data check (done before writing this; numbers from the official DASM meta files and the WavCaps json)
- **DASM train split:** 0 / 280 and 0 / 415 of our clips' YouTube ids are in `meta/audioset_strong/train/train.tsv`
  (103,205 ids), `hierarchical/train.tsv` or `hierarchical/dropped_train.tsv`. Our clips come from the AudioSet-Strong
  **eval** split: 280 / 280 and 413 / 415 are in DASM's `val/validation.tsv` (the 2 others, `g6qq3GHA9t0_150000` and
  `tu3g4ZBt3o0_30000`, are in no DASM list).
- **Caveat 1 — checkpoint selection:** DASM's "best_model" was chosen on its validation set = the AudioSet-Strong eval
  split (`hierarchical/val.tsv` holds 242 / 280 and 352 / 415 of our clips). So the choice of checkpoint saw our clips'
  strong labels (selection only, no gradient). FlexSED has the same kind of exposure risk; not checked here.
- **Caveat 2 — backbone pre-training:** the MGA-CLAP audio tower (DASM's `htast_mga.pt`) was pre-trained on WavCaps,
  whose AudioSet-SL part (108,317 clips) includes 15,863 eval-split ids: **260 / 280 and 393 / 415 of our clips** were
  seen there as audio + one caption (no timestamps). This favours DASM on both sets. Reported next to the result;
  descriptive split on the 415: ΔC-overlap for clips in vs not in WavCaps (n 393 / 22).

## The cells (everything else exactly the round-5 shipped stack)
Shipped stack = BEATs spans (bar 0.175, hysteresis 1.0) ∪ FlexSED spans (bar 0.8) by the twin rule (same family, within
1 s → the BEATs span keeps the earlier start; otherwise the FlexSED span is FlexSED-only); FlexSED clip veto (every span
dropped whose family's FlexSED clip-max < 0.3); BEATs self-veto on FlexSED-only spans (kept iff BEATs clip-max ≥ b =
0.1218); display floor 0.35 on the span's confidence (peak); PANNs off.
- **Baseline** = round 5's gate-1 row at b 0.1218, recomputed and asserted: the 280 → C-overlap 3.036, C-onset 3.850,
  recall 51.3 %, false 4.44 / min, 564 shown; the 415 → 1.928 / 2.207 / 49.7 % / 3.30.
- **D1 — DASM in place of FlexSED.** BEATs spans ∪ DASM spans (bar g, hysteresis 1.0, min duration 0.5 s as always) by
  the same twin rule; **DASM clip veto** (every span dropped whose family's DASM clip-max over the clip's frames < v);
  BEATs self-veto b = 0.1218 on DASM-only spans; display floor 0.35 on BEATs-origin spans.
- **D2 — BEATs ∪ FlexSED ∪ DASM.** The shipped stack is built first, unchanged, up to its vetoes; then DASM spans (bar g)
  are twin-merged against all its spans (BEATs and FlexSED-only; a twin keeps the earlier start); DASM-only spans must
  pass the FlexSED clip veto (0.3), the DASM clip veto (v) and the BEATs self-veto (0.1218). The DASM veto does not touch
  BEATs or FlexSED spans, so D2 shows every span the shipped stack shows (starts may move earlier) plus DASM-only spans.
- **Display floor for DASM-only spans = g.** In the shipped stack FlexSED-only spans pass the 0.35 floor automatically
  (their peak ≥ 0.8). DASM-only spans have peak ≥ g by construction; they are **not** held to 0.35 (if g < 0.35 the 0.35
  floor would silently remove them all). Twin-extended spans keep their own origin's confidence and floor.

## Bars (fitted on the 280 only, label-free: no gold label is read to fit them)
1. **Veto bar v (first).** FlexSED's veto 0.3 keeps a share s_F of all (clip × family) cells on the 280 (FlexSED clip-max
   ≥ 0.3, over the 280 clips × 215 families). v = the DASM clip-max value that keeps the same share: the ⌊s_F · N + 0.5⌋-th
   largest DASM clip-max over the same N = 280 × 215 cells. (Both bars are thus matched to FlexSED's own behaviour on
   the 280 by a count; this is "scaled the same way". It does not assume the two models' score scales are alike, which
   matters because DASM's frame score is multiplied by its clip tag.)
2. **Span bar g (second, v fixed).** N_F = the number of FlexSED-only spans the shipped stack shows on the 280 (after the
   FlexSED veto, the self-veto at 0.1218 and the display floor, counted after the salient non-speech non-music filter
   `D._ev`; twin-rule start extensions are not counted). For g on the grid 0.050, 0.055, …, 0.950, run D1 on the 280 and
   count DASM-only shown spans (same filter); g = the grid value whose count is closest to N_F, tie → the higher g.
3. D2 uses the same g and v (no second fit). b stays 0.1218 in both cells (not refitted).

## Pick and held-out test
- Score D1 and D2 on the 280 with the cost C of amendment 24 (C-overlap = 4 × missed consequential events + 2 × false
  spans per clip; C-onset the same with the onset window [−0.5, +1.0] s). **Eligible = C-overlap below the baseline AND
  C-onset below the baseline** (round-2/5 rule). **Pick = the eligible cell with the lowest C-overlap** (tie → lower
  C-onset, then D1). No eligible cell → the 415 is not scored and FlexSED stays.
- **Only the pick goes to the 415**, with g, v, b frozen. **Pass iff the upper end of the paired clip-bootstrap 95 % CI of
  ΔC-overlap (cell − baseline; 2000 draws, seed 0) is < 0.**
- Reported for every scored cell: C-overlap, C-onset (with ΔC and CI), recall (overlap), onset recall, false spans / min,
  shown spans, DASM-only shown spans, median end error (paired, round-4 `end_errors`, with bootstrap of the median
  difference); on the 415 also the complex vs random clip groups (ΔC-overlap with CI, mean Δ false spans per clip) and the
  WavCaps in / not-in split above. The 280 has no stratum key, so groups are reported on the 415 only.

## What will not be done
No second bar grid, no median-filter variant, no other checkpoint (audio/mix query), no per-family rule, no re-pick after
the 415. Nothing on TEST or DEV clips.

## Result (2026-09-28; `benchmark/detector_round6.json`; jobs 31329824 embed, 31329836 + 31329866 caches, 31329862 + 31329864 c4 only (stopped at c4, no cost), 31329878 check + fit, 31329889 the 415)
**Checks.** c1: DASM and HTS-AT backbone load strict=True with 0 missing / 0 unexpected keys; MGA-CLAP text side fully
loaded (only `position_ids` unexpected). c2: one query alone vs inside the 215 batch, max diff 2e-8 (pass). c3: the
notebook's examples look right (siren clips: alarm/sirens 0.87–0.96; the bird clip: bird 0.57, the rest ≤ 0.22). c4: all
280 + 415 caches present, frames match FlexSED's, labels equal FlexSED's (after the two clarifications above). Gate:
the baseline equals round 5's stack span for span; the 280 3.036 / 3.850 / 51.3 % / 4.44 (564 shown, 56 FlexSED-only);
the 415 1.928 / 2.207 / 49.7 % / 3.30.

**Bars (label-free, the 280).** FlexSED's veto 0.3 keeps 4.65 % of the 60,200 clip × family cells → **v = 0.0839**
(DASM keeps 4.65 %). N_F = 56 FlexSED-only shown spans → **g = 0.575** (55 DASM-only shown; g 0.57 gives 59, 0.58 gives 53).

*The 280 (ΔC = cell − baseline, clip bootstrap 95 % CI; end error = paired median baseline → cell):*

| cell | C-overlap (ΔC) | C-onset (ΔC) | recall | onset-recall | false/min | shown (DASM-only) | end error |
|---|---|---|---|---|---|---|---|
| baseline | 3.036 | 3.850 | 51.3 % | 25.9 % | 4.44 | 564 (0) | — |
| **D1** DASM in place of FlexSED | **2.921 (−0.114 [−0.457, +0.186])** | 3.736 (−0.114 [−0.464, +0.186]) | 55.8 % | 30.4 % | 4.52 | 625 (55) | 1.40 → 1.44 s (n 111) |
| D2 BEATs + FlexSED + DASM | 2.943 (−0.093 [−0.321, +0.057]) | 3.757 (−0.093 [−0.350, +0.086]) | 55.4 % | 29.9 % | 4.54 | 587 (23) | 1.39 → 1.40 s (n 115) |

Eligible (both C below baseline): D1, D2. **Pick = D1** (g 0.575, v 0.0839, b 0.1218), frozen.

*The 415, D1 vs baseline:* C-overlap 1.928 → 2.222, **ΔC-overlap +0.294 [+0.111, +0.511] → fails** (upper CI not < 0; the
CI is wholly above 0, so D1 is reliably worse here). C-onset 2.207 → 2.549 (ΔC +0.342 [+0.145, +0.564]); recall 49.7 →
49.1 %; onset-recall 32.7 → 29.2 %; false/min 3.30 → 4.15 (228 → 287 false spans); shown 672 → 830, of which 137 DASM-only
(FlexSED-only in the baseline: 56 — the count-matched bar did not carry over); end error (81 paired events) 1.64 → 1.50 s
(Δ +0.14 [0.00, +1.55]). Groups: complex ΔC +0.333 [+0.087, +0.595] (n 252, +0.15 false spans / clip); random +0.233
[−0.098, +0.577] (n 163, +0.13); in WavCaps +0.305 [+0.102, +0.524] (n 393); not in WavCaps +0.091 [−0.727, +0.818] (n 22).

**Not adopted: FlexSED stays.** On the 280 both DASM cells looked a little better (−0.09 to −0.11, CIs cross 0). On the
415 the pick is clearly worse: shown spans 672 → 830 = 137 DASM-only in place of 56 FlexSED-only, plus 77 more
BEATs-origin spans (693 vs 616) let through by the looser DASM veto (0.084); false spans +59 and recall −0.6 points. On the
280 the same swap added 61 shown spans but only 4 false ones (recall +4.5 points); on the 415 the extra spans were much
more often false (span counts by origin and true/false in the secondary section below). The training-data caveats (checkpoint chosen on the eval split; 393 /
415 clips in the backbone's WavCaps pre-training) would have favoured DASM, so they do not explain the failure.

## Note added 2026-09-28, after the primary result above and before any secondary number (lead's request after `docs/history/analyses/detector_audit_2026-09-28.md`)
The harness scores with `config.LABEL_FILTER = "lists"`; the shipped system uses `"depictable"` (about 20 % of the
baseline's false spans are labels it never draws). The primary result above stays as it is and decides nothing new.
**Secondary rows, reported only:** the same frozen cells (g 0.575, v 0.0839, b 0.1218 — no refit) re-scored with
`LABEL_FILTER = "depictable"` (it changes both which shown spans count and which gold events count; span extraction does
not read it): baseline, D1 and D2 on the 280; baseline and the pick D1 on the 415 (D2 is not scored on the 415, so the
held-out rule "only the pick goes to the 415" still holds). Same C, ΔC with the same bootstrap. **Extra count per cell,
under both filters:** true spans removed = baseline shown spans that overlap a gold event of their family (not false) and
that no same-family cell span overlaps; false spans removed = the same for the baseline's false spans; also true and
false spans added (cell shown spans that no same-family baseline span overlaps), gross and net.

### Secondary result (job 31329915; `secondary` in `benchmark/detector_round6.json`; reported only, decides nothing)
The "lists" rows reproduce the primary numbers exactly. With the shipped filter (`depictable`):

| set | row | C-overlap (ΔC) | C-onset (ΔC) | recall | onset-recall | false/min | shown | consequential events |
|---|---|---|---|---|---|---|---|---|
| 280 | baseline | 2.664 | 3.479 | 52.5 % | 26.5 % | 3.54 | 515 | 219 |
| 280 | D1 | 2.550 (−0.114 [−0.457, +0.186]) | 3.364 (−0.114 [−0.464, +0.186]) | 57.1 % | 31.1 % | 3.62 | 576 | 219 |
| 280 | D2 | 2.571 (−0.093 [−0.321, +0.057]) | 3.386 (−0.093 [−0.350, +0.086]) | 56.6 % | 30.6 % | 3.64 | 538 | 219 |
| 415 | baseline | 1.692 | 1.971 | 50.0 % | 32.9 % | 2.62 | 600 | 170 |
| 415 | D1 (pick) | 1.986 (+0.294 [+0.111, +0.511]) | 2.313 (+0.342 [+0.145, +0.564]) | 49.4 % | 29.4 % | 3.47 | 758 | 170 |

The filter lowers every C by the same amount in the baseline and the cells, so **every ΔC is identical under both
filters**: the labels the shipped system never draws (Hum, Sine wave, …) are not among the 215 queried families, so
neither FlexSED's nor DASM's veto touches them and they are the same spans in both arms. The conclusion does not change.

*Span churn vs the baseline (identical under both filters, for the same reason). Removed = baseline span with no
same-family cell span overlapping it; added = cell span with no same-family baseline span overlapping it; true = overlaps
a gold event of its family:*

| set | cell | true removed | false removed | true added | false added | net true | net false |
|---|---|---|---|---|---|---|---|
| 280 | D1 | 19 | 60 | 76 | 66 | +57 | +6 |
| 280 | D2 | 0 | 0 | 15 | 8 | +15 | +8 |
| 415 | D1 | 32 | 63 | 130 | 127 | +98 | +64 |

On the 415, D1 removes 32 true spans and adds 127 false ones; about half of what it adds is false (127 / 257), against
46 % on the 280 (66 / 142) — but on the 280 it also removed as many false spans as it added, and on the 415 it did not.
