# Pre-registration: detector round 11 — is the band evidence CAUSED by the speech / music masker?

*Written 2026-09-28, before any round-11 cache existed and before any round-11 cost was computed. Detection only, on the
AudioSet-Strong fit set (the 280, `R.use_set("calib")`) and the held-out set (the 415, `"heldout"`). DEV and the fresh
set only through the conditional steps at the end; TEST never. Harness: `benchmark/detector_round11.py` (reuses
`detector_round8` for the stack, gate 0, scoring, bootstrap and Holm), results in `benchmark/detector_round11.json`, jobs
`slurm/job_round11_*.sh`. Round 10 (`docs/prereg_round10_rescue.md`, running) is not touched; none of its files is edited.*

## Why
The audit (`docs/detector_audit_2026-09-28.md`) found that the largest group of missed needed sounds is sounds FlexSED
scores 0.4–0.8, below its bar 0.8 (the "band group"). Every rule so far that let them in also let in ≥ 2 false spans per
rescued event (break-even under C is 2: a miss costs 4, a false span 2). A Fable consult's diagnosis: the band phantoms
are FlexSED queries that light up on speech-ness or music-ness itself. So round 11 does not ask a second detector; it
tests the **mechanism**: does the candidate's evidence behave like a real sound under a speech / music masker, or like a
response to the masker?

## Fixed parts (as rounds 8–10)
- **Baseline = the shipped stack**, exactly round 8's gate 0 (`detector_round8.stack8` with no option = round 5's stack
  span for span). Round 8's I7 and round 9's J1 / J2 are **off**.
- **Gate 0 (stop if it fails):** 280: C-overlap 3.036, C-onset 3.850, recall 51.3 %, 207 false spans, 564 shown;
  415: 1.928 / 2.207 / 49.7 % / 228 / 672 (`detector_round8.gate_set`).
- **Cost C**, clip lists, paired clip bootstrap (2000 draws, seed 0), one-sided p (share of draws with mean ΔC ≥ 0), the
  primary filter (`LABEL_FILTER = "lists"`) and the secondary "depictable" row: as in round 8.

## Band candidates (identical to round 10's, so the two rounds filter the same objects)
Per clip, per FlexSED column (the 215 families): `_extract_events` at bar 0.4, low 0.4, min duration 0.5 s. A span is a
**candidate** iff (1) its peak is < 0.8; (2) it is FlexSED-only: no BEATs span of the same canonical family (AED 0.175,
before any veto) within 1 s (the shipped twin rule's test); (3) it passes the BEATs self-veto (BEATs clip-max of the family
≥ 0.1218; a family BEATs lacks passes). The FlexSED clip veto (0.3) and the display floor (0.35) pass automatically.
Candidates are computed from the saved caches, no gold read. **Admitted candidates are appended to the baseline's shown
spans** (label, start, end, confidence = peak); nothing is removed, so "true spans removed" is 0 by construction (asserted).
"In the span" = frames with t in [start, end]. "Family curve" = per frame, the max over the FlexSED columns whose canonical
family equals the candidate's (`detector_round8.cols_for`).

Note on the pool: the band group (`detector_round8.hbd_flags`) matches families with `E._same` (also ancestor /
descendant) and ±1 s, while candidates use the exact family. A band event can therefore be rescued by a candidate of a
related family, and some band events have no candidate at all (they are out of reach for every cell).

## Re-inference (GPU, one job per set; only clips with ≥ 1 candidate)
FlexSED is re-run by a copy of `benchmark/gold/flexsed_run.py`'s setup (same checkpoint, local CLAP, the 1-s pad patch of
`split_audio_fixed`, `inference_mode`, the 215 queries in batches of 24 with FlexSED's own "The sound of …" template, no
prompt ensemble — as the calib / heldout caches were made). Outputs are saved as float32 in
`data/work/round11_flexsed_<set>/<id>__<view>.npz`. Four views per clip:
- **orig** — the 16-kHz mono wav from `ffmpeg -i <mp4> -ac 1 -ar 16000` (flexsed_run's command). Plus one **separate**
  batch of two extra queries, "Speech" and "Music" (the 215 vocab has neither), used by M3 / M5.
- **spk** — orig + foreign speech at 0 dB SNR; **mus** — orig + foreign music at 0 dB SNR (below).
- **rev** — orig reversed in time (only clips with a transient-family candidate, list below).
All wavs are temporary (deleted after the npz is written). 0 dB SNR = the foreign signal is scaled so its RMS over the whole
clip equals the RMS of the orig audio over the whole clip; mix = orig + foreign, written as float32 (no clipping; FlexSED
normalises each 10-s chunk by its peak itself). A clip with orig RMS < 1e-4 gets no mix (its candidates are not admitted
by M1 / M1m / M1b; counted).
**Gate R0 (stop if it fails):** on every re-run clip, the orig view has the cache's frame count and max |orig − cache| <
0.02 over all frames and families (the cache is float16 and the calib cache was made on another card class). The M1 / W
deltas compare re-run with re-run (same job, same card), never with the cache.
The 415's re-inference runs in the same way before the picks: it is cache building, not scoring; nothing on the 415 is
scored before the picks are fixed.

### The foreign masker pool (from the fit set; no public speech / music corpus is on the cluster)
Built once from the 280 and used for both sets. A **speech stretch** = a run of consecutive BEATs frames (2-s windows,
0.25-s hop, the saved calib cache) where BEATs "Speech" ≥ **0.6**, "Music" < **0.1**, and every BEATs label that is
salient non-speech non-music under the primary filter < **0.1**; the stretch's audio is [first frame time, last frame
time], kept if ≥ **2.0 s**, and dropped if any gold event of the source clip with a salient non-speech non-music label
(consequential or not) overlaps it ±0.5 s (the 280 gold is fit data; using it to clean the masker touches no score). A
**music stretch** = the same with "Music" ≥ 0.6 and "Speech" < 0.1. Each stretch is cut from the source clip's 16-kHz audio,
scaled to RMS 1, with 20-ms linear fades in and out.
**Fallback, fixed now:** if a pool has < 120 s in total or comes from < 10 source clips, its bars are relaxed once to
(0.5, 0.2, 0.2); if it is still below that, its cells are reported as not run.
**The draw for a target clip** (seed 0): `rng = default_rng([0, crc32(clip id), m])` with m = 0 for speech, 1 for music;
the pool's stretches in `rng.permutation` order, **excluding stretches from the target clip itself**, are concatenated until
the clip's length is reached and the result is cut to the exact sample count.

## The cells (values fixed here; no grids, no fitting)
Definitions per candidate. `mean_v` = mean of the family curve in the span, from view v (re-run).
- **M1 — masker dose-response, speech.** Δ_spk = mean_spk − mean_orig. **Admit iff Δ_spk < 0.**
- **M1m — the same with music.** Admit iff Δ_mus < 0.  **M1b — admit iff Δ_spk < 0 AND Δ_mus < 0.**
- **M2 — argmax margin.** Competitors = the 215 FlexSED columns minus (a) the candidate's own family (same canonical),
  (b) every label related to it by `E._same` (ancestor or descendant in the AudioSet ontology), (c) its siblings (same
  direct parent in `src/labels._parents()`), (d) speech / music labels (`SPEECH_LABELS` or their descendants, `is_music`).
  (The vocab has no Speech or Music query, so (d) removes little; vocal families such as Shout, Yell, Screaming, Crowd stay
  competitors — a speech-driven query should light them up too.) Margin = mean over span frames of (family curve − max
  competitor), on the **cached** scores. **Admit iff margin ≥ 0.15.**
- **M3 — masker-envelope correlation.** r_s, r_m = Pearson r between the cached family curve and the re-run "Speech" /
  "Music" query curves over the span frames (a curve with zero variance → r = 0). **Reject iff r_s > 0.7 or r_m > 0.7;
  admit otherwise.**
- **M5 — two-regime shape** (cached family curve s). *Sharp:* rise = max over span frames t of [s(t) − min s over frames in
  [t − 0.5 s, t]] (frames before the span allowed) > **0.3**, AND the prominence of the span's peak frame (first argmax in the
  span; `scipy.signal.peak_prominences` on the whole-clip curve) > **0.2**. *Long stationary:* CV = std / mean of s over the
  span < **0.2** AND M3's r_s ≤ 0.7 and r_m ≤ 0.7. **Admit iff sharp OR stationary.** (The spec gives no duration for
  "long"; none is added.)
- **M7 — cap.** On top of **the better of M1 / M2 on the 280** (lower C-overlap; tie → lower C-onset; tie → M2): at most one
  admitted candidate per clip, the one with the largest M2 margin (tie → higher peak, then earlier start).
- **C — best bet: M1 AND M2, with the M7 cap.**
- **W — time-reversal (wild card), transient families only.** Only candidates whose family is in the list below can be
  admitted. Rev curve mapped back to the clip by time (orig frame at t ↔ rev frame at L − t − 0.04 s, nearest frame; L =
  clip length). **Admit iff mean_orig − mean_rev ≥ 0.2.** (A clip whose length is not a multiple of 10 s has its chunk
  boundaries at other places after reversal; this adds a small error that is accepted.)
  **Transient families (fixed now, from the 215):** Basketball bounce; Burst, pop; Camera; Single-lens reflex camera; Chink,
  clink; Chop; Clapping; Clunk; Coin (dropping); Cough; Crack; Dishes; Dog; Door; Doorbell; Ding-dong; Drawer open or close;
  Explosion; Finger snapping; Fireworks; Glass; Gunshot; Hammer; Hiccup; Knock; Snap; Sneeze; Sonic boom; Specific impact
  sounds; Splinter; Tap; Thump, thud; Thunk.
- **A0 — reference, every candidate admitted.** Not picked, not in Holm. Its rescue count is the ceiling of every cell.

**Expectation written before any number:** 0 dB foreign speech masks most non-speech sounds, so it will lower the score of
most candidates, true or false, especially *unmasked* ones (where the band evidence is not about speech at all). M1 can
only separate true from false where the phantom is driven by speech-ness (then Δ_spk ≥ 0). If M1 admits nearly every
candidate, it is ≈ A0 and fails for the same reason.

## Reported per cell
C-overlap, C-onset (ΔC, 95 % CI, one-sided p), recall, onset recall, false/min, shown spans; **heard-but-dropped events
rescued** (of the band group, `hbd_flags`) and all rescued events; **false spans added** (gross and net); **true spans
removed** (must be 0); the **"depictable"** row; the candidate table (considered, admitted, admitted true = overlaps a gold
event of its family, admitted false).
**Masked vs unmasked split (added at Adam's request, before any number):** "masked" = BEATs "Speech" or "Music" ≥ 0.3 at
some frame in the span (`detector_round8.peak_in`, round 8's I10 test). Every cell's rescues (by the gold event's span),
false spans added (by the added span's span) and the band pool are split masked / unmasked; the candidate table too. For
M1 and M1m: the share of candidates with Δ < 0, split true / false × masked / unmasked. Adam's point: only 17 % (280) and
59 % (415) of misses are under speech / music, so a masker test can at best help the masked part.

## Decision rules
- **Picked on the 280** iff C-overlap < baseline AND C-onset < baseline (M1, M1m, M1b, M2, M3, M5, M7, C, W).
- Each picked cell goes to the 415 alone, frozen; it **passes** iff the upper 95 % CI of ΔC-overlap < 0. **Holm** across
  the cells sent (one-sided p, family α = 0.025, step-down); a cell counts as passed only if it passes AND Holm rejects it.
- **Before the 415 is scored**, `docs/setup_audit_2026-09-28.md` is read. If it reports a timing / alignment or scorer bug,
  the 415 is not scored and the round stops with a report. If it does not exist yet, the 415 waits.

## Next steps, only for a cell that passes the 415 (fixed in outline now; finished as a dated amendment before computing)
1. **DEV check (DEV only, never TEST)** for **added** spans: the cell's added stage-4 spans rebuilt on each DEV clip from
   the pipeline's caches (FlexSED `data/work/flexsed_cache/<stem>.npz`; the masker / reversal views re-run on the DEV clip's
   `audio.wav` with the same code); an added span only becomes a picture if the stage-5 gate would show it — taken from
   `benchmark/gold/dev_candidates_check.py` (another agent's; reused as is, not edited). Scored with the official
   `score_per_sound` (onset rule, defaults), before / after, for ours ("proposed").
2. **Fresh set** (`R.use_set("fresh")`), scored once, cell frozen; same gate; primary: upper 95 % CI of ΔC-overlap < 0.
3. **Ship rule:** only if the fresh set passes AND on DEV hits do not drop AND DEV wrong pictures do not rise by more than
   2 × the hits gained.

## What will not be done
No other SNR, margin, bar, window or family list; no per-family rule; no re-pick after the 415; no combination other than
M1b, M7 and C; nothing on TEST; no edit of any running job's file or of round 10's files.

## Amendment 1 (2026-09-28, before any re-inference and before any cost): the speech pool
The pool step ran (`benchmark/detector_round11.json`, key `pool`; descriptive, no score read). **Music passed** at the
fallback bars (0.5 / 0.2 / 0.2): 53 stretches, 257.3 s from 43 clips. **Speech failed** both levels: 4.8 s from 2 clips at
(0.6 / 0.1 / 0.1) and 33.8 s from 8 clips at (0.5 / 0.2 / 0.2). A check of why (counts only): the 280 has 270 s where
BEATs Speech ≥ 0.5 and Music < 0.2, but in AudioSet speech almost always has other labelled sounds around it, so the gold
rule removes most of it. Under the rule above M1, M1b, C (and M7 if built on M1) would not run — the round's main cell.
So, before anything is re-run: **the speech pool comes from an official public dataset that is already on the cluster,
the DCASE2025 Task 3 Stereo SELD development set** (`data/dcase2025_task3/stereo_dev/*/` + `metadata_dev/`; derived from
STARSS23, recorded rooms, 13 human-annotated classes at 100-ms frames). A speech stretch = a run of annotation frames
where class 0 or 1 (female / male speech) is active and **no other of the 13 classes** is active, taken as
[first frame × 0.1 s, last frame × 0.1 s] (safe under a 0- or 1-based frame index), kept if ≥ 2.0 s; audio = the stereo
wav, mixed to mono (mean), 16 kHz (librosa); then RMS 1 and 20-ms fades as before. Files in sorted path order; the draw is
unchanged (seed 0, per target clip; no DCASE file is a target, so nothing is excluded). The same size rule applies
(≥ 120 s from ≥ 10 files). Sounds outside the 13 classes (room noise, HVAC) are not annotated and may be present; this is
accepted and disclosed. The music pool stays the 280 one above. Nothing else changes.

## Note 2026-09-28 — HOLD on the 415 (from the lead, before any round-11 cost)
The setup audit (`docs/setup_audit_2026-09-28.md`) found that the AudioSet cost C punishes any added span (on the 415 an
empty output beats the shipped stack; each bark of a run is counted as its own event; there is no time tolerance). A
corrected cost is being defined, and the picks will be re-judged under it. So round 11 runs **the 280 step only, exactly
as registered** (cost C as above), and reports it. **The 415, DEV and the fresh set are not run for any cell** — not even
the 415 re-inference — until the lead lifts the hold. The picks computed on the 280 under C are recorded but not acted on.

## Result on the 280 (2026-09-28; `benchmark/detector_round11.json`; jobs 31330592 + 31330610 re-run, 31330593 score)
Gates: gate 0 passed (3.036 / 3.850, span for span = round 5). Gate R0 passed: 85 clips re-run, max |re-run − cache| =
0.00036 (median 0.00024). Pools: speech = DCASE (amendment 1), 450 stretches, 1532.5 s from 443 files; music = the 280 at
the fallback bars, 257.3 s from 43 clips. Candidates: 209 in 85 clips (15 of a transient family). The A0 row equals round
10's R0 (C-overlap 3.814), so both rounds see the same candidates. Band pool (`hbd_flags`): 61 events, 46 masked / 15 not.

| cell | C-overlap (ΔC, 95 % CI) | C-onset ΔC | admitted (true / false) | band events rescued (masked / not) | false spans added (masked / not) | depictable ΔC-overlap |
|---|---|---|---|---|---|---|
| baseline | 3.036 | 3.850 | — | — | — | (2.664) |
| A0 (all, ref.) | 3.814 (+0.779 [+0.464, +1.064]) | +0.779 | 209 (82 / 127) | 9 (8 / 1) | 127 (96 / 31) | +0.779 |
| M1 speech | 3.607 (+0.571 [+0.379, +0.779]) | +0.571 | 149 (63 / 86) | 3 (2 / 1) | 86 (65 / 21) | +0.571 |
| M1m music | 3.721 (+0.686 [+0.386, +0.957]) | +0.686 | 185 (71 / 114) | 9 (8 / 1) | 114 (87 / 27) | +0.686 |
| M1b both | 3.543 (+0.507 [+0.343, +0.679]) | +0.507 | 134 (57 / 77) | 3 (2 / 1) | 77 (56 / 21) | +0.507 |
| M2 margin | 3.157 (+0.121 [+0.057, +0.200]) | +0.121 | 33 (16 / 17) | 0 | 17 (15 / 2) | +0.121 |
| M3 envelope r | 3.750 (+0.714 [+0.407, +0.993]) | +0.714 | 183 (67 / 116) | 8 (8 / 0) | 116 (86 / 30) | +0.714 |
| M5 shape | 3.757 (+0.721 [+0.414, +1.007]) | +0.721 | 194 (75 / 119) | 9 (8 / 1) | 119 (88 / 31) | +0.721 |
| M7 = cap on M2 | 3.121 (+0.086 [+0.043, +0.136]) | +0.086 | 20 (8 / 12) | 0 | 12 (11 / 1) | +0.086 |
| C = M1 ∧ M2 + cap | 3.114 (+0.079 [+0.036, +0.129]) | +0.079 | 18 (7 / 11) | 0 | 11 (10 / 1) | +0.079 |
| W reversal | 3.050 (+0.014 [−0.029, +0.050]) | +0.014 | 7 (3 / 4) | 1 (0 / 1) | 4 (3 / 1) | +0.014 |

M7 was built on M2 (M2 had the lower C-overlap). True spans removed: 0 in every cell (asserted). ΔC-onset equals
ΔC-overlap in every cell here (every rescued event is also an onset hit). **No cell is below the baseline on either cost,
so nothing is picked; the 415 is not run (and is on hold anyway).**

M1 / M1m, share of candidates whose score drops under the masker (Δ < 0):

| candidates | speech: drop / n (median Δ) | music: drop / n (median Δ) |
|---|---|---|
| true, masked | 47 / 63 (−0.089) | 54 / 63 (−0.232) |
| true, unmasked | 16 / 19 (−0.099) | 17 / 19 (−0.400) |
| false, masked | 65 / 96 (−0.092) | 87 / 96 (−0.346) |
| false, unmasked | 21 / 31 (−0.121) | 27 / 31 (−0.360) |

As expected before the run, a 0 dB masker lowers almost every candidate, real or phantom, masked or not: the speech drop
rate is 75 % for true and 68 % for false candidates, music 87 % vs 90 %. The dose-response does not separate them.
Post hoc, descriptive only (`descriptive_auroc_280` in the json; true vs false candidates): Δ_spk 0.51, Δ_mus 0.47,
M2 margin 0.65, FlexSED peak 0.66, rise 0.60, prominence 0.55; the "long stationary" branch points the wrong way (low CV:
0.35) and so does the speech correlation (0.42: true candidates follow the Speech query curve slightly *more*).
**Why no filter can win here:** even A0 rescues only 9 of the 61 band events (the rest have no candidate: short sounds
under the 0.5-s minimum span, see `docs/dropped_sounds_why_2026-09-28.md`). A perfect filter would gain at most
9 × 4 / 280 = −0.129 and could keep at most 18 phantoms to break even; the best cell by margin (M2) kept 17 phantoms and
lost all 9 rescues.

Checks after the run: W had rev scores for all 15 transient candidates. The two re-run workers used different cards
(H200 and A100), but each clip's four views come from one worker, except one clip (`AMLJZImQ5Xk_30000`, both touched it);
its two candidates have Δ_spk −0.40 / −0.36 and Δ_mus −0.39 / −0.34, far from 0, so no decision depends on the card (the
smallest |Δ| of any candidate is 0.0004 for speech and 0.006 for music, vs ~0.0003 re-run noise). Arithmetic check: every
ΔC equals (2 × false spans added − 4 × band events rescued) / 280, e.g. A0 (254 − 36) / 280 = +0.779, W 4 / 280 = +0.014.
