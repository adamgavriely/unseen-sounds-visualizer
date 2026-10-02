# Pre-registration: detector round 7 — SAM-Audio "remove speech and music, then listen again"

*Written 2026-09-28, before any SAM-Audio output was made and before any cost. Detection only. TEST and DEV clips are
not touched. Harness: `benchmark/detector_round7.py`, results in `benchmark/detector_round7.json`. Cluster jobs:
`slurm/job_round7_sep.sh` (separation), `slurm/job_round7_cache.sh` (QC + BEATs/FlexSED on the residual),
`slurm/job_round7_score.sh` (CPU: gates, cells, pick, held-out).*

## The idea, in one line
Many real sounds are hidden under speech or music. Take speech and music out with a text-queried separator, run the
shipped detectors again on what is left (the "residual"), and add what they now hear to the shipped output.

## The model
SAM-Audio (Meta, Dec 2025; github.com/facebookresearch/sam-audio, commit `bb4c699`; SAM License; weights gated on
Hugging Face — access confirmed for Adam's token on 2026-09-28, `auth_check` OK on all `facebook/sam-audio-*` repos).
Official weights only.
- **Size: `facebook/sam-audio-large`** (checkpoint 14.86 GB). The home disk has ~35 GB free; large + `t5-base`
  (~0.9 GB) fits. Fall-back written now: if `df` shows < 22 GB free right before the download, `sam-audio-base` is used
  and this is stated in the result.
- Loaded with `SAMAudio.from_pretrained(id, visual_ranker=None, text_ranker=None, span_predictor=None)`: no re-ranking
  models (no 12-GB judge, no CLAP, no ImageBind) and no span predictor are built or downloaded. Weights and `t5-base` are
  fetched once on the login node; every job runs with `HF_HUB_OFFLINE=1`, so a hidden download fails loudly instead of
  filling the disk (what killed a running row on 20 Sept).
- Settings: `predict_spans=False`, `reranking_candidates=1`, the default ODE (`midpoint`, step 2/32), fp32,
  `torch.manual_seed(0)` right before every `separate` call (so each call is deterministic).
- Env: the existing `sota` conda env, which already holds `sam_audio` (installed on 20 Sept); nothing is installed or
  upgraded. `torchcodec` cannot load its FFmpeg libraries there, so it is replaced by an import stub; audio is passed to
  the processor as tensors, so torchcodec is never called.

## The residual (how it is made)
- Audio: `ffmpeg` mp4 → mono float32 at 48 kHz (SAM-Audio's rate) = x.
- **Queries: `"speech"` then `"music"`** (the strings of the 20 Sept design). No fall-back wording.
- **Sequential, arithmetic (primary):** t_s = SAM-Audio target for `"speech"` given x; r1 = x − t_s;
  t_m = SAM-Audio target for `"music"` given r1; **residual r = r1 − t_m = x − t_s − t_m**. Sequential, not parallel, so
  content that both queries claim (for example singing) is not subtracted twice.
- Length: SAM-Audio's output can differ from the input by up to one codec hop (1920 samples). Outputs are trimmed or
  zero-padded at the END to the input length; nothing is ever shifted.
- The residual is pure (no 0.1 × original mixed back in; the 20 Sept design had that blend — not used here, because the
  question asked is "input minus the targets").
- The residual is written as raw float32 48 kHz and converted with `ffmpeg` to mono 16 kHz 16-bit wav — the same route
  and format the original caches used (mp4 → ffmpeg → 16 kHz wav). Samples beyond ±1 are clipped by that conversion;
  the count of clipped clips is reported.
- **Model's own residual stem (declared fall-back):** SAM-Audio also returns a generated "residual" stem. Route "stem" =
  the same two queries in the same order, but each step keeps the model's residual stem instead of subtracting the
  target (r1 = stem("speech" | x); r = stem("music" | r1)). Both routes are written for every clip (three `separate`
  calls per clip); only one is scored.

### QC gate that chooses the route (before any cost)
QC clips = the first 10 clips of the 280 (in `detector_round2.usable()` order) whose ORIGINAL BEATs cache has Speech or
Music clip-max ≥ 0.6. For each route, BEATs is run on the QC clips' residuals (msproj env, `infer_beats`):
- (a) length equals the input's (after the trim above) — must hold on 10/10; target/input cross-correlation lag
  (|lag| ≤ 2400 samples at 48 kHz) reported;
- (b) the clip's dominant class (Speech or Music, whichever has the higher original clip-max) falls by ≥ 0.3 in clip-max
  on the residual;
- (c) RMS of residual / RMS of input over the samples inside BEATs windows where the original Speech or Music ≥ 0.5
  (window = [t − 1.5, t + 0.5] s around the stamp t), per clip.
A route **passes** iff (a) holds on 10/10, (b) on ≥ 6 of 10 clips, and the median of (c) < 1.0. The arithmetic route is
used if it passes; else the stem route if it passes; else the round stops here, reported, with no cost computed.

### Identity control (a gate: the route must not change the audio by itself)
The same route with r := x (48 kHz float → ffmpeg → 16 kHz wav → BEATs and FlexSED) on the first 20 clips of the 280.
Gate: the shipped stack on the identity caches gives the same shown spans (`detector_round4.same_spans`) as on the
original caches on ≥ 18 of 20 clips; the max |Δ| frame score per model is reported. A failure is a route bug: stop,
fix, re-run the gate; no cost is read before it passes.

## The residual view (detectors unchanged)
BEATs (`infer_beats`, same 2-s / 0.25-s windows, same stamps) and FlexSED (the 215 families of
`benchmark/gold/depictable_vocab.json`, no prompt ensemble, batch 24, the same wrapper as `benchmark/gold/flexsed_run.py`,
but fed the residual wav path directly — `flexsed_run.py --clip-dir` would silently reuse the original 16 kHz wavs in
`data/work/gold_wav_flat/` with the same stems) are re-run on the residual. Same cache format, new folders:
`benchmark/audioset_{calib,heldout}_windows/beats_r7res/`, `data/work/flexsed_{calib,heldout}_r7res/`
(identity: `beats_r7ident/`, `data/work/flexsed_calib_r7ident/`). No existing cache is overwritten.

**Residual-view spans** = the shipped stack run on the residual caches alone, with the shipped bars and no refit:
BEATs_res at AED 0.175 / display 0.35 / hysteresis 1.0; FlexSED_res at 0.8; FlexSED_res clip veto 0.3; BEATs_res
self-veto b = 0.1218 on FlexSED_res-only spans (`detector_round5.run(..., slot 0, b=0.1218)` on the residual caches).
So a residual span passes the same vetoes, measured on the residual view.

## Cells (two, fixed; no grid)
- **Shipped (baseline)** = the stack on the original caches at b = 0.1218 (`detector_round5.run`), which must reproduce
  3.036 / 3.850 / 51.3 % / 4.44 on the 280 and 1.928 / 2.207 / 49.7 % / 3.30 on the 415 (round 5's tolerances) — gate.
- **S1** = shipped ∪ residual-view spans. Union rule: a residual span that has a shipped "twin" (same canonical family,
  within 1 s: `x.start − 1 ≤ e.end` and `e.start − 1 ≤ x.end`, the stack's own twin test) is absorbed and **the shipped
  span is left untouched** (start and end unchanged); only residual-only spans are added.
- **S2** = S1, but a residual-only span is added only if the ORIGINAL BEATs cache has Speech or Music ≥ 0.3 in any frame
  inside [span start, span end] (round 3's `speechy` test, read on the original audio, since the residual has the
  speech removed by design).

## Pick and test
- Cost C per clip as amendment 24 (`detector_round2.clip_cost`): C-overlap = 4 × missed consequential events (overlap
  rule) + 2 × false spans; C-onset uses the onset window [−0.5, +1.0] s.
- **Pick on the 280:** eligible iff C-overlap < shipped AND C-onset < shipped (point values, as rounds 2 and 5). Pick =
  lowest C-overlap among eligible (tie → lower C-onset, then S1). No eligible cell → the 415 is not scored.
- **The 415 (pick only):** passes iff the paired clip bootstrap (2000 draws, seed 0) upper 95 % CI of ΔC-overlap
  (pick − shipped) < 0. The 415 was used by rounds 2–5 (disclosed).
- Reported per cell: C-overlap, C-onset, ΔC with CI, recall (overlap), onset-recall, false spans and false spans/min,
  shown spans, residual-only spans added (and how many of them are false), complex / random strata on the 415.

### "Previously unheard" events (definition fixed now)
A consequential gold event (salient, non-speech, non-music) is **unheard** iff the shipped stack misses it (overlap rule)
AND on the ORIGINAL caches the same-family BEATs peak in [start − 1, end + 1] s is < 0.35 AND the same-family FlexSED peak
in that window is < 0.8 (neither shipped detector reaches its own bar near it). It is **recovered** by a cell iff the cell
hits it (overlap rule) with a residual-only span. Reported: unheard count, unheard share of all shipped misses (to set
beside the "57 % heard by no detector" figure, which is not from this set or this definition), and recovered counts per
cell (280; the 415 for the pick).

## How this differs from the earlier separation attempts
- *Demucs, 15 Sept:* the detector read only the "other" stem, in place of the mix — recall fell. Here the original view
  is always kept; the residual only adds spans.
- *HTDemucs views (am. 2–3 of `prereg_detector_v5.md`, 20 Sept):* fixed stems (vocals / drums+bass), detector PSED,
  0.9/0.1 blend, a max-over-views score with τ picked on a grid, AUROC go/no-go. Here: a text-queried generative
  separator, the shipped BEATs + FlexSED stack with its own vetoes on the residual, no refit.
- *DeepFilterNet residual (am. 12 of `prereg_v4.md`, 23 Sept):* speech only, BEATs only, scored on DEV gold with false
  labels counted per family, the two views concatenated. Here: speech and music, both detectors, the AudioSet-Strong 280
  / 415 with span-level cost C. Because C counts false *spans*, plain concatenation would double-charge a residual span
  that repeats a shipped one, so the union absorbs twins (the family-level count in am. 12 did the same implicitly), and
  shipped spans are never moved (am. 12's outcome: residual timing pulled mix onsets into the noise floor).

### Note added 2026-09-28, before any residual cache or cost existed (from `docs/history/analyses/detector_audit_2026-09-28.md`)
The harness scores with `config.LABEL_FILTER = "lists"`, but the shipped profile draws with `"depictable"` (about 20 % of
today's false spans carry labels the shipped system never draws). **The primary above is unchanged** (lists; the pick and
the 415 test use it). **Secondary rows:** every cell on the 280, and the pick on the 415, are also scored with
`LABEL_FILTER = "depictable"` (the same spans; the filter applies to spans and gold alike), with their own shipped baseline
and ΔC. These rows are reported, never used to pick or to pass. Also reported for every cell: **true (matched) spans
removed** (shipped spans that overlap a same-family gold event and are missing from the cell; 0 by construction, since the
union only adds) and true spans added.

## Results
*(filled in after the runs; nothing above is changed after a number is seen)*

**Result (2026-09-28): the QC gate fails for both routes → round 7 stops before any cost, as pre-registered.** No
residual cache was made, the identity gate was not needed, and no C, recall or false-span number exists for S1 or S2 on
the 280 or the 415. The shipped stack stays. (`benchmark/detector_round7.json` → `qc`.)

Runs: `facebook/sam-audio-large` (disk had 32 GB free before the download, 18 GB after), env `sota`, offline. Jobs
31329879–82 separated all 280 + 415 clips (three `separate` calls per clip; 11 s per clip on an A100, 5 s on an H200;
1.3 GB of wavs in `data/work/r7_samaudio/`). Job 31329883 = QC; jobs 31329884–88 (caches, scoring) were cancelled
because the QC stop fired.

QC on the 10 fixed clips (the first 10 of the 280 with original BEATs Speech or Music ≥ 0.6; 6 Speech-dominant,
4 Music-dominant):

| route | (a) length ok | (b) dominant class falls ≥ 0.3 (need ≥ 6/10) | (c) median RMS residual / input in speech/music windows (need < 1.0) | pass |
|---|---|---|---|---|
| arithmetic x − t_s − t_m | 10/10 | **1/10** | 0.78 | no |
| model's residual stem | 10/10 | **4/10** | 0.76 | no |

What it means, in plain words: SAM-Audio makes the clips quieter where people talk (about a quarter less sound), but BEATs
still hears the speech or music almost as strongly as before. Subtracting the "speech" and "music" outputs lowered the
BEATs score of the main class by ≥ 0.3 on only 1 of 10 clips (median drop 0.03); the model's own residual did it on 4 of
10 (all four are speech clips; none of the 4 music clips). So the "listen again" view would still be full of speech and
music, and the idea cannot be tested fairly with this separator.

Descriptive numbers over all clips (not part of any rule): the `"music"` target is under 1 % of the input's RMS on 153 of
280 and 209 of 415 clips (median 0.7 % / 0.9 %), so the music query removes almost nothing; the `"speech"` target has a
median RMS of 26 % (280) and 40 % (415) of the input. Residual RMS / input: arithmetic 0.87 / 0.78, stem 0.71 / 0.66
(280 / 415). Target/input lag for speech is 0 samples on 9 of the 10 QC clips (1 on the other). No NaN; every output was
within one codec hop of the input length.

Also pre-computed (baseline only, no cell): the "unheard" definition on the 280 gives 68 of the shipped stack's 109
consequential misses (62 %; beside the "57 %" figure). The secondary label filter gives a shipped baseline of
2.664 / 3.479 / 52.5 % / 3.54 false per min on the 280 (depictable) against 3.036 / 3.850 / 51.3 % / 4.44 (lists). The
true-span count is 0 by construction (the union only adds), and no cell was scored.

---

## Round 7b — retry with descriptive queries and span prediction (pre-registered 2026-09-28, before any 7b output)

*Why:* Adam asked for SAM-Audio to be tested properly, and round 7 failed at the QC gate for a reason that looks like
set-up (the one-word `"music"` query removed almost nothing). 7b is a new round with its own gate. Round 7's result
stays as recorded above. Harness: the same `benchmark/detector_round7.py` with `R7_TAG=7b`; results go to
`benchmark/detector_round7b.json`; jobs `slurm/job_round7b_{sep,cache,score}.sh`; outputs go to new folders
(`data/work/r7b_samaudio/`, `beats_r7bres/`, `flexsed_<set>_r7bres/`, `*_r7bident/`).

**What changes (nothing else does):**
- **Queries, in this order:** `"a person talking"` first, then `"background music"` (the fall-back pair written in the
  20 Sept design). The order is the same as round 7: speech first, then music on what is left.
- **`predict_spans=True`.** This is the official README's setting for text prompts: the span predictor tells the model
  *when* the target sounds. Span predictor = PE-A-Frame-large, the official `facebook/pe-a-frame-large` at revision
  `perception_models` (the format SAM-Audio loads, 6.1 GB). The copy already cached is the transformers-format `main`
  revision (a different file), so the `perception_models` revision is fetched once on the login node (disk: 23 GB free
  before the fetch). The model is loaded with `span_predictor="pe-a-frame-large"`. Still no re-ranking models;
  `reranking_candidates=1`, seed 0 before every call, fp32, offline jobs.
- Model: `facebook/sam-audio-large`, as in round 7. Both routes (arithmetic x − t_s − t_m, and the model's residual stem)
  are made for every clip, as in round 7.

**Gate 1 — QC (as round 7, same 10 clips):** (a) length ok on 10/10; (b) the dominant class (Speech or Music) falls by
≥ 0.3 in BEATs clip-max on ≥ 6/10 clips; (c) median RMS residual / input in speech/music windows < 1.0.

**Gate 2 — erase check (new, fixed now):** does the separation keep the sounds we want to find? Events = the gold events
of the same 10 QC clips that are salient non-speech, non-music (`is_salient_nonspeech` and not `is_music`, the harness's
"lists" filter, consequential or not), counted only if the ORIGINAL BEATs same-family peak in [start − 1, end + 1] s is
≥ 0.3 (so a loss of 0.3 is possible). Checked before any residual output: 11 such events (of 33 salient events) in the
10 clips. Per event, drop = original peak − residual peak (BEATs, same window, same family). An event is **erased** iff
drop > 0.3. Gate 2 passes iff at most 25 % of the counted events are erased (at most 2 of 11). FlexSED is not used in the
gate (it is not run on the QC residuals); the median drop and the per-event rows are reported.

**Route choice:** the arithmetic route if it passes Gates 1 and 2; else the stem route if it passes both; else **7b stops,
recorded as a negative result, with no cost computed**.

**After the gates, everything is exactly as round 7:** the identity gate (≥ 18/20 same shown spans, on 7b's own identity
wavs); the residual view = the shipped stack on the residual caches (shipped bars, same vetoes measured on the residual,
b = 0.1218, no refit); cells S1 and S2 with the same union rule (a twin is absorbed and shipped spans are never moved; S2
only where the ORIGINAL BEATs Speech or Music is ≥ 0.3); the pick on the 280 (C-overlap and C-onset both below shipped;
lowest C-overlap); the 415 for the pick only (paired clip bootstrap, 2000 draws, seed 0; pass iff upper 95 % CI of
ΔC-overlap < 0); the "unheard" definition and recovered counts; the secondary "depictable" rows (never used to pick or
pass); true spans removed and added. The 415 has now been used by rounds 2–7 (disclosed).

**Disk:** keep `sam-audio-large`; the round-7 residual wavs (`data/work/r7_samaudio/`, 1.3 GB) are deleted only after 7b's
own wavs are written.

### Round 7b results
*(filled in after the runs)*

**Result (2026-09-28): Gate 1 (QC) fails for both routes again → 7b stops before any cost. Negative result.** No
residual cache, no identity gate, no S1/S2 number on the 280 or the 415. The shipped stack stays.
(`benchmark/detector_round7b.json` → `qc`.)

Runs: jobs 31330158–61 separated all 280 + 415 clips (sam-audio-large, predict_spans=True with PE-A-Frame-large
`perception_models`; about 11 s per clip on an A100, 5 s on an H200). Job 31330162 = QC (it exits non-zero on the
declared stop). Jobs 31330163–67 were cancelled.

| route | (b) dominant class falls ≥ 0.3 (need ≥ 6/10) | (c) median RMS ratio (< 1) | Gate 1 | erase check: erased / counted (≤ 25 %) | Gate 2 |
|---|---|---|---|---|---|
| arithmetic | **2/10** | 0.88 | fail | 0/11 (median drop 0.000) | pass |
| model's residual stem | **5/10** | 0.79 | fail | 2/11 (median drop 0.04) | pass |

In plain words:
- **Speech:** the stem route now removes speech well enough on 4 of 6 speech clips (BEATs Speech falls by 0.38–0.77). It
  did the same in round 7, so the descriptive wording and span prediction add little here.
- **Music: still not removed.** `"background music"` removes almost nothing, like `"music"` before. Its output is under
  1 % of the input's loudness on 154 of 280 and 222 of 415 clips (median 0.6 %). On the 4 music clips, only one
  (92BvO3) passes. So the gate fails for one reason in both rounds: this separator does not take out music when asked
  with text.
- **Subtracting the targets hardly changes the audio.** Median residual/input loudness is 0.97 (280) and 0.95 (415), and
  the speech target is only 8–13 % of the input's loudness. The arithmetic route therefore keeps the target sounds
  (0/11 erased), but it also keeps the speech and music.
- **The stem route removes some target sounds as well.** Two cash-register events drop from 0.98 and 0.49 to 0.38 and
  0.01. The alarm clock (−0.26) and the heartbeat (−0.19, −0.24) fall too, though below the 0.3 limit.

Both rounds together: SAM-Audio text prompts can take speech out of the model's own residual on about two thirds of
speech-heavy clips, but not music, so a "listen again without speech and music" view cannot be built fairly with it.
This closes the idea as a negative result; no cost was ever computed, so the 280 and the 415 were not spent on it.

Disk: `sam-audio-large` (15 GB) and PE-A-Frame-large `perception_models` (6.1 GB) are kept in the HF cache. The round-7
wavs (`data/work/r7_samaudio/`) were deleted after 7b's wavs were written. 7b's wavs (`data/work/r7b_samaudio/`,
1.3 GB) remain. 18 GB free.
