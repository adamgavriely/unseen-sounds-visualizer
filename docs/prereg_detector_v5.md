# Pre-registration: the last detector attempt (attempt twelve) — generic long/short fusion

*Declared 2026-09-19 after a five-researcher SOTA search (docs: R1 frame-level SED, R2 audio
taggers, R3 separation, R4 fusion rules, R5 audio LLMs; verified links in the session
scratchpad) and a planner/critic round. Adam's constraints: no training; no per-class
rules (a "long vs short sound" rule is allowed, "model X for helicopter" is not); everything
selected on the AudioSet-Strong calibration set first; slice B run once; two-day box.*

## Why this and not the rest
- PSED already contains the recall (74% at 7.1 false spans/min when loosened); the task is
  to keep it while dropping the false alarms. Its misses are long, soft, steady sounds under
  music that a long-window tagger still scores 0.4–0.7.
- Dropped: SpotSound (removes phantoms, cannot rescue misses; "yes" bias; leak check), SAM-Audio
  (gated, slow, thin evidence), stacking (+2–4 pt, below what slice B can see), DASM (a second
  frame model from the same training data; the five-backbone average already showed correlated
  errors). Recorded as options for future work.
- Kept: **SSLAM** (ICLR 2025, MIT, 527-class head, trained on audio mixtures) as a newer
  long-window model next to BEATs; the DCASE-2024 winners' **Sound Event Bounding Boxes**
  idea (extent by change detection, confidence by duration-dependent score); a **confirmation
  gate** (loose PSED kept only if the long-window score agrees).

## Step 0 — the ten-minute check (from cached scores, before building anything)
On the calibration set: PSED boxes at the loose bar (0.05); each box labelled true/false
against gold; mean long-window score (BEATs; SSLAM once cached) per box; AUROC for boxes
longer than L ∈ {1, 2, 3, 4} s. If the AUROC is below 0.75 for both long-window models the
gate cannot buy 7 points and the attempt stops at this table. Also counted: PSED misses with
no loose box at all (unreachable by any rule on PSED's scores).

## The three named variants (no others)
Frame model: PSED (bar-free; boxes from its frames with one shared change-detection setting).
V1 **BEATs gate**: PSED at the loose bar; keep a box only if the BEATs 2-s window score of
the same family inside the box ≥ θ.
V2 **SSLAM gate**: same with SSLAM 10-s window scores.
V3 **Duration routing**: box confidence = PSED max if the box is shorter than L, else the
long-window (SSLAM; BEATs as control) mean inside the box; one threshold on the confidence.
Grid: θ / threshold over 0.05–0.95 in 0.05 steps, L ∈ {1, 2, 3, 4} s — the grid exists only
to match the false-alarm rate; no other knobs.

## Selection (calibration set only)
Fix the false-alarm rate to **2.6 spans/min** (PSED's own bar on slice B; BEATs' 6.4 would let
loose PSED nearly pass by itself). Score each variant by masked-consequential recall at that
rate. Split-half check: settings chosen on 140 clips, recall read on the other 140; the
winner must lead on both halves. Ties under 2 points go to the simpler variant. All
calibration numbers are reported, not only the winner. Settings frozen and written here
before slice B is touched.

## Pass rule (slice B, one run)
Masked-consequential recall **≥ 70% at ≤ 2.6 false spans/min** (PSED alone: 62.7% at the
DCASE bar, 64.8% at 4.05/min at the calibration bar). 66–69% is reported as "not
detectable on 236 events", not as a fail. The full recall-vs-false-alarm curve of PSED and
the winner is the figure; onset MAE is reported as a cost (> 1.5 s noted).

## On fail
The thesis states: a twelfth pre-registered attempt shows that long, soft sounds under music
are not recoverable from frozen public models with generic rules; the contribution is the
benchmark (slice B + calibration set) and the negative result reported in full.

## Budget
~15 engineering hours, ~1 GPU hour (SSLAM caching); everything after caching is CPU on
cached scores. SSLAM checkpoint fetched and hashed first (Google-Drive link-rot risk).

## Amendment (2026-09-19, before selection ran): variant V4 — speech removal as a second witness

Adam asked for a small test of "remove speech, then detect". Design (Fable): SAM-Audio
(Meta, Dec 2025; access granted) with the prompt "speech", `predict_spans=False`, no
reranking; view B = 0.9 × residual + 0.1 × original (the blend that Focus-Then-Listen 2026
found necessary for frozen models; our own Demucs attempt fed pure stems and got worse);
a 1.0 view is a diagnostic only; music is NOT removed (sirens, alarms, whistles are tonal).
Rule (generic): a PSED candidate at the loose bar on the original is kept if
p_orig ≥ τ OR p_viewB ≥ τ for the same family in the same span; τ chosen at 2.6 FP/min on
the calibration set, split-half. Go/no-go before slice B: AUROC(max(p, p_B)) ≥ AUROC(p) + 0.02
AND held-half recall gain ≥ 2 points at equal false alarms. Sanity checks: residual/original
lag = 0; PSED spans on speech-only stretches of the residual (hallucination); energy lost in
the span of buried sounds (removed targets, expected for human-vocal classes; they keep
p_orig so cost nothing). V4 joins V1–V3 in the same selection; the variant count is now four.

## Correction (2026-09-19, after the first calibration pass, before slice B): the gate must be a rescue

The first implementation of V1/V2 *replaced* PSED's decision on long boxes by the long-window
model's; at 2.6 FP/min it raised masked-consequential recall on the calibration set (31
events, 48% → 68%) but lowered recall on all events (45% → 38%): it traded hundreds of
ordinary detections for a few buried ones. That is not the rule Adam described ("low-
confidence PSED verified by the checker") nor V4's. Corrected rule, **rescue**: PSED's own
detections at its calibrated bar are never removed; a *long* loose box that PSED alone would
not pass is added iff the long-window model's score ≥ θ. The winner must also keep
all-events recall within 1 point of PSED alone. The replace-rule numbers are kept in
`fusion_v5_setting.json` as a diagnostic.

## Result of V1–V3 on the calibration set (2026-09-19 evening) — no winner

Step 0 passed for both long-window models (AUROC on boxes ≥ 2 s: BEATs 0.78, SSLAM 0.80;
74% of the buried consequential sounds reachable by a loose box). But at 2.6 false spans per
minute no variant beats PSED alone on both the full set and the held-out half
(`benchmark/fusion_v5_setting.json`): the best rescue (BEATs, L = 1 s, base 0.20) reaches
48.1% all-events recall against PSED's 45.1% and the same 48.4% masked recall, and exceeds
the false-alarm target on the held-out half (3.2/min). The replace-style gate that looked
good in the first pass (masked 48% → 68% on 31 events) did so by giving up 7 points of
recall on the hundreds of other events — an artefact of the tiny masked count, caught by
the all-events guard. Per the pre-registration V1–V3 are not run on slice B. V4 (speech
removal as a witness) is still pending its environment.

## Amendment 2 (2026-09-20, before any separation run): V4 becomes "cleaned-audio views", three views, one threshold

*Declared after a five-reviewer pre-mortem (separation, statistics, engineering, examiner,
systems) and two CPU pre-checks on cached scores. Adam approved the three design choices.*

**Motive and its provenance.** Profiling PSED's 83 misses on slice B (`scripts/psed_miss_profile.py`)
showed that recall depends on what covers the sound, not on its length: speech-covered 59 %
(90 sounds), music-covered 63 % (87), other sounds only 85 % (26); by length 60–78 % with no
trend. This profile was read on the *test* slice, so the motive is post hoc; on the
calibration set the speech-covered group replicates (52 %, 13/25) and the music group is too
small to say (6/6). V4 is therefore confirmatory only through the pass rule below; its motive
is disclosed as test-set inspection.

**Ceiling (pre-check).** 48 of the 83 misses have a PSED family score < 0.05 inside the gold span
on the original audio; a rule that only reconsiders PSED's own loose candidates cannot reach
them: best possible masked recall 80 % (188/236). Reaching the 70 % bar needs 13 of the 35
reachable misses rescued at no extra false alarms.

**Views.** SAM-Audio (`facebook/sam-audio-base`, prompts `"speech"` and `"music"`,
`predict_spans=False`, `reranking_candidates=1`, fixed seed per clip). A = original; B = speech
removed; C = music removed; D = both removed (original minus both stems). Each view = 0.9 ×
residual + 0.1 × original, trimmed to the original length, resampled to 16 kHz. PSED scores all
four. Prompt fallback: if on the first 20 calibration clips more than 25 % of gold
consequential sounds lose > 0.3 of their family score in B or C, the prompts switch to
`"a person talking"` / `"background music"` — decided before the full run, recorded here.

**Rule (single threshold).** Candidate spans come from PSED's loose boxes (bar 0.05, as in
step 0). Score of a candidate = max over {A, B, C, D} of the PSED family score inside its span.
Kept iff score ≥ τ. One τ for all views. PSED's own detections at its calibrated bar (0.15 on
A) are never removed. Two candidate sources, both declared, nothing else:
- **V4-orig**: candidates = loose boxes on A only (ceiling 80 %).
- **V4-union**: candidates = loose boxes on any of A–D (merged per family across views before
  anything else; higher ceiling, more exposure to separation artefacts).

**Selection (calibration set only).** τ on a 0.01 grid = the loosest value with ≤ 2.6 false spans
per minute (same false-alarm definition as before: a span is false if it overlaps no
same-family gold event by ≥ 0.5 s or half its length); split-half both directions (seed 0).
Selection metric = recall over *all* consequential events (the calibration set has only 31
masked ones); masked recall reported beside. Guards as before: FP target met on the full set
and the held-out half; all-events recall ≥ PSED − 1 point. Go/no-go before slice B:
AUROC(max-over-views) ≥ AUROC(A) + 0.02 on the candidate boxes AND held-half consequential
recall ≥ PSED + 2 points at equal false alarms. Between V4-orig and V4-union: the higher
held-half recall; ties under 2 points → V4-orig.

**Slice B (one run).** Pass iff masked-consequential recall ≥ 70 % at ≤ 2.6 false spans/min AND
the paired clip-level bootstrap (1000 resamples, seed 0) 95 % interval of (V4 − PSED) masked
recall lies above 0. 66–69 % or an interval touching 0 = "no detectable gain", reported as such.
Report: false-alarm rate reached on both sets; τ; ceiling counts; recall with the clip holding
the most misses removed; 2×2 event agreement PSED vs V4 (McNemar counts, descriptive only —
events cluster in clips); per-view attribution of every rescued detection; recall-vs-false-alarm
curves for PSED, V4 and each single view; onset error of rescued spans (onset = first frame
≥ τ in the winning view); QC per clip (residual lag, length, RMS ratio, NaN, clipping);
artefact families (families that rise > 0.2 on clips where A has no speech); GPU hours and
cost; checkpoints, prompts, seeds.

**Downstream, if it passes.** Score passed to stage 5 = rescaled max-over-views with
`source_view` recorded; duplicates merged per family across views before the gate; applied to
every clip (no "only if speech present" switch); new row name, nothing else changed; the
100-clip benchmark judged on F1-strict with its CI, since a recall gain can still lower F1.

**Both result paragraphs, written now.** *Pass:* "Removing speech/music with a frozen separator
and re-scoring with the same detector recovers N of the 35 reachable buried sounds at equal
false alarms (CI …); 48 sounds remain inaudible to the detector even after cleaning."
*No gain:* "Cleaning the audio does not help a frozen detector at matched false alarms (CI
includes 0); the buried sounds it misses score below 0.05 on every view — attempt thirteen
closes the detector work; the contribution is the benchmark and the negative result."

## Amendment 3 (2026-09-20, before any separation run): HTDemucs instead of SAM-Audio

SAM-Audio could not be run on the cluster (torchcodec/FFmpeg/NPP library chain, a 2023
ImageBind reranker needing `pkg_resources`, and an undeclared 15-GB "judge" download that
filled the home disk and killed the running v4ab row). A three-reviewer panel judged it the
wrong tool for two fixed classes anyway. Views are now made with **HTDemucs** (`htdemucs_ft`,
Meta, stems vocals / drums / bass / other; already installed): B = original − vocals;
C = original − drums − bass ("other" is kept because it carries ambient sounds as well as
melodic instruments, so C is a partial music removal); D = original − vocals − drums − bass.
Blend, rule, candidate sources, τ selection, guards, go/no-go, pass rule and reporting are
unchanged from amendment 2. The prompt-fallback clause is replaced by the same over-removal
check (share of consequential gold sounds losing > 0.3 of family score in B or C on the first
20 calibration clips), reported; no alternative separator is tried if it fires. Known cost:
"vocals" also removes singing and may remove cries/screams — the max-over-views rule keeps A,
so nothing PSED already hears is lost.

## Result of V4 (HTDemucs views) on the calibration set (2026-09-20 ~07:30) — no winner, not run on slice B

*Correction before selection (same kind as the "gate must be a rescue" correction above):
PSED alone at its shipped bar 0.15 already makes 5.4 false spans/min on the calibration set,
so an add-only rule cannot meet the 2.6/min target; as in fusion_v5, the base bar inside the
rule is on a grid {0.15 … 0.35} and the control is PSED at its loosest bar under the target
(0.28: consequential recall 64.3 %, masked 54.8 %, 2.51/min). Everything else as declared.*

Separation QC (391 clips): lag 0 samples, lengths exact, deterministic (max diff 0.0), no
NaN, stems sum to the original within 8 % (median); over-removal check 0/17 gold sounds lost
> 0.3 in B or C (`benchmark/sep_v4_prompt_check.json`).

Go/no-go 1 — AUROC on the 1 301 candidate boxes: original 0.762, max over views 0.765
(+0.003; needed +0.02), single views B 0.768, C 0.769, D 0.776. **Failed.**
Go/no-go 2 — held-half recall at ≤ 2.6/min: V4-orig 54.5 % vs 46.4 % on one half (+8.1),
82.1 % vs 82.1 % on the other (0; needed ≥ +2 on both). **Failed.** Full set at matched false
alarms: V4-orig (base 0.30, τ 0.30) consequential 68.3 % vs 64.3 %, masked 64.5 % vs 54.8 %
(31 events), all-events 48.5 % vs 46.9 %; single views B/C/D 69–70 %; V4-union the same
within a point (`benchmark/sep_v4_setting.json`).

Reading: cleaning the audio moves a few buried sounds over the bar (+4 points on all
important sounds at equal false alarms) but does not change how well PSED separates true
from false candidates, and the gain is not stable across halves. Per the pre-registration V4
is **not run on slice B**; attempt thirteen closes the detector work. The thesis sentence:
"Removing speech and music with a frozen stem separator before detection does not give a
detectable gain at matched false alarms (AUROC +0.003; held-half gain +8 / 0); 48 of the 83
missed sounds on slice B score below 0.05 on the original and are out of reach of any
re-scoring rule."
