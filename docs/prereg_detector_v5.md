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
