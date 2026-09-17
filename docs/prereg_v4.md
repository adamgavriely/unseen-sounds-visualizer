# Pre-registration: v4 — the SOTA configuration, one swap at a time

*Committed 2026-09-18, before any v4 run. Decided with Adam after a Fable review and a live
check of the public leaderboards (arena.ai 7–13 Sept 2026, Artificial Analysis Sept 2026, HF
Open ASR). The supervisor's ask: the best models, a clear win over the baselines.*

## What v4 is

| stage | v3 (frozen, tag `v3-thesis-2026-09-17`) | v4 | why this one |
|---|---|---|---|
| 2 on-screen hints | OWLv2 (2023) | **SAM 3** (`facebook/sam3`, Nov 2025, text-prompted) | the recognised successor; context-only stage |
| 3 speech (context) | faster-whisper base | **Granite Speech 4.1-2B** (`ibm-granite/granite-speech-4.1-2b`, Apr 2026, WER 5.33, #1 Open ASR) | English assumed (6 languages; Whisper 99) |
| 4 sound detection | BEATs (2022), 527 classes | **FLAM** (Adobe, ICML 2025) with a descriptive query vocabulary and per-query calibration (§4) | the measured bottleneck; the only recognised frame-level open-vocabulary detector with released inference |
| 5 visibility + depiction + describer | Qwen2.5-VL-7B | **Qwen3.8-27B** (Aug 2026; Vision Arena open #4, 1244 ± 10) | already verified on an A100; 7B → 27B + thinking |
| 6 picture | FLUX.1-schnell (2024) | **Qwen-Image-2512** (Dec 2025, Apache-2.0; the only open model in the top 3 of both arenas) | quality; clean licence |
| 7 judge | Mistral-7B-Instruct (2024) | **Gemma-4-31B-it** (text mode; a different family from the Qwen describer and generator) | the describer and the generator are both Qwen in v4; the judge must not be |

Not in v4: MiniMax-H3 (video arm, "later", stays in the plan as the future section); DASM
(weights only, inference code unreleased — cited as a peer).

## Order and attribution

Four model swaps at once would make any gain unattributable. The swaps land in this order,
each with its own bar, on the frozen 100-clip test set and (when it exists) the gold set:

1. **+4 FLAM** → 20-clip check, then the full protocol (`v4a`)
2. **+5 Qwen3.8-27B** → 20-clip check, then full (`v4b`)
3. **+6 Qwen-Image-2512** → full (`v4c`)
4. **+3 Granite, +2 SAM 3** bundled last (`v4`) — context-only stages, expected Δ ≈ 0; run
   for freshness and reported as such

Each step is reported cumulatively (v3 → v4a → v4b → v4c → v4) and, at the end, as a
leave-one-out row per stage. The judge for every v4 row is Gemma-4-31B-it with Qwen3.8-27B
as describer; v3's panels are re-described and re-judged once under the same pair so the v3
row is comparable (the Mistral-judged v3 numbers stay in the thesis as the v3 chapter).

## The headline bar (what "significant improvement" means)

The claim is against the **baselines**, not against v3: under the same evaluation, the
gated system must beat the blind audio-to-image baseline. Pre-declared:

- **primary:** gated − blind judge score on the 100 test clips, paired bootstrap 95% CI
  excluding 0 and ≥ +0.25 on the 0–4 scale (v3: +0.01, a tie);
- **cost:** panel-on time and wasted seconds not worse than v3's ratio to blind (47% vs 60%);
- **gold set (when available):** per-sound "drawn / not drawn" accuracy of the gate vs the
  blind rule, and the human sentence as the reference.

If the primary bar fails after all swaps, the thesis says so with the numbers; the
cumulative table is the result either way.

## §4 FLAM, second attempt (the part that is a new experiment)

The first attempt (docs/prereg_flam.md) used the 527 bare AudioSet names as queries and
failed on precision (FP/min 693–2728 vs BEATs 5.2) while finding 70–85% of buried sounds.
Declared here, before the run:

- **Vocabulary:** a fixed list of ~85 descriptive queries ("a dog barking", "an emergency
  vehicle siren wailing"), each mapped to one AudioSet label so everything downstream
  (families, gate, dedup, depiction) is untouched. Written from the ontology and the
  pipeline's own families; five labels were added so that every *real* labelled detection
  on the dev split has a query (dev is the tuning split; disclosed). The list is
  `src/stage4_audio_event_detection/flam_queries.py` and does not change after the run.
- **Calibration (per query):** on DCASE 2025 task 3 **dev-train-tau** audio (600 five-second
  clips, one per mix and start time, never used for evaluation), the bar for query *q* is the
  lowest bar in {0.05, 0.10, …, 0.95} whose spans not overlapping a gold event of a matching
  class number ≤ 5.2 / |Q| per minute — the BEATs false-positive rate shared equally. A
  query that meets it at no bar gets 0.95. Scores are then rescaled piecewise-linearly so
  that the bar maps to `DISPLAY_THRESHOLD` (0.35) and 1 stays 1; the shipping span rule
  (reach the bar, extend through half of it, ≥ 0.5 s) runs unchanged on the rescaled scores.
- **Evaluation:** the same five bars as the first attempt, on **dev-test-tau** (the 255
  events) and the labelled dev detections:

| measure | BEATs | FLAM-v2 must reach |
|---|---|---|
| masked-event recall (under speech/music) | 9.5% | **≥ 24.5%** |
| clear-event recall | 32.6% | **≥ 32.6%** |
| false positives / min | 5.2 | **≤ 5.2** (a real test now, not by construction) |
| dev real detections still fired (of 23) | 23 | **≥ 21** |
| dev phantoms no longer fired (of 77) | 0 | **≥ 40** (a phantom whose label has no query counts as gone; disclosed) |

PASS iff all five. On pass, `AED_MODEL = "flam"` is v4a. On fail, attempt nine in the
table with the numbers, and v4 continues from stage 5 with BEATs.

## §5 Qwen3.8-27B — the visibility check already ran

docs/prereg_qwen38_visibility.md, thinking-off arm (job 30546271): on-screen recall
**56.3%** (bar ≥ 50% ✔), off-screen recall **82.6%** (bar ≥ 95% ✘) — the same trade the
32B made. The thinking arm (30546328) is still running. Whatever it says, the swap goes
ahead by decision (best models), the numbers are disclosed next to the 7B's, and no prompt
or vote rule is changed for it. The gold set's per-sound labels are the arbiter.

## Not done

No tuning on the test clips; no prompt changes after seeing numbers; the gate bar stays;
v3 stays in every table.
