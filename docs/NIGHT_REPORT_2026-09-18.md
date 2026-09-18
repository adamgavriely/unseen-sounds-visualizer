# Night report, 18 → 19 September 2026

*Written while Adam was away (8 h). Every decision taken alone is marked "(mine)"; every one
that needs him is under "Decisions for the morning". Numbers here are copied from the JSON
files named; nothing is rounded up.*

## What ran and what came out

### Detector (stage 4): three candidates, one held-out table

- **FLAM attempt 2** — FAILED as pre-registered (docs/prereg_v4.md §4): buried-sound recall
  70% ✔, clear 88% ✔, but 62 false spans/min ✘ and 8/23 dev reals lost ✘.
- **PretrainedSED BEATs-strong** (docs/prereg_psed.md) — FAILED on one of eight bars: DCASE
  masked recall 34% (BEATs 9.5%), clear 44% (33%), FP 3.9/min (5.2), onset error 0.19 s
  (0.35), 91% within 0.5 s (80%), phantoms gone 70/77 — all ✔; **dev reals kept 7/23** ✘
  (bar ≥ 21). Diagnosis: genuine low scores on long ambient sounds under music/speech.
- **Gold slice B** (mine, declared before numbers): 111 AudioSet-Strong *eval* clips (real
  YouTube video, every sound human-timed, chosen for a consequential sound at least half
  under speech/music; 39 more were gone from YouTube). Detector table on 236 masked
  consequential events (`benchmark/audioset_detector_eval.json`):

| detector | masked-consequential recall | all consequential | all events | false spans/min | onset MAE | within 0.5 s |
|---|---|---|---|---|---|---|
| BEATs (v3) | 50.0% | 51.1% | 37.9% | 6.76 | 1.48 s | 55% |
| **PretrainedSED** | **62.7%** | **64.2%** | **53.0%** | **2.59** | **1.14 s** | **65%** |
| FLAM-v2 | 54.2% | 60.7% | 46.2% | 37.3 | 2.36 s | 37% |

  PSED beats BEATs on every measure on the held-out real-world slice; the pre-registered
  bar it failed is populated by BEATs' own detections (agreement with BEATs, not recall).
- **Decision (mine, per Fable):** the letter of the pre-registration stands — BEATs is the
  v4 detector — and a **detector arm** was pre-registered with its rule fixed before any
  protocol score: PSED rows (v4a, v4ab) run next to the BEATs rows; PSED is adopted iff
  v4ab's gated score beats v4b's with a paired CI not entirely below 0. Both 20-clip
  checks are queued. Thesis wording is in docs/prereg_v4.md (last section).

### Gate (stage 5)

- Qwen3.8-27B thinking arm still running (12-h job); at 91/258 events it is *worse* than
  the direct arm (on-screen ~44% vs 56%, off-screen ~70% vs 83%), so v4 uses **thinking off**
  (the pre-registered rule: adopt thinking only if ≥ 5 points better).
- The gate's silence rule is being re-set for Qwen3.8 exactly as it was for v3: raw votes on
  the dev split (job running on 3×L4), then the rubric-cost sweep on dev only.

### Context stages (2, 3) — verified on the GPU
SAM 3 (12 GB, 91 s per clip for 27 concepts) and Granite Speech 4.1 (7 GB) both run.

### Evaluation
- v3's panels re-described by Qwen3.8-27B and re-judged (interim judge Mistral; Gemma needs
  disk) → the comparable v3 row, `benchmark/protocol_results_v3_q38_grounded.json`:
  gated **3.28** vs blind **3.35** (−0.07, CI [−0.29, +0.14]: a tie, as before), caption 2.95,
  oracle 3.75. The describer swap lifts every absolute score by ~0.1–0.2 and changes no
  conclusion. (Ungrounded reference: gated 2.91 vs blind 3.24, same pattern as v3.)
- v4b (BEATs + Qwen3.8 gate) 20-clip check running on an A100.

### Disk (still the blocker for Qwen-Image and the Gemma judge)
34 GB free; the deletions need your yes; `/home/fast` and `/home/lab` exist but are
admin-assigned; `/private/keren-lab` needs the `ug_keren` group.

## Decisions for the morning

(filled in at the end)
