# DEV check of the best detector candidates (report only) — 2026-09-28

*Plan written before any candidate number on DEV. Adam approved the check on 28 Sept. DEV only: the 49 human-labelled
DEV clips (`score_per_sound.subsets_of(gold)["dev"]`, gold `gold_AG.json`, 36 needed sounds). TEST (`test_bench`) and
slice B are not touched. Script `benchmark/gold/dev_candidates_check.py`, results `benchmark/gold/dev_candidates_check.json`,
jobs `slurm/job_devcand_*.sh`. Nothing here changes the shipped system: it is a report.*

## Why
The candidates below were tested on AudioSet-Strong clips (the 280 fit set, the 415 held-out set). AudioSet clips are
short and often clean. Our DEV videos are long, busy scenes. A win on AudioSet may not carry over. This check asks: what
would each candidate do to the real DEV pictures, scored by the official per-sound metric?

## The candidates (used unchanged, values frozen on the 280)
| name | round | what changes (everything else = the baseline stack) |
|---|---|---|
| EAT-R | 5 | EAT-large (AS2M fine-tune) replaces BEATs as the frame tagger; display bar 0.515, AED bar 0.2575; self-veto on FlexSED-only spans uses EAT's clip-max, b 0.1659 |
| D1 | 6 | DASM replaces FlexSED: DASM spans at g 0.575; DASM clip veto v 0.0839 on every span (in place of FlexSED's 0.3); BEATs self-veto 0.1218 on DASM-only spans |
| I4 | 8 | parent emission: a parent label's score = 1 − Π(1 − child scores); parent spans emitted where no child and not the parent reaches 0.35 |
| I6 | 8 | VLM scene prior: Qwen3.8-27B lists plausible families from 4 frames (round-8 prompt with amendment 2); FlexSED bar 0.5 for listed families, 0.8 for others |
| I7 | 8 | FlexSED local-contrast veto: a FlexSED-only span is dropped if (mean score inside − mean over ±3 s flanks) < 0.2; no flank → kept |

## Which config the scored DEV run used (checked in its job logs)
The scored DEV renders `data/work/protocol_{proposed,blind_a2i}_dev_monocap_v31/` (jobs 30993798 and 31103904, H200) ran
with `V4=590` (Qwen3.8-27B gate, thinking off; depictable label filter), `FLEXSED_BAR 0.8`, `FLEXSED_VETO 0.3`,
**`PANNS_VETO 0.05`**, `ONSET_MONOTONE True`, `MAX_SPAN None`, onset refinement (ONSET_CAM) on, owlv2 stage 2. No 0.40
display floor, `KINSHIP_DIRECTED` off. This is `config.use_scored()`. Every arm below uses exactly this stage-5 and display
setup (`use_scored()`, then the system switches of `run_protocol.configure`).

## Two baseline rows (a fact checked before this plan: the two vetoes differ on DEV)
The candidates were frozen against the **shipped stack**, which uses the BEATs self-veto (b 0.1218, PANNs off) on
FlexSED-only spans. The scored DEV run used the PANNs veto instead. On DEV the two vetoes disagree on 18 of 50 FlexSED-only
spans in 7 clips (self-veto keeps 16 that PANNs dropped, e.g. 11 Insect spans in `b3_pet_shop`; PANNs keeps 2 that the
self-veto drops). So:
- **B0 = the scored render** (PANNs veto). This is what J2 was compared with.
- **B1 = the shipped stack on DEV** = the scored config with the self-veto swap (b 0.1218, PANNs off). **Primary baseline.**
  Every add/change candidate is built on B1 (the stack it was frozen against), so its Δ is only the candidate's change.
- B1 − B0 is reported. The ship rule is read against B1 (primary) and against B0 (beside, like J2).

## Stage 4: how spans are rebuilt
Per clip, from caches on the clip's own `audio.wav`: BEATs = `data/work/j2_dev_beats/<stem>.npz` (the shipped `infer_beats`,
made for J2), FlexSED = the pipeline's `data/work/flexsed_cache/<stem>.npz`.
- **Gate D0 (every clip):** BEATs spans at 0.175 equal the trace's `extract` step; FlexSED spans at 0.8 equal `flexsed_raw`;
  the twin rule ("absorb") rebuilds `union`; after the FlexSED clip veto, the non-FlexSED-only spans equal those of the
  `veto` step. (Same test as J2's D0, one step further.) A clip that fails D0 is reported; its candidate rows are still
  built from the caches.
- **B0 veto:** which FlexSED-only spans the PANNs veto kept is read from the trace's `veto` step (exact).
- **B1 and candidates:** FlexSED clip veto (or DASM's for D1), then the self-veto on the only-second-detector spans.
- **Onset refinement (occlusion, ONSET_CAM, monotone):** as the pipeline, on spans of tagger origin; FlexSED-only and
  DASM-only spans are not refined (frame-level). A span whose (label, start, end) equals a trace `veto`-step span takes the
  trace's refined start verbatim (reuse, no GPU). Only new or changed spans are refined live, with BEATs' `occlusion_onset`.
  For **EAT-R** the same `occlusion_onset` runs with EAT as the model (the function only needs a batch of 2-s windows →
  class probabilities; EAT's own column for the label). For **I4** a parent span is a BEATs label, so refinement runs on the
  parent's own column, exactly as the pipeline would (count of refinements that fired is reported).
- **New caches (GPU):** EAT on the same 2-s / 0.25-s windows as BEATs (round-5 code; times must equal the BEATs cache);
  DASM with the round-6 queries (`data/work/dasm_text_queries.pt`, 215 families) and the round-6 scorer. DEV clips are 11–28 s
  long, longer than the 10-s AudioSet clips: DASM scores consecutive 10-s pieces; the last piece is the clip's final 10 s
  and only its frames after the previous piece are kept (so no piece is zero-padded; the round-6 clarification showed a
  padded 10-ms piece inflates the clip-max, which is DASM's veto input). I6 answers: the round-8 prompt and model on 4
  frames of each DEV mp4.

## Stage 5: how new spans get a gate decision
Stage 5 is re-run for every arm on the H200 (the scored run's card class): `plan_augmentations` and, for ours,
`decide_subjects` (Qwen3.8-27B; per stretch 3 questions, majority), with the scene (`scene.json`) and speech
(`segments.json`) of the scored run. No picture is drawn (scoring reads only label + time; `image_path` is set to an
existing file so the picture counts as shown, as a placeholder does in the scored run).
- **Gate reuse:** a visibility question for a (label, stretch) that the scored run asked (`gate_votes.json`, within 0.01 s)
  takes the scored run's three votes and verdict. Only new or changed stretches are asked live.
- **Other stage-5 questions** (scene, place, speech, plausibility, depiction, dedup) are asked live once and memoised on
  (prompt, frames); every later arm reuses the same answer for the same question. So an unchanged picture gets the same
  answers in every arm, and gate noise cannot make a difference by itself.
- **Gate D5 (repro):** arm B0r = B0's stage-4 spans through this stage-5 path. It must give the scored render's pictures
  (augment flags and spans) on every clip, both systems. D5 is read and reported before any candidate is scored. If some
  clips differ, they are listed; the candidate Δs still compare arms built by the same code (B1 vs candidate), so they
  stay like for like.
- The pipeline without gate (`blind_a2i`) has no VLM step: its arms are `plan_augmentations` only.
- **I7 (removes only):** simulated like J2, not re-gated: a picture span is removed iff every refined stage-4 event of the
  same family (or ancestor/descendant) that overlaps it is a FlexSED-only span dropped by I7; a picture with no span left
  is not shown. Applied to the B1 render (primary) and to the scored render (B0, beside, comparable to J2).

## Metrics (official `score_per_sound`, onset rule, default settings)
For each arm and both systems (ours = proposed; without gate = blind_a2i): hits / 36, wrong pictures by type (visible,
cross-trigger, phantom), duplicates, F1, viewer cost (β = 2: 4 per missed needed sound + 2 per wrong picture, per clip),
coverage. **Δ viewer cost vs B1:** paired clip bootstrap (2000 draws, seed 0), 95 % CI and one-sided p = share of draws with
Δ ≥ 0; Holm across the 5 candidates (family α 0.025 one-sided, as round 8), for ours; the same for the pipeline without
gate, reported. Δ vs B0 reported beside. Also: new gate questions asked, pictures added / removed by name.
**Ship rule (as for J2):** on DEV, for ours, hits do not drop AND wrong pictures (visible + cross + phantom) drop. Read
against B1 (primary) and B0. Report only: nothing is shipped from this check.

## What will not be done
No bar is refitted on DEV; no TEST or slice-B clip is read; no second variant of any candidate.

## Amendment 1 (2026-09-28, Adam via the lead; written while job 31330563 was 35 s old, before any number of this check existed)
**The check is now confirmatory.** DEV replaces the retired AudioSet test (`docs/history/preregistrations/prereg_round12_v2.md`). The candidates
were fixed before DEV was looked at, with the values frozen on the 280. Main goal: more needed sounds heard, fewer real
sounds dropped.

**Three more candidates (round 10, `docs/history/preregistrations/prereg_round10_rescue.md`, `benchmark/detector_round10.py`), values unchanged:**
- **R1 — DASM agrees with a FlexSED 0.4–0.8 band candidate.** Band candidates per FlexSED column: spans at 0.4 (low 0.4,
  min 0.5 s) with peak < 0.8, no BEATs 0.175 span of the same canonical family within 1 s (such a candidate is discarded,
  not merged), BEATs self-veto (b 0.1218). Admitted iff DASM scores the family ≥ 0.359375 at some frame in
  [start − 0.5, end + 0.5] (nearest frame if none). Admitted spans are added to B1's stage-4 output (after the vetoes) as
  FlexSED-only spans (not refined).
- **R6 — I4 confirmed:** I4's parent spans kept only if R1 OR R2 confirms them (DASM / paraphrase columns related by
  `E._same`); an unconfirmed parent span is removed before the twin rule.
- **R7 — I6 confirmed:** I6's lowered-bar spans (listed family, peak < 0.8) kept only if R1 OR R2 confirms them (the
  family's own columns); an unconfirmed one is removed before the twin rule. I6 spans with peak ≥ 0.8 are untouched.
- **R2 (inside R6 / R7)** = both FlexSED paraphrases (`benchmark/round10_paraphrases.json`) score ≥ 0.5 in the span. It is
  computed on DEV (not replaced by R1 only): FlexSED re-queried with the 430 paraphrases on each DEV clip's 16-kHz audio
  (ffmpeg of the mp4, as `flexsed_run.py`) with round 10's own worker `benchmark/round10_flexsed.py` unchanged (checks c1:
  its copy reproduces the DEV FlexSED cache on 2 clips, max diff < 5e-3; c2: query independence < 1e-3; a failure stops).
- DASM on DEV = this check's DEV DASM cache (job 31330562).

**Primary per candidate (8 candidates: EAT-R, D1, I4, I6, I7, R1, R6, R7; ours = proposed, vs B1):**
(a) Δ viewer cost vs B1 (paired clip bootstrap, 2000 draws, seed 0), one-sided p = share of draws with Δ ≥ 0, **Holm
across all 8** (family α 0.025 one-sided): passes iff Holm rejects (which implies upper 95 % CI < 0); AND
(b) **the ship rule (replaces the J2 wording above):** hits do not drop AND wrong pictures (visible + cross + phantom) do not
rise by more than 2 × the hits gained.
A candidate is **better** iff (a) and (b) hold; **worse** iff Δ cost > 0 with lower CI > 0, or (b) fails; otherwise **same**.

**Reported for each candidate, always next to B1 (ours re-run) and B0 (scored):** needed sounds HEARD by stage 4 (before the
gate: the arm's stage-4 spans at or above its display bar, salient non-speech, scored with `score_per_sound` as if every one
were a picture: hits of that = heard); heard-but-dropped rescued (the needed sounds B1-ours misses whose same-family FlexSED
score reaches 0.4–0.8 within ±1 s — round 8's band flag — and that the candidate hits); hits, misses, wrong by type
(visible / cross / phantom), F1, viewer cost, coverage; the same for the pipeline without gate.

**Order:** job 31330563 (arms B0r, B1, EAT-R, D1, I4, I6) then a new job `slurm/job_devcand_extra.sh` (paraphrase
scores, stage 4 and stage 5 for R1, R6, R7, same memo of stage-5 answers, H200), then the score of all arms. D5 is read
first. No output of job 31330563 is read before this amendment is written.

## Results (2026-09-28; `benchmark/gold/dev_candidates_check.json`; jobs 31330562 caches, 31330563 arms B0r–I6, 31330792 R1/R6/R7 + score)
**Gates.** D0: the stage-4 rebuild equals the pipeline trace on 98 / 98 clip × system pairs. **D5: the repro arm B0r gives
the scored render's pictures on 49 / 49 clips for both systems** (same hits, same wrong pictures), so this stage-5 path
reproduces the real pipeline. R2 worker checks: c1 max diff 2.4e-4 (< 5e-3), c2 4e-7 (< 1e-3). Gate questions: the
scored run's answers were reused 13–126 times per arm; new questions asked live: B1 8, EAT-R 163, D1 70, I4 24, I6 48, R1 29,
R6 22, R7 45. Filters on DEV: R1 admitted 29 of 50 band candidates, R6 23 of 35 parent spans, R7 99 of 129 lowered-bar spans.
"Heard" = needed sounds hit by the arm's stage-4 spans (display bar, salient) before the gate. "Rescued" = needed sounds in
the heard-but-dropped group (B1-ours misses them; FlexSED 0.4–0.8 within ±1 s) that the arm hits.

*ours (with gate)* (heard-but-dropped group = 13 needed sounds)

| arm | heard by stage 4 | hits / 36 | misses | rescued | wrong (visible / cross / phantom) | F1 | viewer cost | Δ cost vs B1 [95 % CI], p | ship rule vs B1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 scored (PANNs veto) | 17 | 14 | 22 | 0 | 24 (6 / 11 / 7) | 0.378 | 2.78 | -0.33 [-0.69, -0.04], p 0.015 | — | — |
| **B1 ours re-run (self-veto)** | 16 | 13 | 23 | 0 | 30 (6 / 18 / 6) | 0.329 | 3.10 | — | — | — |
| EAT-R | 19 | 14 | 22 | 3 | 51 (10 / 26 / 15) | 0.277 | 3.88 | +0.78 [-0.00, +1.63], p 0.975 | fail | worse |
| DASM D1 | 13 | 10 | 26 | 2 | 37 (6 / 22 / 9) | 0.241 | 3.63 | +0.53 [-0.29, +1.39], p 0.903 | fail | worse |
| I4 | 16 | 14 | 22 | 1 | 36 (6 / 22 / 8) | 0.326 | 3.27 | +0.16 [+0.00, +0.37], p 0.982 | fail | worse |
| I6 | 17 | 14 | 22 | 1 | 33 (6 / 19 / 8) | 0.337 | 3.14 | +0.04 [-0.53, +0.57], p 0.590 | fail | worse |
| I7 | 15 | 12 | 24 | 0 | 26 (6 / 14 / 6) | 0.324 | 3.02 | -0.08 [-0.41, +0.29], p 0.343 | fail | worse |
| R1 | 18 | 16 | 20 | 2 | 37 (6 / 25 / 6) | 0.360 | 3.14 | +0.04 [-0.49, +0.49], p 0.619 | fail | worse |
| R6 | 16 | 14 | 22 | 1 | 35 (6 / 21 / 8) | 0.329 | 3.22 | +0.12 [-0.04, +0.33], p 0.960 | fail | worse |
| R7 | 17 | 14 | 22 | 1 | 33 (6 / 19 / 8) | 0.337 | 3.14 | +0.04 [-0.53, +0.57], p 0.590 | fail | worse |

*pipeline without gate* (heard-but-dropped group = 11 needed sounds)

| arm | heard by stage 4 | hits / 36 | misses | rescued | wrong (visible / cross / phantom) | F1 | viewer cost | Δ cost vs B1 [95 % CI], p | ship rule vs B1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 scored (PANNs veto) | 17 | 17 | 19 | 0 | 50 (17 / 21 / 12) | 0.330 | 3.59 | -0.33 [-0.69, -0.04], p 0.015 | — | — |
| **B1 ours re-run (self-veto)** | 16 | 16 | 20 | 0 | 56 (17 / 28 / 11) | 0.296 | 3.92 | — | — | — |
| EAT-R | 19 | 17 | 19 | 3 | 85 (22 / 40 / 23) | 0.246 | 5.02 | +1.10 [+0.08, +2.20], p 0.983 | fail | worse |
| DASM D1 | 13 | 13 | 23 | 2 | 70 (18 / 35 / 17) | 0.218 | 4.73 | +0.82 [+0.00, +1.67], p 0.977 | fail | worse |
| I4 | 16 | 16 | 20 | 0 | 67 (17 / 34 / 16) | 0.269 | 4.37 | +0.45 [+0.20, +0.73], p 1.000 | fail | worse |
| I6 | 17 | 17 | 19 | 1 | 62 (19 / 29 / 14) | 0.296 | 4.08 | +0.16 [-0.45, +0.78], p 0.726 | fail | worse |
| I7 | 15 | 15 | 21 | 0 | 52 (17 / 24 / 11) | 0.291 | 3.84 | -0.08 [-0.41, +0.29], p 0.343 | fail | worse |
| R1 | 18 | 18 | 18 | 2 | 68 (17 / 39 / 12) | 0.295 | 4.24 | +0.33 [-0.12, +0.73], p 0.934 | fail | worse |
| R6 | 16 | 16 | 20 | 0 | 65 (17 / 33 / 15) | 0.274 | 4.29 | +0.37 [+0.16, +0.61], p 1.000 | fail | worse |
| R7 | 17 | 17 | 19 | 1 | 61 (18 / 29 / 14) | 0.298 | 4.04 | +0.12 [-0.45, +0.69], p 0.685 | fail | worse |

Holm over the 8 candidates (α 0.025 one-sided): **no candidate is rejected** (smallest p: I7 0.34 for ours). I6 and R7 give
the same pictures on DEV (R7's filter removed only spans that never became pictures).

**Verdict: no candidate is better. All 8 are worse by the ship rule** (hits do not drop AND wrong pictures rise by at most
2 × the hits gained). EAT-R, I4, I6, R6, R7 each gain 1 hit for +3 to +21 wrong pictures. D1 loses 3 hits. I7 removes 4 wrong
pictures but also loses 1 hit. **R1 is the only one that finds clearly more needed sounds** (hits 13 → 16, heard 16 → 18;
new hits include explosions in `as_explosion_XJ8lc3I6`), but it adds 7 wrong pictures (> 2 × 3 = 6), and its cost does not
move (+0.04 [−0.49, +0.49]). No candidate has a Δ cost CI below 0.

**Side finding (not a candidate): the shipped self-veto is worse than the scored PANNs veto on DEV.** B1 (self-veto 0.1218,
as `use_shipped()`) vs B0 (PANNs 0.05, the scored run): hits 13 vs 14, wrong 30 vs 24, cost 3.10 vs 2.78,
**B0 − B1 = −0.33 [−0.69, −0.04]** (both systems). The self-veto lets 8 FlexSED-only pictures through that PANNs dropped
(Insect ×5 in `b3_pet_shop` / `ambient_nature_rainforest_7629`, Coin, Bicycle, Snoring), and drops 2 that PANNs kept (Train,
Laughter); one Cricket picture becomes an Insect picture. Round 4 found the two vetoes equal on the 415 (ΔC +0.005 [−0.048, +0.067]); on our DEV videos they are not.
Against B0 the ship rule also fails for all 8 candidates.
