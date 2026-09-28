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
