# Round 13: 12-hour detector push (28–29 Sept 2026) — protocol fixed before any new DEV number

*Written 2026-09-28 at the start of the block, before any round-13 idea was run on any clip. Adam: "for the next 12h
you will only try to improve the sounds recognition (decrease needed dropped or increase unheard and so on); consult
with Fable about ideas and suggest yourself and try them."*

## Goal
More needed sounds heard and shown (hits), without more wrong pictures than the hits are worth
(viewer cost = 4 × miss + 2 × wrong, per clip, `benchmark/gold/score_per_sound.py`).

## Baseline
**B0 = the scored config (PANNs veto 0.05)**, i.e. the config the frozen TEST table was scored with and the better arm on
DEV (`docs/dev_candidates_check_2026-09-28.md`: 14 hits, 24 wrong, cost 2.78). Every round-13 candidate is built on top
of B0, so a TEST comparison is not confounded by the veto swap. B1 (self-veto, current `use_shipped()`) is reported
beside. Whether `use_shipped()` returns to the PANNs veto is Adam's decision (TODO), not part of this round.

## Status of the data
- **DEV (49 clips, 36 needed sounds) is spent as a confirmation set.** In this round it is the DEVELOPMENT set: every DEV
  number is exploratory; ideas may be tuned on it.
- **AudioSet (280 / 415 / fresh)** may be used only as a component screen (e.g. does a verifier separate the 61 band
  events), never as a decision.
- **TEST: exactly ONE exposure, at the end of the block, of exactly ONE candidate** (or none), picked by the DEV rule
  below. No second TEST run of any variant. No slice-B clip is read.

## DEV selection rule (to pick the one TEST candidate)
Among round-13 candidates, full pipeline on DEV (stage 4 → gate → shipped display rules, ours re-run in the same job):
eligible iff hits ≥ B0 hits + 1 AND wrong ≤ B0 wrong + 2 × (hits gained) AND cost < B0 cost. Pick the lowest DEV cost;
ties → fewer wrong. If none is eligible, TEST is not touched and the round reports "no candidate".

## TEST decision (one exposure)
Candidate vs B0 re-run in the same job on the TEST split, same scorer, paired clip bootstrap 2000 (seed 0).
**Better** iff hits do not drop AND wrong does not rise by more than 2 × hits gained AND Δ cost < 0 with one-sided
p < 0.05. **Worse** iff Δ cost > 0 with lower 95 % bound > 0 or the hits/wrong rule fails. Otherwise **same**.
Reported always next to B0 and B1: heard by stage 4, hits, misses, wrong (visible / cross / phantom), cost.
Only "better" ships (behind a flag, turned on in `use_shipped()`); the frozen TEST table stays as scored and the new row
is added beside it with this disclosure.

## Log
Every idea tried in this block is listed below with its DEV row (kept even when it fails), so the number of looks at DEV
is on record.

### 2026-09-28/29 — rules R13-1, R13-2, R13-5, R13-6 (Fable consult), DEV, job 31340247 (H200, 20 min)
**What.** Each rule is in the pipeline source behind its own flag in `config.py`, default OFF: `TWIN_MAX` (R13-1),
`MIRROR_VETO` b + `MIRROR_OWN_MAX` 0.4 (R13-2), `IMPULSE_MIN_SPAN` 0.2 + `IMPULSE_FAMILIES` (R13-5), `RETRIGGER`
(1.5 s, FlexSED 0.4, BEATs 0.175) + `RETRIGGER_RAW` (R13-6). Stage 4's union/veto block is now one function,
`fuse_flexsed` (`src/stage4_audio_event_detection/__init__.py`), called by `detect_events` and by the harness
`benchmark/gold/round13_dev.py` on the cached scores, so each arm runs the pipeline's own code. R13-6 records per-family
"breaks" on `AudioEvent`/`AugmentationSpec` (`breaks` field); `merge_by_label` and `_display_spans` never join two bursts
across one (`src/labels.py`, `src/stage6_visual_augmentation/__init__.py`; `score_per_sound.load_pictures` passes it).
Every arm = B0's stage-4 env set explicitly (FLEXSED_BAR 0.8, FLEXSED_VETO 0.3, PANNS_VETO 0.05, BEATS_SELF_VETO 0, MONO,
no cap) + its flags; stage 5 as `dev_candidates_check` (scored gate answers reused, other questions memoised).
**Gates.** Flags off: D0 98/98 (extract, flexsed_raw, union, veto vs the scored trace, PANNs veto recomputed with CNN14),
label/start/end/conf/origin equal to devcand B0r 98/98, **D5 49/49 both systems** (B0r = the scored render).
**Facts found before the run.**
- R13-1 makes 39 absorbed sub-display BEATs spans displayable (every FlexSED twin is >= 0.8 by construction).
- R13-5: FlexSED has queries only for Gunshot, Gasp, Hammer, Explosion, Knock (none for Whack, Clang, Slam, Bang). At
  FlexSED's 0.8 bar none of the four (b)-bucket misses (0.42–0.76) can pass; the rule adds one Gasp span on DEV
  (`b3_ia_youtube_skxtz9foauw_0` 22.44 s), which does not become a picture.
- R13-6: a single stage-4 span cannot hold a 1.5-s evidence gap (BEATs column >= 0.175 throughout, FlexSED >= 0.8, twin
  tolerance 1 s), so the rule can act only where the display merge bridges 1.5–2.0 s. The ambulance Vehicle 0–14.75 s is
  one BEATs run: not splittable. **The Crowd case is not a merge of two families.** Trace of `ly_applause_62ZYD0u`: BEATs
  Applause 6.0–7.75; FlexSED Crowd 0–1.04 and 1.12–11.12; canonical(Applause) = Crowd, so the twin rule pulled Applause to
  1.12 and `consolidate_families` chained the FlexSED-only Crowd 0–1.04 into it (gap 0.08 s) → one Crowd picture at 0.0.
  Minimal fix implemented (`RETRIGGER_RAW`): a later firing of a different sound (not same label / ancestor /
  descendant) is not chained, and the boundary is a break. No detector puts an onset in the hit window [1.4, 2.9] s
  (FlexSED Crowd >= 0.4 runs 13 s from 0.0), so no merge fix can make this a hit. R13-6ev = the evidence rule alone.

*ours (with gate), DEV 49 clips, 36 needed sounds; Δ = paired clip bootstrap 2000, seed 0; eligible = the DEV selection
rule above (hits >= B0 + 1, wrong <= B0 + 2 × gain, cost < B0)*

| arm | heard by stage 4 | hits / 36 | misses | wrong (vis / cross / phantom) | cost | Δ cost vs B0r [95 % CI] | DEV-eligible |
|---|---|---|---|---|---|---|---|
| B0 | 17 | 14 | 22 | 24 (6 / 11 / 7) | 2.78 | +0.00 [+0.00, +0.00] | — |
| B0r | 17 | 14 | 22 | 24 (6 / 11 / 7) | 2.78 | — | — |
| B1 | 16 | 13 | 23 | 30 (6 / 18 / 6) | 3.10 | +0.33 [+0.04, +0.69] | — |
| R13-1 | 18 | 15 | 21 | 24 (6 / 11 / 7) | 2.69 | -0.08 [-0.37, +0.12] | yes |
| R13-2b08 | 16 | 13 | 23 | 22 (6 / 9 / 7) | 2.78 | +0.00 [-0.12, +0.12] | no |
| R13-2b07 | 16 | 13 | 23 | 20 (6 / 8 / 6) | 2.69 | -0.08 [-0.24, +0.08] | no |
| R13-5 | 17 | 14 | 22 | 24 (6 / 11 / 7) | 2.78 | +0.00 [+0.00, +0.00] | no |
| R13-6 | 17 | 14 | 22 | 25 (6 / 12 / 7) | 2.82 | +0.04 [+0.00, +0.12] | no |
| R13-6ev | 17 | 14 | 22 | 24 (6 / 11 / 7) | 2.78 | +0.00 [+0.00, +0.00] | no |
| STACK12b08 | 17 | 14 | 22 | 22 (6 / 9 / 7) | 2.69 | -0.08 [-0.41, +0.12] | no |
| STACK12b07 | 17 | 14 | 22 | 20 (6 / 8 / 6) | 2.61 | -0.16 [-0.49, +0.08] | no |
| STACK1256b08 | 17 | 14 | 22 | 23 (6 / 10 / 7) | 2.73 | -0.04 [-0.37, +0.20] | no |
| STACK1256b07 | 17 | 14 | 22 | 21 (6 / 9 / 6) | 2.65 | -0.12 [-0.45, +0.16] | no |

- **R13-1**: needed gained: `ambient_nature_rainforest_7629` Cricket 0.1 s. Lost: none. Wrong pictures appeared: `bell_miami` Train 8.00–15.00 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50–6.25 (cross). Other picture changes (hit/dup): +`ambient_nature_rainforest_7629` Insect 0.14 (hit) / -none.
- **R13-2b08**: needed gained: none. Lost: `mv_protest_scene_movie` Glass 11.0 s. Wrong pictures appeared: `b3_laundromat` Vehicle 0.30–2.50 (phantom). Disappeared: `b3_favela_rio` Bird 14.00–15.50 (cross); `b3_laundromat` Vehicle 0.30–11.50 (phantom); `mv_protest_scene_movie` Glass 4.75–7.75 (cross). Other picture changes (hit/dup): +none / -`mv_protest_scene_movie` Glass 10.75 (hit).
- **R13-2b07**: needed gained: none. Lost: `mv_protest_scene_movie` Glass 11.0 s. Wrong pictures appeared: `b3_laundromat` Train 1.00–2.50 (phantom); `b3_laundromat` Train 16.25–17.75 (phantom). Disappeared: `b3_favela_rio` Bird 14.00–15.50 (cross); `b3_laundromat` Vehicle 0.30–11.50 (phantom); `b3_laundromat` Vehicle 14.25–17.75 (phantom); `b3_laundromat` Vehicle 23.25–27.75 (phantom); `mv_protest_scene_movie` Glass 4.75–7.75 (cross); `mv_tornado_scene` Vehicle 7.25–9.75 (cross). Other picture changes (hit/dup): +none / -`mv_protest_scene_movie` Glass 10.75 (hit).
- **R13-5**: needed gained: none. Lost: none. Wrong pictures appeared: none. Disappeared: none. Other picture changes (hit/dup): +none / -none.
- **R13-6**: needed gained: none. Lost: none. Wrong pictures appeared: `ly_applause_62ZYD0u` Crowd 0.00–1.50 (cross); `ly_applause_62ZYD0u` Crowd 1.12–7.75 (cross). Disappeared: `ly_applause_62ZYD0u` Crowd 0.00–7.75 (cross). Other picture changes (hit/dup): +none / -none.
- **R13-6ev**: needed gained: none. Lost: none. Wrong pictures appeared: none. Disappeared: none. Other picture changes (hit/dup): +none / -none.
- **STACK12b08**: needed gained: `ambient_nature_rainforest_7629` Cricket 0.1 s. Lost: `mv_protest_scene_movie` Glass 11.0 s. Wrong pictures appeared: `b3_laundromat` Vehicle 0.30–2.50 (phantom); `bell_miami` Train 8.00–15.00 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50–6.25 (cross); `b3_favela_rio` Bird 14.00–15.50 (cross); `b3_laundromat` Vehicle 0.30–11.50 (phantom); `mv_protest_scene_movie` Glass 4.75–7.75 (cross). Other picture changes (hit/dup): +`ambient_nature_rainforest_7629` Insect 0.14 (hit) / -`mv_protest_scene_movie` Glass 10.75 (hit).
- **STACK12b07**: needed gained: `ambient_nature_rainforest_7629` Cricket 0.1 s. Lost: `mv_protest_scene_movie` Glass 11.0 s. Wrong pictures appeared: `b3_laundromat` Train 1.00–2.50 (phantom); `b3_laundromat` Train 16.25–17.75 (phantom); `bell_miami` Train 8.00–15.00 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50–6.25 (cross); `b3_favela_rio` Bird 14.00–15.50 (cross); `b3_laundromat` Vehicle 0.30–11.50 (phantom); `b3_laundromat` Vehicle 14.25–17.75 (phantom); `b3_laundromat` Vehicle 23.25–27.75 (phantom); `mv_protest_scene_movie` Glass 4.75–7.75 (cross); `mv_tornado_scene` Vehicle 7.25–9.75 (cross). Other picture changes (hit/dup): +`ambient_nature_rainforest_7629` Insect 0.14 (hit) / -`mv_protest_scene_movie` Glass 10.75 (hit).
- **STACK1256b08**: needed gained: `ambient_nature_rainforest_7629` Cricket 0.1 s. Lost: `mv_protest_scene_movie` Glass 11.0 s. Wrong pictures appeared: `b3_laundromat` Vehicle 0.30–2.50 (phantom); `bell_miami` Train 8.00–15.00 (cross); `ly_applause_62ZYD0u` Crowd 0.00–1.50 (cross); `ly_applause_62ZYD0u` Crowd 1.12–7.75 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50–6.25 (cross); `b3_favela_rio` Bird 14.00–15.50 (cross); `b3_laundromat` Vehicle 0.30–11.50 (phantom); `ly_applause_62ZYD0u` Crowd 0.00–7.75 (cross); `mv_protest_scene_movie` Glass 4.75–7.75 (cross). Other picture changes (hit/dup): +`ambient_nature_rainforest_7629` Insect 0.14 (hit) / -`mv_protest_scene_movie` Glass 10.75 (hit).
- **STACK1256b07**: needed gained: `ambient_nature_rainforest_7629` Cricket 0.1 s. Lost: `mv_protest_scene_movie` Glass 11.0 s. Wrong pictures appeared: `b3_laundromat` Train 1.00–2.50 (phantom); `b3_laundromat` Train 16.25–17.75 (phantom); `bell_miami` Train 8.00–15.00 (cross); `ly_applause_62ZYD0u` Crowd 0.00–1.50 (cross); `ly_applause_62ZYD0u` Crowd 1.12–7.75 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50–6.25 (cross); `b3_favela_rio` Bird 14.00–15.50 (cross); `b3_laundromat` Vehicle 0.30–11.50 (phantom); `b3_laundromat` Vehicle 14.25–17.75 (phantom); `b3_laundromat` Vehicle 23.25–27.75 (phantom); `ly_applause_62ZYD0u` Crowd 0.00–7.75 (cross); `mv_protest_scene_movie` Glass 4.75–7.75 (cross); `mv_tornado_scene` Vehicle 7.25–9.75 (cross). Other picture changes (hit/dup): +`ambient_nature_rainforest_7629` Insect 0.14 (hit) / -`mv_protest_scene_movie` Glass 10.75 (hit).
- **B1**: needed gained: none. Lost: `un_driving_motorcycle_4O3bZRYO` Laughter 12.7 s. Wrong pictures appeared: `ambient_nature_rainforest_7629` Insect 4.50–6.25 (cross); `ambient_nature_rainforest_7629` Insect 12.08–16.00 (cross); `as_glass_oHil9Ip_` Coin (dropping) 9.00–10.50 (cross); `b3_favela_rio` Bicycle 17.84–19.52 (cross); `b3_pet_shop` Insect 0.00–2.76 (cross); `b3_pet_shop` Insect 7.84–19.28 (cross); `b3_pet_shop` Insect 22.60–24.32 (cross); `mv_detective_crime_scene` Snoring 2.64–4.14 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50–6.25 (cross); `mv_protest_scene_movie` Train 21.48–22.98 (phantom). Other picture changes (hit/dup): +none / -`un_driving_motorcycle_4O3bZRYO` Laughter 13.36 (hit).

Pipeline without gate (heard / hits / wrong / cost / Δ vs B0r): B0r 17/17/50/3.59; B1 16/16/56/3.92; R13-1 18/18/53/3.63
(+0.04 [−0.20, +0.24]); R13-2b08 16/16/46/3.51; R13-2b07 16/16/42/3.35 (−0.24 [−0.53, −0.04]); R13-5 = B0r; R13-6
17/17/52/3.67; R13-6ev = B0r; STACK12b08 17/17/50/3.59; STACK12b07 17/17/46/3.43 (−0.16 [−0.53, +0.12]); STACK1256b08
17/17/52/3.67; STACK1256b07 17/17/48/3.51.

**Best b = 0.7** (R13-2b07 cost 2.69 vs b 0.8 2.78), so **STACK12 = STACK12b07, STACK1256 = STACK1256b07**; the b 0.8
stacks are kept above as run. **Reading.** Only **R13-1** passes the DEV selection rule (+1 hit, Cricket; wrong 24 → 24:
the late Cricket cross picture goes, a Train cross picture in `bell_miami` comes; cost 2.78 → 2.69, Δ −0.08 [−0.37,
+0.12]). R13-2 removes the laundromat Vehicle phantoms (all three at b 0.7) and two cross pictures, but also drops the
needed Glass hit in `mv_protest_scene_movie` (BEATs Shatter/Glass where FlexSED's top query is another family), so hits
fall; at b 0.7 two laundromat Train phantoms appear in place of the Vehicle ones. STACK12b07 has the lowest cost (2.61,
Δ −0.16 [−0.49, +0.08]) but is not eligible: hits stay 14 (Cricket gained, Glass lost). R13-5 and R13-6ev change no
picture; R13-6 is worse by one cross picture (the Crowd split, as predicted). No CI excludes 0.
Files: `benchmark/gold/round13_dev.py`, `benchmark/gold/round13_dev.json`, `slurm/job_round13.sh`, `data/work/r13/`.
On the cluster the round-13 source is only in `~/MscProj_r13` (data/ and ckpts/ link to `~/MscProj`): `~/MscProj` was
not touched because the listener job 31334159 was running; the 9 changed files must be synced into `~/MscProj` after it ends.
R13-6 did not add a general "never merge two families" rule: the only cross-family merge in the pipeline is `_dedup`'s
depiction-similarity merge, left unchanged.

### 2026-09-29 — R13-3 listener rescue (coordinator's rule), DEV, job 31360244 (H200, 3 h 15 min)
**What.** Flag `LISTENER_RESCUE` (default off) + `LISTENER_CACHE` (`benchmark/gold/dev_listener.json`, Qwen3-Omni yes-no logit
per span), `LISTENER_LO`, `LISTENER_TH`, `LISTENER_BEATS_TH`, in `fuse_flexsed` (`src/stage4_audio_event_detection/__init__.py`;
pipeline: `_pipeline_listener`, clip id = work-dir name). (a) A FlexSED 0.4-run (gaps <= 0.24 s merged, as the cache's P2)
with peak in [LO, 0.8), no same-family span over it and score > TH is added as a FlexSED-only span after the PANNs veto (its
run; the cache's 1-s cut when shorter than 0.5 s). (b) A FlexSED >= 0.8 span the PANNs clip veto would drop is kept when the
P2 run containing it scores > TH. (c) With `LISTENER_BEATS_TH` 3: a BEATs run (peak 0.175–0.35, not covered, = the cache's
P3) is added at the display bar 0.35 (onset-refined like any BEATs span). Missing cache key = not rescued, counted.
Arms: LO {0.4, 0.5} × TH {0, 2, 3}, each alone, + R13-1, + P3, + R13-1 + P3 (24 arms), all on B0r.
**Gates.** Flags off: D0 98/98, conf-equality 98/98, D5 49/49 both systems (unchanged).
**Cache coverage.** (a): 0 missing of 512 runs asked at LO 0.5 (0 of 1135 at LO 0.4); R13-1 changes no span, so
P1 coverage and the lookups are the same with + R13-1. (b): 34 vetoed spans asked, 1 missing (`ambient_nature_rainforest_7629`
Insect 12.08–16.00: its 0.4-run overlaps a B0r Insect span, so it is not in P2; not a needed sound here). (c): 0 missing
of 636. The Laughter (b) case is in the cache: score 2.875, kept at TH 0 and 2, not at 3.

*ours (with gate)*

| arm | heard | hits / 36 | misses | wrong (vis / cross / phantom) | cost | Δ cost vs B0r [95 % CI] | rescued (a) / kept (b) / BEATs (c) spans |
|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 22 | 24 (6 / 11 / 7) | 2.78 | — | — |
| B1 | 16 | 13 | 23 | 30 (6 / 18 / 6) | 3.10 | +0.33 [+0.04, +0.69] | — |
| R13-1 | 18 | 15 | 21 | 24 (6 / 11 / 7) | 2.69 | -0.08 [-0.37, +0.12] | — |
| L04t0 | 26 | 19 | 17 | 176 (13 / 113 / 50) | 8.57 | +5.80 [+3.92, +7.80] | 399 / 10 / 0 |
| L04t0+1 | 27 | 19 | 17 | 177 (13 / 114 / 50) | 8.61 | +5.84 [+4.00, +7.80] | 399 / 10 / 0 |
| L04t0+P3 | 26 | 19 | 17 | 179 (14 / 114 / 51) | 8.69 | +5.92 [+4.04, +7.88] | 399 / 10 / 107 |
| L04t0+1+P3 | 27 | 19 | 17 | 180 (14 / 115 / 51) | 8.73 | +5.96 [+4.12, +7.92] | 399 / 10 / 107 |
| L04t2 | 26 | 20 | 16 | 103 (9 / 67 / 27) | 5.51 | +2.73 [+1.47, +4.08] | 210 / 9 / 0 |
| L04t2+1 | 27 | 20 | 16 | 104 (9 / 68 / 27) | 5.55 | +2.78 [+1.51, +4.08] | 210 / 9 / 0 |
| L04t2+P3 | 26 | 20 | 16 | 108 (10 / 70 / 28) | 5.71 | +2.94 [+1.59, +4.37] | 210 / 9 / 107 |
| L04t2+1+P3 | 27 | 20 | 16 | 109 (10 / 71 / 28) | 5.76 | +2.98 [+1.63, +4.37] | 210 / 9 / 107 |
| L04t3 | 24 | 20 | 16 | 79 (8 / 51 / 20) | 4.53 | +1.76 [+0.61, +2.82] | 142 / 8 / 0 |
| L04t3+1 | 25 | 20 | 16 | 80 (8 / 52 / 20) | 4.57 | +1.80 [+0.69, +2.86] | 142 / 8 / 0 |
| L04t3+P3 | 24 | 20 | 16 | 87 (9 / 57 / 21) | 4.86 | +2.08 [+0.90, +3.22] | 142 / 8 / 107 |
| L04t3+1+P3 | 25 | 20 | 16 | 88 (9 / 58 / 21) | 4.90 | +2.12 [+0.94, +3.27] | 142 / 8 / 107 |
| L05t0 | 25 | 22 | 14 | 124 (13 / 81 / 30) | 6.20 | +3.43 [+1.88, +5.06] | 211 / 10 / 0 |
| L05t0+1 | 26 | 22 | 14 | 125 (13 / 82 / 30) | 6.24 | +3.47 [+1.88, +5.10] | 211 / 10 / 0 |
| L05t0+P3 | 25 | 21 | 15 | 130 (14 / 85 / 31) | 6.53 | +3.76 [+2.20, +5.39] | 211 / 10 / 109 |
| L05t0+1+P3 | 26 | 21 | 15 | 131 (14 / 86 / 31) | 6.57 | +3.80 [+2.24, +5.47] | 211 / 10 / 109 |
| L05t2 | 25 | 22 | 14 | 81 (9 / 53 / 19) | 4.45 | +1.67 [+0.49, +2.78] | 129 / 9 / 0 |
| L05t2+1 | 26 | 22 | 14 | 82 (9 / 54 / 19) | 4.49 | +1.71 [+0.49, +2.82] | 129 / 9 / 0 |
| L05t2+P3 | 25 | 21 | 15 | 89 (10 / 59 / 20) | 4.86 | +2.08 [+0.94, +3.22] | 129 / 9 / 109 |
| L05t2+1+P3 | 26 | 21 | 15 | 90 (10 / 60 / 20) | 4.90 | +2.12 [+0.98, +3.27] | 129 / 9 / 109 |
| L05t3 | 24 | 22 | 14 | 65 (8 / 43 / 14) | 3.80 | +1.02 [-0.12, +1.88] | 95 / 8 / 0 |
| L05t3+1 | 25 | 22 | 14 | 66 (8 / 44 / 14) | 3.84 | +1.06 [-0.08, +1.92] | 95 / 8 / 0 |
| L05t3+P3 | 24 | 22 | 14 | 74 (9 / 50 / 15) | 4.16 | +1.39 [+0.20, +2.37] | 95 / 8 / 109 |
| L05t3+1+P3 | 25 | 22 | 14 | 75 (9 / 51 / 15) | 4.20 | +1.43 [+0.24, +2.41] | 95 / 8 / 109 |

Pipeline without gate (hits / wrong / cost): B0r 17/50/3.59; L04t0 23/229/10.41; L04t2 24/149/7.06; L04t3 23/127/6.24; L05t0 25/170/7.84; L05t2 25/123/5.92; L05t3 24/111/5.51.

- **L05t3**: needed gained `ambient_citywalk_nyc_1689` Hammer 13.7 s; `ambient_nature_rainforest_7629` Bird 0.1 s; `as_explosion_XJ8lc3I6` Gunshot, gunfire 0.0 s; `as_explosion_XJ8lc3I6` Walk, footsteps 2.1 s; `as_explosion_XJ8lc3I6` Explosion 2.8 s; `as_explosion_XJ8lc3I6` Explosion 5.6 s; `as_explosion_XJ8lc3I6` Gasp 6.7 s; `b3_carnival_parade` Whistle 6.1 s. Lost none. Wrong pictures appeared 42 (visible 2, cross 33, phantom 7; most common labels Bird ×5, Honk ×4, Footsteps ×3, Shout ×3, Bell ×2, Coin (dropping) ×2; the full list with clip and time is in `round13_dev.json` `picture_changes`). Disappeared: `mv_detective_crime_scene` Alarm 3.92 (cross).
- **L05t3+P3**: needed gained `ambient_citywalk_nyc_1689` Hammer 13.7 s; `ambient_nature_rainforest_7629` Bird 0.1 s; `as_explosion_XJ8lc3I6` Gunshot, gunfire 0.0 s; `as_explosion_XJ8lc3I6` Walk, footsteps 2.1 s; `as_explosion_XJ8lc3I6` Explosion 2.8 s; `as_explosion_XJ8lc3I6` Explosion 5.6 s; `as_explosion_XJ8lc3I6` Gasp 6.7 s; `b3_carnival_parade` Whistle 6.1 s. Lost none. Wrong pictures appeared 57 (visible 3, cross 45, phantom 9; most common labels Bird ×9, Honk ×4, Footsteps ×3, Shout ×3, Bell ×2, Gunshot ×2; the full list with clip and time is in `round13_dev.json` `picture_changes`). Disappeared: `as_explosion_XJ8lc3I6` Gunshot 8.25 (cross); `b3_barbershop` Electric shaver, electric razor 16.00 (cross); `b3_laundromat` Vehicle 14.25 (phantom); `ly_applause_62ZYD0u` Crowd 0.00 (cross); `mv_detective_crime_scene` Alarm 3.92 (cross); `mv_detective_crime_scene` Alarm 9.08 (cross); `mv_protest_scene_movie` Glass 4.75 (cross).
- **L05t0**: needed gained `ambient_citywalk_nyc_1689` Hammer 13.7 s; `ambient_nature_rainforest_7629` Bird 0.1 s; `ambient_snow_walk_930` Laughter 8.1 s; `as_explosion_XJ8lc3I6` Gunshot, gunfire 0.0 s; `as_explosion_XJ8lc3I6` Walk, footsteps 2.1 s; `as_explosion_XJ8lc3I6` Explosion 2.8 s; `as_explosion_XJ8lc3I6` Explosion 5.6 s; `as_explosion_XJ8lc3I6` Gasp 6.7 s. Lost none. Wrong pictures appeared 106 (visible 7, cross 73, phantom 26; most common labels Footsteps ×13, Shout ×7, Bird ×6, Bell ×5, Air horn, truck horn ×5, Honk ×4; the full list with clip and time is in `round13_dev.json` `picture_changes`). Disappeared: `as_fire_alarm_kGKZ0YK4` Alarm 8.89 (visible); `b3_bakery_morning` Chink, clink 0.14 (phantom); `b3_favela_rio` Train 3.80 (cross); `b3_laundromat` Vehicle 23.25 (phantom); `mv_detective_crime_scene` Alarm 3.92 (cross); `mv_protest_scene_movie` Siren 22.75 (phantom); `mv_tornado_scene` Vehicle 7.25 (cross).
- **L04t2**: needed gained `ambient_citywalk_nyc_1689` Hammer 13.7 s; `ambient_nature_rainforest_2179` Bird 6.5 s; `ambient_nature_rainforest_7629` Bird 0.1 s; `ambient_snow_walk_930` Laughter 8.1 s; `as_explosion_XJ8lc3I6` Gunshot, gunfire 0.0 s; `as_explosion_XJ8lc3I6` Walk, footsteps 2.1 s; `as_explosion_XJ8lc3I6` Gasp 6.7 s. Lost `un_driving_motorcycle_4O3bZRYO` Laughter 12.7 s. Wrong pictures appeared 83 (visible 4, cross 59, phantom 20; most common labels Footsteps ×9, Bird ×7, Shout ×7, Air brake ×5, Honk ×4, Bell ×3; the full list with clip and time is in `round13_dev.json` `picture_changes`). Disappeared: `as_explosion_XJ8lc3I6` Gunshot 8.25 (cross); `london_protest_01` Vehicle 0.25 (visible); `mv_detective_crime_scene` Alarm 3.92 (cross); `mv_tornado_scene` Vehicle 7.25 (cross).

**Reading.** The listener finds the band misses: **up to 8 more hits (14 → 22 at LO 0.5)**: the explosions, gunshot,
gasp, footsteps, hammer, whistle and the rainforest bird; Laughter at TH <= 2. But every arm adds far more wrong pictures
than 2 × the hits gained, mostly cross pictures of real sounds at the wrong time (Bird, Footsteps, Shout, Honk, Bell).
**No R13-3 arm is DEV-eligible, and every Δ cost is > 0.** Best R13-3 arm = **L05t3** (LO 0.5, TH 3): hits 22, wrong 65,
cost 3.80, Δ +1.02 [−0.12, +1.88]. + R13-1 adds nothing (the Cricket is already rescued; one more wrong). + P3 on the best
arm (L05t3+P3) is worse: 109 BEATs-band spans admitted, cost 4.16, Δ +1.39 [+0.20, +2.37]. (At LO 0.5 with TH 0 and 2, + P3 costs one
hit, 22 → 21, by the scorer's greedy matching.)
A higher bar or a second check on the admitted spans would be needed before this rescue can pay.

### Amendment A (2026-09-29, written BEFORE any output of these variants exists) — stricter listener questions
The yes/no listener (R13-3) saturates: wrong rescues score 3.25–7.4, hits 3.5–6.0. Fable consult: the question tests
"is X plausible here", not "is X in this cut". Four variants are scored in ONE Qwen3-Omni job on the same candidates
(DEV: FlexSED runs peak >= 0.5 of the P2 pool + the PANNs-vetoed >= 0.8 spans; TEST: the same construction on the
gold-free superset, features only, no scoring). Same model, same audio cut as R13-3 unless stated.
- **V1 multiple choice.** "Listen carefully. Which ONE of these is actually present in this recording? A) {X} B) {s1}
  C) {s2} D) {u} E) none of A-D. Answer with a single letter." s1, s2 = two ontology siblings of X (same parent in
  `src/audioset_ontology.json`, depictable vocab first, `random.Random(hash of clip|family|start)`; if fewer than two,
  cousins via the grandparent); u = a random depictable family from another top-level branch. Asked twice (X at A,
  X at D; the others rotate); score = mean over the two of softmax over the 5 letter logits for X's letter.
  **Rule V1:** accept iff p(X) > 0.5 and p(X) > 2 × max(other options' p).
- **V2 paired cut.** The R13-3 yes/no score on the run cut minus the same question on a control cut from the same clip:
  the nearest window of the same length (>= 1 s) whose FlexSED(X) max < 0.2. **Rule V2:** s_run > 3 and s_run − s_ctrl > 2
  (no control window → reject).
- **V3 localisation.** Cut [start − 3, end + 3]. "If {X} occurs in this recording, reply with the second it starts
  (for example 4.5). If it does not occur, reply none." **Rule V3:** a number is returned and start − 0.5 <= t (clip
  time) <= end + 0.5.
- **V4 open inventory.** "List every distinct non-speech sound you hear in this recording, one per line, most prominent
  first." A line matches X iff it contains X's name or one of its comma-separated AudioSet synonyms, or a child/parent
  name, case-insensitive; else all-mpnet-base-v2 cosine(line, X name) > 0.6. **Rule V4:** some line matches X.
**Selection (on DEV, full pipeline, B0r base):** the rescue uses exactly one rule (V1, V2, V3, V4) or V1 AND V2; no
per-family thresholds; LO fixed at 0.5. Guard (Fable): each rule is also reported on DEV halves A/B (clips sorted by
name, alternate) and its null-control accept-rate must be <= 10 % on both halves. Then the DEV selection rule above
picks at most one candidate for TEST.

### 2026-09-29 — amendment A arms (stricter listener rules), DEV, job 31418824 (H200, 26 min)
**What.** Flag `LISTENER_RULE` ("V1" … "V4", "V12"; default None = the R13-3 yes/no score) + `LISTENER_VCACHE`
(`benchmark/gold/dev_listener_v.json`): R13-3 (a) and (b) take the named rule's accept flag instead of score > TH; LO 0.5.
(a) looks up the P2 run (clip, family, start, end); (b) the PV item of the vetoed span. P1 is never used. Missing key = not
rescued. Arms LR-V1, V2, V3, V4, V12, each alone and + R13-1 (all five run; the best two by cost are V12 and V4).
**Gates.** Flags off: D0 98/98, conf-equality 98/98, D5 49/49 both systems. **Cache coverage: 0 missing** ((a) 512 runs
asked, (b) 34 vetoed spans asked, every arm). **Null-control accept rate** (P2 peak >= 0.5 + PV, halves A / B): V1 1.8 / 0.9 %,
V2 0.7 / 0.0 %, V3 2.1 / 5.0 %, V4 0.4 / 0.6 %, V12 0.4 / 0.0 %: all <= 10 % on both halves (guard passes).
Halves: DEV clips sorted by name, alternate (A = 25 clips, 16 needed; B = 24 clips, 20 needed); B0r A 8 hits / 10 wrong,
B 6 / 14.

*ours (with gate)*

| arm | heard | hits / 36 | wrong (v / c / p) | cost | Δ vs B0r [95 % CI] | eligible | half A: hits / wrong / Δ [CI] | half B: hits / wrong / Δ [CI] | rescued (a) / kept (b) |
|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] | — |
| B1 | 16 | 13 | 30 (6 / 18 / 6) | 3.10 | +0.33 [+0.04, +0.69] | — | 8 / 12 / +0.16 [-0.24, +0.72] | 5 / 18 / +0.50 [+0.08, +0.92] | — |
| LR-V1 | 23 | 20 | 72 (8 / 46 / 18) | 4.24 | +1.47 [+0.49, +2.53] | no | 9 / 26 / +1.12 [+0.56, +1.76] | 11 / 46 / +1.83 [-0.25, +3.75] | 104 / 7 |
| LR-V1+1 | 24 | 20 | 73 (8 / 47 / 18) | 4.29 | +1.51 [+0.53, +2.57] | no | 9 / 26 / +1.12 [+0.56, +1.76] | 11 / 47 / +1.92 [-0.17, +3.83] | 104 / 7 |
| LR-V2 | 22 | 19 | 49 (8 / 30 / 11) | 3.39 | +0.61 [-0.12, +1.22] | no | 9 / 20 / +0.64 [+0.16, +1.20] | 10 / 29 / +0.58 [-0.83, +1.67] | 61 / 2 |
| LR-V2+1 | 23 | 20 | 49 (8 / 30 / 11) | 3.31 | +0.53 [-0.24, +1.22] | no | 9 / 20 / +0.64 [+0.16, +1.20] | 11 / 29 / +0.42 [-1.08, +1.67] | 61 / 2 |
| LR-V3 | 22 | 19 | 58 (7 / 33 / 18) | 3.76 | +0.98 [+0.24, +1.71] | no | 10 / 25 / +0.88 [+0.08, +1.76] | 9 / 33 / +1.08 [-0.17, +2.25] | 66 / 9 |
| LR-V3+1 | 23 | 20 | 58 (7 / 33 / 18) | 3.67 | +0.90 [+0.12, +1.67] | no | 10 / 25 / +0.88 [+0.08, +1.76] | 10 / 33 / +0.92 [-0.50, +2.25] | 66 / 9 |
| LR-V4 | 22 | 19 | 41 (6 / 26 / 9) | 3.06 | +0.29 [-0.37, +0.86] | no | 10 / 19 / +0.40 [-0.24, +1.12] | 9 / 22 / +0.17 [-1.00, +1.08] | 49 / 6 |
| LR-V4+1 | 23 | 20 | 41 (6 / 26 / 9) | 2.98 | +0.20 [-0.49, +0.86] | no | 10 / 19 / +0.40 [-0.24, +1.12] | 10 / 22 / +0.00 [-1.25, +1.08] | 49 / 6 |
| LR-V12 | 21 | 18 | 38 (7 / 21 / 10) | 3.02 | +0.24 [-0.33, +0.78] | no | 9 / 14 / +0.16 [-0.16, +0.48] | 9 / 24 / +0.33 [-0.75, +1.33] | 43 / 1 |
| LR-V12+1 | 22 | 19 | 38 (7 / 21 / 10) | 2.94 | +0.16 [-0.45, +0.73] | no | 9 / 14 / +0.16 [-0.16, +0.48] | 10 / 24 / +0.17 [-1.08, +1.33] | 43 / 1 |

Pipeline without gate (hits / wrong / cost): B0r 17/50/3.59; LR-V1 23/119/5.92; LR-V1+1 24/121/5.92; LR-V2 22/88/4.73; LR-V2+1 23/91/4.78; LR-V3 22/103/5.35; LR-V3+1 23/105/5.35; LR-V4 22/79/4.37; LR-V4+1 23/81/4.37; LR-V12 21/76/4.33; LR-V12+1 22/79/4.37.

**Best arm = LR-V12+1** (V1 AND V2, + R13-1): hits 14 → 19, wrong 24 → 38, cost 2.94, Δ +0.16 [−0.45, +0.74]; not
eligible (wrong 38 > 24 + 2 × 5 = 34, cost above B0r). Needed gained: `ambient_citywalk_nyc_1689` Hammer 13.7 s; `ambient_nature_rainforest_7629` Cricket 0.1 s; `as_explosion_XJ8lc3I6` Walk, footsteps 2.1 s; `as_explosion_XJ8lc3I6` Gasp 6.7 s; `b3_carnival_parade` Whistle 6.1 s. Lost: none. Wrong pictures appeared (16):
`ambient_citywalk_nyc_1689` Sliding door 2.48 (phantom); `ambient_nature_rainforest_2179` Bell 10.60 (cross); `as_glass_oHil9Ip_` Coin (dropping) 9.00 (cross); `as_glass_oHil9Ip_` Coin (dropping) 18.52 (cross); `as_glass_oHil9Ip_` Tools 18.44 (cross); `b3_botanic_garden` Bird 12.80 (cross); `b3_carnival_parade` Whistle 14.64 (phantom); `b3_construction_site` Honk 0.40 (visible); `bell_miami` Train 8.00 (cross); `ly_applause_62ZYD0u` Splash, splatter 3.44 (cross); `movie_blueplanet_115` Laughter 0.00 (cross); `mv_storm_scene_house` Pant 6.16 (cross); `mv_storm_scene_house` Shout 14.48 (cross); `un_driving_motorcycle_4O3bZRYO` Honk 5.12 (cross); `un_driving_motorcycle_4O3bZRYO` Honk 9.40 (cross); `un_driving_motorcycle_4O3bZRYO` Shout 2.36 (phantom). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50 (cross); `mv_detective_crime_scene` Alarm 3.92 (cross).
**Reading.** The stricter rules cut the wrong rescues sharply (V12 admits 43 runs against 95 for L05t3, wrong 38 against 65)
and keep 4–6 of the 8 band hits. **No arm is DEV-eligible and no Δ cost is below 0**, on the whole of DEV or on either half.
R13-1 adds the Cricket hit to every rule at no extra wrong picture (except V1).

### Selection and TEST (2026-09-29, written BEFORE the TEST run)
**DEV selection:** the only DEV-eligible arm of the whole round is **R13-1 (TWIN_MAX)**: 15 hits, 24 wrong, cost 2.69
vs B0r 2.78. Every listener arm (yes/no R13-3 and amendment A V1–V4, V12, each ± R13-1) adds more wrong pictures than
2 × the hits it gains; the best, V12 + R13-1 (19 hits, 38 wrong, 2.94), is worse than B0r on both DEV halves
(+0.16 / +0.17). The listener line is closed: an audio LLM verifier finds 5–8 real dropped sounds but at ~3 wrong
pictures per gained hit under every question form tried (precision ~24 % vs the 33 % break-even).
**TEST (one exposure):** `sbatch slurm/job_r13_test_final.sh R13-1` (B0r, B1, R13-1 in one job). Expectation written
now: R13-1 changes about one picture per 50 clips, so the TEST rule (Δ cost < 0 with one-sided p < 0.05 on 60 clips)
is very unlikely to be met; the expected verdict is **same**, and then nothing ships. The B1 row is the first TEST number
for the current `use_shipped()` (self-veto) and is reported beside, as disclosure for Adam's veto decision.
**TEST attempt 1 (job 31419006, 2026-09-29 05:18) stopped before any score.** Stages 4/5 and the D0/D5 gates passed
(120/120, 60/60). After the `.started` marker, the script loaded the gold file and asserted that its "test" subset equals
`test_stems.txt`; it failed because "test" = gold minus DEV includes the 30 slice-B clips, while the frozen TEST table
(and `test_stems.txt`) is "test_bench" (60 clips). Only clip names were compared; no picture was scored, no hit, miss,
cost or per-clip number was computed or printed. The assertion was changed to "test_bench", the marker removed, and the
same job re-submitted unchanged otherwise. This is recorded as a plumbing failure, not a second exposure.

### TEST result (one exposure, job 31419058; stages reused from attempt 1)
Ours (with gate), TEST 60 clips, 43 needed sounds: **R13-1 = B0r exactly** (16 hits, 25 wrong, cost 2.63; no picture
changed) → verdict **same**, nothing ships, as expected before the run. **B1 (current `use_shipped()`, self-veto) is
worse than B0 on TEST too:** 15 hits, 29 wrong, cost 2.83; B0 − B1 = −0.20 [−0.47, +0.00], one-sided p 0.043 (DEV:
−0.33 [−0.69, −0.04]). Blind (no gate): same picture (R13-1 = B0r; B1 +0.17). Full table: `benchmark/gold/r13_test_final.md`.

## Round 14 (2026-09-29, Adam: "how can we let more sounds that were heard but dropped? we have to do it")
Written BEFORE any round-14 number. TEST is spent; DEV is development; **confirmation = the 100 new annotated clips**
(Adam's annotator job, all sounds tagged) under the same pipeline, scorer and B0 baseline.
Base arms: LR-V4 + R13-1 and LR-V12 + R13-1 (round 13, amendment A). Precision filters on the RESCUED spans only:
- **F1 new-type-once:** rescue only a family that has no B0 picture in the clip, and at most one rescued picture per family
  per clip (the rescued run with the highest FlexSED peak).
- **F4 local winner:** at the rescued run's peak frame, its family must be the top-scoring depictable FlexSED query.
- **F3 scene fit:** the gate VLM (Qwen3.8-27B, frames of the span) is asked "Could the sound of {X} plausibly be heard in
  this scene? Answer yes or no."; rescue only on yes (same voting as the gate's other asks).
Arms: each base × {F1, F4, F1+F4, F1+F4+F3}. Report as always (heard, hits, wrong v/c/p, cost β=2, Δ vs B0r [CI], halves
A/B) plus **cost at β = 1 and the break-even β** (where the arm's cost equals B0r's).
**Two claims, kept apart:** (1) default setting: an arm ships as default only if it passes the DEV selection rule AND
then wins on the new clips at β = 2 (prereg TEST rule). (2) "more sounds" viewer setting: an arm is offered as an option
if its break-even β is ≥ 1.0 on DEV and on the new clips; reported with the full cost-vs-β curve.
**Round 14 addendum (Fable consult 2, written before any round-14 number):** two more filters on rescued spans:
- **F5 co-onset shadow:** drop a rescued run whose peak lies within ±0.2 s of the onset of an already-drawn (B0) picture
  of a different family with a higher score (one impulse lighting many impulsive queries).
- **F6 clip edge:** drop a rescued run whose peak lies in the first or last 0.3 s of the clip.
Tagger agreement at the peak (Fable idea 2) is NOT added: BEATs/PANNs are deaf (<= 0.05) to 4 of the 6 reachable band
hits (round 13 diagnosis), so it would remove the hits it is meant to keep.
Extra arms: best round-14 arm + F5, + F6, + F5 + F6.
**Round 14 amendment B — "swap, don't add" (Adam: add rescued sounds without adding wrong pictures; written before any
round-14 number).** Target on DEV: hits >= B0 + 4 with wrong <= 33 (cost below B0r at β = 2).
- **F7 confirmed mirror veto:** R13-2 (b 0.7) — drop a BEATs-only span when FlexSED's top query there is a different family
  >= 0.7 and its own family < 0.4 — but KEEP it if the listener accepts the span's family (rule V4, else V12; P1 items of
  `dev_listener_v.json`; if the item has no V4/V12 flag, use yes/no score > 3 from `dev_listener.json`).
- **F8 third vote:** a rescued run is kept only if DASM (round 6 / devcand D1 DEV scores) gives the same family >= its D1
  bar anywhere in the run ± 0.5 s.
Arms: best round-14 arm (after F1–F6) + F7, + F8, + F7 + F8.
**Round 14 amendment C — second listener (written before any output of it).** Audio Flamingo Next
(`nvidia/audio-flamingo-next-hf`, NVIDIA OneWay Noncommercial licence accepted by Adam 2026-09-29; official weights).
Scored on exactly the amendment-A candidates (DEV and gold-free TEST): V4 open inventory (same prompt, same matching
rule) and the yes/no question (score = logit yes − logit no). **Rule AF:** V4 match; **Rule AGREE:** the Qwen3-Omni rule
of the arm (V4 or V12) AND AF-Next V4 match. Arms: best round-14 arm with its listener rule replaced by AGREE, and AGREE
alone on the LR-V4+R13-1 and LR-V12+R13-1 bases (+F1+F4). Same report as round 14.
**Round 14 amendment D — FlexSED specific queries (written before any pipeline number; only the component screen
`benchmark/gold/flexsed_extra_dev.json` was seen).** FlexSED gets the 120 "folded" labels (group a of
`benchmark/gold/flexsed_extra_queries.json`: depictable labels that had no query of their own, e.g. "Vehicle horn, car
horn, honking", "Bird vocalization, bird call, bird song") as extra queries, cache `data/work/flexsed_extra_{dev,test}/`,
and they enter stage 4 exactly like the 215 (same 0.8 bar, same min span, same twin rule, same PANNs clip veto, label =
the query's AudioSet label, family by the existing mapping). Group b (27 labels outside the depictable vocab) is NOT
added. Screen facts seen: the 4 unheard misses stay unheard (Whack 0.09, Clang 0.01, Hammer 0.00); at 0.8 the new
queries give 111 runs on same-family gold vs 34 off gold. Arms: **XQ** (B0r + extra queries), **XQ + R13-1**, and the
best round-14 listener arm + XQ. Same full-pipeline DEV test and report.

### Round 14 results (2026-09-29/30), DEV, job 31506544 (H200, 59 min) — filters F1–F8 on the rescued spans
**Implementation (flags default off, `config.py`).** `filter_rescued()` (`src/stage4_audio_event_detection/__init__.py`) runs
after onset refinement in `detect_events` and in the harness, order F4, F6, F5, F8, F1: F4 `LISTENER_LOCAL_WINNER`, F6
`LISTENER_EDGE` (0.3 s), F5 `LISTENER_SHADOW` (±0.2 s, higher conf), F8 `LISTENER_DASM_VOTE` (DASM DEV scores
`data/work/devcand/dasm_cache`, job 31330562, round-6 queries; bar = D1's g 0.575 from `benchmark/detector_round6.json`),
F1 `LISTENER_NEW_TYPE_ONCE`. "B0 picture" in F1/F5 = a non-rescued drawable stage-4 span at or above the display bar (the
stage-4 picture candidates; stage 4 cannot see the gate). F3 `LISTENER_SCENE_FIT` in `reason.decide_subjects` on rescued
specs only: the prereg prompt per gate stretch (the gate's frames), majority of yes. F7 `LISTENER_CONFIRMED_MIRROR`:
mirror veto b 0.7 keeps a span the listener accepts (P1 items of `dev_listener_v.json` carry V1/V2/V12 only, so V12; else
yes/no > 3 from `dev_listener.json`). Rescued spans carry `rescued` on `AudioEvent`/`AugmentationSpec`.
**Gates.** Flags off: D0 98/98, conf-equality 98/98 (offline with every flag present, and on the cluster), D5 49/49 both
systems after every stage of the job. Cache misses: (a) 0/512, (b) 0/34; F7: 30 of 66 mirror-dropped spans have no P1
item, all of them spans the FlexSED clip veto removes later (none is in B0r).
Arms picked in the job: best of the 8 = LR-V12+1+F1F4F3 (cost 2.78, ties broken by fewer wrong) → + F5/F6; best so far
= the same (F5, F6 changed no picture) → + F7/F8.

*ours (with gate); cost = (4 × miss + β × wrong) / 49; break-even β = the β where the arm's cost equals B0r's (below it the
arm is cheaper; "any" = cheaper at every β)*

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible | half A hits / wrong / Δ | half B hits / wrong / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] |
| B1 | 16 | 13 | 30 (6 / 18 / 6) | 3.10 | +0.33 [+0.04, +0.69] | 2.49 | -0.67 | — | 8 / 12 / +0.16 [-0.24, +0.72] | 5 / 18 / +0.50 [+0.08, +0.92] |
| LR-V4+1 | 23 | 20 | 41 (6 / 26 / 9) | 2.98 | +0.20 [-0.49, +0.86] | 2.14 | 1.41 | no | 10 / 19 / +0.40 [-0.24, +1.12] | 10 / 22 / +0.00 [-1.25, +1.08] |
| LR-V12+1 | 22 | 19 | 38 (7 / 21 / 10) | 2.94 | +0.16 [-0.45, +0.73] | 2.16 | 1.43 | no | 9 / 14 / +0.16 [-0.16, +0.48] | 10 / 24 / +0.17 [-1.08, +1.33] |
| LR-V4+1+F1 | 20 | 17 | 37 (5 / 23 / 9) | 3.06 | +0.29 [-0.20, +0.69] | 2.31 | 0.92 | no | 10 / 16 / +0.16 [-0.40, +0.72] | 7 / 21 / +0.42 [-0.33, +1.09] |
| LR-V4+1+F4 | 22 | 19 | 36 (7 / 20 / 9) | 2.86 | +0.08 [-0.57, +0.65] | 2.12 | 1.67 | no | 9 / 15 / +0.24 [-0.32, +0.88] | 10 / 21 / -0.08 [-1.25, +0.83] |
| LR-V4+1+F1F4 | 19 | 16 | 33 (6 / 18 / 9) | 2.98 | +0.20 [-0.20, +0.57] | 2.31 | 0.89 | no | 9 / 12 / +0.00 [-0.40, +0.32] | 7 / 21 / +0.42 [-0.33, +1.08] |
| LR-V4+1+F1F4F3 | 19 | 16 | 29 (6 / 15 / 8) | 2.82 | +0.04 [-0.37, +0.37] | 2.22 | 1.60 | no | 9 / 12 / +0.00 [-0.40, +0.32] | 7 / 17 / +0.08 [-0.58, +0.67] |
| LR-V12+1+F1 | 20 | 17 | 36 (7 / 19 / 10) | 3.02 | +0.24 [-0.16, +0.65] | 2.29 | 1.00 | no | 9 / 13 / +0.08 [-0.16, +0.32] | 8 / 23 / +0.42 [-0.42, +1.17] |
| LR-V12+1+F4 | 20 | 17 | 30 (6 / 15 / 9) | 2.78 | +0.00 [-0.49, +0.41] | 2.16 | 2.00 | no | 8 / 12 / +0.16 [+0.00, +0.40] | 9 / 18 / -0.17 [-1.17, +0.67] |
| LR-V12+1+F1F4 | 19 | 16 | 29 (6 / 14 / 9) | 2.82 | +0.04 [-0.29, +0.33] | 2.22 | 1.60 | no | 8 / 11 / +0.08 [+0.00, +0.24] | 8 / 18 / +0.00 [-0.67, +0.50] |
| LR-V12+1+F1F4F3 | 19 | 16 | 28 (6 / 13 / 9) | 2.78 | +0.00 [-0.33, +0.29] | 2.20 | 2.00 | no | 8 / 11 / +0.08 [+0.00, +0.24] | 8 / 17 / -0.08 [-0.67, +0.42] |
| LR-V12+1+F1F4F3+F5 | 19 | 16 | 28 (6 / 13 / 9) | 2.78 | +0.00 [-0.33, +0.29] | 2.20 | 2.00 | no | 8 / 11 / +0.08 [+0.00, +0.24] | 8 / 17 / -0.08 [-0.67, +0.42] |
| LR-V12+1+F1F4F3+F6 | 19 | 16 | 28 (6 / 13 / 9) | 2.78 | +0.00 [-0.33, +0.29] | 2.20 | 2.00 | no | 8 / 11 / +0.08 [+0.00, +0.24] | 8 / 17 / -0.08 [-0.67, +0.42] |
| LR-V12+1+F1F4F3+F5F6 | 19 | 16 | 28 (6 / 13 / 9) | 2.78 | +0.00 [-0.33, +0.29] | 2.20 | 2.00 | no | 8 / 11 / +0.08 [+0.00, +0.24] | 8 / 17 / -0.08 [-0.67, +0.42] |
| LR-V12+1+F1F4F3+F7 | 19 | 16 | 25 (6 / 11 / 8) | 2.65 | -0.12 [-0.49, +0.16] | 2.14 | 8.00 | yes | 8 / 10 / +0.00 [-0.24, +0.24] | 8 / 15 / -0.25 [-0.92, +0.33] |
| LR-V12+1+F1F4F3+F8 | 18 | 15 | 25 (6 / 12 / 7) | 2.73 | -0.04 [-0.33, +0.16] | 2.22 | 4.00 | yes | 8 / 10 / +0.00 [+0.00, +0.00] | 7 / 15 / -0.08 [-0.67, +0.33] |
| LR-V12+1+F1F4F3+F7F8 | 18 | 15 | 22 (6 / 10 / 6) | 2.61 | -0.16 [-0.49, +0.08] | 2.16 | any | yes | 8 / 9 / -0.08 [-0.24, +0.00] | 7 / 13 / -0.25 [-0.92, +0.25] |


**Best round-14 arm = LR-V12+1+F1F4F3+F7F8** (cost 2.61, Δ −0.16 [−0.49, +0.08], β=1 2.16, cheaper than B0r at every β,
halves A −0.08 / B −0.25; **DEV-eligible**, as are +F7 (2.65) and +F8 (2.73)). Needed gained: `ambient_nature_rainforest_7629` Cricket 0.1 s. Lost: none.
Wrong pictures appeared (4): `b3_laundromat` Train 1.00 (phantom); `b3_laundromat` Train 16.25 (phantom); `bell_miami` Train 8.00 (cross); `mv_storm_scene_house` Shout 23.00 (cross). Disappeared (6): `ambient_nature_rainforest_7629` Cricket 4.50 (cross); `b3_favela_rio` Bird 14.00 (cross); `b3_laundromat` Vehicle 0.30 (phantom); `b3_laundromat` Vehicle 14.25 (phantom); `b3_laundromat` Vehicle 23.25 (phantom); `mv_tornado_scene` Vehicle 7.25 (cross).
**Reading.** The DEV-eligible arms win through R13-1 (the Cricket) and the listener-confirmed mirror veto (laundromat
Vehicle ×3, tornado Vehicle, favela Bird; the listener keeps the Glass hit that plain R13-2 lost), not through the rescue:
R13-1 alone gives 15 hits, the filtered rescue (F1+F4+F3) adds 1 (16), and F8 removes that one again (15). The filters do cut the rescue's wrong pictures (V12+R13-1: 38 → 28 with F1+F4+F3), but
not below B0r's 24 until F7 swaps wrong BEATs pictures out. F5 and F6 change no picture on DEV.
Pipeline without gate (hits / wrong / cost): B0r 17/50/3.59; LR-V4+1 23/81/4.37; LR-V12+1 22/79/4.37; LR-V4+1+F1 20/79/4.53; LR-V4+1+F4 22/69/3.96; LR-V4+1+F1F4 19/66/4.08; LR-V4+1+F1F4F3 19/66/4.08; LR-V12+1+F1 20/77/4.45; LR-V12+1+F4 20/65/3.96; LR-V12+1+F1F4 19/63/3.96; LR-V12+1+F1F4F3 19/63/3.96; LR-V12+1+F1F4F3+F5 19/63/3.96; LR-V12+1+F1F4F3+F6 19/62/3.92; LR-V12+1+F1F4F3+F5F6 19/62/3.92; LR-V12+1+F1F4F3+F7 19/58/3.76; LR-V12+1+F1F4F3+F8 18/55/3.71; LR-V12+1+F1F4F3+F7F8 18/50/3.51.
**Round 14 amendment E — confidence-tiered verification (Adam's idea) + once per family (Fable consult 3; written
before any pipeline number; candidate-level DEV screen seen: TIER + once-per-family-first gives 9 hit-candidates vs 28
wrong-class candidates, vs 10 vs 52 for Qwen V4 alone).**
- **TIER:** a FlexSED band run (peak 0.5–0.8) is rescued if Qwen V4 accepts and peak >= 0.6; if peak < 0.6 it needs Qwen V4
  AND Audio Flamingo Next V4. (0.6 = the midpoint of the band; not tuned.)
- **ONCE (F1b):** at most one rescued picture per family per clip: the EARLIEST accepted run (the sound's first onset is
  the one a viewer needs); no "already shown" condition.
Arms: TIER, TIER+ONCE, TIER+ONCE+R13-1, TIER+ONCE+R13-1+XQ, and (QV4 & AF yes/no > 0)+ONCE+R13-1. Same full-pipeline DEV
test and report. Confirmation of whichever is picked = the 100 new annotated clips.
**Round 14 amendment F — bug fixes from the per-video trace + 3 Fable consults (written before any pipeline number of
these arms; the per-video trace `docs/dev_heard_dropped_2026-09-29.md` and candidate screens were seen).**
Fixes (each a mechanism fix, no new threshold):
- **FIX-FAM (F4):** the local-winner test compares FAMILIES (canonical family; a sibling/child query such as 'Steam whistle'
  counts as the candidate's own family), not raw query names. (Whistle case.)
- **FIX-EARLY (F1):** once-per-family keeps the EARLIEST accepted run, not the strongest (= ONCE of amendment E). (Footsteps.)
- **FIX-CTRL (V12):** when no control window exists (the family is >= 0.2 across the whole clip), the paired-cut leg V2 is
  not evaluable and counts as neutral: V12 = V1 alone. (Both Explosions: V1 yes, V2 rejected only for lack of a window.)
- **FIX-GATE:** a gate "visible" vote that names no object counts as "not visible". (Ambulance: voted seen, named nothing.)
Arms (full pipeline, B0r re-run, same report): **B0r + FIX-GATE** (gate fix alone); **A0** = best round-14 arm
(LR-V12+R13-1+F1F4F3+F7F8) with FIX-FAM, FIX-EARLY, FIX-CTRL; **A0 + FIX-GATE**; **A1** = A0 + FIX-GATE + VLM arbiter
(Fable A): a band candidate with Qwen V4 yes AND AF V4 yes but V12 no is accepted iff the gate VLM, on frames of the span,
answers "plausible" AND names a visual cue to "Is a {family} sound plausible in this scene? Answer plausible or
implausible, then name the visual cue."
**Round 14 amendment G — rule audit (Adam: "maybe some rules are bad"; written before any number of it).** Loosen ONE
shipped rule at a time, full pipeline on DEV, on top of B0r and on top of the best amendment-F arm (A0 + FIX-GATE if it
is best, else A0). Each change is pre-set, no sweep:
G1 BEATs display bar 0.35 → 0.30; G2 picture confidence floor (PICTURE_MIN_CONF 0.40) → off; G3 minimum span 0.5 → 0.3 s
(BEATs and FlexSED); G4 FlexSED clip veto 0.3 → off; G5 PANNs clip veto → off for FlexSED spans >= 0.9; G6 display merge
gap 1.5 → 1.0 s; G7 near-duplicate merge (DEDUP_SIM 0.80) → off; G8 gate: drop a sound as visible only if ALL votes say
visible (instead of the majority).
Report per rule: Δhits, Δwrong (v/c/p), Δcost vs its base, which needed sounds were gained. **A rule change becomes a
candidate iff Δhits >= 1 and Δwrong <= 2 × Δhits.** Candidates are then stacked (in order of Δcost) into one final arm.
**Round 14 amendment H (written after the amendment-E table, before any number of these arms).** (1) Clarification of
FIX-FAM: "same family" = the pipeline's canonical family / `same_family` (ancestor or descendant in the family map), NOT
ontology siblings (Dog and Cat stay different). Arms of amendment F that used the sibling reading are re-run with this one.
(2) Stack: TIER + ONCE + R13-1 gained 5 needed sounds (Hammer, Cricket, Laughter, Gunshot, Explosion) at +10 wrong (cost
= B0r); the swap rule F7 removed wrong pictures in the filter arms. Arms: **TIER+ONCE+R13-1+F7**, **TIER+ONCE+R13-1+F7+F8**,
and the same two with FIX-FAM (canonical) + FIX-CTRL + FIX-GATE.

### Round 14 amendment E (TIER, ONCE, QV4_AFYN), DEV, job 31526089 (27 min)
Flags: `LISTENER_RULE` "TIER" (Qwen V4, and below a FlexSED run peak of 0.6 also AF V4) and "QV4_AFYN" (Qwen V4 AND AF
yes/no > 0), both derived in `listener_from_vcache` with `LISTENER_AFCACHE`; `LISTENER_ONCE` (earliest rescued span per
family per clip) in `filter_rescued`. Gates: flags off D0 98/98, D5 49/49 both systems. Cache misses (a) 0, (b) 0; with XQ
the extra queries' runs have no listener entry (not rescued).

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible | half A: hits / wrong / Δ | half B: hits / wrong / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] |
| TIER | 22 | 19 | 35 (5 / 23 / 7) | 2.82 | +0.04 [-0.65, +0.61] | 2.10 | 1.82 | no | 10 / 15 / +0.08 [-0.56, +0.72] | 9 / 20 / +0.00 [-1.33, +1.00] |
| TIER+ONCE | 21 | 18 | 34 (5 / 22 / 7) | 2.86 | +0.08 [-0.45, +0.57] | 2.16 | 1.60 | no | 10 / 15 / +0.08 [-0.56, +0.72] | 8 / 19 / +0.08 [-0.83, +0.83] |
| TIER+ONCE+1 | 22 | 19 | 34 (5 / 22 / 7) | 2.78 | +0.00 [-0.57, +0.57] | 2.08 | 2.00 | no | 10 / 15 / +0.08 [-0.56, +0.72] | 9 / 19 / -0.08 [-1.17, +0.83] |
| TIER+ONCE+1+XQ | 22 | 17 | 40 (6 / 25 / 9) | 3.18 | +0.41 [-0.33, +1.18] | 2.37 | 0.75 | no | 9 / 21 / +0.72 [-0.08, +1.68] | 8 / 19 / +0.08 [-1.00, +1.25] |
| QV4AFYN+ONCE+1 | 22 | 19 | 35 (5 / 21 / 9) | 2.82 | +0.04 [-0.49, +0.53] | 2.10 | 1.82 | no | 10 / 15 / +0.08 [-0.40, +0.56] | 9 / 20 / +0.00 [-1.00, +0.84] |


TIER+ONCE+R13-1: needed gained `ambient_citywalk_nyc_1689` Hammer 13.7 s; `ambient_nature_rainforest_7629` Cricket 0.1 s; `ambient_snow_walk_930` Laughter 8.1 s; `as_explosion_XJ8lc3I6` Gunshot, gunfire 0.0 s; `as_explosion_XJ8lc3I6` Explosion 2.8 s; lost none. Wrong pictures appeared: `ambient_citywalk_nyc_2627` Train 10.00 (cross); `ambient_market_marrakech_3102` Car passing by 13.52 (cross); `ambient_nature_rainforest_2179` Bell 10.60 (cross); `as_glass_oHil9Ip_` Coin (dropping) 9.00 (cross); `b3_crossing_bells` Insect 15.00 (cross); `b3_crossing_bells` Water 10.00 (cross); `b3_favela_rio` Bird 4.60 (cross); `b3_golf_course` Bird 18.84 (cross); `b3_golf_course` Insect 10.00 (cross); `bell_miami` Train 8.00 (cross); `movie_blueplanet_115` Laughter 0.00 (cross); `mv_tornado_scene` Alarm 13.00 (cross); `un_driving_motorcycle_DgdHSmwA` Gunshot 13.52 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50 (cross); `as_fire_alarm_kGKZ0YK4` Alarm 8.89 (visible); `london_protest_01` Vehicle 0.25 (visible); `mv_detective_crime_scene` Alarm 3.92 (cross).
Without gate (hits / wrong / cost): B0r 17/50/3.59; TIER 22/73/4.12; TIER+ONCE 21/70/4.08; TIER+ONCE+1 22/73/4.12; TIER+ONCE+1+XQ 21/88/4.82; QV4AFYN+ONCE+1 22/73/4.12.
**Reading.** No arm is eligible; TIER+ONCE+R13-1 ties B0r (+5 hits, +10 wrong); + XQ is worse (loses a Crowing and the Gunshot).

### Round 14 amendments F and H (fixes, arbiter; TIER + F7/F8), DEV, job 31543142 (16 min)
Flags: `FIX_FAM` (F4 compares families: canonical equal or ontology ancestor/descendant; amendment H: NOT siblings —
the sibling reading was corrected before any arm ran, so no sibling-reading row exists), `FIX_EARLY` (F1 keeps the earliest),
`FIX_CTRL` (V12 = V1 when the item has no V2 control window), `FIX_GATE` (`reason.decide_subjects`: a "visible" verdict that
names nothing is not visible; 2 scored-run votes on DEV, both the ambulance Vehicle), `LISTENER_ARBITER` (Qwen V4 & AF V4 but
V12 no → kept as `arbiter`; stage 5 asks the prereg prompt on 6 frames of the span, accepts "plausible" + a named cue).
A0 = LR-V12+R13-1+F1F4F3+F7F8 + FIX_FAM + FIX_EARLY + FIX_CTRL; TO1 = TIER+ONCE+R13-1; FIX = FIX_FAM + FIX_CTRL + FIX_GATE.
Gates: flags off D0 98/98, conf-equality 98/98, D5 49/49 both systems.

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible | half A: hits / wrong / Δ | half B: hits / wrong / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] |
| B0r+FIXGATE | 17 | 14 | 25 (6 / 12 / 7) | 2.82 | +0.04 [+0.00, +0.12] | 2.31 | 0.00 | no | 8 / 11 / +0.08 [+0.00, +0.24] | 6 / 14 / +0.00 [+0.00, +0.00] |
| A0 | 18 | 15 | 22 (6 / 10 / 6) | 2.61 | -0.16 [-0.49, +0.08] | 2.16 | any | yes | 8 / 9 / -0.08 [-0.24, +0.00] | 7 / 13 / -0.25 [-0.92, +0.25] |
| A0+FIXGATE | 18 | 15 | 23 (6 / 11 / 6) | 2.65 | -0.12 [-0.45, +0.12] | 2.18 | any | yes | 8 / 10 / +0.00 [-0.24, +0.24] | 7 / 13 / -0.25 [-0.92, +0.25] |
| A1 | 19 | 16 | 24 (6 / 12 / 6) | 2.61 | -0.16 [-0.53, +0.16] | 2.12 | any | yes | 9 / 11 / -0.08 [-0.56, +0.32] | 7 / 13 / -0.25 [-0.92, +0.25] |
| TO1+F7 | 22 | 18 | 32 (5 / 21 / 6) | 2.78 | +0.00 [-0.61, +0.61] | 2.12 | 2.00 | no | 10 / 14 / +0.00 [-0.56, +0.56] | 8 / 18 / +0.00 [-1.17, +1.08] |
| TO1+F7F8 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1+F7+FIX | 22 | 18 | 34 (5 / 23 / 6) | 2.86 | +0.08 [-0.57, +0.69] | 2.16 | 1.60 | no | 10 / 15 / +0.08 [-0.56, +0.64] | 8 / 19 / +0.08 [-1.08, +1.17] |
| TO1+F7F8+FIX | 21 | 18 | 26 (6 / 14 / 6) | 2.53 | -0.24 [-0.78, +0.20] | 2.00 | 8.00 | yes | 9 / 11 / -0.08 [-0.56, +0.32] | 9 / 15 / -0.42 [-1.42, +0.42] |


**Best = TO1+F7F8** (TIER + ONCE + R13-1 + confirmed mirror veto + DASM vote): needed gained `ambient_nature_rainforest_7629` Cricket 0.1 s; `ambient_snow_walk_930` Laughter 8.1 s; `as_explosion_XJ8lc3I6` Gunshot, gunfire 0.0 s; `as_explosion_XJ8lc3I6` Explosion 5.6 s; lost none. Wrong pictures appeared: `b3_golf_course` Bird 18.84 (cross); `b3_laundromat` Train 1.00 (phantom); `b3_laundromat` Train 16.25 (phantom); `bell_miami` Train 8.00 (cross); `movie_blueplanet_115` Laughter 0.00 (cross); `un_driving_motorcycle_DgdHSmwA` Gunshot 13.52 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50 (cross); `b3_favela_rio` Bird 14.00 (cross); `b3_laundromat` Vehicle 0.30 (phantom); `b3_laundromat` Vehicle 14.25 (phantom); `b3_laundromat` Vehicle 23.25 (phantom); `mv_tornado_scene` Vehicle 7.25 (cross).
A1 (arbiter): needed gained `ambient_nature_rainforest_7629` Cricket 0.1 s; `ambient_snow_walk_930` Laughter 8.1 s; lost none. Wrong pictures appeared: `b3_laundromat` Train 1.00 (phantom); `b3_laundromat` Train 16.25 (phantom); `bell_miami` Train 8.00 (cross); `ly_ambulance_(siren)_-yPSgCn` Vehicle 0.00 (cross); `mv_storm_scene_house` Shout 23.00 (cross); `un_driving_motorcycle_DgdHSmwA` Gunshot 13.52 (cross). Disappeared: `ambient_nature_rainforest_7629` Cricket 4.50 (cross); `b3_favela_rio` Bird 14.00 (cross); `b3_laundromat` Vehicle 0.30 (phantom); `b3_laundromat` Vehicle 14.25 (phantom); `b3_laundromat` Vehicle 23.25 (phantom); `mv_tornado_scene` Vehicle 7.25 (cross).
Without gate: B0r 17/50/3.59; B0r+FIXGATE 17/50/3.59; A0 18/50/3.51; A0+FIXGATE 18/50/3.51; A1 19/51/3.47; TO1+F7 22/68/3.92; TO1+F7F8 21/53/3.39; TO1+F7+FIX 22/68/3.92; TO1+F7F8+FIX 21/53/3.39.
**Reading.** TO1+F7F8 is the best DEV arm so far: +4 hits for 0 extra wrong, Δ −0.33 [−0.86, +0.08], cheaper than B0r at
every β, better on both halves; DEV-eligible. A0 gives the same pictures as LR-V12+1+F1F4F3+F7F8 (the three fixes change
nothing once F8 is on). FIX_GATE hurts in every arm: it lets the ambulance Vehicle 0.0 s picture through (a cross; the 7.3 s
onset is still missed). FIX_FAM/FIX_CTRL have no effect on TIER arms (they touch only F4 and V12).

## Confirmation set 1: Adam's tagger set (written 2026-09-29, before any pipeline run on these clips)
16 new clips `data/input/tagger_set/tg_dNNN.mp4`, tags `benchmark/gold/annotations/tagger_AG.json` (31 sounds, 9 needed;
none of the clips is in DEV or TEST). Only counts were read; no pipeline output exists for them. Disclosure: the tagger
tool showed detector suggestions to the annotator (`from_suggestion` / `suggestions_shown_at` fields).
**Candidates (frozen now):** C1 = TO1+F7F8 (the best DEV arm); C2 = the stacked amendment-G arm IF it is DEV-eligible
and beats C1 on DEV cost, else none. Baselines re-run in the same job: B0r (primary) and B1.
**Test:** full pipeline, `score_per_sound` on the 16 clips, paired clip bootstrap 2000 seed 0; Holm over the candidates.
**Better** iff hits do not drop, wrong <= B0r wrong + 2 × hits gained, and Δcost < 0 with Holm-adjusted one-sided
p < 0.05; else **same** unless Δcost > 0 with lower CI > 0 (**worse**). With 9 needed sounds this set is small: it is read
together with the 100-clip set later (same rule), not alone. Scored ONCE; features (caches, listener answers, stage
1–5 for B0r) may be built before, without reading the tags.
**Round 14 amendment I (Fable consult 4; written before any number of these arms; not part of confirmation set 1's
frozen candidates — any winner goes to the 100-clip set).** On top of TO1+F7F8:
- **I1 onset re-localisation:** inside each displayed span, the picture START moves to the frame of the steepest rise of
  the family's evidence (max of BEATs column and FlexSED family/specific queries, 25-fps grid) within the span's first 3 s
  after any dip below half its peak; a second rise after a dip >= 1.5 s splits the span. No new pictures.
- **I5 specific-query evidence:** the rescue tier's "peak" = max over the family query and its specific child queries
  (the 120 folded queries) in the run window; no new spans from child queries by themselves.
- **I6 impulsive short runs:** for impulsive families (Gunshot, Gasp, Explosion, Knock, Hammer, Glass, Door, Slam-type as
  in R13-5), rescue runs shorter than 0.5 s are allowed (cut as 1 s around the peak), still needing the TIER listener rule.
- **I2 activity gate:** a sound the gate calls visible is kept if the VLM, on frames at onset ± 0.5 s, answers "no" to "Is
  a {family} source visibly PRODUCING this sound right now (for example a beak open, a bell swinging, a vehicle moving)?
  Answer yes or no."
Arms: TO1+F7F8 + I1; + I5; + I6; + I2; + all four.
**Confirmation set 1 — REVISED (Adam, 2026-09-29, before any pipeline run or score on these clips): split, not all-test.**
Every tagger-set clip, now and in future batches, goes to **DEV2** or **TEST2** by `sha256(stem) % 2` (0 → DEV2, 1 →
TEST2), fixed now, independent of content. Batch 1: DEV2 = tg_d007, d016, d020, d033, d075, d088 (6); TEST2 = tg_d001,
d011, d013, d017, d040, d076, d077, d078, d079, d080 (10). DEV2 joins DEV for development (its tags may be read and tuned
on). TEST2 stays sealed: scored once per frozen candidate set with the rule above (candidates C1 and C2 as frozen), and
later batches' TEST2 clips are added to it before that single scoring, unless Adam asks to score earlier.
**Round 14 amendment J (Fable consult 4, ideas 3–4; written before any number):**
- **J3 motion-timed impacts:** for impulsive families, a weak run of any detector (BEATs >= 0.1, FlexSED >= 0.3) is
  rescued if the video's frame-difference energy (grayscale, 10 fps) has a peak (>= 3 × the clip median, within the 3 s
  around the run) within 0.5 s of the run's peak; the picture starts at the motion peak; the TIER listener rule is still
  required; no motion-only pictures.
- **J4 audio-LLM timestamps:** for each rescued or weak run, a 4-s window around it; Qwen3-Omni and Audio Flamingo Next are
  asked "At which second of this recording does the {family} sound start? Reply with a number, or none."; if both give a
  number within 1 s of each other, the picture starts at their mean (clip time); if either says none, no change for
  rescued runs and no rescue for weak runs outside the TIER rule.
Arms on top of TO1+F7F8: +J3, +J4, +J3+J4, and + the best amendment-I combination.
**Round 14 amendment K (from the per-sound trace of TO1+F7F8, `docs/dev_heard_dropped_best_2026-09-29.md`; written before
any number of these arms).** On top of TO1+F7F8:
- **K1 F8 bypass:** the DASM vote is skipped for a rescued run that BOTH Qwen V4 and Audio Flamingo V4 accept (two audio
  LLMs outvote one SED). (F8 killed Hammer 13.7 and Explosion 2.8.)
- **K2 covered runs:** a FlexSED band run is still offered to the rescue when a same-family BEATs span BELOW the display
  bar covers it (today it is skipped as "covered"; the listener is never asked). (birds_forest Bird.)
- **K3 high tier OR:** for peak >= 0.6, accept if Qwen V4 OR Audio Flamingo V4 accepts (low tier unchanged: both).
Arms: +K1, +K2, +K3, +K1+K2, +K1+K2+K3. K2 needs listener answers for the newly offered runs: if missing from the caches,
they count as not rescued and the number missing is reported.

### Round 14 amendment G (rule audit), DEV, job 31562318 (29 min)
One rule loosened at a time on B0r and on the best F/H arm (picked in the job by lowest DEV cost: **TO1+F7F8**). Implementation:
G1 `DISPLAY_THRESHOLD`/`AUGMENT_THRESHOLD` 0.30 (AED 0.175 kept); G3 `AED_MIN_DUR` 0.3 (BEATs spans live on a 0.25-s grid, so
this changes FlexSED spans only); G4 `FLEXSED_VETO` 0; G5 new flag `PANNS_VETO_SKIP_ABOVE` 0.9; G6 `MERGE_GAP` 1.0 (B0r's is
2.0, the scored config; applied at score time per arm); G7 `DEDUP_SIM` 1.01; G8 `VISIBILITY_RULE` "unanimous" (reused gate
votes re-decided from their three stored votes). **G2 is a no-op** on both bases (the scored config has no picture floor;
0.40 exists only in `use_shipped`) and was not run. Gates: D5 49/49 both systems. Δ = vs the arm's own base.

| rule | base | Δhits | Δwrong (v / c / p) | Δcost [95 % CI] | gained needed | lost needed | candidate |
|---|---|---|---|---|---|---|---|
| G1 display bar 0.35 → 0.30 (and augment bar) | B0r | +1 | +3 (+2 / +2 / -1) | +0.04 [-0.33, +0.45] | `ambient_nature_rainforest_7629` Bird 0.1 s; `ambient_nature_rainforest_7629` Cricket 0.1 s | `birds_forest` Crowing, cock-a-doodle-doo 10.4 s | no |
| G3 min span 0.5 → 0.3 s (BEATs + FlexSED) | B0r | -1 | +3 (+0 / +3 / +0) | +0.20 [+0.00, +0.57] | — | `b3_favela_rio` Train 14.6 s | no |
| G4 FlexSED clip veto 0.3 → off | B0r | +0 | +11 (+2 / +5 / +4) | +0.45 [+0.12, +0.78] | — | — | no |
| G5 PANNs veto skipped for FlexSED spans >= 0.9 | B0r | +1 | +7 (+0 / +5 / +2) | +0.20 [-0.08, +0.49] | `ambient_snow_walk_930` Laughter 8.1 s | — | no |
| G6 display merge gap 2.0 → 1.0 s | B0r | +0 | +5 (+1 / +4 / +0) | +0.20 [+0.04, +0.37] | — | — | no |
| G7 DEDUP_SIM 0.80 → off | B0r | +0 | +4 (+0 / +2 / +2) | +0.16 [+0.00, +0.41] | — | — | no |
| G8 gate: visible only if all 3 votes say so | B0r | +1 | +6 (+3 / +3 / +0) | +0.16 [-0.12, +0.41] | `bell_miami` Bell 0.2 s | — | no |
| G1 display bar 0.35 → 0.30 (and augment bar) | TO1+F7F8 | +0 | +5 (+2 / +3 / +0) | +0.20 [+0.00, +0.53] | `ambient_nature_rainforest_7629` Bird 0.1 s | `birds_forest` Crowing, cock-a-doodle-doo 10.4 s | no |
| G3 min span 0.5 → 0.3 s (BEATs + FlexSED) | TO1+F7F8 | -1 | +2 (+0 / +2 / +0) | +0.16 [+0.00, +0.49] | — | `b3_favela_rio` Train 14.6 s | no |
| G4 FlexSED clip veto 0.3 → off | TO1+F7F8 | +0 | +7 (+0 / +5 / +2) | +0.29 [+0.08, +0.57] | — | — | no |
| G5 PANNs veto skipped for FlexSED spans >= 0.9 | TO1+F7F8 | +0 | +5 (+0 / +3 / +2) | +0.20 [+0.04, +0.45] | — | — | no |
| G6 display merge gap 2.0 → 1.0 s | TO1+F7F8 | +0 | +5 (+1 / +4 / +0) | +0.20 [+0.04, +0.37] | — | — | no |
| G7 DEDUP_SIM 0.80 → off | TO1+F7F8 | +0 | +3 (+0 / +3 / +0) | +0.12 [+0.00, +0.33] | — | — | no |
| G8 gate: visible only if all 3 votes say so | TO1+F7F8 | +1 | +9 (+4 / +5 / +0) | +0.29 [+0.00, +0.57] | `bell_miami` Bell 0.2 s | — | no |

**No rule change is a candidate** (Δhits >= 1 and Δwrong <= 2 × Δhits) on either base, so no stacked arm was run. Every
loosening adds 3–11 wrong pictures for at most one hit.
**Amendment J screen result (2026-09-30): J3 and J4 dropped before any pipeline arm.** J3: a motion peak lands in the hit
window for 1 of 13 needed impulsive DEV sounds (chance on the same clips 0.21; 17 of 49 clips never reach 3 × median
with a moving camera). J4: the two audio LLMs "agree" mostly by both answering 0 (Audio Flamingo 94 % of its numbers,
Qwen 62 %); runs with no gold sound agree 58 % of the time; the agreed mean lands in the hit window less often than the
runs' own starts (19 vs 24 of 74). Both would make timing worse, so the +J3/+J4 arms are not run.
`benchmark/gold/motion_ts_screen.json`.

### Round 14 amendments I and K (on TO1+F7F8), DEV, job 31562590 (30 min)
Flags: I1 `ONSET_RELOC` (`relocate_onsets`, stage 4 after refinement, before the rescue filters: start at the steepest rise of
the family evidence — BEATs held per window, FlexSED family + specific queries — within the span's first 3 s after the last
dip below half the span's peak; a >= 1.5 s dip below half splits the span), I5 `TIER_SPECIFIC` (tier peak includes the 120
folded queries of `data/work/flexsed_extra_dev` as evidence), I2 `ACTIVITY_GATE` (`reason._not_producing`: a spec the gate
silences is kept if the prereg question on 6 frames at onset ± 0.5 s gets "no"), K1 `F8_BYPASS_BOTH` (rescues accepted by
Qwen V4 AND AF V4 skip F8), K2 `RESCUE_COVERED` (a sub-display BEATs span no longer counts as covering a band run), K3
`TIER_HIGH_OR`. **I6 was not run: it is the existing behaviour** (P2 holds runs of any length; 29 short (< 0.5 s) impulsive
runs — Explosion, Gasp, Gunshot, Hammer — all with cached answers; the rescue already cuts them to the 1-s window).
K2 offered 27 new runs (539 vs 512 asked), all without listener answers → not rescued. Gates: D0 98/98, conf-equality
98/98, D5 49/49 both systems.

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible | half A: hits / wrong / Δ | half B: hits / wrong / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] |
| TO1+F7F8 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1F7F8+I1 | 13 | 11 | 41 (6 / 29 / 6) | 3.71 | +0.94 [+0.16, +1.71] | 2.88 | -0.71 | no | 4 / 20 / +1.44 [+0.32, +2.64] | 7 / 21 / +0.42 [-0.67, +1.42] |
| TO1F7F8+I5 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1F7F8+I2 | 21 | 20 | 41 (12 / 20 / 9) | 2.98 | +0.20 [-0.45, +0.82] | 2.14 | 1.41 | no | 10 / 19 / +0.40 [-0.40, +1.28] | 10 / 22 / +0.00 [-1.00, +0.92] |
| TO1F7F8+I125 | 13 | 11 | 58 (9 / 43 / 6) | 4.41 | +1.63 [+0.82, +2.49] | 3.22 | -0.35 | no | 4 / 28 / +2.08 [+0.96, +3.36] | 7 / 30 / +1.17 [+0.08, +2.25] |
| TO1F7F8+K1 | 22 | 18 | 30 (5 / 19 / 6) | 2.69 | -0.08 [-0.69, +0.49] | 2.08 | 2.67 | yes | 10 / 12 / -0.16 [-0.72, +0.24] | 8 / 18 / +0.00 [-1.17, +1.00] |
| TO1F7F8+K2 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1F7F8+K3 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.82, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.42, +0.17] |
| TO1F7F8+K1K2 | 22 | 18 | 30 (5 / 19 / 6) | 2.69 | -0.08 [-0.69, +0.49] | 2.08 | 2.67 | yes | 10 / 12 / -0.16 [-0.72, +0.24] | 8 / 18 / +0.00 [-1.17, +1.00] |
| TO1F7F8+K1K2K3 | 22 | 18 | 30 (5 / 19 / 6) | 2.69 | -0.08 [-0.65, +0.45] | 2.08 | 2.67 | yes | 10 / 12 / -0.16 [-0.72, +0.24] | 8 / 18 / +0.00 [-1.08, +1.00] |


**Reading.** Nothing beats TO1+F7F8. I5, K2, K3 leave it unchanged (K3 swaps one cross picture). I1 as written is harmful: the
steepest-rise start lands after the annotated onset (e.g. Hammer 13.76 → 15.64 s), 7 hits lost. I2 keeps visible sources
(+2 hits, +6 visible). K1 brings back Hammer 13.7 and Explosion 2.8 but ONCE then drops Explosion 5.6, the favela Train
14.6 is lost, and 6 cross pictures come in: same hits, +6 wrong.
**BEATs weak-band rescue screen (P3, 2026-09-30): no pipeline arm run.** 272 DEV P3 spans scored by Qwen V4/V1/V2 and
Audio Flamingo V4. Even the pool upper bound (every P3 span) recovers NONE of the 18 needed sounds TO1+F7F8 misses; every
rule's accepts are mostly cross/phantom (e.g. both-V4 + once: 2 hit-class / 13 wrong-class, and the 2 hits are sounds
already shown). `benchmark/gold/dev_listener_p3*.json`, `benchmark/gold/listener_p3.py`.

### Round 14 amendments C (AGREE) and D (XQ), DEV, job 31523769 (20 min)
Flags: `LISTENER_AFCACHE` (`dev_listener_afn.json`, merged in `listener_from_vcache`: AGREE_V4 = Qwen V4 AND AF V4, AGREE_V12 =
Qwen V12 AND AF V4; AF covers every amendment-A candidate, 0 missing); `FLEXSED_EXTRA` + `FLEXSED_EXTRA_DIR` +
`FLEXSED_EXTRA_QUERIES` (`add_flexsed_extra`: the 120 group-a queries appended as FlexSED columns, same fps 25 and frame
count as the main cache on all 49 clips; everything downstream treats them like the 215). With XQ the listener has no answer
for the new queries' runs (396 of 940 missing → not rescued). "@AG4" = the arm with its rule replaced by AGREE_V4.
Gates: flags off D0 98/98, conf-equality 98/98, D5 49/49 both systems.

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible | half A: hits / wrong / Δ | half B: hits / wrong / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] |
| XQ | 16 | 13 | 29 (7 / 15 / 7) | 3.06 | +0.29 [-0.04, +0.65] | 2.47 | -0.80 | no | 7 / 15 / +0.56 [+0.16, +1.12] | 6 / 14 / +0.00 [-0.33, +0.42] |
| XQ+1 | 18 | 14 | 33 (7 / 17 / 9) | 3.14 | +0.37 [-0.16, +0.98] | 2.47 | 0.00 | no | 7 / 16 / +0.64 [+0.16, +1.28] | 7 / 17 / +0.08 [-0.83, +1.17] |
| LR-V12+1+F1F4F3+F7F8+XQ | 18 | 14 | 34 (7 / 18 / 9) | 3.18 | +0.41 [-0.12, +1.06] | 2.49 | 0.00 | no | 7 / 16 / +0.64 [+0.16, +1.28] | 7 / 18 / +0.17 [-0.75, +1.33] |
| LR-AG4+1 | 23 | 20 | 32 (5 / 20 / 7) | 2.61 | -0.16 [-0.86, +0.45] | 1.96 | 3.00 | yes | 10 / 13 / -0.08 [-0.64, +0.40] | 10 / 19 / -0.25 [-1.58, +0.75] |
| LR-AG12+1 | 19 | 16 | 32 (7 / 18 / 7) | 2.94 | +0.16 [-0.24, +0.53] | 2.29 | 1.00 | no | 9 / 13 / +0.08 [-0.40, +0.48] | 7 / 19 / +0.25 [-0.50, +0.83] |
| LR-AG4+1+F1F4 | 19 | 16 | 29 (6 / 16 / 7) | 2.82 | +0.04 [-0.33, +0.37] | 2.22 | 1.60 | no | 9 / 11 / -0.08 [-0.48, +0.24] | 7 / 18 / +0.17 [-0.50, +0.83] |
| LR-AG12+1+F1F4 | 18 | 15 | 26 (6 / 13 / 7) | 2.78 | +0.00 [-0.33, +0.24] | 2.24 | 2.00 | no | 8 / 10 / +0.00 [+0.00, +0.00] | 7 / 16 / +0.00 [-0.58, +0.50] |
| LR-V12+1+F1F4F3+F7F8@AG4 | 19 | 16 | 23 (6 / 11 / 6) | 2.57 | -0.20 [-0.57, +0.12] | 2.10 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 7 / 13 / -0.25 [-0.92, +0.25] |
| LR-AG4+1+XQ | 23 | 18 | 36 (6 / 21 / 9) | 2.94 | +0.16 [-0.57, +0.82] | 2.20 | 1.33 | no | 9 / 19 / +0.56 [-0.24, +1.36] | 9 / 17 / -0.25 [-1.50, +0.83] |


**Reading.** XQ adds wrong pictures and loses the `birds_forest` Crowing (10.4 s) in every arm. The best AGREE arm is the old
best round-14 arm with its rule replaced by AGREE_V4 (2.57, Δ −0.20 [−0.57, +0.12]; gains Cricket and the snow-walk
Laughter); it does not beat TO1+F7F8 (2.45).

### Round 14 amendment K2 re-test with listener answers for the offered runs, DEV, jobs 31562960 (scores) + 31563031 (arms)
The 27 band runs that K2 newly offers (a same-family BEATs span below the display bar covers them, so they were never in P2;
list = the arm's own stage-4 log, `data/work/r13/k2_runs.json`) were scored with the amendment-A/C verifiers by importing
`listener_variants.py` and `listener_afnext.py` unchanged (`benchmark/gold/listener_k2.py`; P2 cut rule; null family by
`gold_free_null`, the gold-free rule, since these runs are not in the gold-based DEV cache): `dev_listener_k2.json` (Qwen:
V4 12/27, V12 9/27; nulls V1 1, V3 2, others 0) and `dev_listener_k2_afn.json` (AF V4 19/27, null 0). They are merged at
lookup (`LISTENER_VCACHE` / `LISTENER_AFCACHE` accept "base;supplement"). Arms K2x = TO1+F7F8 + K2, K2K3x = + K3; 0 missing
of 539 asked. (First attempt failed in the P1 lookup, which did not split joined paths; fixed, arms re-run; the scores were
kept.) Gates: D0 98/98, conf-equality 98/98, D5 49/49 both systems.

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible | half A: hits / wrong / Δ | half B: hits / wrong / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — | 8 / 10 / +0.00 [+0.00, +0.00] | 6 / 14 / +0.00 [+0.00, +0.00] |
| TO1+F7F8 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1F7F8+K2 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes | 9 / 10 / -0.16 [-0.56, +0.16] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1F7F8+K2x | 21 | 18 | 25 (6 / 12 / 7) | 2.49 | -0.29 [-0.82, +0.12] | 1.98 | 16.00 | yes | 9 / 11 / -0.08 [-0.56, +0.32] | 9 / 14 / -0.50 [-1.50, +0.25] |
| TO1F7F8+K2K3x | 21 | 18 | 26 (6 / 12 / 8) | 2.53 | -0.24 [-0.73, +0.16] | 2.00 | 8.00 | yes | 9 / 12 / +0.00 [-0.48, +0.40] | 9 / 14 / -0.50 [-1.42, +0.17] |


**Reading.** With answers, K2 still adds no hit: the offered runs mostly overlap BEATs spans of the same family that already
give (or miss) the picture; K2x adds 1 phantom (flea-market Vehicle 20.0 s) and swaps a hair-dryer Computer keyboard phantom
for a Typing one; K2K3x adds 2 more (rainforest Insect 11.24, and a `birds_forest` Bird at 0.52 s, 0.78 s before the needed
Bird's onset at 1.3 s — the K2 target stays a miss). TO1+F7F8 remains the best round-14 arm.

### Round 14 amendment K2 supplement (listener answers for the 27 missing runs), DEV, job 31563031 (6 min)
`benchmark/gold/listener_k2.py` built the 27 runs K2 newly offers (`data/work/r13/k2_runs.json`) like P2 items and scored
them with the unchanged Qwen3-Omni (V1–V4) and Audio Flamingo Next (V4 + yes/no) code → `dev_listener_k2.json`,
`dev_listener_k2_afn.json` (0 missing). `LISTENER_VCACHE` / `LISTENER_AFCACHE` take `;`-joined paths (`_cache_items`, also in
`listener_p1_lookup`). "x" = the K2 arms with these answers merged. Halves were not written by this job.

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ β=2 vs B0r [95 % CI] | cost β=1 | break-even β | eligible |
|---|---|---|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — | 2.29 | — | — |
| TO1+F7F8 | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | -0.33 [-0.86, +0.08] | 1.96 | any | yes |
| TO1F7F8+K2x | 21 | 18 | 25 (6 / 12 / 7) | 2.49 | -0.29 [-0.82, +0.12] | 1.98 | 16.00 | yes |
| TO1F7F8+K2K3x | 21 | 18 | 26 (6 / 12 / 8) | 2.53 | -0.24 [-0.73, +0.16] | 2.00 | 8.00 | yes |

**Reading.** With the answers in, K2 gains no needed sound (the `birds_forest` Bird runs are accepted by AF only, not by
Qwen V4) and adds one phantom (`un_hair_dryer_drying_WWu24rJs` Typing 10.32 s, both listeners say "typing"); K2+K3 adds a
second phantom (`birds_forest` Bird 0.52 s) and a rainforest Insect cross. Nothing beats TO1+F7F8; round 14 amendments
C, D, E–K and P3 are closed on DEV.

## Round 15 screens (2026-09-30, DEV, offline, candidate level; no pipeline arm run) — `benchmark/gold/r15_screen.py`
Adam: "keep trying more". Three offline screens on the pictures of TO1+F7F8 (and B0r), same scorer and cost:
- **Two-listener veto on drawn pictures** (drop a drawn picture when Qwen V1 no AND AF V4 no AND AF yes/no < 0 on its P1/PV
  items): removes 2 wrong (favela Train 3.8, laundromat Train 1.0) and 2 hits (rainforest Insect/Cricket 0.14, favela Train
  14.72). Dead: the listeners reject real sounds as often as phantoms.
- **S1 strict gate** (silent if ANY vote on ANY stretch says visible): B0r 14/24/2.78 → 14/20/2.61; TO1+F7F8 18/24/2.45 →
  17/21/2.41 (loses the rescued Laughter 8.08). This is §10d's any-stretch rule family (GOLD_RERUN_2026-09-22), already swept
  and rejected (higher sensitivity made end-to-end ΔF1 worse); Fable: report-only, not a TEST2 candidate.
- **S2 one sound, one picture** (drop a later same-family picture when the family's BEATs/FlexSED evidence stays >= 0.2 from
  the earlier picture's start): 17/22/2.45, same cost (drops detective Alarm 3.92, laundromat Train 16.25, and the favela
  Train 14.72 hit). Dead.
- **V5 masking-aware listener** ("apart from {louder sounds}, what else?"): not run. Fable: 0 DEV hits reachable (Footsteps
  2.1 and Gasp 6.7 are killed by F8, DASM 0.26 / below bar; Whistle 6.1 needs AF yes). The suspected AF duplicate-answer bug
  ("gunshot and gunfire, explosion" on 41 items) was checked: same cut → same answer, mostly the explosion clip; not a bug.

## Round 15 amendment M — Set-of-Mark crop vote for the gate (Fable idea 2; written 2026-09-30 BEFORE any number of it)
Motive (GOLD_RERUN §10e): 40 of 47 missed visible DEV sources are seen by no stretch at all — perception, not aggregation.
The whole-frame VLM misses small sources (church bell, fire-alarm box, air horn). Mechanism: look closer.
- **Crop vote.** For every (gold sound, stretch) of the cached DEV gate run (`benchmark/gold/gate_gold/Qwen38-27B`, same 6
  frames: stretch ± 1 s, as `gate_gold.run_vlm`), OWLv2 (`google/owlv2-base-patch16-ensemble`) is run with the family's
  `DETECT_QUERY` phrase (`src/stage2_video_understanding/owl.py`; no phrase → no crop vote, the stretch is decided as today).
  Boxes with score >= 0.1, top 2 per frame, cropped with a 20 % margin (at least 224 px on the short side after upscaling).
  Each crop goes to the gate VLM (Qwen3.8-27B, greedy) with: "This is a close-up cut from a video frame. Is it {phrase}? Answer
  yes or no." Crop vote = yes iff some crop gets "yes".
- **Rule M:** a stretch is "seen" iff the shipped majority says seen, OR (crop vote yes AND at least one of name/ab/desc says
  yes). The clip-level verdict stays "silent only if every stretch is seen". Rule M-any (report only): majority OR crop vote.
- **Screen (DEV 54 judge clips, all importance >= 2 gold sounds, `gate_gold.score` style):** seen_silenced and needed_kept for
  majority vs M vs M-any. **Go to a full DEV arm iff** M silences >= 3 more seen sounds AND loses <= 1 needed sound
  (needed_kept drop <= 1/needed). Full arm: TO1+F7F8 and B0r with rule M applied to their stored stage-5 votes plus crop votes
  on their gate stretches; adopted as a candidate for TEST2 only if cost drops on both bases and no needed hit is lost.

## Round 15 amendment N — audio-visual gate vote with Qwen3-Omni (Fable idea 4; written 2026-09-30 BEFORE any number of it)
The gate VLM sees frames but never hears the sound. Qwen3-Omni-30B-A3B gets the same 6 frames of each cached DEV gate
stretch (as amendment M) AND the audio of the stretch (16 kHz, `data/work/devcand/wav16`, stretch ± 1 s), with: "These are
frames from a video, and this is its sound. Is the {label} sound you hear made by something you can see in these frames?
Answer yes or no." Omni vote = (logit yes − logit no) > 0, the listener's yes/no readout. **Rule N:** stretch seen iff the
shipped majority, OR (Omni vote yes AND at least one of name/ab/desc says yes). Report-only: N-any (majority OR Omni yes).
Same screen, same GO bar as amendment M (>= 3 more seen sounds silenced, <= 1 needed sound lost, DEV judge clips).

### Confirmation set 1, DEV2 (6 clips, 3 needed sounds), jobs 31563213–15 (2026-09-30)
DASM restored (GitHub cai525/Transformer4SED c3e883d + HF CPF2/detect_any_sound); restore check: 4 DEV clips re-scored,
max |diff| 0.0 against the old cache (`benchmark/gold/dasm_restore_check.py`). DASM caches built for DEV2 and TEST2; gates
D0 20/20, D5 10/10. TEST2 stage 4/5 were built but NOT scored.

| arm | hits / 3 | wrong (v / c / p) | cost | Δ vs B0r [95 % CI] |
|---|---|---|---|---|
| B0r | 1 | 8 (0 / 8 / 0) | 4.00 | — |
| B1 | 1 | 8 (0 / 8 / 0) | 4.00 | +0.00 |
| TO1+F7F8 | 2 | 8 (1 / 7 / 0) | 3.33 | −0.67 [−2.00, +0.00] |

Same direction as DEV (+1 hit, same wrong), on 3 needed sounds: not a test. `benchmark/gold/dev2_score.json`.

### Round 15 amendment M result, DEV screen, job 31563428 (6 min) — STOP
| rule | seen silenced / 43 | needed kept / 36 |
|---|---|---|
| majority (shipped) | 16 | 31 |
| M | 17 | 30 |
| M-any (report only) | 17 | 28 |

M silences one more seen sound (crossing_bells Train 0.0) and one needed one (as_explosion Gunshot 0.0): below the GO bar
(>= 3 more seen, <= 1 needed lost). The close-up crops do not see the sources the whole frame misses. `gate_gold/som_summary.json`.

### Round 15 amendment N result, DEV screen, job 31563443 — STOP
| rule | seen silenced / 43 | needed kept / 36 |
|---|---|---|
| majority (shipped) | 16 | 31 |
| N | 21 | 28 |
| N-any (report only) | 26 | 25 |

N silences 5 more seen sounds but 3 needed ones (explosion Gunshot 0.0, golf Whack 24.4, motorcycle Laughter 12.7): below the
GO bar (<= 1 needed lost) and below the β = 2 break-even (2 seen per needed). Hearing the sound makes the VLM say "visible"
more often for both classes; it does not separate them. With M this closes Fable's gate ideas 2 and 4.

**Confirmation set 1 — batch 2 (Adam, 2026-09-30; fixed and committed before any pipeline run or output on these
clips).** 15 more clips from Adam's export `tagger_AG_2026-09-30_0347.json` (done, not broken; 2 broken to `_bad/`),
added to `tagger_AG.json` / `data/input/tagger_set/` by `benchmark/gold/tagger_set.py`. Adam asked to split them using
the tags, so from batch 2 on each NEW batch is split **tag-balanced** by `benchmark/gold/tagger_split.py` (ranked by
needed sounds, then all sounds, then sha256; consecutive pairs, alternating which side gets the richer clip). Only
batch 2's own tags were read; batch 1 stays exactly as it was (sha256 % 2; TEST2 tags never read).
Batch 2: **DEV2** = tg_d022, d029, d030, d032, d054, d085, d095 (7; 6 needed sounds); **TEST2** = tg_d009, d014, d019,
d023, d031, d045, d046, d068 (8; 7 needed). Totals: DEV2 13, TEST2 18. Membership: `benchmark/gold/tagger_split.json`.
Disclosure: Adam opened the BEATs-only suggestions on all 15 batch-2 clips while tagging (`suggestions_shown_at`).

## Round 15 amendment O — Whisper-AT as F8's third vote (Fable idea 3; Adam: "do whatever needed"; written 2026-09-30 BEFORE any number of it)
Motive: F8 (DASM) killed two rescued needed sounds (Hammer 13.7, DASM 0.281; Explosion 2.8, DASM 0.516 < 0.575), and the
misses are speech/music-masked (GOLD §10a). Whisper-AT (Gong et al., Interspeech 2023; pip `whisper-at`, official
checkpoints; Whisper large-v2 encoder + AudioSet-2M tagging head) was built to tag sounds under speech.
- **Scores.** Whisper-AT on each clip's 16-kHz audio, `at_time_res` 0.4 s → 527 AudioSet logits per 0.4-s step. Stored as a
  rank score: 1 / (rank of the class at that step), cache `data/work/wat_cache/<clip>.npz` (fw [T, 527], times = step start,
  labels = AudioSet names), same format F8 reads.
- **F8-W:** F8 unchanged (same family by `canonical`, span ± 0.5 s) but with `LISTENER_DASM_DIR` = the Whisper-AT cache and
  `LISTENER_DASM_BAR` = 0.33, i.e. the family is among Whisper-AT's top 3 classes at some step. Top-3 is fixed now (no sweep).
- **Arm O1** = TO1+F7F8 with F8-W in place of F8. Same full DEV pipeline and report; then DEV2 (report). Candidate for TEST2
  iff DEV-eligible and cost < TO1+F7F8's 2.45 with no needed sound lost vs TO1+F7F8.

### Round 15 amendment O result, DEV, job 31563681 — fails
Whisper-AT caches built for DEV (49), DEV2 and TEST2 (65 clips, `data/work/wat_cache/`), 0.4-s steps (the tool warns its head
was trained at 10 s). Gates as before.

| arm | heard | hits / 36 | wrong (v / c / p) | cost β=2 | Δ vs B0r [95 % CI] |
|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — |
| TO1+F7F8 (DASM vote) | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | −0.33 [−0.86, +0.08] |
| TO1F7F8+O (Whisper-AT vote) | 20 | 16 | 26 (5 / 15 / 6) | 2.69 | −0.08 [−0.61, +0.41] |

Whisper-AT keeps Explosion 2.8 (DASM killed it) but not Hammer 13.7; it lets through more cross pictures (crossing_bells
Water, favela Bird ×2 and Vehicle, citywalk Train, rainforest Bell) and loses the favela Train 14.6 and snow-walk Laughter.
Worse than the DASM vote on cost, hits and wrong: not a candidate. DASM stays as F8.

## Merge of the tagger set into DEV and TEST (Adam, 2026-09-30, written before any merged score)
"The purpose of me tagging more is to make dev and test larger." From now on **DEV = DEV (49) + DEV2** and **TEST = TEST (60)
+ TEST2**, with the same per-clip membership as before (batch 1 by sha256 parity, batch 2 tag-balanced, later batches the
same way). All selection happens on the merged DEV. The merged TEST is scored ONCE, at the end, for the final candidate vs
the shipped config (B0) under the prereg TEST rule (paired clip bootstrap, Holm over the candidates). Disclosure: the old
TEST part was read before for other detector candidates (rounds 4–13; R13-1 once); the final candidate (TO1+F7F8 unless a
merged-DEV winner replaces it) was chosen on DEV only and has never been run on TEST.

### Merged-DEV selection (written 2026-09-30, before any merged-DEV number)
Candidates (fixed now; all DEV-eligible arms of rounds 13–15 within 0.2 of the best DEV cost): TO1+F7F8, TO1+F7,
TO1+F7F8+FIX, TO1F7F8+K3, A1, LR-V12+1+F1F4F3+F7F8, LR-V12+1+F1F4F3+F7F8@AG4, R13-1; baselines B0r, B1. Each is run on the
tagger DEV part through `tagger_prep` (TG_ARMS) and scored with `benchmark/gold/merged_dev.py` on DEV + tagger DEV as one
set. **Pick** = the lowest merged-DEV cost among the candidates that are merged-DEV-eligible (hits > B0r, wrong <= B0r + 2 ×
gain, cost < B0r); ties → fewer wrong. The pick is the ONE candidate for the final merged TEST scoring (vs B0).

### Shipping TO1+F7F8 (Adam, 2026-09-30: "the shipped version SHOULD be the best version")
- **Runner** `slurm/run_best.sh NAME DIR [ARM]`: the tagger harness (`tagger_prep.py`, split NAME via TG_EXTRA_SPLITS) on any
  folder of clips, gold-free: FlexSED, scored render, wav16, BEATs, PANNs, B0r stage 4/5, listener pools, Qwen3-Omni, Audio
  Flamingo Next, DASM, then the arm's stage 4/5.
- **Check (job 31564168, 16 min):** 5 DEV clips re-run from scratch as split `shipcheck` (as_explosion, snow_walk,
  rainforest_7629, laundromat, tornado — the clips with rescues, F7 swaps and the PV keep): pictures of B0r and TO1+F7F8
  identical to the DEV harness run on 5/5 clips (`benchmark/gold/shipcheck_compare.py`); D0 10/10, D5 5/5.
- **Switch:** `config.use_shipped()` now sets the TO1+F7F8 flags and LISTENER_REQUIRE_CACHES; per-clip answers via
  `config.set_listener_split(NAME)` / `main.py --listener-split NAME`. A clip without them stops in stage 4 (no silent
  fallback). Stage 6: `scripts/best_to_protocol.py` → `repaint_shipped.py` → `recomposite.py`. Live (cache-free) listener
  and DASM for ComfyUI: TODO. The merged TEST (scored once, end) decides whether the switch stays.

**Confirmation set 1 — batch 3 (Adam, 2026-09-30; fixed and committed before any pipeline run or output on these
clips).** 19 AudioSet-Strong eval clips (`docs/tagger_selection_rule.md`, amendments 1–4; source map
`benchmark/gold/tagger_audioset_sources.json`) from Adam's export `tagger_AG_2026-09-30_0635.json` (done, not broken;
d102, d108 broken to `_bad/`), added by `tagger_set.py`. Split tag-balanced by `tagger_split.py` (same rule as batch 2):
**DEV2** = tg_d107, d120, d121, d125, d127, d128, d129, d133, d149 (9; 10 needed sounds); **TEST2** = tg_d101, d103,
d104, d105, d106, d109, d110, d112, d141, d146 (10; 10 needed). Totals: DEV2 22, TEST2 28.
Disclosure: Adam saw BEATs-only suggestions and/or the clips' AudioSet-Strong labels ("other annotators' tags") while
tagging most of these clips (`suggestions_shown_at`, `annotators_shown_at`, `from_annotators`), so this batch's gold is
anchored partly on AudioSet-Strong; clips are eval split (DASM's checkpoint was picked on that split).

## Round 16 — night arms on merged DEV (Adam asleep, 2026-09-30; Fable night plan; written BEFORE any merged-DEV number)
Base = **TO1+F7F8**. Pass rule for every arm, on merged DEV (DEV 49 + tagger DEV 22): hits ≥ base hits, wrong ≤ base wrong +
2 × hits gained, cost < base cost, AND no needed hit lost on either part (old 49 / tagger 22) separately; each arm also
reported on the tagger part alone (a win only on the old 49 is "not confirmed"). Arms that pass are stacked in order of
merged cost and the stack is scored once. These arms are never added to the frozen 8-candidate merged-DEV pick.
- **N1 scene fit on all pictures** (`SCENE_FIT_ALL`): the F3 prompt ("Could the sound of {X} plausibly be heard in this
  scene? Answer yes or no.", gate VLM, gate frames, majority of stretches) asked for EVERY drawn sound, not only rescued ones;
  "no" → silent. Extra pass condition: wrong ≤ base − 2.
- **N2 masked weak BEATs** (`MASKED_WEAK_VETO`): a BEATs-origin span with conf < 0.5 AND BEATs Speech or Music ≥ 0.3 at some
  frame inside it is dropped unless its FlexSED twin is ≥ 0.8 or the listener accepts it (F7's P1 keep rule). Constants from
  the 280/415 phantom audit, not DEV. Extra pass condition: wrong ≤ base − 3.
- **N3 third listener, 2-of-3** (`LISTENER_RULE` "TIER3"): a third open-inventory audio LLM (Kimi-Audio-7B-Instruct, official
  weights, if the HF id resolves; same V4 prompt and matching) on the same P2/PV cuts; high tier (peak ≥ 0.6): ≥ 2 of
  {Qwen, AF-Next, Kimi} name the family; low tier: all 3.
- **N4 DASM rank readout** (F8 variant): the family is among DASM's top-3 of the 215 queries at some frame in span ± 0.5 s
  (instead of ≥ 0.575).
- **N5 audio-visual gate vote** (DenseAV, if the official weights load): stretch "seen" only if the shipped majority says
  seen AND DenseAV grounds the stretch audio in the frames; screen and GO bar as amendment M.

### Round 16 N2 / N4 on the old DEV 49 (job 31564506, 12 min) — both fail the pre-set rule
| arm | heard | hits / 36 | wrong (v / c / p) | cost | Δ vs B0r [95 % CI] |
|---|---|---|---|---|---|
| B0r | 17 | 14 | 24 (6 / 11 / 7) | 2.78 | — |
| TO1+F7F8 (base) | 21 | 18 | 24 (6 / 12 / 6) | 2.45 | −0.33 [−0.86, +0.08] |
| TO1F7F8+N2 masked weak BEATs | 19 | 17 | 20 (4 / 12 / 4) | 2.37 | −0.41 [−0.94, +0.08] |
| TO1F7F8+N4 DASM rank top-3 | 22 | 18 | 30 (5 / 19 / 6) | 2.69 | −0.08 [−0.69, +0.53] |

N2 removes 6 wrong pictures (london Vehicle and motorcycle Fireworks visible, hair-dryer Keyboard and laundromat Train
phantoms, explosion Gunshot and favela Train cross; 2 come back re-cut) but loses the bakery Door 3.9 hit: lower cost, yet
it fails the pre-set rule (a needed hit lost on a part). N4 lets 7 more cross pictures through: fails. N1's first run was
void (SCENE_FIT_ALL was not in the harness's stage-5 keys; fixed, re-run as job 31564722).
**N3 clarification (written after the Kimi answers exist, before any N3 arm number).** Kimi-Audio answers the V4 prompt with
one comma-separated line of AudioSet-style tags (`Bird,Wild_animals,Animal`) instead of one sound per line. The V4 rule is
applied to its list read as items (commas → lines, `_` → spaces; flag `accept_norm.V4`), the faithful parse of the prompt's
"one per line". Its audio cut is each item's `run_audio` (the cut Qwen and AF heard). Candidate-level counts seen (DEV P2+PV):
Kimi-norm accepts 141 (null 27), Qwen 68 (null 3), AF 129 (null 16). Arm TO1F7F8+N3 = TO1+F7F8 with rule TIER3.

### Merged DEV (DEV 49 + tagger DEV 22 = 71 clips, 55 needed sounds) — pick, jobs 31564839 + CPU scoring (2026-09-30)
Listener caches for all 50 tagged clips rebuilt from scratch (batch-1 answers reproduced exactly: 129/129 DEV2, 208/208
TEST2 items for both V and AF). Coverage: every tagged clip has yes/no, variant, AF answers and DASM.

| arm | hits / 55 | wrong (v / c / p) | cost | Δ vs B0r [95 % CI] | old DEV hits / wrong | tagger DEV hits / wrong |
|---|---|---|---|---|---|---|
| B0r | 16 | 53 (7 / 37 / 9) | 3.69 | — | 14 / 24 | 2 / 29 |
| B1 | 15 | 62 (7 / 47 / 8) | 4.00 | +0.31 [+0.09, +0.62] | 13 / 30 | 2 / 32 |
| **TO1+F7F8** | **23** | **48 (8 / 32 / 8)** | **3.16** | **−0.54 [−0.96, −0.17]** | 18 / 24 | 5 / 24 |
| TO1F7F8+K3 | 23 | 49 (8 / 33 / 8) | 3.18 | −0.51 [−0.90, −0.17] | 18 / 24 | 5 / 25 |
| TO1+F7F8+FIX | 23 | 50 (8 / 34 / 8) | 3.21 | −0.48 [−0.93, −0.09] | 18 / 26 | 5 / 24 |
| A1 | 21 | 48 (8 / 32 / 8) | 3.27 | −0.42 [−0.79, −0.11] | 16 / 24 | 5 / 24 |
| LR-V12+1+F1F4F3+F7F8 | 20 | 46 (8 / 30 / 8) | 3.27 | −0.42 [−0.73, −0.14] | 15 / 22 | 5 / 24 |
| LR-V12+1+F1F4F3+F7F8@AG4 | 20 | 47 (8 / 31 / 8) | 3.30 | −0.39 [−0.73, −0.11] | 16 / 23 | 4 / 24 |
| TO1+F7 | 23 | 63 (7 / 47 / 9) | 3.58 | −0.11 [−0.59, +0.37] | 18 / 32 | 5 / 31 |
| R13-1 | 18 | 53 (7 / 37 / 9) | 3.58 | −0.11 [−0.34, +0.06] | 15 / 24 | 3 / 29 |

**Pick (pre-set rule): TO1+F7F8** (lowest merged-DEV cost, merged-DEV-eligible). On the tagger DEV part alone it also wins:
5 vs 2 hits, 24 vs 29 wrong. It is the one candidate for the merged TEST.
N1 (scene fit on all pictures, job 31564838) on DEV 49: 14 hits / 16 wrong / 2.45 — same cost as the base but 4 hits lost:
fails the pre-set rule.

### Round 16 N3 (Kimi-Audio third listener, TIER3) — fails
DEV 49 (job 31564918): 18 / 24 / 2.45, identical to the base. Merged DEV (job 31564919): 22 hits / 48 wrong / 3.21 vs base
23 / 48 / 3.16 — loses one needed hit on the tagger DEV part (4 vs 5). Fails the pre-set rule.
Merged TEST preparation (job 31564931, gold-free): DASM for the old TEST 60; old TEST stage 4/5 of B0r and TO1+F7F8 in
data/work/r16final, gates pass; tagger TEST part (28 clips) stage 4/5 of B0r and TO1+F7F8, D0 56/56, D5 28/28. Not scored.

## Round 17 (night, 2026-09-30; Fable consult 3; written BEFORE any number of these arms)
Base TO1+F7F8; the Round-16 pass rule (merged DEV, hits ≥ 23, wrong ≤ 48 + 2 × gain, cost < 3.155, no needed hit lost on
either part; tagger part reported alone). Disclosure: R1 and R3 were shaped by reading the base's wrong pictures on the
tagger DEV part. TEST stays TO1+F7F8 (a passing arm is prepared gold-free on merged TEST only, not scored).
- **R1 relabel, don't drop (`RELABEL_2L`):** for a drawn BEATs-origin span (not rescued) whose P1 cut both open-inventory
  listeners answered (Qwen3-Omni V4, same prompt/decoding as amendment A, new job on P1 cuts; Audio Flamingo Next V4, cached):
  if neither names the span's own family and both name the same depictable family F (the V4 matcher — word or cosine > 0.6 —
  run against every depictable family), the span's label becomes F (first such F in Qwen's order). Extra pass: cross ≤ base − 3.
- **R3 co-onset arbitration (`CO_ONSET_ARB`):** two drawn spans of different families whose starts are within ±0.3 s: keep
  only the one with the higher FlexSED family peak inside its own span (F5's evidence, now on all spans), unless the listener
  (P1 V12/yes-no as F7) accepts both. Extra pass: wrong ≤ base − 2.
- Checked and closed without an arm: the two late onsets on the tagger part (Crying 1.25 s vs 0.0; Thunder 8.75 vs 7.4) are
  detection limits, not a bug (BEATs < 0.175 before 1.25 s; the Thunder 8.0 span is below the display bar); hysteresis onsets
  were tried on 14 and 22 Sept (ledger). The 0.25-s uniform lead (screened: +1 hit, −1 wrong on merged DEV for both B0r and
  the base) is not an arm: a 2-value sweep on the selection set, reported as a screen only.

### Round 17 R3 (co-onset arbitration) — fails
DEV 49 (job 31565012): 17 / 23 / 2.49 vs base 18 / 24 / 2.45. Merged DEV (job 31565013): 22 / 47 / 3.18 vs base 23 / 48 /
3.16 — one needed hit lost on the old DEV part, no change on the tagger part. Fails.
P1 V4 job (31565000, 26 min): Qwen V4 on the P1 cuts of DEV 274, tagger DEV 102, TEST 259, tagger TEST 166 items, with both
listeners' family lists (`benchmark/gold/*_listener_p1v4.json`). Candidate-level count seen before the R1 arm: the R1 rule
would relabel 21 DEV and 13 tagger-DEV P1 spans (e.g. bell_miami Train → Bell, helicopter Vehicle → Aircraft, waterfall
Vehicle → Water, tg_d120 Screaming → Cat; also many → Alarm, "Natural sounds", "Arrow").

### Round 17 R1 (relabel via two listeners) — fails
DEV 49 (job 31565655): 18 / 24 / 2.45, unchanged (3 clips relabelled: bell_miami Train → Bell at 8.0 s, helicopter Vehicle →
Aircraft, glass Glass → Alarm; none changes a hit). Merged DEV (job 31565656): 23 / 49 / 3.18 — one more wrong picture on the
tagger part. Fails.

### Merged TEST — the one exposure (decided 2026-09-30 ~07:50 UTC, before reading any TEST number)
All night arms (N1–N4, R1, R3) failed on merged DEV, so the single TEST candidate is the pre-set merged-DEV pick TO1+F7F8 vs
B0r, as registered. Scored now with `benchmark/gold/final_test.py score` (stages built gold-free by job 31564931).

### Merged TEST result (one exposure, 2026-09-30 ~08:00 UTC) — verdict **same**
Old TEST 60 (test_bench) + tagger TEST 28 = 88 clips, 65 needed sounds. `benchmark/gold/final_test.{json,md}`.
(A first attempt stopped before any gold was read: the tagger part's flag mapping refused TEST-mapped paths; fixed so both
parts map from the DEV originals. No `.started` marker existed before the scored run.)

| arm | hits | misses | wrong (v / c / p) | cost | old TEST hits / wrong | tagger TEST hits / wrong |
|---|---|---|---|---|---|---|
| B0r (old shipped) | 21 | 44 | 40 (2 / 31 / 7) | 2.909 | 16 / 25 | 5 / 15 |
| TO1+F7F8 (shipped now) | 22 | 43 | 38 (2 / 30 / 6) | 2.818 | 16 / 25 | 6 / 13 |

Δ cost −0.091 [−0.250, +0.045], one-sided p 0.132 → **same** (hits not lower, wrong lower, but p ≥ 0.05). The best version ran
as intended on the old TEST (46 band rescues accepted, 24 removed by the filters, 13 PANNs-vetoed spans kept, 5 of 60 clips
with different pictures), netting zero change there; the gain is on the tagger TEST part (+1 hit, −2 wrong).
**Decision (Adam's rule: switch back only if TEST shows it is worse):** TO1+F7F8 stays the shipped setting.

**Post-hoc checks of the TEST exposure (descriptive only; no config choice follows).**
- Funnel, gold-free: after the rescue filters, TO1+F7F8 keeps 8 rescued stage-4 rows on the old TEST (6 clips; 2 shown after
  the gate/display) vs 8 on DEV 49 (7 clips; 4 shown). 5 of 60 old-TEST clips have different pictures (aquarium Water added,
  smoke-alarm Alarm moved 10.0 → 9.55 s, film Gunshot removed, air-raid Alarm added, dog-barking Vehicle removed / Siren
  moved); their gains and losses cancel. No sign of a broken path; later arms re-ran TO1+F7F8 on the tagger DEV part and
  reproduced 5 hits / 24 wrong each time.
- Power (approximate, from the bootstrap CI half-width 0.15): SD of the per-clip Δ ≈ 0.71, so the one-sided 80 %-power
  minimum detectable effect on 88 clips is ≈ 0.19 cost; detecting −0.09 would need ≈ 385 clips.
- Winner's curse: the DEV gain (−0.54 on merged DEV) was selected from 8 frozen candidates after 106 DEV arms; on fresh
  tagged clips the gain is smaller (tagger DEV +3 hits / −5 wrong; tagger TEST +1 / −2). Pooled DEV+TEST numbers are
  descriptive only.

### On-the-spot inputs for new videos (Adam: "everything needs to be able to calculate on spot", for ComfyUI)
`src/listener_prep.py` runs the same harness (tagger_prep: FlexSED, scored render, wav16, BEATs, PANNs, B0r stage 4/5, pools,
Qwen3-Omni, Audio Flamingo Next, DASM) as a one-clip split; `main.py` calls it when no `--listener-split` is given, so the
shipped TO1+F7F8 runs on any new video with no manual step. Check (job 31565735, 5 min): as_explosion_XJ8lc3I6 answers
identical to the shipcheck run (yes/no 95/95, variants 37/37, AF 37/37).

## Round 18 (exploratory; TEST is spent — any result here is DEV-only and cannot be shipped without new data)
- **N2b:** N2 (masked weak BEATs veto) with the keep rule widened to either listener: the span is kept if Qwen (F7's P1 rule)
  OR Audio Flamingo Next V4 (P1 cut, cached) names its family. Written before its number; same pass rule as round 16.

### Round 18 N2b — PASSES the pre-set rule on merged DEV (exploratory: TEST is spent)
DEV 49 (job 31565770): 18 hits / 22 wrong (6 / 11 / 5) / 2.37 vs base 18 / 24 / 2.45. Merged DEV (job 31565771): **23 hits /
44 wrong (8 / 30 / 6) / 3.042** vs base 23 / 48 / 3.155 (Δ vs B0r −0.65 [−1.10, −0.25]); no needed hit lost on either part
(old DEV 18 / 22, tagger DEV 5 / 22 vs 18 / 24, 5 / 24). N2 lost the bakery Door because only Qwen was asked; with Audio
Flamingo as a second keep vote the Door stays and 4 wrong pictures still go. Status: a DEV-only improvement found after TEST
was spent — it cannot be confirmed on TEST (one exposure used). Shipping it is Adam's call; its TEST stage 4/5 are prepared
gold-free (not scored) in case he wants a descriptive read, which would be a second TEST look and must be labelled so.
- **N2c (exploratory, written before its number):** N2b without the masking condition — every weak (conf < 0.5) BEATs-only
  span without a FlexSED twin needs either listener (Qwen P1 rule or AF V4 on the P1 cut) to be kept. Same pass rule vs the
  base TO1+F7F8, and reported vs N2b.

### Round 18 N2c — lowest cost, but fails the rule (loses one needed hit)
DEV 49 (job 31565819): 18 / 21 (6 / 11 / 4) / 2.33. Merged DEV (job 31565832): **22 hits / 38 wrong (8 / 26 / 4) / 2.930**
(Δ vs B0r −0.76 [−1.24, −0.34]) vs base 23 / 48 / 3.155 and N2b 23 / 44 / 3.042; tagger DEV part 4 / 17 (one needed hit
lost). Fails the pre-set rule. Note for Adam: it is the cheapest arm on merged DEV (10 fewer wrong pictures for one hit) —
a "fewer false pictures" setting if he prefers precision; DEV-only, TEST spent.
N2b on the tagger TEST part and old TEST: stage 4/5 built gold-free (job 31565818, D0 56/56, D5 28/28; old TEST gates pass).
- **N2d (exploratory, written before its number):** N2b plus the same confirmation for FlexSED-only spans (not rescued):
  kept only if BEATs gives the family ≥ its own bar (0.175) somewhere inside, or either listener accepts it. Same rule.

### Round 18 N2d — fails
DEV 49 (job 31565866): 16 / 22 / 2.53; merged DEV (job 31565867): 20 hits / 43 wrong / 3.18 — three needed hits lost
(FlexSED-only real sounds that neither BEATs nor a listener confirms). Fails.

### Shipping decision for N2b (Fable, 2026-09-30 ~07:30 UTC)
Option (b): TO1+F7F8 stays the default (TEST-guarded); N2b ships as an opt-in (`config.use_n2b()`, `main.py --fewer-false`).
No second TEST read: underpowered for a 0.11 gain (MDE ≈ 0.19) and it would end the one-exposure claim. Thesis disclosure:
"After the single pre-registered TEST exposure, one further variant (N2b: weak masked BEATs-only spans under speech or music
require confirmation by a listener) lowered wrong pictures on merged DEV (44 vs 48, no hit lost) but was found after TEST was
spent and was not scored on TEST; it ships as an opt-in flag and contributes to no reported TEST result."

### End-to-end check of the shipped pipeline (job 31568897, 4 min 38 s on an H200)
`main.py --input ambient_snow_walk_930.mp4` with the shipped setup and no precomputed answers: on-the-spot listener inputs
built (yes/no 71 spans, variants 19, AF 19, DASM), stage 4 rescued 3 band runs and kept 1 vetoed span, the gate kept the
off-screen Laughter (the needed sound TO1+F7F8 gains on DEV), Qwen-Image drew "a person laughing", video written.

## Round 19 — DASM as a third ear for missed sounds (exploratory; TEST spent → opt-in at best; written BEFORE any arm number)
Candidate-level screen seen (merged DEV): of the base's 32 missed needed sounds, 11 have a same-family DASM frame ≥ 0.575
(its F8 bar) within [onset − 0.5, onset + 1.0] s — e.g. tg_d133 Fart ×2, tg_d120 Meow, tg_d107 Laughter, citywalk air horn,
birds_forest Bird, ambulance Vehicle — while 3 of the 11 are gate-visible misses no detector change can fix.
- **P4 pool:** DASM runs (family column ≥ 0.575, gaps ≤ 0.5 s merged) that no B0r stage-4 span of the same family touches
  within ± 0.5 s; cut = run ± 1 s (at least 1 s), as the other pools.
- **Listeners:** Qwen3-Omni V4 and Audio Flamingo Next V4 on each P4 cut (same prompt, decoding and matcher).
- **Arm DR = TO1+F7F8 + DASM rescue:** a P4 run becomes a rescued span (label = its family, confidence = its DASM peak) iff
  BOTH listeners name its family (the base's low-tier rule); then the base's rescue filters (ONCE, F8) apply unchanged.
- Pass rule vs TO1+F7F8 as round 16 (hits ≥ 23, wrong ≤ 48 + 2 × gain, cost < 3.155, no needed hit lost on either part).

### Round 19 DR (DASM third ear) — fails
DEV 49 (job 31580331): 17 / 24 / 2.53; merged DEV (job 31580332): 23 hits / 49 wrong / 3.18 — gains a hit on the tagger part
(6 vs 5) but loses one on the old DEV part (ONCE keeps the earliest rescued span of a family, and a DASM run can come
first). Fails the rule.

### N2b becomes the shipped default (Adam, 2026-09-30 07:50 UTC: "merged means u can push the new best every time u find one
(after documented it)")
`config.use_shipped()` now = TO1+F7F8 + N2b (MASKED_WEAK_VETO, MASKED_WEAK_AF). Evidence: merged DEV 71 clips, 23 hits / 44
wrong / 3.042 vs TO1+F7F8 23 / 48 / 3.155, no needed hit lost on either part; found after the one TEST exposure, so it has
no TEST check (thesis disclosure as written above, "ships as an opt-in" → "ships as the default").

## Round 20 (base = shipped TO1+F7F8 + N2b, merged DEV 23 / 44 / 3.042; exploratory, DEV-only; written before its number)
- **DR2 (DASM third ear, new families only):** as DR, but a confirmed P4 run is added only if its family has no stage-4 span
  anywhere in the clip (DR's failure: a DASM run of an already-present family came first and ONCE then dropped the later hit).
  Pass rule vs the base: hits ≥ 23, wrong ≤ 44 + 2 × gain, cost < 3.042, no needed hit lost on either part.

### TEST reads of later bests (Adam, 2026-09-30 07:50 UTC: "it is winning on dev so its better but need to also check test.
to report the stats")
Each new shipped best is also scored on the merged TEST and REPORTED (`final_test.py score --tag <name>`, its own record
file). These are later TEST reads made after selection on DEV; they are reported, never used to choose between arms, and the
thesis must state how many TEST reads were made. First: N2b (tag n2b).
**N2b on the merged TEST (second TEST read, reported; `benchmark/gold/final_test_n2b.{json,md}`):**

| arm | hits | misses | wrong (v / c / p) | cost | old TEST hits / wrong | tagger TEST hits / wrong |
|---|---|---|---|---|---|---|
| B0r | 21 | 44 | 40 (2 / 31 / 7) | 2.909 | 16 / 25 | 5 / 15 |
| TO1+F7F8 (first read) | 22 | 43 | 38 (2 / 30 / 6) | 2.818 | 16 / 25 | 6 / 13 |
| **TO1+F7F8 + N2b (shipped)** | **22** | **43** | **35 (3 / 27 / 5)** | **2.750** | 16 / 24 | 6 / 11 |

Δ cost vs B0r −0.159 [−0.341, +0.000], one-sided p 0.031 → "better" under the prereg rule (hits not lower, wrong −5, p <
0.05). Caveat for the thesis: this is the second TEST read, made after N2b was chosen on DEV; with two reads a Holm/Bonferroni
correction (α 0.025) would make p 0.031 not significant.

### Round 20 DR2 — PASSES; new shipped best
DEV 49 (job 31586738): 18 / 21 (6 / 10 / 5) / 2.33 vs base 18 / 22 / 2.37. Merged DEV (job 31586739): **24 hits / 44 wrong
(9 / 29 / 6) / 2.986** vs base (TO1+F7F8 + N2b) 23 / 44 / 3.042; no needed hit lost on either part (old DEV 18 / 21, tagger
DEV 6 / 23 vs 18 / 22, 5 / 22). DASM, as a third ear for sound types nothing else detected in the clip, confirmed by both
audio LLMs, adds a needed sound without adding wrong pictures overall. Merged into `use_shipped()`; TEST read 3 prepared.
**DR2 (shipped TO1+F7F8 + N2b + DR2) on the merged TEST (third TEST read, reported; `final_test_dr2.{json,md}`):**

| arm | hits | misses | wrong (v / c / p) | cost | old TEST hits / wrong | tagger TEST hits / wrong |
|---|---|---|---|---|---|---|
| B0r | 21 | 44 | 40 (2 / 31 / 7) | 2.909 | 16 / 25 | 5 / 15 |
| TO1+F7F8 + N2b (read 2) | 22 | 43 | 35 (3 / 27 / 5) | 2.750 | 16 / 24 | 6 / 11 |
| **+ DR2 (shipped)** | **24** | **41** | **37 (3 / 28 / 6)** | **2.705** | 18 / 25 | 6 / 12 |

Δ cost vs B0r −0.205 [−0.455, +0.023], one-sided p 0.051 → "same" by the rule's p < 0.05 (third read; reported only). DR2
adds 2 needed hits on the old TEST for 2 more wrong pictures than N2b: the lowest TEST cost of any version so far.

## Round 21 (base = shipped SHIP2 = TO1+F7F8 + N2b + DR2, merged DEV 24 / 44 / 2.986; Fable ideas; written before numbers)
Pass rule vs the base: hits ≥ 24, wrong ≤ 44 + 2 × gain, cost < 2.986, no needed hit lost on either part.
- **N2e:** the N2 veto triggers on weak (< 0.5), untwinned BEATs-only spans whose family PANNs does not reach its clip-veto
  bar (0.05) inside the span (instead of "under speech/music"); kept if Qwen (P1 rule) or AF V4 names it.
- **N2c-D:** N2c (every weak untwinned BEATs-only span needs a second opinion) with two more keeps: DASM ≥ 0.575 for the
  family within the span ± 0.5 s, or no cached listener answer for the span.

### Round 21 results — both fail (one needed hit lost on the tagger part: tg_d032 Thunder 12.8, a weak BEATs span no second opinion confirms)
| arm | merged hits | wrong (v / c / p) | cost | old DEV | tagger DEV |
|---|---|---|---|---|---|
| SHIP2 (base) | 24 | 44 (9 / 29 / 6) | 2.986 | 18 / 21 | 6 / 23 |
| SHIP2+N2e | 23 | 38 (9 / 25 / 4) | 2.873 | 18 / 20 | 5 / 18 |
| SHIP2+N2c-D | 23 | 40 (9 / 26 / 5) | 2.930 | 18 / 20 | 5 / 20 |
(jobs 31593633 / 31593634). Lower cost, but the rule forbids losing a needed hit.
- **K-V4 (written before its number):** base SHIP2; the F7 and N2b keeps additionally need an open-inventory naming of the
  span's family on its P1 cut (Qwen V4 from `*_listener_p1v4.json` or AF V4). Same pass rule.

### Round 21 K-V4 — PASSES; new shipped best
DEV 49 (job 31593679): 18 / 20 (6 / 10 / 4) / 2.29. Merged DEV (job 31593680): **24 hits / 41 wrong (9 / 27 / 5) / 2.901** vs
base SHIP2 24 / 44 / 2.986; no needed hit lost on either part (old DEV 18 / 20, tagger DEV 6 / 21). Merged into
`use_shipped()`; on-the-spot prep now also builds the P1 open-inventory answers (`listener_p1v4.py`). TEST read 4 prepared.
**K-V4 (shipped TO1+F7F8 + N2b + DR2 + K-V4) on the merged TEST (fourth read, reported; `final_test_kv4.{json,md}`):**
23 hits / 36 wrong (3 / 27 / 6) / 2.727 vs B0r 21 / 40 / 2.909, Δ −0.182 [−0.455, +0.068], p 0.103 → "same". Versus the
previous shipped version on TEST (read 3: 24 / 37 / 2.705) it is one hit and one wrong picture lower (tagger TEST part 5 / 11 vs
6 / 12): on TEST, K-V4's DEV gain does not show. TEST reads are reported, not used to choose, so K-V4 stays shipped by the DEV
rule; the thesis should show all four TEST reads together.

### Statistics of the shipped stack (Fable review; recomputed from saved pictures — no new exposure, no decision)
`benchmark/gold/stack_stats.{py,json}`; paired clip bootstrap 2000, seed 0.

| step | DEV Δ cost [95 % CI] | TEST Δ cost [95 % CI] |
|---|---|---|
| B0r → TO1+F7F8 | −0.535 [−0.958, −0.169] | −0.091 [−0.250, +0.045] |
| + N2b | −0.113 [−0.225, −0.028] | −0.068 [−0.182, +0.045] |
| + DR2 | −0.056 [−0.225, +0.056] | −0.045 [−0.250, +0.136] |
| + K-V4 | −0.085 [−0.197, +0.000] | +0.023 [−0.068, +0.136] |
| **stack vs B0r** | **−0.789 [−1.268, −0.366]** | **−0.182 [−0.455, +0.068]** |

TEST vs B0r over the four reads (one-sided p → Holm): TO1+F7F8 0.132 → 0.206; +N2b 0.031 → 0.124; +DR2 0.051 → 0.153;
+K-V4 0.103 → 0.206 — none significant after Holm; MDE ≈ 0.19 on 88 clips. Every step points the same way on TEST except
K-V4 (+0.02, well inside one SE). About a quarter of the DEV gain survives on TEST (winner's curse after ~120 DEV arms).
Fable: keep K-V4 shipped (reverting to DR2 because of read 3 would be choosing on TEST); stop stacking DEV precision rules —
TEST wrong pictures are flat since N2b; the only TEST signal is recall (DR2 +2 hits). Primary thesis result = the one
pre-registered exposure (TO1+F7F8 vs B0r, "same"); reads 2–4 are a labelled secondary table.

## Round 22 (recall; base = shipped SHIP2+KV4, merged DEV 24 / 41 / 2.901; written before its number)
- **TD (two of three independent opinions):** a FlexSED band run (P2) is rescued if the current TIER rule accepts it, OR if at
  least two of {Qwen V4 names it, Audio Flamingo V4 names it, DASM gives its family ≥ 0.575 within the run ± 0.5 s} agree.
  Screen motive: the citywalk air horn (AF yes, Qwen no, DASM 0.72) is rejected today. Pass rule vs the base: hits ≥ 24,
  wrong ≤ 41 + 2 × gain, cost < 2.901, no needed hit lost on either part.

### Round 22 TD — fails (merged DEV 24 hits / 43 wrong / 2.958 vs base 24 / 41 / 2.901; jobs 31593870 / 31593871)
- **ONCE-G (written before its number):** ONCE keeps the earliest rescued span of a family, and drops a later one only if it
  starts within MERGE_GAP (2.0 s, the scored display's merge gap) of the previous kept one — later rescues farther apart are
  separate events. Base SHIP3; same pass rule (hits ≥ 24, wrong ≤ 41 + 2 × gain, cost < 2.901, no hit lost on either part).

### Round 22 ONCE-G — fails (merged DEV 24 hits / 44 wrong / 2.986 vs base 24 / 41 / 2.901; jobs 31593905 / 31593906):
later same-family rescues are mostly repeats of one sound (3 more wrong pictures), no hit gained.

### Round 23 screens (Fable, candidate level; no arm run)
- **SS short-burst path** (FlexSED ≥ 0.8 runs shorter than 0.5 s, not covered, DASM ≥ 0.575 within ± 0.5 s): merged DEV has
  2 hit-class candidates vs 75 others (a vacuum cleaner alone gives 27). Far above the GO bar (≤ 2 wrong per hit): NO GO.
- **F8-A** (F8 abstains where DASM is deaf): the tagger-DEV Thunder runs (the only targets) are rejected by both listeners
  (Qwen V4 and AF V4 no at 2.04–4.64 and 6.2–8.56 s) — nothing to rescue: not run.
- **FV** (strong BEATs spans removed by the FlexSED clip veto, e.g. tg_d029 Chicken 0.81, FlexSED family max 0.26): merged DEV
  has 3 needed-class candidates (one chicken twice, one cat) vs 47 others; with the listeners' usual precision (~3 wrong per
  hit) it cannot meet wrong ≤ 2 × gain: NO GO.

## Round 24 — Gemma-4-31B as the visibility gate (written before any number)
The gate VLMs tried so far are Qwen3.8-27B (shipped), Qwen2.5-VL-7B, OWLv2 and SAM 3; Gemma-4-31B (cached, the only
judge that passed the trust checks) was never asked the gate's questions. Screen: `gate_gold.py --model google/gemma-4-31B-it
--dev-only` (same six frames, same three votes, majority rule) on the DEV judge clips, scored like amendment M against
Qwen3.8-27B. GO to a full arm iff Gemma silences ≥ 3 more seen sounds AND keeps ≥ as many needed sounds (the gate's
known failure is missed visible sources and wrongly silenced needed ones — the Fart / pet-shop / bell cases).

### Round 24 result — Gemma-4-31B gate: NO GO (job 31594011, 27 min, judge venv torch 2.7)
DEV judge clips (79 sounds: 43 seen, 36 needed), majority rule: Gemma silences 24/43 seen (Qwen3.8 16/43) but keeps only
26/36 needed (Qwen3.8 31/36) — 8 more leaks caught for 5 needed sounds lost, below the break-even and the pre-set bar
(keep ≥ as many needed). Unanimous: 10/43 and 33/36; obvious: 18/43 and 31/36 (+2 seen, same needed, below the ≥ 3 bar).
A fourth VLM moves the gate along the same trade-off line, as the 26 Sept panel predicted.

## Round 25 — cross-validated re-selection of the two never-tuned listener parameters (Fable; written before any number)
Base SHIP3 (= SHIP2+KV4). `config.TIER_SPLIT` (default 0.6, read by `_tier`) and `LISTENER_LO` (0.5). Grid TIER_SPLIT ∈
{0.5, 0.6, 0.7} × LISTENER_LO ∈ {0.4, 0.5, 0.6}, cells with LO ≥ SPLIT void; SHIP3 = (0.6, 0.5), already scored; new cells
CV54, CV64, CV74, CV75, CV76, each run once on merged DEV. Selection: 5-fold over clips stratified by part, seeds 0–9; per fold
the cell with the lowest training cost (ties → SHIP3); report the procedure's mean held-out cost vs the fixed SHIP3 cell.
Candidate = the full-merged-DEV argmin. Adopt iff the procedure's CV cost ≤ SHIP3's CV cost AND the standard pass rule (hits ≥
24, wrong ≤ 41 + 2 × gain, cost < 2.901, no needed hit lost on either part). If the argmin is SHIP3: "stable under CV", no
TEST read. Any adopted cell is reported on TEST afterwards, never chosen on it.

### Round 25 result — stable under CV (jobs 31594170 / 31594171; `benchmark/gold/cv_select.json`)
Merged DEV: CV54, CV64, CV74, CV75 give exactly the shipped pictures (24 / 41 / 2.901); CV76 (LO 0.6) loses 2 hits (22 / 41 /
3.014). 5-fold CV (10 seeds): the procedure picks the shipped cell in 50/50 folds; held-out cost 2.906 = fixed SHIP3 2.906.
The two never-tuned listener parameters do not matter below LO 0.6 (the rescue filters ONCE / F8 / K-V4 decide); the
shipped values are kept. No TEST read.

## Round 26 — Step-Audio-2-mini as an ear (candidate-level screen; written before any of its answers exist)
Step-Audio-2-mini (StepFun, official weights; a different audio encoder lineage from Qwen, AF-Next, Kimi and Dasheng) answers
the V4 prompt on the P2/PV cuts of DEV and the tagger DEV part (same cut, decoding and matcher as Qwen/AF; tag-style answers
normalised as for Kimi). **GO to an arm** only if, on the needed-class candidates of merged DEV, Step names at least as many as
Qwen V4 while accepting no more other-class candidates; the arm would then replace Qwen by Step in the TIER high tier
(base SHIP3, standard pass rule). Otherwise the screen is reported and closed.

### Round 26 result — Step-Audio-2-mini: NO GO (jobs 31594384 smoke, 31594407 full, 3.5 min on an H200)
It answers the V4 prompt with one caption per cut, not a list (mean 1.0 lines). On the merged-DEV P2/PV candidates it names 19
of 35 needed-class candidates (Qwen V4 24, AF 15) and 95 of 969 others (Qwen 107, AF 189): fewer needed than Qwen, so the
pre-set GO bar fails. Files: `benchmark/gold/dev{,2}_listener_step.json`, `listener_step.py`, env `~/venvs/stepaudio`.

## Round 27 — prompt ensemble for the Qwen ear (general; written before its answers exist)
The V4 inventory names the loudest sounds and drops quiet ones under them (Gasp, Footsteps under explosions). A second,
class-uniform prompt V4b: "List every distinct non-speech sound you hear in this recording, including quiet or background
sounds, one per line." (same model, cut, greedy decoding, 64 tokens, same matcher) on the P2/PV cuts. **Arm QE (base SHIP3):**
the Qwen leg of TIER accepts if V4 OR V4b names the family (AF leg unchanged). Standard pass rule (hits ≥ 24, wrong ≤ 41 +
2 × gain, cost < 2.901, no needed hit lost on either part).

### Round 27 result — QE (second Qwen prompt): no change
V4b answers (job 31594441, 62 min): accepts DEV 78/602, tagger DEV 52/402, TEST 145/778, tagger TEST 72/399. Arm SHIP3+QE
(jobs 31594896 / 31594897): merged DEV 24 hits / 41 wrong / 2.901 — the same pictures as the shipped version (the extra Qwen
accepts are all removed later by ONCE, F8 or K-V4). Not adopted.

## Round 28 — DASM clip veto with a listener keep (from the wrong-picture profile; written before its number)
Profile of the shipped version's merged-DEV pictures (`scratchpad wrong_feats`): DASM's same-family score near the picture
start is high for hits (median 0.63) and low for cross (0.38, q25 0.08) and phantom (0.23) pictures; BEATs, FlexSED and PANNs
do not separate them. **DV:** a non-rescued stage-4 span whose family DASM never reaches the round-6 calibrated clip-veto bar
v = 0.084 anywhere in the clip (the 280-clip calibration, same construction as the FlexSED veto 0.3) is dropped, unless
Qwen (F7's P1 rule) or Audio Flamingo V4 accepts it. Base SHIP3; pass rule hits ≥ 24, wrong ≤ 41 + 2 × gain, cost < 2.901,
no needed hit lost on either part.

### Round 28 DV — PASSES; new shipped best
DEV 49 (job 31595098): 18 / 18 (6 / 9 / 3) / 2.20 vs base 18 / 20 / 2.29. Merged DEV (job 31595099): **24 hits / 34 wrong
(9 / 22 / 3) / 2.704** vs base 24 / 41 / 2.901; no needed hit lost on either part (old DEV 18 / 18, tagger DEV 6 / 16).
Merged into `use_shipped()`; TEST read 5 prepared and reported below.
**DV (shipped TO1+F7F8 + N2b + DR2 + K-V4 + DV) on the merged TEST (fifth read, reported; `final_test_dv.{json,md}`):**
23 hits / 34 wrong (3 / 25 / 6) / **2.682** vs B0r 21 / 40 / 2.909, Δ −0.227 [−0.500, +0.023], p 0.051 → "same". Lowest TEST
cost of all reads; versus read 4 (K-V4) −2 wrong, same hits: DV's DEV gain shows on TEST too.

## Round 29 — the weak-sound misses traced (Adam: "inspect the 17 misses that are weak and group them to causes")
All 26 heard-but-missed needed sounds of the shipped version on merged DEV (`scratchpad weak_trace`), grouped with Fable:
gate-visible 5; both listeners no 6; listeners yes but a filter removed it (F8 DASM vote or ONCE) 5 (Hammer 13.7, Explosion 2.8,
tg_d107 Crying, tg_d125 Clapping, tg_d120 Meow/Cat 2.94 — ONCE dropped it after the Cat 0.56 picture); one listener only 2;
never asked (below LO / covered) 3; timing 2; a veto removed a strong span 2; unclear 1 (tg_d030 Motorcycle).
- **F8-U (written before its number; base = shipped SHIP4 = SHIP3+DV):** F8 passes a rescued span if DASM ≥ 0.575 OR both
  Qwen V4 and Audio Flamingo V4 name its family (K1's bypass, round 14), together with ONCE-G (a later rescue of a family
  counts as new if it starts > 2.0 s after the kept one, round 22). Each part alone failed on older bases (K1 +6 wrong, ONCE-G
  +3 wrong); the pair was never run. Pass rule: hits ≥ 24, wrong ≤ 34 + 2 × gain, cost < 2.704, no needed hit lost on either part.
- **DV-L (Fable; written before its number; base SHIP4):** DV with a span-local window: a non-rescued stage-4 span is dropped
  if its family's DASM never reaches v = 0.084 within [start − 0.5, end + 0.5] s, unless Qwen (P1 rule) or AF V4 names it.
- **DV-G (same; base SHIP4):** a non-rescued span is dropped if its family's DASM does not reach g = 0.575 within the span
  ± 0.5 s, unless BOTH Qwen V4 (P1 open inventory) and AF V4 name the family on its cut. Both bars are the round-6 280-clip
  calibration. Same pass rule as F8-U (hits ≥ 24, wrong ≤ 34 + 2 × gain, cost < 2.704, no needed hit lost on either part).
- **V4D (Fable; candidate-level screen, written before any of its answers):** on each P2/PV cut, 20-ms frames with RMS above
  the cut's 85th percentile are attenuated by 20 dB (20-ms fades) — the loud co-occurring event is ducked — and the unchanged
  V4 prompt is asked again (Qwen3-Omni, greedy, 64 tokens, same matcher). GO to an arm iff, on merged-DEV candidates, V4D adds
  ≥ 2 needed-class accepts beyond Qwen V4 with ≤ 2× that many other-class accepts added. The arm would then let the Qwen leg
  of TIER accept on V4 OR V4D.
- **GA (gate audio-identity veto; Fable; written before any number):** for each cached DEV gate stretch whose majority says
  "seen" and whose name vote named a visible thing ("macaws", "church bell", "boxer dog"), Qwen3-Omni hears the stretch audio
  (stretch ± 1 s, NO frames) and answers an a/b in both orderings: "(a) this is the sound of {named} (b) this is a different
  {family} sound or something else". A firm "b" in both orderings vetoes the "seen" vote for that stretch. Screen on the DEV
  judge clips (43 seen, 36 needed; shipped majority 16 / 31). GO to a full arm iff ≥ 3 more needed kept with ≤ 1 fewer seen
  silenced. Otherwise report-only.
- **F8-U result (old DEV 49, proposed):** SHIP4+F8U 20/36 hits, 25 wrong, cost 2.327 vs SHIP4 (=SHIP3+DV) 18/36, 18, 2.204:
  +2 hits for +7 wrong (rule allows +4), cost up. Merged DEV: 26/55, 44 wrong, 2.873 vs 24/55, 34, 2.704. **Fail.**
- **GA screen result (DEV judge clips):** majority 16/43 seen silenced, 31/36 needed kept; GA 9/43 and 32/36. +1 needed kept
  (bell_miami Bell) for 7 fewer seen silenced (the listener says "not the named thing" on real visible sounds: Water, Vehicle,
  Laughter, Machine gun). **STOP.** The listener cannot tell a named visible source from another of the same family.
- **V4D screen result (merged DEV P2/PV, 1004 candidates):** V4D adds 2 needed-class accepts beyond Qwen V4 (nyc_1689 Air
  horn 3.8, carnival Whistle 6.3) and 53 other-class accepts. Bar ≤ 2× → **STOP.** Ducking makes Qwen say yes much more
  often, mostly to wrong or already-drawn sounds.
- **DV-L / DV-G results (merged DEV):** DV-L = SHIP4 exactly (24/55, 34, 2.704; no span changed — the clip veto already
  covers it). DV-G 19/55, 21 wrong (8/12/1), cost 2.620: −13 wrong but −5 hits. **Fail** (hits drop). Noted as a
  fewer-pictures option only.

### Round 30 (Fable, written before any arm number)
- **K4A (`KEEP_NEEDS_V4_ALL`):** K-V4's test on every drawn non-rescued span (conf ≥ display bar): the span is dropped
  when both open-inventory listeners were asked on its P1 cut (`RELABEL_P1V4` item with qwen_fams and af_fams) and neither
  names its family; unasked spans are kept. "exact" = family match (K-V4's own); "onto" = a named family that is a kind of
  the span's family also keeps it (e.g. Aircraft keeps Vehicle). Pre-registered primary: **onto** (K4AO); exact reported.
  Fable's candidate count (seen before the arm, DEV P1 items with gold): 0 of 15 hit_needed items fail the test, 16 non-hit
  items do. Pass rule vs SHIP4 (24/55, 34, 2.704): hits ≥ 24, no needed hit lost on either part, wrong ≤ 34, cost < 2.704.
- **BTP (band-twin pull; screen on saved SHIP4 pictures, CPU):** a drawn non-rescued picture whose family has a FlexSED run
  (score ≥ 0.5, `LISTEN_RUN_BAR`) ending ≤ 1.0 s before the picture's start and starting ≤ 1.5 s before it has its start
  pulled to that run's start (the latest such run). Rescore with score_per_sound on merged DEV. GO to an arm iff 0 current
  hits lost, hits ≥ 25, wrong ≤ 33.
- **PIC-SIM (gate picture-vs-frames similarity; screen on gate_gold DEV judge set):** SigLIP-2 image-image cosine between the
  blind_a2i picture and the gate's stretch frames (max over frames). One pre-set pair: s_hi = 0.80 flips "not seen"→seen,
  s_lo = 0.30 flips seen→not seen. GO iff seen silenced ≥ 20/43 and needed kept ≥ 31/36.
- **BTP screen result (saved SHIP4 pictures, bar 0.5 as pre-registered):** merged 25/55, 33 wrong (9/21/3), cost 2.620; 0 hits
  lost; one outcome change (tg_d107 Crying 1.25→0.08, cross→hit); mv_detective Alarm and tg_d107 Screaming moved, outcomes
  unchanged. **GO.** (src LISTEN_RUN_BAR is 0.4; at 0.4 same totals — reported only.) Arm SHIP4+BTP (`BAND_TWIN_PULL` 0.5,
  in stage 4 after the vetoes, non-rescued spans) runs on merged DEV; same pass rule vs SHIP4.
- **PIC-SIM result:** coverage 34/79 sounds (the rest keep the majority vote). SigLIP-2 cosine seen median 0.518, needed
  0.448, ranges overlap; none ≥ 0.80 or < 0.30, so nothing flips (16/43, 31/36). **STOP.** Exploratory grid: best s_hi 0.60
  gives 18/43, 31/36 — still short.

### Rule change (Adam, 30 Sept 16:02; wording by Fable; written before any arm is judged under it)
Adam: "its ok to take a bit fewer hits if we also take much less wrongs. then we can think how to improve the hits only".
From now on a candidate passes against its base iff EITHER the old rule holds (hits ≥ base, wrong ≤ base + 2×gain, cost <
base, no needed hit lost on either part) OR the **fewer-pictures clause** holds on merged DEV: (a) cost < base; (c) wrong ≤
base wrong − 3 × hits lost (3:1 is a fixed 50% margin over the cost's 2:1 break-even, not fitted); (d) hits ≥ base hits − 3.
The paired bootstrap CI is reported, not required (as for every earlier ship). Disclosure: DV-G (19/21/2.620, 2.6:1, 5 lost)
motivated the change and fails (c) and (d); it is re-run only on a new base and judged as any other arm. Order: BTP, K4A,
N2c on the new base, DV-G on the new base, then new mechanisms (re-time instead of drop first).
- **K4A result (merged DEV):** exact and onto identical: 23/55, 31 wrong (9/19/3), cost 2.676 vs SHIP4 24/34/2.704 (old DEV
  17/18 vs 18/18: one needed hit lost). Fails the old rule; passes the fewer-pictures clause vs SHIP4 (−1 hit, −3 wrong,
  3:1, cost down). Re-judged on the new base (SHIP5+K4AO).
- **BTP arm result (merged DEV):** SHIP4+BTP 25/55, 33 wrong (9/21/3), cost 2.620; old DEV 18/18 unchanged, tagger 7/15 vs
  6/16; no needed hit lost. **Passes the old rule → shipped as SHIP5** (config.use_shipped BAND_TWIN_PULL 0.5). TEST read
  follows (reported). Next on SHIP5: K4AO, N2c, DV-G under the combined rule.
- **RT (re-time instead of drop; GO bar pre-registered: 0 hits lost, cross −3, wrong not up):** on saved SHIP5 pictures, 30
  drawn non-rescued pictures meet the DV-G condition; none has a family DASM peak ≥ 0.575 within ±3 s (max 0.516). No picture
  moves. **STOP** — these pictures are not mistimed; DASM does not hear the family nearby at all.

### Round 31 RPT-S — repeat pictures need a FlexSED silence (Fable, filter group; written BEFORE any number of it)
Base = shipped SHIP5 (SHIP4+BTP): merged DEV 25/55, 33 wrong (9/21/3), cost 2.620. Regrouping of the filter group after
reading the SHIP5 stage-4 trace (`stage4.json` r14_dropped / listener): tg_d107 Crying is a HIT under SHIP5 (BTP pulled it
to 0.38); tg_d125 Clapping was asked (a_asked 7) and refused by the TIER listener rule, not a filter; so the filter group is
3: nyc_1689 Hammer 13.76 (F8, DASM 0.281), as_explosion Explosion 2.84 (F8, 0.516), tg_d120 Meow/Cat 2.94 (ONCE, "not the
first" after Cat 0.56). The "timing" and "veto" cases (birds_forest Bird 2.22 conf 0.22, tg_d032 Thunder 8.0 conf 0.27,
tg_d095 Dishes 16.5 conf 0.37, ly_ambulance Car 8.25 conf 0.32) are all below PICTURE_MIN_CONF 0.40 and would not be drawn
with every veto off (not screenable on saved pictures: no image exists for them); ly_applause Crowd starts 1.9 s early
(0.0 vs 1.9). Display layer: score_per_sound counts a later same-family picture as a free "dup" only if its start falls
in the onset window of an already-matched gold; a later piece of one long sound (onset far back) is a CROSS, so a texture
split into pieces costs one wrong per piece. Disclosure (seen before writing, as round 28's profile): in the SHIP5
pictures 8 wrong pictures are later same-family repeats (barbershop Shaver 16.0; detective Alarm 3.36, 9.08; tg_d020 Rain
7.0, 12.75; tg_d088 Thunder 8.5, 13.25; as_explosion Gunshot 8.25) and 3 hits are (mv_protest Glass 10.75, 16.89;
as_explosion Explosion 9.25); a blanket one-picture-per-family rule would give 22/25/2.563 and fail clause (c) by one.
- **Rule RPT-S:** a placed picture whose family (canonical label) already has an earlier placed picture in the clip is
  dropped unless the family's FlexSED score (max over its queries per frame) is < 0.5 for a contiguous stretch >= 1.0 s
  somewhere inside [previous same-family picture START, this picture's start] (the previous start, not its display end,
  which dwell / MAX_AFTER_END stretch). No FlexSED query for the family -> kept (no evidence of continuity). Rescued and
  non-rescued alike. Constants reused, not fitted: 0.5 = the BTP / LISTEN_RUN bar, 1.0 s = MAX_AFTER_END / the hit
  window's late tolerance. Secondary (reported only): the silence must sit in [start - 1.0, start).
- Screen: `benchmark/gold/rpt_screen.py` on the saved SHIP4+BTP pictures of merged DEV (CPU, FlexSED cache), rescored
  with score_per_sound. GO iff the combined rule vs SHIP5 holds: old rule (hits >= 25, wrong <= 33 + 2 x gain, cost <
  2.620, no needed hit lost on either part) OR fewer-pictures clause (cost < 2.620, wrong <= 33 - 3 x hits lost, <= 3 hits
  lost). If GO the arm lives in stage 4 after BTP (it needs ffw), flag `REPEAT_NEEDS_SILENCE` = 0.5.
- Second idea if STOP: **ONCE-S** — the same silence test on the ONCE-dropped rescues (r14_dropped ONCE lists + FlexSED);
  survivors added as (label, start, start + 1.5) to the base pictures and rescored; same GO bar.
- **RPT-S screen result (saved SHIP5 pictures, merged DEV; `benchmark/gold/rpt_screen_bar0.5_sil1.0.json`):** primary
  ("between") 25/55, 31 wrong (9/19/3), cost 2.563 vs 25/33/2.620; 0 hits lost; old DEV 18/18 unchanged, tagger 7/13 vs
  7/15. Only tg_d020 Rain 7.0 and 12.75 are dropped (Rain never falls below 0.5 in FlexSED); every other repeat (Shaver,
  Alarm rings, Thunder, Gunshot, Glass, Explosion) has a >= 1 s FlexSED silence before it and stays. **Passes the old rule
  -> GO** to an arm (SHIP5+RPTS, `REPEAT_NEEDS_SILENCE` 0.5 in stage 4 after BTP; same pass rule vs SHIP5). Secondary
  ("before", reported only): 24/29/2.563, loses as_explosion Explosion 9.25 (silence 0.84 s before it) — fewer-pictures
  clause only; not adopted. ONCE-S not run (RPT-S is GO).
  Clarifications (no number changes): "previous same-family picture" = the family's FIRST kept picture in the clip (a
  dropped repeat never becomes the previous one; `rpt_screen.apply`). The screen sees placed, gate-kept, image-bearing
  display spans; the arm sees stage-4 spans. Arm definition, written before it exists: "earlier picture" = a same-family
  span with conf >= the display bar that survives stage 4 after BTP; the gate (which can silence the earlier span) is a
  known divergence, reported if arm != screen. Files uncommitted: this entry, `benchmark/gold/rpt_screen.py`,
  `benchmark/gold/rpt_screen_bar0.5_sil1.0.json`.

### Round 31 CONT — continuation veto (Fable, cross group; written 2026-09-30 BEFORE any number of it)
**Group listing (SHIP5 = saved SHIP4+BTP pictures, merged DEV, `benchmark/gold/cross_group.py` → `cross_group.json`; 21
cross pictures, base reproduced 25/55, 33 (9/21/3), 2.620).** Groups: **L late repeat, 9** — the family's sound was
already going on and its onset was shown or missed earlier (barbershop Shaver 16.0 [+15.9 s], detective Alarm 3.36 and
9.08 [Telephone 0.0], tg_d020 Rain 2.0 / 7.0 / 12.75 [Rain 0.0 V/O], tg_d030 Vehicle 13.5 [Motorcycle 7.7 V/O], tg_d088
Thunder 13.25 [+2.15], golf Bird 18.84 rescued [Bird 0.0]); **E early, 4** (explosion Gunshot 8.25 [Machine gun 10.1,
−1.85 s], applause Crowd 0.0 [−1.9, the known chain], protest Glass 4.75 [Glass 11.0; a Baby cry is there], tg_d088 Thunder
8.5 [−2.6]); **X wrong family, 8** — a picture of another family at the moment of a visible/obvious sound (crossing_bells
Steam on the Train, blueplanet Laughter on a Quack, tg_d022 Pant and Dog on Chopping, tg_d088 Explosion on Thunder, tg_d107
Screaming on a bird, tg_d128 Hammer on Clang; tg_d029 Goose on a needed Chicken). Stage: 17 tagger-origin BEATs spans, 2
FlexSED-only (detective Alarm ×2), 2 rescued (golf Bird, blueplanet Laughter; both listeners + DASM say yes). Listeners on
the P1 cut: in L the family is named by Qwen or AF on 8/9 (they hear the sound — it IS there, only late); in X neither
names the family on 3/8 (Pant, Goose, and Dog/Screaming/Hammer/Explosion/Steam are named by at least one). K4A (other
thread) covers X's "nobody names it" cases; the gate-side X cases are amendment M/N territory (closed).
**Rule CONT (`CONTINUATION_VETO`, general, stage 4 after BTP on every drawn span, rescued included):** a picture is a
continuation, not an onset, when its family's FlexSED evidence was already up before it starts: a FlexSED run of the
family (frame score ≥ 0.5 = the BTP / LISTENER_LO bar, gaps ≤ LISTEN_RUN_GAP merged, the `_runs` construction) that starts
≥ 1.5 s before the picture's start (1.5 s = BTP's start window / MERGE_GAP) and reaches it (run end ≥ picture start) drops
the picture. A picture starting in the clip's first 1.5 s is never dropped (no "before" exists). No other constant.
**Screen:** CPU, saved SHIP4+BTP pictures of merged DEV (`benchmark/gold/cont_screen.py`), rescored with score_per_sound.
**GO bar = the combined pass rule vs SHIP5 (25/55, 33, 2.620):** old rule (hits ≥ 25, wrong ≤ 33 + 2 × gain, cost <
2.620, no needed hit lost on either part) OR the fewer-pictures clause (cost < 2.620, wrong ≤ 33 − 3 × hits lost, hits ≥ 22).
Reported only, not judged: the same rule with the BEATs family column in place of FlexSED, and with 1.0 s in place of 1.5 s.
- **CONT screen result (saved SHIP5 pictures, merged DEV, `benchmark/gold/cont_screen.json`):** 24/55 hits, 28 wrong
  (9/17/2), cost 2.535 vs SHIP5 25/33/2.620. Dropped 6 pictures: 4 cross (tg_d020 Rain 2.0 / 7.0 / 12.75, tg_d030 Vehicle
  13.5 — all group L), 1 phantom (mv_protest Siren 22.75), 1 hit (b3_favela_rio Train 14.72: FlexSED hears the train ≥ 0.5
  from > 1.5 s before the annotated onset). Old DEV 17/17/2.245 (one needed hit lost), tagger DEV 7/11/3.182. Old rule
  fails (a hit lost); **fewer-pictures clause passes** (cost down, 5 wrong per hit lost ≥ 3:1, 1 ≤ 3 lost) → **GO** to
  the arm SHIP5+CONT (flag `CONTINUATION_VETO` 0.5 / 1.5 s in stage 4 after BTP), judged under the combined rule vs SHIP5.
  Reported only: BEATs column instead of FlexSED changes nothing (BEATs spans are the pictures' own starts); WIN 1.0 s
  drops 8 (2 hits: + tg_d149 Bee 1.25), 23/27/2.563 — worse than 1.5 s on hits, not adopted. Not caught by CONT: the
  L cases whose family FlexSED is not continuously ≥ 0.5 before the start (barbershop Shaver, detective Alarm ×2, tg_d088
  Thunder 13.25, golf Bird) and every E / X case.
- **RPT-S screen result (saved SHIP5 pictures):** 25/55, 31 wrong (9/19/3), cost 2.563; 0 hits lost; GO → arm SHIP5+RPTS
  (`REPEAT_NEEDS_SILENCE` 0.5, stage 4 after BTP).
- **PMC (written before its number):** 4 heard misses sit just under the picture floor PICTURE_MIN_CONF 0.40 (0.22–0.37).
  One value, not a sweep: floor = DISPLAY_THRESHOLD 0.35 (the pipeline's own display bar). Arm SHIP5+PMC; combined rule.

## Round 31 PTC — peak-tight cut for both ears (Fable, hit group; written 2026-09-30 BEFORE any PTC answer exists)
**Group (HIT side, 11 heard misses of SHIP5 on merged DEV, `scratchpad weak_trace`).** Both listeners no (6): as_explosion
Footsteps 2.1 (8 dB under the explosions; Qwen V12 yes, Kimi yes), Gasp 6.7 (0.2 s long; V12 yes), carnival Whistle 6.1
(under music; Qwen hears "train wheels squealing", AF "squeal"; V12 yes), tg_d032 Thunder 2.8 and 7.4 (all five ears say
wind / ocean waves; DASM 0.01 — treated as closed), tg_d033 Siren 0.0 (10-s Alarm run; Qwen says "Breathing, Footsteps" on
the 0–11 s cut but "Siren" first on the 0–2.8 s Printer cut of the same clip). Never asked (3): rainforest_2179 Bird 6.5
(FlexSED 0.44 < LO 0.5, −40 dB), storm Civil defense siren 16.9 (FlexSED 0.42 < LO, 4 dB under screaming), tg_d095 Dishes 16.6
(FlexSED 0.31, no P2 run). One listener (2): nyc Air horn 3.8 and tg_d107 Laughter 8.4 — both peak ≥ 0.6 (Qwen-only tier),
AF V4 yes ("car horn", "laughing"), Qwen V4 no ("Door closing", "Squeak"). Newer ear not reachable this round: every cached
audio-LLM lineage has been asked, Gemma-4-31B-it has `audio_config: null`, home disk 27 GB free.
**Mechanism.** Qwen V4 names 2–3 sounds and stops (the rest of the 64 tokens is post-EOS noise: "wise wise", "Assistant");
on a long cut the loudest 2–3 sounds crowd the candidate out. The listener cut is [run − 1, run + 1] s; runs < 1 s already get a
1-s window centred on the peak frame (3 s of audio), runs ≥ 1 s get the whole run. Merged-DEV P2/PV candidates with a cut
longer than 4 s: DEV 162 / 602, tagger DEV 100 / 402 (counts only; no class split seen).
**Rule PTC.** For every P2/PV candidate whose listener cut is longer than 4 s, both ears — Qwen3-Omni V4 and Audio Flamingo
Next V4, same prompt, greedy decoding, 64 tokens, same matcher — are asked again on a 3-s window centred on the family's
FlexSED peak frame inside [start, end] (`data/work/flexsed_cache/<clip>.npz`, the item's own query column; PV items: the
vetoed span's own start/end; no column → the span midpoint; clipped to the clip, shifted to keep 3 s). Each ear's flag
becomes original OR tight (no accept can be lost). Arm (only on GO): SHIP5 with TIER on the OR'ed flags.
**Screen (candidate level, merged DEV, as `v4d_screen.py`; `benchmark/gold/ptc_screen.py`).** added = `_tier(OR flags,
peak)` true and `_tier(original flags, peak)` false, class by `gold_class` (hit_needed vs other). **GO iff needed-class added
≥ 2 AND other-class added ≤ 2 × needed-class added.** Reported beside, not decisive: per-leg additions (Qwen-only, AF-only);
of the needed-class additions, how many have DASM ≥ 0.575 within ± 0.5 s (F8's ceiling at arm level). If STOP, a second
idea is written here before its number. Files: `listener_ptc.py`, `slurm/job_ptc.sh`, `dev{,2}_listener_ptc.json`.
  Faithful stage-4 reading (every burst of a spec tested, surviving bursts re-placed through `_display_spans`; `CONT-rows`
  in `cont_screen.json`): identical, 24/55, 28 (9/17/2), 2.535, same hit lost. The screen reads the raw FlexSED cache
  columns (no per-family rescale), the same shortcut the BTP screen used, which its arm then matched exactly.

## Round 31 SUBJ (Fable, gate group; written 2026-09-30 BEFORE any number)
Trace of the shipped SHIP5 pictures on merged DEV (`benchmark/gold/gate_group.py`, `gate_group.json`, frames = the gate's
own six per stretch): 9 visible wrong pictures = 5 with every vote "no" and "nothing" named on every stretch (storm ×2
Thunder, as_church_bell Bell, london_protest Vehicle for the air horn, un_driving Explosion for the starter's bang), 3 one-yes
splits on a plainly active source (fire-alarm pull, elephant splashing Water, kids' Laughter), 1 mixed across stretches
(tg_d088 Thunder: "lightning" seen on one stretch only). 6 gate-silenced needed sounds: macaws, robin, church tower, boxer
dog ×2 (Fart 0.0 and 5.6) — a real same-family thing on screen, unanimous or 2-of-3 "seen", the annotator's "not this one"
inaudible in stills (N and GA already failed there); ly_ambulance Vehicle is unreachable (span starts at 0.0, onset 7.3).
The miss side is at the frame ceiling; the screen goes to the wrong side, where the sound IS the scene (rainstorm,
castle, protest, bath) and the per-stretch gate, asked for an object, finds none.
- **Rule SUBJ (scene-subject silence).** A drawn spec whose every gate stretch said "not seen" is silenced iff the clip's
  scene sentence (the pipeline's own `SCENE_PROMPT` answer, printed as `[stage5] scene:` and computed on the spot for a new
  video) is about the sound: `reason._about_the_sound(scene, event_label)` (word overlap, SUBJ-W) OR the gate VLM
  (Qwen3.8-27B, greedy, text only) answers "yes" to `MAKES_SOUND_PROMPT` with the scene sentence as the thing visible
  (SUBJ). Applies to every drawn picture, rescued or not; no label lists. Report-only variant SUBJ-50: also requires the
  picture to cover ≥ 50 % of the clip.
- **Screen.** Saved SHIP4+BTP (= SHIP5) pictures of merged DEV, scene sentences from the SHIP5 job logs (55 clips with
  pictures, `scene_sentences_ship5.json`), rescored with `score_per_sound` as the BTP screen (`benchmark/gold/subj_screen.py`).
  Base 25 / 33 / 2.620. **GO** to a full arm iff wrong ≤ 30 with 0 current hits lost (hits ≥ 25, cost < 2.620 follow).
  Also reported: the judge-set view (gate_gold DEV clips: seen silenced / 43, needed kept / 36) is not available for a
  whole-clip rule without the scene sentences of the non-picture clips — reported only where a sentence exists.
- If STOP, a second idea is written here before its number.
- **SUBJ result (word path, CPU; `subj_screen.json`):** SUBJ-W drops 3 pictures — as_fire_alarm Alarm 8.89 (visible wrong),
  tg_d127 Water 0.14 (visible wrong) and tg_d120 Cat 0.56, a HIT ("a red alarm clock with a cat picture on the face"):
  24 / 31 / 2.620 vs 25 / 33 / 2.620. SUBJ ⊇ SUBJ-W, so the hit is lost whatever the VLM answers → **STOP** (the VLM
  answers, job 31596035, are reported when they land; SUBJ-50 25 / 32 / 2.592 on the word path, report only). The scene
  sentence names the family of a sound the viewer cannot assume — the "obvious" trade-off again, one level up.

## Round 31 OM — onset-motion tiebreak (Fable, gate group; second idea, written BEFORE any number)
Three of the nine visible wrong pictures are one-yes splits on a plainly active source (fire-alarm pull 8.89, splashing Water
0.14, kids' Laughter 3.08). Honest reach: at most these three; Water starts at the clip start (no onset to see).
- **Rule OM.** A gate stretch with exactly one "yes" among name / a-b / description is re-decided as "seen" iff the video
  itself changes at the stretch's onset: mean |ΔI| between grey frames 0.25 s apart (160 px wide), averaged over
  [onset − 0.5, onset + 0.5] s, ≥ K = 3 × the clip's median frame-to-frame change (same spacing, whole clip). K fixed now.
  The clip verdict stays "silent only if every stretch is seen". No VLM call, no label lists; on the spot for a new video.
- **Screen.** `benchmark/gold/onset_motion_screen.py` on the saved SHIP5 pictures of merged DEV (CPU). **GO** to a full arm
  iff 0 current hits lost and wrong ≤ 31 (≥ 2 of the 3 reachable pictures gone, cost < 2.620). Otherwise report-only, and
  the gate group is closed for this round: the miss side is at the frame ceiling, the wrong side is 4 "obvious" scene
  sounds (no object to see) + 3 splits + 1 label mismatch + 1 annotation-level bell.
- **OM result (`onset_motion_screen.json`, K = 3):** 16 pictures have a one-yes stretch; only the fire-alarm's second
  stretch (14.32–19.75, ratio 4.8 — a camera whip-pan) flips to seen, and its first stretch (8.89–14.32, zero yes) keeps the
  picture; the Water and Laughter splits sit at ratio 0.8–1.0 (a splash or a laugh does not move the frame). 25 / 33 / 2.620 =
  base, nothing dropped → **STOP**. Gate group closed for round 31 (both ideas report-only, no arm run, no TEST read).

## Round 31 SPOT (SOTA scan; written 2026-09-30 BEFORE any SpotSound answer exists)
**Model.** SpotSound-A (Sun et al., arXiv 2604.13023, Apr 2026): LoRA adapter `Loie/SpotSound` (r 8, alpha 16) merged into
`nvidia/audio-flamingo-3-hf`, bf16, greedy, 64 new tokens. Timestamp-interleaved processor vendored from the SpotSound repo
(`processor/af3.py`, "timestamp: t seconds; feature:" + 25 audio tokens per second). Trained to answer "No." for absent queries.
**Questions (the repo's own wording, + " Answer: " as `inference.py`).** EXIST: "This is a sequence of audio stream. Your task
is to identify whether the sound event in the query occurs. The query is: {q}." GROUND: "... identify the temporal window
(start and end timestamps) when the given query appears. The query is: {q}." q = `src.labels.canonical(label)` lowercased;
one rule, no per-family phrasing. Audio = the exact cut the listeners heard: P2/PV `run_audio` (≥ 1 s, as `listener_v4d.py`),
P1 [cut_start − 1, cut_end + 1] clipped, ≥ 1 s (as `dev_listener.py`).
**SPOT-yes (decisive).** EXIST not starting with "no" AND GROUND returns ≥ 1 parsed interval (regex "From a (seconds) to b"
or "(a, b)"; times are cut-relative, shifted by the cut start) overlapping the candidate run [start, end] widened by ± 0.5 s
(the model's 1-s grid). Reported beside, not decisive: SPOT-any (EXIST yes, any interval), GROUND-only overlap, and a null
control (q = item `null_family` on the same cut).
**Data.** Merged DEV P2/PV candidates: `dev_listener_v.json` (~/MscProj_r13, 602) + tagger `dev2_listener_v.json`
(~/MscProj_tg copy, 402; the local copy is an older 92-item version). "Shipped accepts" = Qwen `accept.V4` (as `v4d_screen.py`).
Class by `dev_listener.gold_class` on gold_AG / tagger_AG; candidates dropped for "clip not in gold" are counted and printed.
**Screen (`benchmark/gold/spotsound_screen.py`, `spotsound_screen.json`).** (a) new accepts = SPOT-yes AND V4 false:
**GO iff needed added ≥ 2 AND other added ≤ 1 × needed added.** (b) veto on V4 accepts = V4 true AND SPOT-no:
**GO iff other removed ≥ 3 × needed lost** (and ≥ 1 other removed). (c) SPOT alone vs V4 alone: needed / other accept counts.
Report-only: P1 items of `dev_listener.json` (its `gold` field): SPOT-yes rate on hit_needed vs none. If both STOP, SpotSound
is closed for the listener role; no threshold or prompt is tuned after the numbers.

## Round 31 FLAP (SOTA scan; written 2026-09-30 BEFORE any FineLAP number on project audio)
**Model.** FineLAP (Li et al., ACL 2026; github xiquan-li/FineLAP, HF `AndreasXi/FineLAP` rev b419aa2, MIT, 0.2 B; EAT audio
encoder + RoBERTa text, contrastive dense audio-text). `get_frame_level_score` = sigmoid(cos / 0.1 − 10) per 160-ms frame per
text query. Own env `~/venv_flap` (msproj + transformers 4.51.3; the remote code breaks on transformers 5), HF offline.
Only smoke-tested on the repo's own `resources/1.wav` before this entry.
**Windowing (FineLAP cuts audio at 10.24 s).** The listener wav (`devcand/wav16` DEV, `r13dev2/wav16` tagger DEV) is cut into
10.24-s windows at hop 5.12 s, the last window aligned to the clip end (clip ≤ 10.24 s: one window, zero-padded as FineLAP does);
the fbank block of `load_audio` is applied to each slice in memory. Frame i of a window at w0 covers [w0 + 0.16 i, w0 + 0.16 (i+1));
the score of a time frame is the max over windows covering it (padded frames are dropped).
**Score of an item.** Query = `canonical(label)`. Score = max over frames overlapping [start, end] (PV: the vetoed span's own
start/end); if no frame overlaps, the frame holding the midpoint.
**Calibration (fixed rule, one bar).** P1 items (the drawn BEATs spans; not rescue candidates) of `dev_listener.json`
(~/MscProj_r13) + `dev2_listener.json` (~/MscProj_tg), pooled, kept iff `dev_listener.gold_class == "hit_needed"`. Bar = the
score at index floor(0.10 n) of the ascending list (FineLAP ≥ bar keeps ≥ 90 %). Reported beside: n, per-split bars.
**Candidates.** Every P2 / PV item of `dev_listener_v.json` (r13) and `dev2_listener_v.json` (tg) with gold, AF answers from
`*_listener_afn.json` (same key as `v4d_screen.py`). Shipped accept = candidate-level `_tier({"V4": Qwen V4, "AF_V4": AF V4},
peak)` (TIER_SPLIT 0.6, TIER_HIGH_OR off) — the same proxy as the V4D / PTC screens, not full stage 4 (ONCE, F8, veto order).
Class by `gold_class`: needed = hit_needed, other = none + other_gold.
- **(a) new accept path:** TIER false AND FineLAP ≥ bar AND (Qwen V4 OR AF V4). **GO iff needed added ≥ 2 AND other added
  ≤ 1 × needed added.**
- **(b) veto:** TIER true AND FineLAP < bar. **GO iff other removed ≥ 3 × needed lost** (and other removed ≥ 1).
Each item is listed. Script `benchmark/gold/finelap_screen.py` (`run` on GPU, `score` on CPU), `slurm/job_finelap.sh`, frame
cache `data/work/finelap_cache/<clip>.npz`, result `benchmark/gold/finelap_screen.json`. A GO arm is written here before its run.
- **Results on SHIP5 (25/55, 33, 2.620), combined rule:** SHIP5+K4AO 24/55, 30 (9/18/3), 2.592 — −1 hit, −3 wrong, cost
  down → **passes the fewer-pictures clause** (kept pending the RPT-S / CONT / PMC results, then stacked in order: old-rule
  passes first). SHIP5+N2c 24/31/2.620 — cost not lower, fail. SHIP5+DVG 20/20/2.535 — 5 hits lost (> 3), fail.
### Round 31 PTC result — STOP (job 31596021, 9 min on an H200; `benchmark/gold/ptc_screen.json`)
262 long-cut candidates (5 needed-class). Tight windows: Qwen adds 22 + 19 accepts, AF 39 + 16. Under TIER (original OR
tight): **0 needed-class added, 17 other-class added** (Crowd murmur, Rain, Siren, Train, Cricket … on cuts of wall sounds
already drawn or visible). The Siren 0.0 Alarm run: the Alarm query peaks at 7.6 s, so the tight window [6.12, 9.12] misses the siren at 0 s
(Qwen "Sigh", AF "breathing"); tg_d032 Thunder 6.2 is named ("Rain,
Thunder") but its run start is outside the 7.4 onset window, and its DASM is 0.01 anyway. The mechanism holds (the tight
cut makes both ears name more) but what they name is the wall sound, not the masked one. Not adopted.

## Round 31 H2 — high-tier second opinion (Fable, hit group, second idea; written BEFORE any number of it)
Facts seen: the two "one listener only" misses (nyc Air horn 3.76, tg_d107 Laughter 8.40) sit in the Qwen-only tier (peak ≥
0.6) with AF V4 yes; Kimi V4 also names Laughter ("Laughter,Human_voice"), not the horn ("Bus, Motor vehicle"). K3 (AF alone at
the high tier, round 14) changed no picture on the old DEV, and its tagger-DEV arm ran before the tagger AF answers existed
(08:09) — never really tested there. DASM own-family within ± 0.5 s: Air horn 0.05 (F8 removes it whatever the ears say),
Laughter 0.89. **Rule H2:** at peak ≥ TIER_SPLIT the tier accepts on Qwen V4 OR (AF V4 AND Kimi V4) — two independent ears
agreeing while Qwen is silent; below the split unchanged (Qwen AND AF). Screen (`benchmark/gold/h2_screen.py`, CPU, existing
caches): high-tier merged-DEV P2/PV candidates newly accepted, needed-class vs other-class by `gold_class`. **GO iff needed
added ≥ 2 AND other added ≤ 2 × needed.** Beside: AF-alone additions (K3) and the DASM ≥ 0.575 count among needed additions.
### Round 31 H2 result — STOP (`benchmark/gold/h2_screen.json`, CPU)
551 high-tier merged-DEV candidates. AF alone (K3) adds 3 needed / 85 other. AF AND Kimi adds **2 needed (both the tg_d107
Laughter at 8.40 / 8.80 — one sound) / 21 other** (Bird ×7, Crowd ×6, Laughter/Giggle of visible people, Alarm, Rain,
Splash, Chewing): 10.5 : 1, bar ≤ 2 : 1. Only one of the two needed has DASM ≥ 0.575 (F8). Not adopted.
**Hit-group reading.** Of the 11 heard misses: Thunder 2.8 is named by no ear on any cut; Thunder 7.4 IS named by Qwen on the tight window
("Rain, Thunder") but its run starts at 6.2, outside the onset window, and own-family DASM is 0.01, so timing + F8 block it, not the ears; Footsteps, Gasp,
Whistle, Siren, Air horn have own-family DASM 0.05–0.37, so F8 removes them even when an ear says yes (F8-U already cost
+7 wrong for +2 hits); the 3 never-asked are below LO 0.5 with no P2 run; Laughter 8.4 is the one candidate reachable
(AF + Kimi + DASM 0.89) but every general rule that admits it admits ≥ 10 other-class accepts. The group is closed for
general rules on this data.
- **FLAP screen result:** bar 0.329 (P1 hit_needed 54/59). (a) new accept path: +3 needed, +58 other → STOP. (b) veto on
  shipped-accepted rescue candidates: 5 needed lost, 30 other removed → **GO** (≥ 3×). Arm SHIP5+FLAP: a rescued span whose
  family's FineLAP max over the span < 0.329 is dropped (family not queried → kept). Combined rule vs SHIP5. Risk noted:
  FineLAP is weak on laugh families (4 of 5 losses).
  Smoke note (15 items, no gold read): GROUND answers "from 1.004s to 1.801s"; "from 0.000s to 0.000s" (the model's
  "none") and any b ≤ a count as no interval. Written before the full run's screen.
- **RPT-S arm (merged DEV):** 24/55, 31 (10/18/3), cost 2.620 — at stage 4 it removed an old-DEV hit the screen kept (the
  screen saw placed pictures only). Cost not lower → **fail**.
- **PMC arm:** identical to SHIP5 (25/33/2.620): the four sub-0.40 spans do not reach stage 5 pictures at 0.35 either. No-op.
- **TEST read of SHIP5 (reported, not selected on):** SHIP4+BTP 23 hits / 33 wrong (3/25/5) / 2.659 vs B0r 21/40/2.909;
  d −0.250 [−0.523, +0.000], one-sided p 0.033 → **better** (benchmark/gold/final_test_ship5.md).
- **CONT arm (merged DEV):** SHIP5+CONT 25/55, 28 wrong (9/17/2), cost 2.479 vs SHIP5 25/33/2.620; old DEV 17/17 (one hit
  lost), tagger 8/11 (one gained). Old rule fails (a hit lost on old DEV); **fewer-pictures clause passes** (cost down, 0 net
  hits lost, wrong −5). **Shipped as SHIP6** (config.use_shipped CONTINUATION_VETO 0.5). Next on SHIP6: K4AO, FLAP; TEST read.
- **SPOT screen result:** (a) +8 needed / +324 other → STOP. (b) veto of V4 accepts: 2 needed lost (Belly laugh ×2), 12
  other removed (10 other_gold, 2 phantom) → GO on paper; held until the FLAP veto arm (same slot) reports.

## Round 31 FLAP-F8 (written 2026-09-30 BEFORE any FLAP-F8 number; FineLAP frame cache from the FLAP screen)
**Idea.** F8 drops a listener-accepted rescue when DASM's family max over the span ± 0.5 s is < 0.575; the hit group shows
needed sounds DASM is deaf to (nyc Air horn 3.8: DASM 0.05). **Rule FLAP-F8:** F8 passes iff DASM ≥ 0.575 OR FineLAP family
max over [start, end] ≥ 0.329 (the FLAP P1-calibrated bar, not re-fit). **Variant FLAP-R (reported):** FineLAP replaces
DASM (pass iff FineLAP ≥ 0.329).
**Screen** (`benchmark/gold/finelap_f8_screen.py`, CPU, existing caches). Candidates: merged-DEV P2/PV items accepted by the
candidate-level TIER (`_tier({"V4", "AF_V4"}, peak)`, as the FLAP screen). F8 test as `filter_rescued`: DASM columns whose
`canonical` equals the family, frames with time in [start − 0.5, end + 0.5], max ≥ 0.575; no column / no frame → 0 (drop);
no DASM file → kept. DASM dir: `set_listener_split(split)` → `data/work/dasm_dev2` for tagger DEV; DEV has no `dasm_dev`, so
`data/work/devcand/dasm_cache` (DCC.DASM_DIR, the arms' F8 input). FineLAP score and gold class as the FLAP screen.
Restored = F8 drops, rule passes; newly dropped (FLAP-R only) = F8 passes, FineLAP < bar. **GO (FLAP-F8) iff needed
restored ≥ 2 AND other restored ≤ 2 × needed restored.** Candidate level only (ONCE, veto order not modelled).
- **Results on SHIP6 (25/55, 28, 2.479):** SHIP6+FLAP 25/55, 27 wrong (9/16/2), cost 2.451; old DEV 17/16, tagger 8/11; no
  needed hit lost on either part → **passes the old rule; shipped as SHIP7** (config.use_shipped FINELAP_VETO 0.329,
  FINELAP_DIR per split). SHIP6+K4AO 24/25/2.451 passes the clause too; re-run on SHIP7 (old-rule pass first).
  SHIP5+FLAP 25/32/2.592 (for the record). TEST caches for FineLAP being built; TEST read of SHIP6 and SHIP7 follow.

## Round 32 — joint swap: FineLAP in place of DASM, with the bars it needs (written 2026-09-30 BEFORE any number of it)
**Why joint.** Every new model so far was one extra vote on a pipeline whose bars were tuned for FlexSED + DASM. Adam: a
detector swap may need the mechanism around it re-tuned. Facts read from the code before writing this: the listener band
rescue (`_listener_band`) runs at the END of `fuse_flexsed`, after DV / BTP / CONT, so CONT never sees a rescued span;
rescued spans are FlexSED runs at 0.5 with gaps <= 0.24 s merged (= `LISTEN_RUN_GAP`), so CONT cannot fire on one anyway.
Hence "CONT lets ONCE loosen" has no coupling: **idea 3 (ONCE-G under CONT) is closed on this fact** (ONCE-G failed +3
wrong / 0 hits on SHIP3, round 22); the ONCE-dropped spans of SHIP6 are listed with their gold class for the record only.
Base = SHIP6 (merged DEV 25/55, 28 wrong, 2.479); pass = the combined rule vs SHIP6 (old rule: hits >= 25, wrong <= 28 +
2 x gain, cost < 2.479, no needed hit lost on either part; OR fewer-pictures: cost < 2.479, wrong <= 28 - 3 x lost, hits >= 22).
**Bars (one fixed rule, P1 items only, never the gold of rescue candidates).** As the FLAP screen: P1 items of
`dev_listener.json` (r13) + `dev2_listener.json` (tg) with `gold_class == hit_needed`; bar = the value at index
floor(0.10 n) of the ascending scores (>= 90 % of needed P1 items kept). Span bar (F8 seat): FineLAP family max over the
frames overlapping [start, end] = **0.329** (FLAP, n 59; not re-fit). Clip bar (DV seat): the same rule on the family's
FineLAP max over the WHOLE clip; the number is appended below after `finelap_full.py calib` prints it.
**FineLAP cache 2** (`data/work/finelap_cache2`, `benchmark/gold/finelap_full.py run`, `slurm/job_finelap_full.sh`): the
FLAP windowing, queries per clip = cache-1 queries + every family of the SHIP2+KV4 / SHIP3+DV / SHIP6 stage-4 rows, the
SHIP6 F8- / ONCE-dropped spans and the P4 (DR2) items, so no family the vetoes ask about is unqueried. Same model, same
frames: the scores of cache-1 queries must be equal (checked, reported). `build` writes the same scores in the DASM npz
layout (`fw`, `times` = frame midpoints, `labels`) to `data/work/finelap_as_dasm/<clip>.npz`, so the shipped F8 and DV
code read FineLAP through `LISTENER_DASM_DIR` with no edit to `src/` or `config.py`.
**Arms (real pipeline runs, DEV in ~/MscProj_r13, tagger DEV in ~/MscProj_tg; `benchmark/gold/flap_joint_arms.py`,
`slurm/job_flap_joint.sh`).**
- **SHIP6+FLR** = SHIP6 with `LISTENER_DASM_DIR` = finelap_as_dasm, `LISTENER_DASM_BAR` 0.329, `LISTENER_DASM_PAD` 0.08
  (half a 0.16-s frame: a midpoint within +-0.08 s = a frame overlapping the span, the calibration's window),
  `DASM_CLIP_VETO` = the clip bar. FineLAP replaces DASM in both seats (F8, DV); DR2 keeps its P4 answers (DASM's own finds,
  `DASM_P4_CACHE`, untouched); ONCE, BTP, CONT, K-V4, N2b unchanged.
- **SHIP6+FLR+F1** = the same + `LISTENER_NEW_TYPE_ONCE` (a family already drawn from a non-rescued span is not rescued
  again; the strongest run per new family) — the filter that stops the rescue from repeating families FineLAP re-admits.
**Screens first (CPU, `benchmark/gold/flap_joint_sim.py`, on the saved SHIP6 state; run before the arms, reported beside):**
(A) FLAP-F8 at arm level: SHIP6 F8-dropped rescued spans (`stage4.json` r14_dropped) with FineLAP span max >= 0.329 are
restored (conf = their listener-pool peak), ONCE re-run over kept + restored rescued spans (earliest per family; a
restored earlier span displaces a kept later one), the gate verdict of a restored span = the `augment` flag of the
same-family spec within 0.5 s in the F8-less arm `TO1+F7_proposed` (the harness memo would reuse the same votes); no such
spec -> counted and treated as drawn (pessimistic). Pictures = SHIP6 `augmentations.json` + restored specs through
`_display_spans`, rescored. Variants OR (DASM >= 0.575 or FineLAP >= 0.329) and R (FineLAP alone, the arm's rule).
(B) DV seat at candidate level: SHIP6 non-rescued rows the FineLAP clip bar would drop (no Qwen P1 / AF P1 keep) and the
DASM-DV-dropped rows (SHIP2+KV4 rows absent from SHIP3+DV) it would keep back, each with its gold class.
(C) the ONCE-dropped list (record only).
**Selection.** Between SHIP6+FLR and SHIP6+FLR+F1 (and SHIP6): split-half CV as `cv_select.py` — per-clip costs, halves
stratified by part (10 seeds x 2 halves and 5-fold): the cell with the lowest training cost is scored on the held-out
clips; the procedure's held-out cost vs fixed SHIP6. The full-merged-DEV argmin is the candidate; it is adopted only if
the CV cost <= SHIP6's CV cost AND it passes the combined rule. If the argmin is SHIP6: stable, STOP. No TEST read.
- **TEST read of SHIP6 (reported):** SHIP5+CONT 24 hits / 32 wrong (5/22/5) / 2.591 vs B0r 21/40/2.909; d −0.318
  [−0.614, −0.023], one-sided p 0.025 → **better** (benchmark/gold/final_test_ship6.md).
### Round 32 results — screens (CPU, `benchmark/gold/flap_joint_sim.json`; cache 2 job 31596346)
- **Cache 2 / bars:** 71 clips, every cache-1 score reproduced (71/71 equal); P1 hit_needed n 59; span bar 0.3292 (= FLAP);
  **clip bar 0.4331** (`finelap_full.json`; 54/59 kept under each).
- **(A) FLAP-F8 at arm level** (base re-placed = 25/28/2.479 exactly): of the 51 SHIP6 F8-dropped spans, 24 survive FineLAP +
  ONCE (candidate level had 56: the london_protest horns, applause laughs, tg_d088 thunders etc. collapse to one per family).
  5 are needed-class, but 3 of them the gate silences (snow_walk Giggle, tg_d127 Giggle / Chuckle: the laughing people are on
  screen); Hammer 13.76 and Explosion 2.84 are drawn (the Explosion displaces the kept 5.68 hit under ONCE: net +2 hits).
  11 other-class restored pictures are drawn (Train, Car, Coin, Water, Bird, Alarm, Sigh, Thump ×2, Toilet flush, Chewing).
  **OR: 27/55, 39 wrong (9/27/3), 2.676. R (FineLAP alone): 27/55, 38 (9/26/3), 2.648** (3 kept rescues newly dropped, none a
  drawn hit). Both fail (wrong +11 / +10 for +2 hits, cost up) → **STOP**; even with the 3 no-proxy spans gated, wrong ≥ 36.
  CV: the procedure picks SHIP6 in 20/20 halves and 50/50 folds (held-out 2.476 = fixed SHIP6).
- **(B) DV seat with the FineLAP clip bar 0.4331:** 43 non-rescued rows (conf ≥ 0.40) would be dropped with no listener keep —
  most are non-depictable (Speech, Music, Hum, Animal: never pictures); the picture-level ones are tg_d022 Dog 7.25 and tg_d128
  Hammer 9.0 (both cross-group X pictures), tg_d129 Scissors, un_people_clapping Guitar/Instrument, and **tg_d120 Domestic
  animals 2.97 = the needed Cat hit** (FineLAP clip max 0.031 for the cat: lost). 2 DASM-DV-dropped rows come back (Sigh,
  Hammer; other_gold). The seat swap trades ~3 cross pictures for 1 needed hit + 2 possible wrong: no gain by the cost.
- **(C) ONCE-dropped (record):** 10 spans (golf Bird ×3, arrest Footsteps/Locomotion ×5, tg_d120 Cat ×2, one of them
  needed-class but already hit by the non-rescued Domestic-animals span).
- **Arms SHIP6+FLR / SHIP6+FLR+F1** (job 31596369, queued after the other thread's r16dev / tgarms; real pipeline, both parts)
  are reported below when they finish; the screen's expectation is STOP for both (F8 seat +2 hits / +10 wrong, DV seat −1 hit).
### Round 32 amendment DV2 — FineLAP as a SECOND clip veto beside DASM's (written BEFORE its number; from screen (B))
Screen (B) showed the FineLAP clip bar catching two cross-group X pictures DASM's DV keeps (tg_d022 Dog, tg_d128 Hammer)
at the price of one needed hit (tg_d120 Cat). **Rule DV2:** SHIP6 + a second clip veto: a non-rescued span whose family's
FineLAP max over the whole clip is < 0.4331 (the P1 clip bar) is dropped unless Qwen P1 / AF P1 keeps it (DV's own keep);
DASM's DV stays. **Screen** (`flap_joint_sim.py dv2`, CPU): the saved SHIP6 pictures whose stage-4 row is in the (B)
would-drop list (same family, picture start inside [row start − 1.5, row end]) are removed, rescored on merged DEV.
**GO bar = the combined rule vs SHIP6** (old rule, or fewer-pictures: cost < 2.479, wrong ≤ 28 − 3 × hits lost, hits ≥ 22).
- **DV2 screen result** (`benchmark/gold/flap_dv2_screen.json`): merged **25/55, 26 wrong (9/15/2), cost 2.423** vs SHIP6
  25/28/2.479; removed exactly 2 pictures (tg_d022 Dog 7.25, tg_d128 Hammer 9.0, both cross), 0 hits lost (the Cat hit is a
  Cat picture, not the Domestic-animals row). **Passes the old rule → GO to the arm SHIP6+DV2** (`flap_joint_arms.py`:
  `FINELAP_CLIP_VETO2` 0.4331 applied by a wrapper around the harness's `fuse_flexsed` call — the pipeline's own code plus
  the one veto, listener keep as DV; `slurm/job_flap_dv2.sh`, after the FLR job). DV2 joins the CV selection set. To ship it
  would need a `src/` flag (not this thread's remit): reported as an Adam decision.
- **TEST read of SHIP7 (reported):** SHIP6+FLAP 24 hits / 31 wrong (5/21/5) / 2.568 vs B0r 21/40/2.909; d −0.341
  [−0.636, −0.045], one-sided p 0.015 → **better** (benchmark/gold/final_test_ship7.md).
- **SHIP7+K4AO:** 24/55, 24 wrong (9/13/2), 2.423 vs SHIP7 25/27/2.451: −1 hit (old DEV), −3 wrong. Passes the
  fewer-pictures clause at exactly 3:1; held for Adam's word (his 16:20 message puts hits first).
### Round 32 arm results — all STOP (jobs 31596369 FLR / FLR+F1, 31596404 DV2; `benchmark/gold/flap_joint_arms.json`)
Merged DEV (71 clips, 55 needed), real pipeline on both parts, Δ = paired clip bootstrap vs SHIP6:
| arm | hits | wrong (v/c/p) | cost | old DEV | tagger DEV | Δ cost vs SHIP6 [95 % CI] | verdict |
|---|---|---|---|---|---|---|---|
| SHIP6 | 25 | 28 (9/17/2) | 2.479 | 17/17 | 8/11 | — | base |
| SHIP6+FLR (FineLAP in F8 + DV, bars 0.329 / 0.4331) | 25 | 39 (9/27/3) | 2.789 | 18/22 | 7/17 | +0.31 [+0.03, +0.62] | STOP (tg_d032 Thunder lost; +11 wrong) |
| SHIP6+FLR+F1 | 23 | 39 (9/27/3) | 2.901 | 16/22 | 7/17 | +0.42 [+0.08, +0.82] | STOP |
| SHIP6+DV2 (FineLAP second clip veto) | 24 | 28 (9/17/2) | 2.535 | 17/17 | 7/11 | +0.06 [0.00, +0.17] | STOP (Thunder lost, no wrong removed) |
CV (split-half 10 seeds; 5-fold 10 seeds) over {SHIP6, FLR, FLR+F1, DV2}: the procedure picks SHIP6 in 20/20 halves and
50/50 folds; held-out cost 2.476 = fixed SHIP6. **Full argmin SHIP6: stable, nothing adopted, no TEST read.**
**Why the DV2 screen said GO and the arm says STOP.** The pipeline's P1 keep (`listener_p1_lookup`) falls back to the
R13-3 yes/no score > 3 when a span has no V4/V12 item; on tg_d022 Dog / Domestic animals and tg_d128 Livestock that fallback
keeps the rows (Qwen yes/no says yes), so the arm removes only the undrawn Animal / Livestock 7.5 rows and no picture. The
screen's keep used V4/V12 + AF only (no yes/no fallback) — a screen flaw, recorded; the arm is the truth. The FineLAP clip
bar also removes the weak Thunder 13.75 (conf 0.376, FineLAP clip max < 0.433, no keep) that is the tg_d032 hit.
**Reading.** The swap is worse than the vote: FineLAP in F8's seat re-admits 24 rescues after ONCE (11 drawn other-class:
Train, Car, Coin, Water ×2, Bird, Alarm, Sigh, Thump ×2, Toilet flush, Chewing) for +1 old-DEV hit (Hammer; the Explosion
2.84 displaces the 5.68 hit), and in DV's seat it is deaf to a needed Thunder. FineLAP's calibrated clip bar (0.433) sits
far above DASM's (0.084): FineLAP's clip-wide scores separate needed P1 items from others worse than DASM's do.
**Correction (replaces the "no yes/no fallback" and the (B) Cat sentences above).** The screen DID call the pipeline's
`listener_p1_lookup`; its flaw was elsewhere: `flap_joint_sim.py` did not name SHIP6 in `TG_ARMS`, so `tagger_prep.configure`
left SHIP6's listener caches on the DEV files and every tagger-DEV keep came back "missing". Fixed (TG_ARMS set in `parts()`)
and re-run: screen (B) now drops 34 rows (0 needed-class; Dog / Domestic animals / Livestock kept by the yes/no P1 answer),
the DV2 picture screen removes 0 pictures (25/28/2.479 = SHIP6, STOP), the Cat claim disappears; screen (A) and the CV are
unchanged (they never used the keep). The arm numbers stand; the screens now agree with them on both parts.

## Round 33 — SHIP7 error ledger and SpotSound second-opinion veto (advisor thread; written 2026-09-30 BEFORE any number)
**Base = SHIP7** (= SHIP6 + FineLAP veto; saved pictures = arm `SHIP6+FLAP|proposed`, rows identical to `SHIP7|proposed`):
merged DEV 71 clips, 25/55 hits, 27 wrong (9/16/2), cost 2.451; old DEV 17/16, tagger DEV 8/11.
**Ledger (descriptive, no rule; `benchmark/gold/ship7_errors.py` -> `ship7_errors.json`).** Every wrong picture (27) and every
missed needed sound (30) in one table: part, clip, family, time, class, and the stage that produced / removed it. Producing stage
of a picture = its stage-4 row (`origin` tagger / flex, `rescued`, `pre_start` = BTP moved it) and the arm's listener record
(`a_added` band rescue, `b_kept` PV keep, `f7` mirror keep, DASM P4 = DR2). Removing stage of a miss, in this fixed order:
gate-silenced (same-family spec in the onset window with `augment` false; the gate's reason kept) -> timing (a same-family
picture exists in the clip outside the window) -> below the picture floor (spec conf < 0.40) -> the first arm of the chain
B0r -> TO1+F7F8 -> +N2b -> +DR2 -> +KV4 -> +DV -> +BTP -> +CONT -> +FLAP where a same-family stage-4 row in the window disappears
-> rescue filter (`r14_dropped` F8 / ONCE / FLAP) -> listener refused (P2/PV item, Qwen V4 / AF V4 flags) -> FlexSED family peak
in [onset - 0.5, onset + 1.0] (>= 0.5 covered / never asked; 0.3-0.5 below LO) -> BEATs peak >= 0.175 (weak) -> unheard.
"Ceiling" = gate-visible (annotation says visible), unheard by every detector, or timing of a sound already drawn.
**SPOT-V (arm-level simulation on saved pictures; pre-registered rule).** A RESCUED placed picture of SHIP7 (spec `rescued`
true) is dropped when SpotSound-A's EXIST answer on its own candidate cut (`benchmark/gold/spotsound_raw.json`, matched by
part, clip, pool, canonical family and run overlap as `cross_group.p2_item`) starts with "no". DR2 / P4 rescues and rescued
pictures with no SpotSound item are kept and counted. Non-rescued pictures are never touched (the round-31 SPOT screen (b) was on
V4 accepts only). Reported beside, not decisive: SPOT-no as round 31 (EXIST no OR no GROUND interval overlapping the run +- 0.5 s).
Rescored with `score_per_sound` as the BTP / CONT screens. **GO bar = the combined rule vs SHIP7:** old rule (hits >= 25, wrong <=
27 + 2 x gain, cost < 2.451, no needed hit lost on either part) OR fewer-pictures (cost < 2.451, wrong <= 27 - 3 x hits lost,
hits >= 22). Known before writing: SHIP7 has 14 rescued specs (8 DEV, 6 tagger DEV) in its augmentations.json, so the reach is
at most those; the candidate-level (b) screen listed the snow-walk Belly laugh 8.0 (the Laughter 8.08 hit) among its SPOT-no items.
One new idea is written below AFTER the ledger is read and BEFORE its number.
### Round 33 CPO — change-point onset gate on the saved SHIP7 pictures (written BEFORE its number)
**Research (2024–2026 post-processing of SED frame scores; picked for our ONSET-only score).** Sound Event Bounding Boxes
(Ebbers et al., Interspeech 2024, https://www.isca-archive.org/interspeech_2024/ebbers24_interspeech.pdf, code
https://github.com/merlresearch/sebbs): an event's onset is a local maximum of the "delta" score = the family's frame score
filtered with an ideal step filter of length τ (mean of the next τ/2 s minus the mean of the previous τ/2 s); tentative
events are kept when the contrast between their peak and the neighbouring gap passes a merge threshold (absolute or relative);
tuning grid τ ∈ {0.32, 0.48, 0.64} s, abs ∈ {0.15, 0.2, 0.3}, rel ∈ {1.5, 2, 3}; +4.1 pt PSDS1 over median filtering on all
13 DCASE-2023 systems. nSEBBs (Cui et al. 2025, arXiv 2505.11889) makes τ and the absolute threshold adaptive per class
(τ 0.384–0.8 s by average event duration; abs 0.12–0.30 by duration; rel 2.0–3.2 by a posterior-contrast ratio) for
+≤4 % PSDS1 on SSL models. Boundary-aware inference (arXiv 2601.04178) needs event-proposal-network outputs: not applicable
to cached scores. Class-wise median-filter tuning "tends to overfit the validation set" (RealDESED, arXiv 2607.16736).
**Why this one.** The round-33 diagnostic (`benchmark/gold/cp_diag.py`, record only, no rule applied) on SHIP7's merged-DEV
pictures: of the 30 misses, 23 have NO same-family picture anywhere in the clip (unreachable by any picture post-processing)
and the 7 same-family pictures sit −14.7, −2.3, −1.9, +2.9, +6.4, +9.0, +11.0 s from the gold onset — re-timing (I1's
failure mode) can reach at most one; every family confusion at the right time sits on a visible / non-needed gold sound, so
relabeling (winner-take-all) turns cross into visible at the same cost. The live lever is a DROP: several wrong pictures start
where the family's FlexSED score shows no rise at all (delta peaks ≤ 0.1 near the start: a re-trigger of a sound already
going, or a family confused with a steady sound), which is exactly what cSEBB's change-point onset denies.
**Rule CPO (one rule; constants from the paper, none fitted).** For each placed SHIP7 picture (label L, start a): the family
evidence e(t) = max over the FlexSED cache columns whose canonical family is canonical(L) (cache `data/work/flexsed_cache`,
frames 0.04 s); no column → no evidence → the picture is KEPT (the rule is silent). Delta d(t) = mean e over [t, t+τ/2) −
mean e over [t−τ/2, t), the step filter zero-padded at both clip ends (a sound already present at t = 0 has its onset at frame
0 — the paper's convention, no clip-start exemption). The picture is kept iff d has a local maximum ≥ θabs at some frame in
[a − LATE, a + EARLY + τ/2] = [a − 1.0, a + 0.74] (the score's own collar mirrored — a hit's gold onset lies in
[a − 1.0, a + 0.5] — plus the filter's half-length lag); otherwise it is dropped. Nothing is re-timed or relabeled.
Primary cell: τ = 0.48 s (grid middle), θabs = 0.15 (the lowest grid value = the most permissive = the least hit risk).
Reported: the 3 × 3 grid τ ∈ {0.32, 0.48, 0.64} × θabs ∈ {0.15, 0.2, 0.3}; selection among them ONLY by split-half CV as
`cv_select.py` (per-clip costs, halves stratified by part, 10 seeds × 2 halves and 5-fold × 10 seeds; the cell with the
lowest training cost, ties → SHIP7, scored on held-out clips; the procedure's held-out cost vs fixed SHIP7). Also reported
(record only): the same gate on the family's BEATs tagger frames instead of FlexSED.
**Scope.** The gate cannot touch the 23 no-picture misses nor pictures of visible sources at real onsets; it differs from
the shipped CONT (round 31) in needing no run above 0.5 — it catches continuations whose evidence never reaches the bar
(e.g. a 0.175 re-trigger) and confusions with steady sounds.
**Screen** (`benchmark/gold/cpo_screen.py`, CPU, `~/MscProj_tg`, on the saved SHIP7 = SHIP6+FLAP pictures of merged DEV via
`btp_screen.parts` with `TG_ARMS=SHIP7`; the reproduced base MUST equal 25/55, 27 wrong, 2.451 before any cell is read;
rescored with `score_per_sound`). **Pass = the combined rule vs SHIP7 (25/55, 27, 2.451):** old rule (hits ≥ 25,
wrong ≤ 27 + 2 × gain, cost < 2.451, no needed hit lost on either part) OR fewer-pictures clause (cost < 2.451,
wrong ≤ 27 − 3 × hits lost, hits ≥ 22). The primary cell decides; a grid cell is adopted instead only if the CV picks it and
it passes. GO → arm on the real pipeline (a `src/` flag: not this thread's remit; an Adam decision). STOP → recorded, closed.
- **CPO screen result** (`benchmark/gold/cpo_screen.json`; base re-placed = 25/55, 27 (9/16/2), 2.451 exactly; 53 placed
  pictures; every clip has a FlexSED and a BEATs cache): **primary CPO(0.48, 0.15): 18/55, 22 wrong (9/11/2), cost 2.704 → STOP**.
  It drops 12 pictures: 5 cross (Gunshot 8.25 explosion clip, Electric shaver 16.0, Pant 7.0 + Dog 7.25 tg_d022, Hammer 9.0
  tg_d128) and **7 hits** (bakery Door 4.25, birds_forest Bird 10.25, protest Glass 10.75 + 16.89, storm-house Alarm 2.18,
  tg_d120 Cat 0.56, tg_d149 Bee 1.25): for these needed sounds FlexSED's family score shows no rise ≥ 0.15 near the picture
  start — they are BEATs / listener-rescued faint sounds the text-queried detector barely hears, so a FlexSED change point is
  not a witness of our hits. The whole 3 × 3 grid is STOP (hits 16–18, wrong 16–23, cost 2.620–2.732; the 0.64/0.2 and
  0.64/0.3 cells at 2.620 lose 7–8 hits for 8–10 wrong removed, under the 3:1 clause). CV over {SHIP7 + 9 cells}: split-half
  picks SHIP7 17/20 (procedure 2.519 vs fixed 2.448), 5-fold 45/50 (2.581 vs 2.457); full argmin SHIP7. Record: the same gate
  on BEATs tagger frames 12/55, 19, 2.958 (13 hits lost) — coarser 1-s windows have no usable change points.
  **Closed.** Reading: on this benchmark the wrong pictures are not "onsets that aren't there" more often than the hits are;
  the misses are dominated by families never drawn (23/30), which no post-processing of frame scores can reach.
- **Ledger plumbing note (before any number was read as final):** the first run read DEV specs from the tagger out-dir
  (`tagger_prep.configure` redirects `R.R13`); fixed to explicit roots. Rescued pictures are matched to their P2/PV item by
  `same_family` (the motorcycle Explosion 13.92 picture is the rescued Gunshot child span) and to the SpotSound record by the
  item's own key. Miss order corrected as registered above (rescue filter / listener before "timing"; a BEATs span >= the display
  bar in the window with no B0r row = a B0 veto, FlexSED clip veto when the family's FlexSED clip max < 0.3).
- **K4A-4E (one new idea; written BEFORE its number, after reading the ledger's wrong side).** K4AO (round 30: a drawn
  non-rescued span is dropped when both open-inventory listeners were asked on its P1 cut and neither names its family or a kind
  of it) passed only the fewer-pictures clause on SHIP7 (24/24/2.423: -1 hit, -3 wrong; held). **Rule K4A-4E:** K4AO, except the
  span is KEPT when a non-LLM ear hears the family at the span: DASM family max over [start - 0.5, end + 0.5] >= 0.575 (F8's bar)
  OR FineLAP family max over [start, end] >= 0.329 (FLAP's bar) — the two shipped bars, not re-fit; four ears must all be silent.
  Screen on the saved SHIP7 pictures (`ship7_errors.py`): a placed non-rescued picture is dropped iff EVERY same-family stage-4 row
  under it (conf >= 0.35, non-rescued) fails the test; K4AO alone is reported beside as the check against its arm (24/24).
  **GO bar = the combined rule vs SHIP7** (old: hits >= 25, wrong <= 27 + 2 x gain, cost < 2.451, no needed hit lost on either
  part; OR fewer-pictures: cost < 2.451, wrong <= 27 - 3 x lost, hits >= 22). If GO, the arm needs a `src/` flag (Adam's call).
### Round 33 results (CPU, `benchmark/gold/ship7_errors.py` -> `ship7_errors.json`; base reproduced 25/55, 27 (9/16/2), 2.451)
**Ledger.** 27 wrong = 9 visible (7 BEATs spans + 2 rescued: motorcycle Explosion on visible fireworks, tg_d128 Laughter of
people on screen), 16 cross (15 BEATs / FlexSED spans, 1 rescued: golf Bird 18.84), 2 phantom. Cross by group: late repeat 5
(barbershop Shaver 16.0, detective Alarm 3.36 / 9.08, tg_d088 Thunder 13.25, golf Bird), early 4 (explosion Gunshot 8.25, applause
Crowd 0.0, protest Glass 4.75, tg_d088 Thunder 8.5), wrong family at a visible sound 7 (crossing Steam, tg_d022 Pant + Dog, tg_d088
Explosion, tg_d107 Screaming, tg_d128 Hammer, tg_d029 Goose). 30 misses by the registered order: gate-silenced 5 (macaws, pet-shop
birds, church bell, boxer dog Fart 0.0 and 5.6 — the 5.6 row is merged into the silenced 0.22 spec), unheard by every detector 5
(nyc Hammer 8.1, Clang, golf Whack x2, tg_d125 Explosion), B0 FlexSED clip veto 1 (tg_d029 Chicken, BEATs 0.81, FlexSED clip max
0.26), below the display bar / picture floor 4 (birds_forest Bird 0.22, tg_d032 Thunder 7.4 0.27, tg_d095 Dishes 0.37, ambulance
Vehicle 0.32), rescue filters 3 (F8: Hammer 13.7, Explosion 2.8; ONCE + kinship dedup: tg_d120 Meow 2.9), listener refused 7 (nyc
Air horn and tg_d107 Laughter: AF yes / Qwen no at the Qwen-only tier; Footsteps, Gasp, Whistle, tg_d033 Siren, tg_d125 Clapping),
below LO 2 (rainforest_2179 Bird 0.435, storm siren 0.42), P2 run mistimed 1 (tg_d032 Thunder 2.8, run starts 2.04), CONT 1
(favela Train 14.6), timing 1 (applause Crowd 1.9, span from 0.0).
**What is left.** Hit side: every group is at a ceiling (gate = annotation "visible" for a same-family thing on screen; unheard;
below bar) or on a line already closed by a pre-registered test (listener: H2 / K3 / PTC / V4D / QE / Step / Kimi; filters: F8-U /
K1 / FLAP-F8 / ONCE-G; LO: CV54–CV76; FlexSED clip veto: FV; CONT: bought −5 wrong). No miss is reachable by an untried general
rule on this data. Wrong side: 9 visible = gate perception (M / N / GA / Gemma / SUBJ / OM closed); the 7 wrong-family cross
pictures are the only group a general rule still touches (K4AO takes 3 of them at −1 hit).
**SPOT-V: STOP.** 8 rescued placed pictures: 6 matched to their own P2/PV item — every EXIST answer is "Yes." (snow-walk
Laughter, golf Bird, storm Alarm, tg_d030 Vehicle, tg_d127 Laughter, tg_d128 Giggle); tg_d120 Cat is a DR2 rescue (no P2 item);
the motorcycle Explosion picture is the rescued Gunshot child span (its P2 Gunshot item: "Yes."). No picture changes under EXIST-no
or the round-31 SPOT-no: 25/27/2.451 = SHIP7. The candidate-level (b) losses ("Belly laugh" items) were sibling items the rescue
did not use. SpotSound is closed for the listener role.
**K4A-4E: STOP.** K4AO on the saved pictures = 24/55, 24 (9/13/2), 2.423 — exactly its arm (drops rainforest_7629 Insect 0.14, a
HIT, and Pant 7.0, Goose 11.0, Thunder 8.5, all cross). With the DASM-or-FineLAP keep nothing is dropped (25/27/2.451 = SHIP7):
FineLAP scores 0.95 on the Insect and 0.70–0.77 on the three wrong pictures, so its 0.329 bar keeps all four. Post-hoc, reported
only (not pre-registered, not adopted): DASM alone separates them (Insect 0.59 vs 0.15 / 0.15 / 0.38 at the 0.575 bar) and would
read 25/24/2.366 on the saved pictures ((4 x 30 + 2 x 24) / 71) — a 0.015 margin on one hit; if Adam wants it, it is a new pre-registered arm on SHIP7.

## Round 33 SSL-SaN — audio-visual localisation as a gate vote (SOTA scan; written 2026-09-30 BEFORE any SSL-SaN number)
The gate VLMs only see frames; every audio-visual idea so far (N Qwen3-Omni, GA, PIC-SIM, SUBJ, OM) failed. SSL-SaN (BMVC
2025, `xavijuanola/SSL_SaN`, arXiv 2508.21761, official `sslsan` weights from the authors' Drive folder; the Zenodo record
17064608 holds only the IS3+ dataset) is a sound-source localiser trained with silence / noise / off-screen negatives: its
audio-image cosine map should sit near zero when the sound's source is not on screen — exactly the gate's question.
- **Score.** `benchmark/gold/gate_sslsan.py` (`slurm/job_gate_sslsan.sh`): for every cached Qwen3.8-27B gate stretch of
  all 139 gold clips (`gate_gold/Qwen38-27B`), the same six frames (stretch ± 1 s, `gate_gold.run_vlm` times) and the stretch
  audio (± 1 s, 16 kHz mono, repeated to 10 s as the official test dataloader does; mel 512 / 239 / 257 normalised, dB)
  go through the official encoders; stretch score = max over the 6 frames of the max of the raw 14 × 14 cosine map
  (no sigmoid, no per-image normalisation). Sound score = MIN over its stretches (the gate silences only if every stretch
  is seen). Report-only: the same frames with zero audio (the paper's silence claim, checked on our clips).
- **Threshold t**, fixed on the NON-judge gold clips (the test85 part of `gate_gold`: 90 clips, 176 sounds of importance ≥ 2,
  98 seen / 78 needed): the value with the best balanced accuracy of the SSL-SaN-alone verdict (seen iff score ≥ t), over
  the observed sound scores, highest t on ties. Those clips are consumed by this calibration.
- **DEV judge clips (43 seen, 36 needed; shipped majority 16 / 43, 31 / 36), two pre-set variants:** (a) tie-break: a
  stretch with yes == no among name / a-b / description (1–1 with a None, 0–0) is decided by score ≥ t; (b) fourth vote:
  seen iff yes > no over the four votes, a 2–2 tie decided by SSL-SaN — which means SSL-SaN decides every non-unanimous
  VLM stretch and unanimous stretches stand. Also reported: SSL-SaN alone at t.
- **GO iff seen silenced ≥ 19 / 43 with needed kept ≥ 31 / 36 in (a) or (b).** Otherwise report-only, no arm.
- **Reach, from the cached votes alone (no SSL-SaN number):** of the 27 leaked seen sounds, 0 have only tied not-seen
  stretches, so (a) cannot add a single silenced seen sound and GO is reachable only through (b): 7 leaks have every
  not-seen stretch non-unanimous (+3 needed), 4 already-silenced needed sounds could be rescued, and 6 kept needed sounds
  have all their not-seen stretches non-unanimous (at risk). If the weights or code do not run within about an hour the
  screen stops and says why.
### Round 33 result — SSL-SaN gate vote: STOP (job 31596662, H200/A100 4 h partition, ~25 min; `benchmark/gold/gate_gold/sslsan/`, `sslsan_summary.json`)
The official weights (`sslsan.tar`, epoch 200, keys = `AVENet_ssltie`) run in msproj (torch 2.5.1). The silence claim holds
on our clips: median stretch peak 0.453 with the real audio vs 0.072 with zero audio on the same frames. But the peak does
not separate on-screen from off-screen: calibration on the 176 non-judge sounds gives t = 0.372 with balanced accuracy
0.61 (80 / 98 seen silenced, 32 / 78 needed kept) — the map lights up for nearly every real sound, source on screen or not.
DEV judge clips (43 seen, 36 needed; base 16 / 31): SSL-SaN alone 33 / 43 but 7 / 36; (a) tie-break 16 / 43, 29 / 36 (2
flips, both needed sounds lost: as_explosion Gunshot, golf Whack); (b) fourth vote 20 / 43, 26 / 36 (9 flips: +4 seen —
as_glass Glass, crossing_bells Train, ia_youtube Water and Tap — for 5 needed lost: snow_walk Laughter, as_explosion Gunshot
/ Footsteps / Explosion, golf Whack). Neither variant reaches the bar (≥ 19 / 43 with ≥ 31 / 36) → **STOP**, report-only.
An audio-visual localiser trained on single-source clips answers "does this audio match this image" — on our scenes it
says yes for the scene-typical off-screen sounds (explosions in a war street, a siren over a storm house) as readily as for
visible ones; it is not the missing gate signal. Gate group stays closed.

## Round 34 MOSS — MOSS-Audio-8B-Thinking as a listener (SOTA scan; written 2026-09-30 BEFORE any MOSS number)
The listener rescue (TIER) is at a ceiling with Qwen3-Omni + Audio Flamingo Next; the third-ear tries so far (Kimi-Audio
N3 / TIER3, Step-Audio, H2 / K3 / PTC / V4D / QE) all stopped. MOSS-Audio-8B-Thinking (OpenMOSS-Team, Jun 2026, Apache-2.0,
arXiv 2606.01802; Qwen3-8B backbone, explicit time tokens; MMAU / MMAR / MMSU 71.1 vs Qwen3-Omni 67.9) is the current SOTA
open audio-LLM, so it is screened once in the same seat, the same way.
- **Model / code.** `OpenMOSS-Team/MOSS-Audio-8B-Thinking` (18.1 GB bf16 safetensors) with the official `src/` code
  (github.com/OpenMOSS/MOSS-Audio, `MossAudioModel`, `MossAudioProcessor`, `enable_time_marker=True`) in `~/venv_moss`
  (msproj + transformers 4.57.1, the pinned version; torch 2.5.1 kept). Thinking cannot be switched off for audio requests
  (official usage guide: audio prompts take the template's shortcut branch), so the Thinking model is used as it is.
  Fallback (decided before the smoke): if the smoke shows > 40 s per cut or the code does not run in ~1 h of fixing,
  `MOSS-Audio-8B-Instruct` replaces it and the report says so.
- **Question.** `benchmark/gold/listener_moss.py` (`slurm/job_listener_moss.sh`): the UNCHANGED V4 question
  (`listener_variants.V4_Q`, asserted equal) on exactly the `run_audio` cut Qwen V4 and AF V4 heard, for every P2 / PV item
  of `dev_listener_v.json` (r13) and `dev2_listener_v.json` (tg); the processor's default prompt (audio, then the text = the
  Qwen / AF order); GREEDY; up to 2048 new tokens including the thinking; if no `</think>` came out the thought is closed
  (`\n</think>\n\n` appended) and at most 64 answer tokens follow (flagged `moss_forced`). The answer = the text after the
  LAST `</think>`; only that text goes to the unchanged matcher (`listener_afnext.v4_match`: match_names word match, else
  all-mpnet-base-v2 cosine > 0.6) → accept {V4}; null_accept on the item's cached null_family (the control). No gold in gen / match.
- **Rules screened** (`benchmark/gold/moss_screen.py`, CPU; per candidate Q = Qwen V4, A = AF V4, M = MOSS V4, peak = the
  FlexSED run peak, split = `config.TIER_SPLIT` = 0.6; shipped TIER = Q at peak ≥ 0.6, Q AND A below):
  (a) **MOSS replaces Qwen:** peak ≥ 0.6 → M; below → M AND A (the AF leg kept).
  (b) **MOSS as a third ear:** peak ≥ 0.6 → Q OR (A AND M); below → (Q AND A) OR (M AND (Q OR A)). (b) ⊇ TIER, so it can
  only add; (a) can also lose.
  Counted per rule on merged-DEV P2/PV candidates of the gold clips (`ptc_screen.PARTS`; class by `dev_listener.gold_class`:
  needed = hit_needed, other = other_gold + none): needed-class and other-class accepts ADDED and LOST vs shipped TIER.
  **GO iff needed added ≥ 2 AND other added ≤ 2 × needed added AND needed lost = 0** (per rule). GO → an arm on the real
  pipeline is a `src/` flag = Adam's decision; STOP → recorded, closed. Also printed: the 7 known hit-side misses the ears
  refuse (nyc Air horn, tg_d107 Laughter, carnival Whistle, as_explosion Footsteps / Gasp, tg_d033 Siren, tg_d032 Thunder) with
  Q / A / M and the MOSS text.
- **Sanity (report only).** MOSS on the 274 P1 items of `dev_listener_p1v4.json` (the drawn BEATs spans whose cuts Qwen and
  AF already answered; 29 hit_needed / 43 none / 202 other_gold by the `dev_listener.json` gold field): MOSS yes-rate on
  hit_needed vs none, next to Qwen's and AF's on the same cuts (`qwen_fams` / `af_fams`). A listener that says yes to
  everything is not an ear; the P2/PV null-accept rate is reported for the same reason.
### Round 34 result — MOSS-Audio-8B-Thinking listener: STOP on both rules (job 31597407, H200, 1.6 h for 1278 items; `benchmark/gold/{dev,dev2}_listener_moss.json`, `dev_listener_p1_moss.json`, `moss_screen.json`)
The Thinking model ran as it is (no Instruct fallback: 4 s per cut, thinking 170–185 tokens on average, 3 / 1278 forced
closes, 17.8 GiB, rule identity 602/602 + 402/402 for the Qwen / AF flags). Merged-DEV P2/PV candidates on gold clips: 1004
(35 needed); shipped TIER accepts 24 needed / 94 other; MOSS alone says yes to 19 needed / 125 other (null accept 2).
(a) MOSS replaces Qwen: needed +4 −9, other +63 −50 → **STOP** (it loses snow-walk Laughter, tg_d016 Throat clearing,
tg_d133 Fart ×2, tg_d125 Clapping, nyc Hammer …; it names steady scene sounds — crowd chatter, rushing water, chewing,
bird chirping — on every cut of them).
(b) third ear (Q OR (A AND M) high; (Q AND A) OR (M AND (Q OR A)) low): needed +2 (tg_d107 Laughter 8.40 and Giggle 8.80 —
the same gold laugh, the one known miss the ears refused), other +39, lost 0 → **STOP** (39 > 2 × 2; the additions are the
AF-alone crowd / bird / water / chewing candidates that MOSS co-signs).
Known misses: as_explosion Gasp 6.7 — MOSS yes (Q no, A no: only rule (a) reaches it); Footsteps 2.1 no ('Explosion, Beep');
carnival Whistle 6.1 no ('train'); tg_d032 Thunder no on all five cuts ('heavy rain', 'rushing water'); nyc Air horn 3.8 and
tg_d033 Siren have no P2/PV candidate for any ear. P1 sanity (274 drawn spans, same cuts): yes-rate hit_needed 15/29 (0.52)
vs none 15/43 (0.35) — MOSS 0.52 / 0.35, Qwen 0.55 / 0.44, AF 0.66 / 0.63: MOSS is the most selective ear on the drawn
spans but not a better one on the needed side. Listener seat stays closed; `~/venv_moss` and the 18 GB weights remain on
the cluster (delete on request).
**Ledger note (after the result above; counts unchanged).** The known-misses lookup in `moss_screen.py` matched the label
`"Air horn"` exactly and `startswith("nyc")`; fixed to clip substring + family prefix and re-run. Correction: nyc
'Air horn, truck horn' 3.76 (peak 0.71, hit_needed) IS a P2 candidate — Q no, A yes, MOSS no ('slamming, humming'); tg_d033
has no Siren-family P2/PV item (its Alarm 0.0–10.0 cut: MOSS 'heavy breathing'). Siren stays unreachable by any ear; Air horn
is refused by MOSS too.

## Round 35 DBX — DASM back-extension of late picture starts (written 2026-10-01 BEFORE any DBX number)
**Motivation.** b3_barbershop: gold "Electric shaver" 0.1–27.8 s (needed). The shipped SHIP7 picture starts at 16.0 because BEATs is
only confident from 16 s, while DASM's shaver family score is >= 0.93 in every frame from 0 s. A late onset like this costs a miss
AND a wrong (late-repeat cross): re-timing the start to where DASM's evidence begins turns both.
**Base = SHIP7** (saved arm `SHIP6+FLAP|proposed`, rows = `SHIP7|proposed`): merged DEV 71 clips, 25/55 hits, 27 wrong (9/16/2),
cost 2.451; the screen MUST reproduce this before any DBX line is read.
**Rule DBX (one rule; every constant is a shipped one, none fitted).** For each placed NON-rescued picture (label L, start a,
end b; `rescued` = the representative spec's flag, as BTP): family evidence e(t) = max over the DASM cache columns whose canonical
family is canonical(L) (`data/work/devcand/dasm_cache` for DEV, `data/work/dasm_dev2` for tagger DEV, frames 0.02 s); no column
or no cache -> untouched (the rule is silent; counted). Runs = frames with e >= 0.575 (LISTENER_DASM_BAR, F8's bar), gaps
<= 0.24 s (LISTEN_RUN_GAP) merged, the pipeline's `_runs`; run end = last frame + dt. Take the run [s, r] that covers a
(s - 1e-6 <= a <= r + 1e-6); none -> untouched. If a - s >= 1.0 s (MAX_AFTER_END, the shipped 1.0) the start moves to s
(new picture [s, b], label and end unchanged; s = 0.0 allowed); otherwise untouched. Rescued pictures are never moved.
**DBX-S (variant, report only, not decisive):** as DBX, but a picture is left untouched when any other placed picture of the
clip (any rescue status) has the same canonical family, an original start < a and an original end > s (the moved picture would
overlap an earlier picture of its family).
**Screen** (`benchmark/gold/dbx_screen.py`, CPU, `~/MscProj_tg`, `TG_ARMS=SHIP6+FLAP`, pictures via `btp_screen.parts` and
`btp_screen.placed`, DASM roots explicit from `cross_group.PARTS`; rescored with `score_per_sound`; nothing in `src/` or
`config.py` edited). Reported: base per part and merged; DBX and DBX-S per part and merged; every moved picture (part, clip, family,
old -> new start, class before -> after via `cross_group.classify`), hits lost per part. Known risk: a hit whose start moves
more than 0.5 s (EARLY) before its gold onset becomes an early cross.
**Pass = the combined rule vs SHIP7 (25/55, 27, 2.451):** old rule (hits >= 25, wrong <= 27 + 2 x gain, cost < 2.451, no needed
hit lost on either part) OR fewer-pictures clause (cost < 2.451, wrong <= 27 - 3 x hits lost, hits >= 22). DBX decides; DBX-S is
recorded beside. GO -> an arm on the real pipeline is a `src/` flag = Adam's decision; STOP -> recorded, closed.
### Round 35 result — DBX: STOP on both rules (CPU, `~/MscProj_tg`, `benchmark/gold/dbx_screen.py` -> `dbx_screen.json`)
Base reproduced exactly: merged 25/55, 27 (9/16/2), 2.451; DEV 17/36, 16 (6/8/2), 2.204; tagger DEV 8/19, 11 (3/8/0), 3.000.
53 placed pictures (8 rescued, never moved); every clip has a DASM cache. 45 non-rescued pictures: 37 have DASM < 0.575 at their
start, 6 sit in a run that begins < 1.0 s earlier, 2 move.
**DBX: merged 24/55, 27 (9/15/3), 2.507 | DEV 17/36, 15 (6/7/2), 2.163 | tagger DEV 7/19, 12 (3/8/1), 3.273 -> STOP** (1 hit lost,
cost up). Moved: (1) b3_barbershop Electric shaver 16.0 -> 0.0 (end 23.25): cross -> hit, but the clip already had the shaver
hit at 0.14 (the motivation's premise was wrong for SHIP7: the 0.14–13.25 picture IS the hit; the 16.0 picture is a late-repeat
cross); both shaver pictures now start inside the gold window: one is the hit, the other a dup (clip h1 c1 -> h1 d1, −1 wrong). (2) tg_d149 Bee, wasp 1.25 -> 0.0 (end 3.75): gold Bee onset
1.0, so the new start is 1.0 s early (> EARLY 0.5): hit -> phantom, clip h1 -> p1 (−1 hit, +1 wrong).
**DBX-S (record): merged 24/55, 28 (9/16/3), 2.535 -> STOP**: the barbershop move is blocked by the overlap with the 0.14 picture,
only the tg_d149 loss remains.
**Closed.** Reading: DASM at 0.575 runs from 0 s on a steady sound, so back-extension lands on the clip start, ahead of the gold
onset where the sound ramps up; the one late-repeat it fixes was already a dup-in-waiting next to a hit. No other SHIP7 picture has
DASM evidence at its start that reaches back >= 1 s.

### Visibility re-check (Adam, 30 Sept 23:26; blind to the old tick; tool docs/review/visibility_recheck.html)
14 borderline gate-side gold sounds (9 visible-wrong pictures, 5 gate-silenced needed sounds) re-judged blind:
Q1 source on screen, Q2 obvious without sound. Rule for applying, fixed before scoring: a question answered "unsure"
keeps its old value; seen = visible or obvious. Answers: benchmark/gold/visibility_recheck_answers.json.
Three sounds flip from seen to needed (visible and obvious both no): as_church_bell Bell 0.0, as_fire_alarm Alarm 9.0
(Q2 unsure → old obvious "no" kept), tg_d088 Thunder 1.3. The other 11 keep their class (the 5 gate-silenced stay needed;
storm Thunder ×2, protest horn, bath water, kids' laughter, starter's bang stay seen). DEV gold only (gold_AG / tagger_AG);
TEST untouched. Every system is re-scored on the corrected DEV gold.
Re-scored merged DEV on the corrected gold (58 needed): B0r 18/58, 51 wrong, 3.690; TO1+F7F8 26/45/3.070; SHIP3+DV
27/31/2.620; SHIP4+BTP 28/30/2.535; SHIP5+CONT 28/25/2.394; **SHIP7 (SHIP6+FLAP) 28/58, 24 wrong (6/16/2), 2.366**;
SHIP7+K4AO 27/21/2.338. Order of the shipped chain unchanged; every step still lowers cost.

### Round 35 K4A-D (written before its arm number; disclosure: motivated by the round-33 post-hoc note that DASM separates
K4AO's one lost hit from its three drops on the saved pictures)
K4AO on SHIP7, except a span that DASM hears (family ≥ 0.575, F8's bar, in the span ± 0.5 s; the existing N2c-D keep) is
kept. Arm SHIP7+K4AD vs SHIP7 on the corrected merged DEV gold (base 28/58, 24 wrong, 2.366); combined pass rule.

## Round 35 DBR — DASM-covered repeat drop (written 2026-10-01 BEFORE any DBR number; corrected DEV gold)
**Motivation.** b3_barbershop: gold "Electric shaver" 0.1–27.8 s (needed). SHIP7 draws the shaver twice: 0.14–13.25 (hit) and
16.0–23.25 (late-repeat, counted wrong) because BEATs dips below its bar at 12–16 s (0.08–0.28) while DASM's shaver score is
0.93 in every frame. RPT-S (round 31, FlexSED silence between repeats) failed at arm level; DBR asks DASM the same question.
**Base = SHIP7** (saved arm `SHIP6+FLAP|proposed`) on the corrected DEV gold (visibility re-check, 58 needed): merged 28/58 hits,
24 wrong (6/16/2), cost 2.366; the screen MUST reproduce this before any DBR line is read.
**Rule DBR (one rule; every constant is a shipped one, none fitted).** For each placed NON-rescued picture (label L, start a,
end b; `rescued` = the representative spec's flag, as BTP/DBX): "earlier" = the placed pictures of the clip (any rescue status,
the ORIGINAL placed set, not kept-only) with canonical family = canonical(L) and original start < a; none -> untouched (first of
family). prev_end = the latest end among them. Family evidence e(t) = max over the DASM cache columns whose canonical family is
canonical(L) (`data/work/devcand/dasm_cache` for DEV, `data/work/dasm_dev2` for tagger DEV, frames 0.02 s); no column or no
cache -> untouched (counted). Runs = frames with e >= 0.575 (LISTENER_DASM_BAR, F8's bar), gaps <= 0.24 s (LISTEN_RUN_GAP)
merged, the pipeline's `_runs`, run end = last frame + dt (`dbx_screen.family_runs`, byte-identical to DBX). If prev_end >= a
(the pictures overlap; no gap) -> untouched (counted "overlap"; not a late repeat). Else the picture is DROPPED iff one merged run
[s, r] has s <= prev_end + 1e-6 and r >= a - 1e-6 (every frame of the gap is >= the bar). Rescued pictures are never dropped.
**Screen** (`benchmark/gold/dbr_screen.py`, CPU, `~/MscProj_tg`, `TG_ARMS=SHIP6+FLAP`, pictures via `btp_screen.parts`, DASM roots
from `cross_group.PARTS`; rescored with `score_per_sound`; nothing in `src/` or `config.py` edited). Reported: base per part and
merged; DBR per part and merged; every dropped picture (part, clip, family, start, end, earlier end, class before via
`cross_group.classify`, clip tally before -> after); untouched counts by reason; hits lost per part. Data-quality note: clips whose
DASM family score is flat (max - min < 0.05 over the whole clip), counted on placed-picture families (the score the rule reads)
and on any family. Known risk: the dropped picture, not the earlier one, is the hit (earlier = visible/cross); that shows as a
lost hit.
**Pass = vs SHIP7 on the corrected gold (28/58, 24, 2.366):** old rule (hits >= 28, wrong <= 24 + 2 x gain, cost < 2.366, no needed
hit lost on either part) OR fewer-pictures clause (cost < 2.366, wrong <= 24 - 3 x hits lost, hits lost <= 3). GO -> an arm on the
real pipeline is a `src/` flag = Adam's decision; STOP -> recorded, closed.
### Round 35 result — DBR: GO (old rule) (CPU, `~/MscProj_tg`, `benchmark/gold/dbr_screen.py` -> `dbr_screen.json`)
Base reproduced exactly on the corrected gold: merged 28/58, 24 (6/16/2), 2.366; DEV 19/38, 14 (4/8/2), 2.122; tagger DEV
9/20, 10 (2/8/0), 2.909. 53 placed pictures (8 rescued); every clip has a DASM cache. 44 pictures are first of their family,
8 repeats have a DASM gap that is not covered, 0 overlap, 1 dropped.
**DBR: merged 28/58, 23 (6/15/2), 2.338 | DEV 19/38, 13 (4/7/2), 2.082 | tagger DEV unchanged 9/20, 10, 2.909 -> GO (old rule)**
(hits equal, wrong −1, cost down, no hit lost on either part). Dropped: b3_barbershop Electric shaver 16.0–23.25 (earlier picture
ends 13.25; DASM shaver 0.926–0.945 in every frame): cross -> gone, clip h1 c1 -> h1 (−1 wrong).
**Data-quality note (flat DASM family score, max − min < 0.05 over the whole clip):** 5 of 71 clips on a placed-picture family —
b3_barbershop shaver (0.93–0.95, flat HIGH: DASM never sees the 12–16 s dip BEATs sees), b3_laundromat Train (0.23–0.26),
ambulance Siren (0.15–0.19), helicopter Vehicle (0.12–0.15), tg_d128 Hammer (0.00–0.05) — the last four are flat low, below
the bar, so the rule never fires on them. Every one of the 71 clips has some flat family (absent families sit near 0); only the
placed-family count is informative. The one drop rests on a DASM score that is constant for 28 s, so DBR's win is one picture
on one clip; the merged TEST is spent and cannot confirm it. GO = a `src/` flag on the real pipeline is Adam's decision.
- **DBR arm (after its GO screen):** stage-4 flag `REPEAT_DASM_BRIDGE` 0.575 (before CONT): same rule on stage-4 spans ≥ the
  display bar. Arm SHIP7+DBR on the corrected merged DEV; old rule vs SHIP7. Disclosure: the screen fired only on the
  motivating clip (b3_barbershop).
- **K4A-D result (corrected merged DEV):** SHIP7+K4AD 28/58, 21 wrong (6/13/2), cost 2.282 vs SHIP7 28/24/2.366; old DEV
  19/14 and tagger 9/7; no needed hit lost on either part → **passes the old rule; shipped as SHIP8** (use_shipped
  KEEP_NEEDS_V4_ALL "onto" + KEEP_NEEDS_V4_ALL_DASM_KEEP). K4AO alone 27/21/2.338 (the DASM keep saves the insect hit).
  DBR re-queued on SHIP8; TEST read of SHIP8 follows.

### Visibility re-check round 2 = control sample (written before Adam's answers)
Round 1 re-checked only sounds where the pipeline disagreed with the gold, which can only help the pipeline. Round 2 is
the control: 20 merged-DEV sounds (importance ≥ 2, not in round 1) drawn at random with a fixed seed (20261001), 10 marked
seen and 10 needed (benchmark/gold/visibility_recheck2_items.json), re-judged blind with the same two questions (clips play
0.5 s before to 0.5 s after, Adam 23:43). Same applying rule (unsure keeps the old value). Report: the flip rate here vs
round 1 (3/14); if this sample flips at a similar rate, the round-1 corrections are a general label-noise fix, not a
pipeline-favouring one. Flips here are applied to the gold too and every system re-scored.
- **Round 2 (control) result:** 1 of 20 random sounds changes class (5 %): tg_d120 Caterwaul 0.3 needed → seen (cat on
  screen); it counts against the pipeline (its Cat picture was a hit). Round 1 (pipeline-disagreement sample): 3 of 14 (21 %).
  Reading: random label noise is about 5 %; disagreement cases hold more label errors, as expected, so round 1's gain is
  partly a selection effect — both rounds are applied and disclosed. Merged DEV now 57 needed.

### One gold file (Adam, 1 Oct 00:49)
tagger_AG.json merged into benchmark/gold/annotations/gold_AG.json (50 tg_* clips, marked "set": "tagger"; header "sets");
score_per_sound.subsets_of keeps them out of the old dev/test subsets (checked: dev 49, test_bench 60, tagger 50 = dev2 22 +
test2 28, all present); every reader now points at gold_AG.json. Removed as redundant (in git history): tagger_AG.json,
gold_AG_2026-09-20.json, the re-check answer/item files (their outcome is on the gold entries: recheck_2026_09_30 /
recheck2_2026_09_30 notes; the answers are in commits of 30 Sept).
Check after the merge and the round-2 flip (merged DEV, 57 needed): B0r 18/57, 51, 3.634; SHIP7 (SHIP6+FLAP) 27/25/2.394;
SHIP8 (SHIP7+K4AD) 27/22/2.310.
- **DBR arm (stage 4, on SHIP8):** 27/57, 22 wrong (7/12/3), 2.310 = SHIP8 cost: at stage 4 it also fires on other clips
  (one cross becomes a phantom). Cost not lower → **fail**; not shipped.
- **TEST read of SHIP8 (reported; TEST gold unchanged):** SHIP7+K4AD 23 hits / 29 wrong (4/20/5) / 2.568 vs B0r 21/40/2.909;
  d −0.341 [−0.636, −0.045], one-sided p 0.017 → better. Same cost as SHIP7 on TEST (24/31): −1 hit, −2 wrong.
- **Correction (Adam, 1 Oct 00:59):** tg_d120 caterwaul's source is a clock, not the cat on screen → back to needed. Round 2
  control therefore changes 0 of 20 sounds (0 %) vs round 1's 3 of 14. Final merged DEV (58 needed): B0r 18/51/3.690;
  SHIP7 28/24/2.366; **SHIP8 28/58, 21 wrong (6/13/2), 2.282**.

### Round 36 IMP — DASM impact-type peaks as a listener trigger (written before any number)
**Motivation:** b3_golf_course "Whack, thwack" (6.5 s, 24.4 s) is unheard by BEATs/FlexSED but DASM "Specific impact sounds"
peaks 0.69 / 0.58 there; other SHIP8 misses are short impacts (Clang nyc_2627 3.8, Hammer nyc_1689 8.1/13.7, Explosion
tg_d125 5.4, Dishes tg_d095 16.6). Base = **SHIP8 on the final merged DEV: 28/58 hits, 21 wrong, cost 2.282**
(`benchmark/gold/ledger_ship8.json`, 30 misses listed incl. importance-1). Nothing in `src/` or `config.py` is edited.
**Step 1 (CPU, candidate level, `benchmark/gold/imp_screen.py` -> `imp_screen.json`).** DASM impact-type labels = exact names
present in the 215-label cache: `Specific impact sounds`, `Thump, thud`, `Knock`, `Tap`, `Hammer`. Asked for but absent
from DASM's list: Bang, Slam, Smash/crash, Whack/thwack, Clang. Near-matches deliberately NOT used (named here so they are
not added later): Jackhammer, Engine knocking, Chink/clink, Crack, Crackle, Snap, Burst/pop, Finger snapping. Peak = one
per merged run of the UNION column (max over the 5 columns) >= 0.575 (F8 bar), runs merged over gaps <= LISTEN_RUN_GAP
0.24 s (the pipeline's `_runs`); peak time = argmax frame; per-label firing reported as a breakdown only. Clips = merged
DEV (DCC.dev_stems 49 + dev2_stems.txt 22), DASM caches from `cross_group.PARTS` roots (copied to the laptop unchanged;
assert every clip has one). Each peak goes to ONE bucket, first that applies, window = the scorer's picture-start window
[g.start − 0.5, g.start + 1.0]: (a) ledger miss (needed, in `ledger_ship8.json` misses); (b) needed importance >= 2 sound
SHIP8 already hits (would be a duplicate); (c) needed importance-1 (don't-care); (d) seen (non-needed) sound; (e) nothing.
For every (a) peak print the gold family next to the firing DASM label(s).
**Trigger for step 2:** DISTINCT ledger misses with >= 1 peak >= 3 AND peaks in (b)+(d)+(e) <= 3 x that number. Below
that: STOP, recorded, closed — no GPU job.
**Step 2 (GPU, only if triggered): `benchmark/gold/imp_listen.py` + `slurm/job_imp_listen.sh`.** At every peak (all buckets;
the rule cannot see the gold) ask Qwen3-Omni the shipped V4 open question (`listener_variants.V4_Q`, no ducking,
`listener_v4d.py` loader) on a 2-s cut centred on the peak; map names with `listener_variants.match_names`; a picture is
added (start = peak time, 2 s long) iff the FIRST matched family is impact-type (the gold families above or the 5 DASM
labels' canonical names) and no SHIP8 picture of that family starts within 1.0 s. Arm = saved SHIP8 pictures ("SHIP8"
via `btp_screen.parts`) + added pictures, rescored with `score_per_sound` per part and merged (dbr_screen style).
**Pass vs SHIP8 (28/58, 21, 2.282):** old rule (hits >= 28, wrong <= 21 + 2 x gain, cost < 2.282, no needed hit lost on
either part) OR fewer-pictures clause (cost < 2.282, wrong <= 21 − 3 x hits lost, hits lost <= 3). Known structural
limit, stated before running: a hit needs the drawn family == the gold's canonical family (Whack thwack, Clang, Hammer,
Explosion, Dishes); "Specific impact sounds" is its own family, so a Qwen answer that only names a generic impact cannot
score — step 1 therefore also reports whether Qwen could name each (a) family at all (step 2 reports the name it gave).
### Round 36 result — IMP step 1: STOP (CPU, laptop, `benchmark/gold/imp_screen.py` -> `imp_screen.json`)
71 clips, every one with a DASM cache. 30 union peaks >= 0.575; every one is `Specific impact sounds` alone (Thump/thud, Knock,
Tap, Hammer never reach the bar on merged DEV). Buckets: (a) SHIP8 miss 3, (b) needed already hit 2 (mv_protest Glass 11.0,
Shatter 16.9 = would be duplicates), (c) 0, (d) seen 16, (e) nothing 9. Distinct misses with a peak = 3 (b3_golf_course
Whack thwack 6.5 -> peak 6.74 @0.692 and 24.4 -> 24.72 @0.581; tg_d095 Dishes 16.6 -> 16.88 @0.588): the >= 3 half of the
trigger is met, but elsewhere = 27 > 3 x 3 = 9 -> **STOP, no GPU job, step 2 not built.** Elsewhere peaks sit on tg_d075 (8),
mv_protest_scene_movie (5), tg_d128 (5), tg_d095 (3), b3_golf_course (2) + 4 singles; the seen sounds they land on are Glass,
Shatter, Clicking, Moo, Clang, Whip, Arrow, Fireworks, Water, Stir, Explosion, Engine. The other impact-like misses have no peak:
nyc_1689 Hammer 8.1 / 13.7 (union max in window 0.18 / 0.28), nyc_2627 Clang 3.8 (0.48), as_explosion Explosion 2.8 (0.31),
tg_d125 Explosion 5.4 (0.569, just under the bar — noted, not re-run at a lower bar: that would be a post-hoc bar). Structural
note stands: all three (a) peaks fire on the generic family, so even a triggered step 2 would need Qwen to name Whack/Dishes.

### Round 37 IMP-V — the gate names the maker of a DASM impact peak (written 2026-10-01 BEFORE any number)
**Motivation (Adam):** Round 36 stopped because 27 of 30 DASM impact peaks fall on seen sounds or on nothing — but it never
applied the pipeline's own visibility gate, which is what silences seen sounds. Idea: when DASM hears a generic impact, let
the video frames (the gate VLM) name what made it (golf swing -> whack), then let the gate decide as usual. Base = **SHIP8 on
the final merged DEV: 28/58 hits, 21 wrong, cost 2.282** (the screen must reproduce it before any IMP-V line is read).
Nothing in `src/` or `config.py` is edited.
**Peaks.** Exactly Round 36's: `imp_screen.peaks` on the DASM caches from `cross_group.PARTS` (71 merged-DEV clips, union of
the 5 impact-type columns >= 0.575, runs merged over LISTEN_RUN_GAP, peak = argmax frame); assert the same 30 (clip, t) as
`imp_screen.json`. The rule never reads the gold.
**Skip (no question asked).** A placed SHIP8 picture (`btp_screen.parts`, `TG_ARMS=SHIP8`) whose canonical family is impact-type
(IMPACT_FAMS = canonical of the option labels below + "Specific impact sounds") with start <= t <= end -> "inside existing".
**Closed question (one, both orders).** Frames = the gate's own stretch sampling with a = b = t: 6 frames at t − 1 + 2i/5,
i = 0..5 (`_sample_frames_at`, clamped at 0), clip from `detector_dry.clip_path` (DEV) / `tagger_prep.CLIPS` (tagger DEV).
Model = the shipped gate VLM `Qwen/Qwen3.8-27B` via `reason._load`, `reason._ask(..., images=frames, max_new=6)`, thinking off.
Prompt: "A short impact sound is heard at this moment. Which of these most likely made it?" + lettered options (a)…(l),
"Answer with the letter only." Options (phrase -> drawn AudioSet label): a whack or thwack — a club, bat, racket or hand
striking something -> `Whack, thwack`; a hammer striking -> `Hammer`; a knock, knuckles on a door or wood -> `Knock`;
a door slamming -> `Slam`; a clang, metal struck -> `Clang`; something smashing or crashing, glass breaking -> `Smash, crash`;
dishes, pots or pans clattering -> `Dishes, pots, and pans`; a gunshot -> `Gunshot, gunfire`; an explosion -> `Explosion`;
a thump or thud, a heavy object landing -> `Thump, thud`; a door opening or closing -> `Door`; none of these / cannot tell.
Asked in the forward order and in the fully reversed order (the model's letter-position bias, `reason._ab`); the reply's first
letter after `lstrip("(")` is read. ACCEPT iff both orders map to the same label and it is not "none"; split or none -> no
picture (reported as such).
**Gate.** On the SAME frames, `reason._sound_is_visible(label, frames, mdl, proc)` exactly as `gate_gold.run_vlm` (three votes,
majority = config.VISIBILITY_RULE; `LAST_VOTES` saved). Seen -> no picture. Not seen -> a picture (label, t, t + 2.0 s),
unless a SHIP8 picture of the same canonical family starts within 1.0 s of t (would be a duplicate; skipped, reported).
**Score (CPU, `dbr_screen` style).** Arm = saved SHIP8 pictures + the new pictures, `score_per_sound.score_clip` per part and
merged, class of each new picture by `cross_group.classify`, hits lost per part by per-clip comparison.
**Pass vs SHIP8 (28/58, 21, 2.282):** old rule (hits >= 28, wrong <= 21 + 2 x gain, cost < 2.282, no needed hit lost on either
part) OR fewer-pictures clause (cost < 2.282, wrong <= 21 − 3 x hits lost, hits lost <= 3). GO -> a `src/` flag is Adam's
decision; STOP -> recorded, closed.
**Structural risk, stated before running:** the naming question can only succeed when the frames SHOW the maker — which is
close to the gate's own "name" vote. So the peaks where the VLM names a family are the ones the gate is most likely to silence,
while the three target misses (golf Whack 6.5 / 24.4, tg_d095 Dishes 16.6) are needed precisely because the maker is not
plainly on screen. The screen therefore reports three outcome counts: named -> gate seen (silent); named -> drawn (with class);
unnamed / split / none. A STOP with most peaks in the first bin means the idea folds into the gate; a STOP with the targets in
the third bin means frames alone cannot name an off-screen impact.
**Files:** `benchmark/gold/imp_v_screen.py` (`run` = GPU, one JSON per peak in `benchmark/gold/imp_v/`, resumable; `score` =
CPU -> `imp_v_screen.json`), `slurm/job_imp_v.sh` (H200-4h / A100-4h, from `~/MscProj_tg`, `TG_ARMS=SHIP8`).

## Round 37 BOX — the gate's own box checks its "seen" (Adam's idea, 1 Oct; written BEFORE any BOX number)
**Motivation (Adam):** "the church bell is not seen on screen, we just see the church but no bell. Ask the model for a
rectangle around the object that makes the noise when it says yes (visible), then ask only on that rectangle: is that an X?"
The gate silences needed sounds when a same-type THING is on screen (bell_miami Bell 0.2: tower, no bell; b3_pet_shop
Bird 0.1; tg_d133 Fart). Unlike amendment M (OWLv2 crops used to ADD "seen"), BOX uses the gate VLM's OWN box to CHECK
a "seen" decision; it can only flip seen -> not seen.
**Rule BOX (one rule).** For every stretch whose shipped majority (name/ab/desc, yes > no, cached run
`benchmark/gold/gate_gold/Qwen38-27B`) says seen, the same VLM (Qwen3.8-27B, greedy, `reason._ask`) gets the stretch's
frames (`gate_gold.run_vlm` times: stretch ± 1 s, 6 frames; `_sample_frames_at` may return fewer, frame numbers index the
returned list) and: "These frames (numbered 1..N) are from the moment a sound of {label} was heard. Find the object that is
making that sound. Answer with JSON only: {"frame": k, "bbox_2d": [x1, y1, x2, y2]} on a 0-1000 grid of that frame
(1000 = full width or height). If no such object is visible, answer exactly: none." Reply parse: `none` (first word)
-> no box; a JSON with frame in 1..N and 4 numbers -> box, rescaled to pixels and clamped; anything else -> `unparsed`,
stretch unchanged (garbage cannot masquerade as a flip). The box is cropped (20 % margin, >= 224 px short side after
upscaling, `som_gate.crops`) and asked two LABEL-FREE questions (no sound word, so the crop cannot be re-primed):
direct "This is a close-up cut from a video frame. Is this {obj}? Answer yes or no." (no = reply starts with "n") and
a/b `reason._ab("This is a close-up cut from a video frame. What is it?", "{obj}", "something else")` (no = False;
None = split = not no). **A stretch flips to not-seen iff the box is `none`, or BOTH crop answers are no.** Clip verdict
unchanged: silent only if every stretch is seen. `{obj}` = OBJECT_OF[canonical family] — a NEW table, not DETECT_QUERY
(whose Bell entry is "a church bell tower", the trap itself): Bird "a bird"; Water "water (a river, the sea, waves or a
tap)"; Rain "rain falling"; Drum "a drum"; Walk, footsteps "a person's feet or legs stepping"; Bell "a bell (the bell
itself)"; Laughter "a person laughing"; Motorcycle "a motorcycle"; Machine gun "a gun being fired"; Train "a train";
Rustle "something rustling (leaves, paper or cloth being moved)"; Vehicle "a car or truck"; Crowd "a crowd of people";
Chink, clink "glasses or cutlery touching"; Glass "glass"; Whack, thwack "something being hit"; Whip "a whip"; Air horn,
truck horn "a truck or vehicle horn"; Cellphone buzz, vibrating alert "a mobile phone"; Siren "an emergency vehicle";
Horse "a horse"; any other family: "the thing that makes the sound of {label}".
**Truth = CURRENT gold** (`gold_AG.json`, re-checked 30 Sept / 1 Oct): seen = visible or obvious, re-derived per sound by
matching clip stem / resolved label / start (the `seen` stored in the cache files is NOT read). Importance >= 2 only.
**Screen set:** DEV judge clips (JUDGE100 ∩ cached = 49 clips). No gate cache exists for any tg_* (dev2 tagger) clip, so
tg_d133 Fart cannot be screened; stated, not re-run. Base (shipped majority, cache vs current gold, computed before the
rule was written): seen 41 / silenced 16, needed 38 / kept 33; 54 seen-majority stretches to box.
**Pass:** GO iff needed kept rises by >= 2 (>= 35) AND seen silenced drops by <= 1 (>= 15). Report: table (base vs BOX),
every flipped sound with gold class, counts of none / crop-no / unparsed / unchanged-seen, the raw box replies and crops
of every bell_miami stretch and of every flipped sound (`docs/review/box_crops/`). Script `benchmark/gold/box_gate.py`,
job `slurm/job_box_gate.sh`; nothing in `src/` or `config.py` edited. GO -> a `src/` flag is Adam's decision; STOP ->
recorded, closed.

### Declared secondary: visible-weighted cost (Adam, 1 Oct 01:04–01:06; written before re-scoring)
Adam: a picture of a sound whose source is on screen is roughly half as bad as an unrelated one; show it as a scale.
Secondary cost per clip = (4·miss + w·visible + 2·cross + 2·phantom) / clips, reported as a sweep w ∈ {0, 1, 2} (w = 2 is the
primary). Weights are the owner's elicited value, not literature-derived (precedent for class-weighted errors: PSDS
cross-trigger weight; caption-error severity metrics). Disclosure: docs/metric_per_sound.md recorded the September decision
"visible = full false alarm"; this is a partial, post-hoc reversal, so the primary stays unchanged and every past decision
stays judged on it. Cross is not split (it includes late same-family pictures the onset rule exists to punish). Re-scored for
the whole shipped chain on merged DEV and on TEST (aggregates already reported); sensitivity only, no new "better" claim.
### Round 37 result — IMP-V: STOP (H200, job 31598456, `benchmark/gold/imp_v_screen.py` -> `imp_v_screen.json`, votes in `imp_v/`)
Base reproduced: SHIP8 merged 28/58, 21 (6/13/2), 2.282; DEV 19/38, 14, 2.122; tagger DEV 9/20, 7, 2.636. 30 peaks: 5 inside an
existing impact-type SHIP8 picture (mv_protest Glass x4, motorcycle Explosion); 25 asked. Both orders agreed on a family 8 times:
6 -> gate SEEN (no picture: construction Thump 6.4; golf Whack 6.7 / 11.5 / 19.8 / 24.7 — the golfer is on screen, name+desc
votes yes, a/b split; tg_d128 Clang 5.8 "metal gate"); 2 -> drawn (tg_d075 Thump/thud 1.4 and 9.9, gate not seen: named "water
bottle" / "fire extinguisher") — both score CROSS. 17 split/none, incl. the targets tg_d095 Dishes 16.9 (fwd Whack, rev none).
**IMP-V: merged 28/58, 23 wrong (6/15/2), cost 2.338 | DEV unchanged | tagger DEV 9/20, 9, 2.818 -> STOP** (hits equal, +2 wrong,
cost up). Reading, as pre-stated: the three targets never become pictures — the golf Whacks are named correctly but the gate
(rightly, by its own rule) sees the golfer, i.e. the idea folds into the gate; Dishes is unnamed. Position bias note: the
forward reply was "(a)" (= Whack, first option) in 13 of 25 asks and the reversed reply "(a)" (= none, first option) in 9 of
25 — the two-order agreement rule did its job (only 8 accepted) but the closed question itself is mostly letter bias.
### Round 37 result — BOX: STOP (job 31598457, H200, 5 min; `benchmark/gold/box_gate.py` -> `gate_gold/box_Qwen38-27B/`, `box_summary.json`)
| rule | seen silenced / 41 | needed kept / 38 |
|---|---|---|
| majority (shipped) | 16 | 33 |
| BOX | 9 | 35 |

Needed kept +2 (bar met: bell_miami Bell 0.2, mv_tornado_scene Siren 8.9) but seen silenced −7 (bar <= 1) → **STOP**. The 7 seen
sounds un-silenced: marrakech_3102 Motorcycle 12.7, storm_7200 Rain 0.1, as_explosion Machine gun 10.1, b3_golf_course
Whip 11.2, mv_tornado Cellphone buzz 1.4 + Horse 17.1, un_driving_motorcycle_4O3bZRYO Motorcycle 16.2 — 5 of them by a
`none` box on a source the majority vote (and the gold) sees; the model declines to box what it just said it saw.
54 seen stretches: none 16, crop-no 6, unparsed 15 (replies "Based on the visual evidence…" prose or `<tool_call>`
computer-use garbage — left unchanged by rule), seen kept 17. bell_miami (`docs/review/box_crops/`): stretch 1 `none`;
stretches 2–3 box the IHS medallion / a sign sliver on the tower (1080×1920 frame; boxes 13×30 and 33×30 on the 0–1000
grid), crop answers "no" + a/b False → flipped, as Adam predicted — the mechanism works on the motivating clip but the box
question is too eager to say `none` elsewhere. tg_d133 Fart not screened (no tg_* gate cache). Closed.

## Round 38 E4 CF — the gate VLM answers the gold's own two questions (written 2026-10-01 BEFORE any CF number)
**Motivation:** the shipped gate (`reason._sound_is_visible`: name / a/b / describe, majority) asks questions that differ
from the gold's. The gold's re-check tool (`docs/review/visibility_recheck.html`) asks Q1 "Is the thing that makes THIS
sound on screen while you hear it? A look-alike counts as no" and Q2 "With the sound off, would a viewer already know
this sound is happening?"; gold seen = Q1 yes or Q2 yes. CF asks the same VLM the gold's questions verbatim.
**Rule CF (one design, fixed here).** Same VLM (`Qwen/Qwen3.8-27B` via `reason._load`, greedy `reason._ask`), same 6
frames per stretch as the cached shipped run (`gate_gold.run_vlm`: stretch − 1 s .. end + 1 s, 6 times, clamped at 0,
`_sample_frames_at`), on EVERY stretch of every gold sound of the DEV judge clips (`benchmark/gold/gate_gold/Qwen38-27B`
∩ JUDGE100 = 49 clips; no tg_* cache exists, as in BOX). Two questions, each a closed a/b in BOTH letter orders
(`reason._ab` logic, options "yes" / "no", "Answer with the letter only."), prefixed by one context line:
Q1 = "These frames are from the moment a sound of {label} was heard. Is the thing that makes THIS sound on screen while
you hear it? A look-alike counts as no."; Q2 = "These frames are from the moment a sound of {label} was heard. With the
sound off, would a viewer already know this sound is happening?". A question is YES only if both orders say yes; both
no, or a split (None) -> that question = no. No frames returned -> both no. **stretch seen_CF = Q1 yes OR Q2 yes.**
Clip verdict as shipped: silent only if every stretch is seen.
**Variants.** (a) CF replaces the shipped majority (name/ab/desc) on every stretch. (b) CF is a fourth vote next to the
three cached votes: count True vs False over [name, ab, desc, CF] (cached None excluded); more True -> seen, more
False -> not seen, equal -> the shipped majority decision for that stretch.
**Truth = CURRENT gold** (`gold_AG.json`, seen = visible or obvious, re-derived per sound by stem / resolved label /
start; importance >= 2). **Base (computed before the rule was written, same as BOX): seen 41 / silenced 16, needed 38 /
kept 33.** Both the local and the cluster copies of the gold are checked identical before the job runs; `score` is run
on the laptop on the copied-back votes.
**Pass (per variant):** GO iff seen silenced >= 19 with needed kept >= 32, or needed kept >= 35 with seen silenced >= 15.
Round GO if either variant passes (which one is reported). Report: table (base, a, b), every sound whose verdict differs
from base with gold class and all four raw replies, stretch counts (Q1 yes, Q2 yes, splits, CF seen vs majority seen).
Script `benchmark/gold/cf_gate.py` (`run` GPU -> `gate_gold/cf_Qwen38-27B/`, resumable; `score` CPU ->
`gate_gold/cf_summary.json`), job `slurm/job_cf_gate.sh`; nothing in `src/` or `config.py` edited. GO -> a `src/`
flag is Adam's decision; STOP -> recorded, closed.

## Round 38 BOX-2 — BOX without the `none` flip, unparsed replies re-prompted once (written BEFORE any BOX-2 number)
Refinement of Round 37 (same screen: 49 DEV judge clips, importance >= 2, truth = current gold, base = shipped majority
16/41 seen silenced, 33/38 needed kept; same GO bar: needed kept >= 35 AND seen silenced >= 15). Changes: (1) a `none`
box NO LONGER flips a stretch — the model declining to box is not evidence the source is absent; (2) every stretch whose
round-37 reply was `unparsed` (prose or `<tool_call>` text; 15 in scored sounds, 17 in all) is re-asked ONCE on the same
frames with a strict prompt: "These frames (numbered 1..N) are from the moment a sound of {label} was heard. Reply ONLY
with JSON, no other text, no tools: {"frame": k, "bbox_2d": [x1, y1, x2, y2]} on a 0-1000 grid of frame k (1000 = full
width or height), or {"bbox_2d": null} if the object making that sound is not visible." A still-unparsed reply (or
`null`) leaves the stretch unchanged; a parsed box gets the round-37 crop + the two label-free crop questions unchanged;
(3) a stretch flips to not-seen ONLY when a box was parsed AND both crop checks say no (direct "n…" AND a/b False).
Step A (CPU, cached round-37 answers, rule (1)+(3) without re-prompts) is computed and reported first; step B (GPU,
`box_gate.py reprompt`, job `slurm/job_box_gate2.sh`) adds the re-prompted stretches and is the BOX-2 line. Pass as
round 37. Reported: table (base / BOX-2 step A / BOX-2), every flipped sound with gold class, re-prompt outcomes
(parsed / null / still unparsed). Nothing in `src/` or `config.py` edited.

### Round 38 E5 GBTP — generalised band-twin pull (pre-registered 1 Oct, before numbers)
**Why.** The shipped BTP (round 30) pulls a non-rescued picture's start back to a FlexSED run >= 0.5 of its family that ends
0-1 s before the picture and starts <= 1.5 s before it. Mistimed misses remain (picture drawn, onset outside the hit window
[onset - 0.5, onset + 1.0]): as_explosion Explosion 2.8 (picture 5.68), ly_applause Crowd 1.9 (picture 0.0, early), tg_d032
Thunder 2.8 and 7.4 (picture 13.75), tg_d120 Meow 2.9 (Cat picture 0.56), birds_forest Bird 1.3 (picture 10.25).
**Rule GBTP (one rule; every constant is a shipped one, none fitted).** For each placed picture of the saved SHIP8 arm whose
spec is not `rescued`, fam = `canonical(label)`. For each of the three frame-score caches the evidence is the max over the
cache columns whose `canonical(column label) == fam` (exact family equality, as the shipped BTP); runs = the pipeline's
`_runs(evidence, times, bar, LISTEN_RUN_GAP = 0.24 s)`, run = (s = first frame time, e = last frame time + frame step).
Bars: FlexSED 0.5 (= the shipped `BAND_TWIN_PULL`, read from `round13_dev.arm_cfg("SHIP8")`), BEATs 0.35
(= `config.DISPLAY_THRESHOLD`), DASM 0.575 (= `config.LISTENER_DASM_BAR`, the F8 bar); the script asserts these values.
A run QUALIFIES for a picture starting at a iff `a - 3.0 <= s < a` AND `e >= a - 0.24` (it continues, with gaps <= 0.24 s,
up to the picture's start). A run whose onset is earlier than a - 3.0 does NOT qualify even if it reaches a (the onset must lie
within the 3.0 s). New start = the smallest s over all qualifying runs of all three models; no qualifying run -> unchanged.
The start never moves forward; the end is unchanged. No guard against overlapping an earlier same-family picture (the shipped
BTP has none); any dup change is reported. Rescued pictures are never moved (as shipped).
**Caches.** FlexSED `data/work/flexsed_cache/<clip>.npz` (25 fps; `fw` stored [labels, frames], `DCC.load_fr` transposes);
DASM `cross_group.PARTS[part]["dasm"]/<clip>.npz` (`fw` [frames, labels], `times`); BEATs `cross_group.PARTS[part]["beats"]`
= `data/work/j2_dev_beats` / `j2_dev2_beats` (the shipped `infer_beats`: 2-s window, 0.25-s hop, stamped at window end - 0.5 s,
so a BEATs onset is late-biased and the pull from it is conservative; hop 0.25 > 0.24 means BEATs runs are never gap-merged).
A clip missing one model's cache contributes no runs from that model (counts reported). Truth = `gold_AG.json` (current).
**Reachable by construction** (3.0 s back, never forward): only as_explosion (5.68 - 3.0 = 2.68 <= 2.8) can become a hit.
ly_applause and tg_d120 have the picture BEFORE the sound; tg_d032 (13.75 - 7.4 = 6.35 s) and birds_forest (8.95 s) are out of
reach. Stated before running, so the result is read as "what the rule does to the whole set", not as a fix of these five.
**Screen.** `benchmark/gold/gbtp_screen.py` (CPU, from `~/MscProj_tg`, `TG_ARMS=SHIP8`), on the saved `SHIP8_proposed`
pictures of merged DEV (49 DEV + 22 tagger DEV2 = 71 clips) via `btp_screen.parts()`, rescored with `score_per_sound`;
base must reproduce 28/58 hits, 21 wrong (6/13/2), cost 2.282 (asserted) before the rule runs. Nothing in `src/` or
`config.py` edited. Reported: merged DEV per part; every moved picture (clip, family, old -> new start, per-picture class
before -> after from `cross_group.classify`, clip hits before -> after).
**Pass.** GO iff merged hits >= 28 AND no clip on either part has fewer hits than before (clip-level, so a gain on one clip
cannot offset a loss on another) AND cost at visible weight w = 2 (primary, = 2.282 base) goes down. Secondary: cost(w) =
(4 miss + w visible + 2 cross + 2 phantom) / 71 for w in {1, 2} (base w = 1: 156/71 = 2.197) for the "more hits" profile.
GO -> a `src/` flag is Adam's decision; STOP -> recorded, closed.
### Round 38 result — GBTP: STOP (CPU, login node, `benchmark/gold/gbtp_screen.py` -> `gbtp_screen.json`)
Base reproduced: SHIP8 merged 28/58, 21 (6/13/2), 2.282; DEV 19/38, 14, 2.122; tagger DEV 9/20, 7, 2.636. All three caches present
for all 71 clips. 42 non-rescued pictures; 20 moved (earliest onset from FlexSED 16, BEATs 3, DASM 1); 11 of them to 0.0 s
(the family's run is on from the clip's first frame). 19 moves keep their class (hit -> hit 11, visible 3, cross 2, phantom 2,
dontcare 1); 1 flips: tg_d149 Bees (gold 1.0 s) picture 1.25 -> 0.0, now 1.0 s early (> 0.5 s tolerance): hit -> phantom.
**GBTP: merged 27/58, 22 wrong (6/13/3), cost 2.366 | DEV unchanged 19/38, 14, 2.122 | tagger DEV 8/20, 8, 2.909 -> STOP**
(one needed hit lost, hits < 28, cost up). Sweep: w = 1 base 2.197 -> 2.282; w = 2 base 2.282 -> 2.366. as_explosion (checked after the run): the saved SHIP8 arm does have the Explosion picture at 5.68 (plus 9.25). FlexSED has an
Explosion run 2.88-3.44 at the true onset, but it ends 2.2 s before the picture (no continuation); the runs at the picture
start just AFTER 5.68 (FlexSED 5.76, DASM 5.70; BEATs none), so nothing qualifies and the 5.68 picture stays (still a miss). The listed
mistimed misses are untouched, as predicted by construction. Reading: the frame scores are already "on" at the picture's
start in most cases; pulling to the earliest onset mostly shifts to 0.0 s without changing class, and the one real move on
a target hurts. Closed.
### Round 38 result — BOX-2: STOP (step A CPU from the round-37 cache; step B job 31598492, H200, 3 min; `box2_summary.json`)
| rule | seen silenced / 41 | needed kept / 38 |
|---|---|---|
| majority (shipped) | 16 | 33 |
| BOX-2 step A (cache, no `none` flip) | 14 | 35 |
| BOX-2 (+ re-prompted unparsed) | 13 | 35 |

Needed kept +2 (bar met: bell_miami Bell 0.2, mv_tornado_scene Siren 8.9) but seen silenced −2 at step A (bar <= 1; lost:
mv_tornado Cellphone buzz 1.4, un_driving_motorcycle_4O3bZRYO Motorcycle 16.2 — a box was found, the crop said "no") and
−3 after step B (+ as_glass Chink, clink 18.5: re-prompt boxed a 25×133 strip, crop "no") → **STOP** by one seen sound.
Re-prompt of the 15 unparsed scored stretches: 11 parsed boxes (1 flip), 4 `null`, 0 still unparsed — the strict prompt
fixes the format. The `none` flip was the main damage in round 37 (7 → 2 seen lost); what remains is crops of small boxes
answered "no" on truly visible sources. Closed.
### Round 38 BOX-2 arm — full-pipeline arm of the BOX-2 gate check (pre-registered 1 Oct, before numbers)
**Why.** The BOX-2 screen (above) stopped by one seen sound at gold level (needed kept +2: bell_miami Bell, mv_tornado_scene
Siren; seen silenced −3). In the pipeline that should mean more hits for a few extra on-screen (visible) wrong pictures, which
Adam accepts for a "more hits" profile only if every extra wrong picture is visible-type. This arm measures it end to end.
**Arm.** SHIP8+BOX2 = SHIP8 + `config.GATE_BOX_CHECK = True` (new flag, default False; shipped behaviour unchanged). In
`reason.decide_subjects`, after the gate verdict (and FIX_GATE) of each stretch, a stretch still called seen gets the BOX-2
rule exactly as screened (`box_gate.py` run + reprompt, copied into `reason._box_check`): the gate VLM (Qwen3.8-27B) is asked
`BOX_Q` on the stretch's same 6 frames; if unparsed, once more with the strict `REPROMPT_Q`; a parsed box is cropped (20 %
margin, short side >= 224 px) and asked `DIRECT_Q` ("Is this {object}?") and the a/b crop question; both "no" -> the stretch
is not seen. `none` / `null` / still unparsed / degenerate box -> unchanged. The harness's cached gate verdicts are reused as
before; the new questions go through `_ask` and are memoised under their own prompt + image keys (`ask_memo.json`).
**Run.** `~/MscProj_r13`: `ARMS="SHIP3+DV SHIP8 SHIP8+BOX2" slurm/job_round16_dev.sh` (old DEV 49); then `~/MscProj_tg`:
`TG_ARMS="SHIP3+DV SHIP8 SHIP8+BOX2" slurm/job_tagger_arms.sh` (tagger dev2 22; merged DEV lines). Base must reproduce
SHIP8 merged 28/58 hits, 21 wrong (visible 6 / cross 13 / phantom 2), cost 2.282 at w = 2 (2.197 at w = 1); DEV 19/38, 14,
2.122; tagger DEV 9/20, 7, 2.636. cost(w) = (4 miss + w visible + 2 cross + 2 phantom) / 71.
**Main rule (ship).** GO iff merged cost at w = 2 is lower than SHIP8 AND merged hits >= 28 AND no needed hit is lost on either
part (clip-level: no clip on DEV or tagger DEV has fewer hits than under SHIP8).
**"More hits" rule (profile).** PASS iff merged hits go up AND every extra wrong picture is visible-type (cross and phantom
counts not above SHIP8, on each part) AND merged cost at w = 1 is lower than SHIP8.
Both verdicts are reported. GO on the main rule -> the flag is a shipping candidate (Adam decides; merged TEST stays sealed).
PASS only on the profile rule -> recorded as the "more hits" option for Adam. Neither -> closed.
### Round 39 DETACHED-ADD — a second picture where a drawn family is heard again, far from its picture (pre-registered 1 Oct, before numbers)
**Why.** Six merged-DEV misses are mistimed: the right family IS drawn in the clip, but at another moment (tg_d032 Thunder
2.8 & 7.4, picture 13.75; birds_forest Bird 1.3, picture 10.25; as_explosion Explosion 2.8, pictures 5.68 / 9.25; also
listed: tg_d120 Meow 2.9, ly_applause Crowd 1.9 — if SHIP8 draws no picture of that family there, the rule cannot touch
them by construction, which is reported, not patched). GBTP (round 38) moved pictures and lost a hit; this rule ADDS one
picture per family and clip instead, only where the family is heard again in a detached run.
**Rule (constants reused, none new).** For each clip and each family that SHIP8 already draws there with a NON-rescued
picture: take the family's evidence (max over same-family columns) in each of the three frame caches at the WEAK bars —
BEATs >= `config.AED_THRESHOLD` = 0.175 (half the display bar), FlexSED >= `FLEXSED_VETO` = 0.3 (the clip-veto bar),
DASM >= `config.LISTENER_DASM_BAR` = 0.575 (F8); runs = the pipeline's `_runs` with gaps <= `LISTEN_RUN_GAP` = 0.24 s
merged; a candidate run must be >= 0.24 s long. A run qualifies iff its onset s is > 3.0 s from the start of EVERY
same-family picture (rescued ones included) AND it is DETACHED from every same-family picture: between the run and the
picture (run end -> picture start if the run precedes, picture end -> run onset if it follows) there is a stretch
>= `MERGE_GAP` = 1.5 s where the family is below its weak bar in all three models (checked on a 0.04 s grid, nearest
frame per model; a missing cache counts as "below"). ONE picture per (clip, family): the earliest qualifying onset over
the three models; label = the existing picture's label, start = s, end = s + 2.0 s, appended to the placed list
(`_display_spans`/`_assign_rows` not re-run — an approximation, disclosed). Rescued pictures untouched; nothing in `src/`
or `config.py` edited.
**Screen.** `benchmark/gold/detached_add_screen.py` (copy of `gbtp_screen.py`'s loaders; CPU, `~/MscProj_tg`,
`TG_ARMS=SHIP8`), saved `SHIP8_proposed` pictures of merged DEV (71 clips), `score_per_sound`; base must reproduce 28/58,
21 (6/13/2), 2.282 (asserted). Reported: merged DEV per part; every added picture (part, clip, family, start, source
model(s), class from `cross_group.classify`, clip hits before -> after); dup counts separately (a dup is not "wrong").
**Pass.** Main rule: GO iff merged hits >= 29 AND wrong <= 21 + 2 x (hits - 28) AND cost at w = 2 (base 2.282) lower AND no
clip on either part has fewer hits than SHIP8. "More hits" rule: PASS iff merged hits > 28 AND cross and phantom counts not
above SHIP8 on each part (every extra wrong picture visible-type) AND cost at w = 1 (base 2.197) lower.
cost(w) = (4 miss + w visible + 2 cross + 2 phantom) / 71. GO -> a `src/` flag is Adam's decision; neither -> closed.
### Round 39 result — DETACHED-ADD: STOP (CPU, login node, `benchmark/gold/detached_add_screen.py` -> `detached_add_screen.json`)
Base reproduced: SHIP8 merged 28/58, 21 (6/13/2), 2.282; DEV 19/38, 14, 2.122; tagger DEV 9/20, 7, 2.636; all caches present.
34 (clip, family) pairs eligible; 8 pictures added (earliest from FlexSED 5, BEATs 2, DASM 1): as_explosion Explosion 0.12 (dup),
b3_bakery_morning Door 8.25 (cross), b3_botanic_garden Bird 23.24 (cross), mv_detective_crime_scene Alarm 17.68 (phantom),
un_driving_motorcycle Laughter 9.12 (cross), tg_d022 Dog 0.0 (cross), tg_d088 Thunder 7.08 (cross), tg_d128 Hammer 1.48 (cross).
**DETACHED-ADD: merged 28/58, 28 wrong (6/19/3), cost 2.479 | DEV 19/38, 18, 2.286 | tagger DEV 9/20, 10, 2.909 -> STOP on
both rules** (hits unchanged, +7 wrong, none visible-type; w = 1 2.197 -> 2.394). No needed hit lost. None of the targets
got a picture: tg_d032 Thunder — FlexSED (0.3) runs at 0.68 and 6.04 are far enough, but the family is never below the weak
bars for 1.5 s between them and the 13.75 picture (longest quiet stretch on the grid 1.36 s, ending 12.12); birds_forest Bird — FlexSED on
0.0-6.68 and 7.12-10.32, continuous into the picture; as_explosion — the true-onset run 2.84-3.84 is 2.84 s from the 5.68
picture (< 3.0), the 0.12 run is a dup; ly_applause Crowd — FlexSED on 0.0-13.56 (the 0.0 picture is 1.9 s early, no gap);
tg_d120 — the Cat picture is rescued (family not eligible) and DASM is on 0.56-5.64 anyway. Disclosed: DETACH = 1.5 is the shipped MERGE_GAP in `config.use_shipped()` (the task's named value); `arm_cfg("SHIP8")` merges pictures at 2.0, so the 1.5 was pinned in the script, not read from the arm. Reading: at the weak bars the
detectors hear the mistimed families almost continuously, so "heard again, detached" never fires on the targets and fires
only on repeat textures elsewhere (7 of 8 wrong). Closed.
### Round 39 CONTRAST — forced choice for the refused faint sounds (pre-registered 1 Oct, before any answer)
**Why.** A yes/no or an open inventory on a faint sound leans to "no" (nyc Air horn 3.8, tg_d107 Laughter 8.4, carnival
Whistle 6.1, as_explosion Footsteps 2.1 / Gasp 6.7, tg_d033 Siren, tg_d125 Clapping are all TIER-refused needed sounds). A
forced choice between the candidate's family and the strongest competing family in the same cut may rescue them.
**Rule.** For every merged-DEV P2/PV candidate (`~/MscProj_r13/.../dev_listener_v.json` 49 clips, `dev2_listener_v.json` 22
tagger clips) that the shipped TIER rule REJECTS (peak >= `TIER_SPLIT` 0.6: Qwen V4; below: Qwen V4 AND AF V4 from
`dev{,2}_listener_afn.json`; exactly `moss_screen.rules`): A = the candidate's family; B = the family with the highest raw
frame score inside the candidate's listener cut `run_audio` (the audio Qwen hears) over the FlexSED (`flexsed_cache`), BEATs
(`j2_dev{,2}_beats`) and DASM (`devcand/dasm_cache`, `dasm_dev2`) caches — max over the family's columns and the three
models, scores taken as stored (each in [0, 1], no recalibration) — excluding families related to A
(`listener_variants.related`: same canonical family, `same_family`, ontology descendant either way) and speech / music
(`SPEECH_LABELS` and their descendants, `is_music`, Singing and its descendants). Tie-break: equal score -> FlexSED before
BEATs before DASM -> alphabetical family name. No B (no cache file, nothing left) -> the item is not asked (`no_B`, reported).
Qwen3-Omni (same model, cut and greedy decoding as `listener_v4d`, no ducking, `max_new_tokens` 8) is asked on the cut
"Which sound is in this recording? (a) {A} (b) {B} (c) neither" and again with A and B swapped ((c) stays last). The answer is
the first "(a)" / "(b)" / "(c)" (or lone letter a / b / c) in the reply; unparsed = not A. Accept iff A is chosen in BOTH orders.
**Screen.** `benchmark/gold/listener_contrast.py` (GPU, `slurm/job_contrast.sh`, from `~/MscProj_tg`) writes
`dev{,2}_listener_contrast.json`; `benchmark/gold/contrast_screen.py` (CPU) counts, on the clips with gold
(`annotations/gold_AG.json`, `dev_listener.gold_class`), needed-class (hit_needed) and other-class (other_gold + none)
accepts ADDED on TIER-rejected items (nothing can be lost: TIER accepts are untouched). The TIER base must reproduce the
MOSS screen's 1004 items, 24 needed / 94 other TIER accepts. Reported: both counts, every needed-class accept with its B and
both raw answers, the seven known refused needed sounds with their B and answers, `no_B`.
**Pass.** GO to an arm (TIER's Qwen leg accepts on V4 OR CONTRAST) iff needed added >= 3 AND other added <= 2 x needed added.
Otherwise STOP, recorded, closed.
- **Round 38 E4 CF result:** base 16/41 seen silenced, 33/38 needed kept. CF_a (annotator's Q1/Q2 replace the majority)
  15/41, 32/38; CF_b (fourth vote) 16/41, 31/38. **STOP** both. The VLM's Q2 ("would a viewer know") is split in 30 of 145
  stretches; the bias control removes most of its signal.
### Round 39 MAKER-VIS — the gate asks about the depiction's MAKER, not the sound (pre-registered 1 Oct, before numbers)
**Why.** The shipped gate asks whether the SOUND's source is visible ("is Steam visible?"), but the picture's own depiction
already names a maker ("Train releases steam", "Parrot screaming", "Hammer striking"). If that maker is on screen the picture
only repeats what the viewer sees (visible-type wrong). This screen asks about the maker instead, label-free.
**Rule (fixed, no gold read).** For every placed picture of the saved SHIP8 arm on merged DEV (49 DEV + 22 tagger DEV2 = 71
clips, `btp_screen.parts()` with `TG_ARMS=SHIP8`; placed display spans incl. rescued ones, 50 pictures), the depiction =
the spec's `subject` (fallback: the text after "depiction:" in `reason`), matched to the placed span by label + start.
Maker = the grammatical subject by a fixed whitespace-token rule (`maker_vis_screen.maker_of`; no parser on the cluster):
all tokens before the first VERB token, where token i >= 1 is a verb if (i) it ends in "ing" (> 4 letters), or (ii) it ends
in "s" (not "ss") and the previous token is not plural, or (iii) it does not end in "s" and the previous token is plural
(plural = ends in "s" not "ss", or "people"). No verb -> the whole depiction. Lower-cased. Options: "{a/an} {maker} is
visible in these frames" / "no {maker} is visible"; plural head -> no article + "are"; mass set {water, sky, rain, steam,
snow, wind, fire, smoke, thunder, glass, traffic, people} -> no article. Stem: "These frames are from a video. Judge from
the frames alone." Frames: 6 at picture start -1 s .. +1 s (step 0.4 s, `_sample_frames_at`, as the gate). Asked by the
shipped gate VLM Qwen3.8-27B through `reason._ab` (both option orders); both orders "visible" -> picture dropped; "not
visible" or split -> kept. Frozen table (`makers` command, run before the GPU job): 50 pictures, all matched exactly;
makers: people x5, sky x3, bomb x4, machine gun x2, alarm x4, glass x3, train x3, crowd x2, clouds x2, man x2, person x2,
and bell, fire alarm, cupboard door, bird x2, rooster, emergency vehicle siren (no verb found), car x2, siren, fingers,
insect, dog, alarm clock, parrot, cat, water, hammer, bee.
**Screen.** `benchmark/gold/maker_vis_screen.py run` (GPU, `slurm/job_maker_vis.sh`, one JSON per picture with both raw
replies, resumable) then `score` (CPU), rescored with `score_per_sound` on `gold_AG.json`. Base must reproduce SHIP8 merged
28/58 hits, 21 wrong (6/13/2), cost 2.282 (asserted). Nothing in `src/` or `config.py` edited.
**Reported.** Per dropped picture: clip, label, depiction, maker, class before (`cross_group.classify`), rescued flag, clip
hits before -> after; counts visible / not visible / split; merged DEV per part.
**Pass.** Main rule: GO iff merged hits >= 28 AND no clip on either part has fewer hits than under SHIP8 AND cost at
visible weight w = 2 is lower than 2.282. Fewer-pictures clause: cost lower AND wrong <= 21 - 3 x (hits lost) AND hits
lost <= 3. GO -> a `src/` flag is Adam's decision; STOP -> recorded, closed.
### Round 38 result — E4 CF: STOP on both variants (H200, job 31598491, `benchmark/gold/cf_gate.py` -> `gate_gold/cf_summary.json`, votes in `gate_gold/cf_Qwen38-27B/`)
49 clips, 145 stretches. Base seen 16/41 silenced, needed 33/38 kept. **(a) CF alone: 15/41, 32/38 -> STOP. (b) fourth vote: 16/41,
31/38 -> STOP.** Q1 yes 30 / split 12; Q2 yes 46 / split 30; CF seen 53 stretches vs majority 54 — same volume, different sounds.
(a) flips 14 sounds: 4 needed newly silenced (as_explosion Gunshot 0.0 + Explosion 9.1, golf Whack 24.4, applause Crowd 1.9 —
all through Q2 "obvious" = yes while Q1 said no), 3 needed newly kept (bell_miami Bell: Q1 split x3; ambulance Vehicle; tornado
Siren), 4 seen newly silenced (bakery Crumpling, crossing Train, ia Water), 4 seen newly kept (Machine gun, Glass, aviary Bird
— one stretch splits — , tornado Cellphone). (b) flips only Gunshot 0.0 and golf Whack 24.4, both needed -> silenced. Reading:
the gold's Q2 asked of a VLM is the leak — it answers "a viewer would already know" for explosions, gunfire and applause whose
makers are off screen; Q1 alone is closer to the shipped gate but splits on bells. Closed.

### Found 1 Oct (measurement vs shipped mismatch)
The detector harness scores every arm with MERGE_GAP 2.0 (round13_dev BASE), but config.use_shipped() sets MERGE_GAP 1.5
(commit 8ffd06f, "join repeats within 1.5 s (reviewer)"). SHIP8 merged DEV: at 2.0 → 28/58, 21 wrong (6/13/2), 2.282; at
1.5 → 28/58, 23 wrong (7/14/2), 2.338. All DEV/TEST numbers today are at 2.0. Not changed overnight (display decision is
Adam's): either ship 2.0 (shipped = measured, −2 wrong) or re-score the chain at 1.5.

## Round 38 E1 SYNC — audio-visual synchrony (Synchformer) as a gate vote (written 2026-10-01 BEFORE any gold number)
**Motivation.** Every failed gate vote so far (SSL-SaN, PIC-SIM, GA, BOX pending) tested a semantic match between the
sound word and the frames. Synchrony is a different signal: a visible source that is MAKING the sound moves with it (a
pan dropping, a face laughing, lightning with thunder), an off-screen source over a static picture (bell over a tower,
birds over a still macaw) does not. Amendment 19 (prereg_v4) measured hand-rolled box-local frame difference and was
barely measurable (5/22); this round uses a trained model. Synchformer (Iashin et al. 2024, github v-iashin/Synchformer,
MIT, AudioSet checkpoint `24-01-04T16-39-21`, Acc@1 47.2 % / ±1 class 67.4 % on its own test set) predicts the audio-visual
offset of a 5-s window on a 21-class grid, −2 .. +2 s in 0.2 s steps (class 10 = 0 s).
**Wiring (checked on NON-gold clips before this entry, CPU, login node).** Repo cloned to `~/Synchformer`, run inside env
`msproj` (torch 2.5.1, transformers current) with three shims in `benchmark/gold/sync_gate.py`: stubs for the two removed
head-pruning helpers and `PreTrainedModel.get_head_mask` (never called at inference), `scripts/train_utils.py` loaded by
file path (our repo's `scripts/` package shadows the name). Controls, `smoke` subcommand: `ev_kitchen_pan_drop` (visible pan
drop) gives argmax −0.2 s with p 0.74 at offset 0, +0.8 s at an injected +1.0 s, −1.0 s at −1.0 s; `fx_people_clapping`
and `ambient_harbour_boats` give near-flat distributions (p max 0.06–0.11). The model runs and follows the offset.
**Score per stretch.** For every stretch of the cached shipped gate run (`gate_gold/Qwen38-27B`, all 139 cached gold
clips) the clip's 5-s window centred on the stretch (`(start+end)/2 − 2.5`, clamped to `[0, dur − 5]`; a clip shorter than
5 s is right-padded by cloning the last frame and silence — the demo has no padding rule, this is ours), cut and re-encoded
as the demo does (ffmpeg 25 fps, short side 256, even dims, 16 kHz mono; 5.5 s cut, the model's own test transform slices
exactly 125 frames / 80 000 samples from 0), scored at offset 0 with the test-time transform, half precision on GPU.
**sync = p(|offset| ≤ 0.2 s)** = the summed softmax of classes 9/10/11. Chosen before any gold number because the in-sync
control above peaks at −0.2 s (p(0) = 0.25, p(±0.2) = 0.996): a one-class encode latency must not count as off-screen.
p(0) alone is reported as a secondary (own calibrated t, same table, cannot change GO/STOP). The full 21-class vector and
argmax are stored per stretch; a failed window counts as sync < t. Sound-level sync = MIN over its stretches (the gate
silences only when every stretch is seen).
**Truth = CURRENT gold** (`gold_AG.json`; seen = visible or obvious; re-derived per sound by clip stem / resolved label /
start as `box_gate.gold_index`; the `seen` stored in the cache is not read). Importance ≥ 2 only.
**Calibration.** ONE threshold t = best balanced accuracy of "seen iff sync ≥ t" on the NON-judge cached clips (90 clips,
98 seen / 77 needed sounds, 246 stretches; highest t on ties). The DEV judge clips (JUDGE100 ∩ cached = 49 clips, 41 seen /
38 needed, 145 stretches) are never used to pick t.
**Variants on the DEV judge set, per stretch (clip verdict = every stretch seen):** (a) **vote4**: sync ≥ t is a fourth
vote next to name / a-b / desc; seen iff yes4 > no4; a 2–2 tie keeps the shipped majority decision. (b) **veto**: seen iff
the shipped majority says seen AND (sync ≥ t OR name = ab = desc = True — unanimous, `_ab` None counts as not unanimous).
**Base = shipped majority on current gold, computed before this entry: seen silenced 16/41, needed kept 33/38.**
**Pass (either variant): GO iff seen silenced ≥ 19 with needed kept ≥ 32, or needed kept ≥ 35 with seen silenced ≥ 15.**
Report: base, quartiles of the sound-level sync for seen vs needed sounds (DEV and calibration, plus the fraction of
stretches whose argmax is 0 s), calibration t and its balanced accuracy, both variants, every flip with gold class and
votes, GO/STOP. GO -> a `src/` flag is Adam's decision; STOP -> recorded, closed. Nothing in `src/` or `config.py` edited.
**Files:** `benchmark/gold/sync_gate.py` (`smoke` / `run` = GPU, one JSON per clip in `gate_gold/sync/`, resumable /
`score` = CPU, laptop, -> `gate_gold/sync_summary.json`), `slurm/job_sync_gate.sh` (H200-4h / A100-4h). Stop rule for
the setup: if Synchformer could not run within ~1 h, report why instead — it did (above).
- **Round 39 MAKER-VIS result:** merged 19/58, 15 wrong (4/10/1), cost 2.620 vs SHIP8 28/21/2.282. The maker is "visible"
  for 15 pictures, 9 of them hits (the VLM sees people, sky, clouds for needed off-screen sounds). **STOP.**
### Round 38 result — BOX-2 arm: STOP on both rules (jobs 31598526 DEV + 31598527 tagger DEV, H200)
Base reproduced (SHIP8 merged 28/58, 21 (6/13/2), 2.282). **SHIP8+BOX2: merged 29/58, 26 wrong (9/14/3), cost 2.366 at w = 2
(base 2.282), 2.239 at w = 1 (base 2.197)** | DEV 20/38, 17 (6/8/3), 2.163 (base 19/38, 14, 2.122) | tagger DEV 9/20, 9 (3/6/0),
2.818 (base 9/20, 7, 2.636). No clip lost a hit. Changes: + hit bell_miami; + visible as_glass_oHil9Ip_, b3_aviary_birds,
tg_d054; + cross tg_d085; + phantom b3_construction_site. Main rule: cost up -> STOP. "More hits": hits +1 but the extra wrong
pictures include 1 cross + 1 phantom and w = 1 cost is up -> FAIL. mv_tornado Siren (gained in the screen) did not turn into a hit. Closed.

## Round 39 RELABEL-GATE — listener relabel of a placed picture, then the shipped visibility gate on the new name (written 2026-10-01 BEFORE any number)
**Motivation.** Of SHIP8's 21 wrong pictures, 7 draw the wrong sound at a real moment (Steam for Train, Dog for Chopping, Glass
for Baby cry, Hammer for Clang, Screaming for Bird, Explosion for Thunder, Bird during footsteps); in most, the real sound there
is a SEEN gold sound. If the two open-inventory listeners agree on what the moment really is, the picture can be renamed, and the
shipped gate can then silence it when its true maker is on screen. Known before this entry: the round-31 listing
(`benchmark/gold/cross_group.json`, SHIP5 rows, same stage-4 cuts) is on disk and was read; on its P1 lists the rule below does not
fire on any of the 7 targets (both listeners name A for Steam and d088 Explosion; qwen and AF disjoint for Glass, Hammer,
Screaming; no cut for d022 Dog; golf Bird is rescued). The rule is kept as given — it runs over EVERY placed non-rescued SHIP8
picture (hits included), re-derived from `SHIP8|proposed`, so the score can still move; it is not relaxed to make it fire.
**Rule (CPU, saved SHIP8 pictures, merged DEV, gold `gold_AG.json`, display at `arm_cfg` MERGE_GAP 2.0 as every number today).**
For each placed NON-rescued picture of family A (`btp_screen.parts`, `canonical(label)`):
1. P1 cut = the stage-4 rows of `SHIP8|proposed` with `canonical == A` overlapping [start−0.1, end+0.1] (cross_group's match);
   each row's `(qwen_fams, af_fams)` via `_p1v4_lists` on `{dev,dev2}_listener_p1v4.json`; of the rows with a non-None answer,
   the one whose start is nearest the picture start. No answered row → untouched.
2. "names A" = any listed family with `same_family(x, A)`. Candidates B = families in qwen_fams ∩ af_fams with not
   `same_family(B, A)`, B in `depictable_vocab.json["families"]` (Speech/Music absent), `_specific(B)`. Relabel to B iff
   candidates exist AND at most one listener names A; B = the first candidate in `qwen_fams` order (its prompt lists most
   prominent first). Several candidates → the first one, the rest ignored.
3. Gate on B at the picture: cached iff `gate_gold/Qwen38-27B/<stem>.json` has a gold sound with `same_family(B, label)` and
   picture start in [g.start−0.5, g.end]; verdict = yes > no over name/ab/desc in every stretch overlapping [start−1, start+1]
   (none overlapping → every stretch of the sound); seen only if every such stretch is seen. The cached `seen` fields are not
   read. Not cached (all dev2 `tg_*` clips, no cache) → live gate: `reason._sound_is_visible(B, 6 frames at start−1 … start+1,
   Qwen3.8-27B)` as `gate_gold.run_vlm`, one stretch, same majority; a GPU job (H200-4h/A100-4h, 128G) only for exactly those.
4. Seen → the picture is dropped; not seen → kept, renamed B, same span.
**Score.** `score_per_sound.score_clip` per clip as `dbr_screen`; base must reproduce 28/58, 21 (6/13/2), 2.282 first.
**Pass** (dbr_screen `passes`): main rule — hits ≥ 28, no needed hit lost on either part, cost (visible weight 2) < 2.282;
fewer-pictures clause — cost lower AND wrong ≤ 21 − 3 × (hits lost) AND hits lost ≤ 3. GO → a `src/` flag is Adam's decision;
STOP → recorded, closed. Nothing in `src/` or `config.py` edited.
**Files:** `benchmark/gold/relabel_gate_screen.py` (`scan` = CPU: relabels + cached gates + which need GPU, `gate` = GPU for the
rest, `score` = CPU) → `benchmark/gold/relabel_gate_screen.json`; `slurm/job_relabel_gate.sh` only if a live gate is needed.
### Round 39 result — CONTRAST: STOP (job 31598550, H200, 12 min; `contrast_screen.json`)
Base: 1004 P2/PV candidates on gold clips (602 DEV + 402 tagger DEV2), 118 TIER accepts (26 needed / 92 other under the
current `gold_AG.json`; the MOSS screen had the same 118 as 24 / 94 — the assert was relaxed to the total, no answer changed).
886 TIER-rejected items asked, `no_B` 0, unparsed 0. Order-consistent answers: B in both orders 413, A in both 143, neither
in both 177, order-dependent 153. **Needed added 1, other added 142 -> STOP** (bar: >= 3 with <= 2x other).
The one needed accept: tg_d107 Crying, sobbing 0.04-0.44 (B Laughter, FlexSED 0.86; "(a) Crying" / "(b) Crying").
The seven known refused needed sounds: nyc Air horn 3.8 (B Sliding door; "neither" / Sliding door), as_explosion Footsteps 2.1
(B Gunshot both orders), as_explosion Gasp 6.7 (B Explosion / Gasp: order-dependent), carnival Whistle 6.3 (B Train both
orders), tg_d107 Laughter 8.4 (B Screaming both orders), tg_d125 Clapping 8.3 (TIER-accepted: not asked), tg_d033 Siren (no P2/PV
candidate: not in the pool). Reading: the forced choice does not lean to "neither" — it names the louder competitor
(B wins 413 of 886), so the faint target is still refused; the 143 A-wins are mostly wrong or already-drawn sounds
(142 other-class). Closed.
- **Round 39 CONTRAST result:** the base assertion (24/94, from the MOSS screen) fails only because the gold was corrected
  tonight (TIER base now 26 needed / 92 other on 1004 items). Forced choice on 886 refused candidates: **+1 needed, +142
  other → STOP** (bar ≥ 3 needed with ≤ 2× other).
### Round 38 result — E1 SYNC: STOP (H200, job 31598563, `benchmark/gold/sync_gate.py` -> `gate_gold/sync/`, `sync_summary.json`)
391 windows, 0 failed. Calibration (175 non-judge sounds): t = 0.761, seen silenced 46/98, needed kept 58/77, balanced 0.611
(secondary p0: t = 0.122, balanced 0.604). Sound-level sync quartiles (q1 / median / q3): calibration seen 0.11 / 0.61 / 0.98 vs
needed 0.09 / 0.19 / 0.79; DEV seen 0.11 / 0.23 / 0.96 vs needed 0.06 / 0.11 / 0.19; argmax = 0 s on 28 % (cal) / 21 % (DEV) of seen
stretches vs 16 % / 11 % of needed. A real but weak signal: a confident in-sync peak (sync > 0.9) is near-specific to seen
sources (bakery crumpling / frying / tap, kitchen water / tap, glass), but most seen sources are static and flat
(rain, waves, waterfall, aviary birds, motorcycles).
**DEV judge (base 16/41 silenced, 33/38 kept): (a) vote4 16/41, 33/38, 0 flips; (b) veto 11/41, 36/38, 8 flips; sync alone
13/41, 36/38 -> STOP.** Veto rescues the three target misses (bell_miami Bell, b3_golf_course Whack, ly_ambulance Vehicle:
all 2-1 votes with sync 0.02-0.23) but un-silences five seen sounds with flat sync (marrakech Motorcycle, as_explosion Machine
gun, as_glass Glass 18.8, b3_aviary Bird, mv_tornado Cellphone buzz). Structural note, stated after the fact: variant (a) as
written cannot flip anything — a fourth vote next to three can only make a 2-2 tie, which the rule returns to the shipped
majority; it is identical to the base by construction (the same holds for SSL-SaN's vote4 rule, Round 33). Closed; no `src/` change.
### Round 39 RELABEL-GATE result — STOP (`benchmark/gold/relabel_gate_screen.json`; live gate job 31598609, H200)
50 placed pictures: 8 rescued, 13 no answered P1 cut, 19 both listeners name A, 7 no shared candidate, **3 relabelled** — none of
the 7 targets (as foreseen above). (1) ambient_weather_storm_7200 Thunder → Rain @0.06 (cached gate: Rain 0.1–15.8 seen 3/3):
visible → dropped. (2) ly_applause_62ZYD0u Crowd → Baby laughter @0.0 (cached: Laughter 0.0–13.8 seen 3/3): cross → dropped.
(3) tg_d088 Thunder → Rain @1.0 (live gate: seen 3/3, "a brown and white cow stands in the rain"): **hit → dropped**. Merged
27/58, 19 wrong (5/12/2), cost 2.2817 = base 2.2817 (not lower); DEV 19/38, 12 (3/7/2), 2.041; DEV2 8/20, 7 (2/5/0), 2.818.
Main rule fails (hit lost, cost not lower); fewer-pictures clause fails (19 > 21 − 3). (The job log's "GO (fewer-pictures clause)" line came from the script's first version importing `passes` from dbr_screen with its SHIP7 base 2.366; fixed in 40c0c42 and re-scored on CPU before this number.) Reading: when the listeners agree on
another sound it is a seen texture (rain, laughter) under the detector's event, and the gate then silences a true hit with it.
Closed.

## Round 40 EXPECT — the scene proposes likely off-screen sounds, the ears confirm (written 2026-10-01 BEFORE any number)
**Motivation.** 12 needed SHIP8 misses have no detector run at all and 2 sit below the band floor (`ledger_ship8.json`: nyc_1689
Hammer, favela Train, rainforest_2179 Bird, tg_d033 Siren, tg_d029 Chicken/rooster, ...). A detector cannot be pushed lower
without a flood; instead the scene VLM is asked what one would EXPECT to hear off-screen in this place, and the open-inventory
listener must independently name the same family. Known before this entry: nothing — no VLM proposal or Omni whole-clip answer
exists for these clips; the DASM/BEATs/FlexSED caches exist and were read by earlier screens (DETACHED-ADD).
**Rule (merged DEV = 49 DEV + 22 tagger DEV2 clips, saved SHIP8 pictures, gold `gold_AG.json`, display at MERGE_GAP 2.0 as
every number today).** Per clip:
1. **Propose** (GPU, Qwen3.8-27B via `reason._load/_ask`, the shipped gate VLM): 8 frames at `dur * (i + 0.5) / 8`, i = 0..7
   (`media_duration`, `_sample_frames_at`). One question (verbatim in `expect_screen.PROPOSE_Q`): name at most 5 sound sources
   one would likely HEAR in this scene but whose maker is NOT visible in the frames, answered as names copied from the provided
   list = `depictable_vocab.json["families"]` (215 names, Speech and Music absent), JSON list only. Parse: each answer matched
   case-insensitively to a listed family; else `canonical()` of a depictable label; else discarded (counted). First 5 kept, deduped.
2. **Listen** (GPU, Qwen3-Omni `listener_round.MODEL`): the WHOLE clip wav (dev `data/work/devcand/wav16`, dev2
   `data/work/r13dev2/wav16`), the unchanged V4 open question `listener_variants.V4_Q`, greedy, `V4_NEW` = 64 tokens exactly as
   `listener_v4d.gen` (64 tokens may truncate a long list — accepted, disclosed).
3. **Candidate** (CPU): a VLM family F is "named by both" iff `listener_afnext.v4_match(O, emb, v4_text, F)` is True (the shipped
   V4 matcher: a listed name word-matches a line, or mpnet cosine > 0.6). F is "already drawn" iff any placed SHIP8 picture in the
   clip (rescued included) has `canonical(label) == F` or `same_family(label, F)`; such F is skipped. One candidate per (clip, F).
   Onset = the smallest run start over the three frame caches (`gbtp_screen.caches`: BEATs >= 0.175, FlexSED >= 0.3, DASM >= 0.575;
   family evidence = max over same-family columns; `_runs` with LISTEN_RUN_GAP 0.24 s; **no minimum run length**). No run in any
   cache -> no picture (counted, with the per-model cache-missing count).
4. **Gate** (GPU, same VLM): `reason._sound_is_visible(F, 6 frames at onset − 1 + 0.4 k, k = 0..5)` under the shipped majority rule
   (`VISIBILITY_RULE` = majority in SHIP8), exactly as `gate_gold.run_vlm`. Seen -> dropped; else a picture (F, onset, onset + 2 s)
   is ADDED to the clip's SHIP8 pictures.
**Score.** `score_per_sound.score_clip` per clip as `dbr_screen` / `detached_add_screen`; base must reproduce 28/58, 21 (6/13/2),
2.282 first. Cost at visible weight w via `gbtp_screen.cost_w` (w = 2 is the shipped cost).
**Pass.** Main rule: hits >= 28 AND no needed hit lost on either part AND cost(w = 2) < 2.282. "More hits" rule: hits > 28 AND
cross and phantom not up on either part (every extra wrong picture visible-type) AND cost(w = 1) < base cost(w = 1). GO on either ->
a `src/` flag is Adam's decision; STOP -> recorded, closed. Nothing in `src/` or `config.py` edited.
**Report.** Per target clip: VLM list, Omni lines, intersection, candidate onset (which cache), gate votes, outcome; totals; verdicts.
Noted in advance: birds_forest Bird, tg_d032 Thunder, tg_d120 Meow/Cat, ly_applause Crowd, as_explosion Explosion are excluded by
the "already drawn" clause by design (a same-family picture exists); this round cannot fix those.
**Files:** `benchmark/gold/expect_screen.py` (`propose` / `listen` / `gate` = GPU, `cands` / `score` = CPU, one JSON per clip per
stage under `benchmark/gold/expect/`, resumable) -> `benchmark/gold/expect_screen.json`; `slurm/job_expect.sh` (H200-4h/A100-4h,
one job chaining the five, models unloaded between stages).

## Round 38 SYNC-2 — Synchformer sync as an ADD-seen rule (written 2026-10-01 BEFORE any SYNC-2 number)
**Disclosure.** The rule SHAPE was suggested by the Round 38 DEV flips (every "good" not-seen -> seen flip of plain sync had
sync >= 0.88, the one bad one 0.81), so the DEV judge set has already informed the idea; the threshold therefore comes from the
NON-judge cached clips only (90 clips, 98 seen / 77 needed sounds), and DEV is read once. Same truth (current gold, seen =
visible or obvious, re-derived per sound), same base (16/41 silenced, 33/38 kept), same GO bar (silenced >= 19 & kept >= 32,
or kept >= 35 & silenced >= 15), same score path (`sync_gate.py score2`, CPU, laptop).
**Rule (1) ADD-seen.** A sound the shipped majority keeps (not seen) becomes seen iff its sound-level sync (MIN over stretches,
p(|offset| <= 0.2 s) as in Round 38) >= t_hi, where t_hi = the smallest observed non-judge sync value at which >= 95 % of the
non-judge sounds with sync >= t_hi are gold-seen (if no value reaches 0.95 the rule is void and reported as such). Sounds the
majority already silences are unchanged.
**Rule (2)** = Round 38 (b) veto, unchanged (seen iff majority AND (sync >= t OR unanimous), t = 0.761 from Round 38's
calibration). Reported: (1) alone and (1)+(2) (= veto OR add). Every flip listed with gold class and votes.
**Pictures.** From saved SHIP8 data (`btp_screen.parts`, `TG_ARMS=SHIP8`, DEV part only; the gate cache covers no tg_* clip)
every placed picture whose `cross_group.classify` match is a gold sound that flips under (1) or (1)+(2) is listed with its
class (hit / visible / collision / dontcare): a flip to seen would silence it, a flip to not-seen would let it through.
Report only; the arm is not re-run.
### Round 38 SYNC-2 result — (1) ADD-seen alone: GO on the bar; (1)+(2): STOP (CPU, `sync_gate.py score2` -> `gate_gold/sync2_summary.json`)
t_hi = 0.9976 (precision 1.00 on the non-judge set: all 5 of 175 sounds with sync >= t_hi are gold-seen — a thin calibration).
**DEV judge (base 16/41 silenced, 33/38 kept): (1) add 19/41, 33/38, 3 flips, all correct** (b3_bakery_morning Crumpling 14.1
sync 0.999 votes 0-3; b3_ia_youtube_skxtz9foauw_0 Water 24.6 and Tap 27.3, sync 0.9995, votes 1-2) **= GO, exactly on the bar
(silenced >= 19 with kept >= 32)**. (1)+(2) add+veto 14/41, 36/38, 11 flips (the 3 above + Round 38's veto flips: 3 needed rescued,
5 seen lost) = STOP. Pictures: on the cluster (`~/MscProj_tg`, SHIP8_proposed, 34 DEV pictures) NO SHIP8 picture is matched to
any flipped sound under either variant — the three newly silenced sounds were never drawn, and the three veto rescues (bell_miami
Bell, golf Whack, ambulance Vehicle 7.3) have no SHIP8 picture either (ambulance's is Siren 0.0, golf's Bird 18.8), so the arm's
hits / wrong would not change on DEV. Reading: a confident in-sync peak is a precise but rare "on screen" signal (5/175 on the
calibration set, 3 of 25 kept-but-seen sounds on DEV); the GO is a 3-sound gain on the bar with a threshold set by 5 sounds, and
it changes no shipped picture. Adoption as a `src/` flag is Adam's decision; the shape disclosure above applies.
**Round 40 EXPECT amendment 1 (2026-10-01, written while the GPU job ran stage 1, BEFORE any candidate, gate or score).** The VLM
sometimes ignores "JSON list only" and answers prose with bold family names (seen on the 2nd clip, nyc_2627: a numbered list
"**Vehicle**: ..."), which the parse rule of step 1 discards wholesale. Fallback adopted: when the reply contains no JSON list,
the families are the depictable family names that appear as whole words (case-insensitive) anywhere in the reply, in order of
first appearance, first 5. Replies are stored raw, so `cands` re-parses every clip from the stored reply with this rule; the GPU
proposal stage is not re-run. No other change.

## Round 41 AGREE — the SYNC veto and BOX-2 must BOTH flip (written 2026-10-01 BEFORE any AGREE number)
**Disclosure.** The shape ("flip a seen stretch to not-seen only when two independent checks agree") was chosen AFTER reading
the Round 38 (b) veto flip list (`gate_gold/sync_summary.json`, 8 flips: 3 needed rescued, 5 seen lost) and the Round 38
BOX-2 flip list (`gate_gold/box2_summary.json`, 5 flips: 2 needed rescued, 3 seen lost). At SOUND level those lists share
only bell_miami Bell (needed, good) and mv_tornado_scene Cellphone buzz (seen, bad), so the sound-level intersection is
already known to be at most 15/41 silenced, 34/38 kept — below both GO bars. The screen computes the exact STRETCH-level
rule (which can only agree on fewer stretches, never more) and records it; a STOP here is a confirmation of that bound, not
a fresh finding. DEV judge clips (49) only, since the box cache covers only them.
**Rule.** Per stretch: shipped majority seen (name/ab/desc, yes > no, `gate_gold/Qwen38-27B`); veto flip (Round 38 (b)) iff
majority AND NOT unanimous AND (sync < t = 0.76131 or the sync window failed) (`gate_gold/sync/<stem>.json`, key p_pm02);
BOX-2 flip iff `box_gate.seen_box(st, "BOX2")` is False while `box_gate.majority(st)` is True (parsed box, both crop checks
"no"; the strict re-ask of an unparsed box counts; no box record -> no flip) (`gate_gold/box_Qwen38-27B/<stem>.json`).
AGREE: stretch seen iff majority AND NOT (veto flip AND BOX-2 flip). Records are matched by (clip stem, label, sound start,
stretch start). Sound seen iff every stretch seen (the pipeline rule). Truth = current gold (`gold_AG.json`, seen = visible
or obvious, importance >= 2), base 16/41 seen silenced, 33/38 needed kept, which the script must reproduce first.
**GO** iff silenced >= 19 & kept >= 32, or kept >= 35 & silenced >= 15 (the Round 38 bar). Every flip listed with gold class,
votes, sync and box status. If GO: the SHIP8 misses (saved pictures, `~/MscProj_tg` -> local `data/work/r13*/SHIP8_proposed`,
`cross_group.classify`) whose gold sound flips to not-seen are listed as the sounds a full-pipeline arm could draw; the arm
itself is not built here. **Files:** `benchmark/gold/agree_screen.py` -> `gate_gold/agree_summary.json`. CPU, laptop.

## Round 41 NAMED-VETO — drop a placed picture when the gate names another maker (written 2026-10-01 BEFORE any number)
**Idea.** Every cached gate stretch stores `named` = the VLM's answer to "what makes this sound" (an object noun: "church
bell", "black suv", "chainsaw", "nothing"). A placed SHIP8 picture whose onset falls inside a stretch where the VLM says a
DIFFERENT depictable thing makes the sound is likely a wrong-family picture (cross-trigger) and is dropped.
**Canonicalisation of `named`** (declared here; tuned by eye on the 118 distinct `named` strings of the cache — NOT on any
picture or score): (1) empty / "nothing" -> no family. (2) `score_per_sound.ALIASES` on the whole lower-cased text, then a
small declared noun table for things AudioSet hides or lacks: locomotive, tram, streetcar -> Train; taxi, suv, sedan -> Car;
van, pickup -> Truck; ambulance -> Ambulance (siren); rifle, shotgun, ak-47, gun -> Gunshot, gunfire; tank main gun, tank ->
Artillery fire; cellphone, phone -> Cellphone buzz, vibrating alert; fountain, waves -> Water; cockpit -> Helicopter;
loudspeaker -> Loudspeaker. (3) Otherwise every ontology name (or a comma part of one) that appears as a WHOLE word or phrase in
the text (singular/plural tolerant); among matches a THING name (not in `src.labels.ACTION_LABELS`) beats an action word,
then the longest wins — so "breaking waves" -> Waves, surf (via the table), "cracking glass" -> Glass, "steam locomotive" ->
Train (table), never "yellow taxi" -> Yell (the substring bug of `resolve_label` on free text). (4) The result must be
depictable: `src.labels.is_salient_nonspeech` under `LABEL_FILTER = "depictable"` (set explicitly; `DISPLAY_KEYS` does not
carry it); people ("man", "woman", "the boy") and unresolved nouns are NOT depictable -> the rule does not fire.
**Rule (primary).** For a placed SHIP8 picture (label L, scored placed start a, from `btp_screen.placed`, `TG_ARMS=SHIP8`),
the covering stretches are every cached stretch of that clip with start <= a <= end (any gold sound). The picture is DROPPED
iff at least one covering stretch names a depictable family F with NOT `score_per_sound.same_family(L, F)` AND no covering
stretch names a family that IS same_family with L (the VLM naming the picture's own maker protects it). No covering stretch,
or only "nothing"/unresolved/non-depictable names -> the rule does not fire (counted and reported: the cache holds only
gold-sound stretches, so pictures at other times, and every tg_* (dev2) clip, cannot be judged). Secondary, report only:
the plain variant (any covering stretch naming a different depictable family drops, no protection).
**Score.** `score_per_sound.score_clip` per clip on the kept pictures, both parts, as `dbr_screen`; base must reproduce
28/58 hits, 21 wrong (6/13/2), cost 2.282 first. **Verdict** = `dbr_screen.passes` with BASE = SHIP8: main rule (hits >= 28,
no needed hit lost on either part, wrong <= 21 + 2*max(gain,0), cost(w = 2) < 2.282) or fewer-pictures clause (cost lower,
wrong <= 21 - 3 * hits lost, hits lost <= 3). Every dropped picture listed with its class, the covering stretches and their
`named`. **Files:** `benchmark/gold/named_veto_screen.py` -> `benchmark/gold/named_veto_screen.json`. CPU, laptop; nothing in
`src/` or `config.py` edited.

## Round 41 PRIOR415 — an independent per-family precision prior from the held-out 415 (written 2026-10-01 BEFORE any number)
**Motivation.** 13 of SHIP8's 21 wrong pictures are cross (another sound was there) and the detectors' per-family precision
is unknown for merged DEV; the held-out 415 AudioSet-Strong clips (`benchmark/gold/audioset_heldout.json`, strong labels,
disjoint from DEV/TEST, caches present: FlexSED `~/MscProj_tg/data/work/flexsed_heldout`, BEATs
`~/MscProj/benchmark/audioset_heldout_windows/beats`, 415 each) give one per family and per model without touching gold.
Known before this entry: the SHIP8 base (28/58, 21 (6/13/2), 2.282) and the ledger of its misses; no per-family precision
of either model on the 415 has been read.
**Step 1 (frozen BEFORE any DEV read; `benchmark/gold/prior415_screen.py prior` -> `benchmark/gold/prior415_lists.json`,
committed before step 2).** Clips = the 415 with a cache file for the model (count reported per model). Instances =
strong-label EVENTS (masked included, Speech/Music included — they are never drawn), family = `src.labels.canonical(label)`.
Families with >= 10 instances are scored. FlexSED runs = `_extract_events(ffw, ftimes, flabels, 0.8, None, 0.5, low=0.8)`,
the `fuse_flexsed` line under the shipped flags (`config.use_shipped()`; asserted and dumped: `FLEXSED_FAMILY_BARS None`,
`IMPULSE_MIN_SPAN None`, `AED_HYSTERESIS 1.0`, `AED_MIN_DUR 0.5`). BEATs spans = `_extract_events(fw, times, labels, 0.35,
None, 0.5, low=0.175)` = extraction at `AED_THRESHOLD` 0.175 keeping peak >= 0.35 (= `audioset_detector_eval._spans`
at the display bar). Both are the RAW extractions: before ONSET_CAM, the FlexSED/PANNs/mirror vetoes and the listener,
which need audio and models (disclosed). A run of family F = `canonical(run.label)` is correct iff some event with
`canonical(event.label) == F` has `start` in [run.start − 0.5, run.start + 1.0] (family equality only, no descendant
matching). precision_F = correct / runs. Lists per model: UNRELIABLE = precision < 0.3, RELIABLE = precision > 0.8;
a scored family with 0 runs is in neither (undefined). No minimum run count (the task's rule as given); `n_runs`,
`n_correct`, `n_instances` are recorded per family and every list member with `n_runs < 5` is flagged in the report.
**Step 2 rule (`prior415_screen.py screen`, CPU, `TG_ARMS=SHIP8`, from `~/MscProj_tg`).** Saved SHIP8 pictures of merged
DEV (`btp_screen.parts`, roots from `cross_group.PARTS`), gold `gold_AG.json`, display at MERGE_GAP 2.0; base must
reproduce 28/58, 21 (6/13/2), cost 2.282 (asserted, and `gbtp_screen.cost_w(w=2)` asserted equal to it). Per placed
picture (label, a, b): rescued -> untouched. Contributing rows = `stage4.json["arms"]["SHIP8|proposed"]` rows of the clip
with `canonical(label) == canonical(picture label)` overlapping [a − 0.1, b + 0.1] (cross_group's criterion); no row ->
untouched (counted). Model = `origin` of the row whose start is nearest a (`tagger` = BEATs, `flex` = FlexSED; a BEATs
row that absorbed a FlexSED twin under TWIN_MAX carries `tagger` — disclosed). The picture is DROPPED iff its family is in
that model's UNRELIABLE list AND no listener names its family on the P1 cut of any contributing row: names =
`_p1v4_lists(row)` (`config.RELABEL_P1V4` = `<part>_listener_p1v4.json`, `_CURRENT_CLIP` = clip) over `qwen_fams ∪
af_fams`, named iff `canonical(x) == family` for some x; a row with no P1 item = not named (P1 items exist only for
FlexSED 0.5–0.8 candidates, so BEATs pictures are rarely covered — the counts "no P1 item" vs "P1 item, not named" are
reported). Scored per clip with `score_per_sound.score_clip` as `dbr_screen`; hit loss per part = any clip with fewer
hits than base. **Verdict** = `passes` COPIED into the script with BASE = {28, 21, 2.282} (never imported from
`dbr_screen`, whose BASE is SHIP7): main rule (hits >= 28, no needed hit lost on either part, wrong <= 21 + 2·max(gain, 0),
cost(w = 2) < 2.282) or fewer-pictures clause (cost lower, wrong <= 21 − 3·hits lost, hits lost <= 3). GO -> a `src/`
flag is Adam's decision; STOP -> closed. **RELIABLE, list only, not scored:** every needed SHIP8 miss in
`ledger_ship8.json` whose family is RELIABLE for a model that has a raw run (same extraction as step 1, DEV caches
`cross_group.PARTS[part]["beats"]`, `btp_screen.FLEX_DIR`) with start within [onset − 0.5, onset + 1.0] of the miss
(e.g. a FlexSED clip-veto skip for tg_d029 Chicken) — reported as what RELIABLE would allow. Report: the frozen lists,
every dropped picture with its class before and the clip's counts after, totals, verdict. Nothing in `src/` or
`config.py` edited. **Files:** `benchmark/gold/prior415_screen.py`, `prior415_lists.json` (frozen), `prior415_screen.json`.
### Round 41 AGREE result — STOP (CPU, `agree_screen.py` -> `gate_gold/agree_summary.json`)
Base reproduced (16/41, 33/38; 145 stretches, all with a box record). Stretch flips: veto 17, BOX-2 7, both 4. **AGREE:
15/41 seen silenced, 34/38 needed kept, 2 sound flips** — exactly the sound-level bound disclosed above: bell_miami Bell
0.2 s (needed, good: votes 2-1 x3, sync 0.13/0.12/0.05, box none/box/box) and mv_tornado_scene Cellphone buzz 1.4 s (seen,
BAD: votes 2-0, sync 0.014, box parsed + crops "no"). Neither bar reached (needs 19 & 32, or 35 & 15). The two checks fail
on different seen sounds (sync on the 5 static-looking seen sources, BOX-2 on glass / motorcycle), so requiring both removes
most of the damage but also 2 of the 3 rescues (golf whack, ambulance vehicle: no BOX-2 flip). No SHIP8 picture question
arises (STOP).
### Round 41 NAMED-VETO result — STOP (CPU, `named_veto_screen.py` -> `named_veto_screen.json`)
Base reproduced (28/58, 21 (6/13/2), 2.282). Of the 50 placed SHIP8 pictures: 16 are on dev2 (tg_*) clips with no gate
cache, 5 fall outside every cached stretch, 23 are covered only by stretches naming nothing / people / an unresolved noun,
**6 could be judged and all 6 fired** (the "own family named" protection never applied, so primary = secondary): 3 visible
pictures dropped (storm Thunder over "heavy rain", london_protest Vehicle over "crowd of people", motorcycle Explosion over
"crowd of spectators"), 1 cross dropped (crossing_bells Steam over "train"), and **2 HITS lost** (snow_walk Laughter 8.1 s
over "tram", protest Glass 16.9 s over "crowd of people"). **Merged 26/58, 17 wrong (3/12/2), cost 2.282** (unchanged to 3
d.p.: 2 hits lost against 4 wrong removed at visible weight 2). Main rule fails (hits < 28, needed hits lost on DEV);
fewer-pictures clause fails (cost not lower; 17 > 21 - 3*2). Reading: the gate names the maker of the GOLD sound it was
asked about, which is a different sound from the picture's whenever two sounds overlap — a crowd on screen does not make a
picture of breaking glass wrong. The rule is a sound-overlap detector, not a wrong-family detector. Closed.
### Round 41 PRIOR415 result — STOP (no picture dropped; `benchmark/gold/prior415_lists.json`, `prior415_screen.json`, CPU)
**Step 1 (frozen, commit 31d4ff4).** 415 clips cached for both models, 70 families with >= 10 strong events. FlexSED (0.8 runs,
only 14 families fire at all): UNRELIABLE Groan, Rattle, Shout, Typewriter (1-2 runs each, thin), Vehicle (2/9 = 0.22);
RELIABLE Explosion (1/1, thin), Power tool (5/5). BEATs (0.35 spans): UNRELIABLE 24 families, among the non-thin ones Animal
0.23, Cat 0.29, Dishes 0.29, Domestic animals 0.14, Explosion 0.14, Glass 0.13, Gunshot 0.24, Mechanisms 0/5, Livestock 0/5,
Silence 0/16, Singing 0.19, Speech 0.13, Telephone 0/7, Tools 0.13, Water 0.23 (+ 9 thin: Female singing, Finger snapping,
Footsteps, Groan, Narration, Screaming, Tick, Whoop, Whoosh); RELIABLE Cough, Male speech, Sound effect, Spray, Wind noise
(all thin, 1-7 runs). Vehicle (BEATs 0.36), Bird (0.47), Train (0.53), Dog (0.72), Alarm (0.73), Music (0.67) are in neither.
**Step 2.** Base reproduced (28/58, 21 (6/13/2), 2.282; cost_w(2) = viewer cost, per clip). 50 placed pictures: 8 rescued,
26 BEATs + 8 FlexSED pictures of a non-UNRELIABLE family, **8 pictures of an UNRELIABLE family (all BEATs) and every one is
named by a listener on a P1 cut** (the P1 caches do cover BEATs rows, contrary to the expectation written above): as_explosion
Gunshot 8.25 (cross) and Explosion 9.25 (hit), mv_protest Glass 4.75 (cross), 10.75 (hit), 16.89 (hit), tg_d088 Explosion
10.75 (cross), tg_d107 Screaming 7.0 (cross), tg_d127 Water 0.14 (visible). Dropped 0 -> 28/58, 21, 2.282 = base: main rule
fails (cost not lower), fewer-pictures fails. **STOP.** Without the listener exception the 8 would go: 3 hits lost, 5 wrong
fewer (25/58, 16 wrong) — fails both rules too (wrong 16 > 21 − 9). RELIABLE would allow nothing: of the 30 needed SHIP8
misses only the two Explosions (as_explosion 2.8, tg_d125 5.4) have a RELIABLE family (FlexSED), and neither has a raw 0.8
FlexSED run at its onset. Reading: the held-out precision prior separates families the listener already arbitrates; the
cross pictures it flags (Gunshot/Glass/Explosion/Screaming) are heard by both listeners, so the error is placement or
kinship, not the family. Closed.
### Round 40 result — EXPECT: STOP on both rules (jobs 31598632 propose + 31598677 listen/cands/gate/score, H200)
Base reproduced (28/58, 21 (6/13/2), 2.282). **SHIP8+EXPECT: merged 29/58, 26 wrong (8/15/3), cost 2.366 at w = 2 (base 2.282),
2.254 at w = 1 (base 2.197)** | DEV 20/38, 17 (5/9/3) | tagger DEV2 9/20, 9 (3/6/0). No hit lost. Funnel: 177 proposals (33/71
replies were JSON lists, the rest parsed by amendment 1) → 45 also named by Omni → 13 already drawn, 3 no run → 29 candidates →
gate seen 23, added 6: **bell_miami Bell 0.0 s HIT** (votes no/no/no); un_driving_motorcycle_DgdHSmwA Crowd 0.12 visible; tg_d020 Rain
0.0 visible; marrakech Vehicle 10.0 phantom; waterfall Insect 0.04 cross; tg_d007 Vehicle 4.75 cross. Main rule: cost up -> STOP.
More hits: +1 hit but 2 cross + 1 phantom and w = 1 cost up -> FAIL.
**Target clips.** nyc_1689: VLM Car passing by/Footsteps/Dog/Bird/Vehicle, Omni "Hammering" then junk -> no intersection (the ears
heard the Hammer, the scene never proposed it). nyc_2627: VLM Vehicle, Omni Vehicle/Clicking -> candidate 0.0 s (flex), gate seen
3/3 -> dropped (gold Clang untouched). rainforest_2179: VLM Bird/Water/Insect/Frog/Cricket, Omni "Piano / Stream" -> Water 4.6 s,
gate seen 2/3 -> dropped; Bird not named by Omni. rainforest_7629 / pet_shop: VLM Bird, Omni "Birdsong" -> no word/cos match for
Bird -> nothing. favela: VLM Vehicle/Crowd, Omni "Train wheels squealing / Train moving on tracks" -> no intersection. bell_miami: VLM
Vehicle/Footsteps/Crowd/Bell/Bird, Omni "Church bell / Car" -> Bell 0.0 s added = HIT (the one gain); Vehicle 10.0 s gate seen.
golf: VLM Bird/Cricket, Bird already drawn. tg_d029: VLM Cattle, Omni "Goose honking" -> nothing. tg_d033: VLM Footsteps/Shout/Yell,
Omni "Breathing / Footsteps" -> Footsteps 3.04 s gate seen (Siren never proposed). tg_d095: VLM Cooking/Stir/Splash, Omni "Toilet
flush / Water running" -> nothing. mv_storm: Siren proposed and heard but "already drawn" (Alarm picture). as_explosion, carnival,
ambulance, tg_d107/d125/d133: no intersection.
**Found (not a rule change):** the whole-clip V4 answer (64 tokens, greedy) names 1–2 sounds and then loops on junk tokens ("ive /
ive", "Assistant", ".com") in most clips — the listener's top sound is right on several targets (Hammering, Train, Church bell,
Birdsong, Goose) while the scene VLM proposes generic city/nature families. Closed; motivates Round 40b.

## Round 40b EXPECT-A — the ears alone propose (written 2026-10-01 BEFORE any number; motivated by the Round 40 Omni replies above)
**Rule (merged DEV, saved SHIP8 pictures, gold `gold_AG.json`, MERGE_GAP 2.0).** No VLM proposal. (1) **Listen** (GPU, Qwen3-Omni
`listener_round.MODEL`, whole clip wav as Round 40): one question `expect_a_screen.LIST_Q` ("List every distinct non-speech sound
you hear in this recording as a comma-separated list of short sound names, most prominent first. Name each sound once. Answer with
the list only."), greedy, 96 new tokens, repetition_penalty 1.2. Items = the reply split on commas / newlines / semicolons,
bullets and numbers stripped, lower-cased, leading article dropped. (2) **Families named** = (a) the frozen word map
`expect_a_screen.MAP` (whole item, item without trailing "sound(s)/noise", then each word and its -s/-es singular; Birdsong -> Bird,
Hammering -> Hammer, church bell -> Bell, traffic -> Vehicle, ...; music / speech / wind / breathing -> none), a depictable label or
family name as a whole item (canonical family); PLUS (b) the shipped matcher `listener_afnext.v4_match` over the items (one line
each) for every depictable family. (3) Not already drawn by SHIP8 (canonical / `same_family` vs any placed picture, rescued
included). (4) Onset = earliest weak-bar run over the three caches exactly as Round 40 (no minimum length); no run -> nothing.
(5) Shipped gate at onset − 1 … + 1 s (6 frames, majority) exactly as Round 40; seen -> dropped, else a 2-s picture.
**Score / pass** exactly as Round 40 (base 28/58, 21, 2.282 must reproduce; main rule hits >= 28, no needed hit lost, w = 2 cost
lower; more-hits rule hits up, cross/phantom not up per part, w = 1 cost lower). Report: per target clip Omni list, mapped
families, candidate onset, gate votes, outcome; funnel totals; both verdicts. GO -> a `src/` flag is Adam's decision; STOP -> closed.
Nothing in `src/` or `config.py` edited.
**Files:** `benchmark/gold/expect_a_screen.py` (`listen` GPU / `cands` CPU / `gate` GPU / `score` CPU, re-using
`expect_screen.cmd_gate` / `cmd_score` on `benchmark/gold/expect_a/`) -> `benchmark/gold/expect_a_screen.json`; `slurm/job_expect_a.sh`.
### Round 40b result — EXPECT-A: STOP on both rules (job 31598712, H200)
Base reproduced (28/58, 21 (6/13/2), 2.282). **SHIP8+EXPECT-A: merged 33/58, 104 wrong (16/64/24), cost 4.338 at w = 2 (base 2.282),
4.113 at w = 1 (base 2.197)** | DEV 22/38, 54 (11/32/11) | tagger DEV2 11/20, 50 (5/32/13). No hit lost. Funnel: 306 Omni items ->
326 family namings (94 frozen map, 232 shipped matcher) -> 61 already drawn, 135 no weak-bar run -> 130 candidates -> gate seen 41,
added 89. **5 new hits: nyc_1689 Hammer 13.72 s (map "hammer"; gold Hammer 13.7–16.0), bell_miami Bell 0.0, pet_shop Crow 0.36 (via
"caw", same family as the gold Bird; the Bird candidate itself was gate-seen 3/3), tg_d107 Snicker 9.0 (matcher from "laughter"),
tg_d133 Fart 0.0 (map).** But 83 wrong pictures: the matcher's cosine > 0.6 on short items spreads every heard sound over its ontology
neighbours (tg_d095 "door slam, water running" -> Door, Sliding door, Tap, Toilet flush, Water, Rain, Steam, Natural sounds = 8 wrong;
tg_d107 "laughter" -> Laughter, Belly laugh, Chuckle, Giggle, Gasp; nyc_1689 "mechanisms" -> Printer), and the gate passes most of
them because their makers are indeed off-screen. Main rule: cost up -> STOP. More hits: +5 hits but 64 cross + 24 phantom -> FAIL.
**Target clips.** nyc_1689: "Hammer, Mechanisms" -> Hammer 13.72 added = HIT (+ Printer cross). nyc_2627: "mechanical fan, clatter"
-> Mechanical fan 0.0 (no gold Clang). rainforest_2179: "wind chime, stream" -> Water gate-seen, Bell phantom, Doorbell cross; Bird
never named. rainforest_7629: "birdsong, piano" -> Bird 0.0 gate SEEN 3/3 (the frames show a bird) -> nothing. pet_shop: "birdsong" ->
Bird gate-seen 3/3, Crow 0.36 passes = HIT. favela: "train wheels squealing, train horn" -> Train onset 0.0 (FlexSED run at the start;
gold 14.6 s) -> cross. bell_miami: Bell HIT, Vehicle gate-seen. tg_d029: "wind, goose" -> Goose 0.0, Bird 3.75, Honk 0.8 all phantom
(gold Chicken 6.9 s; no early run matches). tg_d033: "breathing, footsteps" -> Footsteps/Gasp gate-seen, Pant cross; Siren never named.
tg_d095: 8 wrong, Dishes never named. tg_d107: Snicker HIT, 5 cross. tg_d125: Clunk/Thump/Thunk wrong, Explosion/Clapping never named.
tg_d133: Fart HIT. mv_storm: Explosion/Gunshot cross (Siren already drawn as Alarm). golf, as_explosion, carnival, ambulance: nothing new.
**Finding.** The ears alone find 5 of the 14 unreachable needed sounds at the first weak run, which no earlier round did; the price is
the open-inventory match (ontology neighbours) and a gate that cannot refuse an off-screen wrong family. Closed as pre-registered;
a stricter name match (frozen map only, no cosine) would be a new round, not this one (map-only candidates: 59 of 130).

## Round 40c EXPECT-A2 — map-only, two families per clip (written 2026-10-01 BEFORE any number; motivated by the Round 40b flood)
**Rule (merged DEV, saved SHIP8 pictures, gold `gold_AG.json`, MERGE_GAP 2.0; the Round 40b Omni replies in `expect_a/listen/` are
re-used, nothing is re-listened).** Per clip, walk the Round 40b items in list order; a family is named iff `expect_a_screen.map_item`
returns one (the frozen word map, or an exact case-insensitive depictable family / label name as the whole item — NO cosine matcher);
keep the FIRST TWO distinct families so named (an already-drawn family still counts as one of the two). Then exactly as Round 40b: not
already drawn by SHIP8 (canonical / `same_family`), onset = earliest weak-bar run over the three caches (no run -> nothing), shipped
gate at onset − 1 … + 1 s (majority) — every such candidate is a Round 40b candidate with the same onset, so its cached gate answer in
`expect_a/gate/` is re-used verbatim (none re-asked; any uncached one would be asked on GPU, counted); seen -> dropped, else a 2-s
picture. **Score / pass** as Round 40 / 40b (base 28/58, 21, 2.282 must reproduce; main rule hits >= 28, no needed hit lost, w = 2
cost lower; more-hits rule hits up, cross / phantom not up on either part, w = 1 cost lower). Known before this entry (from 40b):
map-only candidates were 59 of 130, and the 40b hits Hammer, Bell, Fart came from the map while Crow and Snicker came from the cosine
matcher and cannot appear here; the two-per-clip cap and the order are fixed now, not after seeing which ones survive.
**Files:** `benchmark/gold/expect_a2_screen.py` (`cands` / `score`, CPU) -> `benchmark/gold/expect_a2_screen.json`.
### Round 40c result — EXPECT-A2: STOP on both rules (CPU, login node; every gate answer re-used from the 40b cache)
Base reproduced (28/58, 21 (6/13/2), 2.282). **SHIP8+EXPECT-A2: merged 31/58, 47 wrong (12/29/6), cost 2.845 at w = 2 (base 2.282),
2.676 at w = 1 (base 2.197)** | DEV 21/38, 29 (9/17/3) | tagger DEV2 10/20, 18 (3/12/3). No hit lost. Funnel: 306 items -> 94 map
namings -> 83 kept (first two per clip) -> 22 already drawn, 6 no run -> 55 candidates (55/55 gate-cached, 0 asked) -> 26 gate-seen,
29 added: **3 hits (nyc_1689 Hammer 13.72, bell_miami Bell 0.0, tg_d133 Fart 0.0)**, 6 visible, 16 cross, 4 phantom. Main rule: cost up
-> STOP. More hits: +3 but 16 cross + 4 phantom -> FAIL. The flood is gone (104 -> 47 wrong) but the right names at the wrong time
remain: the first weak run is 0 s for most clips (favela Train 0.0 vs gold 14.6 s; crossing_bells Cat "hiss" 0.0; d029 Goose 0.0 vs
Chicken 6.9 s; Honk/Clapping/Thunder/Explosion/Gunshot at some other moment), and the gate passes an off-screen wrong name. The
Omni-only hits reachable in this family are Hammer, Bell, Fart; Bird in rainforest_7629 / pet_shop is gate-seen 3/3 (a bird is in
frame), Siren (tg_d033), Dishes (d095), Chicken (d029), Whistle, Clapping (d125) are never named by the whole-clip listener. Closed.

## Round 40d EXPECT-A3 — the 40c names, placed and confirmed by FineLAP (written 2026-10-01 BEFORE any number; motivated by 40c: right names at the first weak run = wrong moment)
**Candidates** = exactly the Round 40c (clip, family) pairs that survived "not already drawn" (`expect_a2/cands.json` rows with outcome
`candidate` or `no run at weak bars` — the weak-bar rule no longer decides anything), 61 pairs expected.
**FineLAP** (text-queried SED, `AndreasXi/FineLAP`, `~/venv_flap`, exactly `finelap_screen.frame_scores`: 10.24-s windows, hop
5.12 s, last window aligned to the end, 0.16-s frames, query phrase = the canonical family name as the shipped FINELAP cache):
frame score at time t = max over the windows covering t, on a 0.16-s grid from 0 to the clip end. Keep a candidate iff its family
score reaches the shipped bar **0.329** (`FINELAP_VETO`) in some frame; **onset = start of the run >= 0.329 (runs merged when the gap
is <= 0.24 s, `_runs` with LISTEN_RUN_GAP) whose maximum frame score is highest** (ties: the earliest). Below the bar everywhere ->
nothing (counted).
**Gate**: shipped visibility gate at onset − 1 … + 1 s (6 frames, majority) as Rounds 40–40c; the Round 40b cached answer for the same
(clip, family) is re-used iff |onset − cached onset| <= 0.5 s, otherwise asked (GPU; counted). Seen -> dropped, else a 2-s picture.
**Score / pass** as Round 40 (base 28/58, 21, 2.282 must reproduce; main rule hits >= 28, no needed hit lost, w = 2 cost lower;
more-hits rule hits up, cross / phantom not up on either part, w = 1 cost lower). Known before this entry: the 40c hits were Hammer
13.72 (FlexSED), Bell 0.0, Fart 0.0; nothing is known about FineLAP on these pairs.
**Files:** `benchmark/gold/expect_a3_screen.py` (`flap` GPU in venv_flap -> `expect_a3/flap/<clip>.npz`; `cands` / `score` CPU msproj;
`gate` GPU msproj for the unmatched onsets) -> `benchmark/gold/expect_a3_screen.json`; `slurm/job_expect_a3.sh` (L4-4h / A100-4h /
H200-4h).
### Round 40d result — EXPECT-A3: STOP on both rules (job 31598858 FineLAP + 17 live gates; score re-run on the login node after a JSON-key crash, numbers identical to the job's printed line)
Base reproduced (28/58, 21 (6/13/2), 2.282). **SHIP8+EXPECT-A3: merged 33/58, 39 wrong (12/22/5), cost 2.507 at w = 2 (base 2.282),
2.338 at w = 1 (base 2.197)** | DEV 22/38, 26 (8/15/3) | tagger DEV2 11/20, 13 (4/7/2). No hit lost. Funnel: 61 pairs -> 13 below the
FineLAP bar -> 48 candidates (31 gate answers re-used within 0.5 s, 17 asked) -> 25 gate-seen, 23 added: **5 hits — nyc_1689 Hammer
13.92 (FineLAP max 0.775, run 13.92–16.0), favela Train 15.04 (0.936, run 15.04–25.6: the placement 40c got wrong at 0.0 is now
right), bell_miami Bell 0.0, tg_d107 Laughter 8.48 (0.978), tg_d133 Fart 0.0** — plus 6 visible, 9 cross, 3 phantom. Main rule: cost up
-> STOP. More hits: +5 but 9 cross + 3 phantom -> FAIL. The wrong pictures are the same off-screen wrong names the gate cannot refuse
(nyc_2627 Mechanical fan, harbour Footsteps 9.76, botanic_garden Footsteps, storm Explosion, motorcycle Gunshot, d029 Goose 10.88
(gold Chicken 6.9–14.8 but a different family), d095 Door / Water, d007 Vehicle) and visible ones the gate missed (snow_walk Vehicle,
ia_youtube Water, arrest Glass, d075 Glass, d020 Rain, motorcycle Crowd). rainforest_7629 Bird: FineLAP run 15.52–15.84 (0.433), gate
not seen there, but gold Bird starts 0.1 s -> cross; pet_shop Bird 0.0 gate-seen (re-used). Closed.
**Across Rounds 40–40d.** The whole-clip listener + a frozen name map + FineLAP placement reaches 5 of the 14 unreachable needed
sounds (hammer, train, bell, laughter, fart) with no hit lost — the best recall any screen since Round 13 has shown on these — but
each variant adds 2–4 wrong pictures per hit, so none passes the cost rule; the limiting part is now the gate's blindness to an
off-screen WRONG name, not detection or placement.

## Round 40e EXPECT-A4 — 40d's added pictures, each confirmed by DASM (written 2026-10-01 BEFORE any number; motivated by 40d: right placement, wrong off-screen names the gate cannot refuse)
**Rule (CPU only, existing caches).** Start from exactly the 23 pictures Round 40d ADDED (`expect_a3_screen.json["added"]`; the 25
gate-dropped ones stay dropped). A picture is kept iff its family's DASM score — max over the DASM columns whose `canonical` equals the
family (`load_fr` on `data/work/devcand/dasm_cache/<clip>.npz` for DEV, `data/work/dasm_dev2/<clip>.npz` for tagger DEV2) — is
**>= 0.575** (`LISTENER_DASM_BAR`, F8's bar) in some frame with **onset − 0.5 s <= t <= onset + 0.5 s**. A family DASM has no column
for in that clip is reported separately and NOT kept (fixed). Disclosed: the shipped `_dasm_keeps` tests span ± 0.5 s, i.e. for a 2-s
picture onset − 0.5 … onset + 2.5 s; that wider window is reported as a secondary number and cannot change the verdict. Score = SHIP8 +
the kept pictures, `score_per_sound.score_clip` as every round today; base 28/58, 21, 2.282 must reproduce. **Pass** as Round 40 (main:
hits >= 28, no needed hit lost, w = 2 cost lower; more-hits: hits up, cross / phantom not up on either part, w = 1 cost lower). Known
before this entry: the 40d hits are Hammer 13.92, Train 15.04, Bell 0.0, Laughter 8.48, Fart 0.0; no DASM value for any of the 23 has
been looked at. **Files:** `benchmark/gold/expect_a4_screen.py` -> `benchmark/gold/expect_a4_screen.json`.
### Round 40e result — EXPECT-A4: **GO on the main rule**, FAIL on more-hits (CPU, login node, existing DASM caches)
Base reproduced (28/58, 21 (6/13/2), 2.282). **SHIP8+EXPECT-A4: merged 31/58, 26 wrong (7/17/2), cost 2.254 at w = 2 (base 2.282),
2.155 at w = 1 (base 2.197)** | DEV 20/38, 18 (4/12/2), 2.204 (base 2.122) | tagger DEV2 11/20, 8 (3/5/0), 2.364 (base 2.636). No hit
lost. DASM within onset ± 0.5 s keeps 8 of the 23 added pictures (no family without a column): **3 hits kept — bell_miami Bell 0.0
(DASM 0.92), tg_d107 Laughter 8.48 (0.847), tg_d133 Fart 0.0 (0.612)** — plus 1 visible (tg_d020 Rain 0.87) and 4 cross
(rainforest_7629 Bird 0.583, arrest Footsteps 0.583, storm Explosion 0.612, motorcycle Gunshot 0.723). Dropped below the bar: the two
FineLAP-placed hits nyc_1689 Hammer (0.273) and favela Train (0.242) and 13 wrong ones (incl. all 3 phantoms, 5 of 6 visible).
**Main rule: hits 31 >= 28, no needed hit lost, cost 2.254 < 2.282 -> GO.** More hits: +3 but cross 13 -> 17 (DEV 8 -> 12) -> FAIL.
Secondary (the shipped span ± 0.5 s window): 31/58, 27 (7/17/3), cost 2.282 — a tie with SHIP8 (tg_d007 Vehicle 0.612 enters as a
phantom); it cannot change the verdict and is reported as pre-registered. Caveat for Adam: the margin is 0.028 cost on 71 clips
(3 hits bought with 5 wrong pictures); DEV alone is worse (2.204 vs 2.122), the gain is on tagger DEV2 (2.364 vs 2.636). As every
round today, GO means a `src/` flag (whole-clip Qwen3-Omni list -> frozen map, first two families -> FineLAP placement >= 0.329 ->
DASM >= 0.575 at onset ± 0.5 s -> shipped gate -> 2-s picture) is Adam's decision; merged TEST is spent and would be read once at the end.

## Round 40e TEST read (reported, not selected on; written 2026-10-01 BEFORE any TEST number)
Rounds 40 → 40e were tuned in sequence on merged DEV, so EXPECT-A4 is read ONCE on the merged TEST exactly as frozen in Rounds 40b–40e
(no change of prompt, decode, map, cap, bars, windows or gate): old TEST (60 clips, `test_stems.txt`) + tagger TEST (28,
`test2_stems.txt`). **Base = the shipped SHIP8 TEST pictures** as `final_test.py` reads them (old TEST arm `SHIP7+K4AD` under
`data/work/r16final`, tagger part under `tagger_prep.out("test2")`, display flags of `arm_cfg("SHIP7+K4AD")`): 23 hits / 42 misses /
29 wrong (4/20/5) / cost 2.568 on 88 clips (`final_test_ship8.json`) must reproduce first. Per TEST clip: Qwen3-Omni whole-clip list
(`expect_a_screen.LIST_Q`, 96 tokens, repetition_penalty 1.2, wav `data/work/r13test/wav16` / `r13test2/wav16`) -> `map_item` (frozen
map + exact names, no cosine) -> first two distinct families -> not already drawn by the SHIP8 TEST pictures -> FineLAP
(`finelap_screen.frame_scores`, query = family name) >= 0.329, onset = start of the highest-max run (gap <= 0.24 s) -> DASM
(`data/work/dasm_test`, `dasm_test2`) >= 0.575 within onset ± 0.5 s (no column -> not kept) -> shipped gate at onset − 1 … + 1 s
(live, no cache exists for TEST) -> 2-s picture. **Gold is read only at the final score step** (gold_AG `test_bench` subset; tagger
gold filtered to the test2 stems), as `final_test.score` does; every earlier stage runs with the gold stubbed. Report: base row,
EXPECT-A4 row (hits, wrong v/c/p, cost at w = 2 and w = 1), paired clip bootstrap of d cost vs base (`DCC.boot`, 2000 draws, seed 0,
one-sided p), every added picture with its class. This is a report, not a selection: the DEV verdict (Round 40e GO on the main rule)
is not revised by it; Adam reads both.
**Files:** `benchmark/gold/expect_test.py` (`listen` GPU msproj / `cands` CPU / `flap` GPU venv_flap / `dasm` CPU / `gate` GPU msproj /
`score` CPU, stage outputs under `benchmark/gold/expect_test/`) -> `benchmark/gold/expect_test.json`; `slurm/job_expect_test.sh`.
### Round 40e TEST read — result (job 31598892, H200; reported, not selected on)
Base reproduced: **SHIP8 TEST 23 hits / 42 misses / 29 wrong (4/20/5) / cost 2.568** on 88 clips (old TEST 18/43, 19; tagger TEST
5/22, 10). **SHIP8+EXPECT-A4 TEST: 25/65, 35 wrong (5/23/7), cost 2.614 at w = 2 (2.557 at w = 1, base 2.523)** | old TEST 19/43,
24 (3/17/4), 2.400 | tagger TEST 6/22, 11 (2/6/3), 3.071. No hit lost. **d cost vs base +0.045 [−0.114, +0.205], one-sided p 0.757**:
not better; the DEV win (−0.028) does not carry to TEST. Funnel: 85 pairs -> 25 below the FineLAP bar, 39 below DASM, 0 without a
column -> 21 candidates -> 13 gate-seen (clay-shoot Gunshot/Laughter, war_fury Gunshot/Explosion, chainsaw, tg_d109 Train, ...) ->
8 added: **2 hits (m5_doc_restrepo_138b Gunshot 0.0 — FineLAP 0.607, DASM 0.989; tg_d101 Bird 5.12)**, 1 visible (aquarium Water
0.0), 3 cross (live_fire Explosion 2.08, air_raid Bird 1.28, tg_d045 Vehicle 0.0), 2 phantom (chainsaw clip Chainsaw 0.0 at FineLAP
0.997 / DASM 0.898 — a needed sound the gold does not time there; airsoft Laughter 2.88). Read as pre-registered: the Round 40e DEV
verdict (GO, main rule) stands as a DEV result; on TEST the same frozen rule is +2 hits, +6 wrong, cost +0.045 (n.s.). Adam's call on
any `src/` flag; this TEST read is recorded here and in `benchmark/gold/expect_test.json`, nothing re-tuned.

## Round 42 HELDOUT-A4 — is the EXPECT-A4 audio chain precise without the gate? (written 2026-10-01 BEFORE any number)
**Question.** Round 40e EXPECT-A4 added +3 hits / +5 wrong on merged DEV and +2 / +6 on merged TEST. Either the AUDIO part (listener
-> map -> FineLAP -> DASM) is precise and the visibility gate lets wrong off-screen names through, or the audio part itself is imprecise.
Gold cannot answer this (every DEV/TEST read is spent and the gate is entangled); the 415 held-out AudioSet-Strong clips
(`benchmark/gold/audioset_heldout.json`, strong labels with times, disjoint from DEV/TEST, 10 s each) can, with no gate at all.
**Clips.** All 415 (compute fits: the Round 40e TEST job listened to 88 clips in 7:40 including model load, so ~25 min for 415; the
150-clip subset clause of the task is NOT triggered). Asserted before any stage: every clip has an mp4 under
`data/input/audioset_heldout/`, a FlexSED cache (`~/MscProj_tg/data/work/flexsed_heldout`), a BEATs cache and a DASM cache
(`~/MscProj/benchmark/audioset_heldout_windows/{beats,dasm_cache}`); the DASM folder holds 416 files, the stray one is ignored.
**DASM frame scores are NOT recomputed:** the Round 6 held-out cache (`benchmark/detector_round6.py cache --set heldout`: the same
`_Dasm` scorer, the same query file `data/work/dasm_text_queries.pt` and 215-label vocab as every DEV/TEST DASM cache; `fw [215, 500]`
fp16 at 50 fps, read by `dev_candidates_check.load_fr`) is that code's output and is used as is.
**Chain (frozen exactly as Round 40e TEST, no gate).** 16-kHz mono wav by ffmpeg from the mp4 -> Qwen3-Omni whole-clip list
(`expect_a_screen.LIST_Q`, 96 tokens, greedy, repetition_penalty 1.2, `items_of`) -> `expect_a_screen.map_item` (frozen map + exact names,
no cosine) -> first two distinct families -> FineLAP (`finelap_screen.frame_scores`, query = family name, `expect_a3_screen.grid_scores`)
>= 0.329, onset = start of the highest-max run (gap 0.24 s, `_runs`) -> DASM (`expect_a4_screen.dasm_max`) >= 0.575 within onset ± 0.5 s
(no column -> not kept, counted). One deviation, disclosed: there are no SHIP8 pictures on these clips, so the "not already drawn"
filter of Rounds 40c–40e is a no-op; every first-two family goes forward. Strong labels are read ONLY in `score`; the earlier stages read
clip ids alone.
**Scoring (per kept detection = (family, onset)).** `correct` iff some strong event (masked included; Speech/Music included — the map
never yields them) with `score_per_sound.same_family(family, event.label)` has `start` in [onset − 0.5, onset + 1.0] (the gold window
of every round); else `wrong time` iff a same_family event exists anywhere in the clip; else `family absent`. Precision = correct / kept.
Reported: overall; per family with >= 3 detections (families with fewer are listed, not judged); the funnel (items, mapped, first-two,
below FineLAP, no DASM column, below DASM, kept); and the same three classes for the FineLAP-placed candidates BEFORE the DASM check
(secondary: how much DASM buys).
**Comparison with the shipped detectors on the same clips (cheap, caches only).** Round 41 PRIOR415 scored FlexSED 0.8 runs and
BEATs 0.35 spans by canonical EQUALITY on families with >= 10 instances; those numbers are quoted as recorded. For a like-for-like row,
the same raw runs (`prior415_screen.flex_runs` / `beats_runs` under `shipped_flags()`) are re-scored under THIS round's rule
(`same_family`, window [−0.5, +1.0], all families) restricted to runs whose canonical family is depictable (`expect_screen.FAMILIES`) —
the only runs that could become pictures. Both detectors are RAW (before vetoes and the listener), as disclosed in Round 41.
**Pass/fail.** None — this is a diagnostic, nothing is selected and no `src/` change follows from it. Reading rule, written now: if the
audio chain's precision on the 415 is clearly above the raw detectors' like-for-like precision, the DEV/TEST wrong pictures are the
gate's problem (right sound, placed on screen or on a visible twin); if it is at or below them, the audio part itself is imprecise.
**Files:** `benchmark/gold/heldout_a4_screen.py` (`wav` CPU / `listen` GPU msproj / `cands` CPU / `flap` GPU venv_flap / `dasm` CPU /
`score` CPU; stage outputs under `benchmark/gold/heldout_a4/`) -> `benchmark/gold/heldout_a4_screen.json`; `slurm/job_heldout_a4.sh`
(H200-4h,A100-4h, 3 h).

## Round 43 HUMAN — the gate asks the annotator's three steps at the onset (written 2026-10-01 BEFORE any HUMAN number)
**Motivation.** Adam on how he labelled visibility: "I watch the video, hear a sound — even a generic one like a tick or a
whack — then I look at the image and understand what is happening, e.g. a person hitting the ball. If I hear whacks WITHOUT
seeing him doing it at that moment, I understand it is other people (off-screen) hitting balls." Three steps: (1) hear a
generic sound and its moment, (2) use the scene to name its most likely source, (3) seen iff that source is visibly ACTING at
that exact moment — not merely present. The shipped gate (name / a/b / desc, majority on 6 frames spread over stretch ± 1 s)
tests presence over a 7-s window, which is why the golfer silences the off-screen whack and the storm clouds do not silence
on-screen thunder. HUMAN samples frames densely AT the onset and asks (2) and (3) directly.
**Set, truth, base (all as Round 38 CF / BOX).** The 49 cached DEV judge clips (`gate_gold/Qwen38-27B` ∩ JUDGE100), every
stretch of every gold sound, importance >= 2; truth = CURRENT gold (`gold_AG.json`, md5 8cbaf53c, identical locally and in
`~/MscProj`; seen = visible or obvious). Base recomputed from the cached name/ab/desc votes and asserted to be exactly
**seen 41 / silenced 16, needed 38 / kept 33** before any HUMAN number prints. Disclosed: no tg_* cache exists, so tg_d133 Fart,
tg_d127 Water and tg_d128 Laughter are OUTSIDE this screen (extending the set would change the base); the named-flip set the
report must list is golf Whack ×2, bell_miami Bell, as_church_bell, Bird (birds_forest, b3_pet_shop, rainforest ×2, b3_aviary_birds),
ambient_weather_storm ×2 Thunder, london_protest_01.
**Frames (one rule).** onset = the stretch start t0 (gold onset for the first stretch of a sound; for later stretches of a long
sound the stretch start, so variant (b) lines up with the cached per-stretch votes — the question's verb is "starts" for the
first stretch and "is still going on" for later ones). Six frames by `_sample_frames_at`, in time order: one CONTEXT frame at
t0 − 1.0 s (t0 + 1.0 s when t0 < 1.0 s), then t0 − 0.3, t0 − 0.1, t0, t0 + 0.1, t0 + 0.3 s, each clamped at 0 (an onset
< 0.3 s collapses the early frames onto 0 s; disclosed, not special-cased). Deviation from the shipped layout, disclosed:
every frame is preceded by a short caption ("context, 1 s before" / "−0.3 s" / … / "0 s: the moment") in the same user turn
(`ask_seq`, a local copy of `reason._ask`'s template + greedy generation with an interleaved content list; `src/` untouched),
so the question can point at "the 0 s frame" without the model counting unlabeled images; a frame ffmpeg fails to return drops
its caption with it. Fewer than 2 frames returned -> not seen.
**Questions (fixed here; `Qwen/Qwen3.8-27B` via `reason._load`, greedy).**
(a) open naming (24 tokens, `_clean_phrase` <= 5 words): "A sound of {label} {starts|is still going on} at the 0 s frame.
Which object, animal or person in these frames could be making that sound? Answer with a short noun phrase of at most 5
words, or exactly: nothing." A reply starting nothing/none/no/not -> candidate = none -> stretch NOT seen, (b) not asked
(as `_sound_is_visible`).
(b) closed, `reason._ab` in BOTH letter orders, options yes/no, context "{candidate} was named as the likely source of the
{label} that {starts|is still going on} at the 0 s frame. Look at the 0 s frame and its neighbours." yes-option = "at that
moment {candidate} is visibly DOING or UNDERGOING the thing that makes {label} (striking, swinging, flowing, flashing, running,
calling...) — the action itself shows in the frames", no-option = "{candidate} is only present, or the action is not visible
at that moment". Both orders yes -> seen; both no or a split -> not seen (the task's rule: seen iff (b) yes in both orders).
Processes count as actions on purpose: thunder's flash, water's flow, an engine running — otherwise the question un-silences
every ambient visible sound, the BOX/SYNC failure.
**Variants.** (a) HUMAN replaces the shipped majority on every stretch; (b) HUMAN is a fourth vote next to the cached
name/ab/desc (True vs False, None excluded; equal -> the shipped majority). Clip verdict as shipped: silent only if every
stretch is seen.
**Pass (per variant, the standing gate bar):** GO iff seen silenced >= 19 with needed kept >= 32, or needed kept >= 35 with
seen silenced >= 15. Round GO if either variant passes. Report: table (base, a, b), every sound whose verdict differs from
base with gold class, the named candidate and the two raw (b) replies per stretch, the named-flip set above, and stretch counts
(nothing-named, (b) yes / no / split, HUMAN seen vs majority seen). GO -> a full-pipeline arm is a separate pre-registration
(the pipeline's onset is the picture start, not the gold start; a frame-sampler + question-set switch in `reason.py` is Adam's
decision); STOP -> recorded, closed.
**Files:** `benchmark/gold/human_gate.py` (`run` GPU -> `gate_gold/human_Qwen38-27B/`, resumable; `score` CPU ->
`gate_gold/human_summary.json`, asserts 49 files and the base), `slurm/job_human_gate.sh` (H200-4h,A100-4h); run in
`~/MscProj` (holds the cf cache and the identical gold); nothing in `src/` or `config.py` edited.

## Round 44 AVNAME — name the sound from audio AND the frames of that moment (written 2026-10-01 BEFORE any number)
**Idea (Adam).** People understand a sound by combining what they hear with what they see: a generic "whack" plus golfers on screen
means a golf strike. The shipped pipeline names sounds from audio alone (BEATs / FlexSED / listeners) and uses the video only for the
visibility gate. This round asks whether an audio-visual NAMER fixes the wrong-family pictures of `docs/review/ledger_ship8.md`
(Steam on a train crossing, Dog on chopping, Glass on a baby cry, Hammer on a clang, Screaming on birds, Explosion on thunder, Bird
during footsteps) without inventing new ones. Names and drops only: picture times, the gate and everything in `src/` are untouched
(the visibility side is Round 43's ground, not repeated here).
**Rule.** For EVERY placed SHIP8 picture on merged DEV (saved pictures as `expect_screen.parts()` = `btp_screen.parts` with
`B.ARM = "SHIP8"`, `TG_ARMS=SHIP8`; rescued pictures included; gold `benchmark/gold/annotations/gold_AG.json` + the tagger DEV2 gold;
base 28/58 hits, 21 wrong (6/13/2), cost 2.282 must reproduce first) Qwen3-Omni-30B-A3B-Instruct (`listener_round.MODEL`, the shipped
listener) gets BOTH the audio cut and the video frames of the window **picture start − 0.5 s … + 1.5 s** and one fixed question:
*"A sound starts at the middle of this clip. Using both what you hear and what you see, what most likely makes this sound? Answer with
one short sound name."* Disclosed: the window puts the onset at 25 % of the cut, not the middle; the text is kept as given by Adam and
the window is not moved to make it true.
**Input format, fixed by the login-node smoke test (processor call succeeds, 616 tokens):** audio = 16-kHz mono cut of the shipped wav
(`data/work/devcand/wav16`, `r13dev2/wav16`), clamped at the clip edges and zero-padded so every cut is exactly 2.0 s with the onset at
0.5 s, passed as `audio=[cut]`; video = **8 frames** at window start + (i + 0.5) × 0.25 s (`_sample_frames_at`, ffmpeg, clamped to the
clip), each resized so the long side is **448 px**, passed as `videos=[frames]` with `video_metadata` fps 4.0 and
`cap_pixels_per_frame=False` (explicit); `use_audio_in_video=False` (the audio is the separate cut, never the mp4 track). Chat content
order video, audio, text. Decode: `model.thinker.generate`, greedy, 16 new tokens; the answer is lower-cased, outer punctuation,
leading articles and a trailing "sound(s)" / "noise(s)" stripped (`expect_a_screen.items_of` style).
**Map to a family F (the shipped matcher, no cosine).** For each depictable family in `expect_screen.FAMILIES` (215), the name set
is `listener_variants.match_names(Onto(), family)`; a family matches when some name matches the answer whole-word with the V4 regex
(`\b name (s|es)? \b`). Several families may match: tie-break fixed now — (1) a family whose own `label_names` match, (2) the longest
matched name, (3) alphabetical; the number of ambiguous answers is reported. Answers matching `nothing | silence | silent | none |
no sound | quiet` are "NONE". An answer that matches no family is "unmapped".
**Variants (scored with `score_per_sound.score_clip`, as every round).** (a) relabel the picture to F when F is a family and F ≠
`canonical(label)` (time unchanged; an unmapped or NONE answer leaves the picture as is); (b) drop the picture when the answer is NONE
or unmapped (a mapped answer leaves the picture as is, name included); (a)+(b) both. Reported for each: merged / DEV / DEV2 line, every
changed picture (clip, old → new name or dropped, class before → after by `cross_group.classify`), and the three verdicts.
**Pass/fail (each variant, as Round 40).** Main rule: hits >= 28, **no needed hit lost** — the set of matched needed gold sounds
(ledger_ship8 `items` matching) of every clip after ⊇ before, on both parts, so a swap of one hit for another in a clip fails — and
cost at w = 2 lower than 2.282. More-hits rule: hits up, cross / phantom not up on either part, cost at w = 1 lower than the base at
w = 1. GO means a `src/` flag is Adam's decision; merged TEST is spent.
**Known before this entry:** the 21 wrong pictures and 28 hits of the ledger; no Qwen3-Omni AV answer on any of them has been seen.
**Files:** `benchmark/gold/avname_screen.py` (`ask` GPU msproj -> `benchmark/gold/avname/answers.json`, gold never read;
`score` CPU) -> `benchmark/gold/avname_screen.json`; `slurm/job_avname.sh` (H200-4h,A100-4h, 1 h).

### Round 43 result — HUMAN: STOP on both variants (job 31598974, H200, 7 min; `benchmark/gold/human_gate.py` -> `gate_gold/human_summary.json`, votes in `gate_gold/human_Qwen38-27B/`, gitignored like every gate cache)
49 clips, 145 stretches, base reproduced (16/41, 33/38). **(a) HUMAN alone: seen silenced 0/41, needed kept 38/38 -> STOP. (b) fourth
vote: 16/41, 33/38 — identical to base on every sound -> STOP.** Why: the open question (a) answers "nothing" on 105 of 145 stretches,
including on-screen rain (storm_7200), waves, the motorcycle, the flea-market rustle; of the 40 stretches with a candidate, (b)
says yes on 6 (the laughing woman ×2, the waterfall, …), no on 24, split on 10. A second failure, disclosed: on 17 stretches the
open reply starts "Based on the visual evidence…" and `_clean_phrase` keeps that clause as the candidate (the onset-dense,
captioned layout invites an explanation); (b) then answers no/split. Named set: golf Whack 6.5 kept (the target flip, but only
because nothing was named — Whack 24.4, the bells, every Bird are "nothing" too), storm Thunder ×3 and london air horn kept as
before, storm Rain and aviary Bird newly kept (visible calls lost). tg_d133 Fart / tg_d127 Water / tg_d128 Laughter outside the
set. Reading: Adam's step (3) — "is the source visibly acting at that moment" — was reached on only 40 stretches and never
silenced a sound the shipped gate keeps; the VLM does not take the first step (name a plausible source at the onset) when a
"nothing" escape is offered on six near-identical frames. (a) with the shipped wide frames, or (b) alone on the shipped
"named" candidate, would be a new round; not run tonight. Closed.
**Round 43 audit (coordinator's question, read from the cached replies, no re-ask).** Not a scoring bug: the letter map is
`reason._ab`'s (order 1 wants "(a)" = yes, order 2 wants "(b)" = yes) and the 6 both-yes stretches (waterfall Water 2nd
stretch, the laughing woman ×2, …) are read as seen. The 0/41 comes from the clip rule (silent only if EVERY stretch is seen):
only 6 of 96 sounds have a candidate on every stretch (the waterfall's first stretch is "nothing"), so at most 6 could be
silenced, and none has (b) yes on all its stretches. (b) answer distribution over the 47 candidate stretches — order 1 (a =
yes): (a) 16, (b) 31; order 2 (a = no): (a) 36, (b) 9, unparsed "Based…" 2. Of the 47 candidates, 35 are the truncated
explanation "Based on the visual evidence" (`_clean_phrase` keeps the first clause; the raw open reply was not stored, so it
cannot be re-parsed from cache); the 141 "nothing" replies are the literal word. The question set, not the parser or the
scorer, is what fails: the open question takes the "nothing" escape and (b) says no even for visible sources.
### Round 44 result — AVNAME: STOP on both rules, all three variants (job 31599021, H200; 50 pictures asked in 75 s)
Base reproduced (28/58, 21 (6/13/2), 2.282). Answers: 18 same family, 27 other family, 0 NONE, 5 unmapped, 23 ambiguous matches
(tie-break applied). Qwen3-Omni with audio + 8 frames mostly names the LOUDEST or most visible thing in the 2-s window, not the
picture's sound. **(a) relabel: 15/58, 34 wrong (9/23/2), cost 3.380** — 27 pictures renamed, 13 needed hits lost (Explosion -> Gunshot
x2, Vehicle -> Explosion, Alarm -> Sigh, Crowd -> Cough, Cough -> Car, Thunder -> Rain x2, Alarm -> Sneeze, Crying -> Bird, Cat -> Alarm,
Laughter -> Splash, Bee -> Sneeze); of the 7 ledger targets it fixes none outright: Steam -> Train (cross -> visible: the right sound,
but it is on screen), Dog -> Cooking and Explosion -> Thunder likewise cross -> visible, Glass -> Chink/clink, Hammer -> Door, Screaming
-> Door stay cross, Bird (footsteps) unmapped ("swing"). **(b) drop unmapped: 27/58, 17 (5/12/0), cost 2.225** — 5 drops: both phantoms
gone (laundromat "washing machine", hair-dryer "click"), golf Bird ("swing") and tg_d128 Laughter ("bull") gone, but snow_walk Laughter
("tram") was a hit -> main FAIL (hit lost, hits 27 < 28); more-hits FAIL. **(a)+(b): 14/58, 30 (8/22/0), cost 3.324.** Reading: the
AV namer is right where the audio was already right (Thunder, Siren, Fireworks) and wrong elsewhere; the one useful signal is "cannot
name it" = phantom (4 of 5 unmapped pictures were wrong), too small to pass. No `src/` change. Files: `benchmark/gold/avname_screen.json`,
`benchmark/gold/avname/answers.json`.
**Standing combined rule (coordinator, after the numbers above; the project's pass rule since 30 Sept 16:02 = old rule OR fewer-pictures
clause: cost lower, wrong <= base wrong − 3 × hits lost, <= 3 hits lost).** (a) relabel: 13 hits lost -> FAIL. (b) drop unmapped:
1 hit lost, wrong 17 <= 21 − 3 = 18, cost 2.225 < 2.282 -> **PASSES the fewer-pictures clause** (fails this round's main rule, recorded
both ways). (a)+(b): 14 hits lost -> FAIL. So variant (b) gets a TEST read.

## Round 44 TEST read — AVNAME variant (b), frozen (reported, not selected on; written 2026-10-01 BEFORE any TEST number)
Exactly the Round 44 (b) rule on the merged TEST (old TEST 60 + tagger TEST 28 = 88 clips, as `expect_test.parts()`: shipped SHIP8 TEST
pictures, arm `SHIP7+K4AD` under `data/work/r16final` + `tagger_prep.out("test2")`; base 23 hits / 42 misses / 29 wrong (4/20/5) /
cost 2.568 must reproduce). Every placed picture: same model, cut (`r13test/wav16`, `r13test2/wav16`), 8 frames, question, decode,
matcher and tie-break as Round 44; a picture is DROPPED iff its answer is NONE or unmapped; nothing is renamed. Gold is read only in
`score` (`final_test` loading, as `expect_test.cmd_score`). Reported: base and new rows (hits, wrong v/c/p, cost at w = 2 and w = 1),
per part, paired clip bootstrap of d cost vs base (`DCC.boot`, 2000 draws, seed 0, one-sided p), every dropped picture with its class
and answer, hits lost. This is a report: the DEV verdict is not revised by it. **Files:** `benchmark/gold/avname_test.py` (`ask` GPU
msproj -> `benchmark/gold/avname/test_answers.json`; `score` CPU) -> `benchmark/gold/avname_test.json`; `slurm/job_avname_test.sh`.

## Round 43b HUMAN-2 — Round 43 with a strict open-question format and majority-of-stretches aggregation (written 2026-10-01 BEFORE any HUMAN-2 number; follows the Round 43 audit: that failure was mechanical — truncated explanations and a free "nothing" escape — not the idea)
**Changes from Round 43 (everything else identical: set, truth, frames, captions, `ask_seq`, (b) wording, both letter orders,
`Qwen/Qwen3.8-27B` greedy).** (a) open question, strict format: "Answer with ONLY a short noun phrase naming the most likely
visible source of the {label} sound in these frames (e.g. 'the golfer', 'the waterfall'), or exactly 'none' if no plausible
source is visible." — max 24 new tokens, the RAW reply stored; candidate = `_clean_phrase` (<= 5 words) of it; a reply starting
none/nothing/no/not -> no candidate -> stretch not seen. Stretch seen iff (b) yes in both orders. **Sound verdict = seen iff a
MAJORITY of the sound's stretches are seen** (more than half; Round 43 and the shipped gate need every stretch — disclosed; the
base stays the shipped all-stretches rule, 16/41, 33/38, asserted). Variants: (a) HUMAN-2 replaces the shipped per-stretch
majority (then majority of stretches); (b) HUMAN-2 is a fourth per-stretch vote next to name/ab/desc (ties -> shipped majority),
then majority of stretches. Pass bar, report (table, flips vs base with raw reply + (b) replies, named set, counts incl. how many
raw replies exceed 5 words) as Round 43. Files: `benchmark/gold/human2_gate.py` (imports Round 43's helpers; `run` GPU ->
`gate_gold/human2_Qwen38-27B/`, `score` CPU -> `gate_gold/human2_summary.json`), `slurm/job_human2_gate.sh`; `src/` untouched.
Note: this commit also carries the Round 44 TEST-read pre-registration another thread had appended but not yet committed.
### Round 44 TEST read — result (job 31599055, H200; 51 pictures in 84 s; reported, not selected on)
Base reproduced: SHIP8 TEST 23/65, 29 wrong (4/20/5), cost 2.568. Answers on the 51 TEST pictures: 18 same family, 33 other family,
**0 NONE, 0 unmapped** (31 ambiguous) — so variant (b) drops nothing: **AVNAME(b) TEST = base, 23/65, 29 (4/20/5), 2.568, d cost +0.000,
p 1.000.** The DEV gain (2 phantoms + 2 wrong dropped for 1 hit) came from 5 un-nameable answers ("swing", "washing machine", "click",
"bull", "tram"); on TEST every answer names some depictable family. The signal does not carry; nothing changes in `src/`.
Files: `benchmark/gold/avname_test.json`, `benchmark/gold/avname/test_answers.json`.

### Round 43b result — HUMAN-2: STOP on both variants; (b) one seen short of the bar (job 31599056, H200; `human2_gate.py` -> `gate_gold/human2_summary.json`, votes in `gate_gold/human2_Qwen38-27B/`, gitignored)
Base reproduced (16/41, 33/38). **(a) HUMAN-2 alone: 5/41 seen silenced, 38/38 needed kept -> STOP. (b) fourth vote: 18/41,
32/38 -> STOP (GO needs 19/32).** The format fix worked: "none" on 98 of 145 stretches (was 105 literal "nothing" + 35 truncated),
only 4 raw replies over 5 words, candidates are now real ("the golfer", "the church", "the bird on the wire", "the rain").
Step (3) now does its job on the targets: golf Whack 6.5 — "the golfer" named, (b) split -> KEPT (base silenced); Whack 24.4
same; bell_miami — "the church" ×3, (b) no -> KEPT (base silenced); pet-shop Bird — "the bird" ×6, (b) no/split -> KEPT (base
silenced). But (b) also says no/split for visible sources: aviary Bird (6 stretches, 0 yes), the marrakech motorcycle, the
flea-market hands, the tornado horse; (b) yes on 12 stretches only (storm Rain ×3, golf-club Whip, the protest man, …). Fourth
vote: +3 silenced (carnival Drum, crossing Train — seen; rainforest_7629 Bird "the macaws" — NEEDED, lost), −1 kept. Reading:
the annotator's question is answered right for the off-screen whack, bell and bird, and wrong for on-screen animals and
engines whose "action" is subtle at ±0.3 s; the net on this bar is one seen sound short. Not GO; no full-pipeline arm. A
follow-up (not run): (b) yes OR shipped majority per stretch, i.e. HUMAN-2 as an ADD-seen rule like SYNC-2. Closed.
### Round 42 HELDOUT-A4 result — the audio chain is PRECISE without the gate (job 31598973, H200, 23 min; `heldout_a4_screen.json`)
All 415 clips (every cache present; the DASM folder's 416th file is the Round 6 log, not a clip). Funnel: 1819 listener items -> 409
mapped -> 334 first-two pairs on 257 clips -> 115 below the FineLAP bar -> 219 placed -> 0 without a DASM column, 127 below DASM ->
**92 kept**. **Kept precision 74/92 = 0.804** (10 wrong time, 8 family absent); FineLAP-placed before DASM 130/219 = 0.594 (DASM
removes 89 wrong for 56 right). Per family (>= 3): Laughter 9/10, Crowd 8/9, Vehicle 5/6, Dog 5/5, Bell 4/4, Cough 4/4, Toilet flush
3/3, Bird 3/5, Typing 2/3, Gunshot 1/5, Cat 1/3, Water 1/3; 25 thin families 27/32. 34 of the 92 onsets are 0.0 s (31 correct).
**Raw detectors, same rule, depictable families: FlexSED 117/297 = 0.394, BEATs 283/754 = 0.375** (PRIOR415 strict: 0.517 / 0.376).
Same families: BEATs Bird 29/53, Gunshot 6/25, Cat 5/17, Water 8/35, Dog 24/32; FlexSED Laughter 18/23, Crowd 17/29, Bell 10/17.
**Reading (pre-registered rule): the chain's precision (0.80) is clearly above the raw detectors' (0.39 / 0.38) on the same clips, so
the DEV/TEST wrong pictures of EXPECT-A4 are the gate's problem, not the audio part's.** Caveats: AudioSet-Strong clips are 10 s and
mostly single-scene (DEV/TEST are longer, off-screen-rich walks), and "correct" here is audio timing only — a right, well-timed sound
whose maker is on screen is still a wrong picture in gold (the 1 visible of 8 on TEST, 1 of 8 on DEV). The weak families on the 415
(Gunshot 1/5, Cat 1/3, Water 1/3) are the same ones that produced cross pictures on DEV/TEST. Diagnostic only; nothing selected.

## Round 43c HUMAN-ADD — HUMAN-2 as an ADD-seen rule (written 2026-10-01 BEFORE computing; CPU only, from the cached Round 43b replies; the shape was suggested by 43b's flips — HUMAN-2 never silenced a needed sound on its own, so it can only add "seen")
**Rule.** A sound is seen iff the shipped majority says seen (every stretch, as base) OR HUMAN-2 (b) is yes in both letter
orders on MORE THAN HALF of its stretches (`human2_gate.sound_majority` over `human_gate.seen_human`). Same set, truth, base
(asserted 16/41, 33/38) and bar (GO iff seen silenced >= 19 with needed kept >= 32, or needed kept >= 35 with seen silenced
>= 15). Report: table, flips vs base with raw + (b) replies. If GO: the SHIP8 placed pictures whose sound has a cached HUMAN-2
stretch (same stem / resolved label, picture start inside a stretch's onset ± 0.5 s) and would now be silenced, by gold class;
pictures on clips or sounds without a cache are listed as not readable. Script `benchmark/gold/human_add_screen.py` ->
`benchmark/gold/human_add_screen.json`; `src/` untouched.

## Round 42b EXPECT-A5 — EXPECT-A4 restricted to the families the held-out 415 vouch for (written 2026-10-01 BEFORE any DEV/TEST number of this round)
**Selector (independent of gold).** From `heldout_a4_screen.json["per_family_kept"]` (Round 42, strong labels only): families with
held-out precision >= 0.8 AND n >= 3 — **Laughter (9/10), Crowd (8/9), Vehicle (5/6), Dog (5/5), Bell (4/4), Cough (4/4), Toilet flush
(3/3)**; every thin family (n < 3) excluded; Bird (3/5), Typing (2/3), Gunshot, Cat, Water excluded. Frozen here, copied into the script.
**Rule.** EXPECT-A4 exactly as frozen (Rounds 40b–40e, 40e TEST read): the same cached candidates and gate answers, nothing re-run; a
picture is added iff it was added by EXPECT-A4 AND its family is in the list above. DEV: the 8 `outcome == "kept"` pictures of
`expect_a4_screen.json` (SHIP8 base 28/58, 21 (6/13/2), 2.282 must reproduce). TEST: the 8 `added` pictures of `expect_test.json`
(SHIP8 TEST base 23/65, 29 (4/20/5), 2.568 must reproduce). Scored with `score_per_sound.score_clip` as every round.
**Verdicts.** DEV: main rule (hits >= 28, no needed hit lost on either part, w = 2 cost < 2.282) and the fewer-pictures clause (cost
lower, wrong <= 21 − 3·hits lost, hits lost <= 3). TEST: reported with the paired clip bootstrap of d cost vs base (`DCC.boot`, 2000
draws, seed 0, one-sided p); not selected on. Disclosed honestly: the per-picture classes of both sets are already recorded in the
Round 40e result sections above, so this round's outcome is deducible before running; it is pre-registered so the family list, not
the gold, is what selects. GO -> a `src/` flag is Adam's decision. **Files:** `benchmark/gold/expect_a5_screen.py` (CPU, from
`~/MscProj_tg`, `TG_ARMS=SHIP8`) -> `benchmark/gold/expect_a5_screen.json`.

### Round 43c result — HUMAN-ADD: STOP, no change at all (CPU, laptop, `benchmark/gold/human_add_screen.py` -> `human_add_screen.json`)
Base reproduced (16/41, 33/38). **ADD-seen: 16/41 seen silenced, 33/38 needed kept — identical to base, no sound flips.** Every
sound HUMAN-2 silences on its own (storm Rain, golf-club Whip, …, 5 of 41) is one the shipped majority already silences; the
13 visible sounds HUMAN-2 gets right but the shipped gate does not are not reached because HUMAN-2 never reaches a majority of
their stretches. SHIP8 placed pictures (merged DEV, 50): 16 on clips without a HUMAN-2 cache (all tg_*: d127 Water, d128
Laughter not readable), 11 with no same-family gold sound at the picture onset ± 0.5 s, 23 matched but HUMAN-2 not seen
(storm Thunder ×2, protest Vehicle/air horn among them), **0 silenced**. Not GO; no full arm warranted. Closed.
### Round 42b EXPECT-A5 result — DEV **GO on both rules**, TEST worse (CPU, login node; `expect_a5_screen.json`)
Family list asserted = {Laughter, Crowd, Vehicle, Dog, Bell, Cough, Toilet flush}. **DEV** (base reproduced 28/58, 21 (6/13/2), 2.282):
**30/58, 21 (6/13/2), cost 2.169 at w = 2 (2.085 at w = 1, base 2.197)** | DEV 20/38, 14, 2.041 | DEV2 10/20, 7, 2.455; no hit lost.
Added 2, both hits: bell_miami Bell 0.0, tg_d107 Laughter 8.48. Skipped 6 (not in the list): the 4 cross (Bird, Footsteps, Explosion,
Gunshot), the visible Rain and the tg_d133 Fart hit. Main rule GO (30 >= 28, cost 2.169 < 2.282), fewer-pictures GO. **TEST** (base
reproduced 23/65, 29 (4/20/5), 2.568): **23/65, 31 (4/21/6), cost 2.614; d cost +0.045 [+0.000, +0.114], one-sided p 1.000** — worse.
Added 2, both wrong: m4_airsoft_24a Laughter 2.88 (phantom; DASM 0.965), tg_d045 Vehicle 0.0 (cross). Skipped 6: the two TEST hits
(restrepo Gunshot, tg_d101 Bird — both excluded families), Water visible, Chainsaw phantom, Explosion cross, Bird cross. Reading: the
held-out list cleans DEV (its two kept pictures are the DEV hits) but on TEST the two kept families are wrong and the two hits are in
excluded families; 4 pictures in all, 2 right — the held-out selector does not transfer. Not shipped; Adam's call as every GO today.

## Round 45 AGREE-EARS — a second, independent ear must also name the EXPECT sound on the whole clip (written 2026-10-01 BEFORE any number)
**Why.** Of the 16 pictures EXPECT-A4 added (DEV 8 + TEST 8), 2 are visible and 9 are cross / phantom: the wrong ones are mostly
AUDIO-side (a sound named that is not there, or not then), which no visibility vote can reach (HUMAN-2 at the EXPECT onsets could touch
at most the 2 visible ones). The held-out 415 (Round 42) showed the audio chain is 80 % precise, so what is needed is a selector that
(1) does not read DEV/TEST gold and (2) can be checked on the 415 first. Rule: a picture of the EXPECT-A4 chain is kept only if a
SECOND listener of a different family — **Audio Flamingo Next** (`nvidia/audio-flamingo-next-hf`, the Round 14 C second ear) — also
names its family on the whole clip, independently of Qwen3-Omni.
**Listen (GPU msproj, one model load).** Every clip of merged DEV (71: `expect_a/listen/*.json` parts dev / dev2), merged TEST (88:
`expect_test/listen/*.json`) and the 415 held-out (`heldout_a4_screen.ids()`), whole 16-kHz wav (`devcand/wav16`, `r13dev2/wav16`,
`r13test/wav16`, `r13test2/wav16`, `heldout_a4/wav16`), the SAME question as Omni (`expect_a_screen.LIST_Q`), AFN decode as Round 14 C
(`listener_afnext.load_model()["gen"]`: greedy, no repetition penalty; 96 new tokens as Omni). Disclosed: Omni used repetition_penalty
1.2, AFN does not (the Round 14 C choice), so only the ear differs, not the question or the map. Parsed with `expect_a_screen.items_of`
-> `map_item` (imported, the frozen Round 40b map, no cosine) -> the clip's AFN family set. Secondary, one forward pass per candidate:
AFN yes/no (`benchmark.listener_round.QUESTION`, logit yes − no as Round 14 C) on the cut [onset − 1, onset + 3] s (the 2-s picture
± 1 s, clipped to the clip).
**Primary rule (AGREE).** An EXPECT-A4 picture is kept iff its family is in the clip's AFN family set. Nothing else re-run: DEV = the 8
`outcome == "kept"` pictures of `expect_a4_screen.json`, TEST = the 8 `added` of `expect_test.json`, held-out = the 92 `outcome ==
"kept"` detections of `heldout_a4_screen.json` (classes already recorded there). Secondary (AGREE+YN): kept iff AGREE and the yes/no
margin > 0 at the onset cut; reported on all three sets, never decides.
**Reads, in this order.** (i) Held-out 415 first: precision of the agreed subset vs 0.804 and how many of the 74 correct survive; reading
rule written now: the selector is "vouched" if agreed precision >= 0.85 with >= half of the 74 correct kept, "not vouched" otherwise
(reported either way; the DEV/TEST reads follow regardless, as pre-registered). (ii) DEV: SHIP8 base 28/58, 21 (6/13/2), 2.282 must
reproduce; main rule (hits >= 28, no needed hit lost on either part, w = 2 cost < 2.282) and the fewer-pictures clause (cost lower,
wrong <= 21 − 3·hits lost, hits lost <= 3), as Round 42b. (iii) TEST: base 23/65, 29 (4/20/5), 2.568 must reproduce; reported with the
paired clip bootstrap of d cost vs base (`DCC.boot`, 2000 draws, seed 0, one-sided p), not selected on. (iv) Pooled DEV + TEST (159
clips, one paired bootstrap over all clips) for EXPECT-A4 and for AGREE, reported as a secondary honesty number (the question of
whether any EXPECT variant has a positive expected effect across both splits); it decides nothing.
**Disclosed honestly** (as Round 42b): the per-picture classes of all three sets are already recorded, so once the AFN lists exist the
outcome is deducible; the AFN list, not the gold, selects. Known before this entry: the DEV kept pictures are Bird, Bell, Footsteps,
Explosion, Gunshot, Rain, Laughter, Fart; the TEST added are Water, Chainsaw, Laughter, Explosion, Gunshot, Bird, Vehicle, Bird. No AFN
whole-clip list exists for any clip (the Round 14 C caches are per-cut). GO -> a `src/` flag is Adam's decision; `src/` untouched.
**Files:** `benchmark/gold/agree_ears_screen.py` (`listen` GPU msproj -> `benchmark/gold/agree_ears/{dev,test,heldout}/<clip>.json`,
resumable; `score` CPU, dev and test in separate processes as `expect_a5_screen.py`) -> `benchmark/gold/agree_ears_screen.json`;
`slurm/job_agree_ears.sh` (H200-4h,A100-4h,L4-4h; from `~/MscProj_tg`, `TG_ARMS=SHIP8`).
### Round 45 result — AGREE-EARS: DEV GO on the main rule, TEST worse, held-out not vouched: STOP (job 31599149, H200, 2.5 min for 574 clips; `agree_ears_screen.py` -> `agree_ears_screen.json`, lists in `agree_ears/` on the cluster)
AFN's whole-clip lists are clean (no repetition loop, e.g. 'Toilet flush, Female speech, Water'). **(i) Held-out 415 (read first):** AGREE
keeps 78 of 92, 65 of 74 correct survive, **precision 0.833** (base 0.804), only 5 of 18 wrong removed -> **not vouched** (bar 0.85).
AGREE+YN (secondary): 69 kept, 59 correct, precision 0.855, 8 of 18 wrong removed -> vouched on the bar, at the price of 15 of 74 correct.
Per family: AFN agrees with every Bird (5/5, 3 correct), Water (3/3, 1 correct) and 4 of 5 Gunshot (1 correct) — the second ear hears the
same wrong families the first one does. **(ii) DEV** (base reproduced 28/58, 21 (6/13/2), 2.282): AGREE keeps 6 of 8 — the 3 hits (bell,
laughter, fart) and 3 cross (rainforest Bird, arrest Footsteps, storm Explosion); drops the motorcycle Gunshot (cross; AFN: Crowd,
Explosion) and the tg_d020 Rain (visible; AFN: Fire). **31/58, 24 (6/16/2), cost 2.197** (w = 1: 2.113 vs 2.197): main rule GO (31 >= 28,
no hit lost, 2.197 < 2.282), fewer-pictures STOP (wrong up). AGREE+YN identical on DEV (every agreed picture has margin > 0). **(iii) TEST**
(base reproduced 23/65, 29 (4/20/5), 2.568): AGREE keeps 7 of 8 — every wrong one (aquarium Water, chainsaw phantom, airsoft Laughter,
live_fire Explosion, air_raid Bird, d045 Vehicle) is also named by AFN, and the one picture it drops is a HIT (tg_d101 Bird; AFN: Honk).
**24/65, 35 (5/23/7), cost 2.659; d cost vs base +0.091 [−0.045, +0.227], one-sided p 0.915** — worse than EXPECT-A4 (2.614, p 0.757).
AGREE+YN: 24/65, 34 (5/22/7), 2.636, d +0.068 [−0.068, +0.205], p 0.879 (the d045 Vehicle goes, margin −0.75). **(iv) Pooled DEV + TEST,
159 clips** (secondary): EXPECT-A4 d +0.013 [−0.126, +0.151] p 0.625; AGREE +0.013 [−0.126, +0.126] p 0.622; AGREE+YN +0.000 [−0.138,
+0.113] p 0.546 — no EXPECT variant has a positive expected effect across both splits. **Reading:** a second ear of a different family
agrees with the first on the wrong off-screen names (chainsaw, laughter, explosion, bird are really audible — the gold does not time
them there, or the maker is on screen); the EXPECT errors are not listener hallucinations, so no audio-side agreement can remove them.
With Round 42b this closes the gold-free selectors for EXPECT: family prior (42b) and second-ear agreement (45) both clean DEV and
both fail TEST. STOP; nothing shipped; `src/` untouched.

## Round 46 GOLD-SCOPE — are EXPECT-A4's extra "wrong" pictures real sounds the gold does not list? (written 2026-10-01 BEFORE any item list is built or any answer seen; follows Round 45's reading that the EXPECT wrongs are audible)
**Why.** EXPECT-A4 (Round 40e) adds 3 DEV / 2 TEST hits but every added wrong picture is a sound two independent ears agree on
(Round 45). If some of those are real, salient, off-screen sounds that the gold simply does not list at that moment, "wrong" there is
gold incompleteness, not a pipeline error. Checking only the arm's extras would favour the arm, so the same blind set also holds
SHIP8's own wrong pictures and random no-picture moments as controls.
**Items (fixed here; built by `benchmark/gold/gold_scope_items.py`, seed 0).** (E) every EXPECT-A4 added picture scored wrong: DEV 5
(`expect_a4_screen.json` kept, class != hit) + TEST 6 (`expect_test.json["added"]`, class != hit) = 11. (S) a seed-0 random sample of
20 SHIP8 wrong pictures (cross or phantom; visible ones are already "real sound, source seen") from the DEV ledger
(`ledger_ship8.json`, 15) + the TEST ledger (`ledger_ship8_test.json`, 25, built by `ledger_ship8_test.py`, a reporting re-read of the
spent TEST, no selection). (C) 10 control moments: a seed-0 random DEV/TEST clip and time with no picture within ± 2 s, named with a
family drawn at random from the (E)+(S) family pool that the clip's gold does not list. 41 items, shuffled, set and arm hidden.
**Question per item** (clip plays onset − 0.5 … onset + 2.5 s, the 2-s picture's span): Q1 "Do you hear a {family} sound start or
play here?" (yes / no / unsure). Q2 (if yes) "Is the thing making it on screen?" (yes / no / unsure). Q3 (if yes) "Would a deaf viewer
want a picture for it?" (yes / no / unsure). **Real-unlisted** = Q1 yes AND Q2 no AND Q3 yes.
**Check before reading (E).** Control (C) Q1-yes rate must be <= 2/10 (the Round 1 visibility control was 0/20); otherwise the check
is uninformative and nothing below is computed.
**Report (no gold edit, no re-selection).** Real-unlisted counts for (E) and (S), side by side (rate and 95 % Wilson interval). A
corrected EXPECT-A4 TEST d cost: each (E) TEST picture rated real-unlisted is counted neutral (neither hit nor wrong), so
d_corr = d − 2·k/88; the same for DEV (2·k/71). SHIP8's own wrongs are common to both arms and cancel in d, so (S) only gives the gold's
general incompleteness rate. Whether EXPECT-A4 ships on the corrected view is Adam's decision after he sees the table; `src/` untouched.
**Files:** `benchmark/gold/gold_scope_items.py` -> `benchmark/gold/gold_scope_items.json` (hidden key) + `docs/review/gold_scope_recheck.html`
(+ clip media in `docs/review/recheck_media/`); answers -> `benchmark/gold/gold_scope_answers.json`; `gold_scope_items.py score`.
**Round 46 amendment 1 (written BEFORE any answer exists; the rating page is unchanged).** (a) "Real-unlisted" also needs the clip's
gold to hold NO sound of the same family anywhere (`score_per_sound.same_family`, the scorer's own matcher). A Q1∧¬Q2∧Q3 item whose
family the gold lists at another time is reported separately as "real, gold times it elsewhere" (a gold onset error or a late
repeat; Adam decides per clip) and does NOT enter d_corr. Known at writing: 5 of the 11 E items are listed-elsewhere (arrest
Footsteps, rainforest_7629 Bird, tg_d020 Rain, air_raid Bird, tg_d045 Vehicle), so at most 3 DEV / 4 TEST E items can move d_corr.
(b) Items with onset 0.0 play 0–3.0 s (not −0.5–2.5). (c) Media are in `docs/review/gold_scope_media/`. (d) The hidden key
`gold_scope_items.json` must not be opened before rating.
