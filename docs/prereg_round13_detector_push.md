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
