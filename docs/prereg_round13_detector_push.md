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
