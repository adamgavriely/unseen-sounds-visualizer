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
