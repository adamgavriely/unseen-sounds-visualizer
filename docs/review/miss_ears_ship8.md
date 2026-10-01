# SHIP8 merged-DEV misses: which ear heard each one (raw, before thresholds)

Script `benchmark/gold/miss_ears.py` (CPU, caches only, run in ~/MscProj_tg), data `benchmark/gold/miss_ears.json`.

Each score cell is **hit window / gold span**: hit window = onset -0.5 .. +1.0 s, gold span = onset .. end. BEATs and FlexSED use columns with `same_family(column, sound)`; DASM uses columns whose canonical equals the sound's canonical (as `expect_a4_screen`, stricter). "n/a" = the model has no column for that family (Clang, Whack, thwack: FlexSED and DASM have none). All caches existed for all 30 misses.

Qwen = Qwen3-Omni whole-clip list (`expect_a/listen`, frozen `map_item`); AFN = Audio Flamingo Next whole-clip list (`agree_ears/dev`). Not the per-span listener caches (`dev_listener_afn.json`, `dev_listener_v.json`), which only score candidate spans. The list ears are whole-clip, so they cannot tell which of two same-family events they heard (e.g. row 2, Hammer 8.1 s, is B only through the lists).

Stage 4 = same-family rows of `stage4.json` arms["SHIP8|proposed"] near the sound (start in onset-0.5 .. end, or overlapping the span). Rows in stage4.json are already past the stage-4 vetoes (no veto flag stored), so a vetoed event shows as "no row".

**Weak levels** (bucket A): BEATs < 0.1, FlexSED < 0.3, DASM < 0.3 (max over window and span), and neither list ear names it. Stage-4 bars used for the B note: BEATs 0.175, FlexSED 0.8; display threshold 0.35. Frame above a bar but no row = min-duration, merge or a veto removed it (which one is not recorded).

Buckets (first that applies): **D** a placed same-family picture overlaps the gold span but starts outside the hit window; **C** a near stage-4 row exists but was dropped/gated; **B** some ear hears it but no near stage-4 row; **A** no ear above the weak levels.

**Counts: A 4, B 16, C 8, D 2** (of 30).

| # | clip | sound | time (s) | BEATs | FlexSED | DASM | Qwen | AFN | stage-4 near rows / what killed them | bucket |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ambient_citywalk_nyc_1689 | Vehicle | 3.8-4.3 | 0.10 / 0.08 | 0.71 / 0.71 | 0.72 / 0.72 | no | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 2 | ambient_citywalk_nyc_1689 | Hammer | 8.1-10.7 | 0.01 / 0.02 | 0.01 / 0.19 | 0.00 / 0.04 | yes | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 3 | ambient_citywalk_nyc_1689 | Hammer | 13.7-16.0 | 0.03 / 0.06 | 0.70 / 0.75 | 0.27 / 0.28 | yes | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 4 | ambient_citywalk_nyc_2627 | Clang | 3.8-4.4 | 0.01 / 0.01 | n/a / n/a | n/a / n/a | no | no | none -  | **A** |
| 5 | ambient_nature_rainforest_2179 | Bird | 6.5-16.0 | 0.03 / 0.07 | 0.43 / 0.46 | 0.49 / 0.61 | no | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 6 | ambient_nature_rainforest_7629 | Bird | 0.1-16.0 | 0.42 / 0.46 | 0.68 / 0.77 | 0.54 / 0.58 | yes | yes | 12 rows (max conf 0.46 Bird vocalization, bird call, bird song@0.89): 9x below display threshold 0.35; 3x stage 5 augment=false (source visible on screen (macaws) - stay silent) | **C** |
| 7 | as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1-11.3 | 0.01 / 0.02 | 0.66 / 0.69 | 0.26 / 0.56 | yes | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 8 | as_explosion_XJ8lc3I6 | Explosion | 2.8-4.2 | 0.14 / 0.14 | 0.62 / 0.62 | 0.52 / 0.52 | yes | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) [placed same family: Explosion@5.68, Explosion@9.25] | **B** |
| 9 | as_explosion_XJ8lc3I6 | Gasp | 6.7-7.0 | 0.41 / 0.04 | 0.76 / 0.76 | 0.07 / 0.07 | no | no | none - above stage-4 bar (BEATs), no row: filtered/vetoed (which veto not recorded) | **B** |
| 10 | b3_carnival_parade | Whistle | 6.1-7.4 | 0.01 / 0.01 | 0.58 / 0.58 | 0.37 / 0.37 | no | no | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 11 | b3_favela_rio | Train | 14.6-28.0 | 0.08 / 0.43 | 0.87 / 0.90 | 0.24 / 0.24 | yes | no | none - above stage-4 bar (BEATs, FlexSED), no row: filtered/vetoed (which veto not recorded) | **B** |
| 12 | b3_golf_course | Whack, thwack | 6.5-7.1 | 0.01 / 0.01 | n/a / n/a | n/a / n/a | no | no | none -  | **A** |
| 13 | b3_golf_course | Whack, thwack | 24.4-25.1 | 0.01 / 0.01 | n/a / n/a | n/a / n/a | no | no | none -  | **A** |
| 14 | b3_pet_shop | Bird | 0.1-27.8 | 0.84 / 0.86 | 0.68 / 0.76 | 0.72 / 0.80 | yes | yes | 3 rows (max conf 0.86 Bird vocalization, bird call, bird song@0.22): 3x stage 5 augment=false (source visible on screen (birds) - stay silent) | **C** |
| 15 | bell_miami | Bell | 0.2-14.5 | 0.70 / 0.70 | 0.98 / 0.98 | 0.92 / 0.92 | yes | yes | 2 rows (max conf 0.70 Church bell@0.22): 2x stage 5 augment=false (source visible on screen (church bell) - stay silent) | **C** |
| 16 | birds_forest | Bird | 1.3-18.0 | 0.21 / 0.65 | 0.71 / 0.75 | 0.60 / 0.64 | yes | yes | 11 rows (max conf 0.65 Crowing, cock-a-doodle-doo@10.25): 8x below display threshold 0.35; 3x placed at 10.25, outside hit window [placed same family: Bird@10.25] | **D** |
| 17 | ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3-8.8 | 0.54 / 0.61 | 0.93 / 0.94 | 0.60 / 0.60 | yes | yes | 3 rows (max conf 0.80 Emergency vehicle@0.0): 3x gate seen (named 'white car') | **C** |
| 18 | ly_applause_62ZYD0u | Crowd | 1.9-10.0 | 0.01 / 0.45 | 0.89 / 0.95 | 0.47 / 0.47 | no | no | 1 rows (max conf 0.45 Applause@1.12): 1x placed at 0.00, outside hit window [placed same family: Crowd@0.00] | **D** |
| 19 | mv_storm_scene_house | Civil defense siren | 16.9-20.4 | 0.01 / 0.01 | 0.42 / 0.41 | 0.02 / 0.03 | yes | yes | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) [placed same family: Alarm@2.18] | **B** |
| 20 | tg_d029 | Chicken, rooster | 6.9-14.8 | 0.81 / 0.81 | 0.26 / 0.26 | 0.11 / 0.13 | no | yes | none - above stage-4 bar (BEATs), no row: filtered/vetoed (which veto not recorded) | **B** |
| 21 | tg_d032 | Thunder | 2.8-5.3 | 0.01 / 0.08 | 0.65 / 0.65 | 0.01 / 0.01 | no | no | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) [placed same family: Thunder@13.75] | **B** |
| 22 | tg_d032 | Thunder | 7.4-10.4 | 0.43 / 0.73 | 0.73 / 0.73 | 0.01 / 0.29 | no | no | 1 rows (max conf 0.27 Thunder@8.0): 1x below display threshold 0.35 [placed same family: Thunder@13.75] | **C** |
| 23 | tg_d033 | Siren | 0.0-18.0 | 0.03 / 0.06 | 0.59 / 0.61 | 0.26 / 0.26 | no | no | none - below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) | **B** |
| 24 | tg_d095 | Dishes, pots, and pans | 16.6-17.0 | 0.37 / 0.29 | 0.31 / 0.31 | 0.19 / 0.19 | no | no | none - above stage-4 bar (BEATs), no row: filtered/vetoed (which veto not recorded) | **B** |
| 25 | tg_d107 | Laughter | 8.2-9.7 | 0.25 / 0.25 | 0.89 / 0.89 | 0.89 / 0.89 | yes | yes | none - above stage-4 bar (BEATs, FlexSED), no row: filtered/vetoed (which veto not recorded) | **B** |
| 26 | tg_d120 | Meow | 2.9-5.6 | 0.81 / 0.83 | 0.12 / 0.13 | 0.92 / 0.95 | yes | yes | 1 rows (max conf 0.64 Domestic animals, pets@2.97): 1x no stage-5 spec holds it (merged/deduped away) [placed same family: Cat@0.56] | **C** |
| 27 | tg_d125 | Explosion | 5.4-5.7 | 0.01 / 0.00 | 0.03 / 0.03 | 0.15 / 0.15 | no | no | none -  | **A** |
| 28 | tg_d125 | Clapping | 8.5-10.0 | 0.02 / 0.02 | 0.79 / 0.80 | 0.34 / 0.35 | no | yes | none - above stage-4 bar (FlexSED), no row: filtered/vetoed (which veto not recorded) | **B** |
| 29 | tg_d133 | Fart | 0.0-1.1 | 0.91 / 0.91 | 0.87 / 0.87 | 0.61 / 0.61 | yes | yes | 2 rows (max conf 0.98 Digestive@0.06): 2x stage 5 augment=false (a kind of Fart, whose source is visible - stay silent) | **C** |
| 30 | tg_d133 | Fart | 5.6-6.4 | 0.86 / 0.86 | 0.86 / 0.86 | 0.58 / 0.58 | yes | yes | 1 rows (max conf 0.91 Fart@6.3): 1x stage 5 augment=false (source visible on screen (boxer dog) - stay silent) | **C** |
