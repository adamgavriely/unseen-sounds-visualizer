# Panel 2 dissection (1 Oct 2026, 22:00 UTC) — what did we NOT try that is worth trying?

Read also: docs/review/panel_2026-10-01/dissection.md (pipeline, scoring, closed rounds up to the morning), panel_2026-10-01/p*.md and round2_votes.md (this afternoon's panel), docs/review/miss_ears_ship8.md (which ear hears each miss), docs/review/kill_flags_ship8.md (which stage-4 step removes heard misses), and the END of docs/prereg_round13_detector_push.md (Rounds 46-61, every pre-registration and result).

## Shipped now: D = ARMS["SHIP8+MD3+WW5"]
Gap 2.5 s, GROUP (Omni same/new, 8 s), min detected span 0.3 s, Round 57 DEPICT-EVENT, Round 53 WEAK-WITNESS (a non-rescued span needs DASM >= 0.35 or both listeners naming it) + Round 60 SCENE-MARGIN (one listener + Qwen3.8 scene-credible keeps it), no picture floor.
**Merged DEV 29/58 hits, 14 wrong (6 visible / 6 cross / 2 phantom), cost 2.028.** TEST (read once, reported): 24/65, 23 wrong, 2.386 (B was 22/26/2.545; p 0.015).

## D's 29 misses (DEV)
| clip | needed sound | onset | same-family pictures drawn |
|---|---|---|---|
| ambient_citywalk_nyc_1689 | Vehicle | 3.8 | - |
| ambient_citywalk_nyc_1689 | Hammer | 8.1 | - |
| ambient_citywalk_nyc_1689 | Hammer | 13.7 | - |
| ambient_citywalk_nyc_2627 | Clang | 3.8 | - |
| ambient_nature_rainforest_2179 | Bird | 6.5 | - |
| ambient_nature_rainforest_7629 | Bird | 0.1 | - |
| as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1 | - |
| as_explosion_XJ8lc3I6 | Explosion | 2.8 | Explosion@5.68, Explosion@9.25 |
| as_explosion_XJ8lc3I6 | Gasp | 6.7 | - |
| b3_carnival_parade | Whistle | 6.1 | - |
| b3_favela_rio | Train | 14.6 | - |
| b3_golf_course | Whack, thwack | 6.5 | - |
| b3_golf_course | Whack, thwack | 24.4 | - |
| b3_pet_shop | Bird | 0.1 | - |
| bell_miami | Bell | 0.2 | - |
| birds_forest | Bird | 1.3 | Bird@10.25 |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3 | - |
| ly_applause_62ZYD0u | Crowd | 1.9 | Crowd@0.0 |
| mv_storm_scene_house | Civil defense siren | 16.9 | Alarm@2.18 |
| tg_d029 | Chicken, rooster | 6.9 | - |
| tg_d032 | Thunder | 2.8 | Thunder@13.75 |
| tg_d032 | Thunder | 7.4 | Thunder@13.75 |
| tg_d033 | Siren | 0.0 | - |
| tg_d095 | Dishes, pots, and pans | 16.6 | - |
| tg_d120 | Meow | 2.9 | Cat@0.56 |
| tg_d125 | Explosion | 5.4 | - |
| tg_d125 | Clapping | 8.5 | - |
| tg_d133 | Fart | 0.0 | - |
| tg_d133 | Fart | 5.6 | - |

## D's 14 wrong pictures (DEV)
| type | clip | picture | gold near it |
|---|---|---|---|
| visible | ambient_weather_storm_16200 | Thunder @0.06 | Thunder |
| visible | ambient_weather_storm_7200 | Thunder @0.06 | Thunder |
| cross | as_explosion_XJ8lc3I6 | Gunshot @8.25 | Walk, footsteps 2.1-11.3 |
| phantom | b3_flea_market | Vehicle @20.0 | nothing |
| cross | b3_golf_course | Bird @3.8 | Bird 0.0-28.0 |
| visible | london_protest_01 | Vehicle @0.25 | Air horn, truck horn |
| cross | ly_applause_62ZYD0u | Crowd @0.0 | Laughter 0.0-13.8 (seen) |
| visible | un_driving_motorcycle_DgdHSmwA | Explosion @13.52 | Fireworks |
| phantom | un_hair_dryer_drying_WWu24rJs | Computer keyboard @11.25 | nothing |
| cross | tg_d088 | Explosion @10.75 | Rain 0.0-14.8 (seen), Thunder 11.1-14.7 (seen) |
| cross | tg_d088 | Thunder @13.25 | Rain 0.0-14.8 (seen), Thunder 11.1-14.7 (seen) |
| cross | tg_d107 | Screaming @6.52 | Bird vocalization, bird call, bird song 6.0-8.9 (seen) |
| visible | tg_d127 | Water @0.14 | Water |
| visible | tg_d128 | Laughter @3.08 | Laughter |

## Adam's blind ratings (Rounds 46, 46b) — what is really there
~1 in 6 'wrong' pictures is a real wanted sound the gold does not list; of D's wrongs: thunder x3 / water / laughter / screaming are real but on screen; Gunshot 8.25 and Crowd 0.0 are real wanted sounds timed away from the gold row; keyboard/flea-market Vehicle not checked or phantom. He recognises some sounds only from what is SEEN LATER in the clip (motorcycle, bees).

## Today's rounds (all pre-registered; DEV = selection, held-out 415 first where possible; TEST never chooses)
Shipped: merge gap 2.5 (Adam), GROUP (47), MD3 min-span 0.3 (48), DEPICT-EVENT (57), WEAK-WITNESS + SCENE-MARGIN (53+60).
Closed: 46 gold-scope (no gold correction), 49 BANDLIST (415 0.69 -> DEV 0 hits +6 cross), 50 HUMAN-BOX, 51 GRP-P (Omni cannot time a 2-s pause), 52 STRONG-KEEP / 52b KIN-KEEP, 53b/c/d witness variants, 54/54b SIGN (VLM letter bias / always 'no'), 55 FRAME-PAUSE, 56 TWIN-SHORT (+18 wrong), 58 TIGHT-CUT (415 0.40), 58b FLEX-WITNESS (415 inverted), 59 CONTEXT +-5 s (Omni names what it SEES), 61 SCENE-EXPLAIN (Adam's 'unlikely here -> visible look-alike -> could be confused' — 0 drops: the VLM reasoned right on the laundromat but the reversed look-alike question was the wrong bias check; 61d with a polarity twin running), listener cache filled (no hit gained).
Key facts: on the 415, a BEATs span named by no listener is right 22 %, one listener 37 %, both 62 %; scene-credible one-ear spans 41 % vs not-credible 8 %. Qwen3.8-27B runs with thinking OFF and a 4-token cap on yes/no questions (truncations become 'None'); thinking ON is slow (~5 min/picture with 3 questions). Bigger VLMs: not on disk; 235B = future work; Qwen3-VL-32B (~66 GB) possible after cleanup (67 GB free now).

## Constraints
General rules only; pre-register before numbers; DEV (71 clips) is small — 1-2 pictures decide; held-out 415 (10-s AudioSet-Strong clips with video, audio labels only) for any rule it can test; TEST only reported after a ship decision. Offline models on the cluster: Qwen3-Omni-30B (audio+video), Qwen3.8-27B (VLM, thinking optional), Gemma-4-31B, Audio Flamingo Next, BEATs, FlexSED (text-query SED), DASM, EAT, CED, SSLAM, AST, CLAP, FineLAP, OWLv2, SigLIP, CLIP, Step-Audio-2-mini, Qwen-Image. Hours, not days.
