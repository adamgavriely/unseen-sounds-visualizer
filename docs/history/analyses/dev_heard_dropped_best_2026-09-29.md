# DEV: where each needed sound is lost — best arm TO1+F7F8 (2026-09-29)

DEV only (49 clips, gold_AG, 36 needed sounds). Arm `TO1+F7F8` = TIER + ONCE + R13-1 + F7 (confirmed mirror veto) + F8 (DASM vote): 18/36 hits, 24 wrong (6 visible / 12 cross / 6 phantom), cost 2.45 (B0r 14/36, 24 wrong, 2.78). Built from the arm's real stage-4 / stage-5 outputs (copied read-only from the cluster). Diagnostic only, no selection.

## The 18 misses: counts per cause

| cause | meaning | n |
|---|---|---|
| unheard | no detector reaches its heard bar in the window | 3 |
| not drawable | label outside the drawable (depictable) vocabulary | 1 |
| below every bar | heard, but under every bar (BEATs display 0.35, FlexSED 0.8) and the listener was not asked | 4 |
| listener rejected | asked; rule TIER said no (Qwen V4 no, or peak < 0.6 and Audio Flamingo V4 no) | 4 |
| killed by F8 (DASM) | rescued by the listener, then dropped: DASM below its bar 0.575 in the span +- 0.5 s | 2 |
| killed by ONCE | rescued, then dropped: not the first rescued span of its family | 0 |
| killed by F7 | BEATs span dropped by the mirror veto b 0.7 and not confirmed by the listener | 0 |
| killed by F4 | local-winner filter (off in this arm) | 0 |
| PANNs veto | FlexSED >= 0.8 span dropped by the PANNs clip veto, PV listener item said no | 0 |
| gate visible | a same-family span reached stage 5 and the gate called the source visible | 3 |
| timing / merge | a same-family picture exists but starts outside [onset - 0.5, onset + 1.0] s | 1 |

## What it shows

- Gained over B0r (4): Cricket 0.1 (R13-1), Laughter 8.1 (PV listener keeps it past the PANNs veto), Gunshot 0.0 and Explosion 5.6 (band rescue). No needed sound is lost.
- F8 kills 2 sounds the rescue had won: Hammer 13.7 (DASM 0.281) and Explosion 2.8 (DASM 0.516, bar 0.575). F7, ONCE and the PANNs veto kill none of the 18; F4 is off. F8 also has a +1 side effect: filter_rescued runs F8 before ONCE, so dropping Explosion 2.84 makes Explosion 5.68 the first rescued Explosion and ONCE keeps it (in TIER+ONCE+1 and TO1+F7, ONCE drops 5.68 as 'not the first'). Net F8 effect on needed sounds: -2 +1.
- The listener (TIER) says no to 4 heard sounds, all on the Qwen V4 leg: Vehicle/air horn nyc_1689 (V4 'Door closing', AF says 'car horn'), Footsteps 2.1 and Gasp 6.7 (V4 names the explosions), Whistle 6.1 (V4 'Train ...'). AF agrees with the no except on the air horn.
- 4 are below every bar and never asked: two FlexSED blips under LO 0.5 (Bird 0.435, Siren 0.42, 0.08 s each), birds_forest Bird (run covered by a weak BEATs span, so the band rescue skips it), and the ambulance Vehicle (in-window Car 0.317 < 0.35; the long Vehicle span was gated visible but starts 7.3 s early anyway).
- 3 gate visible (macaws, pet-shop birds, church bell): gold says not visible. 1 timing/merge (Crowd picture starts 1.9 s early).
- 3 unheard (Hammer 8.1, Whack x2) and 1 not drawable (Clang, heard only by an extra query).

## The 18 misses

| clip | sound | onset s | heard by | cause | evidence |
|---|---|---|---|---|---|
| ambient_citywalk_nyc_1689 | Hammer | 8.1 | – | unheard | no detector reaches its heard bar in the window (BEATs 0.010, FlexSED 0.005, PANNs 0.000, extra 0.000) |
| b3_golf_course | Whack, thwack | 6.5 | – | unheard | no detector reaches its heard bar in the window (BEATs 0.006, FlexSED 0.000, PANNs 0.001, extra 0.000); also outside the drawable vocabulary |
| b3_golf_course | Whack, thwack | 24.4 | – | unheard | no detector reaches its heard bar in the window (BEATs 0.005, FlexSED 0.000, PANNs 0.004, extra 0.000); also outside the drawable vocabulary |
| ambient_citywalk_nyc_2627 | Clang | 3.8 | xq | not drawable | label outside the drawable vocabulary (no FlexSED-215 query); heard only by the extra query Clang 0.826 for 0.24 s (XQ off) |
| ambient_nature_rainforest_2179 | Bird | 6.5 | flex | below every bar | FlexSED Bird 0.435 < 0.8 (run 6.96-7.04, 0.08 s); listener not asked: run peak 0.435 < LO 0.5 |
| birds_forest | Bird | 1.3 | beats, flex, xq | below every bar | in-window BEATs span Bird 2.22-2.75 conf 0.221 < display 0.35; FlexSED Bird 0.708 < 0.8 (run 0.52-6.36, 5.84 s); listener not asked: the run is covered by a same-family stage-4 span; P1 item(s) (not used by the rescue): Bird 2.22-2.75 Qwen V1 yn 6.5; extra query Fowl 0.842 for 0.04 s (XQ off) — The weak BEATs Bird span 2.22-2.75 (0.221) overlaps the FlexSED run 0.52-6.36, so the band rescue skips the run; the P1 item on that BEATs span says yes (V1, yn 6.5) but P1 is not a rescue pool. |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3 | beats, flex, panns | below every bar | in-window BEATs span Car 8.25-8.75 conf 0.317 < display 0.35; FlexSED Ice cream truck, ice cream van 0.939 >= 0.8 but its run 0.0-18.0 starts outside the window; P1 item(s) (not used by the rescue): Police car (siren) 4.75-8.5 Qwen no yn 5.625; Car 8.25-8.75 Qwen V1 yn 4.625 — The long Vehicle span 0.0-14.75 (BEATs 0.611) was gated visible (stretch 4.9-9.8 s seen=True, named 'nothing'), but it starts 7.3 s before the onset, so it could not hit anyway: with FIX_GATE (TO1+F7F8+FIX) it is drawn at 0.0 s as a cross wrong picture and 7.3 s is still missed. The FlexSED Ice cream truck span (0.94, from 0.0 s) was PANNs-vetoed and its PV item said no (Qwen V4 'Siren', AF 'siren'), also out of window. The only in-window candidate is Car 8.25 s (P1 item: Qwen V1 yes, yn 4.6; P1 is not a rescue pool). |
| mv_storm_scene_house | Civil defense siren | 16.9 | flex | below every bar | FlexSED Siren 0.42 < 0.8 (run 16.88-16.96, 0.08 s); listener not asked: run peak 0.42 < LO 0.5 |
| ambient_citywalk_nyc_1689 | Vehicle | 3.8 | flex, xq | listener rejected | P2 Air horn, truck horn 3.76-4.4 peak 0.708: Qwen V4 no ('Door closing'), AF V4 yes ('car horn'), TIER no [Qwen V4 no] |
| as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1 | flex | listener rejected | P2 Footsteps 2.08-2.72 peak 0.658: Qwen V4 no ('Explosion/Gunshot/Shatter'), AF V4 no ('gunshot and gunfire, explosion'), TIER no [Qwen V4 no] |
| as_explosion_XJ8lc3I6 | Gasp | 6.7 | beats, flex | listener rejected | P2 Gasp 6.72-6.92 peak 0.758: Qwen V4 no ('Explosion/Gunshot/Screaming'), AF V4 no ('gunshot and gunfire, explosion'), TIER no [Qwen V4 no]; BEATs Gasp 0.415 for one 0.25-s frame, cut by the 0.5-s min span — Qwen V4 names the explosions, not the gasp; V1/V2/V12 said yes (the old best arm's V12 rescued it, F8 then killed it). |
| b3_carnival_parade | Whistle | 6.1 | flex | listener rejected | P2 Steam whistle 6.32-7.16 peak 0.585: Qwen V4 no ('Train/Train wheels squealing/Train wheel'), AF V4 no ('gunshot and gunfire, explosion'), TIER no [Qwen V4 no]; P2 Whistle 6.32-7.16 peak 0.502: Qwen V4 no ('Train/Train wheels squealing/Train wheel'), AF V4 no ('squeal, metallic clatter, rumble'), TIER no [Qwen V4 no] |
| ambient_citywalk_nyc_1689 | Hammer | 13.7 | flex | killed by F8 (DASM) | rescued Hammer 13.76-15.96 (P2 Hammer 13.76-15.96 peak 0.747: Qwen V4 yes ('Hammer/with'), AF V4 yes ('hammer'), TIER yes); dropped by F8: DASM 0.281 < bar 0.575 — TIER+ONCE+R13-1 (no F8) hits it; F8 then drops it in this arm (DASM far below its bar). |
| as_explosion_XJ8lc3I6 | Explosion | 2.8 | flex, panns, xq | killed by F8 (DASM) | rescued Explosion 2.84-3.64 (P2 Explosion 2.84-3.64 peak 0.621: Qwen V4 yes ('Explosion/Gunshot/Shatter'), AF V4 yes ('gunshot and gunfire, explosion'), TIER yes); dropped by F8: DASM 0.516 < bar 0.575 — TIER+ONCE+R13-1 (no F8) hits it; F8 then drops it in this arm. Its twin at 5.6 s passed F8 and is hit. |
| ambient_nature_rainforest_7629 | Bird | 0.1 | beats, flex, xq | gate visible | span Bird 0.81-4.75 (conf 0.458) reached stage 5; gate: 'source visible on screen (macaws) - stay silent'; votes over the window: 0.8-4.8 s seen=True 'macaws' |
| b3_pet_shop | Bird | 0.1 | beats, flex, panns, xq | gate visible | span Bird 0.14-27.75 (conf 0.856) reached stage 5; gate: 'source visible on screen (birds) - stay silent'; votes over the window: 0.1-4.7 s seen=True 'a small bird' |
| bell_miami | Bell | 0.2 | beats, flex, panns | gate visible | span Bell 0.22-14.5 (conf 0.699) reached stage 5; gate: 'source visible on screen (church bell) - stay silent'; votes over the window: 0.2-5.0 s seen=True 'church bell' |
| ly_applause_62ZYD0u | Crowd | 1.9 | flex | timing / merge | picture Crowd 0.0-7.75 covers the onset but starts 1.9 s early (stage-4 spans: Applause 1.12-7.75 tagger 0.453; Crowd 0.0-1.04 flex 0.867) — The FlexSED Crowd span 0.0-1.04 (0.867) and the Applause span 1.12-7.75 are merged into one picture that starts at 0.0 s; the same picture is a cross wrong picture. |

## Per video (needed sounds)

| clip | needed | hit B0r | hit TO1+F7F8 | misses (cause) |
|---|---|---|---|---|
| ambient_citywalk_nyc_1689 | 3 | 0 | 0 | Vehicle@3.8 (listener rejected); Hammer@13.7 (killed by F8 (DASM)); Hammer@8.1 (unheard) |
| ambient_citywalk_nyc_2627 | 1 | 0 | 0 | Clang@3.8 (not drawable) |
| ambient_nature_rainforest_2179 | 1 | 0 | 0 | Bird@6.5 (below every bar) |
| ambient_nature_rainforest_7629 | 2 | 0 | 1 | Bird@0.1 (gate visible) |
| ambient_snow_walk_930 | 1 | 0 | 1 | – |
| as_explosion_XJ8lc3I6 | 6 | 1 | 3 | Walk, footsteps@2.1 (listener rejected); Explosion@2.8 (killed by F8 (DASM)); Gasp@6.7 (listener rejected) |
| b3_bakery_morning | 1 | 1 | 1 | – |
| b3_barbershop | 1 | 1 | 1 | – |
| b3_carnival_parade | 1 | 0 | 0 | Whistle@6.1 (listener rejected) |
| b3_favela_rio | 1 | 1 | 1 | – |
| b3_golf_course | 2 | 0 | 0 | Whack, thwack@6.5 (unheard); Whack, thwack@24.4 (unheard) |
| b3_pet_shop | 1 | 0 | 0 | Bird@0.1 (gate visible) |
| bell_miami | 1 | 0 | 0 | Bell@0.2 (gate visible) |
| birds_forest | 2 | 1 | 1 | Bird@1.3 (below every bar) |
| ly_ambulance_(siren)_-yPSgCn | 2 | 1 | 1 | Vehicle@7.3 (below every bar) |
| ly_applause_62ZYD0u | 1 | 0 | 0 | Crowd@1.9 (timing / merge) |
| ly_helicopter_-v62cK1 | 1 | 1 | 1 | – |
| mv_detective_crime_scene | 1 | 1 | 1 | – |
| mv_protest_scene_movie | 3 | 3 | 3 | – |
| mv_storm_scene_house | 2 | 1 | 1 | Civil defense siren@16.9 (below every bar) |
| mv_tornado_scene | 1 | 1 | 1 | – |
| un_driving_motorcycle_4O3bZRYO | 1 | 1 | 1 | – |

## The 24 wrong pictures

Origin = the stage-4 span behind the picture: BEATs (tagger span), FlexSED (FlexSED-only span >= 0.8), rescued (listener rescue). 6 are new vs B0r; B0r's other 18 are shared. 1 is a BEATs span the mirror veto dropped and F7 kept.

| type | BEATs | FlexSED | rescued | total |
|---|---|---|---|---|
| visible | 6 | 0 | 0 | 6 |
| cross | 6 | 3 | 3 | 12 |
| phantom | 5 | 1 | 0 | 6 |
| total | 17 | 4 | 3 | 24 |

| clip | picture | start-end s | type | origin | vs B0r | stage-4 span (conf) | BEATs / FlexSED / PANNs at start | gold at that time | note |
|---|---|---|---|---|---|---|---|---|---|
| ambient_weather_storm_16200 | Thunder | 0.06-15.75 | visible | BEATs | same | Thunder 0.06-0.5 (0.451) | 0.45 / 0.86 / 0.83 | Rain visible 0.0-16.0; Thunder visible 0.3-7.5 |  |
| ambient_weather_storm_7200 | Thunder | 0.06-15.75 | visible | BEATs | same | Thunder 0.06-0.5 (0.506) | 0.62 / 0.85 / 0.82 | Thunder visible 0.1-15.8; Rain visible 0.1-15.8 |  |
| as_church_bell_pJRAWkLM | Bell | 0.14-13.75 | visible | BEATs | same | Church bell 0.14-13.75 (0.714) | 0.10 / 0.93 / 0.28 | Bell visible 0.0-15.2 |  |
| as_fire_alarm_kGKZ0YK4 | Alarm | 8.89-19.75 | visible | BEATs | same | Fire alarm 8.89-19.75 (0.696) | 0.53 / 0.92 / 0.15 | Alarm visible 9.0-19.8 |  |
| london_protest_01 | Vehicle | 0.25-1.75 | visible | BEATs | same | Vehicle 0.25-1.5 (0.376) | 0.38 / 0.83 / 0.50 | Crowd visible 0.0-11.1; Air horn, truck horn visible 0.4-11.1 |  |
| un_driving_motorcycle_DgdHSmwA | Fireworks | 13.97-15.47 | visible | BEATs | same | Firecracker 13.97-15.25 (0.408) | 0.41 / 0.46 / 0.35 | Crowd visible 0.0-17.0; Fireworks obvious 13.9-14.2 |  |
| as_explosion_XJ8lc3I6 | Gunshot | 8.25-19.5 | cross | BEATs | same | Fusillade 8.25-9.5 (0.533) | 0.53 / 0.19 / 0.06 | Walk, footsteps needed 2.1-11.3 | starts 1.85 s before the visible machine gun |
| b3_barbershop | Electric shaver, electric razor | 16.0-23.25 | cross | BEATs | same | Electric shaver, electric razor 16.0-23.25 (0.87) | 0.62 / 0.17 / 0.05 | Electric shaver, electric razor needed 0.1-27.8 | second picture of the shaver already hit at 0.1 s |
| b3_crossing_bells | Steam | 0.22-9.0 | cross | BEATs | same | Steam 0.22-4.0 (0.774) | 0.47 / 0.89 / 0.09 | Train visible 0.0-17.0 |  |
| b3_favela_rio | Train | 3.8-5.3 | cross | BEATs | same | Rail transport 3.8-4.0 (0.498) | 0.50 / 0.76 / 0.23 | Walk, footsteps obvious 0.0-28.0; Bird needed 1.8-18.1 |  |
| bell_miami | Train | 8.0-15.0 | cross | BEATs | new | Train horn 8.0-15.0 (0.408) | 0.03 / 0.86 / 0.05 | Bell needed 0.2-14.5 | NEW: R13-1 (TWIN_MAX) lifts the BEATs Train horn span 0.27 -> 0.408 with its FlexSED Train twin (0.86); over the needed church bell |
| mv_protest_scene_movie | Glass | 4.75-7.75 | cross | BEATs (F7 kept) | same | Shatter 4.75-5.25 (0.432) | 0.43 / 0.26 / 0.11 | Crowd needed 0.0-20.0; Baby cry, infant cry visible 3.2-5.1 | BEATs Shatter mirror-vetoed, kept by F7 (listener P1 yes); off the needed Glass onset |
| ly_applause_62ZYD0u | Crowd | 0.0-7.75 | cross | FlexSED | same | Crowd 0.0-1.04 (0.867) | 0.00 / 0.87 / 0.00 | Laughter visible 0.0-13.8 | the merged Crowd picture of the Crowd 1.9 s miss |
| mv_detective_crime_scene | Alarm | 3.92-6.12 | cross | FlexSED | same | Alarm 3.92-6.12 (0.918) | 0.58 / 0.92 / 0.34 | Telephone needed 0.0-9.9 | the ringing phone again (Telephone 0.0 s already hit) |
| mv_detective_crime_scene | Alarm | 9.08-10.58 | cross | FlexSED | same | Alarm 9.08-9.84 (0.911) | 0.41 / 0.91 / 0.59 | Telephone needed 0.0-9.9 | the ringing phone again |
| b3_golf_course | Bird | 18.84-20.34 | cross | rescued | new | Bird 18.84-19.64 (0.785) | 0.12 / 0.79 / 0.02 | Bird needed 0.0-28.0; Walk, footsteps visible 17.1-19.3 | NEW: rescued FlexSED Bird run (0.785, TIER yes) inside a long needed Bird sound (0-28 s), off its onset |
| movie_blueplanet_115 | Laughter | 0.0-1.5 | cross | rescued | new | Laughter 0.0-1.2 (0.674) | 0.13 / 0.67 / 0.12 | Water visible 0.0-16.0; Quack visible 0.2-1.6 | NEW: rescued FlexSED Laughter (0.674) on visible ducks quacking |
| un_driving_motorcycle_DgdHSmwA | Gunshot | 13.52-15.02 | cross | rescued | new | Gunshot 13.52-14.52 (0.74) | 0.18 / 0.74 / 0.01 | Crowd visible 0.0-17.0; Fireworks obvious 13.9-14.2 | NEW: rescued FlexSED Gunshot (0.74) on the obvious fireworks |
| b3_bakery_morning | Chink, clink | 0.14-2.0 | phantom | BEATs | same | Chink, clink 0.14-2.0 (0.489) | 0.42 / 0.44 / 0.06 | – |  |
| b3_laundromat | Train | 1.0-2.5 | phantom | BEATs | new | Train 1.0-1.75 (0.378) | 0.48 / 0.71 / 0.33 | – | NEW: F7 dropped the three Vehicle pictures (listener did not confirm); these BEATs Train spans were hidden inside the Vehicle picture in B0r (FlexSED Train 0.71) |
| b3_laundromat | Train | 16.25-17.75 | phantom | BEATs | new | Train 16.25-16.75 (0.441) | 0.47 / 0.71 / 0.29 | – | NEW: as above (in B0r: 'same picture as Vehicle'); FlexSED Train 0.71 |
| mv_protest_scene_movie | Siren | 22.75-24.02 | phantom | BEATs | same | Siren 22.75-23.75 (0.371) | 0.37 / 0.78 / 0.41 | – |  |
| un_hair_dryer_drying_WWu24rJs | Computer keyboard | 11.25-12.75 | phantom | BEATs | same | Computer keyboard 11.25-11.75 (0.361) | 0.36 / 0.75 / 0.28 | – |  |
| mv_protest_scene_movie | Train | 21.48-22.98 | phantom | FlexSED | same | Train 21.48-22.84 (0.865) | 0.04 / 0.86 / 0.04 | – |  |

## Data and rules

- stage4: data/work/r13/stage4_r14.json = cluster ~/MscProj/data/work/r13/stage4.json (read-only copy): arms['TO1+F7F8|proposed'] (post-filter rows, 'rescued' flag), r14_dropped (F8 DASM value / ONCE), listener (a_added band rescues, b_kept PANNs-vetoed kept), f7 (mirror-vetoed BEATs spans and the listener's keep decision), mirror
- stage5: data/work/r13/TO1+F7F8_proposed (read-only copy of the cluster folder): augmentations.json, gate_votes.json, _stage5_log.json
- pictures: score_per_sound.load_pictures on data/work/r13/TO1+F7F8_proposed: 18 hits / 24 wrong = round13_dev.json rows[proposed][TO1+F7F8]; hits agree with needed_changes on 36/36
- listener: dev_listener_v.json (Qwen3-Omni V1/V2/V3/V4/V12 accept, V4 text, run peak), dev_listener_afn.json (Audio Flamingo Next V4 accept + text + yes/no); TIER recomputed as the pipeline does: Qwen V4 AND (peak >= 0.6 OR AF V4); pools P2 (band runs) and PV (PANNs-vetoed spans) are the ones the rescue reads; P1 shown for reference
- detectors: heard = same-family best frame score in [onset-0.5, onset+1.0] s: BEATs >= 0.175 (benchmark/gold/beats_fw), FlexSED-215 >= 0.4 (data/work/flexsed_cache), PANNs >= 0.1 (benchmark/gold/panns_fw), extra queries >= 0.4 (data/work/flexsed_extra_dev)
- hit: dev_candidates_check.needed_hit: same family, picture start in [onset-0.5, onset+1.0] s
- cause rule: first match: unheard; not drawable; rescued then dropped by F8/ONCE/F4; BEATs span dropped by F7; gate visible (same-family stage-5 span with in-window start); timing/merge (a same-family picture covers the onset but starts outside the window); PANNs veto (FlexSED >= 0.8, no stage-4 span); listener rejected (a P2 item in the window, TIER no); else below every bar
- wrong pictures: type = the change of score_clip's counts when the picture is added in start order (miss_feats.py); origin = the nearest overlapping same-family stage-4 row: rescued (listener band/PV rescue), FlexSED (FlexSED-only span), BEATs (tagger span)
- Nothing was run or written on the cluster (scp from it only). Machine-readable: `benchmark/gold/dev_heard_dropped_best.json`. Earlier trace for the round-14 filter arm: `docs/history/analyses/dev_heard_dropped_2026-09-29.md`.
