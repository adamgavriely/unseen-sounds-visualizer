# DEV: where each needed sound is lost — B0r vs the best round-14 arm (2026-09-29)

DEV only (49 clips, gold_AG, 36 needed sounds). Best arm = `LR-V12+1+F1F4F3+F7F8` (LR-V12 + R13-1 + F1 + F4 + F3 + F7 + F8): 15/36 hits, 22 wrong, cost 2.612 (B0r 14/36, 24 wrong, 2.776). Diagnostic only, no selection.

## Counts per cause

| cause | meaning | B0r | best arm |
|---|---|---|---|
| hit | hit | 14 | 15 |
| A | unheard | 3 | 3 |
| B | heard, below every bar, no listener yes | 10 | 7 |
| C | heard, passed a bar or the listener, killed by a named filter/veto | 2 | 5 |
| D | emitted, gate said visible | 4 | 4 |
| E | emitted, timing / merge | 2 | 1 |
| F | other | 1 | 1 |

C here covers any named filter or veto. Under the strict reading (C = listener said yes, then a filter killed it) C is 0 for B0r (its Laughter = PANNs veto and Gasp = 0.5-s min span move to F) and 4 for the best arm (Laughter, PANNs veto with V12 no, moves to F).

## What it shows

- The best arm's only gain over B0r is Cricket (rainforest_7629 @ 0.1 s), and it comes from R13-1 (TWIN_MAX), not from the listener.
- The listener (LR-V12+1) wins 4 more needed sounds; round-14 filters kill all 4: Hammer (F4: Train is the top query at its peak), Whistle (F4: the sibling query Steam whistle is the top; F1 alone also drops it for the stronger Whistle run at 14.6 s), Walk, footsteps @ 2.1 (F1 keeps the STRONGEST Footsteps run, 4.0-5.2 s, which is outside the window; an earliest-first rule would keep it), Gasp (F8: DASM below its bar; read from the arm chain, the DASM cache is not local).
- Laughter (snow_walk): FlexSED 0.92 for 1.7 s, killed by the PANNs clip veto (PANNs 0.001); the PV listener item says V12 no, but Qwen V3/V4 and Audio Flamingo V4 say yes.
- 4 of the 7 B sounds in the best arm were asked and V12 said no while Audio Flamingo V4 said yes (Vehicle/air horn nyc_1689, Gunshot, Explosion x2 in as_explosion); the other 3 were never asked (run peak < 0.5 twice, run covered by a weak BEATs span once).
- D (4): the gate called the source visible; on ly_ambulance it voted seen=True while naming 'nothing'.
- The 3 A sounds are unheard by every detector; 2 of them (Whack, thwack) and the F sound (Clang) are outside the drawable vocabulary.

## Heard but still dropped in the best arm

| clip | sound | onset s | heard by | cause | where it is lost |
|---|---|---|---|---|---|
| ambient_citywalk_nyc_1689 | Hammer | 13.7 | flex | C | listener V12 yes, rescued Hammer 13.76-15.96 (peak 0.747); killed by F4 local winner: 'Train' is the top query at the peak |
| ambient_snow_walk_930 | Laughter | 8.1 | beats, flex | C | FlexSED Laughter 0.919 (run 1.68 s) >= bar 0.8, dropped by the PANNs clip veto (PANNs 0.001); PV listener asked, V12 no (Qwen yes on V3/V4, AF V4 True) |
| as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1 | flex | C | listener V12 yes, rescued Footsteps 2.08-2.72 (peak 0.658); killed by F1 once-per-family: not the strongest |
| as_explosion_XJ8lc3I6 | Gasp | 6.7 | beats, flex | C | listener V12 yes, rescued Gasp 6.32-7.32 (peak 0.758); killed by F8 DASM vote below bar (from the arm chain; DASM cache not local) |
| b3_carnival_parade | Whistle | 6.1 | flex | C | listener V12 yes, rescued Whistle 6.32-7.16 (peak 0.502); killed by F4 local winner: 'Steam whistle' is the top query at the peak and F1 once-per-family: not the strongest |
| ambient_citywalk_nyc_1689 | Vehicle | 3.8 | flex, xq | B | FlexSED Air horn, truck horn 0.708 < 0.8 (run 0.64 s); listener asked (Air horn, truck horn peak 0.708), V12 no [AF V4 yes: 'car horn']; extra query Subway, metro, underground 0.816 for 0.28 s (XQ not in this arm) |
| ambient_nature_rainforest_2179 | Bird | 6.5 | flex | B | FlexSED Bird 0.435 < 0.8 (run 0.08 s); listener not asked: run peak 0.435 < LO 0.5 |
| as_explosion_XJ8lc3I6 | Gunshot, gunfire | 0.0 | beats, flex, panns | B | FlexSED Gunshot 0.549 < 0.8 (run 0.24 s); listener asked (Gunshot peak 0.549), V12 no [Qwen yes on V2/V3/V4] [AF V4 yes: 'gunshot and gunfire, explosion'] |
| as_explosion_XJ8lc3I6 | Explosion | 2.8 | flex, panns, xq | B | FlexSED Explosion 0.621 < 0.8 (run 0.8 s); listener asked (Explosion peak 0.621), V12 no [Qwen yes on V1/V4] [AF V4 yes: 'gunshot and gunfire, explosion'] |
| as_explosion_XJ8lc3I6 | Explosion | 5.6 | beats, flex, panns | B | FlexSED Explosion 0.557 < 0.8 (run 0.52 s); listener asked (Explosion peak 0.557), V12 no [Qwen yes on V1/V4] [AF V4 yes: 'gunshot and gunfire, screaming'] |
| birds_forest | Bird | 1.3 | beats, flex, xq | B | BEATs span Bird 2.22-2.75 conf 0.221 < display 0.35; FlexSED Bird 0.708 < 0.8 (run 5.84 s); listener not asked: the FlexSED run is covered by a same-family stage-4 span; extra query Fowl 0.842 for 0.04 s (XQ not in this arm) |
| mv_storm_scene_house | Civil defense siren | 16.9 | flex | B | FlexSED Siren 0.42 < 0.8 (run 0.08 s); listener not asked: run peak 0.42 < LO 0.5 |
| ambient_nature_rainforest_7629 | Bird | 0.1 | beats, flex, xq | D | span Bird 0.81-4.75 (conf 0.458) reached stage 5; gate: 'source visible on screen (macaws) - stay silent'; gate votes over the window: 0.8-4.8 s seen=True named 'macaws' |
| b3_pet_shop | Bird | 0.1 | beats, flex, panns, xq | D | span Bird 0.14-27.75 (conf 0.856) reached stage 5; gate: 'source visible on screen (birds) - stay silent'; gate votes over the window: 0.1-4.7 s seen=True named 'a small bird' |
| bell_miami | Bell | 0.2 | beats, flex, panns | D | span Bell 0.22-14.5 (conf 0.699) reached stage 5; gate: 'source visible on screen (church bell) - stay silent'; gate votes over the window: 0.2-5.0 s seen=True named 'church bell' |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3 | beats, flex, panns | D | span Vehicle 0.0-14.75 (conf 0.611) reached stage 5; gate: 'source visible on screen (nothing) - stay silent'; gate votes over the window: 4.9-9.8 s seen=True named 'nothing' |
| ly_applause_62ZYD0u | Crowd | 1.9 | flex | E | picture Crowd 0.0-7.75 covers the onset but starts 1.9 s early (merged span) |
| ambient_citywalk_nyc_2627 | Clang | 3.8 | xq | F | label outside the drawable (depictable) vocabulary: no FlexSED-215 query, cannot be drawn |

## Per video

| clip | needed | hit B0r | hit best | losses (best arm) |
|---|---|---|---|---|
| ambient_citywalk_nyc_1689 | 3 | 0 | 0 | B Vehicle@3.8; C Hammer@13.7; A Hammer@8.1 |
| ambient_citywalk_nyc_2627 | 1 | 0 | 0 | F Clang@3.8 |
| ambient_nature_rainforest_2179 | 1 | 0 | 0 | B Bird@6.5 |
| ambient_nature_rainforest_7629 | 2 | 0 | 1 | D Bird@0.1 |
| ambient_snow_walk_930 | 1 | 0 | 0 | C Laughter@8.1 |
| as_explosion_XJ8lc3I6 | 6 | 1 | 1 | B Gunshot, gunfire@0.0; C Walk, footsteps@2.1; B Explosion@2.8; B Explosion@5.6; C Gasp@6.7 |
| b3_bakery_morning | 1 | 1 | 1 | – |
| b3_barbershop | 1 | 1 | 1 | – |
| b3_carnival_parade | 1 | 0 | 0 | C Whistle@6.1 |
| b3_favela_rio | 1 | 1 | 1 | – |
| b3_golf_course | 2 | 0 | 0 | A Whack, thwack@6.5; A Whack, thwack@24.4 |
| b3_pet_shop | 1 | 0 | 0 | D Bird@0.1 |
| bell_miami | 1 | 0 | 0 | D Bell@0.2 |
| birds_forest | 2 | 1 | 1 | B Bird@1.3 |
| ly_ambulance_(siren)_-yPSgCn | 2 | 1 | 1 | D Vehicle@7.3 |
| ly_applause_62ZYD0u | 1 | 0 | 0 | E Crowd@1.9 |
| ly_helicopter_-v62cK1 | 1 | 1 | 1 | – |
| mv_detective_crime_scene | 1 | 1 | 1 | – |
| mv_protest_scene_movie | 3 | 3 | 3 | – |
| mv_storm_scene_house | 2 | 1 | 1 | B Civil defense siren@16.9 |
| mv_tornado_scene | 1 | 1 | 1 | – |
| un_driving_motorcycle_4O3bZRYO | 1 | 1 | 1 | – |

## Every needed sound

Scores = best same-family frame score in [onset − 0.5, onset + 1.0] s (label shown when ≥ 0.1). Run = best FlexSED-215 0.4-run touching the window (else the best extra-query run, marked xq). Listener = the P2/PV item(s) the rescue reads: Qwen yes/no score, Qwen variants that said yes, Qwen V4 text; AF = Audio Flamingo Next V4 (text) and yes/no score.

| clip | sound | onset | B0r | best | BEATs | FlexSED | extra | PANNs | run (peak, s) | listener | filter | cause B0r | cause best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ambient_citywalk_nyc_1689 | Vehicle | 3.8 | miss | miss | 0.10 Vehicle horn, car  | 0.71 Air horn, truck ho | 0.82 Subway, metro, und | 0.04 | Air horn, truck  0.71, 0.64 | P2 yn -0.25 Q[none] V4 'Door closing' AF V4 yes 'car horn' AF yn -0.75 | – | B | B |
| ambient_citywalk_nyc_1689 | Hammer | 13.7 | miss | miss | 0.03 | 0.70 Hammer | 0.00 | 0.01 | Hammer 0.75, 2.20 | P2 yn 5.75 Q[V1/V2/V3/V4/V12] V4 'Hammer/with' AF V4 yes 'hammer' AF yn 2.375 | F4 (Train) | B | C |
| ambient_citywalk_nyc_1689 | Hammer | 8.1 | miss | miss | 0.01 | 0.01 | 0.00 | 0.00 | – | – | – | A | A |
| ambient_citywalk_nyc_2627 | Clang | 3.8 | miss | miss | 0.01 | 0.00 | 0.83 Clang | 0.00 | xq Clang 0.83, 0.24 | – | – | F | F |
| ambient_nature_rainforest_2179 | Bird | 6.5 | miss | miss | 0.03 | 0.43 Bird | 0.00 | 0.02 | Bird 0.43, 0.08 | – | – | B | B |
| ambient_nature_rainforest_7629 | Bird | 0.1 | miss | miss | 0.42 Bird vocalization, | 0.68 Bird | 0.84 Chicken, rooster | 0.05 | Bird 0.69, 1.64 | P1 yn 3.125 Q[V1/V2/V12] V4 '' AF V4 yes 'bird' AF yn 1.75; P1 yn 3.125 Q[V1/V2/V12] V4 '' AF V4 yes 'bird' AF yn 1.75 | – | D | D |
| ambient_nature_rainforest_7629 | Cricket | 0.1 | miss | hit | 0.32 Cricket | 0.92 Insect | 0.00 | 0.00 | Insect 0.93, 16.00 | P1 yn -1.375 Q[none] V4 '' AF V4 no 'bird' AF yn -0.75; P1 yn -0.375 Q[none] V4 '' AF V4 no 'bird' AF yn -0.5 | – | E | hit |
| ambient_snow_walk_930 | Laughter | 8.1 | miss | miss | 0.21 Snicker | 0.92 Laughter | 0.00 | 0.00 | Laughter 0.92, 1.68 | PV yn 2.875 Q[V3/V4] V4 'Laughter/Train/Assis' AF V4 yes 'laughing' AF yn 0.5; P2 yn 2.875 Q[V3/V4] V4 'Laughter/Train/Assis' AF V4 yes 'laughing' AF yn 0.5 | – | C | C |
| as_explosion_XJ8lc3I6 | Gunshot, gunfire | 0.0 | miss | miss | 0.21 Fusillade | 0.55 Gunshot | 0.16 Artillery fire | 0.24 Fusillade | Gunshot 0.55, 0.24 | P2 yn 4.375 Q[V2/V3/V4] V4 'Explosion/wise' AF V4 yes 'gunshot and gunfire, exp' AF yn 2.875 | – | B | B |
| as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1 | miss | miss | 0.01 | 0.66 Footsteps | 0.15 Run | 0.00 | Footsteps 0.66, 0.64 | P2 yn 3.625 Q[V1/V2/V3/V12] V4 'Explosion/Gunshot/Shatter' AF V4 no 'gunshot and gunfire, exp' AF yn -0.25 | F1 (not the strongest) | B | C |
| as_explosion_XJ8lc3I6 | Explosion | 2.8 | miss | miss | 0.14 Fusillade | 0.62 Explosion | 0.73 Machine gun | 0.12 Boom | Explosion 0.62, 0.80 | P2 yn 5.625 Q[V1/V4] V4 'Explosion/Gunshot/Shatter' AF V4 yes 'gunshot and gunfire, exp' AF yn 3.75 | – | B | B |
| as_explosion_XJ8lc3I6 | Explosion | 5.6 | miss | miss | 0.24 Fusillade | 0.56 Explosion | 0.22 Bang | 0.17 Fusillade | Explosion 0.56, 0.52 | P2 yn 5.75 Q[V1/V4] V4 'Gunshot/Explosion/Shatter' AF V4 yes 'gunshot and gunfire, scr' AF yn 5.3125 | – | B | B |
| as_explosion_XJ8lc3I6 | Gasp | 6.7 | miss | miss | 0.41 Gasp | 0.76 Gasp | 0.00 | 0.00 | Gasp 0.76, 0.20 | P2 yn 3.5 Q[V1/V2/V12] V4 'Explosion/Gunshot/Screaming' AF V4 no 'gunshot and gunfire, exp' AF yn 1.875 | F8 (chain) | C | C |
| as_explosion_XJ8lc3I6 | Explosion | 9.1 | hit | hit | 0.53 Fusillade | 0.61 Explosion | 0.00 | 0.14 Boom | Explosion 0.61, 0.92 | P1 yn 5.375 Q[none] V4 '' AF V4 yes 'gunshot and gunfire, exp' AF yn 3.25; P1 yn 3.625 Q[V1] V4 '' AF V4 yes 'gunshot and gunfire, exp' AF yn 1.0 | – | hit | hit |
| b3_bakery_morning | Door | 3.9 | hit | hit | 0.41 Cupboard open or c | 0.24 Door | 0.31 Squeak | 0.44 Cupboard open or c | – | P1 yn 1.25 Q[none] V4 '' AF V4 yes 'drawer open or close, do' AF yn 0.75 | – | hit | hit |
| b3_barbershop | Electric shaver, electric razor | 0.1 | hit | hit | 0.77 Electric shaver, e | 0.66 Electric shaver, e | 0.00 | 0.21 Electric shaver, e | Electric shaver, 0.66, 10.00 | – | – | hit | hit |
| b3_carnival_parade | Whistle | 6.1 | miss | miss | 0.01 | 0.58 Steam whistle | 0.00 | 0.00 | Steam whistle 0.58, 0.84 | P2 yn 3.0 Q[V1/V3] V4 'Train/Train wheels squealing/T' AF V4 no 'gunshot and gunfire, exp' AF yn 0.625; P2 yn 6.0 Q[V1/V2/V3/V12] V4 'Train/Train wheels squealing/T' AF V4 no 'squeal, metallic clatter' AF yn 1.375 | F1 (not the strongest); F4 (Steam whistle) | B | C |
| b3_favela_rio | Train | 14.6 | hit | hit | 0.08 | 0.87 Train | 0.00 | 0.10 Train | Train 0.90, 28.00 | P1 yn 3.25 Q[none] V4 '' AF V4 no 'wind' AF yn -1.5; P1 yn 3.25 Q[none] V4 '' AF V4 no 'wind' AF yn -1.5 | – | hit | hit |
| b3_golf_course | Whack, thwack | 6.5 | miss | miss | 0.01 | 0.00 | 0.00 | 0.00 | – | – | – | A | A |
| b3_golf_course | Whack, thwack | 24.4 | miss | miss | 0.01 | 0.00 | 0.00 | 0.00 | – | – | – | A | A |
| b3_pet_shop | Bird | 0.1 | miss | miss | 0.84 Bird vocalization, | 0.68 Bird | 0.90 Bird vocalization, | 0.32 Bird | Bird 0.72, 19.60 | – | – | D | D |
| bell_miami | Bell | 0.2 | miss | miss | 0.70 Church bell | 0.98 Bell | 0.12 Ding | 0.54 Church bell | Bell 0.98, 14.24 | P1 yn 8.375 Q[V1] V4 '' AF V4 yes 'church bell, bell' AF yn 4.375 | – | D | D |
| birds_forest | Crowing, cock-a-doodle-doo | 10.4 | hit | hit | 0.45 Crowing, cock-a-do | 0.59 Bird | 0.00 | 0.51 Crowing, cock-a-do | Bird 0.75, 3.16 | P1 yn 5.0 Q[V1] V4 '' AF V4 yes 'rooster crowing' AF yn 4.25; P1 yn 5.125 Q[V1] V4 '' AF V4 yes 'bird vocalization and bi' AF yn 3.6875 | – | hit | hit |
| birds_forest | Bird | 1.3 | miss | miss | 0.21 Bird | 0.71 Bird | 0.84 Fowl | 0.07 | Bird 0.71, 5.84 | P1 yn 6.5 Q[V1] V4 '' AF V4 yes 'bird vocalization and bi' AF yn 3.25 | – | B | B |
| ly_ambulance_(siren)_-yPSgCn | Siren | 0.0 | hit | hit | 0.70 Ambulance (siren) | 0.69 Siren | 0.76 Emergency vehicle | 0.42 Emergency vehicle | Siren 0.89, 18.00 | P1 yn 7.0 Q[V1] V4 '' AF V4 yes 'siren' AF yn 3.25 | – | hit | hit |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3 | miss | miss | 0.54 Emergency vehicle | 0.93 Ice cream truck, i | 0.00 | 0.22 Vehicle | Ice cream truck, 0.94, 18.00 | PV yn -0.375 Q[V3] V4 'Siren/ment' AF V4 no 'siren' AF yn -2.5; P2 yn -0.375 Q[V3] V4 'Siren/ment' AF V4 no 'siren' AF yn -2.5 | – | D | D |
| ly_applause_62ZYD0u | Crowd | 1.9 | miss | miss | 0.01 | 0.89 Crowd | 0.08 | 0.07 | Crowd 0.95, 13.36 | P1 yn 4.0 Q[V1] V4 '' AF V4 no 'laughter, human voice' AF yn 4.0 | – | E | E |
| ly_helicopter_-v62cK1 | Helicopter | 0.0 | hit | hit | 0.16 Vehicle | 0.94 Helicopter | 0.00 | 0.31 Vehicle | Helicopter 0.96, 18.00 | P1 yn 5.5 Q[V1] V4 '' AF V4 no 'airplane' AF yn 2.625; P1 yn 5.625 Q[V1] V4 '' AF V4 yes 'Helicopter' AF yn 4.1875 | – | hit | hit |
| mv_detective_crime_scene | Telephone | 0.0 | hit | hit | 0.33 Telephone bell rin | 0.90 Alarm | 0.84 Telephone dialing, | 0.05 | Alarm 0.90, 1.16 | – | F1 (B0 has the family); F4 (Yawn) | hit | hit |
| mv_protest_scene_movie | Crowd | 0.0 | hit | hit | 0.00 | 0.88 Crowd | 0.65 Applause | 0.00 | Crowd 0.92, 20.00 | – | – | hit | hit |
| mv_protest_scene_movie | Glass | 11.0 | hit | hit | 0.84 Shatter | 0.22 Glass | 0.00 | 0.62 Shatter | – | P1 yn 7.25 Q[V1/V2/V12] V4 '' AF V4 yes 'shatter, glass' AF yn 3.5; P1 yn 7.25 Q[V1/V2/V12] V4 '' AF V4 yes 'shatter, glass' AF yn 3.375 | – | hit | hit |
| mv_protest_scene_movie | Shatter | 16.9 | hit | hit | 0.68 Shatter | 0.46 Glass | 0.00 | 0.34 Shatter | Glass 0.48, 1.28 | P1 yn 6.5 Q[V1/V2/V12] V4 '' AF V4 yes 'shatter, glass' AF yn 3.25 | – | hit | hit |
| mv_storm_scene_house | Siren | 2.6 | hit | hit | 0.37 Siren | 0.93 Siren | 0.76 Police car (siren) | 0.03 | Siren 0.96, 7.72 | – | – | hit | hit |
| mv_storm_scene_house | Civil defense siren | 16.9 | miss | miss | 0.01 | 0.42 Siren | 0.00 | 0.00 | Siren 0.42, 0.08 | – | – | B | B |
| mv_tornado_scene | Siren | 8.9 | hit | hit | 0.35 Siren | 0.79 Siren | 0.00 | 0.39 Siren | Siren 0.92, 11.56 | P1 yn 1.875 Q[none] V4 '' AF V4 yes 'siren' AF yn 2.75 | – | hit | hit |
| un_driving_motorcycle_4O3bZRYO | Laughter | 12.7 | hit | hit | 0.04 | 0.91 Laughter | 0.00 | 0.42 Laughter | Laughter 0.93, 2.88 | P2 yn 6.875 Q[V1/V2/V3/V4/V12] V4 'Laughter/of the same./of the s' AF V4 yes 'laughing' AF yn 4.375; P2 yn 3.75 Q[V2/V3/V4] V4 'Laughter' AF V4 no 'laughing' AF yn 3.6875 | F4 (Laughter) | hit | hit |

## Why, per missed sound (B0r → best arm)

- **ambient_citywalk_nyc_1689 · Vehicle @ 3.8 s** — B0r B: FlexSED Air horn, truck horn 0.708 < 0.8 (run 0.64 s); extra query Subway, metro, underground 0.816 for 0.28 s (XQ not in this arm). Best B: FlexSED Air horn, truck horn 0.708 < 0.8 (run 0.64 s); listener asked (Air horn, truck horn peak 0.708), V12 no [AF V4 yes: 'car horn']; extra query Subway, metro, underground 0.816 for 0.28 s (XQ not in this arm).
- **ambient_citywalk_nyc_1689 · Hammer @ 13.7 s** — B0r B: FlexSED Hammer 0.747 < 0.8 (run 2.2 s). Best C: listener V12 yes, rescued Hammer 13.76-15.96 (peak 0.747); killed by F4 local winner: 'Train' is the top query at the peak.
- **ambient_citywalk_nyc_1689 · Hammer @ 8.1 s** — B0r A: no detector reaches its heard bar in the window. Best A: no detector reaches its heard bar in the window.
- **ambient_citywalk_nyc_2627 · Clang @ 3.8 s** — B0r F: label outside the drawable (depictable) vocabulary: no FlexSED-215 query, cannot be drawn. Best F: label outside the drawable (depictable) vocabulary: no FlexSED-215 query, cannot be drawn.
- **ambient_nature_rainforest_2179 · Bird @ 6.5 s** — B0r B: FlexSED Bird 0.435 < 0.8 (run 0.08 s). Best B: FlexSED Bird 0.435 < 0.8 (run 0.08 s); listener not asked: run peak 0.435 < LO 0.5.
- **ambient_nature_rainforest_7629 · Bird @ 0.1 s** — B0r D: span Bird 0.81-4.75 (conf 0.458) reached stage 5; gate: 'source visible on screen (macaws) - stay silent'; gate votes over the window: 0.8-4.8 s seen=True named 'macaws'. Best D: span Bird 0.81-4.75 (conf 0.458) reached stage 5; gate: 'source visible on screen (macaws) - stay silent'; gate votes over the window: 0.8-4.8 s seen=True named 'macaws'.
- **ambient_nature_rainforest_7629 · Cricket @ 0.1 s** — B0r E: FlexSED Insect 0.929 absorbed by a weak BEATs twin (Insect 0.246 < display 0.35), dropped at the display bar; the later same-family picture is out of window. Best hit: hit.
- **ambient_snow_walk_930 · Laughter @ 8.1 s** — B0r C: FlexSED Laughter 0.919 (run 1.68 s) >= bar 0.8, dropped by the PANNs clip veto (PANNs 0.001). Best C: FlexSED Laughter 0.919 (run 1.68 s) >= bar 0.8, dropped by the PANNs clip veto (PANNs 0.001); PV listener asked, V12 no (Qwen yes on V3/V4, AF V4 True).
- **as_explosion_XJ8lc3I6 · Gunshot, gunfire @ 0.0 s** — B0r B: FlexSED Gunshot 0.549 < 0.8 (run 0.24 s). Best B: FlexSED Gunshot 0.549 < 0.8 (run 0.24 s); listener asked (Gunshot peak 0.549), V12 no [Qwen yes on V2/V3/V4] [AF V4 yes: 'gunshot and gunfire, explosion'].
- **as_explosion_XJ8lc3I6 · Walk, footsteps @ 2.1 s** — B0r B: FlexSED Footsteps 0.658 < 0.8 (run 0.64 s). Best C: listener V12 yes, rescued Footsteps 2.08-2.72 (peak 0.658); killed by F1 once-per-family: not the strongest.
- **as_explosion_XJ8lc3I6 · Explosion @ 2.8 s** — B0r B: FlexSED Explosion 0.621 < 0.8 (run 0.8 s). Best B: FlexSED Explosion 0.621 < 0.8 (run 0.8 s); listener asked (Explosion peak 0.621), V12 no [Qwen yes on V1/V4] [AF V4 yes: 'gunshot and gunfire, explosion'].
- **as_explosion_XJ8lc3I6 · Explosion @ 5.6 s** — B0r B: FlexSED Explosion 0.557 < 0.8 (run 0.52 s). Best B: FlexSED Explosion 0.557 < 0.8 (run 0.52 s); listener asked (Explosion peak 0.557), V12 no [Qwen yes on V1/V4] [AF V4 yes: 'gunshot and gunfire, screaming'].
- **as_explosion_XJ8lc3I6 · Gasp @ 6.7 s** — B0r C: BEATs Gasp 0.415 >= display bar for one 0.25-s frame, cut by the 0.5-s min span (FlexSED 0.758 < 0.8). Best C: listener V12 yes, rescued Gasp 6.32-7.32 (peak 0.758); killed by F8 DASM vote below bar (from the arm chain; DASM cache not local).
- **b3_carnival_parade · Whistle @ 6.1 s** — B0r B: FlexSED Steam whistle 0.585 < 0.8 (run 0.84 s). Best C: listener V12 yes, rescued Whistle 6.32-7.16 (peak 0.502); killed by F4 local winner: 'Steam whistle' is the top query at the peak and F1 once-per-family: not the strongest.
- **b3_golf_course · Whack, thwack @ 6.5 s** — B0r A: no detector reaches its heard bar in the window; also outside the drawable vocabulary. Best A: no detector reaches its heard bar in the window; also outside the drawable vocabulary.
- **b3_golf_course · Whack, thwack @ 24.4 s** — B0r A: no detector reaches its heard bar in the window; also outside the drawable vocabulary. Best A: no detector reaches its heard bar in the window; also outside the drawable vocabulary.
- **b3_pet_shop · Bird @ 0.1 s** — B0r D: span Bird 0.14-27.75 (conf 0.856) reached stage 5; gate: 'source visible on screen (birds) - stay silent'; gate votes over the window: 0.1-4.7 s seen=True named 'a small bird'. Best D: span Bird 0.14-27.75 (conf 0.856) reached stage 5; gate: 'source visible on screen (birds) - stay silent'; gate votes over the window: 0.1-4.7 s seen=True named 'a small bird'.
- **bell_miami · Bell @ 0.2 s** — B0r D: span Bell 0.22-14.5 (conf 0.699) reached stage 5; gate: 'source visible on screen (church bell) - stay silent'; gate votes over the window: 0.2-5.0 s seen=True named 'church bell'. Best D: span Bell 0.22-14.5 (conf 0.699) reached stage 5; gate: 'source visible on screen (church bell) - stay silent'; gate votes over the window: 0.2-5.0 s seen=True named 'church bell'.
- **birds_forest · Bird @ 1.3 s** — B0r B: BEATs span Bird 2.22-2.75 conf 0.221 < display 0.35; FlexSED Bird 0.708 < 0.8 (run 5.84 s); extra query Fowl 0.842 for 0.04 s (XQ not in this arm). Best B: BEATs span Bird 2.22-2.75 conf 0.221 < display 0.35; FlexSED Bird 0.708 < 0.8 (run 5.84 s); listener not asked: the FlexSED run is covered by a same-family stage-4 span; extra query Fowl 0.842 for 0.04 s (XQ not in this arm).
- **ly_ambulance_(siren)_-yPSgCn · Vehicle @ 7.3 s** — B0r D: span Vehicle 0.0-14.75 (conf 0.611) reached stage 5; gate: 'source visible on screen (nothing) - stay silent'; gate votes over the window: 4.9-9.8 s seen=True named 'nothing'. Best D: span Vehicle 0.0-14.75 (conf 0.611) reached stage 5; gate: 'source visible on screen (nothing) - stay silent'; gate votes over the window: 4.9-9.8 s seen=True named 'nothing'.
- **ly_applause_62ZYD0u · Crowd @ 1.9 s** — B0r E: picture Crowd 0.0-7.75 covers the onset but starts 1.9 s early (merged span). Best E: picture Crowd 0.0-7.75 covers the onset but starts 1.9 s early (merged span).
- **mv_storm_scene_house · Civil defense siren @ 16.9 s** — B0r B: FlexSED Siren 0.42 < 0.8 (run 0.08 s). Best B: FlexSED Siren 0.42 < 0.8 (run 0.08 s); listener not asked: run peak 0.42 < LO 0.5.

## Data and rules

- B0r: data/work/r13/stage4.json arm B0r|proposed + data/work/r13/B0r_proposed (augmentations, gate_votes, stage-5 log); local
- best_hits: round13_dev.json needed_changes[proposed][best] applied to B0r (equals the pictures B0r + picture_changes appeared - disappeared: 15 hits / 22 wrong, = rows[proposed][best])
- best_stage4: NOT LOCAL (the F-arm stage-4 rows, r14_dropped and stage-5 folders are on the cluster only). Used: LR-V12+1|proposed rows (local), rescued = rows absent from R13-1|proposed; F1 and F4 replayed locally with the pipeline's filter_rescued (round13_dev.flags/arm_cfg, FlexSED cache as ffw); F8 (DASM cache not local) and F3/F7 read from the arm chain in round13_dev.json needed_changes (F3 and F7 changed no needed sound)
- best_stage5: proxy: data/work/r13/LR-V12+1_proposed specs (gate answers are reused/memoised, so a span that survives the filters gets the same answer)
- listener: dev_listener.json (Qwen3-Omni yes/no score), dev_listener_v.json (V1/V2/V3/V4/V12 accept + V4 text), dev_listener_afn.json (Audio Flamingo Next V4 accept + text + yes/no score); items of the same family overlapping the hit window; the rescue reads P2 (FlexSED run peak >= 0.5) and PV (PANNs-vetoed) only
- heard: same-family best frame score in [onset-0.5, onset+1.0] s: BEATs >= 0.175 (benchmark/gold/beats_fw = the j2 cache, checked equal to dev_miss_table on the 22 misses), FlexSED-215 >= 0.4 (data/work/flexsed_cache), PANNs >= 0.1 (benchmark/gold/panns_fw), FlexSED extra queries >= 0.4 (data/work/flexsed_extra_dev); BEATs times are 2-s window starts
- hit: dev_candidates_check.needed_hit on the proposed pictures (same family, picture start in [onset-0.5, onset+1.0] s)
- cause rule: first match: hit; A unheard; F label outside the drawable vocab; [best arm only] C rescued by the listener (V12) then dropped by F1/F4/F8; D a same-family stage-5 span over the window that the gate called visible; E a picture that covers the onset but starts early (merge), or FlexSED >= 0.8 absorbed by a weak BEATs twin below the display bar; C FlexSED >= 0.8 with no stage-4 span (PANNs clip veto) or BEATs >= 0.35 cut by the 0.5-s min span; else B (below every bar; for the best arm with the listener status). C is used for both the rescue filters and B0r's own vetoes (named in 'why').
- Built from local files only; nothing on the cluster was read or run. Machine-readable: `benchmark/gold/dev_heard_dropped.json`.
