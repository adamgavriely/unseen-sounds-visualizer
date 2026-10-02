# DEV-49: every needed sound (importance ≥ 2) and why it is missed — `dev_monocap_v31`, diagnostic, 2026-09-27

36 needed sounds: hit 14, removed by the gate 3, timing 4, detected, not drawn 0, never detected 15.

Scores = best same-family frame score in [onset − 0.5, onset + 1.0] s. Shipped bars: BEATs 0.35, FlexSED 0.8, PANNs veto 0.05. PE-A-Frame = logit minus the frame's median (amendment 24); not used by the shipped system.

| reason | clip | sound | onset s | imp. | BEATs | FlexSED | PANNs | PE-A-Frame |
|---|---|---|---|---|---|---|---|---|
| hit | as_explosion_XJ8lc3I6 | Explosion | 9.1 | 3 | 0.45 | 0.61 | 0.14 | 8.52 |
| hit | b3_bakery_morning | Door | 3.9 | 2 | 0.41 | 0.24 | 0.44 | 6.61 |
| hit | b3_barbershop | Electric shaver, electric razor | 0.1 | 3 | 0.77 | 0.66 | 0.21 | 20.19 |
| hit | b3_favela_rio | Train | 14.6 | 2 | 0.04 | 0.87 | 0.10 | 7.00 |
| hit | birds_forest | Crowing, cock-a-doodle-doo | 10.4 | 2 | 0.45 | 0.59 | 0.51 | 7.63 |
| hit | ly_ambulance_(siren)_-yPSgCn | Siren | 0.0 | 3 | 0.70 | 0.69 | 0.42 | 13.75 |
| hit | ly_helicopter_-v62cK1 | Helicopter | 0.0 | 2 | 0.02 | 0.94 | 0.04 | 16.61 |
| hit | mv_detective_crime_scene | Telephone | 0.0 | 3 | 0.33 | 0.13 | 0.05 | 6.70 |
| hit | mv_protest_scene_movie | Crowd | 0.0 | 3 | 0.00 | 0.88 | 0.00 | 10.59 |
| hit | mv_protest_scene_movie | Glass | 11.0 | 3 | 0.85 | 0.22 | 0.62 | 8.34 |
| hit | mv_protest_scene_movie | Shatter | 16.9 | 3 | 0.67 | 0.46 | 0.34 | 9.24 |
| hit | mv_storm_scene_house | Siren | 2.6 | 3 | 0.37 | 0.93 | 0.03 | 14.32 |
| hit | mv_tornado_scene | Siren | 8.9 | 3 | 0.35 | 0.79 | 0.39 | 12.65 |
| hit | un_driving_motorcycle_4O3bZRYO | Laughter | 12.7 | 2 | 0.04 | 0.91 | 0.42 | 8.68 |
| removed by the gate | ambient_nature_rainforest_7629 | Bird | 0.1 | 2 | 0.42 | 0.68 | 0.05 | 11.71 |
| removed by the gate | b3_pet_shop | Bird | 0.1 | 2 | 0.84 | 0.68 | 0.32 | 9.30 |
| removed by the gate | bell_miami | Bell | 0.2 | 3 | 0.70 | 0.98 | 0.54 | 6.20 |
| timing | ambient_nature_rainforest_7629 | Cricket | 0.1 | 2 | 0.32 | 0.70 | 0.00 | 12.21 |
| timing | birds_forest | Bird | 1.3 | 2 | 0.21 | 0.71 | 0.07 | 7.23 |
| timing | ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3 | 2 | 0.46 | 0.73 | 0.22 | 13.11 |
| timing | ly_applause_62ZYD0u | Crowd | 1.9 | 2 | 0.01 | 0.89 | 0.07 | 10.77 |
| never detected | ambient_citywalk_nyc_1689 | Vehicle | 3.8 | 2 | 0.10 | 0.53 | 0.04 | 11.09 |
| never detected | ambient_citywalk_nyc_1689 | Hammer | 8.1 | 2 | 0.01 | 0.00 | 0.00 | 5.84 |
| never detected | ambient_citywalk_nyc_1689 | Hammer | 13.7 | 2 | 0.03 | 0.70 | 0.01 | 10.07 |
| never detected | ambient_citywalk_nyc_2627 | Clang | 3.8 | 2 | 0.02 | 0.00 | 0.00 | 0.00 |
| never detected | ambient_nature_rainforest_2179 | Bird | 6.5 | 2 | 0.03 | 0.44 | 0.02 | 9.35 |
| never detected | ambient_snow_walk_930 | Laughter | 8.1 | 2 | 0.10 | 0.92 | 0.00 | 6.91 |
| never detected | as_explosion_XJ8lc3I6 | Gunshot, gunfire | 0.0 | 3 | 0.21 | 0.55 | 0.24 | 9.09 |
| never detected | as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1 | 2 | 0.01 | 0.66 | 0.00 | 8.19 |
| never detected | as_explosion_XJ8lc3I6 | Explosion | 2.8 | 2 | 0.10 | 0.62 | 0.12 | 7.75 |
| never detected | as_explosion_XJ8lc3I6 | Explosion | 5.6 | 3 | 0.04 | 0.56 | 0.06 | 9.25 |
| never detected | as_explosion_XJ8lc3I6 | Gasp | 6.7 | 2 | 0.42 | 0.76 | 0.00 | 3.75 |
| never detected | b3_carnival_parade | Whistle | 6.1 | 2 | 0.01 | 0.50 | 0.00 | 10.89 |
| never detected | b3_golf_course | Whack, thwack | 6.5 | 2 | 0.01 | 0.00 | 0.00 | 0.00 |
| never detected | b3_golf_course | Whack, thwack | 24.4 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| never detected | mv_storm_scene_house | Civil defense siren | 16.9 | 2 | 0.01 | 0.42 | 0.00 | 8.03 |
