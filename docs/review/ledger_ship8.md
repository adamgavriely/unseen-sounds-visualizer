# Shipped pipeline (K4A-D): every miss and every wrong picture

Merged DEV, 71 clips, 58 needed sounds: **28 hits, 30 misses, 21 wrong** (cost 2.282).

A hit needs a picture of the same sound type that starts 0.5 s before to 1.0 s after the sound starts.

## 30 misses

| clip | needed sound | time (s) | pictures of that type we did draw |
|---|---|---|---|
| ambient_citywalk_nyc_1689 | Vehicle | 3.8–4.3 | none |
| ambient_citywalk_nyc_1689 | Hammer | 8.1–10.7 | none |
| ambient_citywalk_nyc_1689 | Hammer | 13.7–16.0 | none |
| ambient_citywalk_nyc_2627 | Clang | 3.8–4.4 | none |
| ambient_nature_rainforest_2179 | Bird | 6.5–16.0 | none |
| ambient_nature_rainforest_7629 | Bird | 0.1–16.0 | none |
| as_explosion_XJ8lc3I6 | Walk, footsteps | 2.1–11.3 | none |
| as_explosion_XJ8lc3I6 | Explosion | 2.8–4.2 | Explosion at 5.68, Explosion at 9.25 |
| as_explosion_XJ8lc3I6 | Gasp | 6.7–7.0 | none |
| b3_carnival_parade | Whistle | 6.1–7.4 | none |
| b3_favela_rio | Train | 14.6–28.0 | none |
| b3_golf_course | Whack, thwack | 6.5–7.1 | none |
| b3_golf_course | Whack, thwack | 24.4–25.1 | none |
| b3_pet_shop | Bird | 0.1–27.8 | none |
| bell_miami | Bell | 0.2–14.5 | none |
| birds_forest | Bird | 1.3–18.0 | Bird at 10.25 |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3–8.8 | none |
| ly_applause_62ZYD0u | Crowd | 1.9–10.0 | Crowd at 0.0 |
| mv_storm_scene_house | Civil defense siren | 16.9–20.4 | Alarm at 2.18 |
| tg_d029 | Chicken, rooster | 6.9–14.8 | none |
| tg_d032 | Thunder | 2.8–5.3 | Thunder at 13.75 |
| tg_d032 | Thunder | 7.4–10.4 | Thunder at 13.75 |
| tg_d033 | Siren | 0.0–18.0 | none |
| tg_d095 | Dishes, pots, and pans | 16.6–17.0 | none |
| tg_d107 | Laughter | 8.2–9.7 | none |
| tg_d120 | Meow | 2.9–5.6 | Cat at 0.56 |
| tg_d125 | Explosion | 5.4–5.7 | none |
| tg_d125 | Clapping | 8.5–10.0 | none |
| tg_d133 | Fart | 0.0–1.1 | none |
| tg_d133 | Fart | 5.6–6.4 | none |

## 21 wrong pictures

- **visible**: the sound is real, but its source is on screen, so no picture is needed
- **cross**: the picture starts while a different sound plays, or too early/late for its own sound
- **phantom**: nothing is happening at that time

| type | clip | our picture (start s) | what is really there (gold) |
|---|---|---|---|
| cross | as_explosion_XJ8lc3I6 | Gunshot (8.25) | Walk, footsteps 2.1-11.3 |
| cross | b3_barbershop | Electric shaver, electric razor (16.0) | Electric shaver, electric razor 0.1-27.8 |
| cross | b3_crossing_bells | Steam (0.22) | Train 0.0-17.0 (seen) |
| cross | b3_golf_course | Bird (18.84) | Bird 0.0-28.0, Walk, footsteps 17.1-19.3 (seen) |
| cross | ly_applause_62ZYD0u | Crowd (0.0) | Laughter 0.0-13.8 (seen) |
| cross | mv_detective_crime_scene | Alarm (3.36) | Telephone 0.0-9.9 |
| cross | mv_detective_crime_scene | Alarm (9.08) | Telephone 0.0-9.9 |
| cross | mv_protest_scene_movie | Glass (4.75) | Crowd 0.0-20.0, Baby cry, infant cry 3.2-5.1 (seen) |
| cross | tg_d022 | Dog (7.25) | Chopping (food) 7.0-8.3 (seen) |
| cross | tg_d088 | Explosion (10.75) | Rain 0.0-14.8 (seen), Thunder 11.1-14.7 (seen) |
| cross | tg_d088 | Thunder (13.25) | Rain 0.0-14.8 (seen), Thunder 11.1-14.7 (seen) |
| cross | tg_d107 | Screaming (7.0) | Bird vocalization, bird call, bird song 6.0-8.9 (seen) |
| cross | tg_d128 | Hammer (9.0) | Clang 5.6-10.0 (seen) |
| phantom | b3_laundromat | Train (1.0) | nothing |
| phantom | un_hair_dryer_drying_WWu24rJs | Computer keyboard (11.25) | nothing |
| visible | ambient_weather_storm_16200 | Thunder (0.06) | Thunder |
| visible | ambient_weather_storm_7200 | Thunder (0.06) | Thunder |
| visible | london_protest_01 | Vehicle (0.25) | Air horn, truck horn |
| visible | tg_d127 | Water (0.14) | Water |
| visible | tg_d128 | Laughter (3.08) | Laughter |
| visible | un_driving_motorcycle_DgdHSmwA | Explosion (13.52) | Fireworks |
