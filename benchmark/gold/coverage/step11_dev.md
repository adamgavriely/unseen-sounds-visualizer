# Step 11: display policies on top of SHIP8+MD3+WW5+SL|AB-m (DEV)

| policy | hits | wrong (vis / other / none) | onset cost | cost_cov | hits on detector-suggested / plain human labels |
|---|---|---|---|---|---|
| base | 32 | 16 (7 / 7 / 2) | 1.915 | 2.421 | 20 / 12 |
| T | 32 | 16 (7 / 7 / 2) | 1.915 | 2.421 | 20 / 12 |
| A | 32 | 16 (7 / 7 / 2) | 1.915 | 2.421 | 20 / 12 |
| TA | 32 | 16 (7 / 7 / 2) | 1.915 | 2.421 | 20 / 12 |
| F | 32 | 14 (5 / 7 / 2) | 1.859 | 2.364 | 20 / 12 |
| TAF | 32 | 14 (5 / 7 / 2) | 1.859 | 2.364 | 20 / 12 |

Flashes per DEV clip (non-zero): ambient_weather_storm_16200 8, ambient_weather_storm_7200 8, as_explosion_XJ8lc3I6 4, as_fire_alarm_kGKZ0YK4 2, b3_aviary_birds 1, b3_cow_farm 3, b3_crossing_bells 1, b3_favela_rio 4, b3_flea_market 3, birds_forest 1, ly_ambulance_(siren)_-yPSgCn 2, ly_applause_62ZYD0u 1, mv_arrest_street_scene 3, mv_protest_scene_movie 1, mv_storm_scene_house 2, mv_tornado_scene 4, tg_d030 1, tg_d095 1, tg_d125 1, tg_d129 1, tg_d149 1, un_driving_motorcycle_4O3bZRYO 2, un_hair_dryer_drying_WWu24rJs 2, un_people_clapping__bAVmK7n 1

Verdict as pre-registered: T and A change no DEV picture. F removes the two storm Thunder pictures (both "source
visible or obvious"; lightning on screen) and no hit: 32 hits / 14 wrong, onset cost 1.859, cost_cov 2.364. That meets
the general bar (>= 2 fewer wrong, <= 1 hit lost) but NOT F's own pre-registered bar (wrong <= 13), so F is not passed.
