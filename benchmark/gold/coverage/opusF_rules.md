# Opus F: conjunction rules to recover v1.7 misses, all 158 clips

v1.7: 59 hits, 13 wrong, cost 1.810. Misses 65; with an in-window dropped candidate of the family 44.

Gate survival (plan B drawn -> still drawn after the v1.7 on-screen decision): hit 0.90 [52, 58], visible 0.31 [11, 36], other 0.52 [31, 60] (an added wrong uses 'visible' or 'other' by its own type).
Bar: a rule pays when it gains >= 2 hits per extra wrong (cost: 1 hit = 4, 1 wrong = 2).

## Rule A: dropped at dasm_vote, both listeners name it, FineLAP >= t

| t | +hits | +wrong (worst) | +hits est | +wrong est | cost (worst) |
|---|---|---|---|---|---|
| 0.5 | 2 | 13 | 1.8 | 5.8 | 1.924 |
| 0.6 | 2 | 13 | 1.8 | 5.8 | 1.924 |
| 0.65 | 2 | 12 | 1.8 | 5.3 | 1.911 |
| 0.7 | 2 | 10 | 1.8 | 3.7 | 1.886 |
| 0.75 | 2 | 9 | 1.8 | 3.2 | 1.873 |
| 0.8 | 1 | 8 | 0.9 | 2.7 | 1.886 |
| 0.9 | 0 | 4 | 0.0 | 1.9 | 1.861 |

CV choices [None, None, None, None, None]; out of fold +hits 0, +wrong 0, cost 1.810

Pictures that change the score at t = 0.9 (label, start, +hit, +wrong, gate verdict):

- ('ambient_snow_walk_930', 'Train', 0.0, 0, 1, None)
- ('ambient_temple_india_1438', 'Crowd', 0.0, 0, 1, None)
- ('as_fireworks_jZ9mYtCA', 'Explosion', 9.04, 0, 1, None)
- ('m5_horror_conjuring_82', 'Clock', 8.6, 0, 1, None)

## Rule B: dropped anywhere after the listeners were asked, both listeners name it, DASM (span +- 0.5 s) >= t

| t | +hits | +wrong (worst) | +hits est | +wrong est | cost (worst) |
|---|---|---|---|---|---|
| 0.2 | 0 | 37 | 0.9 | 16.6 | 2.278 |
| 0.3 | -1 | 37 | 0.0 | 16.1 | 2.304 |
| 0.4 | -2 | 30 | -0.9 | 13.7 | 2.241 |
| 0.5 | -2 | 25 | -0.9 | 11.4 | 2.177 |
| 0.6 | -1 | 21 | -0.9 | 9.3 | 2.101 |
| 0.7 | 0 | 14 | 0.0 | 5.7 | 1.987 |

CV choices [None, None, None, None, None]; out of fold +hits 0, +wrong 0, cost 1.810

Pictures that change the score at t = 0.7 (label, start, +hit, +wrong, gate verdict):

- ('b3_construction_site', 'Truck', 7.75, 0, 1, 0.0)
- ('b3_construction_site', 'Truck', 17.5, 0, 1, 0.0)
- ('ly_applause_62ZYD0u', 'Laughter', 7.4, 0, 1, None)
- ('movie_blueplanet_115', 'Laughter', 0.0, 0, 1, None)
- ('mv_arrest_street_scene', 'Footsteps', 0.12, 0, 1, 0.0)
- ('tg_d020', 'Rain on surface', 7.0, 0, 1, None)
- ('tg_d085', 'Laughter', 11.4, 0, 1, 0.0)
- ('as_alarm_71Xl_uAS', 'Alarm', 15.58, 0, 1, None)
- ('as_alarm_71Xl_uAS', 'Alarm', 18.16, 0, 1, None)
- ('m4_airsoft_24a', 'Explosion', 0.0, -2, 0, None)
- ('m4_clay_shoot_11a', 'Laughter', 1.12, 0, 1, None)
- ('m4_clay_shoot_11a', 'Laughter', 5.24, 0, 1, 0.0)
- ('m4_film_1917_33a', 'Boom', 10.5, 0, 1, 1.0)
- ('m4_live_fire_26a', 'Explosion', 9.94, 1, 0, None)
- ('m5_doc_restrepo_138b', 'Fusillade', 0.06, 1, 0, None)
- ('m5_war_fury_25b', 'Explosion', 10.1, 0, 1, None)
- ('m5_war_fury_25b', 'Gunshot', 10.1, 0, 1, None)
- ('tg_d046', 'Car', 3.75, 0, 1, 1.0)
- ('un_dog_barking_3doKyrCe', 'Howl', 12.75, 0, 1, 0.0)

## Rule C: dropped at band_rescue, Audio Flamingo names it, FlexSED run peak >= t

| t | +hits | +wrong (worst) | +hits est | +wrong est | cost (worst) |
|---|---|---|---|---|---|
| 0.6 | 3 | 45 | 2.7 | 22.9 | 2.304 |
| 0.65 | 2 | 24 | 1.8 | 11.8 | 2.063 |
| 0.7 | 2 | 13 | 1.8 | 6.3 | 1.924 |
| 0.75 | 1 | 7 | 0.9 | 3.2 | 1.873 |

CV choices [None, None, None, None, None]; out of fold +hits 0, +wrong 0, cost 1.810

Pictures that change the score at t = 0.75 (label, start, +hit, +wrong, gate verdict):

- ('as_glass_oHil9Ip_', 'Chink, clink', 8.96, 0, 1, None)
- ('b3_golf_course', 'Bird', 0.0, 0, -1, None)
- ('london_protest_01', 'Crowd', 0.04, 0, 1, None)
- ('mv_tornado_scene', 'Screaming', 17.48, 0, 1, None)
- ('tg_d022', 'Alarm', 0.0, 0, 1, None)
- ('as_glass_GAGt_UEF', 'Chink, clink', 19.76, 0, 1, None)
- ('b3_ia_alcf_heydari_0005_0', 'Whistle', 12.84, 1, 0, None)
- ('mc_riot_scene', 'Boat', 0.0, 0, 1, None)
- ('tg_d045', 'Bird', 8.72, 0, 1, None)
- ('tg_d104', 'Screaming', 0.0, 0, 1, None)

## Rule D: dropped at band_rescue / dasm_rescue / family_merge / continuation_veto, one listener names it, DASM >= t

| t | +hits | +wrong (worst) | +hits est | +wrong est | cost (worst) |
|---|---|---|---|---|---|
| 0.4 | 3 | 53 | 2.8 | 23.5 | 2.405 |
| 0.5 | 3 | 52 | 2.8 | 22.5 | 2.392 |
| 0.6 | 3 | 46 | 2.8 | 19.4 | 2.316 |
| 0.7 | 4 | 27 | 4.6 | 10.8 | 2.051 |
| 0.8 | 3 | 14 | 3.7 | 7.3 | 1.911 |
| 0.9 | 3 | 8 | 2.8 | 3.6 | 1.835 |

CV choices [0.9, None, None, None, None]; out of fold +hits 0, +wrong 5, cost 1.873

Pictures that change the score at t = 0.9 (label, start, +hit, +wrong, gate verdict):

- ('ly_applause_62ZYD0u', 'Laughter', 7.4, 0, 1, None)
- ('tg_d085', 'Waves, surf', 5.0, 0, 1, None)
- ('tg_d085', 'Laughter', 11.4, 0, 1, 0.0)
- ('tg_d085', 'Ocean', 11.76, 0, 1, None)
- ('waves_herzliya', 'Ocean', 1.5, 0, 1, 0.0)
- ('waves_herzliya', 'Waterfall', 5.0, 0, 1, None)
- ('m4_clay_shoot_11a', 'Laughter', 1.12, 0, 1, None)
- ('m4_clay_shoot_11a', 'Laughter', 5.24, 0, 1, 0.0)
- ('m4_live_fire_26a', 'Explosion', 7.26, 1, 0, None)
- ('m5_doc_restrepo_138b', 'Explosion', 0.0, 1, 0, None)
- ('tg_d046', 'Car', 1.5, 0, 1, 1.0)
- ('tg_d046', 'Vehicle horn, car horn, honking', 13.55, 1, 0, 1.0)

## Rule E: dropped at mirror_veto, BEATs peak >= t (no listener asked there)

| t | +hits | +wrong (worst) | +hits est | +wrong est | cost (worst) |
|---|---|---|---|---|---|
| 0.3 | 1 | 61 | 1.8 | 33.4 | 2.557 |
| 0.4 | 0 | 48 | 0.0 | 24.8 | 2.418 |
| 0.5 | 0 | 36 | 0.0 | 19.4 | 2.266 |
| 0.6 | 0 | 24 | 0.0 | 11.3 | 2.114 |
| 0.7 | 2 | 12 | 1.8 | 5.4 | 1.911 |
| 0.8 | 1 | 9 | 0.9 | 3.8 | 1.899 |

CV choices [None, None, None, None, None]; out of fold +hits 0, +wrong 0, cost 1.810

Pictures that change the score at t = 0.8 (label, start, +hit, +wrong, gate verdict):

- ('b3_crossing_bells', 'Hiss', 0.0, 0, 1, None)
- ('mv_arrest_street_scene', 'Shatter', 11.25, 0, 1, None)
- ('mv_protest_scene_movie', 'Shatter', 6.0, 0, 1, None)
- ('mv_tornado_scene', 'Neigh, whinny', 17.5, 0, 1, None)
- ('tg_d022', 'Pant', 0.0, 0, 1, None)
- ('tg_d029', 'Chicken, rooster', 6.75, 1, 0, None)
- ('tg_d075', 'Shatter', 5.5, 0, 1, None)
- ('mv_air_raid_scene', 'Caw', 0.0, 0, 1, 0.0)
- ('tg_d068', 'Cattle, bovinae', 1.0, 0, 1, None)
- ('tg_d068', 'Cattle, bovinae', 8.25, 0, 1, None)
- ('w8_film_hunt_for_red_october_1b', 'Sonar', 6.0, 0, 1, None)

## Stage-5 pool: sounds plan B draws and v1.7 does not (exact)

| clip | label | start | +hit | +wrong | gate yes votes / asked | v1.7 reason |
|---|---|---|---|---|---|---|
| mv_storm_scene_house | Siren | 2.6 | 0 | 0 | 0/6 | same picture as Alarm (fully explained b |
| un_driving_motorcycle_4O3bZRYO | Air horn, truck horn | 8.44 | 0 | 1 | 0/6 | a kind of Vehicle, whose source is visib |
| un_driving_motorcycle_DgdHSmwA | Gunshot | 13.52 | 0 | 0 | 0/3 | same picture as Explosion (similarity 0. |
| un_driving_motorcycle_DgdHSmwA | Fireworks | 13.97 | 0 | 0 | 0/3 | no depiction reads as this sound - dropp |
| tg_d001 | Quack | 0.14 | 0 | 0 | 0/3 | same picture as Duck (same sound under t |
| tg_d001 | Goose | 0.14 | 0 | 0 | 0/3 | same picture as Duck (similarity 0.86) |
| tg_d023 | Insect | 10.25 | 0 | 1 | 0/3 | same picture as Bee, wasp, etc. (same so |
| tg_d110 | Glass | 4.5 | 0 | 1 | 0/3 | a kind of Chink, clink, whose source is  |
| w8_hide_wolves_howl_1a | Crying, sobbing | 6.28 | 0 | 1 | 0/3 | same picture as Baby cry, infant cry (sa |
| ly_helicopter_-v62cK1 | Aircraft | 0.0 | 0 | 0 | 1/12 | same picture as Helicopter (same sound u |
| tg_d127 | Water | 0.14 | 0 | 0 | 1/3 | salient non-speech sound, source not vis |
| tg_d146 | Water | 0.0 | 0 | 0 | 1/3 | stage 2 saw the source somewhere in the  |
| w8_dog_fireworks_window_2a | Fireworks | 0.22 | 0 | 0 | 4/9 | no depiction reads as this sound - dropp |
| ambient_citywalk_nyc_2627 | Vehicle | 0.3 | 0 | 0 | 2/3 | source visible on screen (yellow taxi) - |
| mv_arrest_street_scene | Footsteps | 0.0 | 0 | 1 | 2/3 | source visible on screen (old man) - sta |
| mv_storm_scene_house | Explosion | 17.84 | 0 | 1 | 2/3 | source visible on screen (nothing) - sta |
| tg_d016 | Vehicle | 0.75 | 0 | 0 | 2/3 | source visible on screen (black car) - s |
| tg_d054 | Vehicle | 1.0 | 0 | 0 | 2/3 | source visible on screen (John Deere tra |
| tg_d085 | Laughter | 11.52 | 0 | 1 | 2/3 | source visible on screen (woman in blue  |
| tg_d088 | Rain | 13.75 | 0 | 0 | 4/6 | source visible on screen (rain) - stay s |
| as_alarm_71Xl_uAS | Alarm | 7.5 | 0 | 1 | 6/9 | source visible on screen (sony alarm clo |
| as_fireworks_jZ9mYtCA | Crowd | 0.6 | 0 | 1 | 2/3 | source visible on screen (crowd of peopl |
| m4_clay_shoot_11a | Laughter | 5.44 | 0 | 1 | 2/3 | source visible on screen (man in black j |
| movie_junglebook_95 | Roar | 6.0 | 0 | 1 | 2/3 | source visible on screen (bear) - stay s |
| tg_d019 | Alarm | 10.0 | 0 | 1 | 2/3 | source visible on screen (siren) - stay  |
| tg_d045 | Vehicle | 0.0 | 0 | 0 | 2/3 | source visible on screen (black honda se |
| tg_d104 | Gunshot | 5.56 | 0 | 0 | 2/3 | source visible on screen (rifle) - stay  |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 0.0 | 0 | 0 | 7/9 | source visible on screen (nothing) - sta |
| tg_d079 | Vehicle | 1.0 | 0 | 0 | 7/9 | source visible on screen (nothing) - sta |
| tg_d085 | Water | 8.39 | 0 | 0 | 5/6 | source visible on screen (ocean waves) - |
| tg_d013 | Water | 0.0 | 0 | 0 | 5/6 | source visible on screen (water) - stay  |
| ambient_nature_rainforest_7629 | Bird | 0.81 | 0 | 0 | 3/3 | source visible on screen (macaws) - stay |
| ambient_weather_storm_16200 | Rain | 0.06 | 0 | 0 | 9/9 | source visible on screen (rain) - stay s |
| ambient_weather_storm_7200 | Rain | 0.06 | 0 | 0 | 9/9 | source visible on screen (heavy rain) -  |
| b3_aviary_birds | Bird | 0.3 | 0 | 1 | 15/15 | source visible on screen (birds) - stay  |
| b3_construction_site | Vehicle | 0.0 | 0 | 0 | 12/12 | source visible on screen (excavator) - s |
| b3_crossing_bells | Train | 0.0 | 0 | 1 | 6/6 | source visible on screen (train) - stay  |
| b3_pet_shop | Bird | 0.14 | 1 | 0 | 18/18 | source visible on screen (birds) - stay  |
| ly_applause_62ZYD0u | Laughter | 0.3 | 0 | 0 | 3/3 | source visible on screen (woman) - stay  |
| mv_arrest_street_scene | Human locomotion | 0.1 | 0 | 1 | 3/3 | source visible on screen (man) - stay si |
| un_driving_motorcycle_4O3bZRYO | Vehicle | 4.08 | 0 | 0 | 6/6 | source visible on screen (white car) - s |
| waterfall_kawaida_01 | Water | 0.1 | 0 | 0 | 6/6 | source visible on screen (waterfall) - s |
| waves_herzliya | Water | 0.14 | 0 | 0 | 3/3 | source visible on screen (ocean waves) - |
| ambient_everyday_farm_2166 | Sheep | 14.75 | 0 | 1 | 3/3 | source visible on screen (two sheep) - s |
| ambient_market_bangkok_4332 | Vehicle | 10.0 | 0 | 0 | 3/3 | source visible on screen (tuk-tuk) - sta |
| lx_chainsaw_and_power_tool_BBukw6J | Chainsaw | 0.0 | 0 | 1 | 12/12 | source visible on screen (chainsaw) - st |
| lx_chainsaw_and_power_tool_W1VYWwY | Chainsaw | 0.22 | 0 | 1 | 9/9 | source visible on screen (chainsaw) - st |
| m4_film_blackhawk_32a | Vehicle | 4.0 | 0 | 0 | 3/3 | source visible on screen (military helic |
| m5_war_fury_25b | Gunshot | 5.28 | 0 | 0 | 3/3 | source visible on screen (tank cannon) - |
| mv_air_raid_scene | Crow | 0.0 | 0 | 1 | 3/3 | source visible on screen (crow) - stay s |
| mv_bank_robbery_alarm | Vehicle | 0.0 | 0 | 0 | 6/6 | source visible on screen (white van) - s |
| tg_d031 | Laughter | 0.0 | 0 | 2 | 6/6 | source visible on screen (group of peopl |
| tg_d031 | Giggle | 8.4 | 0 | 1 | 3/3 | source visible on screen (the people) -  |
| tg_d040 | Aircraft | 2.06 | 0 | 0 | 3/3 | source visible on screen (United Airline |
| tg_d076 | Vehicle | 0.0 | 0 | 0 | 3/3 | source visible on screen (silver pickup  |
| tg_d080 | Bird | 5.75 | 0 | 1 | 3/3 | source visible on screen (chicken) - sta |
| tg_d109 | Train | 0.0 | 0 | 1 | 6/6 | source visible on screen (steam train) - |
| tg_d141 | Horse | 0.06 | 0 | 1 | 3/3 | source visible on screen (brown horse) - |
| un_dog_barking_3doKyrCe | Dog | 6.75 | 0 | 1 | 6/6 | source visible on screen (brown dog) - s |
| w8_dashcam_ambulance_behind_1a | Vehicle | 12.5 | 0 | 0 | 6/6 | source visible on screen (yellow van) -  |
- rule G (yes share <= 0.2): +hits 0, +wrong 4 (n 10)
- rule G (yes share <= 0.34): +hits 0, +wrong 4 (n 12)
- rule G (yes share <= 0.5): +hits 0, +wrong 4 (n 13)
- rule G (yes share <= 0.67): +hits 0, +wrong 12 (n 27)

## Each v1.5-v1.7 display rule switched off alone (exact)

| flag | +hits | +wrong | clips gaining |
|---|---|---|---|
| VISIBILITY_RULE | -6 | 3 |  |
| FLASH_RULE | 0 | 3 |  |
| PICTURE_BAN | 1 | 3 | tg_d030 |
| HOLD_FLEXSED | 0 | 0 |  |
| REPEAT_LOCK | 0 | 2 |  |
| UNVERIFIABLE_BAN | 0 | 0 |  |
| COONSET_CONTEST | 0 | 3 |  |
| WEAK_NO_RISE | 0 | 3 |  |
| BEATS_NO_RISE | 0 | 1 |  |
| GATE_DOUBT_DASM | 0 | 6 |  |
