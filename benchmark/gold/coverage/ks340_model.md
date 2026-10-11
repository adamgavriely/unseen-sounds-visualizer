# KS340: a 'real sound?' model trained on 340 AudioSet clips, applied to our 158

340 clips (340 with AudioSet gold rows), 6006 drawable stage-4 candidates, 1155 real (19.2%); 75 features (constant on the 340, left out: ['q_v2', 'at_other']); families one-hot: 25 most common.
Positives with a candidate start < 0.2 s (easy on 10-s AudioSet clips): 228 of 1155 (35.5% of such rows real).
AUROC 340 (clip-grouped 5-fold): all 0.715; stage-4 survivors 0.726 (n 393, real 65.6%); stage-4 dropped 0.653 (n 5613, real 16.0%).
Note: the all-rows AUROC mixes in the survivor / dropped split (base rates differ a lot); the numbers that matter are survivors (use a) and dropped (use b).
AUROC 158 (never trained on; real = any gold sound of the family, needed or not, any importance, onset in window): all 0.805 (n 5868, real 11.8%); survivors 0.651 (n 594); dropped 0.718 (n 5274).
  for scale, beats_peak alone on the 158 (rows where present, n 2599): AUROC 0.700
  for scale, dasm_local alone on the 158 (rows where present, n 1018): AUROC 0.578
  for scale, flex_peak alone on the 158 (rows where present, n 1302): AUROC 0.762

v1.7 on the 158: 59 hits, 13 wrong (visible 1, cross 9, phantom 3), cost 1.810.
## Use (a): silence a v1.7 sound when P(real) < t

95 drawn v1.7 sounds; 0 with no matching stage-4 survivor (never silenced). P quantiles 10/25/50/75: [0.467, 0.584, 0.718, 0.84]
| t | sounds silenced | hits | wrong | cost |
|---|---|---|---|---|
| 0.02 | 0 | 59 | 13 | 1.810 |
| 0.05 | 0 | 59 | 13 | 1.810 |
| 0.08 | 0 | 59 | 13 | 1.810 |
| 0.1 | 0 | 59 | 13 | 1.810 |
| 0.15 | 2 | 59 | 13 | 1.810 |
| 0.2 | 3 | 58 | 13 | 1.835 |
| 0.25 | 3 | 58 | 13 | 1.835 |
| 0.3 | 3 | 58 | 13 | 1.835 |
| 0.4 | 5 | 58 | 12 | 1.823 |
| 0.5 | 13 | 55 | 11 | 1.886 |

CV (a): choices per fold [None, None, None, None, None]
| part | hits | wrong | cost | v1.7 hits | v1.7 wrong | v1.7 cost |
|---|---|---|---|---|---|---|
| all | 59 | 13 | 1.810 | 59 | 13 | 1.810 |
| DEV | 31 | 4 | 1.634 | 31 | 4 | 1.634 |
| TEST | 28 | 9 | 1.954 | 28 | 9 | 1.954 |

Each v1.7 sound whose silencing alone changes the score (P, +hit, +wrong, clip, label, start), sorted by P:

- (0.18, -1, 0, 'b3_bakery_morning', 'Door', 4.25)
- (0.38, 0, -1, 'w8_hide_wolves_howl_1a', 'Human locomotion', 10.0)
- (0.41, -1, 0, 'm4_airsoft_24a', 'Explosion', 4.5)
- (0.426, -1, 0, 'tg_d032', 'Thunder', 13.75)
- (0.477, 0, -1, 'un_hair_dryer_drying_WWu24rJs', 'Computer keyboard', 11.25)
- (0.49, -1, 0, 'm5_war_fury_25b', 'Helicopter', 0.24)
- (0.511, 0, -1, 'w8_hide_wolves_howl_1a', 'Baby cry, infant cry', 6.32)
- (0.524, -2, 0, 'mv_protest_scene_movie', 'Glass', 10.75)
- (0.545, 0, -1, 'tg_d088', 'Explosion', 10.75)
- (0.546, -1, 0, 'tg_d149', 'Bee, wasp, etc.', 1.25)
- (0.554, -1, 0, 'tg_d016', 'Cough', 5.75)
- (0.574, -1, 0, 'tg_d088', 'Thunder', 13.25)
- (0.575, -1, 0, 'tg_d107', 'Crying, sobbing', 0.38)
- (0.584, 0, -1, 'tg_d023', 'Bee, wasp, etc.', 10.25)
- (0.598, 0, 1, 'mv_protest_scene_movie', 'Baby cry, infant cry', 4.02)
- (0.611, -2, 0, 'ambient_nightlife_neon_97', 'Computer keyboard', 0.66)
- (0.622, -1, 0, 'm5_horror_conjuring_82', 'Bell', 4.88)
- (0.637, -1, 0, 'as_explosion_XJ8lc3I6', 'Gunshot', 18.0)
- (0.651, 0, -1, 'm4_film_1917_33a', 'Gunshot', 1.75)
- (0.659, -1, 0, 'mv_protest_scene_movie', 'Crowd', 0.0)
- (0.66, -1, 0, 'tg_d019', 'Siren', 2.25)
- (0.681, -1, 0, 'ly_ambulance_(siren)_-yPSgCn', 'Siren', 0.0)
- (0.684, -1, 0, 'tg_d040', 'Crowd', 14.48)
- (0.696, -1, 0, 'un_dog_barking_3doKyrCe', 'Siren', 0.06)
- (0.706, -1, 0, 'tg_d120', 'Cat', 0.56)
- (0.709, -1, 0, 'w8_helmetcam_chainsaw_roof_2b', 'Chainsaw', 0.06)
- (0.71, -1, 0, 'mc_bridge_scene', 'Helicopter', 9.22)
- (0.718, 0, -1, 'tg_d146', 'Rowboat, canoe, kayak', 0.14)
- (0.72, 0, -1, 'm4_live_fire_26a', 'Gunshot', 8.64)
- (0.723, -1, 0, 'm5_doc_restrepo_138b', 'Gunshot', 3.0)
- (0.729, -1, 0, 'ambient_nature_rainforest_7629', 'Insect', 0.14)
- (0.737, -1, 1, 'tg_d103', 'Explosion', 8.8)
- (0.749, 0, -1, 'tg_d031', 'Bird', 17.5)
- (0.762, -1, 0, 'tg_d001', 'Duck', 0.14)
- (0.775, -1, 0, 'mv_air_raid_scene', 'Siren', 5.5)
- (0.778, -1, 0, 'ly_helicopter_-v62cK1', 'Helicopter', 0.14)
- (0.779, 0, -1, 'un_driving_motorcycle_DgdHSmwA', 'Explosion', 13.92)
- (0.78, -1, 1, 'tg_d107', 'Laughter', 8.92)
- (0.784, -1, 0, 'tg_d127', 'Laughter', 9.32)
- (0.787, -1, 0, 'mc_bridge_scene', 'Dog', 0.1)
- (0.798, -1, 0, 'w8_dog_fireworks_window_2a', 'Explosion', 0.0)
- (0.81, -1, 0, 'tg_d009', 'Laughter', 8.7)
- (0.815, -1, 0, 'w8_kids_ice_cream_truck_2a', 'Ice cream truck, ice cream van', 0.06)
- (0.816, -1, 0, 'mv_storm_scene_house', 'Alarm', 2.18)
- (0.818, -1, 0, 'm4_airsoft_24a', 'Laughter', 3.0)
- (0.826, 0, -1, 'm4_clay_shoot_11a', 'Explosion', 12.18)
- (0.827, -1, 0, 'tg_d104', 'Explosion', 2.72)
- (0.836, -1, 0, 'tg_d075', 'Alarm', 0.14)
- (0.84, -1, 0, 'mv_detective_crime_scene', 'Alarm', 3.36)
- (0.843, -2, 0, 'as_explosion_XJ8lc3I6', 'Explosion', 5.68)
- (0.843, -1, 0, 'mv_tornado_scene', 'Siren', 9.25)
- (0.869, 0, -1, 'b3_golf_course', 'Bird', 3.8)
- (0.873, -1, 0, 'ambient_snow_walk_930', 'Laughter', 8.08)
- (0.876, -1, 0, 'tg_d133', 'Fart', 0.22)
- (0.879, -1, 0, 'bell_miami', 'Bell', 0.22)
- (0.88, 0, -1, 'm4_dog_doorbell_cam_30a', 'Dog', 3.0)
- (0.882, -1, 0, 'as_fire_alarm_kGKZ0YK4', 'Alarm', 8.89)
- (0.891, -1, 0, 'birds_forest', 'Bird', 10.25)
- (0.9, -1, 0, 'ev_smoke_alarm_kitchen', 'Alarm', 10.0)
- (0.903, -1, 0, 'b3_barbershop', 'Electric shaver, electric razor', 16.0)
- (0.921, -1, 0, 'm4_airsoft_24a', 'Gunshot', 0.0)
- (0.922, -1, 0, 'as_church_bell_pJRAWkLM', 'Bell', 0.14)
- (0.925, -1, 0, 'un_driving_motorcycle_4O3bZRYO', 'Laughter', 13.36)
- (0.941, -1, 0, 'w8_kids_fire_alarm_school_1b', 'Alarm', 0.0)
- (0.943, -1, 0, 'ambient_everyday_farm_2166', 'Bird', 0.14)
- (0.955, -1, 0, 'bell_kazansky', 'Bell', 0.14)
- (0.955, -2, 0, 'w8_pet_parrot_phone_ring_2b', 'Dog', 0.14)
- (0.966, -1, 0, 'ambient_waterfall_hike_18078', 'Bird', 0.14)

## Use (b): add a candidate dropped at a stage-4 step when P(real) >= u

5274 eligible dropped candidates; P quantiles 50/90/95/99: [0.133, 0.302, 0.402, 0.663]
| u | candidates | +hits worst | +wrong worst | cost worst | +hits est | +wrong est | cost est |
|---|---|---|---|---|---|---|---|
| 0.3 | 531 | 0 | 140 | 3.582 | 2.7 | 82.8 | 2.790 |
| 0.4 | 269 | 1 | 74 | 2.722 | 0.9 | 38.0 | 2.268 |
| 0.5 | 152 | 0 | 45 | 2.380 | 0.0 | 22.3 | 2.093 |
| 0.6 | 79 | -1 | 23 | 2.127 | -0.9 | 10.6 | 1.967 |
| 0.7 | 45 | -1 | 11 | 1.975 | -0.9 | 4.4 | 1.889 |
| 0.8 | 18 | 0 | 5 | 1.873 | 0.0 | 1.0 | 1.823 |
| 0.9 | 0 | 0 | 0 | 1.810 | 0.0 | 0.0 | 1.810 |

CV (b) worst case: choices per fold [None, None, None, None, None]
| part | hits | wrong | cost | v1.7 hits | v1.7 wrong | v1.7 cost |
|---|---|---|---|---|---|---|
| all | 59 | 13 | 1.810 | 59 | 13 | 1.810 |
| DEV | 31 | 4 | 1.634 | 31 | 4 | 1.634 |
| TEST | 28 | 9 | 1.954 | 28 | 9 | 1.954 |

CV (b) gate-estimated (expected counts): choices per fold [None, None, None, None, None]
| part | hits | wrong | cost | v1.7 hits | v1.7 wrong | v1.7 cost |
|---|---|---|---|---|---|---|
| all | 59.0 | 13.0 | 1.810 | 59 | 13 | 1.810 |
| DEV | 31.0 | 4.0 | 1.634 | 31 | 4 | 1.634 |
| TEST | 28.0 | 9.0 | 1.954 | 28 | 9 | 1.954 |

Pictures that change the score at u = 0.5 (clip, label, start, +hit, +wrong, gate verdict) and the stage-4 step that dropped them:

- ambient_market_marrakech_3102 ('Crowd', 0.0, 0, 1, None) P 0.521 at panns_clip_veto
- ambient_nature_rainforest_2179 ('Bird', 13.08, 0, 1, None) P 0.518 at dasm_rescue
- ambient_nature_rainforest_2179 ('Bird', 14.96, 0, 1, None) P 0.579 at dasm_rescue
- ambient_nature_rainforest_7629 ('Bird vocalization, bird call, bird song', 6.0, 0, 1, None) P 0.504 at mirror_veto
- ambient_nature_rainforest_7629 ('Bird', 11.24, 0, 1, None) P 0.563 at band_rescue
- ambient_snow_walk_930 ('Train', 0.0, 0, 1, None) P 0.706 at dasm_vote
- b3_airport_terminal ('Crowd', 0.0, 0, 1, None) P 0.622 at panns_clip_veto
- b3_construction_site ('Fixed-wing aircraft, airplane', 0.0, 0, 1, 0.0) P 0.511 at dasm_local_veto
- b3_favela_rio ('Bird', 4.6, 0, 1, None) P 0.701 at finelap_veto
- b3_favela_rio ('Bird', 10.08, 0, 1, None) P 0.761 at dasm_vote
- b3_favela_rio ('Bird', 14.0, 0, 1, None) P 0.631 at mirror_veto
- b3_golf_course ('Bird', 0.0, 0, -1, None) P 0.801 at band_rescue
- b3_golf_course ('Bird', 12.84, 0, 1, None) P 0.68 at band_rescue
- b3_golf_course ('Bird vocalization, bird call, bird song', 16.0, 0, 1, None) P 0.662 at mirror_veto
- b3_golf_course ('Bird', 25.08, 0, 1, None) P 0.842 at rescue_once
- b3_ia_youtube_skxtz9foauw_0 ('Gasp', 22.44, 0, 1, None) P 0.555 at dasm_local_veto
- birds_forest ('Bird', 4.75, 0, 1, None) P 0.564 at continuation_veto
- ly_applause_62ZYD0u ('Giggle', 7.4, 0, 1, None) P 0.77 at dasm_vote
- movie_blueplanet_115 ('Laughter', 0.0, 0, 1, None) P 0.75 at finelap_veto
- mv_arrest_street_scene ('Human locomotion', 1.66, 0, 1, None) P 0.665 at rescue_once
- mv_arrest_street_scene ('Human locomotion', 7.3, 0, 1, None) P 0.547 at rescue_once
- tg_d029 ('Fowl', 3.75, 0, 1, None) P 0.577 at mirror_veto
- tg_d085 ('Laughter', 11.4, 0, 1, 0.0) P 0.83 at dasm_rescue
- tg_d095 ('Thump, thud', 0.16, 0, 1, None) P 0.526 at dasm_vote
- tg_d095 ('Bird', 6.75, 0, 1, None) P 0.533 at mirror_veto
- tg_d095 ('Bird', 13.0, 0, 1, None) P 0.631 at mirror_veto
- tg_d107 ('Laughter', 0.04, -1, 1, None) P 0.651 at flexsed_extract
- tg_d125 ('Laughter', 8.96, 0, 1, None) P 0.879 at dasm_rescue
- un_people_clapping__bAVmK7n ('Shout', 8.4, 0, 1, None) P 0.583 at panns_clip_veto
- ambient_temple_india_1438 ('Crowd', 0.0, 0, 1, None) P 0.742 at dasm_vote
- as_glass_GAGt_UEF ('Gasp', 6.12, 0, 1, None) P 0.519 at flexsed_extract
- b3_balloon_burner ('Laughter', 0.32, 0, 1, None) P 0.537 at dasm_local_veto
- b3_ia_87954yourchancetolivenuc ('Bird', 5.75, 0, 1, None) P 0.622 at masked_weak
- dutch_traffic_kids ('Gasp', 15.24, 0, 1, None) P 0.519 at flexsed_extract
- m4_airsoft_24a ('Explosion', 0.0, -2, 0, None) P 0.711 at finelap_veto
- m4_clay_shoot_11a ('Cluck', 0.0, 0, 1, None) P 0.575 at beats_extract
- m4_clay_shoot_11a ('Gasp', 0.16, 0, 1, None) P 0.587 at band_rescue
- m4_clay_shoot_11a ('Giggle', 0.88, 0, 1, None) P 0.789 at dasm_vote
- m4_clay_shoot_11a ('Laughter', 5.24, 0, 1, 0.0) P 0.887 at dasm_rescue
- m4_film_blackhawk_32a ('Alarm', 7.6, 1, 0, None) P 0.539 at dasm_vote
- m5_war_fury_25b ('Shout', 8.12, 0, 1, None) P 0.613 at panns_clip_veto
- m5_war_fury_25b ('Gunshot', 10.04, 0, 1, None) P 0.511 at rescue_once
- mv_air_raid_scene ('Caw', 0.0, 0, 1, 0.0) P 0.818 at mirror_veto
- tg_d031 ('Laughter', 0.0, 0, 1, 0.0) P 0.644 at beats_extract
- tg_d040 ('Shout', 0.0, 0, 1, None) P 0.534 at flexsed_extract
- tg_d045 ('Bird', 3.16, 0, 1, None) P 0.623 at band_rescue
- tg_d045 ('Chirp, tweet', 13.5, 0, 1, None) P 0.53 at masked_weak
- tg_d080 ('Bird', 5.75, 0, 1, 0.0) P 0.552 at masked_weak
- tg_d080 ('Chicken, rooster', 16.75, 0, 1, None) P 0.574 at k4a_inventory
- tg_d101 ('Caw', 5.0, 1, 0, None) P 0.719 at mirror_veto
- tg_d106 ('Giggle', 7.48, 1, 0, None) P 0.68 at dasm_vote
- tg_d141 ('Bird', 3.56, 0, 1, None) P 0.761 at dasm_vote
- un_dog_barking_3doKyrCe ('Giggle', 14.2, 0, 1, None) P 0.575 at dasm_vote
- un_hair_dryer_drying_jmlpavrE ('Gasp', 18.28, 0, 1, None) P 0.519 at flexsed_extract
- w8_hide_wolves_howl_1a ('Crying, sobbing', 0.08, 0, 1, None) P 0.646 at dasm_rescue
- w8_kids_fire_alarm_school_1b ('Laughter', 13.84, 0, 1, None) P 0.545 at dasm_local_veto

Eligible candidates that sit on a v1.7 miss (right family, in window): 85 in 32 clips; their P: [0.04, 0.05, 0.07, 0.07, 0.07, 0.08, 0.09, 0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.17, 0.17, 0.17, 0.18, 0.18, 0.18, 0.18, 0.19, 0.2, 0.21, 0.21, 0.22, 0.24, 0.24, 0.25, 0.27, 0.27, 0.28, 0.29, 0.32, 0.33, 0.36, 0.36, 0.36, 0.39, 0.4, 0.42, 0.47, 0.48, 0.51, 0.54, 0.54, 0.55, 0.56, 0.56, 0.57, 0.57, 0.61, 0.61, 0.62, 0.64, 0.68, 0.72, 0.78]; share of eligible candidates with a higher P: [0.005, 0.007, 0.009, 0.012, 0.014, 0.014, 0.014, 0.019] (best 8).
Each of those with P >= 0.5 added ALONE (worst case): tg_d029 Fowl/Cluck 6.75, m4_film_blackhawk_32a Alarm 7.6, m5_doc_restrepo_138b Explosion 0.0, tg_d101 Caw 5.0/6.0, tg_d106 Laughter/Giggle 7.5, w8_hide_wolves_howl_1a Crying 1.62 -> +1 hit, 0 wrong each (about 6 misses); but u = 0.5 admits 79 merged pictures (+6 visible, +23 cross, +16 phantom wrongs; display joins and same-family additions cancel the hits), so 'off' wins every fold.
Eligible candidates with P >= 0.5 by dropping step: {'dasm_rescue': 32, 'mirror_veto': 29, 'dasm_vote': 20, 'masked_weak': 10, 'panns_clip_veto': 10, 'flexsed_extract': 10, 'band_rescue': 9, 'dasm_local_veto': 9, 'rescue_once': 9, 'finelap_veto': 6, 'beats_extract': 4, 'continuation_veto': 3, 'k4a_inventory': 1}
