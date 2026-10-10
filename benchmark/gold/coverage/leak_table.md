# Leak table: where v1.4 loses needed sounds that a raw candidate had, all 158 clips

needed sounds missed: 65; with no in-window raw candidate of the family: 19; with one: 46

| step that dropped the latest-surviving candidate | sounds |
|---|---|
| band_rescue | 9 |
| kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged | 6 |
| mirror_veto | 5 |
| dasm_vote | 4 |
| beats_extract | 4 |
| family_merge | 3 |
| masked_weak | 3 |
| gate | 2 |
| continuation_veto | 2 |
| scene_margin | 1 |
| label_filter | 1 |
| dasm_local_veto | 1 |
| finelap_veto | 1 |
| panns_clip_veto | 1 |
| family_rule | 1 |
| k4a_inventory | 1 |
| flexsed_extract | 1 |

| clip | gold | step |
|---|---|---|
| ambient_citywalk_nyc_1689 | Vehicle | band_rescue |
| ambient_citywalk_nyc_1689 | Hammer | dasm_vote |
| ambient_nature_rainforest_7629 | Bird | gate |
| as_explosion_XJ8lc3I6 | Walk, footsteps | band_rescue |
| as_explosion_XJ8lc3I6 | Explosion | dasm_vote |
| as_explosion_XJ8lc3I6 | Gasp | beats_extract |
| b3_carnival_parade | Whistle | band_rescue |
| b3_favela_rio | Train | scene_margin |
| b3_pet_shop | Bird | gate |
| birds_forest | Bird | family_merge |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | masked_weak |
| tg_d029 | Chicken, rooster | mirror_veto |
| tg_d030 | Motorcycle | kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged |
| tg_d032 | Thunder | family_merge |
| tg_d033 | Siren | band_rescue |
| tg_d095 | Dishes, pots, and pans | mirror_veto |
| tg_d120 | Meow | label_filter |
| b3_ia_alcf_heydari_0005_0 | Whistle | band_rescue |
| b3_ia_alcf_heydari_0005_0 | Whistle | band_rescue |
| m4_airsoft_24a | Gunshot, gunfire | beats_extract |
| m4_film_1917_33a | Explosion | kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged |
| m4_film_blackhawk_32a | Alarm | dasm_vote |
| m4_fire_bodycam_23a | Siren | dasm_local_veto |
| m4_live_fire_26a | Machine gun | mirror_veto |
| m4_live_fire_26a | Explosion | kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged |
| m4_live_fire_26a | Machine gun | kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged |
| m5_doc_restrepo_138b | Machine gun | family_merge |
| mc_bridge_scene | Door | finelap_veto |
| mc_thriller_basement | Water | dasm_vote |
| tg_d031 | Bird | masked_weak |
| tg_d045 | Ambulance (siren) | masked_weak |
| tg_d046 | Vehicle horn, car horn, honking | beats_extract |
| tg_d046 | Vehicle horn, car horn, honking | continuation_veto |
| tg_d046 | Vehicle horn, car horn, honking | beats_extract |
| tg_d046 | Vehicle horn, car horn, honking | kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged |
| tg_d079 | Tick | band_rescue |
| tg_d101 | Toot | continuation_veto |
| tg_d101 | Crow | mirror_veto |
| tg_d105 | Beep, bleep | band_rescue |
| tg_d106 | Laughter | panns_clip_veto |
| tg_d110 | Chink, clink | family_rule |
| tg_d141 | Neigh, whinny | k4a_inventory |
| w8_film_hunt_for_red_october_1b | Sonar | mirror_veto |
| w8_helmetcam_chainsaw_roof_2b | Alarm | band_rescue |
| w8_hide_wolves_howl_1a | Baby cry, infant cry | flexsed_extract |
| w8_kids_ice_cream_truck_2a | Ice cream truck, ice cream van | kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged |

Undoing a step pays only if at least 1 in 3 of its drops is a real needed sound (one hit = two wrongs).

| step | candidates dropped | in a needed sound's hit window | share |
|---|---|---|---|
| beats_extract | 2338 | 55 | 2% |
| band_rescue | 1591 | 24 | 2% |
| flexsed_extract | 967 | 18 | 2% |
| masked_weak | 733 | 14 | 2% |
| label_filter | 482 | 3 | 1% |
| mirror_veto | 294 | 18 | 6% |
| continuation_veto | 255 | 5 | 2% |
| dasm_rescue | 217 | 19 | 9% |
| gate | 165 | 14 | 8% |
| dasm_local_veto | 136 | 5 | 4% |
| family_merge | 121 | 18 | 15% |
| dasm_clip_veto | 95 | 0 | 0% |
| dasm_vote | 62 | 9 | 15% |
| finelap_veto | 51 | 11 | 22% |
| display_bar | 35 | 0 | 0% |
| scene_margin | 24 | 9 | 38% |
