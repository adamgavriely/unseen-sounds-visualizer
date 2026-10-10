# opusC: audio + decision-logic drop rules on v1.6 (158 clips)

v1.6: 59 hits, 19 wrong, cost 1.886

| rule | t (pre-set) | hits | wrong | cost | dropped wrongs | lost hits | CV out-of-fold hits / wrong / cost | CV t per fold |
|---|---|---|---|---|---|---|---|---|
| GATE_DOUBT_DASM | 0.6 | 59 | 13 | 1.810 | 6: ambient_transport_subway_10800 Train 0.1 (visible); ly_applause_62ZYD0u Crowd 0.0 (cross); mv_protest_scene_movie Baby cry, infant cry 4.0 (visible); tg_d103 Gunshot 1.8 (cross); tg_d110 Chink, clink 3.5 (visible); w8_dashcam_ambulance_behind_1a Siren 0.2 (visible) | 0:  | 59 / 13 / 1.810 | 0.646, 0.748, 0.748, 0.748, 0.748 |
| GATE_SEEN_DASM | 0.6 | 59 | 14 | 1.823 | 5: ambient_transport_subway_10800 Train 0.1 (visible); mv_protest_scene_movie Baby cry, infant cry 4.0 (visible); tg_d103 Gunshot 1.8 (cross); tg_d110 Chink, clink 3.5 (visible); w8_dashcam_ambulance_behind_1a Siren 0.2 (visible) | 0:  | 58 / 14 / 1.848 | 0.646, 0.845, 0.752, 0.752, 0.752 |
| FLEX_CLIP_ALL | 0.1 | 59 | 18 | 1.873 | 1: w8_hide_wolves_howl_1a Human locomotion 10.0 (cross) | 0:  | 59 / 19 / 1.886 | 0.004, 0.089, 0.089, 0.089, 0.089 |
| DASM_CLIP_HARD | 0.084 | 59 | 18 | 1.873 | 1: tg_d088 Explosion 10.8 (cross) | 0:  | 59 / 19 / 1.886 | 0.124, 0.124, 0.060, 0.124, 0.124 |
| V1_ABSENT | 0.05 | 59 | 18 | 1.873 | 1: m4_live_fire_26a Gunshot 2.5 (cross) | 0:  | 56 / 17 / 1.937 | 0.863, 0.058, 0.974, 0.058, 0.058 |
| CONF_BAR | 0.37 | 59 | 17 | 1.861 | 2: un_hair_dryer_drying_WWu24rJs Computer keyboard 11.2 (phantom); tg_d031 Bird 17.5 (cross) | 0:  | 59 / 18 / 1.873 | 0.376, 0.376, 0.376, 0.376, 0.364 |
| AB_CONF | 0.69 | 59 | 15 | 1.835 | 4: mv_protest_scene_movie Baby cry, infant cry 4.0 (visible); tg_d103 Gunshot 1.8 (cross); tg_d110 Chink, clink 3.5 (visible); tg_d146 Rowboat, canoe, kayak 0.1 (cross) | 0:  | 58 / 16 / 1.873 | 0.612, 0.691, 0.691, 0.691, 0.734 |

Rule text:
* GATE_DOUBT_DASM: the on-screen check said 'seen' (majority or description) in >= 1 stretch, yet the picture is drawn: DASM's family max over the picture must be >= t
* GATE_SEEN_DASM: as above, majority 'seen' votes only
* FLEX_CLIP_ALL: FlexSED's family max over the whole clip must be >= t, for every origin (the FlexSED clip veto, 0.3, already applies to some)
* DASM_CLIP_HARD: DASM's family max over the whole clip must be >= t, no listener override
* V1_ABSENT: Qwen V1 multiple choice p(family) must be >= t when it was asked
* CONF_BAR: stage-5 confidence >= t (display bar is 0.35)
* AB_CONF: an a/b override of a majority-'visible' verdict needs stage-5 confidence >= t

Stack GATE_DOUBT_DASM + FLEX_CLIP_ALL: pre-set t -> 59 hits, 12 wrong, cost 1.797; CV out-of-fold 59 hits, 13 wrong, cost 1.810; CV t per fold [[0.6465, 0.003999], [0.7475, 0.089], [0.7475, 0.089], [0.7475, 0.089], [0.7475, 0.089]]

Stack GATE_DOUBT_DASM + FLEX_CLIP_ALL + DASM_CLIP_HARD + V1_ABSENT: pre-set t -> 59 hits, 10 wrong, cost 1.772; CV out-of-fold 56 hits, 11 wrong, cost 1.861; CV t per fold [[0.6465, 0.003999, 0.124, 0.8634128719568253], [0.7475, 0.089, 0.124, 0.0584556243265979], [0.7475, 0.089, 0.059999, 0.9736461043357849], [0.7475, 0.089, 0.124, 0.0584556243265979], [0.7475, 0.089, 0.124, 0.0584556243265979]]
