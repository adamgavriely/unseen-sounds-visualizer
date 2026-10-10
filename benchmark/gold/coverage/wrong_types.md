# v1.4 wrong pictures: root causes and trail counter-rules (158 clips)

## Root-cause types (34 wrongs)

| type | n | wrongs |
|---|---|---|
| D. sibling-label confusion (a real sound plays, wrong family) | 11 | Gunshot/Explosion x5 (m4_film_1917_33a 1.8, m4_live_fire_26a 2.5, m4_clay_shoot_11a, tg_d103 1.8, tg_d088 Explosion); mv_air_raid_scene Shofar for siren; tg_d001 Honk for duck; tg_d045 Siren for Ambulance (siren); tg_d107 Screaming for parrot; tg_d146 Rowboat for water; mv_protest_scene_movie Glass 4.8 |
| A. visible source, onset matched, gate missed | 7 | a/b flips: ambient_transport_subway_10800 Train, mv_protest_scene_movie Baby cry, tg_d110 Chink; tg_d128 Laughter, un_driving_motorcycle Explosion, m4_film_1917_33a Explosion 10.9, w8_dashcam_ambulance_behind_1a Siren 0.2 |
| B. re-trigger: later burst of a sound whose source the gate named in some stretch | 6 | as_explosion_XJ8lc3I6 8.2, tg_d088 Thunder 13.2, m4_film_1917_33a Gunshot 11.5, m4_live_fire_26a 8.6, w8_dashcam_ambulance_behind_1a Siren 11.0, tg_d103 Gunshot 8.8 |
| C. right family, picture start outside -0.5..+1 s of a long needed sound | 5 | b3_golf_course Bird, ly_applause Crowd (1.9 s early), tg_d031 Bird, w8_hide_wolves_howl_1a Baby cry, w8_hide_wolves_howl_1a Human locomotion |
| C'. same family as an on-screen sound, late | 2 | w8_dashcam_avalanche_road_2b Siren (Alarm), w8_dashcam_ambulance_behind_1a Shofar |
| E. phantom | 3 | un_hair_dryer Computer keyboard, m4_dog_doorbell_cam_30a Dog, tg_d023 Bee |

Not separable from the trail: C (rescued or two-listener confirmed, good DASM), A without a flip (gate 0 seen),
E Dog / keyboard (strong peaks, DASM >= 0.59, both listeners name them).

## Single rules (all 158 clips)

| rule | hits | wrong | cost | wrongs removed | hits lost |
|---|---|---|---|---|---|
| v1.4 | 59 | 34 | 2.076 | - | - |
| R2d | 59 | 29 | 2.013 | 5 | 0 |
| R2c | 58 | 28 | 2.025 | 6 | 1 |
| R11 | 59 | 32 | 2.051 | 2 | 0 |
| R7 | 56 | 24 | 2.025 | 10 | 3 |
| R7b | 57 | 26 | 2.025 | 8 | 2 |

R7 / R7b read the listener lists over all matched candidates (a family named in any candidate's list counts). The first exploration let the last candidate decide; that also dropped m4_film_1917_33a Explosion 10.9 (wrong) and, for R7 only, the w8_kids_fire_alarm_school_1b Alarm hit (R7b combination then 57 / 19).

## Combinations of {R2d, R11, R7b} (all 158 clips; DEV / TEST hits-wrong)

| rules | hits | wrong | cost | DEV | TEST |
|---|---|---|---|---|---|
| none (v1.4) | 59 | 34 | 2.076 | 31/12 | 28/22 |
| R7b | 57 | 26 | 2.025 | 29/10 | 28/16 |
| R11 | 59 | 32 | 2.051 | 31/12 | 28/20 |
| R11+R7b | 57 | 24 | 2.000 | 29/10 | 28/14 |
| R2d | 59 | 29 | 2.013 | 31/10 | 28/19 |
| R2d+R7b | 57 | 22 | 1.975 | 29/8 | 28/14 |
| R2d+R11 | 59 | 27 | 1.987 | 31/10 | 28/17 |
| R2d+R11+R7b | 57 | 20 | 1.949 | 29/8 | 28/12 |

## Clip-grouped 5-fold CV (choice by training cost)

| fold | test clips | chosen | train cost |
|---|---|---|---|
| 0 | 32 | R2d+R11+R7b | 1.937 |
| 1 | 32 | R2d+R11+R7b | 1.857 |
| 2 | 32 | R11+R7b | 1.746 |
| 3 | 31 | R2d+R11+R7b | 2.031 |
| 4 | 31 | R2d+R11 | 2.173 |

Out-of-fold: 57 hits, 27 wrong, cost 2.038 (v1.4 59 / 34 / 2.076)
Paired clip bootstrap (100000, seed 0), out-of-fold minus v1.4 cost per clip: -0.038 [95% CI -0.139, +0.063], P(diff >= 0) = 0.2626

The rules were found by reading the wrongs of all 158 clips (TEST included); the CV is a sanity check, not a held-out test.
