# DASM "Specific impact sounds" rescue, relabelled to the sound both listeners name (PREREG_sis_relabel.md)

44 DASM runs of the family on the 158 clips; both listeners name a common family for 18 (sis_relabel.py, the pipeline's
V4 matcher). Each relabelled run against the gold and the v1.7 pictures:

| clip | start s | new label | effect on v1.7 | rescue path | gold playing |
|---|---|---|---|---|---|
| b3_construction_site | 6.40 | Vehicle | wrong | DR2 (family already in clip); banned | - |
| b3_golf_course | 6.70 | Bird | wrong | DR2 | Bird (importance 1), Whack |
| b3_golf_course | 19.82 | Bird | wrong | DR2 | Bird (importance 1), Arrow (on screen) |
| mv_protest_scene_movie | 5.06 | Screaming | wrong | passes | Crowd, Baby cry (on screen) |
| mv_protest_scene_movie | 11.12 | Chink, clink | already a v1.7 hit | passes | Crowd, Glass |
| mv_protest_scene_movie | 18.66 | Bell | wrong | passes | Crowd, Shatter |
| un_driving_motorcycle_DgdHSmwA | 14.00 | Explosion | same as the v1.7 picture | DR2 | Crowd, Fireworks (on screen) |
| mc_bridge_scene | 4.32 | Car passing by | wrong | passes | Door (on screen) |
| mc_bridge_scene | 5.74 | Car passing by | wrong | passes | Door |
| mv_bank_robbery_alarm | 10.92 | Burst, pop | wrong | passes | Ping (on screen) |
| w8_kids_fire_alarm_school_1b | 7.88 | Alarm | wrong (no new onset) | DR2 | Alarm, Clang |
| tg_d095 | 0.28 | Arrow | wrong | passes | Stir (on screen) |
| tg_d103 | 3.96 | Explosion | wrong | DR2 | Aircraft, Clang |
| tg_d103 | 6.36 | Train | wrong | passes | Aircraft, Clang |
| tg_d103 | 7.66 | Burst, pop | wrong | passes | Aircraft |
| tg_d106 | 5.20 | Cough | wrong | passes | Thump, Scrape (on screen) |
| tg_d110 | 3.14 | Glass | the only possible new hit (Breaking) | DR2 blocks it | Chink (on screen), Breaking |
| tg_d112 | 0.08 | Chink, clink | wrong | passes | Shatter (on screen) |

Ceiling: at most 1 new hit (tg_d110, and the unchanged rescue path's DR2 blocks it); 11 relabelled runs that pass DR2
and the bans land where no needed sound of their family starts. The pre-registered step-1 bar (>= 2 possible new hits)
fails, so the on-screen gate was not run. Not adopted.
