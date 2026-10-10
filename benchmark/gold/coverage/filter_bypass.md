# Which filter drops too many right candidates? (all 158 clips)

Current system: 59 hits / 34 wrong, onset cost 2.076.

## Attribution: let one step's dropped candidates through (on top of the current system)

| step | dropped cands | peak >= q | hits / wrong if bypassed | d hits | d wrong | onset cost |
|---|---|---|---|---|---|---|
| band_rescue | 1554 | 0.0 | 71 / 1209 | +12 | +1175 | 16.646 |
| band_rescue | 1554 | 0.5 | 71 / 1209 | +12 | +1175 | 16.646 |
| band_rescue | 1554 | 0.7 | 63 / 263 | +4 | +229 | 4.873 |
| beats_extract | 1445 | 0.0 | 66 / 778 | +7 | +744 | 11.316 |
| beats_extract | 1445 | 0.5 | 59 / 52 | +0 | +18 | 2.304 |
| beats_extract | 1445 | 0.7 | 59 / 35 | +0 | +1 | 2.089 |
| continuation_veto | 111 | 0.0 | 59 / 62 | +0 | +28 | 2.430 |
| continuation_veto | 111 | 0.5 | 59 / 50 | +0 | +16 | 2.278 |
| continuation_veto | 111 | 0.7 | 59 / 49 | +0 | +15 | 2.266 |
| dasm_clip_veto | 94 | 0.0 | 59 / 82 | +0 | +48 | 2.684 |
| dasm_clip_veto | 94 | 0.5 | 59 / 73 | +0 | +39 | 2.570 |
| dasm_clip_veto | 94 | 0.7 | 59 / 67 | +0 | +33 | 2.494 |
| dasm_local_veto | 131 | 0.0 | 60 / 98 | +1 | +64 | 2.861 |
| dasm_local_veto | 131 | 0.5 | 60 / 89 | +1 | +55 | 2.747 |
| dasm_local_veto | 131 | 0.7 | 60 / 79 | +1 | +45 | 2.620 |
| dasm_rescue | 209 | 0.0 | 63 / 151 | +4 | +117 | 3.456 |
| dasm_rescue | 209 | 0.5 | 63 / 151 | +4 | +117 | 3.456 |
| dasm_rescue | 209 | 0.7 | 61 / 78 | +2 | +44 | 2.582 |
| dasm_vote | 54 | 0.0 | 63 / 73 | +4 | +39 | 2.468 |
| dasm_vote | 54 | 0.5 | 63 / 73 | +4 | +39 | 2.468 |
| dasm_vote | 54 | 0.7 | 62 / 54 | +3 | +20 | 2.253 |
| depict_event | 1 | 0.0 | 59 / 35 | +0 | +1 | 2.089 |
| depict_event | 1 | 0.5 | 59 / 35 | +0 | +1 | 2.089 |
| display_bar | 27 | 0.0 | 59 / 51 | +0 | +17 | 2.291 |
| family_merge | 98 | 0.0 | 63 / 53 | +4 | +19 | 2.215 |
| family_rule | 6 | 0.0 | 59 / 35 | +0 | +1 | 2.089 |
| family_rule | 6 | 0.5 | 59 / 35 | +0 | +1 | 2.089 |
| family_rule | 6 | 0.7 | 59 / 35 | +0 | +1 | 2.089 |
| finelap_veto | 49 | 0.0 | 60 / 67 | +1 | +33 | 2.468 |
| finelap_veto | 49 | 0.5 | 60 / 67 | +1 | +33 | 2.468 |
| finelap_veto | 49 | 0.7 | 59 / 44 | +0 | +10 | 2.203 |
| flexsed_cross_veto | 7 | 0.0 | 60 / 38 | +1 | +4 | 2.101 |
| flexsed_cross_veto | 7 | 0.5 | 60 / 38 | +1 | +4 | 2.101 |
| flexsed_extract | 800 | 0.0 | 60 / 221 | +1 | +187 | 4.418 |
| flexsed_extract | 800 | 0.5 | 60 / 221 | +1 | +187 | 4.418 |
| flexsed_extract | 800 | 0.7 | 60 / 221 | +1 | +187 | 4.418 |
| gate | 104 | 0.0 | 61 / 58 | +2 | +24 | 2.329 |
| gate | 104 | 0.5 | 60 / 55 | +1 | +21 | 2.316 |
| gate | 104 | 0.7 | 60 / 50 | +1 | +16 | 2.253 |
| k4a_inventory | 16 | 0.0 | 60 / 38 | +1 | +4 | 2.101 |
| k4a_inventory | 16 | 0.5 | 60 / 38 | +1 | +4 | 2.101 |
| k4a_inventory | 16 | 0.7 | 59 / 36 | +0 | +2 | 2.101 |
| label_filter | 141 | 0.0 | 60 / 123 | +1 | +89 | 3.177 |
| label_filter | 141 | 0.5 | 60 / 77 | +1 | +43 | 2.595 |
| label_filter | 141 | 0.7 | 59 / 51 | +0 | +17 | 2.291 |
| masked_weak | 313 | 0.0 | 60 / 236 | +1 | +202 | 4.608 |
| mirror_veto | 235 | 0.0 | 63 / 151 | +4 | +117 | 3.456 |
| mirror_veto | 235 | 0.5 | 61 / 76 | +2 | +42 | 2.557 |
| mirror_veto | 235 | 0.7 | 61 / 48 | +2 | +14 | 2.203 |
| panns_clip_veto | 17 | 0.0 | 60 / 43 | +1 | +9 | 2.165 |
| panns_clip_veto | 17 | 0.5 | 60 / 43 | +1 | +9 | 2.165 |
| panns_clip_veto | 17 | 0.7 | 60 / 43 | +1 | +9 | 2.165 |
| rescue_once | 16 | 0.0 | 60 / 44 | +1 | +10 | 2.177 |
| rescue_once | 16 | 0.5 | 60 / 44 | +1 | +10 | 2.177 |
| rescue_once | 16 | 0.7 | 60 / 41 | +1 | +7 | 2.139 |
| scene_margin | 21 | 0.0 | 61 / 39 | +2 | +5 | 2.089 |
| scene_margin | 21 | 0.5 | 60 / 36 | +1 | +2 | 2.076 |
| scene_margin | 21 | 0.7 | 60 / 35 | +1 | +1 | 2.063 |

## Clip-grouped 5-fold CV (selection on training clips only)

- fold 1: [('scene_margin', 0.7)]
- fold 2: [('scene_margin', 0.7)]
- fold 3: []
- fold 4: [('scene_margin', 0.7)]
- fold 5: [('scene_margin', 0.7)]

Out of fold, all 158 clips: 59 hits / 35 wrong, onset cost 2.089 (current 2.076), J 2.948
