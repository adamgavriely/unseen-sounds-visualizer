# Which SHIP8 stage-4 step removes 6 strongly heard DEV misses (CPU, caches only)

Script `benchmark/gold/kill_flags.py`, data `benchmark/gold/kill_flags.json`. Stage-4 rows of SHIP8|proposed rebuilt in memory (build, filter_rows) per variant; a sound is *restored* when a same-family row starts in [onset - 0.5, onset + 1.0]. Extra rows = rows with conf >= display threshold (0.35) not in SHIP8, over all 71 DEV clips (both parts; depictable labels in brackets).

Rebuild check (rebuilt == cached rows, label/pre_start/end/conf/origin/rescued, and start): dev 196/196; dev2 88/88

Variants that crashed (flag reverted alone leaves an incoherent config): `-LISTENER_VCACHE` (71 clips)

| sound | in B0r? | killer step (SHIP8) | variants that restore it (row conf) | smallest chain that restores it (row label start conf) |
|---|---|---|---|---|
| dev as_explosion_XJ8lc3I6 Gasp @6.7 | no | minimum span AED_MIN_DUR 0.5 (shared with B0r: BEATs span 6.50-6.75 = 0.25 s; FlexSED 0.758 < bar 0.8); behind it N2 masked weak veto, then DASM clip veto (SHIP8 only) | none | AED_MIN_DUR=0.1, MASKED_WEAK_VETO=False, DASM_CLIP_VETO=None (Gasp 6.5 0.415 unrefined) |
| dev b3_favela_rio Train @14.6 | yes | CONT continuation veto (SHIP8 only) | `B0r` (0.429, 0.401); `-CONTINUATION_VETO` (0.429, 0.401); `-[CONT continuation veto]` (0.429, 0.401); `-[all vetoes (F7 mirror, N2b, DV, K4A-D, CONT, FLAP)]` (0.429, 0.401) | CONTINUATION_VETO=None (Train 14.72 0.429; Rail transport 14.72 0.401) |
| dev2 tg_d029 Chicken, rooster @6.9 | no | mirror veto MIRROR_VETO 0.7 (SHIP8 only); behind it the cross-detector veto FLEXSED_VETO 0.3 (shared: FlexSED never reaches 0.3 for the bird family, so B0r drops it too) | none | MIRROR_VETO=None, FLEXSED_VETO=0.0 (Chicken, rooster 6.75 0.813 unrefined; Fowl 6.75 0.794 unrefined; Cluck 6.75 0.741 unrefined; Crowing, cock-a-doodle-doo 6.75 0.606 unrefined) |
| dev2 tg_d095 Dishes, pots, and pans @16.6 | yes | three SHIP8-only vetoes, each enough alone: mirror veto, N2 masked weak veto, K4A-D (KEEP_NEEDS_V4_ALL); mirror+N2 off gives only Cutlery 0.268 (< display 0.35) | `B0r` (0.373, 0.268); `-[all vetoes (F7 mirror, N2b, DV, K4A-D, CONT, FLAP)]` (0.373, 0.268) | MIRROR_VETO=None, MASKED_WEAK_VETO=False, KEEP_NEEDS_V4_ALL=None (Dishes, pots, and pans 16.5 0.373; Cutlery, silverware 16.5 0.268) |
| dev2 tg_d107 Laughter @8.2 | no | minimum span AED_MIN_DUR 0.5 (shared: BEATs 9.00-9.25 = 0.25 s, FlexSED 8.92-9.40 = 0.48 s) | `+[AED_MIN_DUR 0.5->0.1]` (0.388); `+[AED_MIN_DUR 0.5->0.3]` (0.886); `+[FLEXSED_BAR 0.8->0.75]` (0.886) | AED_MIN_DUR=0.3 (Laughter 8.92 0.886) |
| dev2 tg_d125 Clapping @8.5 | no | FlexSED bar 0.8 + minimum span + PANNs veto on FlexSED-only spans (all shared with B0r): clap frames 0.74-0.80, pieces 0.08 s at bar 0.7 | none | FLEXSED_BAR=0.7, AED_MIN_DUR=0.05, PANNS_VETO=0.0 (Clapping 8.96 0.776; Clapping 9.48 0.792) |

| variant (restores >= 1) | extra rows (depictable) | lost SHIP8 rows (depictable) |
|---|---|---|
| `+[AED_MIN_DUR 0.5->0.1]` | 74 (67) | 3 (3) |
| `+[AED_MIN_DUR 0.5->0.3]` | 3 (3) | 1 (1) |
| `+[FLEXSED_BAR 0.8->0.75]` | 34 (34) | 11 (11) |
| `-CONTINUATION_VETO` | 66 (66) | 1 (1) |
| `-[CONT continuation veto]` | 66 (66) | 1 (1) |
| `-[all vetoes (F7 mirror, N2b, DV, K4A-D, CONT, FLAP)]` | 216 (136) | 1 (1) |
| `B0r` | 211 (131) | 52 (52) |

Variant names: `-FLAG` = SHIP8 with that flag at its B0r value; `-[group]` = a group reverted; `+[step]` = a step SHIP8 shares with B0r loosened. Per-sound trace / listener / filter detail in the JSON.

Not determined (needs GPU, skipped): CAM onset refinement of new BEATs rows; a restored row with no cached refined start keeps its unrefined start (marked 'unrefined'). Listener caches were not refilled for spans a variant creates. `-LISTENER_VCACHE` alone crashes (LISTENER_RULE=TIER needs it). MERGE_GAP, PICTURE_MIN_CONF, GROUP_* (display-only) ignored.
