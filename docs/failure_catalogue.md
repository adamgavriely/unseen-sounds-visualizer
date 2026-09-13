# Failure catalogue

Rule 2 of docs/plan_robustness.md: every failure seen in a video is logged here with clip,
stage and confidence. A design change needs >= 3 entries of one pattern, or a metric move
on the dev split. Entries marked FIXED were general bugs, not tuning.

| # | date | clip | what | stage | conf | pattern | status |
|---|---|---|---|---|---|---|---|
| 1 | 09-13 | mc_bridge_scene | real dog bark vetoed as "out of place in a parking lot" | place veto | 0.44 | veto kills real sound | open (1 of 3) |
| 2 | 09-13 | b3_jungle_hike | sheep in a jungle shown | detector | 0.41 | phantom above bar | open (1 of 3) |
| 3 | 09-13 | mv_tornado_scene | phone alert tone never detected | detector vocabulary + masking | - | no AudioSet class; under music | limitations |
| 4 | 09-13 | mv_tornado_scene | siren heard only 10.5-13.5 s, present longer | detector, masked by score/announcer | 0.65 peak | masked by music | -> upgrade B |
| 5 | 09-13 | as_fire_alarm | dog nobody hears | detector | 0.60 | confident phantom | open (1 of 3) |
| 6 | 09-13 | station_passenger | ice cream truck at a station | detector | 0.63 | confident phantom; place veto caught it | veto working as intended |
| 7 | 09-13 | b3_quarry_blast | horse, train, truck; blast missed | detector | 0.84 | confident phantom family | veto caught 2 of 3 |
| 8 | 09-13 | several | nothing detectable before 2 s | detector timing | - | window stamping | FIXED (lead-in pad) |
| 9 | 09-13 | several | pictures a second before / after the sound | detector timing | - | window stamping | FIXED (offset 0.5) |
| 10 | 09-13 | b3_jungle_hike | birdsong merged into an earlier owl | ontology merge | - | family/member merge | FIXED (member explains family only where they overlap) |
| 11 | 09-13 | several | square images cropped in 2-row cells | panel | - | layout | FIXED (fit on white) |
