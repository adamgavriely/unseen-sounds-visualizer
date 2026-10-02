# DEV misses and wrong pictures of B0 (scored render, ours with gate) — 2026-09-28

*Diagnostic, DEV only (49 clips, 36 needed sounds). No TEST data. B0 = `dev_monocap_v31` with the PANNs veto 0.05, the arm that the DEV check calls B0 (docs/dev_candidates_check_2026-09-28.md): 14 hits, 22 misses, 24 wrong pictures (6 visible / 11 cross / 7 phantom). All three numbers are reproduced here with `score_per_sound`. Data: `benchmark/gold/dev_miss_table.json`.*

## How it was made
- **Pictures**: `data/work/protocol_proposed_dev_monocap_v31/<clip>/augmentations.json`, copied from the cluster. They equal `devcand/B0r_proposed` on 49 / 49 clips. Read with `score_per_sound.load_pictures`. Hit or miss per needed sound: `dev_candidates_check.needed_hit` (same hits as `score_clip`). The type of each wrong picture = the change in `score_clip`'s counts when that picture is added (pictures in start order, the scorer's own order).
- **Stage 4**: arm `B0r|proposed` of `data/work/devcand/stage4.json`. These are the spans after the vetoes and onset refinement; origin *tagger* = BEATs, *flex* = FlexSED-only. The scored run's `onset_trace.json` gives the FlexSED 0.8 spans before the vetoes. `gate_votes.json` and each picture spec's `reason` give stage 5.
- **Scores**: BEATs `data/work/j2_dev_beats` (2-s windows, 0.25-s hop; time = window start). FlexSED `data/work/flexsed_cache` (25 fps). PANNs `benchmark/gold/panns_fw` (100 fps). Each score = the best same-family frame (same canonical label, or ancestor / descendant below the top level: `score_per_sound.same_family`, the hit rule) with frame time in the hit window **[onset − 0.5, onset + 1.0] s**, as in the 09-27 table. *FlexSED run* = the longest stretch ≥ 0.4 of the best FlexSED label that touches the window. *Near* = [onset − 1, onset + 2] s.
- Scripts: `benchmark/gold/miss_table/` (miss_feats.py -> build_out.py -> build_md.py); inputs partly copied read-only from the cluster.

**Buckets.** The first match wins, in this order. Every other match is listed under *also*.
1. **(e)** Stage 4 has a same-family span that could be shown (conf ≥ 0.35, or FlexSED-only) and it **starts in the window**, but there is no picture: the gate or stage 5 removed it.
2. **(f)** A same-family span or picture that could be shown **covers the onset** (it starts earlier and is still on), or it starts up to 1 s after the window. A much later picture of the same sound does not count.
3. **(h)** A FlexSED 0.8 span near the onset is in the trace's `flexsed_raw` step but not in its `veto` step, and it has no same-family BEATs twin: the stage-4 clip veto removed it.
4. **(d)** A same-family BEATs span with 0.175 ≤ conf < 0.35 starts in the window (emitted, but below the display bar). *d-score* under *also* = BEATs is 0.175–0.35 in the window, but there is no span.
5. **(c)** FlexSED at a 0.4 bar (`_extract_events`, 0.5-s min span) gives a same-family span that starts in the window, peak < 0.8.
6. **(b)** FlexSED is 0.4–0.8 in the window, but gives no such span (its run is shorter than 0.5 s).
7. **(g)** A span of a sibling label (same direct parent), conf ≥ 0.35, starts in the window.
8. **(a)** BEATs < 0.3 and FlexSED < 0.4 near the onset. Anything else is **(h)**.

## Counts

| bucket | what | misses |
|---|---|---|
| (a) | unheard | 4 |
| (b) | FlexSED 0.4-0.8, blocked by 0.5-s min span | 4 |
| (c) | FlexSED 0.4-0.8 span reachable | 6 |
| (d) | BEATs 0.175-0.35 (emitted, below display) | 2 |
| (e) | emitted, removed by gate / stage 5 | 3 |
| (f) | emitted, onset outside window | 2 |
| (g) | emitted, wrong (near) family | 0 |
| (h) | other (here: the stage-4 veto) | 1 |
| | **total** | **22** |

Wrong pictures: 6 visible, 11 cross, 7 phantom. 20 come from BEATs spans, 4 from FlexSED-only spans. 7 of the 11 cross pictures show the right family of a real gold sound, but not at its onset.

## Where the levers are
1. **FlexSED 0.4–0.8 band: 10 of 22** (c 6, b 4). Of the 6 reachable spans, only the 2 explosions pass R1's own filter. The other 4 (air horn, hammer, footsteps, whistle) are removed by the BEATs self-veto, because BEATs does not hear them at all (clip peak ≤ 0.055). So a band rescue needs a check that does not lean on BEATs. The 4 (b) sounds are short: FlexSED runs of 0.08–0.24 s, and two of them reach only 0.42–0.43.
2. **The gate says visible: 3** (e: Bird ×2, Bell). The detectors heard them well (BEATs 0.46–0.86). The gate named a source on screen (macaws, a small bird, a church bell) where the annotator said it is not visible. No detector change helps these.
3. **Heard at the onset, then lost in stage-4 bookkeeping: 5** (d 2, f 2, h 1). The twin rule keeps the BEATs conf 0.31 of an Insect span that FlexSED scores 0.93 (Cricket). The display bar is 0.35 but BEATs gives 0.22 (Bird). A long or merged span starts too early (Crowd, Vehicle). Both vetoes drop a FlexSED 0.92 Laughter. Two of these also make a cross wrong picture (Cricket at 4.5 s, Crowd at 0.0 s), so one fix there could turn 2 misses into hits and remove 2 wrong pictures.
4. **Unheard: 4** (a: Hammer, Clang, golf Whack ×2): no detector reaches 0.3 / 0.4. **Wrong family: 0** (g). Only 3 misses have another family's span at the onset, and none of those became a picture, so relabelling does not help the misses.
5. **Wrong pictures: 20 of 24 come from BEATs.** 6 visible sounds pass the gate (thunder ×2, church bell, fire alarm, a vehicle on a visible air horn, fireworks). 7 of 11 cross pictures show the right family at the wrong time (a ringing phone labelled Alarm ×2, a second shaver picture, Bird, Cricket, Crowd, Gunshot before a visible machine gun): this is a cost of the onset rule, not a label error. Right timing would stop 6 of the 7 costing as false alarms, but only 2 would turn into hits (Cricket, Crowd): Alarm x2 and the shaver would become duplicates, the Bird would be don't-care (importance 1), and the Gunshot would still be a visible false alarm (the machine gun is on screen). 3 of the 7 phantoms are one BEATs Vehicle label on washing-machine rumble in `b3_laundromat`.

## 1. The 22 missed needed sounds

Scores = the best same-family frame in the hit window (the label is in brackets when the score is ≥ 0.1). *Cross-label* = another family's stage-4 span (salient, conf ≥ 0.175) that starts in the window; *shown* = conf ≥ 0.35.

| # | clip | gold label | onset s | dur s | imp | BEATs | FlexSED | FlexSED run ≥ 0.4 (s) | PANNs | bucket | also | cross-label at onset | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ambient_citywalk_nyc_1689` | Vehicle | 3.8 | 0.5 | 2 | 0.10 (Vehicle horn, car horn, honking) | 0.71 (Air horn, truck horn) | 0.64 | 0.04 | **(c)** FlexSED 0.4-0.8 span reachable | — | — | FlexSED Air horn 0.71, span 3.76-4.40 s. Not an R1 band candidate: BEATs self-veto (Air horn clip peak 0.04 < 0.1218). |
| 2 | `ambient_citywalk_nyc_1689` | Hammer | 13.7 | 2.3 | 2 | 0.03 | 0.70 (Hammer) | 0.84 | 0.01 | **(c)** FlexSED 0.4-0.8 span reachable | — | — | FlexSED Hammer span 14.20-15.04 s. Not an R1 band candidate: self-veto (BEATs Hammer clip peak 0.055). |
| 3 | `ambient_citywalk_nyc_1689` | Hammer | 8.1 | 2.6 | 2 | 0.01 | 0.01 | 0.00 | 0.00 | **(a)** unheard | — | — | No detector hears it. |
| 4 | `ambient_citywalk_nyc_2627` | Clang | 3.8 | 0.6 | 2 | 0.01 | 0.00 | 0.00 | 0.00 | **(a)** unheard | — | — | No detector hears it. |
| 5 | `ambient_nature_rainforest_2179` | Bird | 6.5 | 9.5 | 2 | 0.03 | 0.43 (Bird) | 0.08 | 0.02 | **(b)** FlexSED 0.4-0.8, blocked by 0.5-s min span | — | — | FlexSED 0.43 for 0.08 s only. |
| 6 | `ambient_nature_rainforest_7629` | Bird | 0.1 | 15.9 | 2 | 0.42 (Bird vocalization, bird call, bird song) | 0.68 (Bird) | 1.64 | 0.05 | **(e)** emitted, removed by gate / stage 5 | d, c | Insect 0.14 s (0.25); Cricket 0.14 s (0.20); Cricket 1.00 s (0.32); Insect 0.14 s (0.31); Insect 0.14 s (0.30); Insect 0.14 s (0.23) | BEATs Bird vocalization span at 0.89 s (0.46) in the window; the gate said visible ('macaws'). Gold says not visible. |
| 7 | `ambient_nature_rainforest_7629` | Cricket | 0.1 | 9.4 | 2 | 0.32 (Cricket) | 0.92 (Insect) | 16.00 | 0.00 | **(d)** BEATs 0.175-0.35 (emitted, below display) | — | Bird vocalization, bird call, bird song 0.14 s (0.26); Bird 0.14 s (0.26); Bird vocalization, bird call, bird song 0.89 s (0.46, shown); Bird 0.81 s (0.36, shown); Chirp, tweet 0.89 s (0.34) | BEATs Insect span at 0.14 s has conf 0.31 < 0.35. FlexSED Insect 0.93 (raw span 0-10 s at 0.8) was absorbed into it by the twin rule, which keeps the BEATs conf. A Cricket picture comes only at 4.50 s (counted as a cross wrong picture). |
| 8 | `ambient_snow_walk_930` | Laughter | 8.1 | 1.3 | 2 | 0.21 (Snicker) | 0.92 (Laughter) | 1.60 | 0.00 | **(h)** stage-4 second-detector veto | c, d-score | — | FlexSED Laughter 0.92, raw span 8.08-8.76 s at the 0.8 bar, dropped by the PANNs clip veto (PANNs Laughter clip peak 0.001 < 0.05). The shipped self-veto drops it too (BEATs Laughter clip peak 0.105 < 0.1218). |
| 9 | `as_explosion_XJ8lc3I6` | Gunshot, gunfire | 0.0 | 2.2 | 3 | 0.21 (Fusillade) | 0.55 (Gunshot) | 0.24 | 0.24 (Fusillade) | **(b)** FlexSED 0.4-0.8, blocked by 0.5-s min span | d-score | — | FlexSED Gunshot 0.55 for 0.24 s. BEATs Fusillade 0.21 for one 0.25-s frame, also below the min span. |
| 10 | `as_explosion_XJ8lc3I6` | Walk, footsteps | 2.1 | 9.2 | 2 | 0.01 | 0.66 (Footsteps) | 0.64 | 0.00 | **(c)** FlexSED 0.4-0.8 span reachable | — | — | FlexSED Footsteps span 2.08-2.72 s. Not an R1 band candidate: self-veto (BEATs Footsteps clip peak 0.016). |
| 11 | `as_explosion_XJ8lc3I6` | Explosion | 2.8 | 1.4 | 2 | 0.14 (Fusillade) | 0.62 (Explosion) | 0.68 | 0.12 (Boom) | **(c)** FlexSED 0.4-0.8 span reachable | — | — | R1 band candidate (Explosion 2.84-3.52 s, 0.62). |
| 12 | `as_explosion_XJ8lc3I6` | Explosion | 5.6 | 1.1 | 3 | 0.24 (Fusillade) | 0.56 (Explosion) | 0.52 | 0.17 (Fusillade) | **(c)** FlexSED 0.4-0.8 span reachable | d-score | — | R1 band candidate (Explosion 5.68-6.20 s, 0.56). |
| 13 | `as_explosion_XJ8lc3I6` | Gasp | 6.7 | 0.3 | 2 | 0.41 (Gasp) | 0.76 (Gasp) | 0.20 | 0.00 | **(b)** FlexSED 0.4-0.8, blocked by 0.5-s min span | — | — | FlexSED Gasp 0.76 for 0.20 s. BEATs Gasp 0.41 (above the display bar) for one 0.25-s frame. Both are cut by the 0.5-s min span. |
| 14 | `b3_carnival_parade` | Whistle | 6.1 | 1.3 | 2 | 0.01 | 0.58 (Steam whistle) | 0.84 | 0.00 | **(c)** FlexSED 0.4-0.8 span reachable | — | — | FlexSED Steam whistle 0.58, span 6.32-7.16 s. Not an R1 band candidate: self-veto (BEATs 0.003). |
| 15 | `b3_golf_course` | Whack, thwack | 6.5 | 0.6 | 2 | 0.01 | 0.00 | 0.00 | 0.00 | **(a)** unheard | — | — | No detector hears it. |
| 16 | `b3_golf_course` | Whack, thwack | 24.4 | 0.7 | 2 | 0.01 | 0.00 | 0.00 | 0.00 | **(a)** unheard | — | — | No detector hears it. |
| 17 | `b3_pet_shop` | Bird | 0.1 | 27.7 | 2 | 0.84 (Bird vocalization, bird call, bird song) | 0.68 (Bird) | 15.56 | 0.32 (Bird) | **(e)** emitted, removed by gate / stage 5 | c | — | BEATs Bird 0.86 over the whole clip; the gate said visible ('a small bird') on all 6 stretches. |
| 18 | `bell_miami` | Bell | 0.2 | 14.3 | 3 | 0.70 (Church bell) | 0.98 (Bell) | 14.24 | 0.54 (Church bell) | **(e)** emitted, removed by gate / stage 5 | — | — | BEATs Church bell 0.70; the gate said visible ('church bell'). |
| 19 | `birds_forest` | Bird | 1.3 | 16.7 | 2 | 0.21 (Bird) | 0.71 (Bird) | 5.12 | 0.07 | **(d)** BEATs 0.175-0.35 (emitted, below display) | b | — | BEATs Bird span 2.22-2.75 s, conf 0.22 < 0.35. The FlexSED 0.4 span starts at 0.52 s, 0.28 s before the window. |
| 20 | `ly_ambulance_(siren)_-yPSgCn` | Vehicle | 7.3 | 1.5 | 2 | 0.54 (Emergency vehicle) | 0.93 (Ice cream truck, ice cream van) | 18.00 | 0.22 (Vehicle) | **(f)** emitted, onset outside window | h-veto, d | — | The Vehicle span (BEATs 0.61) runs 0.0-14.75 s through the onset, and the gate called it visible. A new Car span starts at 8.25 s with conf 0.32 < 0.35. |
| 21 | `ly_applause_62ZYD0u` | Crowd | 1.9 | 8.1 | 2 | 0.01 | 0.89 (Crowd) | 13.12 | 0.07 | **(f)** emitted, onset outside window | — | — | The Crowd picture 0.0-7.75 s starts 1.9 s early: a FlexSED Crowd span 0.0-1.04 s (0.87) is merged with the Applause span (BEATs, 1.12 s). The same picture is a cross wrong picture at 0.0 s. |
| 22 | `mv_storm_scene_house` | Civil defense siren | 16.9 | 3.5 | 2 | 0.01 | 0.42 (Siren) | 0.08 | 0.00 | **(b)** FlexSED 0.4-0.8, blocked by 0.5-s min span | — | Vehicle 16.50 s (0.32) | FlexSED Siren 0.42 for 0.08 s. |

## 2. The 24 wrong pictures

*Detector* = the origin of the stage-4 span that sets the picture's start: a BEATs span (its start may have been pulled earlier by a FlexSED twin) or a FlexSED-only span. *Score* = that span's conf (BEATs peak or FlexSED peak). BEATs / FlexSED / PANNs = the best score of the picture's family in [start − 0.5, start + 1.0] s. *Same family, off its onset* = a gold sound of the picture's family is already on, or starts up to 2 s after the picture.

| # | clip | shown label | start–end s | type | detector | score | BEATs | FlexSED | PANNs | collided with (gold at that time) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ambient_nature_rainforest_7629` | Cricket | 4.50–6.25 | cross | BEATs | 0.36 | 0.36 (Cricket) | 0.93 (Insect) | 0.00 | Bird (needed, 0.1-16.0 s); Cricket (needed, 0.1-9.5 s) | same family as gold Cricket 0.1 s (needed, imp 2), off its onset; the late Cricket picture of the Cricket 0.1 s miss (d) |
| 2 | `ambient_weather_storm_16200` | Thunder | 0.06–15.75 | visible | BEATs | 0.45 | 0.45 (Thunder) | 0.86 (Thunder) | 0.83 (Thunder) | Rain (visible, 0.0-16.0 s); Thunder (visible, 0.3-7.5 s) | — |
| 3 | `ambient_weather_storm_7200` | Thunder | 0.06–15.75 | visible | BEATs | 0.51 | 0.62 (Thunder) | 0.85 (Thunder) | 0.82 (Thunder) | Thunder (visible, 0.1-15.8 s); Rain (visible, 0.1-15.8 s) | — |
| 4 | `as_church_bell_pJRAWkLM` | Bell | 0.14–13.75 | visible | BEATs | 0.71 | 0.10 | 0.93 (Bell) | 0.28 (Church bell) | Bell (visible, 0.0-15.2 s) | — |
| 5 | `as_explosion_XJ8lc3I6` | Gunshot | 8.25–19.50 | cross | BEATs | 0.53 | 0.53 (Fusillade) | 0.19 (Gunshot) | 0.06 | Walk, footsteps (needed, 2.1-11.3 s) | same family as gold Machine gun 10.1 s (visible, imp 2), off its onset; starts 1.85 s before the visible machine gun (10.1 s) |
| 6 | `as_fire_alarm_kGKZ0YK4` | Alarm | 8.89–19.75 | visible | BEATs | 0.70 | 0.53 (Fire alarm) | 0.92 (Alarm) | 0.15 (Whistle) | Alarm (visible, 9.0-19.8 s) | — |
| 7 | `b3_bakery_morning` | Chink, clink | 0.14–2.00 | phantom | BEATs | 0.49 | 0.42 (Chink, clink) | 0.44 (Chink, clink) | 0.06 | — | — |
| 8 | `b3_barbershop` | Electric shaver, electric razor | 16.00–23.25 | cross | BEATs | 0.87 | 0.62 (Electric shaver, electric razor) | 0.17 (Electric shaver, electric razor) | 0.05 | Electric shaver, electric razor (needed, 0.1-27.8 s) | same family as gold Electric shaver, electric razor 0.1 s (needed, imp 3), off its onset; second picture of the shaver that is already hit at 0.1 s |
| 9 | `b3_crossing_bells` | Steam | 0.22–9.00 | cross | BEATs | 0.77 | 0.47 (Steam) | 0.89 (Steam) | 0.09 | Train (visible, 0.0-17.0 s) | near label of the visible train (Steam) |
| 10 | `b3_favela_rio` | Train | 3.80–5.30 | cross | BEATs | 0.50 | 0.50 (Rail transport) | 0.76 (Train) | 0.23 (Train) | Walk, footsteps (obvious, 0.0-28.0 s); Bird (needed, 1.8-18.1 s) | — |
| 11 | `b3_favela_rio` | Bird | 14.00–15.50 | cross | BEATs | 0.51 | 0.51 (Bird) | 0.36 (Bird flight, flapping wings) | 0.01 | Walk, footsteps (obvious, 0.0-28.0 s); Bird (needed, 1.8-18.1 s) | same family as gold Bird 1.8 s (needed, imp 1), off its onset; Bird picture inside a long importance-1 Bird sound (don't care), off its onset |
| 12 | `b3_laundromat` | Vehicle | 0.30–11.50 | phantom | BEATs | 0.48 | 0.48 (Vehicle) | 0.71 (Train) | 0.32 (Vehicle) | — | BEATs Vehicle; FlexSED Train 0.71 at the same time (machine rumble heard as a vehicle) |
| 13 | `b3_laundromat` | Vehicle | 14.25–17.75 | phantom | BEATs | 0.47 | 0.22 (Vehicle) | 0.73 (Train) | 0.14 (Vehicle) | — | same as above (FlexSED Train 0.73) |
| 14 | `b3_laundromat` | Vehicle | 23.25–27.75 | phantom | BEATs | 0.46 | 0.28 (Vehicle) | 0.77 (Train) | 0.19 (Vehicle) | — | same as above (FlexSED Train 0.77) |
| 15 | `london_protest_01` | Vehicle | 0.25–1.75 | visible | BEATs | 0.38 | 0.38 (Vehicle) | 0.83 (Air horn, truck horn) | 0.50 (Vehicle) | Crowd (visible, 0.0-11.1 s); Air horn, truck horn (visible, 0.4-11.1 s) | Vehicle on the visible air horn |
| 16 | `ly_applause_62ZYD0u` | Crowd | 0.00–7.75 | cross | FlexSED-only | 0.87 | 0.00 | 0.87 (Crowd) | 0.00 | Laughter (visible, 0.0-13.8 s) | same family as gold Crowd 1.9 s (needed, imp 2), off its onset; the early Crowd picture that also causes the Crowd 1.9 s miss (f) |
| 17 | `mv_detective_crime_scene` | Alarm | 3.92–6.12 | cross | FlexSED-only | 0.92 | 0.58 (Telephone bell ringing) | 0.92 (Alarm) | 0.34 (Telephone) | Telephone (needed, 0.0-9.9 s) | same family as gold Telephone 0.0 s (needed, imp 3), off its onset; the ringing phone again (Alarm is Telephone's parent; BEATs Telephone bell ringing 0.58); the Telephone at 0.0 s is already hit |
| 18 | `mv_detective_crime_scene` | Alarm | 9.08–10.58 | cross | FlexSED-only | 0.91 | 0.41 (Telephone bell ringing) | 0.91 (Alarm) | 0.59 (Telephone bell ringing) | Telephone (needed, 0.0-9.9 s) | same family as gold Telephone 0.0 s (needed, imp 3), off its onset; the ringing phone again (BEATs Telephone bell ringing 0.41) |
| 19 | `mv_protest_scene_movie` | Glass | 4.75–7.75 | cross | BEATs | 0.43 | 0.43 (Shatter) | 0.26 (Glass) | 0.11 (Shatter) | Crowd (needed, 0.0-20.0 s); Baby cry, infant cry (visible, 3.2-5.1 s) | — |
| 20 | `mv_protest_scene_movie` | Train | 21.48–22.98 | phantom | FlexSED-only | 0.86 | 0.04 | 0.86 (Train) | 0.04 | — | FlexSED-only Train 0.87; BEATs 0.04 and PANNs 0.04 at that time |
| 21 | `mv_protest_scene_movie` | Siren | 22.75–24.02 | phantom | BEATs | 0.37 | 0.37 (Siren) | 0.78 (Siren) | 0.41 (Siren) | — | — |
| 22 | `mv_tornado_scene` | Vehicle | 7.25–9.75 | cross | BEATs | 0.42 | 0.42 (Vehicle) | 0.54 (Car passing by) | 0.08 | Cellphone buzz, vibrating alert (visible, 1.4-7.6 s) | — |
| 23 | `un_driving_motorcycle_DgdHSmwA` | Fireworks | 13.97–15.47 | visible | BEATs | 0.41 | 0.41 (Firecracker) | 0.46 (Fireworks) | 0.35 (Fireworks) | Crowd (visible, 0.0-17.0 s); Fireworks (obvious, 13.9-14.2 s) | picture of an obvious sound (fireworks seen) |
| 24 | `un_hair_dryer_drying_WWu24rJs` | Computer keyboard | 11.25–12.75 | phantom | BEATs | 0.36 | 0.36 (Computer keyboard) | 0.75 (Computer keyboard) | 0.28 (Typing) | — | — |

## TLDR
- 22 misses: FlexSED band 10 (reachable 6, too short 4), gate said visible 3, stage-4 bookkeeping 5 (below the display bar 2, onset outside the window 2, veto 1), unheard 4, wrong family 0.
- Only 2 of the 6 reachable band spans survive R1's BEATs self-veto (the explosions); BEATs is deaf to the other 4.
- 24 wrong pictures: 20 BEATs, 4 FlexSED-only; 7 of the 11 cross pictures are the right family at the wrong time; fixing their timing gives only 2 hits.
