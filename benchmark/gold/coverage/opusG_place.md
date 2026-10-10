# opusG: where / what / how merged we draw, v1.7 pictures, all 158 clips

v1.7: hits 59 / 135 needed-listed, wrong 13, cost 1.810

## A  back-extend the start to where the family was first heard

Best 10 settings on all clips (reading only; CV below is the result):

| det | bar | gap | N | rise<= | found | hits | wrong | cost | focus fixed |
|---|---|---|---|---|---|---|---|---|---|
| beats | 0.1 | 0.25 | 5.0 | 0.1 | False | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.25 | 5.0 | 0.1 | True | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.25 | 30.0 | 0.1 | False | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.25 | 30.0 | 0.1 | True | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.5 | 5.0 | 0.1 | False | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.5 | 5.0 | 0.1 | True | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.5 | 30.0 | 0.1 | False | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 0.5 | 30.0 | 0.1 | True | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 1.0 | 5.0 | 0.1 | False | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |
| beats | 0.1 | 1.0 | 5.0 | 0.1 | True | 59 | 12 | 1.797 | b3:Bird 3.8->0.2 |

CV: choices [('beats', 0.1, 0.25, 5.0, 0.1, False), ('beats', 0.1, 0.25, 5.0, 0.1, False), None, ('beats', 0.1, 0.25, 5.0, 0.1, False), ('beats', 0.1, 0.25, 5.0, 0.1, False)]; out of fold hits 59, wrong 13, cost 1.810
- ('beats', 0.1, 0.25, 5.0, 0.1, False): b3_golf_course hit+0 wrong-1

Focus cases (picture start vs gold) under the most frequent CV choice:
- b3_golf_course Bird: [3.8] -> [0.25]; clip hit 0->0, wrong 1->0
- tg_d031 Bird: [17.5] -> [17.5]; clip hit 0->0, wrong 1->1
- w8_hide_wolves_howl_1a Baby cry, infant cry: [4.0] -> [4.0]; clip hit 0->0, wrong 2->2
- w8_hide_wolves_howl_1a Human locomotion: [10.0] -> [10.0]; clip hit 0->0, wrong 2->2

Most focus cases any setting puts in the onset window: 2 of 4 (5 settings); cheapest such ('fd', 0.5, 2.0, 3.0, None, False): hits 56, wrong 15, cost 1.911
- b3_golf_course hit+0 wrong-1; birds_forest hit-1 wrong+1; mv_tornado_scene hit-1 wrong+1; tg_d149 hit-1 wrong+1; tg_d019 hit-1 wrong+1; w8_hide_wolves_howl_1a hit+1 wrong-1

## Second-burst misses

65 misses. 6 are sounds of a family already drawn by a picture that started earlier in the clip (and no picture of the family in the onset window); 2 of them start while that earlier picture is still up (merged / held over the new onset). 4 misses have a same-family picture starting > 1 s late inside the sound.

| clip | gold | onset-end | earlier pictures of the family | still up at onset |
|---|---|---|---|---|
| mv_storm_scene_house | Civil defense siren | 16.9-20.4 | Alarm 2.2-11.5 | no |
| tg_d120 | Meow | 2.9-5.6 | Cat 0.6-2.2 | no |
| m4_airsoft_24a | Gunshot, gunfire | 1.8-2.5 | Gunshot 0.0-2.4 | yes |
| m4_live_fire_26a | Machine gun | 7.1-9.3 | Gunshot 2.5-6.0 | no |
| m4_live_fire_26a | Machine gun | 11.4-13.3 | Gunshot 2.5-6.0 | no |
| w8_kids_ice_cream_truck_2a | Ice cream truck, ice cream van | 10.3-18.0 | Ice cream truck, ice cream van 0.1-17.8 | yes |

## S  split a picture at a fresh rise inside it

| det | bar | quiet | lead | hits | wrong | cost |
|---|---|---|---|---|---|---|
| flex | 0.5 | 2.0 | 1.0 | 60 | 16 | 1.823 |
| flex | 0.5 | 2.0 | 2.0 | 60 | 16 | 1.823 |
| flex | 0.5 | 2.0 | 3.0 | 60 | 16 | 1.823 |
| fd | 0.4 | 2.0 | 3.0 | 60 | 19 | 1.861 |
| fd | 0.5 | 2.0 | 3.0 | 60 | 19 | 1.861 |
| flex | 0.3 | 2.0 | 3.0 | 59 | 18 | 1.873 |

CV: choices [None, None, None, None, None]; out of fold hits 59, wrong 13, cost 1.810

## L  parent label for disputed siblings

Scorer: same_family(picture, gold) is true for ancestor/descendant pairs, so an 'Explosion' picture is a hit for a Gunshot, gunfire / Machine gun / Fireworks / Burst gold sound; the family name 'Gunshot' is not an ontology name (no ancestors), so a 'Gunshot' picture never matches Explosion or Fireworks gold. Fire is not under Explosion (Natural sounds > Fire). Siren is a child of Alarm; Shofar is under Music.

pictures of these labels in v1.7: {'Gunshot': 5, 'Explosion': 9, 'Alarm': 6, 'Siren': 5}
- ontology name: hits 60, wrong 12, cost 1.772; m4_film_1917_33a hit+1 wrong-1
- weapon: hits 60, wrong 12, cost 1.772; m4_film_1917_33a hit+1 wrong-1
- siren: hits 59, wrong 13, cost 1.810; 
- both: hits 60, wrong 12, cost 1.772; m4_film_1917_33a hit+1 wrong-1

## M  merging

| setting | hits | wrong | dup | cost |
|---|---|---|---|---|
| v1.7 (MERGE_GAP 2.5, GROUP 8) | 59 | 13 | 1 | 1.810 |
| MERGE_GAP 0.8 | 59 | 17 | 1 | 1.861 |
| MERGE_GAP 1.0 | 59 | 17 | 1 | 1.861 |
| MERGE_GAP 1.5 | 59 | 17 | 1 | 1.861 |
| MERGE_GAP 2.0 | 59 | 14 | 1 | 1.823 |
| MERGE_GAP 3.0 | 58 | 13 | 1 | 1.835 |
| MERGE_GAP 4.0 | 55 | 13 | 1 | 1.911 |
| GROUP off | 60 | 16 | 1 | 1.823 |
| GROUP_MAX_GAP 4 | 60 | 14 | 1 | 1.797 |
| MERGE_GAP 1.0 + GROUP off | 60 | 20 | 1 | 1.873 |

CV: choices ['v1.7 (MERGE_GAP 2.5, GROUP 8)', 'GROUP_MAX_GAP 4', 'GROUP_MAX_GAP 4', 'GROUP_MAX_GAP 4', 'GROUP_MAX_GAP 4']; out of fold hits 59, wrong 14, cost 1.823
