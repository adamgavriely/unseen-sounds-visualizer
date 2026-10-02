# VLM visibility-gate audit (Qwen3.8-27B): what it deemed "seen"

Sources: benchmark/gold/gate_gold/Qwen38-27B/*.json (votes, named phrase), cluster log ~/MscProj/logs/gate_gold_30967699.out (describe sentences, first 60 chars), gold benchmark/gold/annotations/gold_AG.json, DEV pipeline votes ~/MscProj_r13/data/work/r13/SHIP8+MD3+WW5+SL_proposed/*/gate_votes.json and ~/MscProj_tg/data/work/r13dev2/.../gate_votes.json, human2/human/cf/box caches for alternative raw replies. Gold "seen" = visible OR obvious (gate_gold.py); "needed" = neither. Importance>=2 only, as gate_gold.score. Frames for the DEV error cases were extracted with ffmpeg at the gate's own times and looked at (scratchpad/frames/sheet_*.jpg).

Error classes: (i) object present but not the sound's maker/action now; (ii) look-alike / other instance of the same kind; (iii) answer-format / readout; (iv) frames at the wrong moment; (v) the question or the gold definition itself.

## 1. Prompts verbatim (src/stage5_cross_modal_analysis/reason.py)

**Q1 open naming (VISIBLE_PROMPT; 6 frames, max 24 tokens, cleaned to 5 words):**
```
These frames are from the moment a sound of {label} was heard.
Name the thing in these frames that is making that sound. Answer with a short noun phrase of at most 5 words. If nothing that could make that sound is visible in these frames, answer exactly: nothing.
```

Readout: a reply starting with nothing/none/no /not -> name vote NO. Else, if the phrase shares a word with the label (_about_the_sound) -> name vote YES with no further check ("label echo"). Else a text-only world-knowledge question (no frames):
```
Sound heard: {label}
Thing visible in the video: {named}
Could that thing be what is making that sound? Answer yes or no.
```
vote = reply starts with "y".

**Q2 a/b (6 frames, asked in both orderings, max 6 tokens; yes only if both orderings agree, else split -> None):**
```
These frames are from the moment a sound of {label} was heard. Judge from the frames alone.
(a) you can SEE {label} happening on screen -- the source is in the frames and visibly making that sound
(b) {label} is not visibly happening in these frames
Answer with the letter only.
```

**Q3 describe (6 frames, max 48 tokens, cut to 25 words):**
```
Describe what is happening in these frames in one sentence. Name what you see and what it is doing. No more than 25 words.
```
Readout: desc vote YES if the sentence shares a word with the label, or with the named phrase when the name vote was yes; otherwise the sentence is put to MAKES_SOUND_PROMPT (text only, "Could that thing be what is making that sound?") and vote = reply starts with "y".

**Verdict:** majority of (name, a/b, desc); None counts for neither side. A sound is silenced only if every 5-s stretch is "seen". FIX_GATE (named nothing -> not seen), GATE_BOX_CHECK, ACTIVITY_GATE, CONCEALED_ACTION are all OFF in the shipped config (config.py).

**Gate-gold extra, not on the inference path ("obvious" a/b):** `These frames are from the moment a sound of {label} was heard. Judge from the frames alone, as a viewer who hears nothing. (a) a viewer with no sound would already assume that {label} is happening -- the thing making it, or its action, is plainly shown (b) a viewer with no sound would not know that {label} is happening`

**Round 43b human2 naming (strict):** `Answer with ONLY a short noun phrase naming the most likely visible source of the {label} sound in these frames (e.g. 'the golfer', 'the waterfall'), or exactly 'none' if no plausible source is visible.` then the Round-43 action a/b on captioned frames: `{cand} was named as the likely source of the {label} that {verb} at the 0 s frame. Look at the 0 s frame and its neighbours. (a) at that moment {cand} is visibly DOING or UNDERGOING the thing that makes {label} (striking, swinging, flowing, flashing, running, calling...) -- the action itself shows in the frames (b) {cand} is only present, or the action is not visible at that moment`. **Round 50/50L crop question:** `This is a close-up cut from a video frame. Is this {phrase} making the {label} sound right now? Answer yes or no.` (50L reads it as a logit margin).

## 2. Gate-gold DEV (dev54 = judge100 clips): 145 stretches; gate said seen 54, of which gold agrees 38, WRONG (gold needed) 16; gold-seen but kept (missed) 38

Name-vote path on the wrong silences: {'could-question yes': 6, 'label-echo (word overlap, no check)': 9, 'nothing': 1}; desc-vote path: {'names the named thing': 8, 'names sound': 1, 'could-question yes': 7}; a/b vote on the wrong silences: {'yes': 9, 'split': 3, 'no': 4}.

Label-echo rate over ALL gate-seen stretches (DEV): 30/54 named a phrase that repeats the label word (name vote YES with no check).

Error classes on the 16 wrong stretches: {'ii': 10, 'i': 5, 'v/iv': 1}

### 2a. Wrong silences (gate seen, gold needed)

| clip | sound | stretch | name [path] | a/b | desc [path] | named (Q1) | describe (Q3, 60 chars) | gold vis/obv | class |
|---|---|---|---|---|---|---|---|---|---|
| ambient_nature_rainforest_7629 | Bird | 0.1-5.4 | yes [could-question yes] | yes | yes [names the named thing] | macaws | Two colorful macaws perch on a bare tree branch in a lush fo | 0/0 | ii |
| ambient_nature_rainforest_7629 | Bird | 5.4-10.7 | yes [could-question yes] | yes | yes [names the named thing] | macaws | Two scarlet macaws perch on a bare tree branch in a lush gre | 0/0 | ii |
| b3_golf_course | Whack, thwack | 6.5-7.1 | yes [could-question yes] | split | yes [names the named thing] | golf club | A man in a black shirt and cap is practicing his golf swing  | 0/0 | i |
| b3_pet_shop | Bird | 0.1-4.7 | yes [label-echo (word overlap, no check)] | yes | yes [names the named thing] | a small bird | A small robin with an orange breast stands on a mossy tree s | 0/0 | ii |
| b3_pet_shop | Bird | 4.7-9.3 | yes [label-echo (word overlap, no check)] | yes | yes [names sound] | a small bird | A small red-breasted bird | 0/0 | ii |
| b3_pet_shop | Bird | 9.3-13.9 | yes [label-echo (word overlap, no check)] | yes | yes [could-question yes] | a small bird | A European robin lands on a mossy stump | 0/0 | ii |
| b3_pet_shop | Bird | 13.9-18.6 | yes [could-question yes] | yes | yes [names the named thing] | European robin | A European robin hops onto a mossy tree stump in a garden | 0/0 | ii |
| b3_pet_shop | Bird | 18.6-23.2 | yes [label-echo (word overlap, no check)] | yes | yes [could-question yes] | a bird | A small robin with an orange breast stands on a mossy tree s | 0/0 | ii |
| b3_pet_shop | Bird | 23.2-27.8 | yes [label-echo (word overlap, no check)] | yes | yes [could-question yes] | birds | A robin and a blackbird take turns perching on a mossy tree  | 0/0 | ii |
| bell_miami | Bell | 0.2-5.0 | yes [label-echo (word overlap, no check)] | no | yes [names the named thing] | church bell | A camera pans right across a pink church | 0/0 | i |
| bell_miami | Bell | 5.0-9.7 | yes [label-echo (word overlap, no check)] | no | yes [could-question yes] | church bell | A camera pans left across a pink cathedral and surrounding c | 0/0 | i |
| bell_miami | Bell | 9.7-14.5 | yes [label-echo (word overlap, no check)] | no | yes [names the named thing] | church bell | A camera pans left across a pink church and a white skyscrap | 0/0 | i |
| ly_ambulance_(siren)_-yPSgCn | Vehicle | 7.3-8.8 | no [nothing] | yes | yes [could-question yes] | nothing | A car drives down a wet | 0/0 | i (+iii: name said nothing; a/b letter + desc could-question carried it) |
| mv_protest_scene_movie | Crowd | 10.0-15.0 | yes [could-question yes] | split | yes [could-question yes] | people | A man in a blue vest and white shirt stands in a dimly lit r | 0/0 | ii |
| mv_protest_scene_movie | Crowd | 15.0-20.0 | yes [label-echo (word overlap, no check)] | no | yes [names the named thing] | crowd of people | A man in a blue vest stands in a dimly lit room while people | 0/0 | ii |
| mv_tornado_scene | Siren | 8.9-15.6 | yes [could-question yes] | split | yes [could-question yes] | loudspeaker | A man in a red shirt and white cowboy hat sings into a micro | 0/0 | v/iv (siren horn shown 7.9-9.6 s; gold says not visible 8.9-15.6) |

### 2b. Correct silences (gate seen, gold seen)

| clip | sound | stretch | name | a/b | desc | named | describe | gold vis/obv |
|---|---|---|---|---|---|---|---|---|
| ambient_market_marrakech_3102 | Motorcycle | 12.7-16.0 | yes | yes | no | motorcycle | A man and a woman walk through a busy outdoor market area wi | 1/1 |
| ambient_weather_storm_7200 | Rain | 0.1-5.3 | yes | yes | yes | rain | Heavy rain falls intensely on a dark | 1/1 |
| ambient_weather_storm_7200 | Rain | 5.3-10.6 | yes | yes | yes | rain | Heavy rain falls on a dark | 1/1 |
| ambient_weather_storm_7200 | Rain | 10.6-15.8 | yes | yes | yes | rain | Heavy rain falls on a dark | 1/1 |
| as_explosion_XJ8lc3I6 | Machine gun | 10.1-13.9 | yes | split | yes | ak-47 rifle | A player in a first-person shooter game uses an AK-47 to sho | 1/1 |
| as_explosion_XJ8lc3I6 | Machine gun | 13.9-17.8 | yes | no | yes | AK-47 rifle | A player is playing a first-person shooter game | 1/1 |
| as_glass_oHil9Ip_ | Chink, clink | 18.5-19.0 | yes | yes | yes | glass | A man in a kitchen pours a drink into a glass and then reach | 1/1 |
| as_glass_oHil9Ip_ | Glass | 18.8-19.3 | yes | no | yes | glass | A man in a kitchen is placing ice from a tray into a glass o | 1/1 |
| b3_aviary_birds | Bird | 0.0-4.6 | yes | yes | yes | bird | A red-crested woodpecker with a grey body and white belly is | 1/1 |
| b3_aviary_birds | Bird | 4.6-9.3 | yes | yes | yes | bird | A red-crested woodpecker sits on a branch next to a nest | 1/1 |
| b3_aviary_birds | Bird | 9.3-13.9 | yes | yes | yes | birds | A bird flies into a large woven nest | 1/1 |
| b3_aviary_birds | Bird | 13.9-18.5 | yes | yes | yes | birds | A camera pans across a bird aviary | 1/1 |
| b3_aviary_birds | Bird | 18.5-23.2 | yes | split | yes | bird | A small bird is perched on a branch inside a large | 1/1 |
| b3_aviary_birds | Bird | 23.2-27.8 | yes | yes | yes | birds | Several small birds | 1/1 |
| b3_carnival_parade | Drum | 0.0-4.8 | yes | split | yes | drum | A group of people wearing matching orange shirts march down  | 1/1 |
| b3_carnival_parade | Drum | 4.8-9.6 | yes | no | yes | drum | A group of people | 1/1 |
| b3_carnival_parade | Drum | 19.2-24.0 | yes | yes | yes | drums | A group of people wearing orange shirts play drums in a stre | 1/1 |
| b3_crossing_bells | Train | 0.0-5.7 | yes | yes | yes | train | A train passes through a level crossing while cars wait and  | 1/1 |
| b3_crossing_bells | Train | 5.7-11.3 | yes | yes | yes | train | A train passes through a level crossing | 1/1 |
| b3_favela_rio | Walk, footsteps | 4.7-9.3 | yes | split | yes | man in white shirt | A person wearing a white shirt and a backpack stands in the  | 0/1 |
| b3_favela_rio | Walk, footsteps | 9.3-14.0 | yes | yes | yes | man | A man in a blue and white shirt walks down a steep | 0/1 |
| b3_favela_rio | Walk, footsteps | 14.0-18.7 | yes | yes | yes | person walking down stairs | A person in a blue shirt walks down a steep | 0/1 |
| b3_flea_market | Rustle | 0.0-1.1 | yes | yes | no | paper money | A person counts a stack of cash at an outdoor flea market ta | 1/1 |
| b3_flea_market | Rustle | 1.2-2.9 | yes | split | yes | plastic bags | A person is counting a stack of US dollar bills at an outdoo | 1/1 |
| b3_golf_course | Whip | 11.2-12.0 | yes | split | yes | golf club | A man in a black shirt and cap swings a golf club at a drivi | 1/1 |
| london_protest_01 | Air horn, truck horn | 0.4-5.8 | yes | yes | no | red air horn | A crowd of people marches down a city street | 1/1 |
| ly_applause_62ZYD0u | Laughter | 0.0-4.6 | yes | yes | yes | woman | A woman in a white top laughs hysterically | 1/1 |
| ly_applause_62ZYD0u | Laughter | 4.6-9.2 | yes | yes | yes | woman | A woman in a white top laughs and gestures animatedly while  | 1/1 |
| ly_applause_62ZYD0u | Laughter | 9.2-13.8 | yes | yes | yes | the woman | A woman in a white top talks and gestures while another woma | 1/1 |
| mv_tornado_scene | Cellphone buzz, vibrating alert | 1.4-7.6 | yes | split | yes | cellphone | A crowd of people | 1/0 |
| mv_tornado_scene | Horse | 17.1-19.8 | yes | yes | yes | horse | A man in a red shirt and cowboy hat speaks into a microphone | 1/1 |
| un_driving_motorcycle_4O3bZRYO | Vehicle | 4.8-10.2 | yes | yes | yes | white car | A white Toyota Avanza is shown maneuvering and driving on a  | 1/0 |
| un_driving_motorcycle_4O3bZRYO | Motorcycle | 16.2-20.0 | yes | yes | yes | motorcycle | A young boy in a school uniform stands on the road next to a | 1/1 |
| waterfall_kawaida_01 | Water | 0.0-5.6 | yes | yes | yes | waterfall | The camera zooms out to reveal a wide shot of a waterfall ca | 1/1 |
| waterfall_kawaida_01 | Water | 5.6-11.2 | yes | yes | yes | waterfall | The camera pans right and zooms out to reveal a group of peo | 1/1 |
| waves_herzliya | Water | 0.0-4.2 | yes | yes | yes | ocean waves | Gentle ocean waves roll onto a sandy beach under a cloudy sk | 1/1 |
| waves_herzliya | Water | 4.2-8.4 | yes | yes | yes | ocean waves | Gentle ocean waves roll onto a sandy beach under an overcast | 1/1 |
| waves_herzliya | Water | 8.4-12.6 | yes | yes | yes | ocean waves | Gentle ocean waves roll onto a sandy beach under a cloudy sk | 1/1 |

### 2c. Missed (gold seen, gate kept)

| clip | sound | stretch | name | a/b | desc | named | describe | gold vis/obv |
|---|---|---|---|---|---|---|---|---|
| ambient_weather_storm_16200 | Thunder | 0.3-7.5 | no | no | no | nothing | Heavy rain falls on a dark | 1/1 |
| ambient_weather_storm_16200 | Thunder | 10.4-16.0 | no | no | no | nothing | Heavy rain falls steadily on a dark | 1/1 |
| ambient_weather_storm_7200 | Thunder | 0.1-5.3 | no | no | no | nothing | Heavy rain falls intensely on a dark | 1/1 |
| ambient_weather_storm_7200 | Thunder | 5.3-10.6 | no | no | no | nothing | Heavy rain falls on a dark | 1/1 |
| ambient_weather_storm_7200 | Thunder | 10.6-15.8 | no | no | no | nothing | Heavy rain falls on a dark | 1/1 |
| as_church_bell_pJRAWkLM | Bell | 0.0-5.1 | no | no | no | nothing | The camera pans across the red brick facade of a large | 1/0 |
| as_church_bell_pJRAWkLM | Bell | 5.1-10.1 | no | no | no | nothing | The camera pans across the red brick facade of a large | 1/0 |
| as_church_bell_pJRAWkLM | Bell | 10.1-15.2 | no | no | no | nothing | A camera pans across a large red brick building with turrets | 1/0 |
| as_fire_alarm_kGKZ0YK4 | Alarm | 9.0-14.4 | no | no | no | nothing | A man in a striped shirt uses a tool to work on a wall in a  | 1/0 |
| as_fire_alarm_kGKZ0YK4 | Alarm | 14.4-19.8 | yes | no | no | fire alarm | A person walks down a hallway decorated with stars | 1/0 |
| as_glass_oHil9Ip_ | Glass | 9.0-9.6 | yes | no | no | glass | A man in a grey jacket is leaning over a kitchen counter | 1/1 |
| b3_bakery_morning | Crumpling, crinkling | 10.0-12.1 | no | no | no | nothing | A person is preparing a sheet of dough on a floured wooden c | 1/1 |
| b3_bakery_morning | Crumpling, crinkling | 14.1-15.6 | no | no | no | nothing | A person is pressing and smoothing a rectangular sheet of ye | 1/1 |
| b3_bakery_morning | Frying (food) | 17.6-18.3 | no | no | no | nothing | A person uses a long metal ruler to measure and trim the edg | 1/1 |
| b3_bakery_morning | Chink, clink | 20.3-20.8 | no | no | no | nothing | A person is preparing to cut a rectangular sheet of dough on | 1/1 |
| b3_bakery_morning | Tap | 20.7-21.9 | no | no | no | nothing | A person is preparing to cut a rectangular sheet of dough on | 1/1 |
| b3_botanic_garden | Walk, footsteps | 0.0-4.4 | no | no | no | nothing | A camera moves forward along a dirt path in a forest | 0/1 |
| b3_botanic_garden | Walk, footsteps | 4.4-8.9 | no | no | no | nothing | A camera moves forward along a dirt path in a sunny forest | 0/1 |
| b3_botanic_garden | Walk, footsteps | 8.9-13.3 | no | no | no | nothing | A camera pans right along a forest path | 0/1 |
| b3_botanic_garden | Rustle | 24.8-25.7 | no | no | yes | nothing | A camera slowly pans across a dense | 1/1 |
| b3_carnival_parade | Drum | 9.6-14.4 | no | no | yes | nothing | A group of people wearing orange shirts are marching down a  | 1/1 |
| b3_carnival_parade | Drum | 14.4-19.2 | yes | no | no | drum | A group of people wearing matching orange shirts are gathere | 1/1 |
| b3_construction_site | Honk | 0.7-1.3 | no | no | no | nothing | A yellow Caterpillar 365C excavator digs into a pile of red  | 1/0 |
| b3_crossing_bells | Train | 11.3-17.0 | yes | no | no | train | Cars are parked along a street in a town | 1/1 |
| b3_favela_rio | Walk, footsteps | 0.0-4.7 | no | no | no | nothing | A camera pans down a steep | 0/1 |
| b3_favela_rio | Walk, footsteps | 18.7-23.3 | no | no | yes | nothing | A person walks down a steep | 0/1 |
| b3_favela_rio | Walk, footsteps | 23.3-28.0 | no | no | no | nothing | A camera moves down a steep | 0/1 |
| b3_golf_course | Arrow | 19.7-20.2 | no | no | no | nothing | A man in a black jacket and cap stands on a golf mat | 1/1 |
| b3_ia_youtube_skxtz9foauw_0 | Water | 24.6-27.8 | no | yes | no | red wine | A woman sits on a couch holding a camera on a gimbal | 1/1 |
| b3_ia_youtube_skxtz9foauw_0 | Tap | 27.3-28.0 | yes | no | no | wine glass | A woman sits on a couch | 1/1 |
| london_protest_01 | Air horn, truck horn | 5.8-11.1 | no | no | no | nothing | A crowd of people marches down a city street | 1/1 |
| movie_blueplanet_115 | Quack | 0.2-1.6 | no | no | no | nothing | A large colony of seals rests on a beach while a seagull sta | 1/0 |
| movie_blueplanet_115 | Duck | 2.3-3.5 | no | no | no | nothing | A seagull stands on rocks | 1/0 |
| mv_arrest_street_scene | Glass | 11.3-14.8 | no | no | no | nothing | A young man in a red sweater runs outside | 1/1 |
| mv_protest_scene_movie | Baby cry, infant cry | 3.2-5.1 | no | no | yes | nothing | A man in a vest and white shirt stands up from a seated posi | 1/0 |
| mv_storm_scene_house | Ding | 13.3-18.6 | no | no | no | nothing | A tornado approaches a town as people film it with cameras w | 1/1 |
| mv_storm_scene_house | Ding | 18.6-24.0 | no | no | no | nothing | A massive tornado rips through a city street | 1/1 |
| un_driving_motorcycle_DgdHSmwA | Fireworks | 13.9-14.4 | no | no | no | nothing | A crowd of runners gathers at the starting line under a blue | 0/1 |

## 2. Gate-gold TEST (dev54 = judge100 clips; TEST = the other gate-gold clips, listed for the audit only, no selection on them): 247 stretches; gate said seen 83, of which gold agrees 63, WRONG (gold needed) 20; gold-seen but kept (missed) 71

Name-vote path on the wrong silences: {'could-question yes': 6, 'label-echo (word overlap, no check)': 14}; desc-vote path: {'names the named thing': 5, 'could-question yes': 9, 'names sound': 4, 'no': 2}; a/b vote on the wrong silences: {'yes': 5, 'no': 11, 'split': 4}.

Label-echo rate over ALL gate-seen stretches (TEST): 49/83 named a phrase that repeats the label word (name vote YES with no check).

Error classes on the 20 wrong stretches: {'v': 2, 'i': 9, 'ii': 7, 'iii': 2}

### 2a. Wrong silences (gate seen, gold needed)

| clip | sound | stretch | name [path] | a/b | desc [path] | named (Q1) | describe (Q3, 60 chars) | gold vis/obv | class |
|---|---|---|---|---|---|---|---|---|---|
| 1J_j6c2uQaM_30000 | Thunderstorm | 3.7-5.9 | yes [could-question yes] | yes | yes [names the named thing] | lightning | A lightning storm rages at night | 0/0 | v (lightning on screen; gold says not visible) |
| bell_kazansky | Bell | 0.0-4.5 | yes [label-echo (word overlap, no check)] | no | yes [could-question yes] | bells | A person is sitting on a bench in a park in front of a large | 0/0 | i |
| bell_kazansky | Bell | 4.5-9.0 | yes [label-echo (word overlap, no check)] | split | yes [could-question yes] | bell | A person walks through a park with a large white cathedral a | 0/0 | i |
| bell_kazansky | Bell | 13.5-18.0 | yes [label-echo (word overlap, no check)] | split | yes [could-question yes] | bell | People are walking around a town square featuring a large wh | 0/0 | i |
| fZhywChBkq0_30000 | Laughter | 6.9-7.6 | yes [could-question yes] | no | yes [could-question yes] | the girl with pigtails | The cartoon characters | 0/0 | ii |
| fZhywChBkq0_30000 | Human voice | 8.6-9.1 | yes [could-question yes] | no | yes [names the named thing] | the boy | A girl in a black ninja outfit watches a boy in a black suit | 0/0 | ii |
| m4_airsoft_24a | Gunshot, gunfire | 1.8-2.5 | yes [could-question yes] | no | yes [could-question yes] | rifle | A person in camouflage crouches behind an orange barrel | 0/0 | i |
| m4_live_fire_26a | Machine gun | 7.1-9.3 | yes [label-echo (word overlap, no check)] | no | yes [could-question yes] | Machine gun | Two M1 Abrams tanks are positioned on a paved area | 0/0 | i |
| m4_live_fire_26a | Machine gun | 11.4-13.3 | yes [label-echo (word overlap, no check)] | no | yes [could-question yes] | machine gun | A tank drives across a snowy field | 0/0 | i |
| m5_doc_restrepo_138b | Machine gun | 0.0-1.5 | yes [label-echo (word overlap, no check)] | yes | yes [could-question yes] | machine gun | A soldier in green shorts steps over a pile of spent bullet  | 0/0 | ii (mounted gun on screen, heard fire is distant) |
| m5_doc_restrepo_138b | Machine gun | 2.4-6.3 | yes [label-echo (word overlap, no check)] | yes | yes [names sound] | machine gun | A soldier operates a mounted machine gun while another shirt | 0/0 | ii (mounted gun on screen, heard fire is distant) |
| mc_bridge_scene | Door | 6.0-7.1 | yes [label-echo (word overlap, no check)] | no | yes [names the named thing] | car door | A man with a gun enters a car and points it at a woman sitti | 0/0 | i |
| mYI2QzLce_s_30000 | Bicycle bell | 9.0-10.0 | yes [label-echo (word overlap, no check)] | no | yes [names sound] | bicycle | A group of people are walking and riding bicycles along a pa | 0/0 | ii |
| q4Z8j3IalYs_100000 | Fire engine, fire truck (siren) | 5.0-10.0 | yes [label-echo (word overlap, no check)] | no | yes [names sound] | fire truck | A fire truck drives down a road | 0/0 | v (fire truck on screen; gold not visible) |
| tyrj_wZor4U_0 | Bell | 1.0-2.1 | yes [could-question yes] | no | yes [names the named thing] | bicycle | A man in a suit and hat rides a bicycle down a street past a | 0/0 | ii |
| w8_dog_fireworks_window_2a | Fireworks | 5.4-10.7 | yes [label-echo (word overlap, no check)] | yes | no [no] | fireworks | A person greets their dog | 0/0 | iii (label echo: named "fireworks", describe "A person greets their dog") |
| w8_dog_fireworks_window_2a | Fireworks | 10.7-15.9 | yes [label-echo (word overlap, no check)] | yes | no [no] | fireworks | A person praises their German Shepherd | 0/0 | iii (label echo: named "fireworks", describe "A person greets their dog") |
| w8_kids_fire_alarm_school_1b | Alarm | 10.5-15.7 | yes [label-echo (word overlap, no check)] | split | yes [names sound] | fire alarm control panel | A person interacts with a wall-mounted fire alarm control pa | 0/0 | i |
| w8_pet_parrot_phone_ring_2b | Dog | 5.9-11.5 | yes [label-echo (word overlap, no check)] | split | yes [names the named thing] | brown dog | A man works on a brown Mini Cooper in a driveway while two p | 0/0 | ii |
| zbiJEml563w_30000 | Siren | 0.0-5.0 | yes [could-question yes] | no | yes [could-question yes] | white police suv | A man in a light shirt runs away from a yellow car with its  | 0/0 | i |

### 2b. Correct silences (gate seen, gold seen)

| clip | sound | stretch | name | a/b | desc | named | describe | gold vis/obv |
|---|---|---|---|---|---|---|---|---|
| 1ghXWnSJibU_0 | Bicycle bell | 1.7-5.0 | yes | yes | yes | bicycle bell | A hand presses the button on a chrome bicycle bell mounted o | 1/1 |
| 2gbNCkCPl7w_380000 | Rustle | 1.1-1.6 | yes | no | yes | tall grass | A man and a woman walk together through a grassy area | 1/1 |
| 58JwiVM8bYM_30000 | Quack | 3.1-3.6 | yes | split | yes | ducks | A first-person shooter game shows a player aiming a shotgun  | 1/0 |
| 58JwiVM8bYM_30000 | Gunshot, gunfire | 3.5-4.4 | yes | no | yes | shotgun | A player in a first-person hunting game aims a shotgun at a  | 1/1 |
| 6sRQnpEEQ0E_390000 | Firecracker | 2.3-6.2 | yes | yes | yes | fireworks | A man in a gold hat watches fireworks explode in a fountain  | 1/1 |
| 6sRQnpEEQ0E_390000 | Firecracker | 6.2-10.0 | yes | yes | yes | fireworks | A man throws a firework into a fire pit | 1/1 |
| 8PsvY3TZkBQ_290000 | Steam whistle | 8.3-10.0 | yes | split | yes | steam locomotive | A black steam locomotive pulls a line of freight cars along  | 1/1 |
| 97aoiaWwRVk_20000 | Train wheels squealing | 0.0-1.8 | yes | split | yes | train | A train moves along an elevated track while a couple embrace | 1/1 |
| 9NYsJxa4wjA_40000 | Croak | 8.8-10.0 | yes | split | yes | two frogs | A cartoon frog watches a swarm of flies buzzing around its h | 1/0 |
| 9P6-DKN1XLA_320000 | Snicker | 5.0-5.8 | yes | split | yes | woman | A man in a green shirt and a woman in a black dress are havi | 1/1 |
| _U8kAFAm8tQ_30000 | Firecracker | 0.0-5.9 | yes | yes | yes | firecracker | A firework is burning on the grass | 1/1 |
| ambient_everyday_farm_2166 | Sheep | 9.9-15.8 | yes | yes | yes | white sheep | A white lamb grazes on a green pasture | 1/1 |
| as_alarm_71Xl_uAS | Alarm | 0.6-5.4 | yes | split | yes | sony alarm clock | A person picks up a black Sony digital alarm clock | 1/0 |
| as_alarm_71Xl_uAS | Alarm | 5.4-10.2 | yes | yes | yes | sony alarm clock | A person is holding and manipulating a black Sony alarm cloc | 1/0 |
| as_alarm_71Xl_uAS | Alarm | 10.2-15.0 | yes | split | yes | sony alarm clock | A person is holding and adjusting a black Sony digital alarm | 1/0 |
| as_alarm_71Xl_uAS | Alarm | 15.0-19.8 | yes | split | yes | sony alarm clock | A Sony digital clock radio displays the time 8:51 in green n | 1/0 |
| Br1s0Ye8kKs_30000 | Door | 0.0-0.8 | yes | no | yes | wooden cabinet | A red prohibition symbol is overlaid on a wooden cabinet | 1/1 |
| Br1s0Ye8kKs_30000 | Slam | 0.0-0.8 | yes | no | yes | cabinet door | A red prohibition symbol is overlaid on a wooden cabinet | 1/1 |
| ev_smoke_alarm_kitchen | Frying (food) | 0.1-5.5 | yes | no | yes | pot on the stove | A camera pans across a kitchen counter showing a plate of sa | 1/1 |
| g9Qah25_yH0_14000 | Baby cry, infant cry | 0.2-4.6 | yes | yes | yes | the baby | A man holds a baby up in the air | 1/1 |
| g9Qah25_yH0_14000 | Baby cry, infant cry | 4.6-9.0 | yes | yes | yes | the baby | A man holds a crying baby on his lap while he lies on a couc | 1/1 |
| lx_chainsaw_and_power_tool_BBukw6J | Chainsaw | 0.6-6.4 | yes | yes | yes | chainsaw | A worker uses a chainsaw to cut a large log held in a yellow | 1/1 |
| lx_chainsaw_and_power_tool_BBukw6J | Chainsaw | 6.4-12.2 | yes | yes | yes | chainsaw | A worker in a yellow hard hat uses a chainsaw to cut firewoo | 1/1 |
| lx_chainsaw_and_power_tool_BBukw6J | Chainsaw | 12.2-18.0 | yes | yes | yes | chainsaw | A worker in a yellow hard hat uses a chainsaw to cut a large | 1/1 |
| lx_chainsaw_and_power_tool_W1VYWwY | Chainsaw | 0.0-4.5 | yes | yes | yes | chainsaw | A man uses a small chainsaw to carve a face into a giant pum | 1/1 |
| lx_chainsaw_and_power_tool_W1VYWwY | Chainsaw | 4.5-8.9 | yes | yes | yes | chainsaw | A man in a green hoodie uses a chainsaw to carve a face into | 1/1 |
| lx_chainsaw_and_power_tool_W1VYWwY | Chainsaw | 8.9-13.4 | yes | yes | yes | chainsaw | A man in a green hoodie uses a chainsaw to carve a large pum | 1/1 |
| ly_alarm_qCVdVav | Door | 3.8-4.7 | yes | split | yes | car door | A person is filming the interior of a yellow Volkswagen truc | 1/1 |
| m4_clay_shoot_11a | Laughter | 5.4-6.0 | yes | yes | yes | man in black jacket | A group of men are at a shooting range | 1/1 |
| m4_film_blackhawk_32a | Helicopter | 0.0-4.8 | yes | yes | yes | military helicopter | A soldier is being lowered from a helicopter by a rope over  | 1/1 |
| m4_film_blackhawk_32a | Helicopter | 4.8-9.5 | yes | yes | yes | military helicopter | A soldier is inside a military helicopter | 1/1 |
| m4_film_blackhawk_32a | Helicopter | 9.5-14.2 | yes | yes | yes | military helicopter | A military helicopter flies low over a city | 1/1 |
| m4_film_blackhawk_32a | Helicopter | 14.2-19.0 | yes | yes | yes | military helicopter | A military helicopter flies low over a war-torn city while s | 1/1 |
| m4_fire_bodycam_23a | Fire | 18.7-20.0 | yes | yes | no | fire | A firefighter in full gear | 1/1 |
| m5_war_fury_25b | Artillery fire | 5.2-5.8 | yes | no | yes | tank | A tank commander shouts "Fire!" as his crew aims and fires t | 1/1 |
| m5_war_fury_25b | Artillery fire | 10.1-10.8 | yes | split | yes | tank | A tank fires its main cannon | 1/1 |
| m5_war_fury_25b | Artillery fire | 14.6-15.1 | yes | split | yes | tank main gun | A tank crewman shouts from his turret as his tank | 1/1 |
| mc_break_in_scene | Glass | 13.2-17.8 | yes | no | yes | large glass window | A man in a black sweater stands with arms outstretched over  | 1/1 |
| mc_break_in_scene | Crack | 8.1-9.9 | yes | split | yes | cracked glass window | A large glass window shatters | 1/1 |
| mc_bridge_scene | Door | 2.1-4.9 | yes | no | yes | car door | A man with a gun holds a woman hostage in the back of a car  | 1/1 |
| mc_riot_scene | Helicopter | 0.0-5.3 | yes | yes | yes | helicopter | A man looks up at a helicopter flying at night over a buildi | 1/1 |
| mc_riot_scene | Car | 11.4-16.0 | yes | yes | yes | car | A car drifts around a corner at night | 1/1 |
| mv_air_raid_scene | Crow | 0.0-5.2 | yes | yes | yes | crow | A car drives down a snowy street while several birds fly ove | 1/1 |
| mv_bank_robbery_alarm | Truck | 1.4-8.2 | yes | yes | yes | white van | A person is seen inside and then exiting a white van at nigh | 1/1 |
| mv_bank_robbery_alarm | Screech | 8.3-9.9 | yes | split | yes | white van | A white van is shown at night | 1/1 |
| mYI2QzLce_s_30000 | Bicycle bell | 0.5-1.6 | yes | yes | yes | bicycle bell | A person holding a bike bell with "I MY BIKE" on it walks th | 1/1 |
| rDju9bBFvA4_80000 | Keys jangling | 1.3-1.8 | yes | yes | yes | brass keys | A person is holding a small lock and then a set of keys | 1/1 |
| rDju9bBFvA4_80000 | Keys jangling | 2.9-3.6 | yes | yes | yes | metal keys | A person is holding and manipulating a set of keys | 1/1 |
| rDju9bBFvA4_80000 | Keys jangling | 5.7-6.2 | yes | split | yes | metal key | A person holds a silver key | 1/1 |
| snR9TvRupjU_0 | Caterwaul | 4.1-10.0 | yes | yes | no | orange cat | A small | 1/1 |
| tyrj_wZor4U_0 | Bicycle bell | 3.4-4.8 | yes | no | yes | bicycle | A man on a bicycle rides past a brick building and stops to  | 1/1 |
| un_dog_barking_3doKyrCe | Howl | 0.0-6.0 | yes | split | yes | brown dog | A leashed brown dog stands in a field of dry grass | 1/1 |
| un_dog_barking_3doKyrCe | Howl | 6.0-12.0 | yes | split | yes | brown dog | A brown dog on a leash stands in a field of dry grass | 1/1 |
| un_dog_barking_3doKyrCe | Bark | 12.0-13.3 | yes | no | yes | brown dog | A brown dog on a leash is being pulled by a person while a s | 1/0 |
| w8_dashcam_ambulance_behind_1a | Siren | 6.3-12.3 | yes | split | yes | ambulance | A yellow and green emergency van drives down a tree-lined ro | 1/1 |
| w8_dashcam_ambulance_behind_1a | Vehicle | 13.1-20.0 | yes | yes | yes | police car | A police car with flashing lights follows a yellow van down  | 0/1 |
| w8_duel_diner_horn_1a | Engine | 0.0-5.1 | yes | yes | yes | truck | A man drives a truck down a desert highway | 1/1 |
| w8_kids_fire_alarm_school_1b | Keys jangling | 5.4-7.7 | yes | yes | yes | keys | A person inserts a key into a wall-mounted security panel | 1/1 |
| yszKc4m-W9U_30000 | Animal | 0.7-1.5 | yes | no | yes | crow | A man wearing a beanie and ear protection lights a small exp | 1/1 |
| yszKc4m-W9U_30000 | Aircraft | 6.4-6.9 | yes | yes | no | airplane | A man looks down and then to the side | 1/1 |
| yszKc4m-W9U_30000 | Machine gun | 8.4-9.7 | yes | yes | no | machine gun | A man in a black robe stands in a rocky | 1/1 |
| yszKc4m-W9U_30000 | Laughter | 9.7-10.2 | yes | split | yes | man | A man in a dark robe stands in a rocky | 1/1 |
| z32IXFJ4sKk_80000 | Crying, sobbing | 2.2-6.2 | yes | no | yes | the child | A man in a white tank top playfully picks up and hugs a smal | 1/1 |

### 2c. Missed (gold seen, gate kept)

| clip | sound | stretch | name | a/b | desc | named | describe | gold vis/obv |
|---|---|---|---|---|---|---|---|---|
| -8pCMgGKZY8_12000 | Bark | 0.7-1.2 | yes | no | no | dog | A black bull and a white and black cow are eating from a tro | 1/0 |
| -8pCMgGKZY8_12000 | Dog | 3.1-4.1 | no | no | yes | nothing | A black sheep chases a small dog around a grassy yard | 1/1 |
| 0YsC6M4GFoc_4000 | Explosion | 3.2-3.8 | no | no | no | nothing | A car performs a burnout in a parking lot at night | 1/1 |
| 0YsC6M4GFoc_4000 | Car alarm | 3.9-4.5 | no | no | yes | nothing | A car performs a burnout in a parking lot at night | 1/0 |
| 0YsC6M4GFoc_4000 | Car alarm | 4.8-5.5 | no | no | no | nothing | A vehicle performs a burnout in a parking lot at night | 1/0 |
| 0YsC6M4GFoc_4000 | Car alarm | 5.9-6.5 | no | no | no | nothing | A police officer stands in a dark parking lot at night as th | 1/0 |
| 0YsC6M4GFoc_4000 | Car alarm | 6.9-7.7 | no | no | no | nothing | Smoke billows from a row of parked police vehicles in a dark | 1/0 |
| 0YsC6M4GFoc_4000 | Car alarm | 7.8-9.5 | no | no | no | nothing | Thick smoke drifts across a wet | 1/0 |
| 2gbNCkCPl7w_380000 | Whistle | 2.0-2.5 | no | no | no | airplane | A man and woman look at a leaf | 1/1 |
| 2gbNCkCPl7w_380000 | Slam | 2.6-3.1 | yes | no | no | door | A man carries a woman through a field | 1/1 |
| 2gbNCkCPl7w_380000 | Creak | 3.4-4.0 | no | no | no | nothing | A commercial airplane is shown taking off from a runway | 1/1 |
| 4A5RpztQf-A_30000 | Bark | 6.6-7.3 | yes | no | no | black dog | A man sits at a desk with a Gizmo plushie | 1/1 |
| 6sRQnpEEQ0E_390000 | Laughter | 4.6-5.7 | no | no | no | nothing | A man stands near a palm tree at night while a large firewor | 1/1 |
| 97aoiaWwRVk_20000 | Gasp | 1.1-1.6 | no | no | no | nothing | A train moves along an elevated track while a man and woman  | 1/1 |
| 9NYsJxa4wjA_40000 | Crying, sobbing | 0.0-2.1 | no | no | no | nothing | A praying mantis and a frog sit together until a blue butter | 1/1 |
| 9NYsJxa4wjA_40000 | Slap, smack | 4.3-4.8 | no | no | no | nothing | A small frog catches a blue butterfly in its mouth while a l | 1/1 |
| 9NYsJxa4wjA_40000 | Crying, sobbing | 4.8-6.8 | no | no | no | the large green frog | A small frog watches a large frog scream at a butterfly | 1/1 |
| 9NYsJxa4wjA_40000 | Insect | 7.0-8.8 | yes | no | no | flying insects | A large | 1/1 |
| b3_aquarium_walk | Water | 0.0-4.7 | no | split | yes | nothing | A school of fish swims through the blue water | 1/1 |
| b3_aquarium_walk | Water | 4.7-9.3 | no | split | yes | nothing | A school of fish swims through the blue water | 1/1 |
| b3_aquarium_walk | Water | 9.3-14.0 | no | split | yes | nothing | A school of fish swims through the blue water of an aquarium | 1/1 |
| b3_aquarium_walk | Water | 14.0-18.7 | no | split | yes | nothing | A shark and a manta ray swim through the blue water alongsid | 1/1 |
| b3_aquarium_walk | Water | 18.7-23.3 | no | split | no | nothing | A shark and a manta ray swim together through a blue underwa | 1/1 |
| b3_aquarium_walk | Water | 23.3-28.0 | no | split | yes | nothing | A shark swims along the ocean floor while a school of small  | 1/1 |
| b3_leafblower_fall | Vehicle | 0.0-5.2 | no | no | no | nothing | A camera pans across a patch of green grass scattered with b | 0/1 |
| b3_leafblower_fall | Vehicle | 5.2-10.5 | no | no | no | nothing | A camera pans across a patch of green grass with scattered b | 0/1 |
| b3_leafblower_fall | Vehicle | 10.5-15.8 | no | no | no | nothing | A camera pans across a patch of green grass and weeds with s | 0/1 |
| b3_leafblower_fall | Vehicle | 15.8-21.0 | no | no | yes | nothing | A leaf blower is clearing a pile of dry | 0/1 |
| ev_smoke_alarm_kitchen | Frying (food) | 5.5-10.8 | no | no | yes | nothing | A camera pans across a kitchen counter showing a plate of sa | 1/1 |
| lx_chainsaw_and_power_tool_W1VYWwY | Chainsaw | 13.4-17.8 | yes | split | no | chainsaw | A person in a green hoodie uses a tool to carve a large pump | 1/1 |
| m4_clay_shoot_11a | Fire | 12.1-12.6 | no | no | yes | shotgun | A man in a shooting stand aims a shotgun at targets in a fie | 1/1 |
| m4_clay_shoot_11a | Fire | 14.1-14.8 | no | no | no | shotgun | A man in a cap and sunglasses aims a shotgun from a wooden s | 1/1 |
| m4_film_1917_33a | Explosion | 7.2-9.6 | no | split | no | nothing | A man in a trench coat walks through a muddy trench while so | 1/1 |
| m4_film_1917_33a | Explosion | 10.8-11.6 | no | no | no | nothing | Soldiers in a trench charge up a steep | 1/1 |
| m4_film_1917_33a | Explosion | 15.0-15.7 | no | no | no | nothing | A young soldier stands frozen in a trench while other soldie | 1/1 |
| m4_fire_bodycam_23a | Crack | 0.0-4.9 | no | no | yes | nothing | Firefighters are actively battling a house fire | 1/1 |
| m4_fire_bodycam_23a | Crack | 4.9-9.8 | no | no | no | nothing | A firefighter in full gear uses a hose to spray water on a h | 1/1 |
| m4_fire_bodycam_23a | Crack | 9.8-14.7 | no | no | yes | nothing | A firefighter in full gear is actively battling a house fire | 1/1 |
| m4_fire_bodycam_23a | Crack | 14.7-19.6 | no | no | yes | nothing | A firefighter in full gear approaches a house with visible f | 1/1 |
| m4_live_fire_26a | Fire | 2.1-3.6 | no | no | no | nothing | Two military tanks are positioned on a wide | 1/1 |
| mc_break_in_scene | Glass | 4.1-8.7 | yes | split | no | cracked glass screen | A group of people | 1/1 |
| mc_break_in_scene | Glass | 8.7-13.2 | yes | no | no | cracking glass | A group of people | 1/1 |
| mc_flood_scene | Water | 0.0-4.3 | no | no | yes | nothing | A massive wave of water and ice crashes down from the mounta | 1/1 |
| mc_flood_scene | Water | 4.3-8.5 | no | no | no | nothing | A massive | 1/1 |
| mc_flood_scene | Water | 8.5-12.8 | no | no | no | nothing | A plane is on the tarmac at night while air traffic controll | 1/1 |
| mc_riot_scene | Helicopter | 5.3-10.7 | no | no | no | nothing | A car with its headlights on drives through a dark | 1/1 |
| mc_riot_scene | Helicopter | 10.7-16.0 | no | no | no | nothing | A red car with its headlights on is driving at night | 1/1 |
| mc_thriller_basement | Water | 10.9-14.9 | no | no | no | nothing | A woman with blonde hair is seen in a dark | 1/1 |
| mc_thriller_basement | Water | 14.9-18.9 | no | no | no | nothing | A woman walks through a dark | 1/1 |
| movie_junglebook_95 | Roaring cats (lions, tigers) | 7.4-9.0 | no | no | no | nothing | A young boy runs through a dark field at night holding a fla | 1/1 |
| mv_bank_robbery_alarm | Truck | 2.4-3.3 | no | no | no | nothing | A person is seen getting into the back of a white van at nig | 1/1 |
| mv_bank_robbery_alarm | Door | 4.8-5.5 | no | split | yes | white van | A person is seen getting into the back of a white van at nig | 1/1 |
| mv_bank_robbery_alarm | Ping | 8.7-11.9 | no | no | no | nothing | A white van is shown at night | 1/1 |
| mv_bank_robbery_alarm | Door | 11.9-12.4 | no | no | yes | nothing | A vehicle is shown crashing through a barrier at night | 1/1 |
| mYI2QzLce_s_30000 | Bicycle bell | 6.3-7.4 | yes | no | no | bicycle bell | A person holds up an "I love my bike" button while walking o | 1/1 |
| qCMCR45l55c_30000 | Doorbell | 0.0-0.9 | yes | split | no | doorbell | A cartoon boy in a green shirt points at a picture on the wa | 1/1 |
| qCMCR45l55c_30000 | Doorbell | 2.2-3.5 | yes | no | no | doorbell | A cartoon boy in a green jacket points at a picture on the w | 1/1 |
| qCMCR45l55c_30000 | Doorbell | 4.8-6.3 | no | no | no | nothing | A cartoon boy in a green jacket points at a picture on the w | 1/1 |
| qCMCR45l55c_30000 | Doorbell | 7.5-8.9 | no | no | no | nothing | A cartoon boy in a green jacket points at a picture on the w | 1/1 |
| snR9TvRupjU_0 | Hiss | 3.2-4.3 | no | no | no | cat | A camera zooms in on a row of metal cages | 1/1 |
| snR9TvRupjU_0 | Hiss | 9.2-10.0 | no | yes | no | orange cat | An orange tabby cat lies on a white surface | 1/1 |
| w8_dashcam_ambulance_behind_1a | Siren | 0.2-6.3 | no | no | no | nothing | A silver car is parked on the side of the road while other v | 1/1 |
| w8_dashcam_ambulance_behind_1a | Shofar | 1.6-5.1 | no | no | no | nothing | A silver car pulls out from the curb into traffic | 1/0 |
| w8_dashcam_ambulance_behind_1a | Honk | 5.8-8.0 | no | no | yes | nothing | A dark grey hatchback pulls out from a side road on the righ | 1/0 |
| w8_dashcam_avalanche_road_2b | Alarm | 3.8-8.5 | no | no | yes | nothing | A massive avalanche of snow and ice crashes down a mountains | 1/0 |
| w8_dashcam_avalanche_road_2b | Alarm | 8.5-13.3 | no | no | yes | nothing | A massive avalanche of snow and debris crashes down a mounta | 1/0 |
| w8_dashcam_avalanche_road_2b | Alarm | 13.3-18.0 | no | no | no | nothing | People in red jackets are shoveling a massive pile of snow t | 1/0 |
| w8_duel_diner_horn_1a | Aircraft | 2.2-7.9 | no | no | no | nothing | A man with a mustache and glasses drives a car | 1/0 |
| w8_kids_fire_alarm_school_1b | Alarm | 17.6-18.1 | no | split | no | nothing | A person's hand presses a red button on a wall-mounted secur | 1/1 |
| w8_kids_fire_alarm_school_1b | Clang | 7.7-8.2 | yes | split | no | keys | A person inserts a key into a wall-mounted Simplex security  | 1/1 |
| yszKc4m-W9U_30000 | Firecracker | 1.9-2.7 | no | split | no | nothing | A man wearing ear protection feeds a crow a small object | 1/1 |

## 3. The "nothing" escape at Q1 (DEV, 145 stretches; coordinator: Round 50L counted 98/145 unnamed under its own strict question)

Under the shipped VISIBLE_PROMPT: 'nothing' on 84/145 stretches = 32 of 76 gold-SEEN stretches (42%) and 52 of 69 gold-NEEDED (75%). On the 32 seen-but-'nothing' stretches the a/b voted no 32/32 and desc yes only 4/32, so all 32 were kept (= 32 of the 38 DEV misses). The describe sentences there name the scene, never the maker: Thunder -> 'Heavy rain falls on a dark'; Bell (as_church_bell) -> 'The camera pans across the red brick facade of a large'; Crumpling -> 'A person is preparing a sheet of dough on a floured wooden c'; Footsteps (POV walker) -> 'A camera moves forward along a dirt path'; Honk -> 'A yellow Caterpillar 365C excavator digs into a pile'; Glass -> 'A young man in a red sweater runs outside'. On needed stretches the 'nothing' answer is right 52/52 at Q1 and only 1 of those 52 was flipped to seen by a/b+desc (ly_ambulance Vehicle).

Round 50L is consistent with this: once a phrase IS named, the crop question read as a logit separates seen/needed (AUROC 0.77); the loss is upstream, at the refusal. The 32 seen-but-nothing stretches split into (a) maker on screen but the model will not commit: church facade with bell tower x3, excavator horn, bakery dough actions x5 (Crumpling x2, Frying, Chink, Tap), drum in a marching band, air horn in a march, rustle, arrow, fire alarm, baby cry = 16; (b) POV sounds whose maker is the camera-holder: footsteps x6 (gold vis=0 obv=1); (c) thunder with only rain in the frames x5; (d) gold-generous or off-frame: seagull for Quack/Duck, tornado for Ding x2, fireworks over a start line (vis=0 obv=1) = 5. Total 32.

## 4. Shipped DEV arm (SHIP8+MD3+WW5+SL_proposed; 49 r13 clips + 22 tg clips): every gate_votes.json stretch the gate silenced

Matched to gold by resolved family label + time overlap. "no gold sound" = detector output the annotator did not list (not a gate error; the picture would have been a false alarm anyway).

| part | clip | label | stretch | name | a/b | desc | named | gold | verdict |
|---|---|---|---|---|---|---|---|---|---|
| DEV1 | ambient_citywalk_nyc_2627 | Vehicle | 0.3-3.5 | yes | yes | no | yellow taxi | vis=1 obv=1 imp=1 (Vehicle 0.3-16.0) | OK |
| DEV1 | ambient_nature_rainforest_7629 | Bird | 0.8-4.8 | yes | yes | yes | macaws | vis=0 obv=0 imp=2 (Bird 0.1-16.0) | WRONG |
| DEV1 | ambient_transport_subway_10800 | Train | 0.1-5.1 | yes | no | yes | train | vis=1 obv=1 imp=1 (Train 0.1-15.8) | OK |
| DEV1 | ambient_transport_subway_10800 | Train | 5.1-10.2 | yes | no | yes | train | vis=1 obv=1 imp=1 (Train 0.1-15.8) | OK |
| DEV1 | ambient_transport_subway_10800 | Train | 10.2-15.2 | yes | no | yes | train | vis=1 obv=1 imp=1 (Train 0.1-15.8) | OK |
| DEV1 | ambient_weather_storm_16200 | Rain | 0.1-5.3 | yes | yes | yes | heavy rain | vis=1 obv=1 imp=1 (Rain 0.0-16.0) | OK |
| DEV1 | ambient_weather_storm_16200 | Rain | 5.3-10.5 | yes | yes | yes | rain | vis=1 obv=1 imp=1 (Rain 0.0-16.0) | OK |
| DEV1 | ambient_weather_storm_16200 | Rain | 10.5-15.8 | yes | yes | yes | rain | vis=1 obv=1 imp=1 (Rain 0.0-16.0) | OK |
| DEV1 | ambient_weather_storm_7200 | Rain | 0.1-5.3 | yes | yes | yes | rain | vis=1 obv=1 imp=2 (Rain 0.1-15.8) | OK |
| DEV1 | ambient_weather_storm_7200 | Rain | 5.3-10.5 | yes | yes | yes | heavy rain | vis=1 obv=1 imp=2 (Rain 0.1-15.8) | OK |
| DEV1 | ambient_weather_storm_7200 | Rain | 10.5-15.8 | yes | yes | yes | heavy rain | vis=1 obv=1 imp=2 (Rain 0.1-15.8) | OK |
| DEV1 | as_explosion_XJ8lc3I6 | Gunshot | 0.0-1.0 | yes | split | yes | AK-47 rifle | vis=0 obv=0 imp=3 (Gunshot 0.0-2.2) | WRONG |
| DEV1 | as_explosion_XJ8lc3I6 | Gunshot | 8.2-12.4 | yes | split | yes | ak-47 rifle | NO GOLD MATCH | no gold sound |
| DEV1 | as_explosion_XJ8lc3I6 | Gunshot | 12.4-16.5 | yes | split | yes | AK-47 rifle | NO GOLD MATCH | no gold sound |
| DEV1 | b3_aviary_birds | Bird | 0.3-5.8 | yes | yes | yes | red crested bird | vis=1 obv=1 imp=2 (Bird 0.0-27.8) | OK |
| DEV1 | b3_aviary_birds | Bird | 5.8-11.3 | yes | yes | yes | birds | vis=1 obv=1 imp=2 (Bird 0.0-27.8) | OK |
| DEV1 | b3_aviary_birds | Bird | 11.3-16.8 | yes | yes | yes | birds | vis=1 obv=1 imp=2 (Bird 0.0-27.8) | OK |
| DEV1 | b3_aviary_birds | Bird | 16.8-22.3 | yes | yes | yes | birds | vis=1 obv=1 imp=2 (Bird 0.0-27.8) | OK |
| DEV1 | b3_aviary_birds | Bird | 22.3-27.8 | yes | yes | yes | birds | vis=1 obv=1 imp=2 (Bird 0.0-27.8) | OK |
| DEV1 | b3_construction_site | Vehicle | 0.0-4.9 | yes | yes | yes | excavator | NO GOLD MATCH | no gold sound |
| DEV1 | b3_construction_site | Vehicle | 4.9-9.9 | yes | yes | yes | excavator | NO GOLD MATCH | no gold sound |
| DEV1 | b3_construction_site | Vehicle | 9.9-14.8 | yes | yes | yes | excavator | NO GOLD MATCH | no gold sound |
| DEV1 | b3_construction_site | Vehicle | 14.8-19.8 | yes | yes | yes | excavator | NO GOLD MATCH | no gold sound |
| DEV1 | b3_crossing_bells | Train | 0.0-5.2 | yes | yes | yes | train | vis=1 obv=1 imp=2 (Train 0.0-17.0) | OK |
| DEV1 | b3_crossing_bells | Train | 5.2-10.5 | yes | yes | yes | train | vis=1 obv=1 imp=2 (Train 0.0-17.0) | OK |
| DEV1 | b3_crossing_bells | Steam | 0.2-4.6 | yes | yes | yes | steam train | NO GOLD MATCH | no gold sound |
| DEV1 | b3_pet_shop | Bird | 0.1-4.7 | yes | yes | yes | a small bird | vis=0 obv=0 imp=2 (Bird 0.1-27.8) | WRONG |
| DEV1 | b3_pet_shop | Bird | 4.7-9.3 | yes | yes | yes | a small bird | vis=0 obv=0 imp=2 (Bird 0.1-27.8) | WRONG |
| DEV1 | b3_pet_shop | Bird | 9.3-13.9 | yes | yes | yes | a small bird | vis=0 obv=0 imp=2 (Bird 0.1-27.8) | WRONG |
| DEV1 | b3_pet_shop | Bird | 13.9-18.5 | yes | yes | yes | European robin | vis=0 obv=0 imp=2 (Bird 0.1-27.8) | WRONG |
| DEV1 | b3_pet_shop | Bird | 18.5-23.1 | yes | yes | yes | a small bird | vis=0 obv=0 imp=2 (Bird 0.1-27.8) | WRONG |
| DEV1 | b3_pet_shop | Bird | 23.1-27.8 | yes | yes | yes | birds | vis=0 obv=0 imp=2 (Bird 0.1-27.8) | WRONG |
| DEV1 | bell_miami | Bell | 0.2-5.0 | yes | no | yes | church bell | vis=0 obv=0 imp=3 (Bell 0.2-14.5) | WRONG |
| DEV1 | bell_miami | Bell | 5.0-9.7 | yes | no | yes | church bell | vis=0 obv=0 imp=3 (Bell 0.2-14.5) | WRONG |
| DEV1 | bell_miami | Bell | 9.7-14.5 | yes | no | yes | church bell | vis=0 obv=0 imp=3 (Bell 0.2-14.5) | WRONG |
| DEV1 | ly_ambulance_(siren)_-yPSgCn | Vehicle | 0.0-4.9 | yes | yes | yes | white car | vis=0 obv=0 imp=2 (Vehicle 7.3-8.8) | WRONG |
| DEV1 | ly_ambulance_(siren)_-yPSgCn | Vehicle | 4.9-9.8 | no | yes | yes | nothing | vis=0 obv=0 imp=2 (Vehicle 7.3-8.8) | WRONG |
| DEV1 | ly_ambulance_(siren)_-yPSgCn | Vehicle | 9.8-14.8 | no | yes | yes | nothing | vis=0 obv=0 imp=2 (Vehicle 7.3-8.8) | WRONG |
| DEV1 | ly_applause_62ZYD0u | Laughter | 0.3-4.2 | yes | yes | yes | woman | vis=1 obv=1 imp=2 (Laughter 0.0-13.8) | OK |
| DEV1 | mv_arrest_street_scene | Human locomotion | 0.1-0.4 | yes | yes | yes | man | NO GOLD MATCH | no gold sound |
| DEV1 | mv_arrest_street_scene | Footsteps | 0.0-1.0 | yes | split | yes | old man | vis=1 obv=1 imp=1 (Footsteps 0.0-10.5) | OK |
| DEV1 | mv_protest_scene_movie | Baby cry, infant cry | 4.0-6.0 | yes | no | yes | baby | vis=1 obv=0 imp=2 (Baby cry, infant cry 3.2-5.1) | OK |
| DEV1 | mv_storm_scene_house | Explosion | 17.8-19.9 | no | yes | yes | nothing | NO GOLD MATCH | no gold sound |
| DEV1 | un_driving_motorcycle_4O3bZRYO | Vehicle | 4.1-8.0 | yes | yes | yes | white car | vis=1 obv=0 imp=3 (Vehicle 4.8-10.2) | OK |
| DEV1 | un_driving_motorcycle_4O3bZRYO | Vehicle | 8.0-12.0 | yes | yes | yes | white car | vis=1 obv=0 imp=3 (Vehicle 4.8-10.2) | OK |
| DEV1 | waterfall_kawaida_01 | Water | 0.1-5.5 | yes | yes | yes | waterfall | vis=1 obv=1 imp=2 (Water 0.0-11.2) | OK |
| DEV1 | waterfall_kawaida_01 | Water | 5.5-11.0 | yes | yes | yes | waterfall | vis=1 obv=1 imp=2 (Water 0.0-11.2) | OK |
| DEV1 | waves_herzliya | Water | 0.1-1.8 | yes | yes | yes | ocean waves | vis=1 obv=1 imp=2 (Water 0.0-12.6) | OK |
| DEV2 | tg_d016 | Vehicle | 0.8-3.5 | yes | yes | no | black car | NO GOLD MATCH | no gold sound |
| DEV2 | tg_d030 | Vehicle | 5.7-11.3 | yes | yes | no | scooter | NO GOLD MATCH | no gold sound |
| DEV2 | tg_d030 | Vehicle | 11.3-17.0 | yes | yes | no | scooter | NO GOLD MATCH | no gold sound |
| DEV2 | tg_d054 | Vehicle | 1.0-5.8 | yes | split | yes | John Deere tractor | NO GOLD MATCH | no gold sound |
| DEV2 | tg_d085 | Laughter | 11.5-12.2 | yes | split | yes | woman in blue sari | NO GOLD MATCH | no gold sound |
| DEV2 | tg_d085 | Water | 2.8-3.8 | yes | yes | yes | ocean waves | vis=1 obv=1 imp=1 (Water 0.0-17.0) | OK |
| DEV2 | tg_d085 | Water | 8.4-10.2 | yes | yes | no | ocean waves | vis=1 obv=1 imp=1 (Water 0.0-17.0) | OK |
| DEV2 | tg_d088 | Rain | 0.1-4.2 | yes | yes | no | rain | vis=1 obv=1 imp=1 (Rain 0.0-14.8) | OK |
| DEV2 | tg_d088 | Rain | 13.8-14.8 | yes | split | yes | rain | vis=1 obv=1 imp=1 (Rain 0.0-14.8) | OK |
| DEV2 | tg_d133 | Fart | 0.2-2.5 | yes | no | yes | boxer dog | vis=0 obv=0 imp=2 (Fart 0.0-1.1) | WRONG |
| DEV2 | tg_d133 | Fart | 6.3-8.0 | yes | no | yes | boxer dog | vis=0 obv=0 imp=2 (Fart 5.6-6.4) | WRONG |

Counts: {'OK': 29, 'WRONG': 16, 'no gold sound': 14} of 59 silenced stretches. WRONG = pet_shop Bird x6 (ii), rainforest Bird x1 (ii), bell_miami Bell x3 (i), ly_ambulance Vehicle x3 at the SOUND level: only the 4.9-9.8 stretch overlaps gold's needed Vehicle 7.3-8.8 (named nothing, a/b letter + describe-could carried it); the 0-4.9 and 9.8-14.8 stretches were matched label-only (a white van IS on screen at 0-4.9), so count 1 stretch wrong, 1 sound wrongly silenced, as_explosion Gunshot 0-1 s x1 (i: AK-47 on screen, the heard shot at 0-2.2 s is off-screen; its later Gunshot stretches match gold's visible Machine gun 10-18 s), tg_d133 Fart x2 (i: boxer dog on screen, action concealed). Same clips and same mechanism as gate-gold.

## 5. Alternative raw replies on the DEV wrong silences (human2 strict naming + Round-43 DOING a/b; cf = counterfactual a/b; box = Round 38 crop)

- rainforest_7629 Bird: human2 named 'the macaws' / 'the parrots' / 'none'; DOING a/b -> (b),(a) = NO in 2/3; box: 'None' in stretch 1, box on a macaw in stretch 2 (crop said Yes). Frames: two scarlet macaws perched, not calling (checked).
- b3_pet_shop Bird (6): human2 'the bird' / 'the robin'; DOING -> NO in 4/6, split (b)(b) in 2/6; cf q1/q2 yes; box always boxes the robin, crop 'Yes'. Frames: a European robin on a mossy stump, beak closed; the audio is pet-shop birds (b3 = foreign ambient audio).
- bell_miami Bell (3): human2 'the church' -> DOING NO 3/3; box: 'None' / tiny box on the facade, crop 'no' -> flip 3/3. Describe: 'A camera pans right across a pink church'. Shipped Q1 'church bell' = label echo, no check.
- b3_golf_course Whack 6.5: human2 'the golfer', DOING split (b),(b) [letter bias]; box on the club, crop 'yes'. Frames 5.5-8.1 s: golfer addressing the ball, no swing (checked); gold: strike off-screen.
- mv_protest Crowd: human2 'the group of people' -> DOING NO 4/4; cf NO; box: 'none' / unparsed ('Frames 1-5 show a small, formal gathering in an interior room ... not typically described as a [crowd]'). Frames: 5-8 people in a candle-lit room (checked).
- mv_tornado Siren 8.9: human2 'the siren speaker array' -> DOING NO; box on 'an emergency vehicle', crop 'no' -> flip. Frames 7.9-9.6 s DO show a tornado-siren horn array, then a singer (checked); the gold 'not visible' is disputable here.
- ly_ambulance Vehicle 7.3: human2/human 'none' / 'nothing'; cf NO; box on 'a car or truck', crop 'no'. Shipped: name NO, a/b YES, describe 'A car drives down a wet' -> could-question YES. Frames: a white van ahead leaves the frame by 7.7 s (checked).

## 6. Vote-rule re-scoring on gate-gold DEV (sound level, imp>=2; 43 seen / 36 needed): re-weighting the three votes does not fix it

| rule | seen silenced | needed kept | balanced |
|---|---|---|---|
| majority (shipped) | 16/43 (0.37) | 31/36 (0.86) | 0.617 |
| majority AND a/b yes (a/b veto) | 10/43 | 34/36 | 0.589 |
| majority AND a/b not-no | 14/43 | 32/36 | 0.607 |
| a/b alone | 11/43 | 34/36 | 0.600 |
| name AND a/b | 10/43 | 35/36 | 0.602 |
| unanimous | 8/43 | 35/36 | 0.579 |
| majority AND named != nothing (FIX_GATE) | 16/43 | 32/36 | 0.630 |

The a/b production question is right on the wrong silences only 7/16 (DEV) and 15/20 (TEST) and wrong on many correct silences, so vetoing on it trades one error for another: the information is not in the three answers. FIX_GATE is a free +1 needed-kept (ly_ambulance) with no loss.

## 7. Classification and what it means

### Counts (stretch level, importance>=2)

| class | DEV gate-gold (16 wrong) | DEV pipeline extra (tg_d133 Fart, as_explosion Gunshot) | TEST gate-gold (20 wrong, audit only) |
|---|---|---|---|
| (i) object present, not the maker/action now | 5 (bell_miami x3, golf Whack, ly_ambulance Vehicle) | +3 (Fart x2, Gunshot 0-1 s) | 9 (bell_kazansky x3, airsoft rifle, live-fire machine gun x2, car door, fire-alarm panel, police SUV for Siren) |
| (ii) other instance / look-alike of the same kind | 10 (pet_shop robin x6, perched macaws x2, people in a room for a mob x2) | 0 | 7 (restrepo mounted gun vs distant fire x2, bicycle for Bell / Bicycle bell x2, brown dog for another dog, cartoon characters for a laugh track x2) |
| (iii) answer-format / readout | 0 pure; 1 shared with (i): ly_ambulance (Q1 nothing, a/b letter yes, describe-could yes) | 0 | 2 (w8_dog_fireworks: named "fireworks" with describe "A person greets their dog": label echo) |
| (iv) frames at the wrong moment | 0 pure; 1 shared with (v): mv_tornado (horn at 7.9-9.6 s, stretch 8.9-15.6) | 0 | 0 |
| (v) question vs gold definition / gold disputable | 1 (mv_tornado Siren: a siren horn IS on screen) | 0 | 2 (1J_j6 lightning for Thunderstorm; q4Z8 fire truck for its own siren) |

Under the task's strict definition ('visible' = the maker is on screen, not 'obvious'), 4 more silences become class (v): b3_favela footsteps x3 (DEV) and w8_dashcam Vehicle (TEST) have gold vis=0 obv=1 (gate_gold scores them as correct because seen = visible OR obvious).

Class (iii) bound: the shipped a/b letters are not stored (only the booleans), so letter bias on the shipped question is observable only as the split rate: 12/145 DEV stretches split (yes 38, no 95); Q1 truncation is bounded by the 24-token budget + the 5-word cut and shows up as 'Based on the visual evidence' in Round 43's cache, not in the shipped one.

Dominant: (ii) on DEV (10/16, but 6 of those are one clip, b3_pet_shop, whose video and audio come from different places), (i) across DEV+TEST (14 + 9 = 23 of 36 when the pipeline extras are added), i.e. **presence is taken for production**. (iii) and (iv) are small. (v) is 3 cases where the gate is arguably right and the gold strict.

### The mechanism, in the answers themselves
1. Q1 asks "name the thing making that sound". The model names the most plausible object: 'church bell' when it sees a church, 'machine gun' when it sees a gun, 'golf club' when it sees a golfer, 'a small bird' when it sees a robin. 30/54 DEV and 49/83 TEST "seen" stretches are label echoes (the named phrase repeats the label word) and then by_name = yes with no check at all. For the other names the check is the text-only MAKES_SOUND_PROMPT, "Could that thing be what is making that sound?", which is a world-knowledge "could", not a "is it, now, in these frames" (macaws could; people could; a loudspeaker could; a boxer dog could fart).
2. Q3's vote is not independent: if the describe sentence mentions the named object (robin / bird, church, people, macaws) it is an automatic yes (8/16 DEV wrong silences), otherwise it goes through the same "could" question (7/16). So two of the three votes measure presence/plausibility.
3. Only Q2 asks about production ("visibly making that sound") and it is noisy: yes on 9/16 DEV wrong silences, no/split on 15/20 TEST ones, and no/split on many correct silences (ab-veto costs 6 of 16 correct silences on DEV, section 6). Majority therefore = presence.
4. On the other side (misses), Q1 answers 'nothing' on 42% of gold-seen stretches (32/76 DEV) and then a/b says no 32/32: the model refuses to commit when the maker is small, partly hidden or the action is fine-grained (bell tower on a facade, excavator horn, dough crinkling, drum in a march), when the maker is the camera-holder (POV footsteps), or when only the effect is in frame (rain for thunder). Those 32 are 32 of the 38 DEV misses, so the miss problem is a refusal problem, not a judgement problem (Round 50L: once named, the crop logit separates at AUROC 0.77).

## 8. Three fixes implied by the audit (with the rounds already run on each)

History that bounds them: Round 43/43b (production a/b, generated letters) STOP at 0/41 and 5/41 silenced; Round 50 (crop question, text) STOP; Round 50L (crop question as logit margin) STOP at step 0 because 98/145 stretches had no phrase from Round 50's naming, while on named sounds the margin separated at AUROC 0.77 (bell_miami 'the church' -6.0, pet_shop Bird -2.5, mv_protest Crowd -1.9 vs waterfall +2.6, rain +1.5, rustle +2.3); Round 38 SYNC/BOX/CF STOP; Round 44 AVNAME (Omni names the picture from audio+video) STOP; Round 64 CONCEALED-ACTION {Bell} PASS on DEV (rescues bell_miami by label rule).

**Fix 1 (readout, class i: 5/16 DEV + 9/20 TEST + Fart/Gunshot in the pipeline): a production VETO on the shipped 'seen' stretches, read as a logit margin on the SHIPPED named phrase.**
Not a new prompt and not a replacement: keep the shipped majority, and for every stretch it calls seen (53/54 DEV have a named phrase; the 54th is FIX_GATE) ask once, logit-read as in Round 50L, "Is this {named} making the {label} sound right now?" on the same 6 frames (or the Round-50 crop), and un-see the stretch when the margin is below t. This is the 50L signal where coverage is not the problem: the wrong silences are exactly the stretches that DID name a phrase (15/16), so 50L's coverage failure does not apply, and 50L never tried the veto form (it tried REPLACE and 4TH VOTE). Also switch FIX_GATE on (free +1 needed kept, section 6). Would fix by sound: bell_miami/bell_kazansky (church named, margin -6), golf Whack, machine gun not firing (airsoft, live-fire), car door, fire-alarm panel, police SUV, Fart, Gunshot 0-1 s; at risk: correct silences with weak action (alarm clock, cellphone buzz, flea-market Rustle, motorcycle 'split'). Test: CPU/one forward pass per stretch on the cached gate-gold dev54 frames (~54 reads), sweep t, Round-38 GO bar (kept >= 35 & silenced >= 15, or silenced >= 19 & kept >= 32); if GO, one DEV pipeline arm = SHIP8+MD3+WW5+SL + veto, scored with merged_dev.py; TEST once at the end as usual.

**Fix 2 (model input, class ii: 10/16 DEV, 7/20 TEST): a kind-MATCH question with the audio in the prompt, not a naming question.**
The frames cannot show that the robin is not the pet-shop bird, that the perched macaws are not the calling bird, or that eight people in a room are not the mob; the only cue is what the sound sounds like. Round 44 asked Qwen3-Omni to NAME the picture's sound from audio+video and STOPPED; the untried form is a match: audio cut + the stretch's frames + the shipped named phrase, "You hear {label}. Is the sound you hear coming from the {named} you see, or from something off screen? Answer: this / off-screen", logit-read. Bound the expectation: 6 of the 10 DEV cases are one clip (b3_pet_shop, b3 = video with foreign ambient audio) and Round 38 SYNC (Synchformer) found no usable synchrony signal, so this is the lowest-confidence fix; it is listed because class ii is the largest DEV class and nothing else can reach it. Test: gate-gold dev54, the 10 class-ii stretches + the 38 correct silences, GPU (Omni), same GO bar; the b3 unseen_ambient clips are the acid test.

**Fix 3 (readout for the misses, the 'nothing' escape: 32/38 DEV misses): a second naming pass seeded by the describe sentence, before 'nothing' is accepted.**
When Q1 says nothing, re-ask with the describe sentence as context and a candidate list of its nouns ("You described: 'A camera pans across the red brick facade of a large church'. Which of these could be making {label}: church / facade / camera / none?"), then put the chosen phrase through Fix 1's margin. Honest ceiling: bucket (a) of section 3 = 16 stretches (church facade x3, excavator Honk, bakery x5, drum, air horn, rustle, arrow, alarm, baby cry); the describe sentence names a plausible maker in about half of them (excavator, dough/sheet, marching group, facade), so expect ~6-8 stretches (~3-4 sounds) regained; POV footsteps (6), thunder without lightning (5) and the gold-generous 5 are out of reach (footsteps are a label rule like Round 64: Walk/footsteps with a moving camera = 'obvious', keep). Round 43b's strict format and Round 43c HUMAN-ADD did not try the describe seed. Test: gate-gold dev54 (cached frames, one extra generation per 'nothing' stretch = 84 calls), then DEV arm.

Not recommended from this audit: re-weighting the three votes (section 6: every re-weighting is at or below the shipped majority), frame shifting (class iv is 0-1 cases; the gateshift B.3 run also found it stable), or another VLM alone (the errors are in what is asked and read, not in what the model can see).
