# Worked examples for the meeting — with the real numbers

All three demo clips are test clips and are Figure 2 of the thesis. Scores are the judge's
(0–4) under the human-grounded reference; "blind" = a picture for every sound, no gate.

## A. Police car on screen — correct silence (`demos/clean_un_police_car_siren…`)

- Sounds detected: Siren (0.84), Vehicle. The police car is in frame.
- Gate: in every 5-second stretch, at least two of the three votes said the source is visible → **silenced**. Panel stays empty.
- Score: ours **4**, blind **3** (it drew a police car anyway — redundant), caption 3.
- Say: "The absence of a picture is the feature. The blind baseline loses one point here."
- Trap: under the proposal's *original* (circular) reference, this correct silence scored **0** — the reference said the siren was 'missing' because the detector heard it. That is why the reference was replaced.

## B. New York street — wrong silence (`demos/clean_ambient_citywalk_nyc…`)

- Sounds detected: Vehicle (0.36), Speech, Music (below bar). Cars are visible on the street.
- Gate: Vehicle judged visible in every stretch → **silenced**. The label says the sound that matters is off-screen traffic → a picture was due.
- Score: ours **0**, blind **3**, caption 2.
- Say: "This is the honest failure. Two readings: the gate is defensible — cars *are* visible — and the label is debatable. Four of the 11 withheld pictures on test are this kind; the other seven are the detector never firing."
- Debug version shows: `0.36 Vehicle — visible`. The detector fired; the gate declined.

## C. Ambulance behind the camera — correct picture (`demos/clean_ly_ambulance…`)

- Sounds detected: Siren (0.84, 0–17.8 s), Vehicle (0.61).
- Gate for Siren: all four stretches → "nothing visible is making it" → picture "ambulance siren blaring" for the whole sound. Vehicle: visible in 3 of 4 stretches → also shown (the rule silences only if visible in *every* stretch) → second picture "car horn beeps".
- Score: ours **4**, blind **3** (its description talked about brake lights, not a siren), caption 2.
- Say: "The siren gets a picture from the first second. The second picture, a horn, is the Vehicle sound — shown because the road was out of view in one stretch. Defensible, but it is a redundant picture; that is the kind of cost the cost table counts."

## Likely questions at the demos

- **"Why is the picture a cartoon / generic?"** — FLUX draws a 2–5-word phrase on a white background; a generic picture avoids claiming details the sound cannot support (a *red* ambulance would be invented). Stock photos scored +0.17 higher but 55% had restrictive licences.
- **"Why does the ambulance clip also show a horn?"** — the Vehicle detection was out of view in one of four stretches; "silence only if visible everywhere" keeps it. It is counted as cost.
- **"Why silent on the New York clip?"** — the gate saw cars; the label says off-screen traffic. Label ambiguity; four such clips on test.
- **"What does the detector miss?"** — sounds under loud speech or music: crowd noise at 0.22 under Speech 0.84, rain at 0.23 under Music 0.81; the bar is 0.35. Seven of the 11 withheld pictures.
- **"Why not lower the bar?"** — the trade-off figure: lowering it adds phantoms (a whale at a Christmas market at 0.36); the sweep on the development split found 0.35 already best.
- **"Why not a bigger vision model?"** — tried (32B): it silences more visible sources (67% vs 35%) but wrongly silences 25% of off-screen sounds (was 3%); under the score that is worse.
- **"Why no user study?"** — the proposal specified automatic evaluation; I state that helpfulness to deaf viewers is untested (decision 4).
