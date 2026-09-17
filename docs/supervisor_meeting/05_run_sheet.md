# Run sheet — 30 minutes

Decisions must be reached by minute 20. Play the **clean** demos; open a **debug** one only
if he asks "why".

| min | do | say (short) | show |
|---|---|---|---|
| 0–2 | Opening | "This is my first update and it is late — sorry. The system is built and evaluated. It **ties** the simplest baseline, 3.17 vs 3.16 out of 4, and the thesis explains why: the sound detector. I plan to submit on the 18th unless you object, and I need four decisions." | nothing |
| 2–5 | What it does | "Seven existing models, none retrained. One decision matters: is the thing making this sound visible? If not, a picture beside the video." | `figures/fig_pipeline.png` |
| 5–7 | Demo A | "Police car on screen, siren loud — the panel stays **empty**. That is the feature." | `demos/clean_un_police_car_siren…` |
| 7–9 | Demo B | "Cars visible, but the labelled sound is off-screen traffic. My gate saw cars and stayed silent — the score gives that **0**. Honest failure; also a debatable label." | `demos/clean_ambient_citywalk_nyc…` (debug if asked) |
| 9–11 | Demo C | "No ambulance in frame; the siren gets a picture for the whole sound. Scored 4." | `demos/clean_ly_ambulance…` |
| 11–15 | Findings 1 + 2 | "Both systems are 0.45 below a perfect gate, in opposite places. Mine loses where a picture is due — 7 of 11 are sounds buried under speech or music. Seven fixes, bars declared first, none passed: every setting moves along this curve." | thesis p. 9 table; `figures/fig_detector_tradeoff.png` |
| 15–20 | Decisions 1–4 | Ask in order; write the answer in `06_questions_form.md`; move on. | form |
| 20–25 | Decisions 5–6, departures | "Two smaller things: format and the AI-disclosure wording. And here is what changed from the proposal." | update p. 1 |
| 25–30 | Next steps + parking lot | "Freeze numbers today 18:00, submit the 18th. Anything I could not answer goes here." | form, parking lot |

Skip unless asked: finding 3 (why automatic references failed), finding 4 (cost-sensitivity —
the crossover is poorly determined and invites a tangent), `fig_cost_sensitivity`.

**If the videos do not play:** `examples/*.jpg` are mid-clip frames; `figures/fig_examples.png`
is the same three clips frozen with the votes and scores.

**Numbers you may need (all in the report):** tie 3.17 vs 3.16, CI −0.15..+0.18; caption 2.89
(+0.28); oracle 3.62; picture-due 2.78 vs 3.02, no-picture 3.56 vs 3.30; DCASE 66.7% ± 6
(97% off-screen kept, 35% on-screen silenced); panel on 47% vs 60%; wasted 357 s vs 506 s;
274 clips / 100 test; self-agreement 78%, κ 0.60; 7 of 11 withheld = detector.
