# Work log, 2026-09-24/25 (Adam away five hours; supervisor meeting next day)

Mandate: *"fix everything we said, and consult Fable on everything you can improve and do it … we will
prepare for another supervisor meeting tomorrow."* Panel of five (P1 timing, P2 statistics, P3 models,
P4 evaluation, P5 DHH/supervisor), two rounds: `docs/panel_2026-09-25_brief.md`, `…_round2.md`.

## 1. The reproduction check (before any TEST number)

`dev_repro_v32` (every new switch off, code as of tonight) against `dev_repro_v31`: 23 of 23 starts
identical, every metric identical (F1 0.293, P 0.282, R 0.306, cost 3.18). The new code changes nothing
when its switches are off, so the TEST comparison below is between configurations, not code versions.
Both jobs' logs confirm the switches reached the run (`ONSET_MONOTONE True`, `MAX_SPAN None`, vetoes
0.3 / 0.05, FlexSED bar 0.8).

## 2. TEST, the second deliberate look (rule frozen in `docs/test_second_look.md`)

F1 0.400 -> 0.381, dF1 -0.019 [-0.127, +0.082]; recovered 2, lost 2 (one each to the onset rule, one
each to the cap). **Verdict: inconclusive** — the onset rule stays as a trace-proven bug fix, its gain is
DEV-only; the cap stays off (one chainsaw lost, one church bell gained). Full table in
`docs/test_second_look.md`. Supervisor sentence (P2): "On the 60 held-out TEST clips the timing fix
neither helped nor hurt (one sound gained, one lost, F1 -0.02 [-0.13, +0.08]); it stays as a bug fix
proven by the trace, its DEV gain is reported as DEV-selected, and removing the picture cap traded one
chainsaw for one church bell."

## 3. Pictures — what V3 got wrong, found before any picture was seen

V3's subjects on the 50 never-annotated clips invented things the audio never established (a Whoosh →
"a person swinging a sword" twice, a bare Siren → "a police car", a Thunk → "a heavy door", smashes →
"a person smashing a glass") and lost what the scene had established (a Vehicle on a farm had been a
tractor; now "a car driving down the road"). The panel's verdict (all five):

* Adam rates **today / N0 / N** as pre-registered (primary N vs today); N's failures are a result.
* **V3.1** (`PICTURE_SCENE`): the scene may add at most two qualifier words to the heard word ("car
  door", "farm vehicle"), never a new noun or a person; a list guard (`labels.names_forbidden`) refuses
  siblings, unfired kinds, kinds inside a tie; no model is asked to approve its own answer; a sound name
  with no qualifier is drawn as its visible effect. Committed before any N picture was seen.
* V3.1 is rated on a **fresh set** (the 30 slice-B clips), not on the 50 its rules were written from.
* Seeds 1–2 are drawn for every arm for the checker's column (seed noise); Adam sees seed 0 only.

## 4. The judge, rebuilt

No describer: Gemma-4-31B looks at the pictures the viewer was shown, with their times, against a
reference that is a fixed template over the annotator's ticks (`benchmark/gold/judge_direct.py`). This
removes both circular links (Qwen3.8 describing its own subjects; a reference written from the
detector's events). Trust checks B1/B2 rerun on the 139-clip `v4b4` renders before any ranking.

## 5. Fair baselines

Blind (draw every detected sound) and caption re-rendered on DEV with the onset rule on and the cap
off, so gate-vs-blind is like for like again (`*_dev_monocap_v31`). A TEST render of the baselines is
a further look: Adam's decision.

DEV 49, paired clip bootstrap (2000, seed 0), gate minus blind, both arms at the same timing:

    timing          F1 gate/blind   dF1                    dP                     dFA/clip               dcost/clip
    old (v30)       0.29 / 0.26     +0.033 [-0.043,+0.098]  +0.079 [+0.007,+0.144]  -0.47 [-0.69,-0.27]    -0.78 [-1.27,-0.29]
    new (monocap)   0.38 / 0.33     +0.048 [-0.054,+0.131]  +0.115 [+0.020,+0.204]  -0.53 [-0.78,-0.31]    -0.82 [-1.39,-0.20]

The onset fix helps the blind arm too (F1 0.26 -> 0.33), as the panel predicted; the gate's advantage
is unchanged in kind: precision, false alarms, viewer cost and clean-clip accuracy (+0.23 [+0.07,
+0.39]) significant, F1 not. Versus silence on DEV: F1 +0.38 [+0.24, +0.50]; viewer cost -0.16
[-0.74, +0.41], not significant.

## 6. Not done, on purpose

No end rule (ends already follow the sound: −0.06 s median); the bell lost to the visibility vote is
booked as the cost of removing the cap. Gate stability under frame shifts (P1) and the final
like-for-like reporting render (P2) wait until the picture setup is frozen.
