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

## 7. The rebuilt judge: trust checks (139 clips, `v4b4` renders)

`judge_direct.py` (Gemma-4-31B sees the pictures; reference = template over the annotator's ticks;
0 of 417 replies unparsed). Bars unchanged from `docs/judge_plan.md`:

    system          B1 Spearman rho vs viewer cost      B2 clean - wrong-picture clips
    proposed        -0.722 [-0.815, -0.617]  PASS       +1.11 [+0.61, +1.58]  PASS
    blind           -0.661 [-0.763, -0.542]  PASS       +1.33 [+0.80, +1.84]  PASS
    caption         -0.730 [-0.799, -0.644]  PASS       +0.89 [+0.41, +1.36]  PASS

Far stronger than the old describe-then-judge chain (Gemma there: B1 passed narrowly; Mistral failed B2).
Paired judge scores on these renders (a superseded row, used to calibrate the instrument; it contains
TEST clips already read on 22 Sept, so it is not a new look at the current system):

    all 139     ours 2.05   vs blind +0.27 [+0.07, +0.50] *   vs caption -0.04 [-0.22, +0.15]
    DEV 49      ours 2.04   vs blind +0.39 [+0.08, +0.73] *   vs caption +0.10 [-0.18, +0.39]
    seen 44     ours 2.45   vs blind +1.00 [+0.59, +1.45] *   vs caption +0.23 [-0.18, +0.64]
    unseen 33   ours 1.27   vs blind -0.15 [-0.61, +0.27]     vs caption -0.21 [-0.61, +0.15]

Caveats: captions are judged as text tags naming the sound, pictures as pixels the judge must
recognise; at most 4 pictures per clip are shown to the judge.

**B3, self-agreement** (`judge_direct_v4b4_repeat.json`: 20 clips x 3 systems re-judged with sampling on,
temperature 1.0): 59 of 60 scores identical, 1 off by one point.

**The current DEV system** (`judge_direct_dev_monocap_v31.json`; all three arms at the new timing, 49
clips; B1/B2 pass for all three arms here too, e.g. ours rho -0.71 [-0.87, -0.50], gap +0.98 [+0.24, +1.68]):

    DEV 49      ours 2.29   vs blind +0.20 [-0.18, +0.61]     vs caption +0.06 [-0.31, +0.39]
    seen 18     ours 3.11   vs blind +1.00 [+0.33, +1.67] *   vs caption +0.56 [0.00, +1.11]
    unseen 14   ours 0.93   vs blind -0.50 [-1.36, +0.29]     vs caption -0.50 [-1.36, +0.29]

A tie with both baselines overall; the gate's value is on the clips whose source is on screen, and the
pictures are weakest on the off-screen clips the project exists for.

## 8. Two controls

**Hardware.** `dev_repro_v32L` (switches off, on an L40S with the 27B gate CPU-offloaded) vs
`dev_repro_v31` (A100): every shared start identical, no needed sound changed, but 21 vs 23 spans
(F1 0.293 -> 0.297). So the card class moves about one picture in 49 clips. The TEST base ran on A100
and the TEST arm on L40S; on TEST the gate differed on 1 of 60 clips. Rule from now on: comparison arms
run on the base's card class (H200/A100).

**Is V3 gate-neutral?** No. On the H200 (`dev_v3gate_v32h`) vs `dev_monocap_v31`: the same 14 hits,
but 2 clips gain a picture (FA/clip +0.08 [0.00, +0.20], dF1 -0.019 [-0.049, 0.00]). Cause: the
duplicate check compares the subjects' wording; today's generic "Vehicle" was depicted as the train
the scene showed, so it merged with "Train"; V3 depicts it as a generic vehicle, so it no longer merges,
and the family keeps its bursts outside the train as a second picture (b3_laundromat, ly_helicopter).
Not fixed tonight: changing the duplicate rule on the two clips that showed it would be tuning. Stated
as V3's gate cost; a fix (merge by the audio — overlapping bursts with an ancestor/descendant source —
rather than by wording) is a separate DEV change for the panel.
