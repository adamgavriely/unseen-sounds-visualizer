# Ceiling analysis of SHIP7 on merged DEV — why can't we do better?

Script: `benchmark/gold/ceiling_ship7.py` (release v1.2.0) → `ceiling_ship7.json` (CPU, on the saved SHIP7 pictures, arm `SHIP6+FLAP|proposed`;
base reproduced 25/55 hits, 27 wrong (9 visible / 16 cross / 2 phantom), cost 2.451 on 71 clips; ledger = `ship7_errors.json`).
(labels as of that run; the final label file has 58 needed sounds)
An **oracle** fixes ONE stage perfectly with the gold and leaves every other stage as shipped. Cost = (4 × misses + 2 × wrong) / 71,
so one wrong picture = 0.028, one miss = 0.056.

## The oracle rows (each alone)

| row | hits / 55 | wrong (v/c/p) | cost | Δ cost | what it did |
|---|---|---|---|---|---|
| shipped SHIP7 | 25 | 27 (9/16/2) | 2.451 | — | |
| O1 perfect visibility gate | 30 | 18 (0/16/2) | 1.915 | **−0.536** | drops the 9 visible pictures; gives back 5 gate-silenced needed sounds (macaws, pet-shop birds, church bell, boxer dog ×2) |
| O2 perfect listener (ONCE / F8 / FLAP / gate as shipped) | 26 | 24 (7/15/2) | 2.310 | −0.141 | rejects 3 wrong rescues; 11 needed candidates accepted, but 8 are then blocked by F8 (DASM < 0.575), 1 by the gate (Fart 0.08) and 1 by ONCE (Fart 5.7, the family's second rescue); only Laughter 8.4 gets through |
| O2 with F8 / FLAP lifted | 34 | 24 (7/15/2) | 1.859 | −0.592 | the same 11 candidates, filters off: +9 hits (gate and ONCE still block the two Farts) |
| O3 perfect vetoes / filters | 30 | 27 (9/16/2) | 2.169 | −0.282 | gives back Hammer 13.8 and Explosion 2.8 (F8), Cat 2.9 (ONCE), Train 14.7 (CONT), Chicken 6.9 (B0 FlexSED clip veto). Vehicle 8.25 (0.32) and Dishes 16.5 (0.37) are under the 0.40 floor: not restorable |
| O4 perfect timing | 26 | 20 (11/7/2) | 2.197 | −0.254 | moves 9 cross pictures of the right family to their gold onset: +1 hit (applause Crowd 1.9); 6 late repeats become free duplicates (Shaver, Alarm ×2, Glass, Thunder 13.25, golf Bird); 2 land on a visible sound (Gunshot → Machine gun, Thunder 8.5 → 11.1) |
| O5 perfect family | 25 | 26 (14/10/2) | 2.423 | −0.028 | relabels 6 cross pictures; every gold at that moment is visible (Steam → Train, Pant/Dog → Chopping, Explosion → Thunder, Screaming → Bird, Crowd → Laughter): cross becomes visible, cost almost unchanged |
| O5+O4 family AND timing | 27 | 25 (9/14/2) | 2.282 | −0.169 | two pictures need both: Gunshot 8.25 → Footsteps 2.1, Goose 11.0 → Chicken 6.9 |

## Waterfall (all oracles together, applied in sequence: timing → family → both → gate → vetoes → listener)

| step | hits | wrong (v/c/p) | cost | marginal Δ | the stage is responsible for |
|---|---|---|---|---|---|
| shipped SHIP7 | 25 | 27 (9/16/2) | 2.451 | | |
| + O4 timing | 26 | 20 (11/7/2) | 2.197 | −0.254 | 1 hit, 7 wrong (6 are repeats of a sound already drawn) |
| + O5 family | 26 | 18 (14/2/2) | 2.141 | −0.056 | 2 wrong (5 cross pictures become visible pictures) |
| + O5+O4 joint | 27 | 17 (14/1/2) | 2.056 | −0.085 | 1 hit, 1 wrong |
| + O1 gate | 32 | 3 (0/1/2) | 1.380 | **−0.676** | 5 hits + 14 wrong (all visible pictures, including the 5 that O5 / O4 turned from cross into visible) |
| + O3 vetoes | 36 | 3 (0/1/2) | 1.155 | −0.225 | 4 hits |
| + O2 listener (filters perfect) | 43 | 3 (0/1/2) | 0.761 | −0.394 | 7 hits |
| **ceiling of the pipeline on this data** | **43 / 55** | **3** | **0.761** | | |

The order matters for the split between O4/O5 and O1: a cross picture at the moment of a visible sound is "family wrong" AND "visible";
here timing/family get it first (timing before family: a cross picture with an own-family gold is a timing error). Read the "alone" table
for the order-free view. In O4, barbershop Shaver and detective Alarm 3.36 read `new_class: hit` in the log because the moved copy takes the
nearest-onset match and the original picture becomes the duplicate: net zero, not a gain.

## What no oracle reaches (the residual: 12 misses, 3 wrong)

- **5 unheard** (O6): nyc Hammer 8.1 (BEATs 0.006, FlexSED 0.005, DASM 0.003, FineLAP 0.031), tg_d125 Explosion 5.4 (BEATs 0.004,
  FlexSED 0.03, DASM 0.15, FineLAP not queried), nyc_2627 Clang 3.8 and golf Whack ×2 (BEATs ≤ 0.015; FlexSED / DASM / FineLAP never
  queried for these families). 4 of 5 are under every model's lowest bar; the Explosion only clears DASM's 0.084 clip bar.
- **8 faint or mistimed**: below LO 0.5 (rainforest Bird 0.435, storm siren 0.42), under the picture floor (birds_forest Bird 0.22, tg_d032
  Thunder 7.4 0.27, ambulance Vehicle 0.32, tg_d095 Dishes 0.37), tg_d032 Thunder 2.8 (P2 run starts 2.04, outside the window). These need a
  lower bar, and every lower bar tried (CV54–CV76, PMC, N2c) bought wrong pictures. (That is 7; the 8th "faint" item, tg_d029 Chicken, O3 restores.)
- **3 wrong**: 2 phantoms (laundromat Train 1.0, hair-dryer Computer keyboard 11.25 — nothing in the gold at that time) and tg_d128 Hammer 9.0
  (a Clang is on screen since 5.6: wrong family AND late; no relabel or move fixes it).

## The three biggest ceilings

1. **The visibility gate: −0.68 of the 2.45 cost** (alone −0.54). 9 wrong pictures are of sounds the annotator calls visible, and 5 needed
   sounds are silenced because a same-family thing is on screen. Every gate idea (M / N / GA / Gemma / SUBJ / OM / PIC-SIM / SSL-SaN) failed:
   the frames cannot tell "this bell" from "that bell". This is also the group where the gold itself is a judgement call (see O7).
2. **The listener + F8 pair: −0.39 (with filters lifted −0.59 alone).** 11 needed sounds have a P2/PV candidate in the right window; the shipped
   ears refuse 7 of them (the ledger's "listener refused" rows), and even a perfect ear would lose 8 to F8 (DASM under 0.575: Air horn 0.05, Whistle 0.02, Gasp 0.07, Siren 0.20,
   Footsteps 0.26, Hammer 0.28, Clapping 0.35, Explosion 0.52). F8 stays because every relaxation (F8-U, K1, FLAP-F8) bought 3–10 wrong per hit.
3. **Vetoes and timing: −0.23 and −0.20.** Five needed sounds are removed by a veto that was shipped because it removed more wrong than right
   (F8 ×2, ONCE, CONT, B0 clip veto). Six wrong pictures are late repeats of a sound already drawn: a "one picture per sound" rule would remove
   them, but RPT-S (screen GO, arm fail) shows the arm-level version costs a hit.

Detector recall (O6) is the floor: 5 of 55 needed sounds (9 %) are heard by no model, and 8 more are heard too faintly. So even a perfect
decision layer tops out at 43 / 55 hits on this data.

## O7 — how sure is the gold?

- Annotator flags on the DEV gold: 10 needed sounds are marked **masked** (hard to hear: rainforest Bird ×2 + Cricket, as_explosion Footsteps and
  Explosion 5.6, storm siren, tornado Siren, tg_d029 Chicken, tg_d033 Siren, tg_d125 Clapping); 1 clip marked unsure (b3_favela_rio, the CONT Train).
  7 of those masked sounds are SHIP7 misses.
- 3 needed sounds carry a free-text label that we mapped ourselves ("clank" → Clang; "golf swings / ball strikes" → Whack, thwack ×2): all three
  are in the unheard group — no model was asked for that family.
- The 9 visible-wrong pictures are scene sounds (rainstorm thunder ×3, church bell, fire alarm, protest air horn, starter's bang, bath water,
  kids' laughter) and the 5 gate-silenced needed sounds have a same-family thing on screen (macaws, robin, church tower, boxer dog ×2). Both
  groups sit one tick away from the other class. If all 14 were ticked the other way the cost would land near 1.9 with no pipeline change
  (hit and needed counts both shift: two of the visible golds are importance 1 and one is "obvious", not "visible") — the whole size of the O1 ceiling. Low estimate of doubtful items: 3 (labels); high: 14 + 10 masked = 24 of the 55 needed + 27 wrong.

## TEST

`final_test_ship7.json` holds only the aggregate rows (SHIP7 24 hits / 41 misses, 31 wrong (5/21/5), cost 2.568 vs B0r 21 / 40 / 2.909 on 88
clips). No per-item TEST data is stored and TEST gold was not read, so the TEST breakdown is skipped.

## Notes on method

- Restored spans (O1 / O3 / O2) have no generated image (the stage that removed them ran before stage 6); the oracle assumes the picture would
  have been drawn. The shipped gate's verdict on a restored span is the `augment` flag of a same-family spec within 0.5 s (SHIP7 or the F8-less
  TO1+F7 arm); no such spec → drawn (pessimistic for the wrong side, none appeared).
- A moved late repeat becomes a free duplicate under `score_per_sound` (a picture of an already-matched sound is `dup`, not wrong); that is the
  "merge repeats" credit and is what O4's −7 wrong mostly is.
- Needed-class rule for listener candidates (family match and run start in the onset window) counts 17 DEV P2/PV items vs the file's 15
  `hit_needed` (two family-level matches the file's stricter rule did not count); it only decides which candidates a perfect ear accepts.
- The Chicken 6.9 span (B0 veto, no row exists) is reconstructed from BEATs frames ≥ 0.35 in the onset window.
