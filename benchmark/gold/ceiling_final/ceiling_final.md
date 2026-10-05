# Ceiling analysis of the FINAL system (D' = SHIP8+MD3+WW5+SL) on merged DEV

Method = report Section 8.1 (`ceiling_ship7.py`, release v1.2.0), re-run on the final arm's saved DEV pictures.
Scripts: `ceiling_final/ledger_final.py` -> `ledger_final.json`, `ceiling_final/ceiling_final.py` -> `ceiling_final.json`.
Base reproduced exactly: **29 / 58 hits, 15 wrong (6 on screen / 7 other sound / 2 nothing), cost 2.056** on 71 clips.
Cost = (4 x misses + 2 x wrong) / 71: one wrong = 0.028, one miss = 0.056.

Changes vs the SHIP7 script (all needed for the final arm, none change an oracle's rule):
- picture loader passes `clip=` to `_display_spans` (else GROUP / DEPICT display rules are skipped: base came out 29/18/2.141);
- ledger chain extended SHIP6+FLAP -> SHIP7+K4AD -> SHIP8 -> +MD3 -> +WW -> +WW5 -> +SL (no miss is caused by these new stages);
- floor 0.40 -> 0.35 (the final arm has no picture floor; display / augment bar 0.35);
- gate-vote proxy also reads the SHIP8 chain arms (final first, then TO1+F7 as before). Inert here: the only "silenced" verdict it found is the final arm's own Fart (Digestive) spec.
- sanity line as in the original: needed-class rule on DEV P2/PV items 19 vs the file's hit_needed 15 (SHIP7 run: 17 vs 15).

## Each oracle alone

| row | hits / 58 | wrong (v/c/p) | cost | change | what it did |
|---|---|---|---|---|---|
| final D' | 29 | 15 (6/7/2) | 2.056 | - | |
| O1 perfect gate | 34 | 9 (0/7/2) | 1.606 | **-0.450** | drops 6 on-screen pictures; gives back 5 gate-silenced needed sounds (macaws, pet-shop birds, church bell, Fart x2) |
| O2 perfect listener (filters as shipped) | 29 | 11 (4/6/1) | 1.944 | -0.112 | rejects 4 wrong rescues; 10 needed candidates accepted but every one blocked (F8 x8, two of them also FLAP; gate 1; ONCE 1) |
| O2 with F8 / FLAP lifted | 37 | 11 (4/6/1) | 1.493 | **-0.563** | same candidates, filters off: +8 hits (the 2 Farts stay blocked by gate / ONCE) |
| O3 perfect vetoes | 35 | 15 (6/7/2) | 1.718 | -0.338 | gives back Hammer, Explosion (F8), Train (CONT), Chicken (B0 clip veto), Dishes 0.37 (TO1+F7F8 stage: mirror veto / rescue filters), Meow (ONCE); ambulance Car 0.32 is under the 0.35 bar |
| O4 perfect timing | 30 | 12 (8/2/2) | 1.915 | -0.141 | 5 moves: applause Crowd -> hit; protest Glass -> free duplicate; golf Bird -> importance-1 sound; Gunshot and Thunder land on on-screen sounds |
| O5 perfect family | 29 | 15 (9/4/2) | 2.056 | 0.000 | 3 relabels, each onto an on-screen sound: other-sound becomes on-screen, cost unchanged |
| O5+O4 family and timing | 30 | 14 (6/6/2) | 1.972 | -0.084 | Gunshot 8.25 -> Footsteps 2.1 |

## Waterfall (timing -> family -> joint -> gate -> vetoes -> listener)

| step | hits | wrong (v/c/p) | cost | step change |
|---|---|---|---|---|
| final D' | 29 | 15 (6/7/2) | 2.056 | |
| + O4 timing | 30 | 12 (8/2/2) | 1.915 | -0.141 |
| + O5 family | 30 | 11 (9/0/2) | 1.887 | -0.028 |
| + O5+O4 joint | 30 | 11 (9/0/2) | 1.887 | 0.000 |
| + O1 gate | 35 | 2 (0/0/2) | 1.352 | **-0.535** |
| + O3 vetoes | 41 | 2 (0/0/2) | 1.014 | -0.338 |
| + O2 listener (filters perfect) | 47 | 1 (0/0/1) | 0.648 | -0.366 |
| **ceiling on this data** | **47 / 58** | **1** | **0.648** | |

## Residual (no oracle reaches it): 11 misses, 1 wrong

- **5 unheard**: nyc Hammer 8.1, nyc_2627 Clang 3.8, golf Whack x2, tg_d125 Explosion 5.4. 4 of 5 are under every model's lowest bar; the Explosion clears only DASM's clip bar (0.152).
- **6 faint or mistimed**: rainforest Bird (FlexSED 0.435 < LO 0.5), storm siren (0.42 < 0.5), birds_forest Bird (row 0.22), tg_d032 Thunder 7.4 (row 0.27), ambulance Vehicle (row 0.32, N2b), tg_d032 Thunder 2.8 (listener run starts 2.04, outside the window).
- **1 wrong**: hair-dryer Computer keyboard 11.25 (nothing in the gold at that time).

## Compared with the report's SHIP7 ceiling (Section 8.1)

| | SHIP7 (report) | final D' |
|---|---|---|
| base | 25/55, 27 wrong, 2.451 | 29/58, 15 wrong, 2.056 |
| ceiling | 43/55, 3 wrong, 0.761 | 47/58, 1 wrong, 0.648 |
| biggest stage (waterfall) | gate -0.676 | gate -0.535 |
| unheard | 5 | 5 (same sounds) |

## Compared with the cloud approximation (from the decision trail, no saved pictures)

| row | cloud | exact | why they differ |
|---|---|---|---|
| timing | 34/12/1.690 | 30/12/1.915 | most likely the cloud did not rescore moved pictures. Rescored, only 1 of 5 moves becomes a hit (Glass is a duplicate, golf Bird is importance 1, 2 land on on-screen sounds) |
| family | 29/15/2.056 | 29/15/2.056 | same (wrong split moves 6/7/2 -> 9/4/2) |
| gate | 34/9/1.606 | 34/9/1.606 | same |
| vetoes | 32/15/1.887 | 35/15/1.718 | +3, most likely because the exact run rebuilds spans the trail does not hold (Chicken from BEATs frames, Meow ONCE-dropped row, Dishes 0.37 above the 0.35 bar) |
| listener (filters lifted) | 39/15/1.493 | 37/11/1.493 | same cost, different split: the exact oracle also rejects 4 wrong rescues (-4 wrong) and the 2 Farts stay blocked by gate / ONCE (-2 hits). 2 misses = 4 wrong in cost, so the totals match |
| waterfall end | 52/3/0.423 | 47/1/0.648 | the exact residual has 11 misses: 5 unheard + 6 faint / mistimed. The cloud (6 never-heard) most likely treated the faint ones as reachable |
