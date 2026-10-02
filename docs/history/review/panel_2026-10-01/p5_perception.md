# Panelist 5 — cross-component / DHH perception (round 1)

Yardstick: at w=2 break-even is 2 wrong per hit, at w=1 4 visible per hit. Every kill-flag relaxation sits at or above it (MD3 3 rows/1 hit; CONT 66/1; FlexSED 0.75 34/1). What is left: how uncertain evidence is SHOWN, how the gate is JUDGED, what the gold calls visible.

**P1 — Tiered output (confident / tentative) instead of the floor drop-or-keep.** Confident = conf >= 0.40 and gate not-seen (as shipped). Tentative = conf 0.35–0.40 (optionally flat-texture pictures: FlexSED family level >= 0.6 in the 1.5 s before start, rise < 0.1), rendered as a compact "?" chip, <= 1 per clip, in its own display pass (never chained with pictures). Band pictures DEV 5 = 2 hits / 2 phantom / 1 visible; the floor was selected on the held-out 415 (dC −0.227). Both DEV phantoms are flat textures. Why new: floor was only drop/keep; CONFIDENCE_FADE (continuous) was worse; this is a binary tier on ~5 % of pictures (ASR-caption pattern DHH users prefer). Test: reporting rule pre-registered; 10-chip glance check by Adam. ~2 h CPU.

**P2 — Gate judged under the task cost.** Silence iff P(seen) > 4/(4+w) = 0.67 at w=2, 0.80 at w=1. From votes: P(seen | majority-silenced) 0.76 DEV / 0.82 on 176 non-DEV calibration sounds; unanimous 0.89/0.89; split 0.67/0.78; kept 0.47/0.44. At w=2 the shipped majority rule already satisfies the cost rule: no change. At w=1 the unanimous rule is the w=1 profile's gate by derivation. DEV arm expectation: +1 hit (bell), +2 visible; about −2 units at w=1, tie at w=2. 1 h CPU.

**P3 — Gold secondary column "timing inferable from the frames"** (evaluation side). The 6 visible wrongs are scene sounds; the 5 gate-silenced needed are object-visible / action-invisible. Thunder under rain judged both ways across clips. Blind re-rating by Adam of 11 disputed items + >= 20 random controls, reported as a second column, never the main number.

Checked, dead: dominant-visible suppression; onset-rise veto; audio-strength gate tiebreak.

**First: P1.**
