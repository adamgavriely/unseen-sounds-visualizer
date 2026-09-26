# Chapter 5 — Results (draft, 27 Sept 2026)

*Order fixed before the final numbers were read (`docs/panel_2026-09-26_plan.md` §A, signed 5/5): the primary metric
first, then the declared secondary family with a multiple-testing correction, then the operating curve, then the
analyses that explain the result. Every number below is read from a committed file named beside it. Items marked
**[pending]** are filled when their run or sitting finishes.*

## 5.1 Set-up in one paragraph

Three systems share every stage up to the visibility gate: the same detector stack (BEATs ∪ FlexSED with two
cross-model vetoes), the same onset rule, the same label filter and the same renderer. **Ours** draws a sound only if the
gate finds its source off screen; **blind** draws every detected sound; **silence** shows nothing. A fourth arm, text
tags, is discussed in §5.8. Gold: one annotator, per-sound labels (label, onset, visible, obvious, importance). A needed
sound (importance ≥ 2, neither visible nor obvious) is *hit* when a picture of its family starts within [−0.5, +1.0] s of
its onset. Viewer cost per clip = 4 × missed needed sounds + 2 × wrong pictures. All intervals are paired clip bootstraps
(2000 draws, seed 0). TEST = 60 clips held out and scored once for this table (amendment 21); DEV = 49 clips used during
development, shown beside it for transparency.

## 5.2 Primary metric: F1 against drawing every sound — no significant difference

| | ours | blind | Δ (ours − blind) | 95 % CI | p |
|---|---|---|---|---|---|
| TEST, 60 clips, 43 needed sounds | 0.381 | 0.322 | **+0.059** | [−0.030, +0.144] | 0.18 |
| DEV, 49 clips, 36 needed sounds | 0.378 | 0.330 | +0.048 | [−0.054, +0.131] | 0.30 |

*Source: `benchmark/gold/holm_test_final_v33_test_bench.json`, `holm_dev_monocap_v31_dev.json`.*

The difference is positive on both halves and significant on neither. The design could not have detected it: with 60
clips the smallest effect detectable at 80 % power is about +0.13 (from the interval width), and about 320–570 clips would
be needed for a true +0.03–0.05. The reason the gate cannot move F1 much is structural (§5.6): F1 weighs a removed wrong
picture and a lost right one equally, and the detector misses most needed sounds before the gate sees them.

## 5.3 Declared secondary family (ours − blind), Holm-corrected, per table

| row | TEST Δ [95 % CI] | p | Holm | DEV Δ [95 % CI] | Holm |
|---|---|---|---|---|---|
| Precision | **+0.137** [+0.043, +0.256] | 0.002 | **0.010 ✓** | +0.115 [+0.020, +0.204] | 0.080 |
| Wrong pictures / clip | **−0.517** [−0.800, −0.283] | < 0.001 | **< 0.001 ✓** | −0.531 [−0.776, −0.306] | < 0.001 ✓ |
| Viewer cost / clip | **−0.833** [−1.434, −0.300] | 0.003 | **0.012 ✓** | −0.816 [−1.388, −0.204] | 0.040 ✓ |
| Clean-clip accuracy | **+0.219** [+0.086, +0.364] | 0.001 | **0.006 ✓** | +0.231 [+0.074, +0.391] | 0.030 ✓ |
| F0.5 | +0.110 [+0.017, +0.210] | 0.023 | 0.069 | +0.093 [−0.003, +0.170] | 0.186 |
| Recall | −0.070 [−0.154, +0.000] | 0.103 | 0.206 | −0.083 [−0.200, +0.000] | 0.186 |
| Weighted F1 | +0.022 [−0.074, +0.103] | 0.580 | 0.580 | +0.016 [−0.094, +0.102] | 0.705 |

✓ = survives Holm at 0.05 within its table. p is two-sided from the bootstrap draws (ties counted on both sides;
resolution 1/2000). Negative wrong-picture and cost differences are gains. The recall row is the price: the gate
silences some needed sounds (not significant after correction). Precision survives on TEST but not on DEV, so it is
stated as a TEST result, not a replicated one.

## 5.4 Against showing nothing (family 2, Holm over three rows)

| row | TEST Δ [95 % CI] | Holm |
|---|---|---|
| F1 (all clips; equals ours' F1, silence has none) | +0.381 [+0.217, +0.557] | < 0.001 ✓ |
| Viewer cost, all clips | −0.233 [−1.000, +0.467] | 0.585 |
| Viewer cost, clips whose sound is off screen (13)* | **−3.077 [−4.769, −1.385]** | **< 0.001 ✓** |

\*One pre-declared subgroup (amendment 5) of a post-hoc metric (amendment 9). On these clips 11 of the 13 pictures shown
are correct. The value is unchanged by the timing fix (the 23 Sep table gave the same cell). Over all clips the system
does not beat silence on cost: clips with nothing to draw can only cost, and busy "mixed" clips are its weak category
(ours 6.27 vs silence 5.33 per clip).

**Per category (TEST, cost per clip, CIs only):** off screen — ours 4.00, blind 4.00, silence 7.08; mixed — 6.27 / 8.00 /
5.33 (ours − blind −1.73 [−3.73, −0.13]); source on screen — 0.67 / 2.50 / 0.00 (−1.83 [−2.83, −1.00]); nothing to draw —
0.20 / 0.30 / 0.00.

## 5.5 The whole trade-off, not one weight

*Figure: `benchmark/gold/cost_curve_test_final_v33.png` (DEV twin `cost_curve_dev_monocap_v31.png`).*
The price of a wrong picture (β, a missed sound = 4) is not known for deaf viewers. Ours is cheaper than blind for any
β above 0.39 (DEV 0.46) and cheaper than silence for any β below 2.56 (DEV 2.33). The declared β = 2 lies inside both
ranges. The gate given the annotator's own sound list would cost 1.23 per clip at β = 2 — the headroom lies in detection.

## 5.6 Why F1 does not move: the detector, not the gate

- **Oracle diagnostics** (`docs/GOLD_RERUN_2026-09-22.md` §9, 22 Sep renders): replacing the detector with the
  annotator's sound list makes the gate's F1 gain significant (+0.067 [+0.012, +0.117]); adding only the missed sounds
  back, keeping every false alarm, already does (+0.075 [+0.029, +0.119]). Misses, not false alarms, cause the null.
- **Where needed sounds are lost** (DEV autopsy, 21 misses): 11 never detected, 5 timing, 3 gate, 2 label filter.
- **Detector on 280 human-labelled AudioSet-Strong clips** (descriptive, shipped bars): **[pending — D6 table]**.
- **Detector threshold and TEST overlap.** FlexSED's bar was chosen on a split overlapping 35 of the 60 TEST clips; on
  the clean DEV-49 it passes two of its three adoption rules and misses the third by 0.011 (`flexsed_recheck_dev49.json`).

## 5.7 The gate itself

| visibility judge (DEV gold sounds) | balanced accuracy |
|---|---|
| OWLv2 (object detector) | 0.50 |
| Qwen2.5-VL-7B, 3 votes | 0.61 |
| Qwen3.8-27B, 3 votes (shipped) | 0.62 |

Each vote alone: open naming 0.64, a/b in both orders 0.60 (keeps 94 % of needed sounds), description 0.56 — all within
the ±0.11 interval of the shipped majority (`gate_vote_table.py`). Three further mechanisms (any-stretch rule, OWLv2 and
SAM 3 per-stretch votes) each removed exactly two wrong pictures per needed sound lost — the break-even at β = 2
(amendments 14, 18). The remaining errors are *presence without source*: a visible bell tower silences an off-screen bell;
a visible tank silences a helicopter through the "kind of vehicle" rule (3 such kinship silences on DEV).
**Stability under a 0.5-s frame shift: [pending — B.3]**.

## 5.8 Pictures versus text

With the same gate decisions and the same spans (derived arm, identical on 49/49 DEV clips), the automatic judge
(Gemma-4-31B, trust checks passed) scores pictures and text tags alike: −0.04 [−0.22, +0.14]. The judge reads a text tag
as the label itself, so this measures no advantage for pictures; whether a picture is *understood* is measured by people
(§5.9). The ungated text baseline stays in the tables with its confound stated.

## 5.9 Pictures: blind human recognition

54 sounds from 50 never-annotated clips, each picture shown 1.5 s at 384 px, the rater typing what makes the sound,
three versions interleaved and hidden: today's generator (FLUX.1-schnell) 14 of 54 recognised; Qwen-Image-2512 with the
same text 26 (+0.22 [+0.07, +0.37]); with the V3 text 32 (+0.33 [+0.22, +0.46]); 9 of 10 repeated pictures answered the
same. V3 was rejected for one invented object (a thud drawn as a door), and the new generator alone shows one wrong
object (a ringing phone drawn as a desk bell), so neither is adopted as clean. Three automatic picture checkers failed
calibration against the human rater. **Final frozen setup: [pending — blind confirmation sitting]**.

## 5.10 Timing

A traced bug moved 223 of 442 picture starts earlier than their sound (worst 8.98 s). The fix (a start may never move
before its anchor) raised DEV F1 by +0.101 [+0.022, +0.196]; on TEST the change was inconclusive by its pre-written rule
(−0.019 [−0.127, +0.082]). It is kept as a bug fix; the gain is reported as DEV only. The 8-second picture cap was removed
on principle (a picture should last as long as its sound): 4 of 15 TEST pictures now outlast their sound by > 2 s.

## 5.11 Disclosures

One annotator (second annotator on 30 clips: **[pending]**). TEST exposures: 10 in total, 5 deliberate, all dated
(`docs/LEDGER_2026-09-26.md`, finding 1); this table is the fifth deliberate read and replaces the 23 Sep table by a rule
written before it was rendered. Viewer cost was declared after F1 came out null (its weights predate the gold). 49 DEV
clips were used in development. TEST holds more clips with nothing to draw than DEV (20 vs 8). Card class moves about one
picture in 49 clips.
