# Detector + visibility improvements, 30 Sept – 1 Oct 2026 (for the supervisor)

Score: a **hit** is a picture of the right sound type that starts 0.5 s before to 1.0 s after the sound starts. **Wrong** =
a picture of a sound whose source is on screen (visible), a different sound at that moment / wrong time (cross), or nothing
(phantom). Cost per clip = (4·miss + 2·wrong) / clips (lower is better). DEV = 71 clips (49 old + 22 tagger), TEST = 88
clips (60 old + 28 tagger). Every rule was written down before its numbers (docs/prereg_round13_detector_push.md).

## Shipped chain

| step | what it does | DEV hits / wrong / cost | TEST hits / wrong / cost (p vs old) |
|---|---|---|---|
| old pipeline (B0r) | — | 18 / 51 / 3.690 | 21 / 40 / 2.909 |
| TO1+F7F8 | audio-LLM listeners rescue weak detections | 26 / 45 / 3.070 | 22 / 38 / 2.818 |
| + N2b, DR2, K-V4 | listener checks on weak / masked spans | 27 / 38 / 2.817 | 23 / 36 / 2.727 |
| + DV | drop a sound DASM never hears in the clip | 27 / 31 / 2.620 | 23 / 34 / 2.682 |
| + BTP | start a late picture at an earlier weak run of the same sound | 28 / 30 / 2.535 | 23 / 33 / 2.659 (0.033) |
| + CONT | drop a picture that is only a later piece of a sound already going on | 28 / 25 / 2.394 | 24 / 32 / 2.591 (0.025) |
| + FineLAP veto | a 2026 detector double-checks rescued sounds | 28 / 24 / 2.366 | 24 / 31 / 2.568 (0.015) |
| **+ K4A-D (shipped)** | drop a picture neither listener names, unless DASM hears it | **28 / 21 / 2.282** | **23 / 29 / 2.568 (0.017)** |

DEV numbers are on the corrected gold (Adam's blind visibility re-check: 3 of 14 disputed sounds changed; random control 0 of 20).

## Tried and rejected (each pre-registered, each failed its bar)

| idea | why it failed |
|---|---|
| newer audio-LLM ears (MOSS-Audio, SpotSound, Kimi, Step-Audio) | add ~3–10 wrong per extra hit |
| FineLAP in place of DASM | worse separation: 25 / 39 |
| onset post-processing (SEBB 2024) | −7 hits |
| visibility votes: SSL-SaN, SigLIP picture-vs-frame, audio identity, box-and-crop (BOX) | each un-silences real visible sounds |
| DASM impact peaks → picture (+ VLM naming) | peaks fire 9× more often on other moments |
| repeat-picture rules (RPT-S, DBR) | no net gain at full pipeline |

## Where the remaining errors are (shipped, DEV)

| misses (30) | count | wrong pictures (21) | count |
|---|---|---|---|
| no model hears it | 12 | source on screen | 6 |
| visibility check silences it | 5 | wrong sound at that moment | 7 |
| listeners say no / F8 filter | 7 | repeat, early or late | 6 |
| picture at the wrong time | 6 | nothing there | 2 |

Ceiling study: with every stage perfect on the same detections, 43 hits / 3 wrong / cost 0.761; 12 misses no rule can reach.

## Secondary cost (Adam, 1 Oct): a picture of an on-screen sound counts half

See benchmark/gold/visible_weight_sweep.md. No past decision changes. A "more hits" profile will hold changes that win only
when on-screen pictures count half, and only if every extra wrong picture is an on-screen one.

## Night 1 Oct (running log)

| experiment | idea | result |
|---|---|---|
| IMP-V | DASM impact peak → VLM names it → visibility check | golf whack named right, but the check sees the golfer; 28 / 23: rejected |
| BOX (Adam) | VLM boxes the sound's source; "is this an X?" on the crop | fixes the church bell, un-silences 7 visible sounds: rejected |
| BOX-2 (screen) | BOX, but "no box" no longer flips; strict re-ask | +2 needed kept (church bell, siren), 3 visible calls lost: rejected (one too many) |
| BOX-2 arm | BOX-2 inside the full pipeline (SHIP8+BOX2) | 29 hits (+1, church bell) but 26 wrong (+5: 3 visible, 1 cross, 1 phantom), cost 2.282 → 2.366: rejected under both rules |
| E1 SYNC | does on-screen motion move in time with the sound (Synchformer) | as a veto: +3 needed kept (church bell, golf whack, ambulance) but 5 visible calls lost: rejected; strong signal for visible sounds (sync ≥ 0.88 = on screen), two-sided version next |
| E1 SYNC-2 | Synchformer sync ≥ 0.998 (precision 1.0 on non-judge) ADDS "seen" to the gate | +3 visible sounds silenced (bakery crumpling, kitchen water, tap), nothing lost: 19/41 seen, 33/38 needed — on the GO bar exactly; with the veto 14/41, 36/38: rejected; touches no SHIP8 picture |
| E4 CF | ask the VLM the annotator's own two questions | no better (15/41 seen, 32/38 needed vs 16/41, 33/38): rejected |
| E5 GBTP | start a late picture where any detector first hears its sound | most pictures just move to 0 s; one hit lost (27 / 22): rejected |
| MAKER-VIS | ask if the picture's maker (train, parrot) is on screen | 9 hits lost (19 / 15): rejected |
| CONTRAST | ask the listener "(a) X (b) strongest other sound (c) neither" on refused faint sounds | +1 needed, +142 other accepts: rejected |
| DETACHED-ADD | add a 2-s picture where a drawn family is heard again, far from its picture | fires only on repeat textures: 28 / 28, +7 wrong, targets untouched: rejected |
| EXPECT | the scene VLM names sounds one would expect off-screen; the listener must name the same; shipped gate at the first weak detector run | +1 hit (church bell) but +5 wrong (2 visible, 2 cross, 1 phantom), cost 2.282 → 2.366: rejected under both rules. Found: the listener's top sound is often the missed one (hammer, train, bell), the scene VLM never proposes it |
| EXPECT-A | the listener alone lists the sounds (cleaner decode), word map to families, shipped gate at the first weak run | +5 hits (hammer, church bell, pet-shop crow, snicker, fart) but +83 wrong (64 cross, 24 phantom), cost 2.282 → 4.338: rejected under both rules. The open match spreads each heard sound over its neighbours and the gate cannot refuse off-screen wrong names |
| EXPECT-A2 | EXPECT-A with the word map only (no cosine) and at most two sounds per clip | +3 hits (hammer, church bell, fart) but +26 wrong (6 visible, 16 cross, 4 phantom), cost 2.282 → 2.845: rejected under both rules. Right names, wrong moment: the first weak run is usually 0 s |
| EXPECT-A3 | the EXPECT-A2 names, placed and confirmed by FineLAP (text-queried detector, bar 0.329, strongest run), then the shipped gate | +5 hits (hammer, favela train now placed right, church bell, laughter, fart) but +18 wrong (6 visible, 9 cross, 3 phantom), cost 2.282 → 2.507: rejected under both rules. Best recall of the series; the gate cannot refuse an off-screen wrong name |
| RELABEL-GATE | rename a picture to the sound both listeners heard, then ask if that is on screen | fires on 3 pictures, none of the 7 targets; drops 2 wrong but also a true hit (27 / 19, cost same): rejected |
| PRIOR415 | per-family precision of each detector on 415 held-out AudioSet clips (strong labels); drop a picture whose family the detector gets wrong > 70 % of the time, unless a listener heard it | lists frozen first (BEATs: 24 bad families, e.g. Glass, Gunshot, Water; FlexSED: Vehicle + 4 rare ones); 8 SHIP8 pictures hit a bad family but every one was heard by a listener, so nothing dropped (28 / 21, cost same): rejected |
| AGREE | a seen call flips only when the SYNC veto AND BOX-2 both say so | 15/41 seen, 34/38 needed: church bell rescued, phone buzz lost; below both bars: rejected |
| NAMED-VETO | drop a picture when the gate says a different thing makes the sound at that moment | 6 of 50 pictures judged, all dropped: 4 wrong gone but 2 true hits lost (26 / 17, cost same 2.282): rejected |

**Found:** the shipped display joins repeats within 1.5 s, but every score today was measured at 2.0 s. At 1.5 s the best has 2 more wrong pictures (28 / 23 / 2.338). Needs Adam's call: ship 2.0 s, or re-score at 1.5 s.

**Engineering (1 Oct night):** new clips now build their FineLAP scores automatically (src/listener_prep.py), so the shipped pipeline runs on clips outside DEV/TEST (needed for ComfyUI). Checked: identical scores on a DEV clip; stage-4 cache check passes on a live clip.
