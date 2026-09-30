## Detector push: rounds 13–15 (listener rescue, filters, and negative results)

*Continues §5.8. Numbers are read from `docs/prereg_round13_detector_push.md` (rounds 13–15, amendments A–O),
`docs/dev_heard_dropped_best_2026-09-29.md` and `benchmark/gold/r13_test_final.md`. Intervals are paired clip bootstraps
(2000 draws, seed 0). All rows are **ours** (with the gate).*

### Goal and protocol

The detector was still the main cause of misses (§5.7–5.8). Rounds 13–15 asked: can the pipeline show **more needed
sounds** without more **wrong pictures** than those sounds are worth? The score is the modelled viewer cost of §5.1
(β = 2 assumed): (4 × missed needed sounds + 2 × wrong pictures) / clips. A wrong picture is **visible** (it
matches a sound the gold marks as not needed, e.g. thunder over a visible storm), **cross** (a real sound is playing,
but no unmatched sound of the picture's family starts there, e.g. a second shaver picture after the first one hit) or
**phantom** (no gold sound at all, e.g. "Train" in a laundromat). A hit saves 4 and a wrong picture costs 2, so one gained hit pays for at most two wrong pictures: this is
the **break-even**.

The baseline **B0r** is the configuration the frozen TEST table was scored with (PANNs veto 0.05), re-run in the same job
as each candidate; **B1** (the self-veto in `use_shipped()`) is shown beside it. The protocol was written before any
round-13 number:

- **DEV (49 clips, 36 needed sounds) is the development set.** It had already been spent as a confirmation set, so every
  DEV number here is exploratory.
- **Pre-registration.** Each round and amendment (A–O) was committed before its first output. An arm is **DEV-eligible**
  only if hits ≥ B0r + 1, wrong ≤ B0r + 2 × (hits gained), and cost < B0r.
- **TEST: one exposure** of one round-13 candidate, at the end of round 13. After that, TEST was spent.
- **New tagged clips** (the "tagger set") were split into DEV2 and TEST2 (batch 1 by a hash of the clip name, batch 2
  balanced by tags) and later **merged**: DEV = DEV + DEV2, TEST = TEST + TEST2. Selection happens on the merged DEV; the
  merged TEST is scored once, at the end, for the final candidate against the shipped configuration.

**The one TEST exposure.** Only R13-1 (defined below) was DEV-eligible in round 13. On TEST (60 clips, 43 needed) R13-1 equals B0r exactly (16 hits, 25 wrong, cost 2.63; no picture
changed): verdict "same", as predicted before the run; nothing shipped. B1 was worse on TEST too (15 hits, 29 wrong, cost 2.83; B0r − B1 = −0.20 [−0.47, +0.00]).
The interval touches zero; the pre-registered rule's one-sided p (0.043) is not read as significance here. This is a
**new TEST exposure after row 10 of §5.15**. (An earlier attempt stopped on a clip-list check before any score existed: a
plumbing failure, not an exposure.)

*Source: `benchmark/gold/r13_test_final.md`; prereg "Selection and TEST", "TEST result".*

### The best DEV arm: TO1+F7F8

**BEATs** is a tagger (it scores 527 AudioSet classes in each short window); its spans are drawn above 0.35. **FlexSED**
is a text-queried detector (it scores a sound named in words, e.g. "Gunshot"); its spans need 0.8. Many real sounds sit
in FlexSED's **band** (peak 0.5–0.8): heard, but below the bar. The **listener** is an audio language model
(Qwen3-Omni-30B-A3B, later also Audio Flamingo Next) that hears a short audio cut and answers a question.
**TO1+F7F8** stacks five parts:

1. **Confidence-tiered listener rescue (TIER).** A band run becomes a new span if the listener's **open inventory**
   (V4: "List every distinct non-speech sound you hear") names its family. At peak ≥ 0.6 Qwen alone decides; below 0.6
   both listeners must agree. A second leg keeps a FlexSED span ≥ 0.8 that the PANNs veto (a second tagger, PANNs, drops a
   FlexSED span it does not also hear) would remove, when the listener says yes (this gained a snow-walk Laughter).
2. **Once per family (ONCE).** At most one rescued picture per family per clip: the *earliest*, because the viewer needs
   the first onset.
3. **Twin lift (R13-1).** A weak BEATs span is raised over the display bar by a strong FlexSED twin of the same family
   (e.g. a BEATs Train-horn span lifted from 0.27 to 0.408 by FlexSED Train at 0.86).
4. **Listener-confirmed mirror veto (F7).** A BEATs-only span is dropped when FlexSED's top query there is another family
   (≥ 0.7) and its own family is weak (< 0.4), unless the listener confirms it. It removes three laundromat "Vehicle"
   phantoms but keeps a Glass hit that the plain veto lost.
5. **DASM vote (F8).** A rescued run is kept only if another text-queried detector, DASM, scores its family ≥ 0.575 within the run
   ± 0.5 s.

**DEV result.** TO1+F7F8: **18 of 36 hits, 24 wrong (6 visible / 12 cross / 6 phantom), cost 2.45**; B0r: 14 hits, 24
wrong (6 / 11 / 7), cost 2.78; **Δ −0.33 [−0.86, +0.08]**. It gains four needed sounds (a Cricket, the Laughter, a
Gunshot, an Explosion) and loses none; six wrong pictures appear and six go. It is cheaper than B0r at every β (β = 1:
1.96 vs 2.29) and on both DEV halves (A −0.16 [−0.56, +0.16]; B −0.50 [−1.50, +0.25]).

**DEV2 (batch 1: 6 clips, 3 needed).** 2 of 3 hits against B0r's 1, 8 wrong in both (cost 3.33 vs 4.00; Δ −0.67 [−2.00,
+0.00]). Same direction, but three sounds are not a test.

**Caveats.** (1) It is the best of many arms: counted from the log's tables (single-rule changes of amendment G
included), **106 full-pipeline DEV arms** were run, 80 up to and including the job that produced TO1+F7F8 and 26 after it,
none of which beat it. The winner of so many looks is biased upward. (2) The interval includes zero. (3) It has **not
been run on TEST**.

*Source: prereg "Round 14 amendments F and H", "Confirmation set 1, DEV2".*

### Main arms across rounds 13–15 (DEV, 49 clips, 36 needed sounds)

| arm | what it adds | hits / 36 | wrong (v / c / p) | cost β = 2 | Δ vs B0r [95 % CI] | eligible |
|---|---|---|---|---|---|---|
| B0r | baseline (scored config, re-run) | 14 | 24 (6 / 11 / 7) | 2.78 | — | — |
| B1 | self-veto (`use_shipped()`) | 13 | 30 (6 / 18 / 6) | 3.10 | +0.33 [+0.04, +0.69] | — |
| R13-1 | twin lift (round 13 rules) | 15 | 24 (6 / 11 / 7) | 2.69 | −0.08 [−0.37, +0.12] | yes |
| L05t3 | yes/no listener rescue (R13-3, best) | 22 | 65 (8 / 43 / 14) | 3.80 | +1.02 [−0.12, +1.88] | no |
| LR-V12+1 | stricter questions (amend. A, best) | 19 | 38 (7 / 21 / 10) | 2.94 | +0.16 [−0.45, +0.73] | no |
| LR-V12+1+F1F4F3+F7F8 | rescue filters (round 14, best) | 15 | 22 (6 / 10 / 6) | 2.61 | −0.16 [−0.49, +0.08] | yes |
| TIER+ONCE+1 | tiered listener (amend. E, best) | 19 | 34 (5 / 22 / 7) | 2.78 | +0.00 [−0.57, +0.57] | no |
| LR-V12+1+F1F4F3+F7F8@AG4 | two-listener AGREE (amend. C, best) | 16 | 23 (6 / 11 / 6) | 2.57 | −0.20 [−0.57, +0.12] | yes |
| XQ | 120 extra FlexSED queries (amend. D) | 13 | 29 (7 / 15 / 7) | 3.06 | +0.29 [−0.04, +0.65] | no |
| **TO1+F7F8** | TIER + ONCE + R13-1 + F7 + F8 (amend. F/H) | **18** | **24 (6 / 12 / 6)** | **2.45** | **−0.33 [−0.86, +0.08]** | **yes** |
| TO1F7F8+K1 | F8 bypass (amend. K) | 18 | 30 (5 / 19 / 6) | 2.69 | −0.08 [−0.69, +0.49] | yes |
| TO1F7F8+O | Whisper-AT vote instead of DASM (amend. O) | 16 | 26 (5 / 15 / 6) | 2.69 | −0.08 [−0.61, +0.41] | yes |

*Source: prereg round-13, R13-3, A, round-14, E, C/D, F/H, I/K and O tables.* v / c / p = visible / cross / phantom.
No interval excludes zero in favour of any arm. Eligible is not picked: several eligible arms lose to TO1+F7F8 on cost.

### Negative results, by mechanism

**Audio-LLM question forms.** The yes/no listener ("is X present?") finds band misses (up to 14 → 22 hits), but every
arm adds far more wrong pictures than 2 × the hits (best L05t3: 65 wrong, Δ +1.02). Four stricter forms (multiple
choice, paired control cut, localisation, open inventory) and one combination cut the wrong rescues, but none is
eligible (best 19 hits / 38 wrong, Δ +0.16). Under every form, about **3 wrong pictures per gained hit** (precision
about 24 % against the 33 % break-even).

**Specific queries (XQ).** 120 extra FlexSED queries (e.g. "Bird vocalization, bird call, bird song") add wrong pictures
and lose a Crowing hit in every arm; the four unheard misses stay unheard.

**Onset re-localisation (I1).** Starting each picture at the steepest rise of the evidence lands after the annotated
onset (e.g. Hammer 13.76 → 15.64 s): 7 hits lost, Δ +0.94 [+0.16, +1.71].

**Activity gate (I2).** Keeping a "visible" sound when the VLM says the source is not *producing* it right now: +2
hits, +6 visible wrong pictures (Δ +0.20 [−0.45, +0.82]).

**Motion timing (J3) and LLM timestamps (J4)**, both dropped at the screen. A motion peak lands in the hit window for 1
of 13 needed impulsive sounds (chance 0.21). The two listeners "agree" on a start time mostly by both answering 0, and
the agreed time hits the window less often than the run's own start (19 vs 24 of 74).

**BEATs weak band (P3).** Adding BEATs spans below the display bar made the best listener arm worse (Δ +1.39 [+0.20,
+2.37]). A screen of 272 such spans: even taking all of them recovers **none** of the 18 remaining misses.

**Obvious fixes.** Letting two listeners outvote DASM (K1) brings back two killed sounds but loses two others: same
hits, +6 wrong. Treating a "visible" gate vote that names no object as "not visible" lets the ambulance Vehicle cross picture
through in every arm (+1 or +2 wrong) and gains no hit.

**Rule audit (G).** Seven shipped rules were loosened one at a time on B0r and on TO1+F7F8. Every loosening adds 3–11
wrong pictures for at most one hit; none passed.

**Round-15 screens (offline, on TO1+F7F8's pictures).** A two-listener veto on drawn pictures removes 2 wrong and 2 hits.
A strict gate (silent if any vote says visible) gives 17 hits / 21 wrong / 2.41, but loses the rescued Laughter and
belongs to a rule family already rejected (§5.9); report only. "One sound, one picture" gives 17 / 22 / 2.45: same cost,
one hit lost.

**Set-of-Mark crop gate (M).** OWLv2 close-up crops shown to the gate VLM silence one more visible sound (16 → 17 of 43)
and lose one needed sound (31 → 30 of 36 kept). The GO bar (≥ 3 more silenced) failed.

**Audio-visual gate (N).** Qwen3-Omni with frames and sound silences 5 more visible sounds (16 → 21) but loses 3 needed
ones (31 → 28), below the break-even of 2 visible per needed: hearing the sound makes "visible" likelier for both kinds.

**Whisper-AT vote (O).** Whisper-AT, a tagger built to hear under speech, replaced DASM as the third vote. It keeps one
Explosion that DASM killed but, among other changes, loses a Train and the Laughter: 16 hits, 26 wrong, 2.69, worse than DASM on all three.

### Where the remaining 18 misses are lost

The per-sound trace of TO1+F7F8 (diagnostic only, no selection) gives each miss one cause, first match wins:

| cause | what it means | DEV misses |
|---|---|---|
| unheard | no detector reaches its heard bar near the onset | 3 |
| not drawable | label outside the drawable vocabulary | 1 |
| below every bar | heard, under every bar, listener never asked | 4 |
| listener rejected | asked; TIER said no | 4 |
| killed by F8 (DASM) | rescued, then dropped by the DASM vote | 2 |
| gate visible | the gate called the source visible (gold: not visible) | 3 |
| timing / merge | a picture exists but starts outside the hit window | 1 |
| **total** | | **18** |

*Source: `docs/dev_heard_dropped_best_2026-09-29.md`.* ONCE, F7 and the PANNs veto cause none.

**What it implies.** Four misses (a Hammer and two golf Whacks unheard, a Clang not drawable) are beyond any filter.
Four sit in a weak band under every bar: two FlexSED blips of 0.08 s (0.435 and 0.42, under the rescue's 0.5), a forest
Bird run skipped because a weak BEATs span covers it, and an ambulance Vehicle just under the BEATs bar (0.317 < 0.35).
Six are lost at the rescue decision: the listener names the louder sound (for Footsteps and Gasp it names the
explosions), or DASM scores below its bar (Hammer 0.281, Explosion 0.516). Three are gate errors (macaws, pet-shop birds,
a church bell) and one is a merge that starts a Crowd picture 1.9 s early. So the ceiling has four parts: unheard or not
drawable (4), weak band (4), gate (3) and the rescue decision (6). No filter tried in rounds 14–15 recovered one of these
without adding more wrong pictures than it was worth. Of the arm's 24 wrong pictures, 18 are B0r's own.

### Threats to validity

- **DEV reuse.** DEV was already spent as a confirmation set. It then selected among 106 arms, and its per-sound traces
  shaped later fixes (amendments F and K). Halves A and B are the same 49 clips, so their agreement is not replication.
  The DEV gain of TO1+F7F8 is an upper estimate.
- **Small DEV2.** Batch 1 holds 3 needed sounds; its interval [−2.00, +0.00] can neither confirm nor refute DEV.
- **Tagger suggestions.** The tagging tool showed detector suggestions to the annotator (`from_suggestion`,
  `suggestions_shown_at`); BEATs-only suggestions were opened on all 15 batch-2 clips. Gold on the new clips may lean
  towards what BEATs hears.
- **TEST history.** The old TEST part was read for earlier detector candidates (rounds 4–13; R13-1 once). The final
  candidate was chosen on DEV only and has never been run on TEST.
