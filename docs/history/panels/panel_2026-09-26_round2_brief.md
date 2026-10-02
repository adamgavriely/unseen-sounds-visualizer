# Panel 2026-09-26 — round 2 brief: critique each other, then vote

Read all five round-1 answers: `docs/history/panels/panel_2026-09-26_round1.md`.

## Corrections to the round-1 brief (verified)
- **CAPTION is not gated** (P4 is right): `benchmark/run_protocol.py` l.156–161 sets `GATE_ENABLED=False` for
  `audio_caption`. Every "pictures vs caption" number compares gated pictures with *ungated* text.
- **The brief quoted an old TEST row** (P5 is right). The single deliberate TEST look of 23 Sep (`docs/history/preregistrations/prereg_v4.md`
  "THE SINGLE TEST LOOK", final cell, old timing, 60 clips): ours vs blind ΔF1 +0.047 [−0.057, +0.134] null,
  **ΔP +0.143 [+0.034, +0.264] sig, Δcost +0.73 [+0.20, +1.33] sig**; vs silence all 60 +0.33 [−0.33, +1.07] null,
  **unseen 13 clips +3.08 [+1.08, +5.38] sig** (precision 0.846 there). Then on 24 Sep TEST was read again for the
  timing change (inconclusive).

## Where you already agree (5/5 or 4/5)
1. ΔF1 vs blind cannot become significant at any n reachable in days (MDE ≈ 0.13 on DEV, ~320–570 clips for a
   true +0.03–0.05). Report it null; the structural reason is the oracle-label diagnostic.
2. ΔP, ΔFA, Δcost, clean-clip accuracy are already significant; report them as declared secondaries with a
   multiplicity rule (Holm over the declared family), the β cost curve, and the history of the primary.
3. Do not touch the frozen picture sitting; do not build a 4th automatic picture instrument.
4. Do not retune the gate on the 6 named losses; do not buy recall with lower bars or new detector rules.
5. A second annotator (25–30 stratified clips, κ on `needed` / `visible`) is the thing an examiner asks first.

## Where you disagree — vote and argue (≤ 6 lines each)
- **D1 Gate treatment.** (a) P3: an *attribution* vote ("the {named} in frame is making this sound now / it
  comes from out of view / can't tell", both orders) as a 4th vote, bar: DEV needed_kept ≥ 0.92 and
  seen_silenced ≥ 0.40. (b) P1: restrict the kinship silence (`reason.py` ~1324–1342) to cases where the logged
  `named` phrase is about the descendant; CPU re-decide from `gate_votes.json`; adopt iff 2 × recovered >
  re-admitted visible pictures. (c) P5: only the pre-declared "obvious" vote under the §10e stop rule (oracle ΔF1
  > +0.12, ≤ 1 hit lost). (d) nothing. Choose one (or an ordered pair), and write the exact bar. P2's warning:
  three mechanisms already landed at ratio 2.00.
- **D2 TEST reporting.** (a) P2: render blind + caption (and ours) on TEST at the current timing now, write
  before scoring that this table **replaces** the 23 Sep table whatever it shows, score only with Adam's yes.
  (b) P5: headline the 23 Sep table; onset fix footnoted as "DEV-selected, TEST inconclusive"; no new TEST table.
- **D3 More gold.** P1: ~40 more unseen/mixed clips to power cost-vs-silence on unseen. P2: only with N fixed
  in advance and one analysis. Adam's rule: no extra work unless necessary. Is it necessary? If yes, how many,
  which analysis, who annotates?
- **D4 Second annotator.** Who (Adam must recruit a person), which ticks, how many clips, what is reported.
  Worth asking Adam for?
- **D5 Gated caption arm on DEV** (P4): re-render caption with the gate on, judge vs pictures, B1/B2 first.
  Worth it? What exactly can it claim?
- **D6 AudioSet-Strong detector report** (P1): report (never select) the shipped detector stack on the 280-clip
  calibration set. Worth it?
- **D7 Synthetic off-screen mixes** as a labelled diagnostic of the gate's false-silence rate. Worth it?
- **D8 P5's count**: how many of the 81 frozen confirmation subjects carry a qualifier beyond the family name
  (bounds where a picture can beat a caption; answers RQ2). Worth it?

## Also answer
- **The one thing that would most raise the thesis's significance claims honestly**, given the corrected
  numbers above. It may be a writing/reporting change, not a build.
- **What I (the assistant) can do tonight with plenty of GPU and zero work for Adam**, in order.

End with your votes as a line: `D1=… D2=… D3=… D4=… D5=… D6=… D7=… D8=…`.
