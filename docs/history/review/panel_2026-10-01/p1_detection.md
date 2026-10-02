# Panelist 1 — detection recall (round 1)

**Ceiling first.** Of the 15 B misses, 7 are ear refusals on the cut (Vehicle 3.8 Qwen "Door closing"; Footsteps/Gasp named as "Explosion/Gunshot"; Whistle "Train"; tg_d033 Siren; Clapping; Laughter now MD3), 2 sub-LO (Bird 0.435, Siren 0.42), 1 mistimed run (Thunder 2.8), 1 rooster = Round 52, 1 Dishes (3 vetoes), 1 unheard (Hammer 8.1). Only 3 are reachable by a general stage-4 change: Hammer 13.7 and Explosion 2.8 (TIER yes by both ears, killed by F8: DASM 0.281 / 0.516 < 0.575) and Gasp 6.7 (min span, then N2 + DV). Each proposal below is honestly +1.

**P1 — Agreement-first short spans (twin before min span; twins exempt from the DASM clip veto).** `_extract_events` applies min span per detector before the twin union (`fuse_flexsed` l.650, 671–701), so two short agreeing detections die separately: Gasp = BEATs 6.50–6.75 (0.415) + FlexSED 6.72–6.92 (0.758) -> union 0.42 s >= 0.3. Change: extract both at min span 0, union twins (1-s tolerance as now), apply 0.3 s to the union; singles unchanged. Mirror and N2 already skip twins; DV does not (Gasp DASM clip max 0.065 < 0.084). Differs from R13-5 (cross-detector agreement satisfies duration). 415 bar: twin-only short spans precision >= 0.375 (Round 42 rule), plus twinned spans with DASM clip max < 0.084 >= 0.375. DEV vs SHIP8+MD3 main rule. Risk: Gunshot-type crosses. ~30 lines; 415 CPU 10 min; DEV rebuild + gate ~30 GPU-min.

**P2 — F8 hybrid: keep a rescued span if DASM >= 0.575 OR (local >= 0.25 AND z >= ~20 vs the family's own clip background, MAD-scaled).** Hammer 0.281 vs clip median 0.001 -> recovered; Explosion not. Re-admits ~6 non-needed before downstream filters. 415 bar on the added zone: precision >= 0.70 (0.59 without z). Weakness: near-degenerate on 10-s clips.

**P3 — Tight listener cut for refused runs.** Ear hears [cut − 1, cut + 1]; Footsteps' cut contains the Explosion at 2.84; Gasp's contains the Fusillade. Re-ask refused runs on run − 0.25 … run end (min 0.6 s). Rides on P2/FLAP for the DASM vote. GPU 1–2 h.

**Checked and dead:** per-family DASM bars; CONT variants (Train unreachable); global F8 lowering = K1; N4 rank.

**Run first: P1.**
