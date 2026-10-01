# Panel 2 — Fable A (audio), 1 Oct

Audio-reachable D misses = B 16 + C non-gate 4 + D 2; gate kills and the A bucket (Clang, golf whacks, tg_d125 Explosion) are out of reach for audio changes. All ideas online per video; constants fixed on the 415 only.

**1. TAG-ENS (first pick).** Calibrated frame-level ensemble of AudioSet taggers (EAT-large, SSLAM, CED if present) on BEATs' 2-s / 0.25-s windows; per-label quantile matching onto BEATs' scale on the 415 frames (label-free); span source = mean of calibrated scores; downstream unchanged (vetoes read raw BEATs). New: the only earlier ensemble (09-23 AST/CED) was discarded as a calibration artefact. Targets: nyc Vehicle 3.8, as_explosion Explosion 2.8, tg_d032 Thunder 7.4, rainforest_7629 Bird, Gasp; wrongs from lone-model spikes (keyboard, flea-market Vehicle, storm Thunder). Expected DEV +1..2 hits, −1..3 wrong (honest chance of zero). Selection: 415 half A calibrate / half B read: span count within ±10 % of BEATs' 754, precision >= 0.42, onset recall >= BEATs'; then DEV main rule vs D. Per video: 2–3 extra tagger passes (seconds). Risk: vetoes tuned to BEATs' profile.

**2. RHYTHM.** Label-free periodic-impact witness (multi-band onset envelope autocorrelation, 1–6 Hz, >= 3 cycles; speech/music masked) + an impulsive family named by the Omni whole-clip list -> rescued-origin span. The only route to nyc Hammer 8.1 (unheard by every detector). Targets: Hammer 8.1/13.7, Footsteps 2.1, Clapping 8.5. 415 bar precision >= 0.50. Per video: CPU < 1 s + whole-clip lists (~1 min GPU). Risk: walking scenes.

**3. JOINT.** 4-constant logistic score over BEATs/FlexSED/DASM logits for sub-bar twins, fitted on 415 half A, admitted only as the family's FIRST occurrence in the clip. Targets: Vehicle 3.8, Explosion 2.8, rainforest_2179 Bird, Footsteps, Whistle. Risk: BANDLIST déjà vu.

Checked, not new: separation (5 negatives), extra ears (Step-Audio, Kimi, MOSS, CLAP), per-family bars, repetition bars, TTA views, local contrast, onset cells.
