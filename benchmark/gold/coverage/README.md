# Scorer v2 (coverage) and the end-hold sweep

`score_coverage.py` = the onset scorer (`../score_per_sound.py`, unchanged: hits, misses, wrong pictures) plus how
long each needed sound (importance 2-3) is covered.

Rules:
- **coverage** of a needed sound = share of its labelled time under a same-family picture whose *start* is in this
  sound's onset window (-0.5 .. +1.0 s). A miss has 0. A picture that stays up through repeats adds coverage, never a
  second hit. One picture may cover several same-family sounds (all of them get its coverage).
- **cost_cov** = (4 x sum over needed sounds (1 - coverage) + 2 x wrong pictures) / clips. Lower is better.
  Equals the onset cost when every hit is fully covered.
- **hit coverage** = mean coverage over hit sounds; **needed-time coverage** = covered seconds / needed seconds.
- **wrong seconds** = full on-screen time of wrong pictures (visible / other sound / no sound), per clip.
  Duplicate and don't-care pictures count in neither wrong seconds nor stale seconds.
- **stale seconds** = time a matched picture stays up more than 1 s after the latest end of the sound(s) it matched,
  per clip.

Baseline display rule = the scoring harness (`MAX_AFTER_END` None), which produced the reported 2.056 / 2.391.
The shipped display (`MAX_AFTER_END` 1.0) is in `frozen_v2_shipped_ends.json` (only TEST cost_cov moves: 2.626).

Inputs: `pics_frozen.json` (pictures of the frozen system, tag detector-frozen-2026-10-02, by `dump_pictures.py` on
the cluster), `../annotations/gold_AG.json`, `../dev_stems.txt` (71), `../test_stems.txt` (87).

    python benchmark/gold/coverage/score_coverage.py              # frozen system, DEV + TEST
