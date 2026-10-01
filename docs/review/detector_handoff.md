# Detector thread handoff (1 Oct 2026, morning)

## Shipped best = SHIP8 (config.use_shipped(), on main)
Chain: BEATs tagger + FlexSED (0.8, band rescue 0.5–0.8) → listener rescue TIER (Qwen3-Omni V4; below peak 0.6 also Audio
Flamingo Next) → ONCE, F8 DASM vote 0.575 → N2b, DR2, K-V4, DV (DASM clip veto 0.084) → BTP (pull late start to FlexSED
run ≥ 0.5) → CONT (drop continuation pieces) → FineLAP veto 0.329 on rescued spans → K4A-D (drop a picture neither
listener names unless DASM ≥ 0.575 hears it) → stage-5 visibility gate (Qwen3.8-27B majority of name/ab/desc).
- Merged DEV (71 clips, 58 needed): **28 hits / 21 wrong (6 vis / 13 cross / 2 phantom) / cost 2.282** (old B0r 18/51/3.690)
- Merged TEST (88 clips, 65 needed): **23 / 29 (4/20/5) / 2.568**, p 0.017 vs B0r 21/40/2.909 (final_test_ship8.md)
- Per-item errors: docs/review/ledger_ship8.md. Supervisor log: docs/review/improvements_log_2026-09-30.md.
- New clips: src/listener_prep.py builds listener answers + DASM + FineLAP (~/venv_flap) on the spot (ComfyUI-ready).

## How to run / score
- DEV old part: from ~/MscProj_r13 `ARMS="SHIP3+DV SHIP8 <arm>" sbatch --export=ALL slurm/job_round16_dev.sh`
- DEV tagger part + merged lines: from ~/MscProj_tg `TG_ARMS="SHIP3+DV SHIP8 <arm>" sbatch --export=ALL --dependency=afterany:<id> slurm/job_tagger_arms.sh`
- TEST read: from ~/MscProj_tg `ARM=<arm> TG_ARMS=<arm> sbatch --export=ALL slurm/job_final_test_prep.sh`, then
  `python benchmark/gold/final_test.py score --arm <arm> --tag <name>` (reported, not selected on)
- Arms: benchmark/gold/round13_dev.py ARMS; new flags need a default in config.py and must be copied to all 3 checkouts
  (~/MscProj, ~/MscProj_r13, ~/MscProj_tg). Screens on saved pictures: benchmark/gold/dbr_screen.py style (roots from
  cross_group.PARTS, TG_ARMS=SHIP8).
- Gold: ONE file benchmark/gold/annotations/gold_AG.json (tagger clips merged, tg_*; subsets_of keeps them out of old dev/test).
  Adam's visibility re-checks applied (round 1: 3/14 flipped; random control 0/20).

## Rules in use
- Cost = (4·miss + 2·wrong)/clips. Hit = same-family picture starting −0.5…+1.0 s from the gold onset. Strict: a second picture
  of a sound still playing counts wrong (Adam).
- Pass vs base (pre-register every arm first in docs/prereg_round13_detector_push.md): old rule (hits ≥ base, wrong ≤ base +
  2×gain, cost < base, no needed hit lost on either DEV part) OR fewer-pictures clause (cost < base, ≥ 3 wrong removed per hit
  lost, ≤ 3 lost). Hits first (Adam).
- Secondary: visible-weighted cost (w = 0/1/2 for on-screen pictures; benchmark/gold/visible_weight_sweep.md). A change that
  wins only at w = 1 AND whose extra wrongs are all visible-type → second "more hits" profile + git tag, not main.
- Every new best: merge to main and TEST-read it.

## Cluster
No jobs running at handoff. Home disk ~30 GB free. Use H200-4h / A100-4h (msproj torch 2.5.1; no Blackwell); small models
also L4-4h / generic-48G.

## Overnight results (all pre-registered, all closed)
| test | DEV hits/wrong/cost | TEST | verdict |
|---|---|---|---|
| EXPECT-A4: Omni whole-clip list → map → FineLAP onset → DASM confirm → gate | 31/26/2.254 | 25/35/2.614 | DEV only |
| EXPECT-A5: A4, families ≥ 0.8 precise on 415 held-out | 30/21/2.169 | 23/31/2.614 | DEV only |
| AGREE-EARS: A4 + Audio Flamingo must agree | 31/24/2.197 | 24/35/2.659 | DEV only |
| HELDOUT-A4: A4 sound chain on 415 held-out, no gate | precision 0.80 (detectors 0.38) | — | finding |
| HUMAN-2 (Adam): is the source acting at the onset? (gate) | 18/41 seen, 32/38 needed (4th vote) | — | 1 short of GO |
| AVNAME: Omni audio+video names each picture | relabel 15/34 | drop-only: 0 change | rejected |
| BOX-2 arm, SYNC, CF, MAKER-VIS, CONTRAST, DETACHED-ADD, GBTP, RELABEL-GATE, NAMED-VETO, PRIOR415, IMP-V, AGREE | — | — | rejected |
Reading: missing sounds can be heard (80 % precise held-out) but additions become cross/visible pictures on DEV/TEST
because the gate can't separate a wrong off-screen sound from a right one, and the gold lists only salient sounds.
DEV-only wins do not transfer (1–2 pictures decide on ~70 clips).

## Open decision for Adam
Shipped MERGE_GAP 1.5 s, but all scores use 2.0 s (merge_gap_sens.py; 1.0 = 1.5 because MIN_DWELL 1.5):
1.5 → DEV 28/23/2.338, TEST 23/30/2.591; **2.0 → 28/21/2.282, 23/29/2.568 (recommended)**; 3.0 → 28/18/2.197, 22/27/2.568.

## Next ideas (not started)
1. HUMAN-2 restricted to off-screen-type families (whack, bell, birds) as a gate override — only if pre-registered from a
   family list fixed outside DEV/TEST.
2. Gold-scope check: Adam re-tags whether EXPECT-A4's extra DEV/TEST pictures are real but unlisted sounds (tool:
   docs/review/visibility_recheck.html pattern) — decides whether "wrong" there is gold incompleteness.
3. "More hits" profile candidates: none qualify yet (EXPECT extras are cross, not visible).
4. Larger selection set: pool DEV + held-out style labels for any detector-only rule before touching TEST.
