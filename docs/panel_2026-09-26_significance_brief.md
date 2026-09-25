# Panel 2026-09-26 — how to improve the pipeline and win more clearly over the baselines (round 1 brief)

Adam: *"consult 5 fables 4 rounds how can we improve the pipeline and be more significant improvement over
baselines."* Standing wishes: *"when possible dont make more work for me, only what necessary"*; significance
must never depend on a cut we chose after seeing numbers (*"If so, it's manipulation"*).

You are one of five reviewers. Repository `P:\MscProj` (read anything, change nothing). Run data lives on
the BIU cluster; the numbers you need are below or in the cited docs.

## The system (one line per stage)
Clip (10–20 s video) → audio detectors (BEATs ∪ FlexSED at 0.8, FlexSED veto 0.3, PANNs veto 0.05) → label
filter (depictable families) → onset rule (`ONSET_MONOTONE`), no span cap → **gate**: Qwen3.8-27B looks at 6
frames per 5 s and votes (majority of 3 questions) whether the source is visible; visible → no picture →
picture subject (Qwen3.8) → Qwen-Image-2512 picture in a side panel. Baselines: **SILENCE** (no panel),
**BLIND** (draw every detected sound, no gate), **CAPTION** (text tag instead of a picture, gated).

## The measured state (all numbers real)
**Primary metric** (`benchmark/gold/score_per_sound.py`, declared before gold existed): a needed sound
(annotator importance 2–3) is hit when a picture of its family starts within [−0.5, +1.0] s of its onset;
per-sound P/R/F1; viewer cost = 4 × miss + 2 × wrong picture; paired clip bootstrap 2000, seed 0.
Gold: one annotator (Adam), 139 clips = DEV 49 (33 needed sounds), TEST 60, slice B 30.

**DEV 49, current system, gate − blind, same timing** (`docs/NIGHT_REPORT_2026-09-25.md` §5):
F1 0.38 vs 0.33, ΔF1 +0.048 [−0.054, +0.131] **null**; ΔP +0.115 [+0.020, +0.204] sig; ΔFA/clip −0.53
[−0.78, −0.31] sig; Δcost −0.82 [−1.39, −0.20] sig; clean-clip accuracy +0.23 [+0.07, +0.39] sig.
**vs SILENCE**: F1 +0.38 [+0.24, +0.50] sig; viewer cost −0.16 [−0.74, +0.41] **null**.
**TEST 60** (read twice deliberately; further looks need Adam's yes): the old-timing gate vs silence F1 +0.29
[+0.20, +0.38], vs blind ΔF1 +0.028 null, ΔP +0.062 sig, ΔR −0.076 sig (gate loses 6 needed sounds), FA −0.55
sig (`docs/GOLD_RERUN_2026-09-22.md`). Current TEST F1 ≈ 0.38–0.40. Blind/caption at new timing not rendered on TEST.

**Why F1 is null vs blind** (anatomy, TEST + DEV): the gate removes 2/3 of on-screen pictures (31 → 10) but
silences 6 needed sounds (a bird called "visible" because other birds are on screen; a visible bell while
another rings; "a tank is visible" → helicopter silenced). 70 of 80 remaining false alarms are **detector
label errors both systems share** ("Cooking" on a cow farm, "Owl" in a pet shop) — no gate can remove them.
Oracle-label diagnostic: with a perfect vocabulary ΔP triples but ΔF1 stays +0.03 → the null is the
equal-weight F1 (each removed false alarm is worth less than each lost hit). Most misses: the detector never
fired (hammer, clang, whistle, honk, footsteps, door). Detector sensitivity table (§4c of the gold doc):
PSED has the best recall but a cross-fit showed no out-of-fold gain → declared detector kept.

**Judge** (Gemma-4-31B sees the pictures; B1/B2/B3 trust checks pass): current DEV ours 2.29 vs blind +0.20
null, vs caption +0.06 null; on clips whose source is **seen** +1.00 sig vs blind; on **unseen** (off-screen —
the clips the project exists for) −0.50 null (pictures weakest there).

**Pictures (human, blind, Adam)**: round 2 on 50 fresh clips — today 14/54 understood, new generator 26/54
(+0.22 sig), generator + V3 text 32/54 (+0.33 sig, but V3 vetoed for one invented object). A frozen final
setup (Qwen-Image-2512 + guarded scene-kind prompt + templates + word cards) awaits Adam's one confirmation
sitting (168 cards, 50 frozen clips; `docs/freeze_picture_setup_2026-09-25.md`). Three automatic picture
checkers failed calibration against Adam.

**Timing**: onset rule on DEV F1 +0.10 sig (selected, 3/4 recoveries in-sample); TEST second look
inconclusive (`docs/test_second_look.md`). No span cap (Adam's principle); 4/15 TEST pictures linger > 2 s.

**Rules we keep** (break one only with a stated reason): a shared stage (detector, timing, display bar) is
never chosen by ΔF1 — only by the BLIND system's own DEV F1; primary metric not changed after seeing numbers
(F0.5 exists as a declared secondary, not promoted); TEST reads only with Adam's yes, decision rule committed
first; every arm copies the base env and runs on the base's card class (H200/A100); Adam's time is scarce —
a new annotation or rating task must be clearly worth it. Compute is plentiful (BIU H200/A100, 4-h jobs,
no internet on compute nodes). Thesis must be written soon (days, not weeks).

## Questions — answer each in at most eight lines, with one concrete recommendation

1. **The biggest honest lever.** Where is the headroom: detector recall/label errors (shared), the gate's
   6 lost sounds, timing, pictures, or the test's power (49/60 clips, 33 needed sounds on DEV)? Name the ONE
   change you would build first and the expected size of its effect on which comparison.
2. **Gate improvements that are treatment, not shared tuning.** How to stop silencing an off-screen sound
   because a same-family thing is on screen (birds, bells, vehicles)? E.g. ask "is the thing making *this*
   sound visible *and* sounding", audio-visual sync, localisation, per-instance counting. What is principled
   and testable on DEV without fitting to the 6 known losses?
3. **Power.** Can more evidence be obtained without Adam annotating much? E.g. AudioSet-Strong / other
   human-labelled SED sets with frame-level labels as extra gold for the shared stages; synthetic off-screen
   clips (mix a known sound under a video without its source — ground truth by construction); a second
   annotator; re-using slice B. What is honest, and what would a thesis examiner accept?
4. **Metrics.** Is there a pre-declared or principled way to report the gate's real value (cost, precision,
   clean-clip accuracy are significant; F1 is not) that is not HARKing? Is a user study (deaf/HoH viewers or
   hearing viewers with sound off) feasible in days, and would it beat the per-sound metric as evidence?
5. **Pictures vs caption.** The judge ties pictures with a caption. What would make pictures beat captions
   measurably (e.g. the frozen setup, combining picture + short word, off-screen-specific depiction), and how
   to test it without another heavy Adam sitting?
6. **What NOT to do.** One thing that would look like progress but would weaken the thesis.

Write your answers under a heading with your reviewer letter. Be concrete: file names, flags, the exact test
and its decision rule. Cite code you read.
