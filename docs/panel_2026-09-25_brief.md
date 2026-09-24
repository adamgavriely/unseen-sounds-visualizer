# Panel, 2026-09-25 (night before the supervisor meeting) — round 1 brief

Adam, before leaving for five hours: *"fix everything we said, and consult Fable on everything you can
improve and do it … we will prepare for another supervisor meeting tomorrow."* Earlier today: *"use all
the computation needed to fix all the stuff and rerun tests as needed and consult fables on decisions
until u feel its fixed to present me another working pipeline with significance that also has good
generations."* And on timing: *"time limit is not a thing we should use, it depends on the sound … the
evidence significance should not be dependent on the cut. If so it's manipulation."*

You are one of five reviewers. The repository is at `P:\MscProj` (read anything; change nothing). Run
data lives on the cluster and is not local; the numbers you need are below.

## Where things stand (facts, all measured)

**Headline metric** (unchanged, `benchmark/gold/score_per_sound.py`): a needed sound is hit when a
picture of the same family starts within [−0.5, +1.0] s of its onset; viewer cost = 4 × missed + 2 ×
wrong; paired clip bootstrap, 2000 draws, seed 0. Gold: one annotator, 139 clips (DEV 49, TEST 60,
slice B 30). It never looks at what the picture shows.

**Timing.** `ONSET_MONOTONE` (a later stage may never move a start earlier than its anchor —
`docs/onset_timing.md`) on DEV 49: F1 0.293 → 0.395, P +0.093, R +0.111, cost −0.45, all CIs exclude
0, 4 recovered / 0 lost. 3 of the 4 recoveries are the cases the bug was found on. With the 8-s
picture cap also removed (Adam's decision, final — `docs/test_second_look.md`): F1 0.378, 4 recovered,
1 lost (a church bell, lost because the longer span makes the visibility vote sample frames where the
bell tower is on screen), not significant on DEV. Ends: median end error on drawn specs −1.70 s →
−0.06 s; on the rendered panel median displayed end −2.65 s → −0.25 s, 1 of 16 lingering > 2 s.
A switches-off reproduction run matches the base exactly (23/23 starts, every metric), twice.
**TEST 60 (the approved second look, onset rule on + cap off) is finishing now**; the decision rule is
frozen in `docs/test_second_look.md`.

**Earlier gate result (2026-09-22, old timing):** vs SILENCE F1 +0.290 [+0.199, +0.377]; vs BLIND
(draw every detected sound, no gate) ΔF1 +0.028 [−0.022, +0.074] null, ΔP +0.062 significant, ΔR
−0.076 significant, FA/clip −0.55 significant. The blind and caption arms were rendered with the
**old timing** (no onset rule, 8-s cap). Our rule: a shared stage is never tuned by our F1.

**Pictures.** Adam's blind round 1 (family name shown): shipped 12/31 "yes", new instructions +
Qwen-Image 18/32; FLUX ≈30 % vs Qwen-Image ≈48 % pooled over seeds. His complaint: pictures show the
family tag, not the specific sound (train → "vehicle", ambulance → generic siren), and ignore the
scene (a car door heard in a car drawn as a house door; a rooster drawn as a generic bird).
`PICTURE_V3` (signed by a five-reviewer panel over three rounds, `docs/picture_v3_prereg.md`):
specific source from the detector's own sub-labels (`labels.choose_source`), a depiction prompt for
the whole source caught making this sound with no place, Qwen-Image-2512 with a negative prompt.
A round-2 blind rating (no name shown, Adam types what he sees) on **50 never-annotated clips**, three
arms (today / N = V3 / N0 = today's subjects on Qwen-Image), is being drawn now. A checker
(GLM-4.6V-Flash) logs what it sees; it may not act until calibrated against Adam. The car-door case
cannot be fixed by V3 (AudioSet has no car-door class; only the scene can tell) — the scene step
("RESOLVE": frames choose between the audio's candidates, may add one qualifier where the place
determines the kind, never add a noun the detector did not hear) was designed and deferred.

**Judge.** Describe (a VLM writes what the rendered panel tells a viewer) → judge (an LLM compares it
to a reference built from the annotator's ticks, 1–5) → rubric cap. Mistral-7B failed the trust check
(cannot separate clips carrying a known-wrong picture, even on 139). Gemma-4-31B passes B1 (tracks the
annotator's cost) and B2 (separates wrong-picture clips). Gemma on the old system: caption-only
control beats us (−0.41 [−0.66, −0.16]); on the current DEV system: tie (+0.08). **The describer is
Qwen3.8-27B — the same weights that write the picture subject** (circular).

**Compute.** BIU Slurm: H200 / A100-80 / L40s, 4-h jobs, no internet on compute nodes. Qwen-Image
≈20 s/picture on H200. Adam: "use all the computation needed".

## Questions — answer each in at most six lines, with a recommendation

1. **Ends.** With the cap gone, is any further end work needed? If yes, what principled rule that
   follows the sound (not a number of seconds)? The bell: is it an end problem at all, or a gate
   problem (the visibility vote samples frames across the whole span)? Should we touch it now, and if
   so, how, without it being a fix tuned to the one sound we lost?
2. **Fair baselines.** The onset rule and the cap are *shared* stages. Should the BLIND and CAPTION
   arms be re-rendered with them on, so "gate vs blind" and "pictures vs caption" compare like with
   like? On DEV only, or on TEST too — and is a TEST render of a *baseline* arm a further look at TEST
   that needs Adam's yes? What is honest to claim tomorrow if it is not done?
3. **Judge describer.** Replace Qwen3.8 as the describer with what — GLM-4.6V-Flash (the checker's
   family), Gemma-4-31B (the judge's own family, multimodal), or something else? Does the trust check
   (B1/B2) need to be rerun, and on how many clips?
4. **Pictures beyond this round.** What should be built *now*, while Adam rates, so the next round is
   ready: the scene RESOLVE step, per-burst sources, best-of-N seeds with the checker, a stronger
   generator (name one, with licence and VRAM), or nothing? What is the smallest thing that fixes the
   car door and the rooster honestly?
5. **The supervisor meeting.** Given all the above, what are the two or three claims we can make
   tomorrow with a straight face, and which numbers must be labelled "DEV, selected" or "one rater"?
6. **Anything else** broken or weak that you would fix with a lot of GPU time before the thesis is
   written. One item, the most important.
