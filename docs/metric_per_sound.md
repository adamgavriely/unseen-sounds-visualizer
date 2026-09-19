# Per-sound evaluation — the rule after a ten-reviewer debate (19 Sept 2026)

*Ten independent Fable reviewers (DHH research, DCASE metrics, statistics, captioning, CV
detection, decision analysis, adversarial, annotation quality, thesis examiner, IR) answered
the same brief without seeing our results, then debated eight disagreements. Below: what all
ten agree on, the one split, and the rule we adopt. The brief and both rounds are kept in the
session scratchpad; positions are summarised in docs/scoring_panel_2026-09-19.md.*

## Unit and data
The unit is **one sound in one clip**. Two gold sets, reported separately:
Set 1 = 111 AudioSet-Strong clips (every sound human-timed; Adam adds needed / obvious /
importance); Set 2 = the 100 benchmark clips (Adam's per-sound annotation). A system's output
per clip = panel events (start, end, depicted label).

## Step 1 — match each picture to the gold (time first, label second)
1. **Time (Adam's rule, 19 Sept).** A picture matches a gold sound only if it starts within
   **[sound onset − 0.5 s, sound onset + 1.0 s]**. The panel argued for the gate's 5-s stride;
   Adam overruled: a glass shatter or an explosion shown 3 s late has lost its effect — 0.5 s
   is the aim, 1 s is forgivable, more is not good enough. Overlap with a long sound does not
   count. The picture's end is ignored. Sounds that start before the clip use the clip start
   as onset. Sensitivity rows: +0.5 s, +2 s, +5 s (the last shows what the gate's design
   costs). Where lateness comes from in our pipeline: the detector's onset (BEATs 0.35 s mean
   error on DCASE, PretrainedSED 0.19 s) and, for a sound that spans two 5-s stretches, a
   first stretch judged "visible" delays the picture to the second — such cases count as
   misses under this rule, which is the right pressure on the design.
2. **Label.** Same AudioSet family (parent or child within two hops) = match. Labels at
   ontology depth < 2 (root-level words such as "sound", "vehicle") **never** match — so
   "vehicle" shown for a car horn is a cross-trigger. Exact-class numbers in a second column.
3. **One-to-one.** Each gold sound takes at most one picture (the nearest in time); one
   picture may cover several overlapping sounds of the **same** family only (two barks, not
   rain and traffic). Extra pictures on a sound already matched are **duplicates**: not
   credited, not false alarms, counted in the clutter rate.
4. **"Dog at 45 s" rule.** A picture is never judged by itself: first look for a dog-family
   sound in the gold around 44–50 s. If one exists and is needed → hit (with its lateness
   recorded). If it exists but is obvious → a visible-picture. If none exists → phantom.

## Step 2 — classes
Per needed gold sound: **hit** / **miss**. Per picture: **hit** / **visible-picture** (a real
sound that was obvious, not needed) / **cross-trigger** (a real sound is there, wrong family;
that sound stays a miss) / **phantom** (no gold sound of any family in the window) /
**duplicate**. Lateness (picture start − onset) is an attribute of a hit, reported as a
median; no decaying credit (unanimous in round 2).

## Step 3 — missing gold labels
Adam's decision: verified results only. The annotation must therefore be exhaustive (every
sound the annotator hears is added, including quiet ones); a picture with no gold match is
a **phantom**, full stop. No lenient bound, no detector-score rule (the detector under test
may not excuse its own pictures).

## Step 4 — numbers per set
- Pool hits / misses / false alarms over all sounds of a set (**micro**); clips with no needed
  sound contribute false alarms only — they are the test of restraint.
- **Precision** = hits / (hits + false alarms). **Recall** = hits / needed sounds. **F1**.
- **The one split (5 vs 5), decided by Adam:** a picture of a visible sound **is a false
  alarm**. Headline = `F1-strict` (false alarms = phantoms + cross-triggers + visible-pictures).
  `F1-phantom` (phantoms + cross only) is reported beside it for comparison, with the
  visible-picture rate. The panel's split is recorded in the thesis.
- **Importance** (1–3) enters once, as a weight: a needed sound of importance 3 counts as
  three sounds, importance 1 as one (its picture carries the same weight; phantoms weigh 1)
  → weighted P / R / F1 beside the unweighted ones. Plainly: missing a siren costs three
  times missing background traffic.
- **F0.5 and F2** beside F1: if the ranking of gated / blind / caption holds across β = 0.5,
  1, 2, the equal-cost assumption of F1 does not drive the conclusion.
- **Confidence intervals** by clip-level bootstrap (2000 draws; sounds inside a clip are not
  independent). With ~50 informative clips, differences under ~0.10 in F1 are not reliable.
- **Clean-clip accuracy**: share of clips with no needed sound on which nothing was shown.
- The caption baseline is scored by the same rule; each caption tag = one picture, one label
  per event.

## The thesis table (per set)
System | P | R | F1-strict (with CI) | F1-phantom | weighted F1 | F0.5 | F2 | visible pictures |
cross-triggers | phantoms | duplicates | median lateness | clean-clip accuracy

## Anti-gaming (why each rule exists)
Show everything → phantoms and visible-pictures sink precision. Show nothing → recall 0,
precision defined as 0. Show a vague label → depth < 2 never matches. Re-fire every dwell →
duplicates are not credited. Two labels per event → one label per event.

## Worked example
Gold: siren 12–18 s (needed, importance 3); dog 30–32 s (obvious, importance 1). System:
SIREN 13–16, DOG 30–32, HORN 40–42. → SIREN = hit (1 s late); DOG = visible-picture;
HORN = phantom. Recall 1/1. Precision-strict 1/3, F1-strict 0.50.
Precision-phantom 1/2, F1-phantom 0.67. Weighted (siren 3): P-strict 3/5 = 0.60, F1 = 0.75.

## Gold set 2 run declared (2026-09-20 08:00, before any slice-B annotation is received)

Before receiving any slice-B annotations (obvious / importance), the frozen v4ab pipeline
(commit of this note; PSED at 0.15 chosen on the calibration set, not on slice B; Qwen3.8-27B
gate; FLUX; systems proposed / blind / audio-caption; render only, no judge, no parameter
changes; `slurm/job_sliceb_render.sh`, `--clip-dir data/input/audioset_strong`, tag
`v4ab_sliceB`) is run once on all 111 slice-B clips. Per-sound results on slice B are
reported separately from the 100-clip benchmark, noting that slice B was earlier used to
*evaluate* the detector (never to select its bar).

**Addendum (2026-09-20 09:10):** the detector-arm rule (docs/prereg_v4.md) kept BEATs (v4ab
gated 2.60 < v4b 2.85, CI below 0), so the adopted v4 configuration is v4b. The v4b pipeline
(BEATs + Qwen3.8-27B; FLUX; same three systems; render only) is therefore rendered once on
slice B as well (`slurm/job_sliceb_render_v4b.sh`, tag `v4b_sliceB`), queued before any
slice-B annotation is received. Both rows are scored per sound on both gold sets; the
v4ab-vs-v4b per-sound comparison is the second, pre-declared view of the detector question.
