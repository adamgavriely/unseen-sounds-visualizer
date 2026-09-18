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
1. **Time.** A picture can match a gold sound if the picture starts within
   **[sound onset − 1 s, sound onset + 5 s]** *or* the picture overlaps the sound's interval by
   ≥ 0.5 s. The +5 s is our gate's 5-second stride (disclosed, cited from the design); the
   picture's end is ignored (set by the 1.5-s dwell). Sounds that start before the clip use
   the clip start as onset. Sensitivity row: +3 s and +7 s.
2. **Label.** Same AudioSet family (parent or child within two hops) = match. Labels at
   ontology depth < 2 (root-level words such as "sound", "vehicle") **never** match — so
   "vehicle" shown for a car horn is a cross-trigger. Exact-class numbers in a second column.
3. **One-to-one.** Each gold sound takes at most one picture (greedy by earliest picture
   start); one picture may cover several overlapping sounds of the same family (rain,
   traffic). Extra pictures on a sound already matched are **duplicates**: not credited, not
   false alarms, counted in the clutter rate.
4. **"Dog at 45 s" rule.** A picture is never judged by itself: first look for a dog-family
   sound in the gold around 44–50 s. If one exists and is needed → hit (with its lateness
   recorded). If it exists but is obvious → a visible-picture (see the split). If none exists →
   phantom (Set 1) or "unverified" (Set 2, see step 3).

## Step 2 — classes
Per needed gold sound: **hit** / **miss**. Per picture: **hit** / **visible-picture** (a real
sound that was obvious, not needed) / **cross-trigger** (a real sound is there, wrong family;
that sound stays a miss) / **phantom** (no gold sound of any family in the window) /
**duplicate**. Lateness (picture start − onset) is an attribute of a hit, reported as a
median; no decaying credit (unanimous in round 2).

## Step 3 — missing gold labels (Set 2 only)
Adam's annotation starts from detector suggestions, so a quiet real sound may be unlabelled.
A picture with no gold match is **unverified**. Two bounds are reported: **strict** (unverified
= phantom; the headline) and **lenient** (unverified excluded from precision), plus the count.
No detector-score rule: the detector under test may not excuse its own pictures (R7, R8, R10).
If time allows, one logged blind re-listen pass by the annotator turns unverified into gold
or phantom.

## Step 4 — numbers per set
- Pool hits / misses / false alarms over all sounds of a set (**micro**); clips with no needed
  sound contribute false alarms only — they are the test of restraint.
- **Precision** = hits / (hits + false alarms). **Recall** = hits / needed sounds. **F1**.
- **The one split (5 vs 5):** does a visible-picture count as a false alarm in precision?
  Yes: DHH, decision-analysis, adversarial, annotation, examiner ("the task is to show what
  the video does not show; if this is free, the blind system is never charged for its
  defining fault"). No: DCASE, statistics, captioning, CV, IR ("it is a real sound, not a wrong
  picture; report it as its own clutter rate"). **Adopted: report both** —
  `F1-strict` (visible-pictures and duplicates excluded, phantoms + cross + visible-pictures
  as false alarms) and `F1-phantom` (phantoms + cross only) — and a **visible-picture rate**
  column. Which is the headline is Adam's decision (see the note in the thesis).
- **Importance** (1–3) enters once, as a sample weight on every sound and on the picture it
  matches (phantoms weight 1) → weighted P / R / F1 beside the unweighted ones (7 of 10).
- **F0.5 and F2** beside F1: if the ranking of gated / blind / caption holds across β = 0.5,
  1, 2, the equal-cost assumption of F1 does not drive the conclusion.
- **Confidence intervals** by clip-level bootstrap (2000 draws; sounds inside a clip are not
  independent). With ~50 informative clips, differences under ~0.10 in F1 are not reliable.
- **Clean-clip accuracy**: share of clips with no needed sound on which nothing was shown.
- The caption baseline is scored by the same rule; each caption tag = one picture, one label
  per event.

## The thesis table (per set)
System | P | R | F1-strict | F1-phantom | weighted F1 | F0.5 | F2 | visible-picture rate |
phantoms per clip | cross-triggers | median lateness | clean-clip accuracy | unverified (Set 2)

## Anti-gaming (why each rule exists)
Show everything → phantoms and visible-pictures sink precision. Show nothing → recall 0,
precision defined as 0. Show a vague label → depth < 2 never matches. Re-fire every dwell →
duplicates are not credited. Two labels per event → one label per event.

## Worked example
Gold: siren 12–18 s (needed, importance 3); dog 30–32 s (obvious, importance 1). System:
SIREN 13–16, DOG 30–32, HORN 40–42. → SIREN = hit (1 s late); DOG = visible-picture;
HORN = phantom (Set 1) / unverified (Set 2). Recall 1/1. Precision-strict 1/3, F1-strict 0.50.
Precision-phantom 1/2, F1-phantom 0.67. Weighted (siren 3): P-strict 3/5 = 0.60, F1 = 0.75.
