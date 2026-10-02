# Topic 3 — How to evaluate a task nobody has evaluated before (3 reviewers × 3 rounds)

Read `docs/history/panels/panel3_common_context.md` first. Adam: *"this kind of work hasn't been done before and we need to think how to
evaluate it properly."*

## What the system must get right (and what can go wrong)
A useful picture needs all of: the right sound (detection), shown only when its source is not visible (gate), at the right
moment (timing), understood at a glance (picture). Errors: a missed needed sound; a wrong picture (sound not there, wrong
kind, or source already visible — redundant); a late/early picture; an unreadable or misleading picture.

## What exists now
- **Per-sound gold metric** (`docs/history/plans/metric_per_sound.md`, `benchmark/gold/score_per_sound.py`): a needed sound is a hit if a
  same-family picture starts within [−0.5, +1.0] s of its onset; P/R/F1 per sound; **viewer cost** = 4 × misses + 2 × wrong
  pictures per clip (post hoc, weights from a September rubric); β operating curve (`cost_curve.py`); Holm-corrected
  secondary family; paired clip bootstrap. It never looks at what the picture shows.
- **Gold:** one annotator (Adam), 139 clips; intra-rater κ 0.60 on an older clip-level label; second annotator (30 clips,
  packet ready) to be discussed with the supervisor.
- **Picture recognition:** blind glance test by Adam (1.5 s, 384 px, typed answer, answer sheet committed first).
- **Automatic judge:** Gemma-4-31B looks at the panel and scores it 1–5 against a reference built from the gold ticks;
  passes three trust checks (tracks the annotator's cost ρ −0.72; separates clips with a known wrong picture; stable);
  cannot separate pictures from text tags (it reads a text tag as the label itself).
- **Failed instruments (all recorded):** model-made references (circular); independent 4-model reference (chance on
  silence); "with vs without sound" audio-visual reference (49.5 %); gap-closing score (AUROC 0.53); six automatic picture
  checkers.
- **A viewer study** was designed (`docs/history/plans/beta_specification.md`) but not run: no DHH participants available; hearing
  viewers with sound off would be a proxy.
- **Integrity issues to manage:** TEST was exposed 10 times (5 deliberate); the primary (F1) is null by construction (MDE ≈
  0.13 at n = 60; ~320–570 clips needed); viewer cost was declared after F1 came out null; detector bars chosen on a split
  overlapping TEST.

## Questions (round 1 — ≤ 10 lines each, concrete)
1. What is the **right evaluation framework** for this new task — which components (detection, gate, timing, picture,
   end-to-end usefulness), which metrics for each, and which one is the headline? Cite analogous fields (sound event
   detection PSDS, caption quality, accessibility/DHH sound visualisation studies, information-retrieval style costs).
2. What **human evaluation** is feasible in 6 days and would convince an examiner (who, how many, what task, what
   measure — e.g. hearing participants with sound off answering comprehension questions; a small DHH pilot)? Power and
   design.
3. How should the thesis **present** what already exists (null F1, significant precision/cost, one annotator, TEST
   exposures) so it is honest and strong? What must be added or relabelled?
4. Can the benchmark and metric themselves be a **contribution** (a released gold set + protocol for "off-screen sound
   visualisation")? What would make it reusable?
5. The one thing not to do.
