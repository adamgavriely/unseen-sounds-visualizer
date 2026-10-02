# Scoring panel, 19 September 2026 — five independent Fable reviewers

*Brief given to each: the current scoring only (describer → reference → 0–4 LLM judge; empty
panel on a no-due clip = 4; cost axis) and the gold set being built. They were NOT told our
results or what we changed. Asked for: a better, fairer, ideally continuous score; how to
set its parameters without tuning on the test clips; how to validate it.*

## What all five said (independently)

1. **Score each sound, not each clip sentence.** Inputs per gold sound: when it happens,
   whether the *picture already makes it obvious* ("event visibility", not just "is the
   source in shot" — a phone in a pocket ringing is not obvious; a visible dog barking is),
   whether it should be drawn, and how important it is (1–3, from the "what a deaf viewer
   misses" sentence). Per system: each panel's on/off time and what it shows.
2. **Four cases per needed sound**: hit (right thing, on time) · miss · wrong identity ·
   clutter (a picture where none was needed). Timing is continuous: credit falls with the
   delay (full credit within ~1–2 s, none after ~6 s); a panel after the sound ends is a miss.
3. **Silence when nothing is needed scores 0, not 4.** Showing something there is a pure
   cost. Wrong pictures cost more than nothing ("misleading is worse than nothing").
4. **The prices (how much a clutter picture or a wrong picture costs, relative to one missed
   sound) must not be chosen by the researcher.** Three acceptable ways: (a) estimate them
   from a small pairwise-preference study (viewers see the same clip with two panels and
   pick the better one; fit a Bradley–Terry model) on clips *outside* the 100; (b) fix them by
   convention and show a sensitivity band over the whole range; (c) report a **curve** —
   benefit (needed sounds conveyed) against attention cost (panel-on time) over the system's
   operating points — and its area, which needs no price at all (the PSDS idea from sound
   event detection).
5. **Validation**: human pairwise preferences on ~20 held-out clips (DHH viewers; hearing
   viewers with muted audio as a disclosed proxy), Kendall τ between metric differences and
   human choices; plus sanity checks on synthetic edits (delay a panel, add a redundant panel,
   swap the picture — each must lower the score). Pre-register the metric family, freeze on
   the dev clips, score the 100 test clips once, add a sensitivity appendix.
6. **The LLM judge stays only as a secondary check** (is the picture legible / does it match
   the sound), reported next to its agreement with humans.

## The metric they converge on (one formula, names vary: MIU / NIS / NVB / VTU)

For clip c with needed sounds e (weight w_e = importance × not-obvious):

    benefit_c  = Σ_e w_e · match_e · timing_e   /  Σ_e w_e        (1 = perfect; 0 if nothing needed)
    cost_c     = λ · (panel-on time on sounds that were not needed, or on nothing) / clip length
                 + κ · (wrong pictures)  [+ σ · number of panel switches]
    score_c    = benefit_c − cost_c

Report: (i) the benefit-vs-cost curve and its area (parameter-free); (ii) score at prices
λ, κ from the preference study, with a band over the plausible range; (iii) benefit and
cost separately, so a reader sees what drives a difference.

## What changes for us

- Gold tool: two extra ticks per sound — "obvious from the picture alone?" and importance
  1–3. Everything else is already collected.
- A per-sound scorer on the gold set replaces the LLM judge as the primary number; the
  judge becomes a secondary column.
- A small preference study on ~20 dev clips (not the 100) to set the prices — or the
  curve, which needs none.
- Statistician's power note: only clips with a needed sound inform the benefit term
  (~50 of 100); with n = 50 the smallest detectable difference is ≈ 0.12, with 100 ≈ 0.085.
  Aim for 100 *informative* clips (slice B helps).

## Second round (three reviewers): "per minute?" and how exactly to compute P / R / F1

**Per minute.** A rate divides by clip length. It is only needed when clips differ in length
(our benchmark: 20–60 s) or when two sets are compared (10-s YouTube vs 20–60 s benchmark);
when all clips are the same length it is just a count scaled by a constant. All three:
report the two clip sets separately. Reviewer A: for a thesis reader, "unneeded pictures
per clip" (plain count, averaged within a set) is clearer; keep per-minute in a footnote
for readers from sound-event detection. Reviewer C: per minute is the field's convention
(distraction grows with time watched). Decision: per clip within each set; per minute in a
footnote.

**Precision / recall / F1 — the rule (all three agree):**
- a shown picture **matches** a needed sound if the labels are the same family (ontology)
  and the picture starts within [onset − 0.5 s, onset + 2 s] (or overlaps the sound);
  the picture's end is ignored (set by the display's dwell, not by the sound);
- one-to-one: each needed sound matches at most one picture; a second picture for the same
  sound is a false positive;
- unmatched picture = **false positive** (this includes a picture of a real but *visible*
  sound); unmatched needed sound = **false negative**; a picture of the wrong sound at the
  time of a needed one = one FP **and** one FN (cross-trigger; count them separately too);
- pool TP/FP/FN over all clips of a set (**micro**), because half the clips have no needed
  sound and clips have 0–6 of them; macro in the appendix;
- confidence interval by **clip-level bootstrap** (2000 draws; sounds in one clip are not
  independent); with ~50 informative clips the interval on F1 is about ±0.08, so a
  difference under ~0.10 is not reliably detectable;
- **importance** enters once, as weighted recall (a missed importance-3 sound counts three
  times a missed importance-1 sound), reported beside plain recall; precision unweighted;
- F1 assumes a miss and a false alarm cost the same — show **F0.5 and F2** beside it;
- a clip with nothing needed and nothing shown adds nothing to P/R/F1; report
  **clean-clip accuracy** (share of such clips left blank) so silence is visible;
- the caption baseline is scored by the same rule, each caption tag = one "picture";
- sensitivity: T = 1 s and 3 s beside 2 s.

**The thesis table (per clip set):** System | P | R | F1 (T = 2 s) | F0.5 | F2 | weighted R |
unneeded pictures per clip | cross-triggers | clean-clip accuracy | 95% CI on F1.

Worked example (reviewer C): gold needs HORN at 12.0 s and DOG at 30.0 s; the system shows
HORN at 13.1 (match), CAT at 30.2 (FP, and DOG at 30 is an FN), DOG at 45 (FP: 15 s late).
P = 1/3, R = 1/2, F1 = 0.40.
