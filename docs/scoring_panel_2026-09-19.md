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
