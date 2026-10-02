# How to win more clearly over the baselines — simple summary (26 Sept)

Five reviewers, four rounds (`docs/history/panels/panel_2026-09-26_plan.md`, signed 5 of 5).

## The answer
- **F1 against "draw everything" cannot become significant.** It would need about 320–570 clips; we have 60 on
  TEST. No model change fixes that. It is a size problem, not a pipeline problem.
- **We already have significant wins, and they should lead the thesis** (the one TEST look, 23 Sept):
  - precision (share of pictures that are right) **+0.14**, significant;
  - viewer cost (4 × missed + 2 × wrong) **0.73 better**, significant;
  - on the 13 clips where the sound is off screen (the clips the project is for): **3.08 cheaper than showing
    nothing**, significant; there 5 of every 6 pictures are right.
- **The biggest honest step is how we report, not a new build:**
  - F1 first, reported as "no difference", with the reason;
  - then all seven secondary numbers with a correction for testing many numbers (Holm), so nobody can say we
    picked the good ones;
  - then the cost curve for every price of a wrong picture, not one chosen point.
- **One mistake found:** the "caption" baseline never used the gate (it shows text for *every* detected sound).
  So "pictures vs caption" mixed two differences. Fixed on the supervisor page. A fair version (the same gate,
  shown as text) is planned and costs no GPU.

## What the reviewers said NOT to do
- More annotation "until it is significant". That is cheating by stopping when we like the result.
- New gate rules made to recover the 6 lost sounds.
- Lower detector thresholds.
- Another automatic picture judge.
- Any change to your pending rating sitting.

## Waiting for the VPN (no work for you)
The cluster was not reachable, so none of this has run yet:
- the Holm tables;
- the fair text baseline;
- a detector report on 280 AudioSet clips;
- a check of the "a kind of" gate rule;
- the count of pictures more specific than the word.

## Actions for you
1. **Turn the VPN on** — everything above runs by itself after that.
2. **Yes or no: one last like-for-like TEST table?** The reviewers voted 4–1 yes.
   - What it is: all three systems on TEST at the final settings. Today, TEST has "ours" at the new timing but the
     baselines only at the old timing.
   - The rule is already written (amendment 21): the new table replaces the old one whatever it shows. The 4
     earlier TEST readings stay listed.
   - The risk: precision may lose its significance at the new timing.
   - The one reviewer against: "a fifth reading of TEST".
   - My recommendation: **yes**. One table, one system, is what an examiner expects.
3. **One name:** a person (not you) to annotate 30 clips for about an hour, so we can report agreement between
   two people. The page will be ready; you only forward the link.
4. Still pending from yesterday: your one rating sitting, https://claude.ai/artifact/3UV2qKvwA8rkhLudtrgVPf.
