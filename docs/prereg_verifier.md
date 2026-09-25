# Picture verifier — declared before it runs (2026-09-25)

Adam: *"clearly a VLM could see most images here are bad; no need for a human to say this."*

**What it is.** Gemma-4-31B is shown one picture at the size the viewer sees (384 px) and told which sound
it is meant to show; it answers yes/no: would a viewer who cannot hear recognise that sound at a glance?
(`benchmark/gold/verifier.py`, prompt frozen there.) This differs from the two blind checkers that failed
(GLM: kappa 0.45; the per-picture judge: 502 of 567 "can't tell"): it answers the *named* question, the one
Adam answered in round 1.

**Calibration set.** Adam's round-1 ratings (2026-09-24; picture + family name shown; "would a viewer
understand this sound?"): 170 yes/no answers on DEV pictures of several arms; the 19 "unsure" are left out.
Never used to design this prompt. The prompt is not changed after the result is seen.

**Bars (the checker's frozen ones):** catches >= 70 % of Adam's "no", rejects <= 15 % of his "yes",
Cohen's kappa >= 0.5.

**If it passes:** it screens the GP-2 generator and prompt arms on picture-DEV in place of Adam's screening
sitting; the screening bar (right >= control + 8/54) is applied to its yes-rate. Adam still rates the final
50-clip confirmation, blind — the thesis claim rests on a human. **If it fails:** Adam screens, as planned;
the verifier is reported as a failed instrument.
