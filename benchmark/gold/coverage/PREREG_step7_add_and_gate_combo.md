# Pre-registration: Step 7, two cheap new routes on saved answers (DEV only, CPU)

Written and committed 2026-10-08 before either was computed. Adam (8 Oct): "keep trying, maybe other approaches".
Base = the a/b candidate (AB-m, 32 hits / 16 wrong, onset cost 1.915). TEST untouched; adopt nothing.

## C. Add-only rescue (keep the rule chain, add back what it dropped)

Step 4 tried "remove from the chain's output" and "replace the chain". Not tried: keep every AB-m picture and ADD the
bursts the chain dropped whose keep-score is high. Score = Step 4's out-of-fold probability (same 24 features, same
clip-grouped 10-fold CV; both models). A dropped burst (no AB-m picture of the same family overlapping it) becomes a
picture (family label, burst start, burst end) when its probability > t. Curve over t; reported: hits / wrong at the t
giving the most hits with wrong <= 16, <= 18, <= 20, and the clear-win test (>= 35 hits at <= 16 wrong, or >= 32 at <= 13).

## G. Two-opinion gate (a/b answer + Qwen3-Omni onset look)

For each gate record (sound the gate judged), "seen" score = mean of (share of stretches a/b says seen, split ->
majority) and the Omni onset-gate P(yes) of the overlapping burst (Step 3b (2); max if several; missing -> the a/b
share alone). The sound is silenced iff score >= s; otherwise drawn as its gate-free spec (the Step 2 replay, both
directions). s in {0.4, 0.5, 0.6, 0.7}. Reported: hits / wrong / onset cost per s, against AB-m. Clear win as above.
