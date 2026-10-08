# Pre-registration: Step 8, relabel a picture when Qwen3-Omni names a close relative instead (DEV only, CPU)

Written and committed 2026-10-08 before it was computed. Base = the a/b candidate (32 / 16). After Step 7 (no win).
Seven of the frozen system's DEV wrong pictures are "a different sound": often a near relative (Vehicle vs Train).
Step 3b (1) already asked Qwen3-Omni, for every candidate burst, to pick {its family, its AudioSet siblings, none}.

Rule: for each a/b-candidate picture, the overlapping burst(s) of the same family (max-probability one); option
probabilities = mean over the two option orders. If the top option is another family (not "none") and
P(top) - P(own) > m, the picture's label becomes that family (start / end unchanged). m in {0.0, 0.2, 0.4}.
Reported: hits / wrong / onset cost per m vs 32 / 16; clear win = >= 35 hits at <= 16 wrong or >= 32 at <= 13.
