# Step 14: AudioSep verifier (DEV)

1196 bursts, 209 real.

| score | AUROC real vs not |
|---|---|
| (1) stem energy fraction in span | 0.609 |
| (2) stem in span vs outside | 0.579 |
| (3) CLAP(stem) - CLAP(residual) | 0.648 |
| logistic (1)-(3), clip-grouped 10-fold | 0.677 |

Bar 1 (>= 0.80): FAIL
