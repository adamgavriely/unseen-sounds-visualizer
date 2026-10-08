# Step 4: keep-score, DEV (71 clips), clip-grouped 10-fold CV

1196 bursts, 59 good (would be hits).

Variant (b) representation baseline (bursts the frozen system draws, as bursts): 25 hits / 20 wrong, onset cost 2.423, cost_cov 2.588

| model | variant | AUROC (good) | hits @ wrong<=5 | @ <=10 | @ <=15 | at p > 1/3: hits / wrong / onset cost / cost_cov |
|---|---|---|---|---|---|---|
| trees | a reorder-only | 0.886 | 8 (4 w) | 18 (10 w) | 27 (15 w) | 14 / 9 / 2.732 / 2.945 |
| trees | b replace vetoes | 0.886 | 4 (5 w) | 9 (10 w) | 10 (15 w) | 13 / 19 / 3.070 / 3.170 |
| logistic | a reorder-only | 0.870 | 14 (5 w) | 22 (10 w) | 30 (15 w) | 17 / 8 / 2.535 / 2.764 |
| logistic | b replace vetoes | 0.870 | 6 (4 w) | 9 (8 w) | 11 (14 w) | 15 / 31 / 3.296 / 3.381 |

Reference: frozen 29 hits / 15 wrong (onset 2.056, cost_cov 2.509); a/b candidate 32 / 16 (1.915, 2.421).
Clear win = >= 35 hits at <= 16 wrong, or >= 32 hits at <= 13 wrong (any threshold on the CV curve).
Clear wins: NONE -- the keep-score does not beat the a/b candidate by a clear margin.
