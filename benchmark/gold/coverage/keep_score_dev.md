# Step 4: keep-score, DEV (71 clips), clip-grouped 10-fold CV

1196 bursts, 59 good (would be hits).

Variant (b) representation baseline (bursts the frozen system draws, as bursts): 25 hits / 20 wrong, onset cost 2.423, cost_cov 2.588

| model | variant | AUROC (good) | hits @ wrong<=5 | @ <=10 | @ <=15 | at p > 1/3: hits / wrong / onset cost / cost_cov |
|---|---|---|---|---|---|---|
| trees | a reorder-only | 0.886 | 7 (5 w) | 18 (10 w) | 30 (15 w) | 16 / 10 / 2.648 / 2.834 |
| trees | b replace vetoes | 0.886 | 1 (4 w) | 7 (9 w) | 10 (15 w) | 14 / 24 / 3.155 / 3.261 |
| logistic | a reorder-only | 0.863 | 13 (5 w) | 21 (10 w) | 31 (15 w) | 16 / 9 / 2.620 / 2.841 |
| logistic | b replace vetoes | 0.863 | 8 (5 w) | 9 (6 w) | 11 (13 w) | 15 / 29 / 3.239 / 3.325 |

Reference: frozen 29 hits / 15 wrong (onset 2.056, cost_cov 2.509); a/b candidate 32 / 16 (1.915, 2.421).
Clear win = >= 35 hits at <= 16 wrong, or >= 32 hits at <= 13 wrong (any threshold on the CV curve).
Clear wins: NONE -- the keep-score does not beat the a/b candidate by a clear margin.
