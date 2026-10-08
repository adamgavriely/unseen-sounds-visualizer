# Step 4: keep-score, DEV (71 clips), clip-grouped 10-fold CV

1196 bursts, 59 good (would be hits).

Variant (b) representation baseline (bursts the frozen system draws, as bursts): 25 hits / 20 wrong, onset cost 2.423, cost_cov 2.588

| model | variant | AUROC (good) | hits @ wrong<=5 | @ <=10 | @ <=15 | at p > 1/3: hits / wrong / onset cost / cost_cov |
|---|---|---|---|---|---|---|
| trees | a reorder-only | 0.888 | 7 (5 w) | 15 (10 w) | 24 (15 w) | 13 / 10 / 2.817 / 3.024 |
| trees | b replace vetoes | 0.888 | 5 (4 w) | 7 (10 w) | 9 (14 w) | 12 / 27 / 3.352 / 3.455 |
| logistic | a reorder-only | 0.882 | 12 (5 w) | 20 (9 w) | 31 (15 w) | 18 / 8 / 2.479 / 2.708 |
| logistic | b replace vetoes | 0.882 | 3 (2 w) | 8 (10 w) | 9 (12 w) | 16 / 34 / 3.324 / 3.388 |

Reference: frozen 29 hits / 15 wrong (onset 2.056, cost_cov 2.509); a/b candidate 32 / 16 (1.915, 2.421).
Clear win = >= 35 hits at <= 16 wrong, or >= 32 hits at <= 13 wrong (any threshold on the CV curve).
Clear wins: NONE -- the keep-score does not beat the a/b candidate by a clear margin.
