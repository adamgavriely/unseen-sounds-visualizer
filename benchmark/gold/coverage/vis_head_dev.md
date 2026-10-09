# Step 12 v1: trained visibility head (DEV)

Training: AVATAR 2645 frames (1983 on-screen, 662 off-screen). 5-fold CV AUROC: logistic 0.776, mlp 0.781; chosen mlp; cut bar s (90% precision on CV) 0.810.

Bar A (DEV gate records with a visibility label, 24 visible / 35 not):

| score | AUROC |
|---|---|
| trained head (mean over seconds) | 0.630 |
| current gate (share of stretches seen) | 0.780 |
| a/b rule | 0.789 |

Bar A (>= 0.85): FAIL -> bar B not run; version 2 (Synchformer, CAV-MAE) is the declared next step.
