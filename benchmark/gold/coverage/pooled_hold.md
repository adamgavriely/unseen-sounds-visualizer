# All 158 clips: picture ends from the ears' evidence, clip-wise cross-validation

Base = a/b gate + flash + plain texture ban. Starts never moved. J = cost_cov + 0.5 x (wrong + stale seconds) per clip.

| fold | chosen setting (F, B, D, tau, mode) | held-out J: base -> hold | cost_cov | hits / wrong | |end err| median | hit cover |
|---|---|---|---|---|---|---|
| DEV->TEST | (0.5, None, False, 0.0, 'extend') | 3.031 -> 3.086 | 2.515 -> 2.407 | 28/22 -> 28/22 | 1.15 -> 0.50 | 0.80 -> 0.88 |
| TEST->DEV | (0.5, 0.175, False, 0.0, 'both') | 2.727 -> 2.755 | 2.364 -> 2.183 | 31/12 -> 31/12 | 0.65 -> 0.70 | 0.71 -> 0.81 |
| random 1/5 | (0.5, None, False, 0.0, 'extend') | 2.951 -> 3.124 | 2.491 -> 2.398 | 16/8 -> 16/8 | 1.30 -> 0.95 | 0.82 -> 0.86 |
| random 2/5 | None | 2.753 -> 2.753 | 2.675 -> 2.675 | 12/2 -> 12/2 | 0.58 -> 0.58 | 0.72 -> 0.72 |
| random 3/5 | (0.5, 0.175, False, 0.0, 'both') | 4.264 -> 4.555 | 3.404 -> 3.320 | 12/15 -> 12/15 | 0.34 -> 0.18 | 0.77 -> 0.83 |
| random 4/5 | None | 2.238 -> 2.238 | 1.924 -> 1.924 | 7/3 -> 7/3 | 1.30 -> 1.30 | 0.66 -> 0.66 |
| random 5/5 | None | 2.224 -> 2.224 | 1.703 -> 1.703 | 12/6 -> 12/6 | 1.88 -> 1.88 | 0.73 -> 0.73 |

5-fold out-of-sample, all 158 clips: J 2.894 -> 2.988; cost_cov 2.447 -> 2.412; hits/wrong 59/34 -> 59/34; start err median +0.06 s (|0.15|); end err median -0.05 -> +0.00 s (|0.73| -> |0.73|); hit cover 0.75 -> 0.78; wrong s/clip 0.84 -> 1.03; stale s/clip 0.06 -> 0.12

Setting chosen on all 158 clips (for the shipped rule, not a score): None
