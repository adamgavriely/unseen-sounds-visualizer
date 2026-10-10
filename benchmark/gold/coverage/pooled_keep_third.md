# All 158 clips: keep-score trained on DEV + TEST bursts, used only to remove current-system pictures

| fold | bar t chosen on training clips | held-out J: current -> filtered | hits / wrong |
|---|---|---|---|
| 1/5 | 0.3333333333333333 | 3.124 -> 3.001 | 16/8 -> 11/3 |
| 2/5 | 0.3333333333333333 | 2.726 -> 2.850 | 12/2 -> 9/1 |
| 3/5 | 0.3333333333333333 | 4.410 -> 4.359 | 12/15 -> 4/9 |
| 4/5 | 0.3333333333333333 | 2.216 -> 2.336 | 7/3 -> 6/3 |
| 5/5 | 0.3333333333333333 | 1.994 -> 2.049 | 12/6 -> 7/2 |

All 158 clips out of sample: J 2.904 -> 2.928; hits/wrong 59/34 -> 37/18; onset cost 2.076 -> 2.430; cost_cov 2.300 -> 2.563
