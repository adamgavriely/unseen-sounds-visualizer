# Hold from Qwen3-Omni 'still heard', all 158 clips

v1.4 (FlexSED hold): J 2.904, cost_cov 2.300, cover 0.85, |end err| 0.49 s, wrong s/clip 1.04, stale s/clip 0.17
CV choices [None, None, ('omni|flex', 0.9, 1), None, None]
out of fold: J 2.990, cost_cov 2.290, cover 0.86, |end err| 0.58 s, wrong s/clip 1.17, stale s/clip 0.23

| mode, tau, gap | J | cost_cov | cover | wrong s | stale s |
|---|---|---|---|---|---|
| ('omni', 0.5, 1) | 3.192 | 2.270 | 0.87 | 1.39 | 0.46 |
| ('omni', 0.5, 2) | 3.312 | 2.263 | 0.87 | 1.39 | 0.71 |
| ('omni', 0.7, 1) | 3.111 | 2.297 | 0.85 | 1.23 | 0.40 |
| ('omni', 0.7, 2) | 3.208 | 2.272 | 0.87 | 1.25 | 0.62 |
| ('omni', 0.9, 1) | 2.993 | 2.334 | 0.83 | 1.12 | 0.20 |
| ('omni', 0.9, 2) | 3.061 | 2.334 | 0.83 | 1.14 | 0.32 |
| ('omni|flex', 0.5, 1) | 3.207 | 2.234 | 0.89 | 1.44 | 0.51 |
| ('omni|flex', 0.5, 2) | 3.318 | 2.233 | 0.89 | 1.44 | 0.73 |
| ('omni|flex', 0.7, 1) | 3.119 | 2.241 | 0.89 | 1.31 | 0.45 |
| ('omni|flex', 0.7, 2) | 3.223 | 2.241 | 0.89 | 1.33 | 0.64 |
| ('omni|flex', 0.9, 1) | 2.990 | 2.251 | 0.88 | 1.20 | 0.28 |
| ('omni|flex', 0.9, 2) | 3.047 | 2.251 | 0.88 | 1.22 | 0.38 |
