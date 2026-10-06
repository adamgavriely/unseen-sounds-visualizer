# One-model outside baselines (merged DEV 71 clips, TEST 87 clips)

Cost = (4 x miss + 2 x wrong) per clip, report rule (a repeat inside the onset window is not wrong). Bar set once on DEV; TEST scored once. Paired clip bootstrap, 100,000 draws, seed 0. Pictures are not drawn: the score checks only which sound is shown and when, for every row.

| System | DEV cost | TEST hits / needed | TEST wrong | TEST cost | final - row, TEST [95% CI] | p |
|---|---|---|---|---|---|---|
| Show nothing | 3.268 | 0/66 | 0 | 3.034 |  |  |
| Direct audio-to-image | 2.451 | 28/66 | 59 | 3.103 |  |  |
| Final system | 2.056 | 25/66 | 22 | 2.391 |  |  |
| PretrainedSED M2D-strong (bar 0.7) | 3.465 | 5/66 | 13 | 3.103 | -0.713 [-1.287, -0.115] | 0.0225 |
| Qwen3-Omni-30B, audio | 12.366 | 28/66 | 325 | 9.218 | -6.828 [-8.529, -5.264] | 0.0000 |
| Qwen3-Omni-30B, video + audio, off-screen only | 5.099 | 10/66 | 84 | 4.506 | -2.115 [-2.897, -1.379] | 0.0000 |

One model vs show nothing / vs direct audio-to-image (TEST, d [CI] p):
- m2d: vs nothing +0.069 [-0.276, +0.345] p 0.6716; vs a2i +0.000 [-0.690, +0.667] p 1.0000
- qwen: vs nothing +6.184 [+4.713, +7.793] p 0.0000; vs a2i +6.115 [+4.552, +7.793] p 0.0000
- qwen_av: vs nothing +1.471 [+0.943, +2.023] p 0.0000; vs a2i +1.402 [+0.598, +2.276] p 0.0007

M2D DEV threshold sweep (hits of 58 / wrong / cost):

- 0.05: 38 / 1007 / 29.493
- 0.1: 40 / 667 / 19.803
- 0.15: 36 / 518 / 15.831
- 0.2: 31 / 388 / 12.451
- 0.25: 30 / 306 / 10.197
- 0.3: 23 / 245 / 8.873
- 0.35: 18 / 193 / 7.690
- 0.4: 14 / 145 / 6.563
- 0.5: 11 / 71 / 4.648
- 0.6: 9 / 42 / 3.944
- 0.7: 4 / 15 / 3.465

Per-clip rows (cost and counts for every system): `per_clip` in result_m2d.json, result_qwen.json, result_qwen_av.json.
