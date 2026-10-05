# One-model outside baselines (merged DEV 71 clips, TEST 87 clips)

Cost = (4 x miss + 2 x wrong) per clip, report rule (a repeat inside the onset window is not wrong). Bar set once on DEV; TEST scored once. Paired clip bootstrap, 100,000 draws, seed 0. Pictures are not drawn: the score checks only which sound is shown and when, for every row.

| System | DEV cost | TEST hits / needed | TEST wrong | TEST cost | final - row, TEST [95% CI] | p |
|---|---|---|---|---|---|---|
| Show nothing | 3.268 | 0/65 | 0 | 2.989 |  |  |
| Direct audio-to-image | 2.451 | 26/65 | 61 | 3.195 |  |  |
| Final system | 2.056 | 24/65 | 23 | 2.414 |  |  |
| PretrainedSED M2D-strong (bar 0.7) | 3.465 | 4/65 | 14 | 3.126 | -0.713 [-1.310, -0.092] | 0.0241 |
| Qwen3-Omni-30B, audio | 12.366 | 27/65 | 326 | 9.241 | -6.828 [-8.529, -5.264] | 0.0000 |
| Qwen3-Omni-30B, video + audio, off-screen only | 5.099 | 9/65 | 85 | 4.529 | -2.115 [-2.874, -1.402] | 0.0000 |

One model vs show nothing / vs direct audio-to-image (TEST, d [CI] p):
- m2d: vs nothing +0.138 [-0.184, +0.391] p 0.3889; vs a2i -0.069 [-0.759, +0.598] p 0.8760
- qwen: vs nothing +6.253 [+4.782, +7.862] p 0.0000; vs a2i +6.046 [+4.506, +7.724] p 0.0000
- qwen_av: vs nothing +1.540 [+1.034, +2.092] p 0.0000; vs a2i +1.333 [+0.552, +2.161] p 0.0007

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
