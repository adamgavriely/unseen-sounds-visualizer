# All 158 clips: learned ban list and gate re-check (flash + adopted hold in every row)

J = cost_cov + 0.5 x (wrong + stale s) per clip; all numbers on held-out clips.

## gate M

| fold | learned ban list | held-out J: none / texture list / learned | hits / wrong (learned) |
|---|---|---|---|
| DEV->TEST | Bird, Computer keyboard, Crowd, Gunshot, Screaming, Vehicle, Water | 3.444 / 2.985 / 2.980 | 20 / 12 |
| TEST->DEV | Baby cry, infant cry, Bee, wasp, etc., Gunshot, Honk, Human locomotion, Shofar, Siren, Vehicle, Water | 2.847 / 2.610 / 2.665 | 24 / 9 |
| random 1/5 | Bee, wasp, etc., Bird, Dog, Gunshot, Honk, Screaming, Shofar, Siren, Vehicle, Water | 2.870 / 2.870 / 3.119 | 12 / 4 |
| random 2/5 | Baby cry, infant cry, Bee, wasp, etc., Crowd, Dog, Gunshot, Human locomotion, Screaming, Shofar, Siren, Vehicle, Water | 3.281 / 2.794 / 3.245 | 5 / 1 |
| random 3/5 | Baby cry, infant cry, Honk, Human locomotion, Screaming, Shofar, Vehicle | 5.280 / 4.444 / 4.734 | 10 / 15 |
| random 4/5 | Baby cry, infant cry, Bee, wasp, etc., Gunshot, Honk, Human locomotion, Shofar, Siren, Vehicle, Water | 1.899 / 1.895 / 1.777 | 6 / 1 |
| random 5/5 | Baby cry, infant cry, Bee, wasp, etc., Crowd, Gunshot, Honk, Human locomotion, Screaming, Shofar, Siren, Vehicle, Water | 2.485 / 2.026 / 2.081 | 8 / 4 |

M no ban: J 3.175, hits 54/124, wrong 35, onset cost 2.215, cost_cov 2.398, cover 0.87, |end err| 0.50 s, wrong s/clip 1.34

M texture list: J 2.816, hits 53/124, wrong 27, onset cost 2.139, cost_cov 2.322, cover 0.86, |end err| 0.49 s, wrong s/clip 0.83

M learned (5-fold out-of-sample): J 3.005, hits 41/124, wrong 25, onset cost 2.418, cost_cov 2.581, cover 0.84, |end err| 0.47 s, wrong s/clip 0.75

Ban list learned on all 158 clips (for shipping, not a score): Baby cry, infant cry, Bee, wasp, etc., Gunshot, Honk, Human locomotion, Screaming, Shofar, Siren, Vehicle, Water

## gate AB-m

| fold | learned ban list | held-out J: none / texture list / learned | hits / wrong (learned) |
|---|---|---|---|
| DEV->TEST | Baby cry, infant cry, Bird, Computer keyboard, Crowd, Gunshot, Screaming, Train, Vehicle | 3.575 / 3.086 / 3.034 | 21 / 14 |
| TEST->DEV | Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Gunshot, Honk, Human locomotion, Rowboat, canoe, kayak, Shofar, Siren, Vehicle | 2.819 / 2.680 / 2.693 | 27 / 10 |
| random 1/5 | Baby cry, infant cry, Bee, wasp, etc., Bird, Chink, clink, Dog, Gunshot, Honk, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle | 3.124 / 3.124 / 3.083 | 12 / 4 |
| random 2/5 | Baby cry, infant cry, Bee, wasp, etc., Crowd, Dog, Gunshot, Human locomotion, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle | 3.213 / 2.726 / 3.213 | 7 / 2 |
| random 3/5 | Baby cry, infant cry, Chink, clink, Gunshot, Honk, Human locomotion, Screaming, Shofar, Train, Vehicle | 4.811 / 4.410 / 3.755 | 10 / 9 |
| random 4/5 | Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Gunshot, Honk, Human locomotion, Rowboat, canoe, kayak, Shofar, Siren, Vehicle | 2.531 / 2.216 / 2.099 | 6 / 2 |
| random 5/5 | Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Crowd, Gunshot, Honk, Human locomotion, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle | 2.453 / 1.994 / 1.952 | 9 / 4 |

AB-m no ban: J 3.236, hits 60/124, wrong 41, onset cost 2.139, cost_cov 2.363, cover 0.85, |end err| 0.50 s, wrong s/clip 1.53

AB-m texture list: J 2.904, hits 59/124, wrong 34, onset cost 2.076, cost_cov 2.300, cover 0.85, |end err| 0.49 s, wrong s/clip 1.04

AB-m learned (5-fold out-of-sample): J 2.831, hits 44/124, wrong 21, onset cost 2.291, cost_cov 2.470, cover 0.84, |end err| 0.46 s, wrong s/clip 0.62

Ban list learned on all 158 clips (for shipping, not a score): Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Gunshot, Honk, Human locomotion, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle

## gate AB-s

| fold | learned ban list | held-out J: none / texture list / learned | hits / wrong (learned) |
|---|---|---|---|
| DEV->TEST | Baby cry, infant cry, Bird, Computer keyboard, Crowd, Footsteps, Gunshot, Rain, Screaming, Train, Vehicle | 3.861 / 3.372 / 3.235 | 21 / 17 |
| TEST->DEV | Alarm, Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Gunshot, Honk, Human locomotion, Roar, Rowboat, canoe, kayak, Shofar, Siren, Vehicle | 3.165 / 2.758 / 3.130 | 23 / 14 |
| random 1/5 | Baby cry, infant cry, Bee, wasp, etc., Bird, Chink, clink, Dog, Gunshot, Honk, Rain, Roar, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle | 3.210 / 3.210 / 3.169 | 12 / 5 |
| random 2/5 | Baby cry, infant cry, Bee, wasp, etc., Crowd, Dog, Footsteps, Gunshot, Human locomotion, Rain, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle | 3.331 / 2.843 / 3.330 | 7 / 3 |
| random 3/5 | Baby cry, infant cry, Chink, clink, Crowd, Footsteps, Gunshot, Honk, Human locomotion, Roar, Screaming, Shofar, Train, Vehicle | 5.540 / 4.840 / 4.484 | 10 / 13 |
| random 4/5 | Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Footsteps, Gunshot, Honk, Human locomotion, Rain, Roar, Rowboat, canoe, kayak, Shofar, Siren, Vehicle | 2.857 / 2.543 / 2.326 | 6 / 4 |
| random 5/5 | Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Crowd, Footsteps, Gunshot, Honk, Human locomotion, Rain, Roar, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle | 2.759 / 1.994 / 1.952 | 9 / 4 |

AB-s no ban: J 3.549, hits 60/124, wrong 51, onset cost 2.266, cost_cov 2.490, cover 0.85, |end err| 0.50 s, wrong s/clip 1.90

AB-s texture list: J 3.096, hits 59/124, wrong 41, onset cost 2.165, cost_cov 2.389, cover 0.85, |end err| 0.49 s, wrong s/clip 1.25

AB-s learned (5-fold out-of-sample): J 3.064, hits 44/124, wrong 29, onset cost 2.392, cost_cov 2.571, cover 0.84, |end err| 0.46 s, wrong s/clip 0.89

Ban list learned on all 158 clips (for shipping, not a score): Baby cry, infant cry, Bee, wasp, etc., Chink, clink, Crowd, Footsteps, Gunshot, Honk, Human locomotion, Rain, Roar, Rowboat, canoe, kayak, Screaming, Shofar, Siren, Train, Vehicle

## Gate comparison (out-of-sample, learned ban, paired clip bootstrap of per-clip J)

- AB-m - M: -0.174 [-0.412, +0.061], p 0.14
- AB-s - M: +0.059 [-0.243, +0.373], p 0.72
- AB-m - AB-s: -0.233 [-0.442, -0.070], p 0.00
