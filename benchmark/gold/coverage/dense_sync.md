# Dense frames at the sound's start (Qwen3.8-27B), all 158 clips

pictures scored: 94

| scope | question | bar | hits | wrong | cost |
|---|---|---|---|---|---|
| v1.4 | - | - | 59 | 34 | 2.076 |
| all | sync | -4.562 | 32 | 15 | 2.519 |
| all | sync | -4.125 | 36 | 16 | 2.430 |
| all | sync | -3.758 | 38 | 16 | 2.380 |
| all | sync | -3.125 | 41 | 18 | 2.329 |
| all | sync | -2.607 | 43 | 19 | 2.291 |
| all | sync | -1.875 | 45 | 21 | 2.266 |
| all | sync | -1.42 | 47 | 21 | 2.215 |
| all | sync | -0.875 | 47 | 26 | 2.278 |
| all | sync | -0.467 | 49 | 27 | 2.241 |
| all | sync | 0.365 | 51 | 28 | 2.203 |
| all | sync | 1.3 | 55 | 28 | 2.101 |
| all | sync | 2.598 | 57 | 30 | 2.076 |
| all | sync | 5.655 | 58 | 33 | 2.089 |
| all | change | -2.25 | 28 | 17 | 2.646 |
| all | change | -1.75 | 33 | 17 | 2.519 |
| all | change | -1.375 | 36 | 17 | 2.443 |
| all | change | -1.125 | 40 | 17 | 2.342 |
| all | change | -0.75 | 45 | 17 | 2.215 |
| all | change | -0.25 | 49 | 18 | 2.127 |
| all | change | -0.023 | 50 | 18 | 2.101 |
| all | change | 0.5 | 54 | 19 | 2.013 |
| all | change | 0.94 | 55 | 21 | 2.013 |
| all | change | 1.74 | 56 | 23 | 2.013 |
| all | change | 2.725 | 56 | 27 | 2.063 |
| all | change | 3.25 | 57 | 31 | 2.089 |
| all | change | 4.14 | 58 | 33 | 2.089 |
| short | sync | -4.562 | 45 | 22 | 2.278 |
| short | sync | -4.125 | 48 | 22 | 2.203 |
| short | sync | -3.758 | 49 | 22 | 2.177 |
| short | sync | -3.125 | 50 | 23 | 2.165 |
| short | sync | -2.607 | 51 | 24 | 2.152 |
| short | sync | -1.875 | 52 | 24 | 2.127 |
| short | sync | -1.42 | 53 | 24 | 2.101 |
| short | sync | -0.875 | 53 | 28 | 2.152 |
| short | sync | -0.467 | 54 | 29 | 2.139 |
| short | sync | 0.365 | 54 | 30 | 2.152 |
| short | sync | 1.3 | 57 | 30 | 2.076 |
| short | sync | 2.598 | 59 | 32 | 2.051 |
| short | sync | 5.655 | 59 | 33 | 2.063 |
| short | change | -2.25 | 48 | 26 | 2.253 |
| short | change | -1.75 | 51 | 26 | 2.177 |
| short | change | -1.375 | 52 | 26 | 2.152 |
| short | change | -1.125 | 52 | 26 | 2.152 |
| short | change | -0.75 | 54 | 26 | 2.101 |
| short | change | -0.25 | 56 | 26 | 2.051 |
| short | change | -0.023 | 56 | 26 | 2.051 |
| short | change | 0.5 | 58 | 26 | 2.000 |
| short | change | 0.94 | 58 | 26 | 2.000 |
| short | change | 1.74 | 59 | 27 | 1.987 |
| short | change | 2.725 | 59 | 31 | 2.038 |
| short | change | 3.25 | 59 | 33 | 2.063 |
| short | change | 4.14 | 59 | 33 | 2.063 |

CV by cost: choices [('short', 'change', 1.74), ('short', 'change', 0.5), ('short', 'change', 1.74), ('all', 'change', 0.94), ('all', 'change', 0.5)]; out of fold hits 54, wrong 25, cost 2.089

CV by no hit lost: choices [('short', 'change', 1.74), ('short', 'change', 0.5), ('short', 'change', 1.74), ('short', 'change', 1.74), ('short', 'change', 1.74)]; out of fold hits 58, wrong 27, cost 2.013

## Short pictures, "change" > 1.74 (the bar four of five CV folds chose)

Drops 7 pictures, all wrong, no hit: tg_d088 Explosion 10.8 s (margin 1.75) and Thunder 13.2 s (3.25), m4_film_1917_33a
Gunshot 11.5 s (1.88), tg_d103 Gunshot 1.8 s (2.38) and 8.8 s (1.88), tg_d110 Chink, clink 3.5 s (3.25),
w8_dashcam_ambulance_behind_1a Siren 11.0 s (7.12). On all clips 59 hits / 27 wrong, cost 1.987; paired clip bootstrap
vs v1.4 -0.089 [-0.177, -0.013], p 0.013 (bar chosen on the same clips: optimistic). Out of fold 58 / 27, cost 2.013.
42 of 94 pictures are short (<= 2.5 s); two VLM calls each.
