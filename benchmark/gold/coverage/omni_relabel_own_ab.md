# Omni closed-choice relabel / drop (R3) on top of R1 own_ab, all 158 clips

pictures with an Omni closed-choice vote: 90

| scope, mode, m | hits | wrong | cost (w=2) | cost (w=4) |
|---|---|---|---|---|
| v1.4 | 58 | 31 | 2.063 | 2.456 |
| ('weapon', 'relabel', 0.0) | 56 | 30 | 2.101 | 2.481 |
| ('weapon', 'relabel', 0.2) | 56 | 30 | 2.101 | 2.481 |
| ('weapon', 'relabel', 0.4) | 56 | 31 | 2.114 | 2.506 |
| ('weapon', 'drop', 0.0) | 55 | 25 | 2.063 | 2.380 |
| ('weapon', 'drop', 0.2) | 55 | 25 | 2.063 | 2.380 |
| ('weapon', 'drop', 0.4) | 55 | 26 | 2.076 | 2.405 |
| ('all', 'relabel', 0.0) | 26 | 50 | 3.114 | 3.747 |
| ('all', 'relabel', 0.2) | 30 | 46 | 2.962 | 3.544 |
| ('all', 'relabel', 0.4) | 38 | 43 | 2.722 | 3.266 |
| ('all', 'drop', 0.0) | 25 | 13 | 2.671 | 2.835 |
| ('all', 'drop', 0.2) | 29 | 14 | 2.582 | 2.759 |
| ('all', 'drop', 0.4) | 37 | 18 | 2.430 | 2.658 |

CV by w=2: choices [None, None, ('weapon', 'drop', 0.0), None, None]; out of fold hits 55, wrong 27, cost w=2 2.089, w=4 2.430

CV by w=4: choices [('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0)]; out of fold hits 55, wrong 25, cost w=2 2.063, w=4 2.380
