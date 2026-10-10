# Omni closed-choice relabel / drop (R3), all 158 clips

pictures with an Omni closed-choice vote: 94

| scope, mode, m | hits | wrong | cost (w=2) | cost (w=4) |
|---|---|---|---|---|
| v1.4 | 59 | 34 | 2.076 | 2.506 |
| ('weapon', 'relabel', 0.0) | 57 | 33 | 2.114 | 2.532 |
| ('weapon', 'relabel', 0.2) | 57 | 33 | 2.114 | 2.532 |
| ('weapon', 'relabel', 0.4) | 57 | 34 | 2.127 | 2.557 |
| ('weapon', 'drop', 0.0) | 56 | 28 | 2.076 | 2.430 |
| ('weapon', 'drop', 0.2) | 56 | 28 | 2.076 | 2.430 |
| ('weapon', 'drop', 0.4) | 56 | 29 | 2.089 | 2.456 |
| ('all', 'relabel', 0.0) | 27 | 53 | 3.127 | 3.797 |
| ('all', 'relabel', 0.2) | 31 | 49 | 2.975 | 3.595 |
| ('all', 'relabel', 0.4) | 39 | 46 | 2.734 | 3.316 |
| ('all', 'drop', 0.0) | 26 | 16 | 2.684 | 2.886 |
| ('all', 'drop', 0.2) | 30 | 17 | 2.595 | 2.810 |
| ('all', 'drop', 0.4) | 38 | 21 | 2.443 | 2.709 |

CV by w=2: choices [None, None, ('weapon', 'drop', 0.0), None, None]; out of fold hits 56, wrong 30, cost w=2 2.101, w=4 2.481

CV by w=4: choices [('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0), ('weapon', 'drop', 0.0)]; out of fold hits 56, wrong 28, cost w=2 2.076, w=4 2.430
