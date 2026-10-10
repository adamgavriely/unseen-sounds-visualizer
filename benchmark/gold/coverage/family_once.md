# One picture per group per gap (R2), all 158 clips

| group, gap s | hits | wrong | cost (w=2) | cost (w=4) |
|---|---|---|---|---|
| v1.4 | 59 | 34 | 2.076 | 2.506 |
| ('label', 3.0) | 59 | 34 | 2.076 | 2.506 |
| ('label', 6.0) | 58 | 34 | 2.101 | 2.532 |
| ('label', 10.0) | 56 | 30 | 2.101 | 2.481 |
| ('label', 1000000000.0) | 54 | 28 | 2.127 | 2.481 |
| ('family', 3.0) | 59 | 34 | 2.076 | 2.506 |
| ('family', 6.0) | 58 | 34 | 2.101 | 2.532 |
| ('family', 10.0) | 56 | 30 | 2.101 | 2.481 |
| ('family', 1000000000.0) | 54 | 28 | 2.127 | 2.481 |
| ('coarse', 3.0) | 57 | 30 | 2.076 | 2.456 |
| ('coarse', 6.0) | 54 | 31 | 2.165 | 2.557 |
| ('coarse', 10.0) | 52 | 26 | 2.152 | 2.481 |
| ('coarse', 1000000000.0) | 50 | 24 | 2.177 | 2.481 |

CV by w=2: choices [('coarse', 3.0), None, None, ('coarse', 3.0), None]; out of fold hits 57, wrong 34, cost w=2 2.127, w=4 2.557

CV by w=4: choices [('coarse', 3.0), ('label', 1000000000.0), None, ('coarse', 3.0), ('label', 1000000000.0)]; out of fold hits 53, wrong 34, cost w=2 2.228, w=4 2.658
