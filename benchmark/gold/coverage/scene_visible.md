# Scene-visible suppress (R1), all 158 clips

| variant | hits | wrong | cost (w=2) | cost (w=4) |
|---|---|---|---|---|
| v1.4 | 59 | 34 | 2.076 | 2.506 |
| own_ab | 58 | 31 | 2.063 | 2.456 |
| own_any | 41 | 16 | 2.304 | 2.506 |
| other | 55 | 30 | 2.127 | 2.506 |
| own_ab+other | 54 | 29 | 2.139 | 2.506 |
| own_any+other | 39 | 16 | 2.354 | 2.557 |

CV by w=2: choices ['own_ab', 'own_ab', 'own_ab', 'own_ab', None]; out of fold hits 58, wrong 32, cost w=2 2.076, w=4 2.481

CV by w=4: choices ['own_ab', 'own_ab', 'own_ab', 'own_ab', 'own_any']; out of fold hits 54, wrong 30, cost w=2 2.152, w=4 2.532
