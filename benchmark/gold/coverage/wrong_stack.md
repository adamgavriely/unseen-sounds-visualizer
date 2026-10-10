# Wrong-dropping stack on top of R1 + R3, all 158 clips

| added dropper | hits | wrong | cost (w=2) | cost (w=4) |
|---|---|---|---|---|
| R1 + R3 | 55 | 25 | 2.063 | 2.380 |
| ('rise', 'full', -3.0) | 52 | 23 | 2.114 | 2.405 |
| ('rise', 'full', 0.0) | 46 | 18 | 2.203 | 2.430 |
| ('rise', 'full', 2.0) | 43 | 13 | 2.215 | 2.380 |
| ('rise', 'full', 4.0) | 43 | 11 | 2.190 | 2.329 |
| ('rise', 'hi', -3.0) | 50 | 21 | 2.139 | 2.405 |
| ('rise', 'hi', 0.0) | 49 | 17 | 2.114 | 2.329 |
| ('rise', 'hi', 2.0) | 45 | 13 | 2.165 | 2.329 |
| ('rise', 'hi', 4.0) | 40 | 11 | 2.266 | 2.405 |
| ('peak', None, 0.3) | 55 | 24 | 2.051 | 2.354 |
| ('peak', None, 0.4) | 55 | 24 | 2.051 | 2.354 |
| ('peak', None, 0.5) | 52 | 24 | 2.127 | 2.430 |
| ('peak', None, 0.6) | 52 | 23 | 2.114 | 2.405 |

CV by w=2: choices [None, ('peak', None, 0.3), ('peak', None, 0.3), ('peak', None, 0.3), ('peak', None, 0.3)]; out of fold hits 55, wrong 25, cost w=2 2.063, w=4 2.380

CV by w=4: choices [None, ('rise', 'full', 4.0), ('rise', 'hi', 2.0), ('rise', 'hi', 0.0), ('rise', 'hi', 0.0)]; out of fold hits 44, wrong 18, cost w=2 2.253, w=4 2.481
