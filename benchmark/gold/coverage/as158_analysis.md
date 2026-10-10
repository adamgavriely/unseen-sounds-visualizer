# BEATs-strong (PretrainedSED) on the 158 benchmark clips

## A. Wrong-dropper on v1.7 pictures

Family probability per picture (median, and the lowest hit):

| feature | hits median | lowest hit | wrongs median | wrongs below the lowest hit |
|---|---|---|---|---|
| BEATs_in | 0.410 | 0.014 | 0.293 | 1 of 13 |
| BEATs_on | 0.286 | 0.011 | 0.204 | 1 of 13 |
| mean_in | 0.391 | 0.012 | 0.330 | 1 of 13 |
| mean_on | 0.270 | 0.010 | 0.254 | 1 of 13 |

No family class in the 447 (feature empty, never dropped): 

v1.7: 59 hits, 13 wrong, cost 1.810

| feature | bar | hits | wrong | cost |
|---|---|---|---|---|
| BEATs_in | 0.0157 | 58 | 12 | 1.823 |
| BEATs_in | 0.0323 | 56 | 12 | 1.873 |
| BEATs_in | 0.0645 | 54 | 11 | 1.911 |
| BEATs_in | 0.1306 | 52 | 11 | 1.962 |
| BEATs_in | 0.1538 | 49 | 10 | 2.025 |
| BEATs_in | 0.1779 | 47 | 10 | 2.076 |
| BEATs_in | 0.1948 | 44 | 10 | 2.152 |
| BEATs_in | 0.2011 | 44 | 8 | 2.127 |
| BEATs_in | 0.2187 | 42 | 7 | 2.165 |
| BEATs_in | 0.2337 | 40 | 7 | 2.215 |
| BEATs_in | 0.2806 | 37 | 7 | 2.291 |
| BEATs_in | 0.3096 | 36 | 6 | 2.304 |
| BEATs_on | 0.0126 | 58 | 12 | 1.823 |
| BEATs_on | 0.0185 | 56 | 12 | 1.873 |
| BEATs_on | 0.0409 | 53 | 12 | 1.949 |
| BEATs_on | 0.0707 | 52 | 11 | 1.962 |
| BEATs_on | 0.0876 | 50 | 10 | 2.000 |
| BEATs_on | 0.1314 | 48 | 10 | 2.051 |
| BEATs_on | 0.1614 | 44 | 10 | 2.152 |
| BEATs_on | 0.1662 | 42 | 10 | 2.203 |
| BEATs_on | 0.1802 | 40 | 9 | 2.241 |
| BEATs_on | 0.1913 | 38 | 9 | 2.291 |
| BEATs_on | 0.201 | 37 | 7 | 2.291 |
| BEATs_on | 0.2079 | 36 | 6 | 2.304 |
| mean_in | 0.016 | 58 | 12 | 1.823 |
| mean_in | 0.0418 | 56 | 12 | 1.873 |
| mean_in | 0.0901 | 54 | 11 | 1.911 |
| mean_in | 0.1645 | 53 | 10 | 1.924 |
| mean_in | 0.1758 | 51 | 9 | 1.962 |
| mean_in | 0.1855 | 49 | 9 | 2.013 |
| mean_in | 0.1984 | 47 | 8 | 2.051 |
| mean_in | 0.2231 | 45 | 8 | 2.101 |
| mean_in | 0.2425 | 41 | 8 | 2.203 |
| mean_in | 0.2483 | 39 | 8 | 2.253 |
| mean_in | 0.2542 | 37 | 7 | 2.291 |
| mean_in | 0.2834 | 35 | 7 | 2.342 |
| mean_on | 0.0139 | 58 | 12 | 1.823 |
| mean_on | 0.0212 | 56 | 12 | 1.873 |
| mean_on | 0.0449 | 54 | 11 | 1.911 |
| mean_on | 0.1048 | 52 | 11 | 1.962 |
| mean_on | 0.1151 | 49 | 11 | 2.038 |
| mean_on | 0.1316 | 47 | 11 | 2.089 |
| mean_on | 0.1506 | 45 | 10 | 2.127 |
| mean_on | 0.1689 | 43 | 10 | 2.177 |
| mean_on | 0.1824 | 41 | 9 | 2.215 |
| mean_on | 0.1919 | 40 | 8 | 2.228 |
| mean_on | 0.224 | 38 | 7 | 2.266 |
| mean_on | 0.2425 | 35 | 7 | 2.342 |

CV choices [None, None, None, None, ('BEATs_in', 0.0157)]; out of fold 58 hits, 13 wrong, cost 1.835
Pass (0 hits lost, >= 2 wrongs removed out of fold): NO

## B. BEATs-strong as an extra detector (upper bound)

needed sounds 124; v1.7 misses 65; of these, 65 have a family class among the 447.

| bar t | recoverable misses | runs | extra runs where an on-screen sound of the family plays | other extra runs | 4 x rec - 2 x other | pass |
|---|---|---|---|---|---|---|
| 0.2 | 16 | 685 | 111 | 457 | -850 | no |
| 0.3 | 9 | 431 | 72 | 262 | -488 | no |
| 0.4 | 4 | 271 | 61 | 150 | -284 | no |
| 0.5 | 1 | 144 | 27 | 86 | -168 | no |
| 0.6 | 0 | 75 | 15 | 42 | -84 | no |

Recoverable: a v1.7-missed needed sound with a BEATs-strong run of its family starting -0.5..+1.0 s of its onset.
Extra runs: runs that match no needed onset and no v1.7 picture of their family; before any veto or gate (upper bound of wrongs).
