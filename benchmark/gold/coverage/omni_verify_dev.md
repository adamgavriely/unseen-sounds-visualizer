# Step 3b: Qwen3-Omni verification of every DEV candidate burst (measure only)

## (1) verifier: P(own family) in a closed choice {family, siblings, none}

| bursts | n | right | AUROC |
|---|---|---|---|
| all | 1196 | 209 | 0.652 |
| drawn by the frozen system | 50 | 44 | 0.455 |
| not drawn | 1146 | 165 | 0.626 |
| has a beats member | 537 | 136 | 0.656 |
| has a flexsed member | 135 | 54 | 0.734 |
| has a flexsed band member | 610 | 72 | 0.664 |
| has a dasm member | 67 | 27 | 0.555 |
| all, score = 1 - P(none) | 1196 | 209 | 0.743 |

## (2) onset gate: P(yes, the maker or a visible event is on screen), audio + 4 fps video

| set | n | visible | Omni AUROC | current gate AUROC | a/b rule AUROC |
|---|---|---|---|---|---|
| right-family bursts with a visibility label | 201 | 104 | 0.781 | - | - |
| of these, judged by the current gate | 109 | 45 | 0.797 | 0.816 | 0.796 |

## (3) still heard? per second over burst + 3 s

7278 seconds, 1309 inside a same-family gold sound; AUROC 0.753
