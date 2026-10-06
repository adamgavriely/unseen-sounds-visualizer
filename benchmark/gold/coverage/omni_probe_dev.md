# Step 3 prep: Qwen3-Omni probes on DEV (measure only)

## (a) is the source visibly making the sound (gate records)

24 visible, 35 not visible; left out: {'obvious only': 2, 'no gold sound': 11}

| score | AUROC (visible vs not) |
|---|---|
| Omni, mean over seconds | 0.810 |
| Omni, max over seconds | 0.804 |
| current gate (share of stretches seen) | 0.780 |
| a/b rule (share seen, split -> majority) | 0.789 |

## (b) closed choice: which of these heard families, or none (audio)

| set | n | right | AUROC (first-token prob of own letter) | named in both orders: right / wrong |
|---|---|---|---|---|
| all candidates | 3131 | 993 | 0.746 | 600/993 / 483/2138 |
| kept (drawn) | 116 | 105 | 0.739 | 89/105 / 8/11 |
| dropped | 3015 | 888 | 0.734 | 511/888 / 475/2127 |
| frozen DEV pictures (right = not other-sound / no-sound) | 45 | 36 | 0.519 | 26/36 / 6/9 |
