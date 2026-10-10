# Upper bound for Fable's rescue ideas on v1.4, all 158 clips (10 Oct)

Each rescue can only bring back pictures that one v1.4 step removes. Turning that step fully off shows the most it can add.

| v1.4 with | hits | wrong | onset cost |
|---|---|---|---|
| nothing changed | 59 | 34 | 2.076 |
| no texture ban (ceiling of the "sudden texture" exception) | 60 | 41 | 2.139 |
| no on-screen gate (ceiling of a gate rescue) | 61 | 62 | 2.380 |

A rescue that kept every extra hit and added no wrong would gain at most 1 hit (ban) or 2 hits (gate): 0.025 / 0.051
cost, below the 0.19 noise floor (noise_floor.md). Not run. The third idea (bypass the listener when BEATs and FlexSED
agree) falls under leak_table.md: the listener step's drops are 2 % real sounds.
