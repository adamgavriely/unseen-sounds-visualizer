# Candidates for fresh confirmation (post-hoc on DEV; test only on newly labelled clips)

Rules found AFTER looking at DEV results. They are not adopted and do not count as DEV wins. Each is tested once, as
written here, only on clips that nobody has scored yet (e.g. if Adam labels a new set).

| rule | found in | DEV numbers (on top of the a/b candidate, 32 hits / 16 wrong) | why post-hoc |
|---|---|---|---|
| Plain texture ban: never draw a picture labelled exactly Vehicle, Water, Engine, Rain, Wind, Liquid, Mechanisms or "Domestic sounds, home sounds" (no listener exception) | Step 11 check, 8 Oct | 31 hits / 14 wrong (removes 3 Vehicle pictures: 1 hit, 2 wrong) | the pre-registered rule had an Audio Flamingo "child named" exception and changed nothing; the plain version was read off the per-picture list |

## Result on TEST (9 Oct)

The plain texture ban above was written here (commit ee75a49, 8 Oct) before it was ever run on TEST. On the 87 TEST
clips it removes 5 wrong pictures and no hit (frozen 25 / 22 -> 25 / 17). This is a clean held-out confirmation.
Pooled over all 158 clips, see pooled_dev_test.md.
