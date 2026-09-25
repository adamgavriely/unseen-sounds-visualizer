Confirmation clip order (GP-3 step 1), fixed 2026-09-25 before any picture or gate run on these clips:
every `data/input/benchmark/unsorted/*.mp4` not in gold_dev, gold_test, gold139/all or pic_fresh (912 clips),
sorted by sha256("confirm-2026-09-25/" + filename). The confirmation set is the first 50 clips in this order
on which the shipped detector + gate (pictures off) shows at least one off-screen sound in group (a) or (b).
