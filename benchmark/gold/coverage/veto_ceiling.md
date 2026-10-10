# Upper bound for any remove-only veto on v1.4, all 158 clips (10 Oct)

v1.4: 59 hits, 65 misses; wrong pictures: 7 visible, 24 cross-trigger, 3 phantom (1 duplicate). Onset cost 2.076.

| perfect veto of | wrongs removed | best possible cost | gain |
|---|---|---|---|
| visible pictures (e.g. SigLIP picture-vs-frame) | 7 | 1.987 | 0.089 |
| phantom pictures (e.g. energy rise) | 3 | 2.038 | 0.038 |
| cross-trigger pictures (e.g. sibling drop) | 24 | 1.772 | 0.304 |
| all wrongs | 34 | 1.646 | 0.430 |

A paired difference must reach about 0.19 to show (noise_floor.md). A visible-only or phantom-only veto cannot reach it even
if perfect and losing no hit, so SigLIP and salience vetoes are not run again. Only cross-trigger removal could, and every
tried cross-trigger veto (sibling_drop, agree_veto, keep-score) lost about one hit per wrong removed.
