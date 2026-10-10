# Five ideas, in order, clip-grouped 5-fold CV on all 158 clips (out of fold)

Current system: 59 hits / 34 wrong, onset cost 2.076, J 2.904, cover 0.85, |end err| 0.49 s, start err +0.06 s

| idea | options | chosen per fold | out-of-fold hits / wrong | onset cost | J | adopted? |
|---|---|---|---|---|---|---|
| 1 restart at raw onsets | [None, (1.0, 'drawn'), (2.0, 'drawn'), (3.0, 'drawn'), (1.0, 'any'), (2.0, 'any'), (3.0, 'any')] | [None, None, None, None, None] | 59 / 34 | 2.076 | 2.904 | no |
| 2 evidence-bound ends | [(0.5, None, False, 0.0, 'extend'), (0.5, None, False, 0.0, 'both'), (0.5, None, False, 0.5, 'both'), (0.4, None, False, 0.0, 'both'), (0.4, None, False, 0.5, 'both')] | [(0.5, None, False, 0.0, 'extend'), (0.5, None, False, 0.0, 'extend'), (0.5, None, False, 0.0, 'extend'), (0.5, None, False, 0.0, 'extend'), (0.5, None, False, 0.0, 'extend')] | 59 / 34 | 2.076 | 2.904 | no |
| 3 parent on sibling disputes | [None, 0.5, 1.0] | [None, None, None, None, None] | 59 / 34 | 2.076 | 2.904 | no |
| 5 best-timed guess | ['current', 'strongest', 'flexfirst'] | ['current', 'current', 'current', 'current', 'current'] | 59 / 34 | 2.076 | 2.904 | no |
| 4 contrast dropper | - | - | - | - | - | skipped: TEST Omni answers not ready |

Final config: {'restart': None, 'parent': None, 'time': 'current', 'hold': (0.5, None, False, 0.0, 'extend')}
