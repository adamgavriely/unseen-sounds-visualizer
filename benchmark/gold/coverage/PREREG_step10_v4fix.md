# Pre-registration: Step 10, fix the Qwen3-Omni V4 open-list decoding and re-run the a/b candidate (DEV only)

Written and committed 2026-10-08 before any fixed answer was generated.

Bug (counted before writing this, `v4fix.py count`): the Qwen V4 open-list answers ("List every distinct non-speech
sound you hear ...") in the caches the frozen pipeline reads are almost all degenerate: generation had no end-of-turn
stop and ran on with "Assistant" / repeated words until 64 tokens, so the list holds 1-2 real names.

| cache | DEV Qwen V4 answers | degenerate | dev2 | degenerate |
|---|---|---|---|---|
| listener_v (TIER band rescue, V4 accept) | 626 | 611 | 404 | 388 |
| listener_p1v4 (K-V4, two-witness names, K4A) | 274 | 271 | 102 | 102 |
| listener_p4 (DASM rescue) | 76 | 73 | 38 | 37 |

Fix (`v4fix.py`): same prompt, same audio cut, greedy, 64 tokens, plus eos_token_id = [<|im_end|>, <|endoftext|>] and
repetition_penalty 1.1; repeated lines and role words removed. Every derived field is recomputed with the frozen code's
own matchers (listener_variants V4 block, listener_p1v4.fam_parse, listener_afnext.v4_match). TEST caches are fixed too
(test2; test if present) and never scored here.

Test: arm SHIP8+MD3+WW5+SL+V4FIX = the frozen arm with the three DEV / dev2 caches swapped for the fixed ones, nothing
else changed; harness stage 4 + stage 5 (new gate stretches asked live, memoised); then the a/b gate re-decision (Step 2).
Pass bar: hits >= 32, wrong <= 16, and cost_cov below the a/b candidate's 2.421. Reported also without the a/b rule, and
hits split by label provenance where the gold records it.

## Amendment (8 Oct, written after the V4FIX DEV result, before the combination was computed)

Combination run, no new tuning: V4FIX + a/b gate (AB-m) + the Step 11 flash rule F exactly as pre-registered there.
Reported with the same pass bar as Step 10 (hits >= 32, wrong <= 16, cost_cov below 2.421).
