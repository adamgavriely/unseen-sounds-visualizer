# Pre-registration: Qwen3.8-27B for the visibility decision (scope v2, point 1, second swap)

*Committed 2026-09-17 night, before the run. Smoke test: Qwen3.8-27B (Aug 2026, native
vision-language, Apache-2.0) loads in our environment on an A100-80 (51 GB) and answered the
ambulance clip's visibility question correctly in 5 s with thinking off.*

## What is swapped

Stage 5's vision-language model, Qwen2.5-VL-7B → **Qwen3.8-27B**. The three-vote visibility
question, the prompts and the "silence only if visible in every stretch" rule are unchanged.
The 7B's weakness on DCASE gold is on-screen recall: it silences only 35% of sounds whose
source is on screen (off-screen recall 97%). The 32B Qwen2.5-VL tried earlier raised that to
67% but wrongly silenced 25% of off-screen sounds and failed.

## Two arms, both run

- **thinking on** (primary; Adam's preference, and Qwen's own guidance for visual questions):
  the model reasons first, then gives the final short answer; the pipeline reads the text after
  the reasoning. Cost: several times slower per question.
- **thinking off** (control): the same prompts, direct answer.

The prompts are not changed between arms except for the instruction to end with the final
answer; nothing is rewritten after seeing the numbers.

## Measured on DCASE 2025 gold, the same 258 events as before

| measure | 7B (current) | 32B (failed) | Qwen3.8-27B must reach |
|---|---|---|---|
| on-screen recall (visible source correctly silenced) | 35% | 67% | **≥ 50%** |
| off-screen recall (unseen source correctly kept) | 97% | 75% | **≥ 95%** (drop ≤ 2 points) |

**PASS iff both, in at least one arm.** If both arms pass, the thinking arm is adopted only if
it beats the other by ≥ 5 points of on-screen recall (otherwise the faster one). On pass:
the model becomes stage 5 and the 100-clip protocol is re-run once as v4 (together with any
other passing swap). On fail: attempt nine in the table, with the numbers.

## Not done

No prompt tuning on DCASE; no change of the vote rule; the gate's bar stays.

## Outcome (added 2026-09-19 morning, after both arms ran)

`benchmark/eval_dcase_visibility_q38_direct.json`, `_think.json` (258 events each).

| arm | on-screen recall (bar ≥ 50%) | off-screen recall (bar ≥ 95%) |
|---|---|---|
| Qwen2.5-VL-7B (v3) | 35% | 97% |
| Qwen3.8-27B, thinking off | **56.3%** ✔ | 82.6% ✘ |
| Qwen3.8-27B, thinking on | 51.2% ✔ | 78.0% ✘ |

**FAILED as pre-registered in both arms** (off-screen recall falls 14–19 points; the same
trade the 32B made). Thinking is not better on either measure and ~3× slower, so by the
declared rule the direct arm is the one used. The swap went ahead by decision (best models)
with these numbers disclosed; on the 100-clip protocol the Qwen3.8 gate changed nothing
(v4b: gated 2.85 vs blind 2.75 under the rubric judge, v3 2.83 vs 2.73), and it still
draws a picture on 28 of the 50 clips where none is due (dev-sweep accuracy 52.7% vs the
7B's 44.9%).
