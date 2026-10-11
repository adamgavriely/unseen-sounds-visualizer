# Pre-registration: DASM "Specific impact sounds" rescue, relabelled to the sound both listeners name (Adam, 11 Oct 2026)

Problem: round-19 DASM rescue (stage 4, dasm_rescue_events) keeps a DASM run only when BOTH open-list listeners
(Qwen3-Omni V4, Audio Flamingo V4) name its family. DASM's "Specific impact sounds" is a parent class: the listeners name
the actual sound (door, glass shatter, gunshot), never the parent, so all 44 such runs on the 158 clips are dropped.

Fix (fixed before counting anything): for a DASM run of family "Specific impact sounds", both listeners' answers are
resolved to families with the pipeline's existing V4 matcher (benchmark/gold/listener_open_inventory.py fam_parse: word
match on listener_variants.match_names, or mpnet cosine > listener_variants.COS, over depictable_vocab.json). A family
named by both is the new label (if several, the one Qwen lists first). The run then follows the normal rescue path
unchanged: DR2 (the family must not already be found in the clip), the display bans (v1.4 texture ban, v1.5
unverifiable families), the a/b on-screen gate, and every v1.4-v1.7 display rule on the whole clip.

Bar (Adam): adopt only if, on all 158 clips after full re-display, hits >= v1.7 + 2 and wrong <= v1.7 + 1
(v1.7: 59 hits, 13 wrong). Step 1, ceiling: if fewer than 2 candidates start within -0.5..+1.0 s of a needed sound
v1.7 misses, stop without the gate run.
