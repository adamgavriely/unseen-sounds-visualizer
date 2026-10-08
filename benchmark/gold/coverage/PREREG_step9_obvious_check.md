# Pre-registration: Step 9, "would a viewer already expect this sound from the picture?" (DEV only)

Written and committed 2026-10-08 before any answer was generated. Base = the a/b candidate (32 hits / 16 wrong).
7 of its 16 wrong pictures are "source visible or obvious" (Train in a subway, Thunder in a storm, ...): the gate asks
whether the maker is visible, never whether the sound is already expected from what is on screen.

Items: every a/b-candidate picture on DEV (label, start). Model: Qwen3-Omni-30B-A3B, video only (no audio), the clip cut
[start - 1, start + 2] s at 2 frames/s. Question: "These frames are from a video shown without sound. Would a viewer
already expect to hear a {label} sound here, just from what is on screen? Answer yes or no." Score = P(yes) (softmax of
the max yes / no logits). Rule: a picture is removed when P(yes) >= q, q in {0.5, 0.7, 0.9}. Reported: hits / wrong /
onset cost per q; AUROC of P(yes) for wrong-visible-or-obvious vs hit pictures. Clear win = >= 32 hits at <= 13 wrong.
