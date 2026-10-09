# Pre-registration: Step 14, target-separation verifier with AudioSep (DEV only)

Written and committed 2026-10-09 before any separation was run. TEST untouched; adopt nothing.
Base = a/b + flash (32 hits / 14 wrong; leading DEV candidate, missed its own bar by one wrong, not adopted).

Model: AudioSep (MIT, Audio-AGI/AudioSep, audiosep_base_4M_steps.ckpt, its CLAP encoder
music_speech_audioset_epoch_15_esc_89.98.pt), 32 kHz. For every DEV candidate burst of Step 3b (1196): cut the clip
audio [start - 2, end + 2] s, query "{family}" (lower case), stem = AudioSep output, residual = mix - stem.
Features: (1) stem energy / mix energy inside the burst; (2) stem energy inside the burst / stem energy in the rest of
the cut; (3) CLAP(stem inside the burst, text) - CLAP(residual inside the burst, text) (cosine of the CLAP audio and
text embeddings).
Truth (from gold, after the run): real = a same-family gold sound overlaps the burst (0.5 s slack), as Step 3b (1).

Bar 1: AUROC >= 0.80 real vs not, for feature (3) alone or for a logistic regression of (1)-(3) with clip-grouped
10-fold CV (both reported).
Bar 2 (only if bar 1 passes), on top of a/b + flash: (i) drop a picture whose burst score < t; (ii) add a dropped burst
(no picture of its family overlapping) whose score > u, as a picture (family, burst start, burst end).
t, u = the out-of-fold score quantiles {0.1, 0.2, 0.3} and {0.95, 0.98, 0.99}. Pass: hits >= 33 at wrong <= 14, or
wrong <= 12 at hits >= 32.
