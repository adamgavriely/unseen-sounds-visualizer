# Panelist 2 — visibility gate & vision (round 1)

**Ground facts (cached DEV judge set: 41 seen / 38 needed, base 16 silenced / 33 kept; gate_gold caches)**
- Kills (5): golf Whack 6.5, pet-shop Bird (3-0 on all 6 stretches), bell_miami (2-1 ×3), ambulance Vehicle 7.3, tornado Siren. 6 cached signals per kill (votes, HUMAN-2 action, CF Q1/Q2, BOX-2, Synchformer sync, SSL-SaN) vs the 8 silenced-seen sounds with the same 2-x vote pattern: nothing separates them. Ambulance is span timing. Fart ×2: no cache. Kill side at the signal ceiling.
- Leaks (25): 18 are "nothing" named on every stretch, 16 are 0-3 everywhere. Every closed question requires a SOURCE object first; thunder, air horn, fireworks, POV footsteps, bakery hands have none -> the "nothing" escape is the structural gap.
- The gate-gold screen asks with GOLD labels, the pipeline with stage-4 labels -> gate-gold GOs (SYNC-2, BOX-2's bell) moved no DEV picture; 9/18 DEV wrongs are invisible to the screen by construction.
- LOCO-CV fusion of all cached signals: unstable -> not proposed.

**P1 SIGN — effect-centric gate vote, source-free, ADD-seen.** Qwen3.8-27B via reason._ab both orders: "(a) the visible SIGN of {label} — the effect or motion that sound makes — is in these frames, even with no audio (b) nothing in the frames shows that sound happening". Seen iff shipped majority OR sign=yes (both orders) on EVERY stretch; 4th-vote variant reported. Targets the 18 nothing-leaks and in DEV thunder ×2 / water / laughter / fireworks visible wrongs. Selection: gate-gold bar then DEV arm. Risk: as_explosion war street holds 6 needed sounds; thunder flash may be missed by 6 frames. ~3 h, one H200 job.

**P2 ONSET + SYNC-ADD (CPU).** (i) seen iff the FIRST stretch is majority-seen: cached 19/32 (+Drum, +crossing Train, +protest Air horn; −rainforest_7629 Bird, already a DEV miss). (ii) SYNC-ADD with t from leave-one-clip-out on DEV: 25/33. Combined 28/41, 32/38. Expected zero DEV picture change.

**P3 DEPICT-EVENT.** For each placed picture ask whether the depiction's EVENT ("train releasing steam", "fireworks bursting") is visibly happening at picture start ±1 s, both orders; drop iff yes both. Targets the 9 DEV wrongs the gate-gold screen can't see. Risk: hits with look-alike activity. GPU 2 min, ~2 h.

**First: P1 SIGN.**
