# Panel 2026-09-25 — round 2: the merged plan, and new evidence on V3 (sign or amend)

## New evidence since round 1 (read this first)

The V3 subjects for the 54 sounds of the 50 never-annotated clips were written before any picture was
drawn or seen (`data/work/picture_bench_fresh/subjects_V3.json`, cluster). Reading them, V3 has two
failures that the round-3 design did not foresee — both are the scene question Adam raised:

**(a) V3 invents a thing the audio never established.**

| # | clip | heard (candidates) | today's subject | V3 subject |
|---|---|---|---|---|
| 20 | dog_barking_01 (place: science lab) | Whoosh 0.36 | Air rushing | **A person swinging a sword** |
| 27 | movie_bladerunner (a city) | Whoosh 0.56 | Wind blowing | **A person swinging a sword** |
| 29 | mv_basement_scene_noise (storage room) | Siren 0.31 only | Siren blaring | **Police car with siren** |
| 39 | rx_phone_call_bad_news (dark house) | Thunk 0.58 | Fist hits wall | **Heavy door slamming shut** |
| 41, 53 | war series / thriller | Smash, crash | Glass smashing | **A person smashing a glass** |

Causes: DEPICT_V3 says "if what the detector heard is the name of a sound rather than of a thing, name
the thing that makes it" with no place to tell it which thing, so it guesses; "if a word could mean
something else, add the word that says which kind" turns Siren into a police car; "the whole thing a
person would point at as making it" adds a person to a smash.

**(b) V3 loses what the scene established.**

| # | clip | heard | today | V3 |
|---|---|---|---|---|
| 10 | b3_chicken_coop (farm) | Vehicle 0.42 | **Tractor driving** | A car driving down the road |
| 19 | cobblestone road traffic (parking lot) | Vehicle 0.59 | Car engine idling | A car driving down the road |
| 27 | bladerunner (a city) | Whoosh | Wind blowing | a sword |

Today's subjects came from the v1 kind question on the frames ("who or what is making it") and from
the place in the depiction prompt. V3 removed both, by design, to stop places being painted in.

Also noted, not failures: #45 (a police-car clip) fired Police car 0.854 and Ambulance 0.722, a tie at
margin 0.8, so the source became "Emergency vehicle" — and DEPICT then wrote "police car" anyway (right
by luck, and exactly the specificity the tie rule forbids). #9 chicken coop: Chicken/rooster 0.71, Cluck
0.60, Crowing 0.43 → source Cluck → "a chicken clucks" (the rooster case works as designed when the
detector fires the sub-label).

The N arm (V3 subjects, Qwen-Image) is being drawn now for the record, **not yet shown to anyone**.

## Round 1 answers, merged

1. **Ends** — all five: no further end work; ends already follow the sound (−0.06 s specs, −0.25 s
   panel, 1/16 lingering). The bell is a gate problem (stretch vote, end-relative grid), booked as the
   cost of a principled change. Later, as its own DEV change: onset-anchored fixed 5-s stretches (P1).
2. **Fair baselines** — all five: re-render BLIND (and CAPTION) on DEV with the onset rule on and the cap
   off. **Submitted** (jobs 31103904/5). TEST baseline render = a further look → Adam's yes + a written
   rule first. Until then: gate vs blind is the 09-22 like-for-like result at old timing, never combined
   with the new-timing gain. P2: *rendering* baselines on TEST is not a look, *scoring* is — render now
   under the frozen config, score only with Adam's yes and a written "no decision hangs on it". P2 also
   quotes the 23 Sept single TEST look: gate vs blind precision +0.143 [+0.034, +0.264], viewer cost
   +0.73 [+0.20, +1.33], F1 null; and wants a slide listing every TEST read.
3. **Judge describer** — split: GLM-4.6V-Flash (P3, P5), Gemma-4-31B as describer (P1, P2 — GLM would
   read checker-approved pictures with the checker's own eyes), no describer — Gemma-4-31B
   judges the rendered panel directly, with a template reference built from the annotator's ticks, no
   LLM (P4). P3: the reference is currently written by the describer's text side from the *detector's*
   events. Everyone: rerun B1/B2 on the 139-clip `v4b4` renders (already exposed), plus a B3 repeat.
4. **Pictures** — all: build RESOLVE (qualifier only, never a new noun) now; no best-of-N until the
   checker is calibrated; no new main generator. P3, P5: seed noise (25–30 %) is the largest threat —
   draw all arms at 3 seeds.
5. **Supervisor** — three claims: the trace-proven timing fix (DEV selected; TEST under frozen rule);
   pictures beat silence (+0.29); the gate cuts wrong pictures vs blind at a recall cost, F1 null, old
   timing both arms. Pictures and judge: "one rater", "provisional".
6. **One item** — seeds (P3, P5); the describer-free judge chain (P4); gate stability under frame shifts
   (P1); one final like-for-like reporting render — all arms, 139 clips, one frozen config, one tag,
   three seeds, Gemma evaluation, declared in advance with no decision hanging on it (P2).
7. **P2, forking paths on pictures:** once Adam's round-2 answers are opened those 50 clips become
   picture-DEV; reserve a fresh clip set now, before opening, and run RESOLVE only there.

## Proposal for decision: V3.1 = V3 + the scene, with guards (to be drawn as arm N1)

Built behind a switch `PICTURE_SCENE` on top of `PICTURE_V3`; V3 stays reproducible.

1. **RESOLVE** (one vision call per sound, on the frames already sampled for that sound's visibility vote,
   plus the clip's place): "A sound detector heard {source} ({chain}). It may be out of view. Name what
   is making it, at most 4 words. If the frames show it, name that. If it is out of view and this place
   makes one kind of {source} far more likely than any other, name that kind. If {source} is the name of
   a sound rather than of a thing, name the thing in this place most likely to make it. Otherwise answer:
   unknown."
2. **Accept the answer only if** (mechanical, no model judgement):
   * it names no ontology **sibling** of the source and no **descendant** the detector did not fire
     (the answer-sheet lists, used as a guard) — so no police car from a bare Siren, no choosing inside a
     tie;
   * its head noun is the source's own name, a fired candidate's name, or a thing the place names as a
     kind of the source, confirmed by a text yes/no asked in **both orders** ("Is a tractor a kind of
     vehicle?") — or, for a sound-name source (Whoosh, Thunk, Smash), a thing that makes that sound,
     confirmed the same way;
   * otherwise the source stays as it is.
3. **DEPICT_V3.1** = DEPICT_V3 given the RESOLVE answer as the thing, with three wording fixes: no "add
   the word that says which kind" (that invents kinds); "show a person only if the sound comes from a
   person's own body or voice"; for a sound-name source with no accepted RESOLVE answer, draw the
   visible effect of the sound (shattered glass, a swinging door) rather than inventing a maker.
4. **The same mechanical guard** runs on the DEPICT output: a sibling or unfired descendant in the
   phrase → one retry with "do not name a more specific kind than {thing}", then the family name.

**The forking-path problem with V3.1 (P2's point 7, sharpened):** the V3.1 fixes above were written
after reading V3's *subjects* (text only — no picture drawn, no rating seen) on these same 50 clips. So
N1's gain on these 50 is partly in-sample. The unannotated pool is thin: all `mixed` and `unseen_ambient`
clips are already in the 50; what is left is `seen_ambient` / `no_ambient`, which rarely have an
off-screen sound to draw.

## Questions (at most six lines each)
1. Sign or amend V3.1 points 1–4. Is point 2's second bullet (the model's own yes/no, both orders)
   honest enough, or should a scene-named kind be allowed only when it is also a fired candidate?
2. The round Adam will rate: arms today / N0 / **N1 (V3.1)**, dropping N (V3), or all four? His time is
   the scarce thing (≈160 cards for three arms with repeats).
3. The describer: pick one of GLM / Gemma-as-describer / no describer (Gemma sees the panel, template
   reference). One line why.
4. Seeds: 3 seeds for which arms, and is seed 0 still "the picture shown" for the rating?
5. Given the forking-path problem: rate N1 on these 50 anyway with the in-sample caveat, or hold N1 for a
   fresh set (which set? the gold DEV 49 is where every earlier rule was written; the 30 slice-B clips?),
   and rate only today / N0 / N now?
6. Anything in this merged plan you would stop.
