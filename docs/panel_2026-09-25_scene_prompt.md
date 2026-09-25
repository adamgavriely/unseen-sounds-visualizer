# Panel of five, three rounds — the generator, and a scene-aware prompt written by the VLM (2026-09-25)

Adam: *"which image generator can we use among SOTA models, AND how can we refine the prompt to be more
specific — maybe let the VLM generate the prompt? If fire is heard and the scene is a forest, it could
generate fire crackling in a forest."* Also: *"most images are very bad in my opinion"* and *"clearly a VLM
could see most images here are bad; no need for a human to say this."*

Reviewers: G1 (generators, SOTA), G2 (visual communication, prompts), P2 (statistics, forking paths),
P4 (automatic evaluation), P5 (deaf and hard-of-hearing viewers). Read-only; repo `P:\MscProj`.

## Where we are (all measured)

* Adam's blind round 2 (54 sounds, 50 never-annotated clips, no name shown, he typed what he saw):
  FLUX today 14/54 right; Qwen-Image-2512 with today's text 26/54; Qwen-Image + V3 text 32/54 (+33 points vs
  today, significant; the generator gave +22 of it). V3 is vetoed by one false message (a Thunk drawn as a
  door). Failures: "can't tell" on sounds with no single object (whoosh, thunder, car alarm, smash, rain on a
  surface, train horn); "vague" on bird actions read as "bird"; misreadings (sheep as goat).
* A 2-reviewer generator panel (G1, G2; `docs/panel_2026-09-25_generator.md`) signed plan **GP-2**: compute
  gate, then screen HiDream-O1-Image (MIT) / Qwen-Image-2.1 (research licence) / Mage-Flow (MIT, gated)
  against Qwen-Image-2512 on the 54, prompt arms on the winner (whole thing in frame, mid-act with its
  visible effect, one large subject; guarded LLM expansion), fixed template subjects for sounds with a
  canonical maker (thunder → lightning bolt), a burst-with-word card for sourceless sounds, rule 2 widened
  to unnamed inventions, confirmation on 50 fresh unannotated clips, pass bar written first.
* Automatic judges: blind "what is this?" checkers failed against Adam (GLM kappa 0.45; the per-picture
  judge said "can't tell" to 502/567). A **named yes/no verifier** ("this should show X — would a deaf
  viewer recognise it at a glance?") is being calibrated right now on Adam's 170 round-1 yes/no answers
  (`docs/prereg_verifier.md`); if it passes, it screens generators instead of Adam.

## Why the place was removed from the picture (history the panel must weigh)

The depiction step used to get the *place* ("palace", "street") and the generator painted it: two crowds in
front of a palace, a fire engine placed in flames, scenery instead of the source; the place strips then had
to scrub it. The current rule: no place in the subject, a plain white background, a negative prompt against
scenery. The scene is still used — V3.1 lets it add one qualifier to the heard word ("car door", "farm
vehicle") under a list guard — but never as a background. The picture sits **beside** the video, whose scene
the viewer already sees.

## Adam's proposal

Let the VLM (it sees the clip's frames) write the image prompt itself, specific to the scene: fire heard in a
forest → "fire crackling in a forest"; possibly a longer, richer prompt (G1: these generators were trained on
40-80-word captions, so a 5-word prompt is out of distribution).

## Round 1 questions (at most eight lines each; your lens first)

1. **Generator:** does GP-2's candidate list stand, or is there a better SOTA choice now? (G1 leads.)
2. **Scene in the picture:** should the picture include the setting (forest fire vs a generic fire), and if
   so how much — a background, a small hint, or only a qualifier word? What does it cost a deaf viewer who
   already sees the scene beside it? (G2, P5 lead.)
3. **VLM-written prompt:** should the VLM (with the frames) write the whole image prompt? How do we stop it
   inventing a source the audio never heard (the sword, the police car, the door) — which guard, which
   check? (G1, G2, P4.)
4. **Measuring it without Adam where possible:** can the named verifier (if it passes) or another automatic
   check screen the prompt variants, and what must still be a human? (P4, P2.)
5. **Forking paths:** we have now changed the picture rules several times on the same clips. What is the
   cleanest order and set of clips so the final claim holds? (P2 leads.)
