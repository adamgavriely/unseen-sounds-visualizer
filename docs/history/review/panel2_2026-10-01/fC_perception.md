# Panel 2 — Fable C (perception / system), 1 Oct

System finding: the gold applies two human rules the pipeline never encodes: (a) "obvious" = the sound EVENT is inferable from the picture, not "the source object is on screen" (bell in a visible tower, fart from a visible dog = not obvious; thunder under visible rain = obvious); (b) a same-family return after a > 2-s pause is a NEW event (12/58 needed are returns; D hits 3; 7 of D's 29 misses have a same-family picture drawn elsewhere).

**1. CONCEALED-ACTION families (first pick).** Fixed ontology list the visibility gate may not silence: families whose sound-producing action is hidden inside the visible source (Bell, Church bell, Bicycle bell, Change ringing; Digestive: Fart, Burping, Hiccup, Stomach rumble; not Cowbell/Jingle bell/Chime/Tuning fork/Chewing/Biting/Gargling). Exemption at the family's stage-5 verdict (both kill paths: kinship propagation and VLM vote). Targets bell_miami Bell, tg_d133 Fart x2. Evidence outside DEV: Bell 2 needed silenced, 0 seen silenced on gate-gold calibration; Digestive none outside DEV2. 0 GPU.

**2. PERCEPTUAL-ONSET RETURN.** Signal-level onset-strength peak (clip-relative) with sub-bar same-family evidence within ±0.3 s and >= 2.0 s from the nearest same-family picture -> a stage-4 row exempt from group_bursts, MERGE_GAP and GROUP. Targets tg_d032 Thunder 7.4, as_explosion Explosion 2.8, birds_forest Bird 1.3, tg_d120 Meow 2.9. 415 STOP guard on strong-label returns. ~1 s CPU per video.

**3. CO-OCCURRENCE WITNESS.** P(family | the clip's own confirmed families) from AudioSet-Strong (minus screening ids) as a VLM-free one-ear witness; 415 one-ear population binned by lift.

Checked, dead: event-vs-texture gate exemption; top-k salience; clip consensus -> timing; within-clip self-prior; tiered output.
