# Pre-registration: Step 11, two display policies and a flash rule (DEV only, CPU)

Written and committed 2026-10-08 before any of them was computed. Base = the a/b candidate (32 hits / 16 wrong); each
policy also re-applied on top of Step 10's V4 fix if that passes. TEST untouched. Every result also reports hits split by
label provenance (gold field `from_detector`: the annotator started from a detector suggestion, or not).

## T. Texture-parent ban

Generic continuous-texture families, fixed now: Vehicle, Water, Engine, Rain, Wind, Liquid, Mechanisms,
Domestic sounds, home sounds. A picture of one of them is removed unless Audio Flamingo Next's V4 open list on an
overlapping P1 cut (listener_p1v4 `af_fams`; Qwen's list is the buggy one) names a strict descendant family of it.

## A. Common-ancestor picture

Sibling groups and their parent, fixed now: {Gunshot, Explosion, Artillery fire, Machine gun} -> Explosion;
{Siren, Shofar, Alarm, Air horn, truck horn} -> Alarm; {Honk, Gull, seagull, Bird, Duck} -> Bird. When a picture's
family is in a group and any DEV candidate (decision trail, any origin, kept or dropped) of ANOTHER family of the same
group starts within 1 s of the picture's start, the picture is drawn as the parent instead (times unchanged).

## F. Visible flash

Per-frame mean luminance (Y) of each DEV clip at its full frame rate (OpenCV). Flash = a frame whose Y exceeds the
median of the previous 1 s by more than 5 robust sigma (1.4826 x MAD of that second; floor 2 grey levels), in a run of
at most 4 frames. Rule: a Thunder picture is silenced if a flash lies in [onset - 2 s, picture end]; an Explosion /
Fireworks / Gunshot, gunfire / Machine gun / Artillery fire picture is silenced if a flash lies within +-0.3 s of its
onset. Flash counts per clip are reported (camera cuts and headlights show up there).

## Runs and pass bar

T, A, T+A, F, and T+A+F on top of the a/b candidate. Pass: wrong reduced by >= 2 with hits down by <= 1
(F: wrong <= 13 with hits >= 31).
