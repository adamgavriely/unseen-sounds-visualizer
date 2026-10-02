# Round 52 STRONG-KEEP near-miss: why it lost tg_d120 Cat and gained 3 wrongs (1 Oct, CPU, DEV2 caches)

Rebuild: stage 4 of `SHIP8+MD3` and `SHIP8+MD3+SK7` re-run on the login node (`round13_dev.build -> reloc_rows -> filter_rows`,
arm configs taken from the cached `r13dev2/stage4.json` `cfg`). The rows match the cached rows. Stage-5 specs come from
`<ARM>_proposed/<clip>/augmentations.json`. Evidence is measured over the span `[pre_start, end)`. DASM is checked at onset ± 0.5 s.

| clip, picture | stage-4 row (label, start, end, BEATs conf) | veto skipped by SK7 | FlexSED own family (span / clip) | FlexSED top other family in span | DASM own at onset (top) | result |
|---|---|---|---|---|---|---|
| tg_d029 rooster | Chicken, rooster 6.75-11.0 0.813 (+Fowl 0.794, Cluck 0.741) | mirror (and cross behind it) | Bird 0.258 / 0.258 | **Goose 0.852**, Honk 0.83, Gasp 0.80 | 0.073 (Cattle 0.27, Goose 0.25) | +1 hit, Bird 6.89 |
| tg_d120 Cat | Meow 2.25-6.75 0.830 | cross (Cat clip max 0.145 < 0.3) | Cat 0.134 / 0.145 | Dog 0.577 (< 0.7, mirror silent) | 0.891 (Cat) | -1 hit (see below) |
| tg_d022 Pant | Pant 0.22(0.0)-1.5 0.862 | mirror | Pant 0.389 / 0.663 | Alarm 0.80 | 0.148 (Alarm 0.30) | +1 cross |
| tg_d032 Thunder | Thunder 8.75-9.5 0.727, Thunderstorm 0.704 | mirror | Thunder 0.395 / 0.73 | Bicycle 0.819, Rain 0.813 | 0.013 (Water 0.42) | +1 cross |
| tg_d128 Cattle | Cattle 1.5-9.25 0.988, Moo 0.980 | mirror | Cattle 0.069 / 0.069 | Laughter 0.916 | 0.768 (Cattle) | +1 visible |

**tg_d120: why the Cat picture moved from 0.56 to 2.25.** It was not merging, relabelling, an onset change or dedup.
The cause is the DASM rescue rule DR2 ("new families only"). In the base arm, the cross-detector veto drops Meow 0.83 and
Cat 0.668. That leaves the clip with no Cat span, so the DASM rescue adds Cat at 0.56, 2.94 and 3.94 (DASM peaks 0.89 / 0.92 /
0.95; both listeners name it). ONCE then keeps the earliest one, 0.56-2.24, and that picture is the hit. In SK7, Meow (0.83 >= 0.7)
skips the cross veto. Now Cat is already "present", so `DASM_RESCUE_NEW_ONLY` skips all three rescues. The only Cat left is
BEATs' own Meow row, and its onset is 2.25. Stage 5 maps it to Cat 2.25-6.75, which is scored phantom. In short, a kept span
with a late onset pushed out a rescue with a better onset. STRONG-KEEP changed what a later rule counts as "present".
The other changes: tg_d032 joins the kept 8.75 span and the base's 13.75 Thunder into one picture (spans
[[8.75, 9.5], [13.75, 14.75]], conf 0.727), so its onset moves to 8.75. tg_d128 Cattle is a real sound (DASM 0.77) that the
gate called not visible. That is a gate miss, not a detector error.

**What separates the rooster from Pant, Thunder and Cattle at stage 4.** BEATs conf does not (Pant 0.862 and Cattle 0.988 are
both above the rooster's 0.813). FlexSED own-family max does not (the rooster is the lowest, 0.258). DASM does not (the rooster
is the lowest, 0.073). Listener names cannot be used: the rooster and Cattle have no P1 item ("missing"), and the yes/no
lookup changes with the start (Pant: false at 0.0 in base F7, true at the refined 0.22). That is unresolved, so nothing is
built on it. **The one separator is ontology kinship of FlexSED's winning query.** For the rooster, Goose and Honk share the
non-root parent **Fowl** with Chicken and Cluck. FlexSED heard the same source under a sibling name. The "other family" only
exists because of the canonical map (Chicken -> Bird, Goose -> Goose). Pant/Alarm and Cattle/Laughter share no ancestor.
Thunder/Bicycle share none, and Thunder/Rain share only the root "Natural sounds".

**Proposed refinement: KIN-KEEP.** A span with BEATs conf >= 0.7 skips the mirror veto and the cross-detector veto **only if**
FlexSED's top other-family query in the span is >= 0.7 (the mirror bar) **and** is ontology kin of the span's label. Kin means
the two labels share an AudioSet ancestor that is not one of the 7 top-level roots. Checked on the five DEV2 cases:
- rooster: kept (Goose 0.852, Fowl). Hit stays.
- Meow: not kept, because no query reaches 0.7 (Dog 0.577). The cross veto drops it as in base, the DASM rescue at 0.56
  comes back, and the Cat hit stays.
- Pant, Thunder and Cattle: vetoed as in base (competitor is not kin). No new wrongs.
- The 3 "longer span, same score" cases are expected to go back to base, but this is not checked.

**How the numbers are fixed on the held-out 415 (not DEV).** The two bars are inherited and not tuned: 0.7 = STRONG_BEATS_KEEP
= MIRROR_VETO. The only free choice is the kin depth: either "non-root common ancestor" or "same direct parent". It is
pre-registered and chosen on the 415. Step 1 re-runs `strongkeep_415.py` on its 64 proxy spans (27 correct, 0.422) and splits
them into kin-kept and the rest. Pass bar: kin subset precision >= 0.375 (raw BEATs on the 415), with n >= 10, and the non-kin
rest must sit below 0.375. Supporting sign: the cross-only part, the Meow type, was already 4/17 = 0.235. Choose the kin depth
by kin-subset precision. Ties go to the stricter one, because `labels.same_source` warns that a shared parent can join
Dog/Sheep, and Meow/Dog share "Domestic animals, pets". DEV is run once, only after Step 1 passes, with the Round 52 rule.
TEST is not read.
