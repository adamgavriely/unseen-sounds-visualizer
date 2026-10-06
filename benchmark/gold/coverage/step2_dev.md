# Step 2, DEV (71 clips)

| cell | hits | wrong | onset cost | cost_cov | hit cover | wrong s/clip | stale s/clip | pass (>= 31 hits, <= 16 wrong) |
|---|---|---|---|---|---|---|---|---|
| D' | 29 | 15 | 2.056 | 2.509 | 0.72 | 1.11 | 0.12 | no |
| AB-m | 32 | 16 | 1.915 | 2.421 | 0.72 | 1.29 | 0.13 | yes |
| TL-onset+AB-m | 32 | 17 | 1.944 | 2.449 | 0.72 | 1.33 | 0.13 | no |
| TL-span+AB-m | 31 | 17 | 2.000 | 2.456 | 0.74 | 1.33 | 0.12 | no |
| AB-s | 32 | 21 | 2.056 | 2.561 | 0.72 | 1.47 | 0.13 | no |
| TL-span | 29 | 16 | 2.085 | 2.537 | 0.72 | 1.16 | 0.12 | no |
| TL-onset | 29 | 16 | 2.085 | 2.537 | 0.72 | 1.16 | 0.12 | no |
| TL-onset+AB-s | 32 | 22 | 2.085 | 2.590 | 0.72 | 1.52 | 0.13 | no |
| TL-span+AB-s | 31 | 22 | 2.141 | 2.597 | 0.74 | 1.52 | 0.12 | no |

passing: ['AB-m']; selected: AB-m

Clips that change against D' (hits before -> after, wrong before -> after):
- TL-onset+AB-m: ambient_transport_subway_10800 0->0 / 0->1; b3_laundromat 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_protest_scene_movie 3->3 / 1->2; tg_d127 1->1 / 1->0; tg_d133 0->2 / 0->0
- TL-onset+AB-s: ambient_transport_subway_10800 0->0 / 0->1; b3_laundromat 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_arrest_street_scene 0->0 / 0->1; mv_protest_scene_movie 3->3 / 1->2; tg_d054 0->0 / 0->1; tg_d085 0->0 / 0->1; tg_d088 1->1 / 2->4; tg_d127 1->1 / 1->0; tg_d133 0->2 / 0->0
- TL-onset: b3_laundromat 0->0 / 0->1
- TL-span+AB-m: ambient_transport_subway_10800 0->0 / 0->1; b3_laundromat 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_protest_scene_movie 3->3 / 1->2; tg_d127 1->1 / 1->0; tg_d133 0->1 / 0->0
- TL-span+AB-s: ambient_transport_subway_10800 0->0 / 0->1; b3_laundromat 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_arrest_street_scene 0->0 / 0->1; mv_protest_scene_movie 3->3 / 1->2; tg_d054 0->0 / 0->1; tg_d085 0->0 / 0->1; tg_d088 1->1 / 2->4; tg_d127 1->1 / 1->0; tg_d133 0->1 / 0->0
- TL-span: b3_laundromat 0->0 / 0->1
- AB-m: ambient_transport_subway_10800 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_protest_scene_movie 3->3 / 1->2; tg_d127 1->1 / 1->0; tg_d133 0->2 / 0->0
- AB-s: ambient_transport_subway_10800 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_arrest_street_scene 0->0 / 0->1; mv_protest_scene_movie 3->3 / 1->2; tg_d054 0->0 / 0->1; tg_d085 0->0 / 0->1; tg_d088 1->1 / 2->4; tg_d127 1->1 / 1->0; tg_d133 0->2 / 0->0

AB-m vs D', paired clip bootstrap of onset cost: -0.141 [-0.451, +0.085], p 0.35 (DEV, 71 clips; not significant).
The TL arms ran through the harness with every gate answer reused (stage 5 asked no new question). TL changes one clip
(b3_laundromat, +1 wrong) and no hit.
