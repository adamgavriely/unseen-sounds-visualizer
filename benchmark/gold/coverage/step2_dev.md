# Step 2, DEV (71 clips)

| cell | hits | wrong | onset cost | cost_cov | hit cover | wrong s/clip | stale s/clip | pass (>= 31 hits, <= 16 wrong) |
|---|---|---|---|---|---|---|---|---|
| D' | 29 | 15 | 2.056 | 2.509 | 0.72 | 1.11 | 0.12 | no |
| AB-m | 32 | 16 | 1.915 | 2.421 | 0.72 | 1.29 | 0.13 | yes |
| AB-s | 32 | 21 | 2.056 | 2.561 | 0.72 | 1.47 | 0.13 | no |

passing: ['AB-m']; selected: AB-m

Clips that change against D' (hits before -> after, wrong before -> after):
- AB-m: ambient_transport_subway_10800 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_protest_scene_movie 3->3 / 1->2; tg_d127 1->1 / 1->0; tg_d133 0->2 / 0->0
- AB-s: ambient_transport_subway_10800 0->0 / 0->1; bell_miami 0->1 / 0->0; mv_arrest_street_scene 0->0 / 0->1; mv_protest_scene_movie 3->3 / 1->2; tg_d054 0->0 / 0->1; tg_d085 0->0 / 0->1; tg_d088 1->1 / 2->4; tg_d127 1->1 / 1->0; tg_d133 0->2 / 0->0
