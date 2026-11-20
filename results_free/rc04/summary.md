## Surrogate stage (iCEM, no CARLA)

| scene | role | struct best (brake / lane_change / staged) | free supported / unique samples | of which warm / cold | diverse kept | min pair dist | free best robust cost | struct best robust cost | kept supported robust over worlds |
|---|---|---|---|---|---|---|---|---|---|
| near_obstacle | all_families_refuted | FAIL / FAIL / FAIL | 0 / 2189 | warm 0 / cold 0 (kept) | 0 | nan | 1013.6 | 1017.0 | 0 / 0 |
| obstacle_40 | all_families_refuted | FAIL / FAIL / FAIL | 0 / 2189 | warm 0 / cold 0 (kept) | 0 | nan | 1013.6 | 1017.3 | 0 / 0 |
| obstacle_44 | all_families_refuted | FAIL / FAIL / FAIL | 0 / 2189 | warm 0 / cold 0 (kept) | 0 | nan | 1013.7 | 1017.0 | 0 / 0 |
| no_follower | multi_family_supported | PASS / PASS / PASS | 1261 / 2189 | warm 4 / cold 1 (kept) | 3 | 1.36 | 1.2 | 1.0 | 4 / 5 |
| easy | multi_family_supported | PASS / FAIL / PASS | 711 / 2189 | warm 1 / cold 4 (kept) | 3 | 0.72 | 1.0 | 0.9 | 1 / 5 |
| truck_far_ahead | mixed_near_boundary | FAIL / PASS / PASS | 831 / 2189 | warm 1 / cold 4 (kept) | 3 | 1.23 | 1.0 | 0.6 | 5 / 5 |
| brake_boundary | mixed_near_boundary | PASS / FAIL / FAIL | 157 / 2189 | warm 4 / cold 0 (kept) | 2 | 0.60 | 57.0 | 68.8 | 2 / 4 |
| lane_change_boundary | mixed_near_boundary | FAIL / FAIL / PASS | 624 / 2189 | warm 2 / cold 3 (kept) | 3 | 0.73 | 1.8 | 1.2 | 5 / 5 |

Budget per scene: free 1446 cold + 810 warm robust evaluations; structured families about 1350 (3 families x 2 seeds), each robust evaluation = 5 surrogate rollouts.

## CARLA stage

CARLA-run candidates: free 159 of 159, structured 90 of 90

### Surrogate transfer (false-safe = surrogate PASS, CARLA not PASS; false-negative = surrogate FAIL, CARLA PASS; INVALID/missing excluded)

| family / stratum | surrogate PASS: n | false-safe | 95% CI | surrogate FAIL: n | false-negative | 95% CI |
|---|---|---|---|---|---|---|
| free random_prior | 9 | 6 (67%) | (0.354, 0.879) | 23 | 0 (0%) | (0, 0.143) |
| free iCEM-selected (w0) | 24 | 9 (38%) | (0.212, 0.573) | 17 | 1 (6%) | (0.01, 0.27) |
| free world replays | 73 | 32 (44%) | (0.33, 0.552) | 13 | 7 (54%) | (0.291, 0.768) |
| structured iCEM-selected (w0) | 17 | 1 (6%) | (0.01, 0.27) | 15 | 3 (20%) | (0.07, 0.452) |
| structured world replays | 50 | 1 (2%) | (0.004, 0.105) | 8 | 3 (38%) | (0.137, 0.694) |

### Coverage per scene (nominal world w0, CARLA labels)

| scene | role | struct PASS (best+boundary) / run | free PASS (best, diverse, boundary) / run | free refuted / mitigation / random PASS | free supported in all worlds | struct supported in all worlds |
|---|---|---|---|---|---|---|
| near_obstacle | all_families_refuted | 0 / 3 | 0 / 1 | [] / ['FAIL'] / 0 of 4 | 0 | 0 |
| obstacle_40 | all_families_refuted | 0 / 3 | 0 / 1 | ['FAIL'] / ['FAIL'] / 0 of 4 | 0 | 0 |
| obstacle_44 | all_families_refuted | 0 / 3 | 0 / 1 | [] / ['FAIL'] / 0 of 4 | 0 | 0 |
| no_follower | multi_family_supported | 4 / 5 | 5 / 5 | ['PASS'] / ['FAIL'] / 0 of 4 | 4 | 4 |
| easy | multi_family_supported | 4 / 5 | 4 / 5 | ['FAIL'] / ['FAIL'] / 1 of 4 | 4 | 2 |
| truck_far_ahead | mixed_near_boundary | 5 / 5 | 2 / 5 | ['FAIL'] / ['FAIL'] / 1 of 4 | 1 | 5 |
| brake_boundary | mixed_near_boundary | 2 / 4 | 2 / 4 | ['FAIL'] / ['FAIL'] / 1 of 4 | 2 | 2 |
| lane_change_boundary | mixed_near_boundary | 4 / 4 | 2 / 5 | ['FAIL'] / ['FAIL'] / 0 of 4 | 2 | 4 |
