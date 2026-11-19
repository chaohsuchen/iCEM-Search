# RC03 representative presets (chosen from the surrogate scan; used for the `presets_22mps` batch and checked in CARLA)

Scan: `tools/rc03_scan.py` (short iCEM per family, nominal cost). Results: `results_search/rc03/scans/scan_d0_x_esc_gap.json` and `scan_d0_x_intrude.json`, both at v_ego 18 and 22 m/s.

## What drives recoverability in RC03
1. `d0` (distance to the spinning car at the cue): dominant. d0 <= ~22 m: braking fails at both speeds; d0 >= ~28-35 m: braking succeeds.
2. `intrude` (dimensionless: victim rest position = (1 - intrude) x lane width from the ego lane centre on its own side; 0.3 = victim mostly in its own lane, 1.0 = victim rests ON the ego lane centre line, yaw 0.8 rad so it also sticks ~2.4 m into the escape side): decides whether a lane change can work at all. 0.3: lane change succeeds at every d0 tested; 0.6: usually fails (clearance ~0.7-1.0 m); 1.0: fails (the victim blocks the whole ego lane and part of the escape lane).
3. `v_ego` (18 vs 22): second order here; shifts the braking boundary slightly (22 m/s fails at d0=22 with contact).
4. `esc_gap` (free window in the escape lane): almost no effect while the lane is free (14-40 m all the same); matters only through the occupied-lane variants (tight_gap, braking_lead, tailgater).

## Presets (v_ego 18 first, repeat at 22)
| id | d0 | intrude | escape lane | expected | why it is interesting |
|---|---|---|---|---|---|
| easy | 50 | 1.0 | free | brake OK, lane change fails | plain stopping works; steering is not needed |
| both | 35 | 0.3 | free | brake OK, lane change OK | two feasible strategies |
| steer_only | 22 | 0.3 | free | brake fails, lane change OK | braking is too late, steering becomes necessary |
| steer_blocked | 22 | 0.3 | tight_gap / tailgater / braking_lead | both fail or mitigate | the one remaining option is closed |
| hard | 22 | 1.0 | free | both fail (mitigate/contact) | unrecoverable reference |
| brake_boundary | 22 -> 28 (bisect) | 1.0 | free | flips fail -> success | recovery boundary in d0 |
| lc_boundary | 35 | 0.3 -> 0.6 (bisect) | free | flips success -> fail | recovery boundary in intrude |

## Open points
- d0=28, intrude=0.6 gives brake FAIL with positive margin while d0=28 intrude=1.0 gives brake success (non-monotone). Check in CARLA whether this is real (victim yaw/footprint geometry) or a surrogate artefact before using it.
- Speed range is kept at 18/22 m/s (decision: no extension to 28 m/s for now).

## CARLA check of the 18 m/s presets (batch presets_18mps, 24 runs; batch deleted in the clean-up, numbers below kept for the record)
- Surrogate PASS -> CARLA PASS in 11/11 (no false-safe). Surrogate FAIL -> CARLA PASS in 3 brake cases at d0=22 (steer_only, steer_blocked, hard): CARLA stops 2.1 m short; the surrogate was too conservative (padding applied to every actor, and its 1 g braking check counted the low-speed stop snap). Fixed in `search/rc03_sim.py` (pad only on actors ahead, snap ticks ignored) -> in-sample on all 58 verified rows: false-safe 0, false-unsafe 2.
- Consequences for the presets: at 18 m/s braking with the optional lateral `shift` passes even at d0=14 when intrude=0.3, so "brake fails, steering needed" does NOT appear at 18 m/s. It appears at 22 m/s: d0 14-22, intrude 0.3 -> brake fails, lane change passes at d0=22. `hard` = intrude 1.0 at 22 m/s (both fail). `steer_blocked` with esc_gap=12 does not block the lane change in CARLA (clearances 1.7-3.8 m): needs a smaller gap or `lead_brake`.
- Revised proposal: run the presets at 22 m/s (steer_only d0=22 intrude 0.3; hard d0=22 intrude 1.0; both d0=35 intrude 0.3; easy d0=50 intrude 1.0) and use esc_gap<=8 or lead_brake=4 for steer_blocked.

## Boundary exploration (v22, results_search/rc03/{search,carla}/boundary_r3, 55 runs)
Out-of-sample after the surrogate fixes (measured throttle->accel table, v_bias=-0.5, controller integrator start transient it0=0.25): surrogate PASS 21 -> CARLA PASS 21 (false-safe 0%); surrogate FAIL 34 -> CARLA PASS 5 (conservative near the boundaries).
CARLA boundaries (optimised policy per point): brake vs d0: fails <= 21 m, passes >= 24 m (surrogate boundary ~25 m); lane change vs intrude: passes up to 0.86, fails at 1.0; lane change vs esc_gap: passes >= 15.5 m and at 13 m, fails at 10 m and below (the 15 m point failed: non-monotone, policy-dependent).
Earlier iterations (boundary_r1, boundary_r2; deleted, superseded by boundary_r3) had false-safe 27.8% / 26.3% with older surrogates; their CARLA labels remain valid data (see results_search/summary/carla_runs.json).
