# Per-run evidence schema (`rcx-run-1`)

Every CARLA execution of a policy writes one folder `runs/<run_id>/` (written by `rcx/common.py:run_engine` + `rcx/evidence.py`, for every RC):

| file | contents | for the world model? |
|---|---|---|
| `summary.json` | configuration, labels, metrics (see below) | metadata only |
| `ego_observable.csv` | one row per tick: what the ego knows | **yes** (only this + `decisions.jsonl`) |
| `ground_truth.csv` | one row per tick: true state of every actor, scene measurements | **no** (hidden information; labelling / analysis) |
| `decisions.jsonl` | one JSON record per decision tick | yes |
| `run.mp4` | chase-camera video with overlay | no |

## summary.json
`schema`, `run_id`, `rc` (e.g. RC03-01), `rc_case`, `variation_id` (scenario name from the search, e.g. `steer_only`), `candidate_kind` (best / boundary), `family`,
`scenario_params` (the varied scene parameters) and `scenario` (full scene parameters incl. defaults), `scene_seed`, `policy` (name, version, params) and `policy_params`,
`policy_source` (optimizer, search seed, pop, iters, robust cost of the iCEM run that produced the policy), `surrogate_prediction`, `calibration`,
`label` / `reason` / `end_reason` (CARLA verdict, reference only: final recovery-profile labels are produced downstream), `metrics`, `times`, `contact` (frames, actors), `evidence_meta`, `files`.

### Constraint block, scene name and hidden settings (added fields)
- `scene_name`: scene family, separate from the RC number: `street_van` (rc01), `spun_car` (rc03), `stopped_car_truck_alongside` (rc04), `lane_change_abort` (rc05), `broken_car_passer` (rc06), `tunnel_squeeze` (rc09), `sidewalk_pedestrian_dart` (rc13). `variation_id` is the scene variation inside it.
- `hidden_settings`: names of the scene parameters (keys of `scenario`) that are not observable in the recorded history at the decision point. Declared per case from `20RCs.md` ("Hidden at the trigger"): rc01 none; rc03 crash, intrude, psi_rest, T_slide, lead_brake, lead_react; rc04 and rc06 side_acc, side_react, fol_headway, fol_react; rc05 rear_acc, rear_vmax; rc09 rear_delay_s, rear_decel; rc13 dart_speed, dart_dist.
- `constraints`: every check the CARLA judge applied, as a list of `{name, kind, measured, required, op, unit, passed}`. `kind` is `contact` (frames in contact, required 0), `clearance` (distance in m), `limit` (filtered acceleration against the physics envelope, m/s2: forward 0.5 g, braking 1 g, lateral 1 g, planar 1 g) or `state` (yes/no end checks such as settled, stopped, not closing). `op` compares `measured` with `required`. `constraints_pass` is true when every listed constraint passed; it is null for INVALID or UNRESOLVED runs, whose `constraints` is null.
- The block is computed from the same recorded numbers as the label (`rcx/constraints.py`); `tools/backfill_constraints.py` checks that `constraints_pass` agrees with the label for every run (PASS if and only if all constraints passed). Some checks are only decisive when there is no contact (clearances, settled, not closing, stopped): a run that ends in a contact fails on `no_contact` first.

## Units, signs and reference points
| quantity | unit / sign / reference |
|---|---|
| time (`sim_time`, `t_since_cue`) | seconds; tick 0.05 s; `t_since_cue` is empty before the decision point (the tick at which the policy is first told about the hazard) |
| `ego_s`, `ego_t` | metres in the scene lane frame: s forward along the road, t lateral, positive to the right; ego lane centre t = 0 (rc03/04/06: the escape lane is at t = -side * lane_w) |
| `ego_yaw_rel` | heading relative to the road (sign convention not yet documented) |
| `speed`, `accel_*` | m/s and m/s2; `accel_long` positive forward, `accel_lat` positive to the ego's right; `*_f` = 0.2 s trailing mean |
| `a_req`, `a_req_clipped` | requested longitudinal acceleration in m/s2, negative = braking |
| `seen_<actor>_rel_*` | the other actor relative to the ego: `rel_s` forward, `rel_t` right, `rel_vs`/`rel_vt` relative velocities (m/s), `yaw_rel` rad, `length`/`width` m. Reference point of the position: see the open question in the hand-off notes (the offset to the true centre distance is a few cm) |
| `*_clear` (ground truth) | footprint-to-footprint distance in metres between the ego and that actor (rotated rectangles); 0 at contact |
| `throttle`, `brake`, `steer`, `hand_brake` | CARLA vehicle control values as applied |

## ego_observable.csv (per tick, dt = 0.05 s)
- time: `frame`, `sim_time`, `t_since_cue` (empty before the cue).
- ego state (own sensing): `ego_s`, `ego_t` (lane frame; t positive to the right), `ego_yaw_rel`, `speed`, `accel_long`, `accel_lat`, `*_f` (0.2 s mean), `accel_planar_f`, `contact` (collision sensor bit).
- policy view: `obs_*` = the exact observation dict the policy received this tick (e.g. cue, hazard).
- perception: `seen_<actor>_visible` (0/1) and, when visible, `seen_<actor>_{rel_s,rel_t,rel_vs,rel_vt,yaw_rel,length,width}` (ego-relative). Visibility rule (provisional): euclidean distance <= 80 m AND the line of sight from the ego centre to the actor centre is not blocked by another tracked actor footprint. Static map geometry (buildings) is not modelled. Hidden (invisible) actors have empty cells.
- controls: requested `a_req`, `a_req_clipped`, `t_des`, `phase`; applied `throttle`, `brake`, `steer`, `hand_brake`.

## ground_truth.csv (per tick)
`frame`, `sim_time`, `contact_actor`, per actor `actor_<name>_{s,t,yaw_rel,vs,vt,clear,length,width}` (lane-frame centre, velocity components, footprint distance to the ego),
and every scene-specific measurement (clearances, front/rear s of actors, ...). Hidden scene parameters (e.g. RC03 `intrude`, `psi_rest`) are in `summary.json: scenario`.

## decisions.jsonl (one record per tick)
`frame`, `query_time`, `t_since_cue`, `observation` (what the policy was given), `observation_history_frames` (the previous 20 frames: look up their rows in `ego_observable.csv`),
`policy_output` (phase, a_req, t_des, t_rate, lane_keep, info), `controller_state` (integrator / brake-frame counter / throttle memory / accel history of the low-level controller),
`applied_command`, `pending_commands` (commands issued during the last `delay_frames` ticks of the actuator delay window, i.e. possibly not yet effective).

## Notes
- Names of actors are the scene roles (`victim`, `esc_lead`, `esc_follow`, `rear`, ...); duplicates get a numeric suffix.
- No preprocessing, recovery-profile labelling or splitting is done here.
- The same files are written for RC01 / RC05 / RC09 / RC13 / RC03 because they share `run_engine`.
