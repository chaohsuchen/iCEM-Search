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
