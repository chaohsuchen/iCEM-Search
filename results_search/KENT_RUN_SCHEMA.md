# Run folder schema (`rcx-run-1`, Kent's deliveries)

One folder per policy executed once in one scene: `<rc>/carla/<batch>/runs/<NNN>_<variation>_<family>_<kind>/`. Written by `carla_rc/teachers/evidence.py`; CARLA 0.9.15, 20 Hz ticks (0.05 s), the policy decides at 10 Hz.

**Compression in this repository.** `ground_truth.csv`, `ego_observable.csv` and `decisions.jsonl` are stored gzipped (`.csv.gz`, `.jsonl.gz`); `summary.json` is plain. Numbers with more than 7 significant digits are written at float32 precision, CARLA's own (relative change < 5e-7; columns, rows and text unchanged). Read them with `tools/kent/runs_io.py` (either form), or directly: `pandas.read_csv("ground_truth.csv.gz")`, `gzip.open("decisions.jsonl.gz", "rt")`. Repeats near the boundary (R6) are under `<rc>/carla/<batch>/repeats/runs/`.

## Frames, units, signs

- **Scenario frame** (every `*_s`, `*_t`, `ego_*` field): `s` / `x` along the site's reference lane centreline [m], from the site's frame origin; `t` / `y` lateral [m], **positive to the right**; yaw [rad], **positive turning to the right** (towards +t), 0 = along the lane. On a bend the frame follows the centreline.
- **Reference point:** every position is the vehicle's **footprint centre** (bounding-box centre). The ego is 4.9 m × 2.1 m (EGO_L × EGO_W). The driver's eye (line of sight) is 0.4 m forward and 0.4 m left of the centre.
- **Relative state** of an actor (`seen_*`, `actor_*`): `s`, `t` = actor centre − ego centre; `yaw_rel` = actor yaw − ego yaw; `vs`, `vt` = actor velocity − ego velocity, both in the scenario frame [m/s].
- **Time:** `sim_time` [s] is 0 at t0, the end of the pre-roll; the pre-roll (`phase = prefix`) is recorded from `-prefix_s` (`summary.json` → `scenario.prefix_s`). `frame` is CARLA's frame number. `t_since_cue` is the time since the policy's trigger (the decision point), empty before it.
- **CARLA world** (`ego_world_*` in `ground_truth.csv`): Unreal coordinates [m]; yaw in degrees.
- **Controls:** `throttle`, `brake` ∈ [0, 1]; `steer` ∈ [-1, 1], positive = right (CARLA). Requested `a_req` [m/s²] (positive = accelerate) and `r_req` = yaw rate [rad/s] (positive = right).

## summary.json

| field | meaning |
|---|---|
| `label`, `reason`, `end_reason` | CARLA's verdict (PASS / FAIL / INVALID), why, and how the run ended (`horizon`, `contact:<actor>`, `collision_sensor:<actor>`, `left_corridor`) |
| `constraints` | **R1:** every constraint the label was judged on: `name`, `measured`, `required`, `unit`, `passed`, `definition` (clearance, road, friction, terminal, no_contact with the impact speed, goal, envelope). PASS = all passed; `pass_rule` restates it. |
| `metrics` | margin `r` = min(clearance, road, friction, terminal margins) and each minimum, speeds, progress, envelope statistics, the scene's goal details |
| `contact` | frames, actors, `impact_speed` [m/s] = closing speed at the contact |
| `scene` | **R3:** `name` (the scene, e.g. `blowout_onramp`, independent of the RC number), `uid` (= `variation_id`), `site`, `split` (train / heldout), `history_id` (scenes differing only in hidden settings share it), `hidden_world` (null for the searched scene), `hidden_params` (**R2**) |
| `scenario` | scene id (`sid`), town, site frame, lane width, horizon, `prefix_s`, description, every scene parameter (`params`) |
| `policy` | family, all its parameters, trigger time and hand-back time / choice |
| `candidate_kind` | `best` (the family's most robust policy), `boundary` (its smallest-margin recovery), `shared` (**R4:** the family's first fixed setting, run in every scene), `ref` (a sweep's starting policy kept fixed) |
| `hidden_params` | names of the scene parameters hidden at the decision point |
| `surrogate_prediction` | the surrogate's verdict for this candidate (PASS / FAIL / UNCERTAIN, or WORLD for a hidden-world run), margins, worlds |
| `policy_source`, `scene_seed` | how the policy was found (optimizer, rng) and how the scene was drawn |
| `observation` | what the ego is told: line of sight, 0.3 s track hold, no delay |

## ego_observable.csv (world-model input), one row per tick

`frame, sim_time, t_since_cue, phase` (prefix / before_trigger / stage1 / handback); ego `ego_s, ego_t` (lateral from the ego's own lane centre), `ego_yaw_rel, speed` [m/s], `accel_long` [m/s², filtered], `yaw_rate` [rad/s], `accel_long_f, accel_lat_f` (measured, envelope monitor); requested `a_req, r_req`; applied `throttle, brake, steer`; `contact` (1 on the contact tick); `obs_*` the policy's observation vector (names in the spec's `obs_names`); per actor `seen_<actor>_visible` (line of sight this tick) and, when visible, `seen_<actor>_{s,t,yaw_rel,vs,vt,length,width}`. Background traffic (`bg_*`) is never delivered.

## decisions.jsonl (world-model input), one line per decision (0.1 s)

`frame, query_time, t_since_cue, observation` (named), `observation_history_frames` (frames of the history the policy saw, last 10) and `observation_history_len`, `policy_output` (phase, family, envelope action z, a_req / r_req, hand-back choice), `controller_state`, `applied_command` (first tick), `pending_commands` (every tick the command was held over).

## ground_truth.csv (analysis and labels only: hidden information), one row per tick

`frame, sim_time, contact_actor`; the margins `m_clear` (clearance with reserve) `m_road` `dyn` and the nearest actor; corridor bounds `bound_lo, bound_hi` (relative to the home lane centre); ego `ego_x, ego_y, ego_psi, ego_v` (scenario frame) and `ego_world_x, ego_world_y, ego_world_yaw`; every actor's true relative state `actor_<n>_{s,t,yaw_rel,vs,vt,length,width}` and `actor_<n>_clear` = signed footprint distance to the ego [m] (negative = overlap; within 30 m only).

## Batches and repeats

- Hidden worlds (**R5**): a candidate run in another hidden world of its history has `variation_id` `<history>~<world>` and `scene.hidden_world` set; its surrogate verdict is `WORLD` (not part of the safety score).
- Shared policies (**R4**): every family's first fixed setting runs in every scene and hidden world. When it coincides with the family's `best` or `boundary` policy it runs once, under that kind; `carla/<batch>/shared_runs.json` lists every run that is the shared setting (`{run_id: family}`).
- Repeats near the boundary (**R6**): `carla/<batch>/repeats/runs/<run>_x<dx>_v<dv>/`, the same candidate with the ego's start shifted (CARLA itself is deterministic for a given start).
- No `run.mp4` (camera off).
