# Reply to PIPELINE_HANDOFF.md (Victor's side, state of results_search/ after the 2026-11-19 update)

Delivered in `rcx-run-1`: 1206 CARLA runs, all with the full evidence files, for RC01, RC03, RC04, RC05, RC06, RC09, RC13 (RC02 is derived from RC01). Labels: PASS 616, FAIL 583, INVALID 7 (rc01 5, rc09 2). Index: `results_search/summary/carla_runs.json`. Videos are not part of the repository (about 9.5 GB); everything else is about 1.6 GB.

## Section 5: what a run folder carries

| ID | Item | State | Where |
|---|---|---|---|
| R1 | constraint block | **done** | `summary.json: constraints` (name, kind, measured, required, op, unit, passed) and `constraints_pass`; code `rcx/constraints.py`; existing runs filled by `tools/backfill_constraints.py`. Checked on all 1206 runs: `constraints_pass` is true exactly for the PASS runs and false exactly for the FAIL runs (INVALID runs: null). It is derived from the same recorded numbers as the label, so it is a consistency check, not an independent measurement |
| R2 | hidden settings | **done (declared)** | `summary.json: hidden_settings`, from `20RCs.md` "Hidden at the trigger", written with the implementation's parameter names. rc01 none; rc03 crash, intrude, psi_rest, T_slide, lead_brake, lead_react; rc04 and rc06 side_acc, side_react, fol_headway, fol_react; rc05 rear_acc, rear_vmax; rc09 rear_delay_s, rear_decel; rc13 dart_speed, dart_dist. This is the design list, not a measurement; Fred's measured list for rc03 (intrude, lead_brake, crash) is a subset |
| R3 | scene name separate from RC number | **partly** | `summary.json: scene_name` (for example `spun_car`, `street_van`) and `variation_id` (the variation) are there. Kent's `uid`, `site`, `split` are not: the runs do not start from Kent's variation files |
| R4 | shared set of policies on every scene | **not done** | needs new CARLA runs. Today 183 of 600 distinct (case, family, parameters) appear in more than one scene (the `ref` runs), but that is not a fixed shared set |
| R5 | each candidate in every hidden world of its history | **not done** (only `ref` runs, 323) | needs new CARLA runs; the search does not group scenes by history |
| R6 | more than one scene seed near boundaries | **not done** | every run has `scene_seed` 7 and no repeats. It is not yet known how much a new seed changes the scripted-actor cases (rc03, rc04, rc06); this should be tested first |
| R7 | whole pre-roll recorded | **not done** | recorded before the cue: rc01 about 9 to 10 s, rc05 2.0 s, rc09 and rc13 1.5 s, rc03, rc04 and rc06 0.5 s. A longer pre-roll changes the whole run, so those cases must be re-run |
| R8 | `RUN_SCHEMA.md` | **done** | `docs/RUN_SCHEMA.md`, now with the new fields and a units and signs table. The sign convention of `ego_yaw_rel` is still not written down |

## Section 6: Victor's questions (answers from the code; please correct)

**Q1. Reference point of `seen_*_rel_s`.** `rel_s` = the other actor's footprint centroid s minus the ego's footprint centroid s (mean of the four footprint corners, `rcx/evidence.py`). `ego_s` in the CSV is different: the ego's CARLA actor location. The few-cm offset Fred sees against `ground_truth.csv` is therefore most likely the distance between the vehicle origin and its footprint centroid. This comes from reading the code; I did not measure it.

**Q2. `seen_*` and `obs_*`.** `seen_*` is line of sight: within 80 m (`VIS_RANGE`) and not blocked by another tracked actor, centre to centre. `obs_*` is what the policy is told (the observation dict). Both are intended. The statement that 19 of 68 RC01 objects are never visible I have not checked.

**Q3. Thresholds** (all in `rcx/scenarios.py`, `rc01/evaluate.py`, `rc01/config.py`):
- Physics envelope: 0.2 s trailing mean of the measured acceleration; forward 0.5 g, braking 1 g, lateral 1 g and planar 1 g, each checked separately (not one combined 1 g).
- Ego settled (rc03, rc04, rc06): stopped (speed < 0.3 m/s), or over the last 10 ticks (0.5 s) the lateral offset is within `settle_tol` = 0.6 m of the start lane centre or of the escape lane centre. Heading is not checked. rc05 uses the last 20 ticks (1 s) within 0.5 m of a lane centre.
- Still closing (rc03, rc04, rc06, last tick): the ego is moving (> 0.3 m/s), overlaps the other vehicle laterally (lateral gap after the half widths < 0.3 m), the gap is positive and gap divided by closing speed is below 2.0 s. For a car behind (`esc_follow`, `rear`, `fol`) the same test with its closing speed. Function `still_closing` / `rear_closing` in `rcx/rc03_laws.py` and `rcx/rc04_laws.py`.
- Ego left the road (rc03, rc04, rc06): the largest |lateral offset| after the cue exceeds 1.5 lane widths. rc05: left foot edge beyond lane_w + 1.75 m or right edge beyond 1.75 m.
- Minimum clearance: rc03, rc04, rc06, rc09 0.5 m; rc05 rear-left 1.0 m and lead 2.0 m; rc01 2.0 m during the hold (hold: speed below 0.1 m/s for 2 s); rc09 stop = speed below 0.1 m/s at the end.
- rc13: pedestrian clearance 0.5 m (only when the pedestrian darts into the lane), unnecessary slowing = speed at the pass below 0.9 of the target speed or a stop although the pedestrian never entered the lane.
These are exactly the checks listed in `constraints` of each run.

**Q4. Scene and policy identity.** Yes: a scene is identified by its scene parameters (`scenario_params`), a policy by family plus `policy_params`. 9 identical (scene, family, parameters) triples occur more than once in the data (the same run in two batches).

**Q5. Schedule.** All RCs built so far are delivered. Not built: RC07, RC08, RC10, RC11, RC12, RC14 to RC20. The 7 INVALID runs (rc01 5, rc09 2) have not been re-run; they can be, on request. Videos are not in the repository.

**Q6, Q7** are for Victor and Kent together. For the record, the pass rules in these runs are the thresholds listed under Q3.
