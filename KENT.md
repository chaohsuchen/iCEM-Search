# Branch `kent`: Kent's additions to the search, and Kent's results

This branch adds Kent's work on top of `main`. Victor's code is unchanged except for one optional argument in `search/cem.py`. The search runs Victor's iCEM over Kent's scenes, surrogate and CARLA driver (`carla_rc/teachers`). Every label comes from CARLA, as on `main`.

RCs added here: RC07, RC10, RC11, RC14, RC15 (work in progress), RC08, RC12, RC16, RC18 and RC19. RC20 is covered as a definition and a library-recall measure in `results_search/KENT_README.md`. RC03 stays Victor's.

## What is added

| path | content |
|---|---|
| `search/cem.py` | one optional `stop` callback for `icem` (early stop); default behaviour unchanged |
| `search/kent_bridge.py` | bridge to `carla_rc/teachers`: surrogate rollouts, the policy parameter spaces, and the results folder (`results_search/`, or `$RC_RESULTS`) |
| `search/kent_cases.py` | per-RC `Case` classes for the scenes in `carla_rc/teachers`: ranges, sites, presets, perturbations for the hidden worlds |
| `search/pipeline.py` | batch pipeline: random / presets / boundary stages, hidden worlds (R5), shared setting per family (R4), candidate selection for CARLA |
| `search/rc19_batch.py` | RC19: anchors of RC03 / RC10 / RC07 batches rerun under cases A-D and the semantic control S |
| `search/experimental/` | experimental program-search code (not used by the pipeline) |
| `carla_rc/` | Kent's CARLA side; see below |
| `RC_scenarios_HighwayEnv_CARLA.ipynb` | the notebook whose code cells (actor laws, controllers) `options.py` / `laws.py` load at run time; outputs stripped |
| `tools/kent/pick.py` | selective CARLA: picks the meaningful histories of a batch (defining / contrast / hard / easy) |
| `tools/kent/shared_index.py` | writes `shared_runs.json` (which CARLA runs are each family's shared setting, R4) |
| `tools/kent/runs_io.py` | reads a run folder's files, gzipped or plain |
| `tools/kent/rc20.py` | RC20 on Victor's RC04 scene: restricted / full family library / generic planner / fixed settings, then labels |
| `results_search/rc07`, `rc08`, `rc10`, `rc11`, `rc12`, `rc14`, `rc15`, `rc16`, `rc18`, `rc19` | Kent's results, next to Victor's RCs; `results_search/KENT_README.md` and `KENT_RUN_SCHEMA.md` describe them |

## `carla_rc/`: only what the search and the CARLA replay import

| file | role |
|---|---|
| `teachers/rc*_opts.py` | **one file per RC scene.** Each defines a spec class: the geometry and actors built from the scene parameters (`build`), the road bounds, the parameter ranges, the CARLA sites (town, start point, lane, train / held out), the trigger, the option families (`modes`, e.g. keep / brake / stop_end / overtake) and the hidden parameters. `rc07_opts`: RC07 (blowout + on-ramp); `rc08_opts`: the garbage truck; `rc10_opts`: RC10 / RC11 (spinning car, adjacent lanes); `rc14_opts`: RC14; `rc15_opts`: RC15 (merge); `rc16_opts`: the double-parked van; `rc19_opts`: RC19's cases A-D / S around the RC03 / RC10 / RC07 scenes; `rc03_opt(s)`: Kent's RC03 scene (the base of RC12 and RC19 `on_rc03`); `rc01_opts`: the roadworks corridor (registered, not used here). |
| `teachers/rc03.py`, `rc07.py`, `rc11.py`, `rc15.py`, `rc19.py` | the earlier fixed-scenario versions; still imported by `scenarios.py`, which registers every scene |
| `teachers/scenarios.py` | the scene registry (`SPECS`: scene id -> spec) and the shared actor helpers |
| `teachers/options.py` | the option families and their controllers (lane change, swerve, overtake, wait-and-overtake, ...), the parameter spaces `PARAM_SPACE` and the trigger / delay logic |
| `teachers/surrogate.py`, `rc_search.py` | the surrogate (the same scene and controllers in a kinematic model) and the rollout / scoring used by the search |
| `teachers/env.py`, `frame.py`, `envelope.py`, `envelope_qual.json`, `laws.py`, `sight.py`, `expert.py` | the CARLA environment: lane frame on a real map, vehicle envelope, actor laws, visibility, the notebook loader |
| `teachers/evidence.py`, `carla_verify.py` | per-tick run folders (R1-R8) and the CARLA command line: `select` / `run` / `rerun` / `score` / `summarize` |
| `bridge.py`, `accel_map.json` | CARLA control mapping and the measured acceleration map |
| `tests/fake_carla/` | CARLA stand-in so the surrogate and search run without CARLA |
| `shard_loop.sh`, `restart_srv.sh` | run one `carla_verify.py` shard on a fresh server per round until it is done (survives server crashes and the server memory leak); they assume the VM layout (`~/rc`, `~/carla0915`, `~/rcenv`) |

## Results

`results_search/<rc>/search/<batch>/` holds the search output: `scenarios_*`, `profiles_*` and `candidates_*.json`. `results_search/<rc>/carla/<batch>/` holds the CARLA side: `selected.json`, `results_shard*.jsonl` (one line per CARLA run, including the label and the surrogate verdict), `repeats/` and `shared_runs.json`. Batches: `vsearch` for every RC; `vsearch_rear` (RC10, RC11) and `vsearch_rc14rear` (RC14) add a rear vehicle; RC19 has `on_rc03`, `on_rc10` and `on_rc07`.

Not included:
- The per-run folders of the variations that were not selected. The selected ones are included (below); the rest stay on the CARLA VM, ask Kent for any you need.
- The mid-search checkpoints (`partial_*.jsonl`).
- Kent's RC03 and RC07's earlier batches (`baseline`, `validation`, `pilot2`). RC12 and RC19 `on_rc03` were built on Kent's RC03 scene, so regenerating them (`search/rc19_batch.py rc03`) needs Kent's RC03 batch, not Victor's `results_search/rc03`.

At the time of this commit, the CARLA "complete" fill (every remaining candidate, not only the selected ones) is still running for several batches. Some `results_shard*.jsonl` files will grow.

## Run folders (per-tick data)

`results_search/<rc>/carla/<batch>/runs/<run_id>/` (and `repeats/runs/` for the repeats near the boundary, R6) for the **selected variations** of each batch: about 40-50 per batch, picked by `tools/kent/pick.py` (defining / contrast / hard / easy, spread over presets, random and boundary stages), each with every candidate in every hidden world (R5) and its repeats; RC19 complete. The selected variation ids are in each batch's `selected_variations.txt`; for RC07, RC10, RC11, RC14 and RC15 `vsearch` (where every candidate was run in CARLA) the run folders of 40 picked variations are included.

Each folder: `summary.json` (plain: label, constraints R1, hidden parameters R2, scene name / uid / site / split R3, policy, surrogate prediction) and `ground_truth.csv.gz`, `ego_observable.csv.gz`, `decisions.jsonl.gz`. The per-tick files are gzipped, with numbers at float32 precision (CARLA's own; relative change < 5e-7): about 20 GB raw, ~2 GB here. Read them with `tools/kent/runs_io.py`; details in `results_search/KENT_RUN_SCHEMA.md`.

## Running

The surrogate and the search need no CARLA (the stand-in is used):

```
python -c "from search import kent_bridge as kb; print(kb.OUT)"
python -m search.pipeline rc08 --stage random --n 150 --seed 308 --batch vsearch   # stages: see search/pipeline.py
python -m search.rc19_batch rc10 vsearch_rear --n 12
```

CARLA replay (CARLA 0.9.15 and ScenarioRunner on the machine):

```
python carla_rc/teachers/carla_verify.py select --rc rc08 --batch vsearch [--keep keep.txt]
bash carla_rc/shard_loop.sh run rc08 vsearch 0 2000          # shard 0 of 4, server port 2000
python carla_rc/teachers/carla_verify.py score --rc rc08 --batch vsearch
```
