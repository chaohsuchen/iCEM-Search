# Recovery-Condition (RC) policy search with CARLA 0.9.15

Goal: for each Recovery Condition (RC) and scene variation, find several recovery strategies that work, plus strategies that only mitigate or fail, and record every CARLA run as evidence. The result is a recovery-profile dataset for later world-model training.

Two layers:
1. **Surrogate** (`search/`): a fast numpy simulator plus an iCEM optimizer proposes policy parameters. It never produces labels.
2. **CARLA** (`rcx/`, `rc01/`): replays every proposed policy in CARLA 0.9.15. **Every label (PASS / FAIL / INVALID / UNRESOLVED) comes from CARLA.**

Quality gate: per RC and batch, the false-safe rate (surrogate PASS but CARLA not PASS) must be below 10% before moving on (`tools/summarize_verify.py`).

## Status

| RC | Scene | State |
|---|---|---|
| RC01 | stop before a blocking van (Town01) | done (baseline, presets, boundary) |
| RC02 | RC01 scenes where no policy reaches PASS | derived from RC01 (`tools/rc02_negatives.py`) |
| RC03 | braking fails, lateral escape remains (Town04) | done |
| RC04 | stopped car ahead, truck alongside, follower (Town04) | done |
| RC05 | abort a lane change | done |
| RC06 | broken-down car ahead, passer alongside (shares the RC04 scene and surrogate) | done |
| RC09 | front/rear squeeze, no exit | done |
| RC13 | pedestrians that may dart | done |
| others | see `20RCs.md`, grouping in `docs/RC_GROUPS.md` | not started |

## Where to start (data and docs)

- **The data:** `results_search/` (read `results_search/README.md` first). `results_search/summary/carla_runs.json` indexes every CARLA run; each run folder holds `summary.json`, `ego_observable.csv`, `decisions.jsonl`, `ground_truth.csv`, `run.mp4`.
- **What a run folder contains and what the world model may use:** `docs/RUN_SCHEMA.md` (schema `rcx-run-1`). World-model input is `ego_observable.csv` and `decisions.jsonl` only; `ground_truth.csv` is for analysis and labelling.
- **Scene definitions:** `20RCs.md` (all 20 RCs: scenario, options, variations, hidden parameters).
- **RC grouping and what can be reused:** `docs/RC_GROUPS.md`. **Adding a new RC:** `docs/NEW_RC_CHECKLIST.md`.

## Code layout

| path | content |
|---|---|
| `rcx/` | CARLA scenes (`scenarios.py`), scripted actor laws (`rc03_laws.py`, `rc04_laws.py`), policies, run engine and evidence writer (`common.py`, `evidence.py`) |
| `rc01/` | low-level controller, calibration, RC01 street scene, pass/fail judge, video overlay (imported by `rcx`) |
| `search/` | surrogate simulators (`*_sim.py`), per-RC `Case` classes and the `CASES` registry (`cases.py`), iCEM (`cem.py`), policy families, `template_case.py` for a new RC |
| `tools/` | `run_cem.py` (search), `scan_params.py` (sensitivity scan), `run_boundary.py` (boundary sweeps, configs in `boundary_configs/`), `verify_carla.py` (CARLA replay), `summarize_verify.py` (false-safe), `index_runs.py`, `build_dataset.py`, `check_layout.py`, `rc03_scan.py`, `rc02_negatives.py` |
| `slurm/`, `slurm_logs/` | **not part of this repository** (private cluster scripts, see "Running CARLA verification" below) |
| `results_v3/` | CARLA calibration files (`<rc>/calibration*.json`) that the surrogates load. Produced by the legacy driver `run_v2.py` |
| `results_search/` | all search and CARLA results (see its README) |

## The standard flow (same four steps for every RC)

```bash
PY=<python with numpy and the carla egg>      # e.g. a private venv; the repo has no install step
# 1. baseline = the RC's original scene list
$PY tools/run_cem.py --case rc04 --batch baseline                       # surrogate search -> results_search/rc04/search/baseline
$PY tools/verify_carla.py --case rc04 --batch baseline --port 2000 --shard 0 --nshards 3   # CARLA replay, one call per shard (see below)
$PY tools/summarize_verify.py results_search/rc04/carla/baseline        # false-safe must be < 10%
# 2. presets_<tag>: hand-picked easy / mid / boundary / hard scenes   (run_cem.py --scen-file <scenes.json>)
# 3. boundary_r<N>: sweep one scene parameter, bisect where PASS flips (tools/run_boundary.py, config tools/boundary_configs/<rc>.json)
# 4. $PY tools/index_runs.py; $PY tools/build_dataset.py; $PY tools/check_layout.py
```

Only iCEM is used. Scene parameters (gap, speed, delay, d_obs, ...) are never searched by iCEM; batches differ only in how scenes are chosen. There are no conditional or guarded programs: the optimizer only picks numbers.

## Running CARLA verification (the Slurm scripts are not included)

The Slurm scripts (`slurm/`, `slurm_logs/`) are specific to the author's private cluster and are **not published in this repository**. They only did three things, which you can reproduce on any machine:

1. Start a CARLA 0.9.15 server (off-screen rendering, `-RenderOffScreen`; needs a GPU with Vulkan, tested on NVIDIA only; AMD is untested) and wait until it accepts connections on a port.
2. Set the controller environment variables used by every verification run: `RC_SUBSTEP=0.002 RC_BRAKE_SCALE=3.0 RC_BRAKE_CAP_MAX=0.85`.
3. Run the verifier, one call per shard (shards only split the candidate list; run them in parallel on separate servers or one after another):
   ```bash
   RC_SUBSTEP=0.002 RC_BRAKE_SCALE=3.0 RC_BRAKE_CAP_MAX=0.85 \
   $PY tools/verify_carla.py --case rc04 --batch baseline --port 2000 --shard 0 --nshards 3
   ```
   Output goes to `results_search/<rc>/carla/<batch>/` (`results_shard*.jsonl`, `runs/`). If a shard hangs (CARLA simulator timeout), remove its error lines from `results_shard*.jsonl` and run it again; finished candidates are skipped.

## Environment

- CARLA 0.9.15 server and the matching `carla` Python 3.10 client, plus numpy. There is no install step in the repo.
- The data under `results_search/` and the surrogate search (`tools/run_cem.py`) do **not** need CARLA or a GPU; only `tools/verify_carla.py` does.

## Conventions

- All files, comments and logs are in English.
- Ego speeds stay at the original 18 / 22 m/s (RC03, RC04, RC06).
- INVALID runs (CARLA scene did not match the declared setup) are ignored for labels and for false-safe.
