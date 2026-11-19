# results_search: quick guide for readers

**What this is.** For each Recovery Condition (RC, e.g. `rc03` = braking fails but a lateral escape remains) we searched for recovery policies and ran them in CARLA 0.9.15. A fast surrogate simulator (iCEM search) proposes policies; CARLA executes them; **every label comes from CARLA**, never from the surrogate. The point is a dataset of recovery profiles: per RC and scene variation, policies that recover, policies that only mitigate, and policies that fail.

**Where to start**
1. `summary/carla_runs.json` - one row per CARLA run (RC, batch, scene parameters, family, policy parameters, CARLA label, path to the run folder). Filter here first.
2. Open a run folder `<rc>/carla/<batch>/runs/<NNN>_<scene>_<family>_<kind>/` (what each file contains: `docs/RUN_SCHEMA.md`):

| file | what it is | who may use it |
|---|---|---|
| `summary.json` | RC ID, scene/variation ID, scene parameters, policy family and parameters, seeds, CARLA label + reason, metrics | metadata |
| `ego_observable.csv` | per tick (0.05 s): ego state, what the policy observed, what it perceived (with visibility flags), requested and applied controls | world-model input |
| `decisions.jsonl` | per decision tick: observation history, query time, controller state, pending commands | world-model input |
| `ground_truth.csv` | per tick: true state of every actor, clearances, scene measurements (includes hidden information) | analysis / labelling only |
| `run.mp4` | chase-camera video | looking only |

**What each directory means**

| path | meaning |
|---|---|
| `<rc>/` | one Recovery Condition (`rc01` stop before a blocking van, `rc03` lateral escape, `rc04` stopped obstacle ahead, side vehicle and follower, `rc05` abort a lane change, `rc06` broken-down car 60-90 m ahead and a faster car passing alongside: let it pass, then move in behind it, `rc09` front/rear squeeze in a tunnel, `rc13` pedestrians that may dart) |
| `<rc>/search/<batch>/` | what the surrogate search produced: `scenarios.json` (the scenes or axes), `candidates.json` (the policies sent to CARLA), `profiles.json`/`rollouts.pkl` (search statistics) or `table.json` (boundary sweeps) |
| `<rc>/carla/<batch>/` | the CARLA execution of the same-named search batch: `results_shard*.jsonl` (one line per candidate with surrogate prediction and CARLA result), `runs/` (the evidence folders) |
| `<rc>/scans/` | one-off sensitivity scans used to choose the 2-3 dominant parameters and the presets |
| `summary/` | cross-RC files: run index, per-scene recovery profiles, optimizer comparison |
| `logs/` | surrogate search logs |

**Batches** (same name under `search/` and `carla/`): `baseline` = the RC's original scene list; `presets_<tag>` = hand-picked easy / mid / near-boundary / hard scenes; `boundary_r<N>` = sweeps along one scene parameter (warm-started iCEM at every point, plus the easy-end policy kept fixed, bisection where PASS flips to not-PASS). A candidate's `kind`: `best` = optimized for that scene, `ref` = a fixed policy carried to a harder scene (fixed-policy flip), `boundary` = the smallest-margin PASS found by the search.

**Reading a result.** `label` in `summary.json` is the CARLA verdict: PASS (recovered: no contact, clearances kept, settled), FAIL (contact, clearance violated, or physics envelope exceeded; the `reason` field says which), INVALID (the CARLA scene did not match the requested setup, e.g. cue gap off or vehicle overlaps a wall; ignore for labels, not a policy failure) or UNRESOLVED (the cue was never delivered or evidence is missing without any failure evidence; not used as a label). Trust level: for each RC the surrogate was checked against CARLA (false-safe rate = surrogate says PASS but CARLA does not; target < 10%, `tools/summarize_verify.py`).

---

# results_search layout and naming rules

```
results_search/
  README.md  logs/  summary/
  <rc>/                      rc01 rc03 rc04 rc05 rc06 rc09 rc13 ...  (two digits)
    search/<batch>/          surrogate (iCEM) output
        scenarios.json       the scene list / axes of the batch (the batch is self-contained)
        candidates.json      policies sent to CARLA (kind = best | boundary | ref)
        profiles.json, rollouts.pkl      (run_cem batches)      table.json      (boundary batches)
    carla/<batch>/           CARLA verification of search/<batch> (same batch name)
        results_shard*.jsonl, calibration_*.json
        runs/<NNN>_<scenario>_<family>_<kind>/
            summary.json  ego_observable.csv  ground_truth.csv  decisions.jsonl  run.mp4      (see docs/RUN_SCHEMA.md)
    scans/                   one-off sensitivity scans used to choose presets
  logs/cem_<rc>_<batch>.log  surrogate search logs
  summary/                   cross-RC outputs: carla_runs.json (index of every CARLA run), recovery_profiles.json, rollouts.npz, OPTIMIZER_COMPARISON.md, rc02_unrecoverable.json (RC02 = rc01 scenes where nothing reaches PASS; tools/rc02_negatives.py)
```

## Batch names (identical under search/ and carla/)
| name | meaning |
|---|---|
| `baseline` | the original scene list of the RC (the case's `SCEN`). Re-running it in the new format replaces the old runs. |
| `presets_<tag>` | representative easy / hard / near-boundary presets, `<tag>` = what distinguishes the batch (e.g. `22mps`) |
| `boundary_r<N>` | boundary exploration (warm start + bisection), round N (`r1`, `r2`, ...). Keep the last round that passed the false-safe check; labels of older rounds stay valid data (labels are always CARLA's). |

`<NNN>` in a run folder = index of the candidate in candidates.json. Every run folder has the full evidence files (`tools/check_layout.py` checks this).

## Commands (run from the project root)
```
python tools/run_cem.py --case rc03 --batch presets_22mps --scen-file <scenes.json>      # -> search/<batch>/ (+ scenarios.json)
python tools/run_boundary.py --case <rc> --batch boundary_r4                          # boundary batch, axes from tools/boundary_configs/<rc>.json
CASE=rc03 BATCH=presets_22mps SHARD=0 NSHARDS=2 sbatch --export=ALL --exclude=gpunode16 --gres=gpu:rtx_4090:1 slurm/verify.sbatch   # -> carla/<batch>/
python tools/summarize_verify.py results_search/rc03/carla/<batch>        # false-safe of that batch
python tools/index_runs.py        # summary/carla_runs.json
python tools/build_dataset.py     # summary/recovery_profiles.json etc. (baseline batches)
python tools/check_layout.py      # verifies these naming rules
```
Every new RC / batch must pass `tools/check_layout.py` before it is considered done.

## Notes on rc04 and rc06 (stopped obstacle ahead, side vehicle, follower)

Both share one scene and one surrogate (`search/rc04_sim.py`); actors follow the scripted laws in `rcx/rc04_laws.py`; the ego dynamics calibration of rc03 is reused. Policy families: `brake`, `lane_change`, `staged` (slow down first, then change lane).

- **Finding (rc04, dry road, 18 and 22 m/s):** `staged` never beats `brake`; wherever a staged policy passes, a plain brake policy passes too. Staged runs are kept as data, they are not a better recovery.
- **rc06 surrogate calibration.** The first rc06 search had a high false-safe rate (batch values 11.6 / 19.2 / 5.6 %). Cause: for a positive requested acceleration the surrogate accelerated the ego much more than CARLA does (surrogate about 26 m/s, CARLA ego stays near 22.4 m/s), so in the surrogate the ego outran the faster follower but in CARLA it was rear-ended. Fix, applied to rc06 only (`RC06Case.SUR` in `search/cases.py`): `acc_gain=0.7` (scales positive requested acceleration), `it0=0.05`, and wider pads (`fol_pad=0.8`, `side_pad=0.5`, `obs_pad_lc=0.5`), fitted on CARLA speed traces. Raising the pads alone made the surrogate too strict (it rejected policies that pass in CARLA). rc04 keeps its original surrogate settings.
- **Result of the gate (false-safe = surrogate PASS but CARLA not PASS):** rc06 baseline 2.6 %, presets_main 4.2 %, boundary_r1 2.0 %; rc04 baseline, presets_main and boundary_r2 are all below 10 %.
- In `boundary_r1` of rc06 the surrogate is conservative: 16 of 73 surrogate-FAIL candidates pass in CARLA. This lowers the number of feasible strategies found, it does not hurt trust in the PASS labels.
