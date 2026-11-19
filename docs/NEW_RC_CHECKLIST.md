# Checklist for adding a new RC

Template: `search/template_case.py` (copy to `search/<rcNN>_case.py`; do not edit the template in place).
The Slurm scripts are not part of the repository (private cluster); see README "Running CARLA verification". All files, comments and logs are written in English.

## 0. Classify first (see RC_GROUPS.md)
- [ ] Which group (A/B/C/D)? Group D: write only a wrapper, skip to step 3.
- [ ] Can it be done with only new parameters + a new cost? If yes, do not write a new simulate.

## 1. Scenario definition (in the Case docstring; source: 20RCs.md)
- [ ] theta_s: variable names (CARLA names), ranges, units
- [ ] Hidden parameters (not observable at the trigger): `perturbed()` must perturb them
- [ ] Benign counterpart (companions): same scene without the threat, to catch over-conservative policies
- [ ] Keep negative (unsolvable) cells in SCEN; never drop them to look better

## 2. CARLA scene (`rcx/scenarios.py`)
- [ ] New `BaseScn` subclass with `defaults`; town / road / lane
- [ ] t0 = first time the threat is *seen*, not the spawn time
- [ ] Register in the `TOWN` table and the `SCN` branch of `tools/verify_carla.py`
- [ ] Run one scripted policy in CARLA to confirm the scene reproduces (before touching the surrogate)

## 3. Surrogate
- [ ] Reuse the plant (LongPlant / lateral plant), `LowLevelController`, and the same policy class as CARLA
- [ ] New `simulate(theta_s, kw, plant, calib)` returns a dict with the CARLA summary field names (verdict, contact, rel_speed, min_*)
- [ ] Calibrate with 8-12 CARLA runs -> `results_v3/<rc>/calibration.json`; compare surrogate and CARLA traces tick by tick (actor motion models are the usual culprit, e.g. the RC13 walker ramp)
- [ ] Record known surrogate bias and include it in `margin()`
- [ ] Mirror CARLA's physics-envelope checks (filtered accel limits) in the surrogate verdict

## 4. Action space (structure and ranges only, never the answer)
- [ ] Define families as FAM dicts (policy class + parameter names + lo/hi + fixed), like `search/families.py` and the per-case FAM in `search/cases.py`; no conditional/guarded programs (the optimizer only picks numbers)
- [ ] Always include a trivial family (keep speed / plain brake) so the optimizer can pick the baseline
- [ ] Ranges must be wide enough to contain wrong choices; never shrink them to contain only the solution
- [ ] Guards may use observables only, never hidden parameters

## 5. Cost / class / margin
- [ ] `outcome_class` -> success / mitigate / fail (contact with rel_speed < 2 = mitigate)
- [ ] Cost: contact >> everything else; keep progress out of the safety cost and store it as a separate field
- [ ] `margin()`: remaining slack of a PASS, used to choose boundary candidates

## 6. Register and search
- [ ] Add to `CASES`; 6-10 entries in SCEN; `perturbed()` includes hidden-parameter perturbations
- [ ] `python tools/run_cem.py --case <rc>` (iCEM only by default)
- [ ] Output: `results_search/<rc>/{search,carla}/<batch>/` (see results_search/README.md)
- [ ] `python tools/check_layout.py` prints "layout OK" (batch names, run folders, logs follow results_search/README.md)

## 7. CARLA verification and false-safe
- [ ] Run `tools/verify_carla.py --case <rc> --batch <batch> --port P --shard i --nshards n` against a running CARLA server (README "Running CARLA verification")
- [ ] After a timeout: remove the error lines from `results_shard*.jsonl` before re-running (verify_carla now also skips error rows when deciding what is done)
- [ ] false-safe = surrogate PASS and CARLA not PASS; if > 10%, go back to step 3 and do not expand

## 8. Dataset
- [ ] Add the case to the `cases` tuple in `tools/build_dataset.py`
- [ ] Labels always come from CARLA; surrogate-only cells are marked `source=surrogate`
- [ ] Record the library version; missing evaluations are NaN, never failure

## Frozen output format and standard flow (do not change per RC)
The output format is frozen as schema `rcx-run-1` (docs/RUN_SCHEMA.md) and the layout of results_search/README.md. Every RC runs the same four steps; only the Case, the CARLA scene and the scene lists differ:
1. `tools/run_cem.py --case <rc> --batch baseline` -> verify (`tools/verify_carla.py`, --case/--batch/--shard/--nshards) -> `tools/summarize_verify.py` (false-safe < 10%)
2. `tools/scan_params.py` to pick the 2-3 dominant parameters -> `presets_<tag>` (`run_cem.py --scen-file`) -> verify -> false-safe
3. `tools/boundary_configs/<rc>.json` -> `tools/run_boundary.py --case <rc> --batch boundary_r1` -> verify -> false-safe (next round `boundary_r2` if > 10%)
4. `tools/index_runs.py`, `tools/build_dataset.py`, `tools/check_layout.py` (also validates the required summary keys / CSV columns / decision fields of every run)
If the format must change, bump the schema version and re-run old batches; never mix formats within results_search/.
