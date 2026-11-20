# RC04 Free-Form Recovery Trajectory Search: Pilot Proposal

- Status: final proposal for team approval; not yet implemented
- Owner: Victor (policy search and CARLA execution)
- Interfaces: Kent (scenarios), Fred (data validation and recovery profiles), Yige (downstream world-model requirements)
- Workstream: broader recoverability research, separate from the frozen CSC2626 coursework experiments
- Version: 2.0, 19 November 2026

## 1. Decision and objective

Recommendation: GO for a bounded RC04 pilot. Add a new `free` policy family alongside `brake`, `lane_change` and `staged`. Keep the existing controller families, previously validated runs, physical limits and the CARLA evaluator unchanged.

We want to test whether a more flexible action representation can discover additional physically verified and meaningfully different recovery policies beyond the existing hand-designed families. The goal is not to maximise the number of PASS runs or a surrogate score. We want:

- a wider set of feasible recovery strategies within a given scenario;
- informative supported / refuted and near-boundary contrasts across parameterised scenarios;
- full-horizon, causally valid outcomes that can be grouped into recovery profiles;
- policy descriptions and trajectories that Fred can process and a future world model can learn from;
- measured search efficiency and trustworthy transfer to CARLA.

The search method is deliberately open. iCEM with the current surrogate is one baseline, not a required method. If direct CARLA search, a hybrid / multi-fidelity strategy, CMA-ES, CEM or MPPI-style sampling proves more reliable or cost-effective, Victor should use it and report the comparison fairly. Only verified CARLA executions determine reference labels.

## 2. Research question and scope

The current optimiser chooses a few parameters inside predefined manoeuvre templates. It cannot create a qualitatively different temporal sequence of braking and lateral motion.

A failure of `brake`, `lane_change` and `staged` does not imply the scene is physically unrecoverable. It establishes only that the tested policies and optimisation budgets did not find a supported response.

The RC04 pilot asks:

1. Coverage: can a free-form response succeed where all tested structured families fail? Does it add feasible options even where structured policies already pass?
2. Diversity: can it find multiple distinct supported, near-boundary, refuted and mitigation responses, not only one best trajectory?
3. Recovery boundaries: do small, physically meaningful changes in traffic conditions or policy shape change which continuations are supported?
4. Search reliability: do optimiser-approved candidates remain valid in CARLA, especially when the optimiser explores motions outside the surrogate's calibration distribution?
5. Downstream usability: can each free trajectory be fully described, replayed, checked across compatible hidden worlds, and represented as a candidate policy in a recovery profile?
6. Cost: how many surrogate evaluations, CARLA executions and wall-clock seconds are needed per verified diverse policy?

Pilot only: RC04, selected development scenarios and variations. Do not expand to all RCs or retrain the world model until the pilot's results justify it.

## 3. Policy representation: six-point free response plus feedback continuation

### 3.1 Initial representation

Implement a new `TrajectoryPolicy` compatible with the existing `PolicyOutput` interface. Start with K = 6 time-indexed control points over a fixed, declared emergency-response prefix.

Each point contains:
- requested longitudinal acceleration `a_req` (m/s2);
- desired lateral target / offset `t_des` (m), in the existing controller's coordinate frame and sign convention.

The first pilot uses uniformly spaced knot times and one deterministic interpolation rule (piecewise-linear is a simple initial choice). Freeze the prefix duration before search using the existing RC04 response timeline; do not search duration and knot count simultaneously. The executed trajectory still passes through the current low-level controller, tracking logic, actuator delay and command queue.

These are requested controls / targets, not guaranteed realised acceleration or steering.

### 3.2 Complete recovery, not just a short prefix

The optimised free sequence must be followed by an explicitly defined feedback stabilisation / continuation policy, executed to the original full recovery horizon. Record exactly how and when the free response hands over, whether its final target is held, and how pending commands and controller memory persist.

A successful short evasive prefix with an unsafe or unstable later continuation is not a supported recovery.

The free prefix is open-loop in its time-indexed requests. That is acceptable for this pilot, but it must not be described as a general adaptive feedback policy. The feedback continuation is a separate, named component of the complete candidate policy.

### 3.3 Physical feasibility and smoothness

Two complementary layers:

- During search: reasonable hard bounds on requested commands and target rates, plus a fixed, modest smoothness / rate penalty or filter. Avoid rapid acceleration-braking oscillation and untrackable lateral target changes. The rule is specified and frozen before the search comparison.
- During validation: the unchanged CARLA evaluator checks realised longitudinal / lateral / combined acceleration, body clearance, road limits, contacts and terminal behaviour under the existing time-window rules.

Do not assume that individually bounded requests keep the realised vehicle inside its joint envelope. Do not loosen limits or reinterpret a FAIL as PASS. Preserve instantaneous physical peaks as diagnostics even where the registered evaluator uses averaging.

If the optimiser regularly proposes invalid trajectories, tighten candidate-generation feasibility or improve the model; do not alter physical reference labels.

## 4. Search strategy: use whichever method is credible

Victor compares candidate strategies on a small common development panel and chooses the main approach based on CARLA-confirmed results and compute cost.

| Strategy | Advantages | Main risk | Use |
|---|---|---|---|
| Existing surrogate + iCEM | Cheap exploration, time-correlated sampling | Free trajectories may exploit surrogate errors | Reproducible reference |
| Direct CARLA search (CEM / iCEM, CMA-ES, ...) | Actual physics in the loop | More expensive and noisy | Credible alternative for small budgets |
| Hybrid / multi-fidelity | Cheap broad search, CARLA verification, then local refinement | Needs careful fidelity accounting | Preferred if it improves verified yield |
| MPPI-style sequence sampling | Handles correlated control sequences | Needs suitable dynamics, constraints and tuning | Optional, not a prerequisite |

Do not implement four optimisers for the sake of it. Start from the working iCEM infrastructure, run a small fidelity / cost comparison, and use the simplest method that produces trustworthy CARLA-qualified results.

The objective prioritises constraint satisfaction, then verified headroom / progress, then comfort / diversity. It uses informative continuous violation scores, not a flat PASS / FAIL reward. INVALID and unresolved attempts are preserved separately from qualified failures.

### If the surrogate is used

The current surrogate is calibrated mainly on structured manoeuvre shapes. Before trusting free-form elites:

- Run an initial approximately 40-candidate CARLA cross-check covering both random free trajectories and optimiser-selected elites. Report the strata separately.
- Measure surrogate PASS -> CARLA not-supported (false-safe) and surrogate FAIL -> CARLA supported (false-negative) with actual denominators.
- Treat 40 mixed checks as a smoke / fidelity pilot, not as proof of a false-safe rate below 10%. Include an uncertainty bound and say "inconclusive" if there are too few predicted-positive samples.
- If optimised elites show unacceptable transfer: recalibrate on development-only CARLA evidence, restrict the trusted search region, verify more candidates in CARLA, or switch to direct CARLA / hybrid optimisation.
- After recalibration, check a separate candidate sample; do not validate a new calibration on the traces used to fit it.

Surrogate outcomes never become the world model's physical reference labels.

## 5. RC04 experiment design

### Phase A: freeze representative cases

Choose three roles, not three outcome guarantees:

1. Previously all-family-refuted (for example `near_obstacle`, if the existing construction is qualified).
2. Benign / multiple-family-supported (for example `no_follower`).
3. Mixed or near-boundary (for example `truck_far_ahead`, if qualified).

For each role, select 2 to 3 representative parameter variations from Kent's documented ranges. Keep actor mechanisms and scenario meaning fixed. Freeze exact configurations, seeds, hidden-world definitions, budgets and evaluator version before comparing search methods. Include easy controls; do not force every case to have a failure.

Also rerun or supplement the structured-family optimiser with a matched incremental search budget on the selected cases. If free succeeds where a historical structured-family attempt failed, distinguish genuinely added expressivity from simply giving the new optimiser more samples or better initialisation. Preserve old results as historical evidence.

### Phase B: implementation and mechanical qualification

Verify:
- K = 6 controls reproduce a simple known braking profile within a declared tolerance;
- knot times, units, interpolation, lateral reference frame, bounds and smoothness;
- command delay, pending queue, feedback-tracker state and continuation handoff;
- parameter changes produce different applied actions, not only different metadata;
- full-horizon CARLA replay, physical checks and complete run logging.

### Phase C: search, diversity and boundary refinement

Search selected configurations, then independently execute and preserve representative candidates:
- best supported candidates by margin / progress;
- several behaviourally distinct supported candidates, if found;
- refuted and near-threshold candidates;
- mitigation-only outcomes when no tested candidate is supported;
- a common set of reference policies to compare across scene variations.

Reuse good candidates as search warm starts but verify them afresh in each variation. Refine around interesting supported / refuted transitions rather than sweeping an unbounded parameter grid. Record how much extra search was spent on each case.

Diversity is measured on executed behaviour (applied acceleration, lateral trajectory, response timing, progress) rather than only on distance between control-point vectors. State any clustering or separation thresholds.

For hidden-world cases, execute the exact same complete policy in every declared compatible world. A trajectory optimised with privileged knowledge of the sampled world cannot be counted as robust from an indistinguishable observation history unless it passes across the full compatible set. A qualified counterexample refutes that policy; missing or invalid worlds remain unresolved.

Repeat selected near-boundary candidates under additional initialisation seeds. Report sign stability and raw margins; repetitions do not create independent scene families.

### Phase D: compare against existing families

Report at minimum:

| Metric | Meaning |
|---|---|
| Added coverage | Configurations where free is supported and none of the tested structured-family candidates is |
| Number and diversity of supported policies | Whether free adds genuinely different recoveries, not just one elite |
| Boundary coverage | Nearby supported / refuted policies and the relevant physical constraint changes |
| Complete behaviour | Progress, contact, road, envelope, stabilisation and terminal outcomes |
| Robustness | Qualification across compatible worlds and repeat checks |
| Search efficiency | Verified supported candidates per CARLA call / wall-clock budget |
| Surrogate transfer, if used | Random and elite false-safe / false-negative rates with denominators |
| Dataset quality | Complete query histories, shared-policy coverage, masks, unresolved entries |

A negative coverage result is still useful. Not finding PASS does not prove global physical impossibility, only failure of the tested control-point representation, controller, search budget and worlds.

## 6. Downstream data contract (Victor -> Fred -> world model)

Each physically evaluated candidate / world execution needs a reproducible run folder and an identifier linking all policies evaluated from the same public decision history.

| Category | Required fields |
|---|---|
| Scenario provenance | RC, Kent's base-scenario ID / UID and site, variation settings, world ID, declared hidden settings, random seed, source commit |
| Policy definition | `family=free`; K; knot times; `a_req`; `t_des`; units, reference frame, interpolation, prefix duration, feedback continuation, controller / queue version |
| Public decision information | Query time, complete pre-query ego-observable history, track / visibility masks, actual delivered cue events, past applied controls, pending commands and known controller state |
| Execution evidence | Per-tick requested and applied controls, ego / actor telemetry, collisions, road and vehicle constraints, progress, complete terminal continuation |
| Reference labels | Raw per-constraint measured value and threshold, signed normalised margins, first violation timing, status and validity / qualification mask |
| Search metadata | Optimiser, random seed, evaluation budget, surrogate score if applicable, physical result, candidate category (best / boundary / ref / mitigation), cost |
| Reproducibility | Base-group and query IDs, hashes, matched-history checks, replay receipts, provenance |

Do not export only `family=free` and PASS / FAIL. The full control-point sequence and continuation must be a learnable, replayable policy description. Fred's existing 15-slot descriptor cannot necessarily represent it: add a versioned control-point tensor / mask while preserving the original schema. Existing coursework model weights will not automatically accept this new policy encoding.

Keep per-world physical margins separate from per-history robust profiles. Hidden settings and future ground-truth actor trajectories are for reference / auditing only, not initial-query model inputs.

For future learning experiments, split by complete base scenario / counterfactual family. Keep neighbouring variations, policy-search seeds, hidden worlds, timing variants and repeat executions together. Reserve independent groups and policy shapes / parameters for prospective transfer testing.

## 7. Go / no-go decision after the pilot

Proceed beyond RC04 only if the new family adds credible CARLA-verified recovery coverage and / or useful policy diversity at reasonable search cost, with complete, consistent dataset records. It does not have to find a new PASS in every difficult scene.

If the surrogate is unreliable, switch search methods; do not abandon the concept merely to preserve the surrogate. If free produces only existing manoeuvre shapes, excessive invalid controls, weak diversity or unjustifiable compute cost, retain the null result and reconsider whether observation-conditioned free policies are the more valuable next research step.

Do not expand to all RCs automatically; review RC04's coverage, transfer, cost and dataset usefulness first.

## 8. Required deliverables from Victor

1. Implementation and unit / replay tests for `TrajectoryPolicy`, interpolation, physical command handling and the feedback continuation.
2. Frozen pilot manifest: exact selected RC04 cases / variations, known hidden worlds, seeds, optimiser budgets and evaluator version.
3. Search-fidelity report: measured random / elite surrogate transfer, any calibration performed, and a direct-CARLA / hybrid comparison if needed.
4. CARLA reference results: all assigned candidates and statuses; comparisons with matched-budget structured-family searches; full constraint components.
5. Policy-diversity and coverage report: distinct supported policies, near-boundary / refuted examples, mitigation behaviour, added coverage, progress and computation.
6. Complete Fred-compatible run package: per-tick public observation and ground-truth channels, control-point descriptions, policy / world / profile identifiers, resolution masks and hashes.
7. Illustrative synchronised CARLA videos / plots with trace IDs, including a successful example and a refuted or near-boundary example where available.
8. Short GO / NO-GO recommendation about extending free search to RC03, RC06 or additional mechanisms.

## 9. Final principle

The research outcome is not "iCEM found a larger score" or "a free trajectory passed one scene".

It is whether expanding the finite policy library reveals additional, physically verified, causally valid recovery choices and informative differences between those choices, which can become reliable supervision for a recovery-aware world model.

The pilot strengthens our broader research programme while remaining separate from the already frozen CSC2626 coursework results.

## 10. Implementation status (added after the proposal was accepted)

Pilot code and the surrogate stage are done; the CARLA stage is queued. Pilot design choices that fill in details this proposal leaves open (continuation definition, prefix timing, bounds, hidden-world set, scene choice, warm starts), the deviations from the plan and the interim surrogate results are in `results_free/README.md` and `results_free/rc04/manifest.json`.
