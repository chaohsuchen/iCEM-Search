# Grouping of the 20 RCs (by whether simulate / plant can be shared)

Status: DONE = has a Case, a CARLA scene and CARLA-verified batches (rc01, rc02 (derived from rc01), rc03, rc04, rc05, rc06, rc09, rc13). RC01/02/05/09/13 follow the as-built scenes (20RCs.md was updated to match); the other RCs follow 20RCs.md. Still to do: RC07, RC08, RC10, RC11, RC12, RC14, RC15, RC16, RC17, RC18, RC19, RC20.

| Group | Shared core | Members | Reuse / new |
|---|---|---|---|
| **A. Longitudinal braking / stopping** (1D ego, LongPlant, ProfilePolicy) | 1D kinematic front/rear actors in the style of `rc09_sim` | RC01/02 (DONE), RC09 (DONE), RC08 (the hold part) | New A-group RCs need only new parameters + cost. RC02 is RC01's gap range: nothing new to write |
| **B. Longitudinal + pedestrians / occlusion** | `rc13_sim` (kinematic pedestrians + dart) | RC13 (DONE) | Reuse `rc13_sim` |
| **C. Lateral lane change / abort / escape** (2-3 lanes, lateral plant, footprint) | **New `lane_sim`**, written once: lateral plant + rotated-rectangle distance from `rc05_sim`, pluggable actor laws (constant speed / accelerate / brake on seeing / cut in) | RC03 (DONE), RC04 (DONE), RC05 (DONE, not migrated), RC06 (DONE), RC07, RC10, RC14, RC15, RC18 | Each RC only adds an actor list + hidden parameters + cost |
| **C'. In-lane lateral shift (footprint)** | `lane_sim` single-lane mode + static narrow-passage geometry | RC16 | Add passage obstacles to `lane_sim` |
| **C''. Oncoming-lane overtake** | `lane_sim` + oncoming stream | RC08 | Add oncoming actors; the hold part reuses group A |
| **D. Wrappers (no new simulate)** | Wrap an existing simulate | RC02 (=RC01), RC11 (=RC10 with late t_reveal), RC12 (=RC03 + delay), RC17 (=RC16 + delta_obs/delta_cmd; an RC01 delay variant is already built), RC19 (=RC03 + nuisance; label-consistency check only, no search), RC20 (=RC04 + restricted family) | Case flags / wrappers: delay buffer, restricted FAM, nuisance renders |

## Suggested first batch (at least one per group)
1. RC13 pre_slow fix (done in this session; see the report).
2. RC03 (group C: builds `lane_sim`; RC10/11/12/18/19 depend on it).
3. RC06 (group C: shares the RC04 scene and surrogate; no conditional programs, the optimizer only picks numbers).
4. RC04 (group C: staged plan; base of RC20).
5. RC16 (group C': footprint; base of RC17).
6. RC08 (A + C'': hold vs overtake, safety and progress stored separately).
7. RC02 + RC12 (group D: nearly free; validates the wrapper mechanism).

Acceptance: each new Case must be calibrated to false-safe < 10% before the next batch.

## Self-contained rule
- RC03 and all other not-yet-built RCs are defined only by `20RCs.md` plus this project's own code; no earlier specs, READMEs, helper scripts or site names are assumed. CARLA scenes go in `rcx/scenarios.py`, surrogates in `search/`.

## Numbering decisions
- RC01/02/05/09/13 use the as-built project scenes; all others follow 20RCs.md.
- The original md RC13 (over-approximated predictions on the RC06 scene) has no number now; if needed later it becomes a planner-side wrapper over RC06.
