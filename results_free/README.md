# results_free: RC04 free-form pilot

Separate from `results_search/` (the verified structured-family data is untouched). Proposal: [docs/FREE_SEARCH_PROPOSAL.md](../docs/FREE_SEARCH_PROPOSAL.md). Code: `rcx/free_policy.py` (policy), `tools/run_free.py` (search and selection), `tools/summarize_free.py` (tables), `tools/verify_carla.py` (CARLA replay, unchanged except for extra fields).

**Status (2026-11-19): surrogate stage and CARLA stage done** (159 free + 90 structured + 3 mechanical runs, all completed, 0 errors). Not done: repeat seeds for near-boundary candidates, videos/plots, direct-CARLA search comparison. Tables: `rc04/summary.md` (`python tools/summarize_free.py`) and `rc04/diversity.json` (`python tools/free_diversity.py`).

## What exists

| Path | Content |
|---|---|
| `rc04/manifest.json` | frozen pilot manifest: 8 scenes in 3 roles, hidden-world set, budgets, evaluator |
| `rc04/search/free_pilot/` | free-family search per scene (`<scene>.json`) and `candidates.json` (159 candidates) for CARLA |
| `rc04/search/struct_matched/` | structured families re-searched on the same scenes at about the same budget (`candidates.json`, 90 candidates incl. world replays) |
| `rc04/search/free_cold/` | earlier free search without warm starts (kept as evidence) |
| `rc04/search/free_big/` | free search with about 6.7x budget on 4 scenes (sensitivity run) |
| `rc04/search/mech/` | mechanical test candidates (known brake policy as control points) |
| `rc04/logs/` | search logs |

## Surrogate-stage results (not CARLA)

- Mechanical test (surrogate): control points `from_brake(a)` reproduce `EscapePolicy(mode=brake, t_react=0.1)` exactly (same verdict, minimum gap and final speed on 3 scenes x 3 decelerations). CARLA version (`mech`): the brake family and the equivalent free policy give the same label and the same minimum front gap (9.992354634904444 m, identical), a different free policy gives a different outcome (contact), so parameters change the applied behaviour.
- Budget per scene: free 1446 cold + 810 warm robust evaluations (each = 5 surrogate rollouts), structured families about 1350 (3 families x 2 seeds). So free used about 1.7x the structured budget in total; the cold part alone is matched.
- **All-families-refuted scenes** (`near_obstacle` d_obs 42, `obstacle_40`, `obstacle_44`): the surrogate finds **no supported free policy** among 2189 unique samples per scene, and none among about 9400 samples per scene at 6.7x budget (`free_big`). Best free robust cost is no better than the structured best (about 1013 vs 1017, both contact). On the surrogate, free search does not open a new recovery here. The surrogate is only a proxy; the best / refuted / mitigation / random candidates of these scenes were run in CARLA (see below).
- **Cold iCEM in 12 dimensions is inefficient where the feasible band is narrow:** at `brake_boundary` (d_obs 48) the structured brake family passes, but cold free search finds 1 supported sample in 1403 (1 in 9371 at 6.7x budget). Warm starts from the structured policies (projected to control points) raise this to 157 supported samples. The first part of the pilot therefore favours a hybrid (structured warm start, then free refinement) over cold free search.
- **Multi-family and mixed scenes** (`no_follower`, `easy`, `truck_far_ahead`, `lane_change_boundary`): hundreds of supported free samples per scene; 2 to 3 behaviourally distinct supported policies kept per scene (behaviour distance >= 0.5, see `tools/run_free.py`). Free best cost is close to, not better than, the structured best (e.g. 1.2 vs 1.0, 0.6 vs 1.0).
- Hidden worlds (5 declared: `w0` nominal, `w1` side_acc -3, `w2` fol_react 0.5, `w3` both, `w4` side_acc -1.5 + fol_react 1.2; scenes without a follower use 3): in the surrogate, the kept supported policies are robust over all worlds in 17 of 24 kept supported policies overall (see `python tools/summarize_free.py`, last column); the CARLA replay is below.

## Deviations from the proposal (stated, not hidden)

1. **Continuation:** after the prefix (0.1 s to 5.1 s) the policy holds the last lateral target and holds the speed reached at the hand-over (the same `speed_hold(v, v)` as `a_post = 0` of the structured families); hand brake once stopped. It does not brake further, so a free policy must be stopped or at speed that is safe by 5.1 s. Total horizon 7 s as in the existing RC04 runs.
2. **Search methods:** iCEM on the surrogate only (cold and structured-warm-start). The direct-CARLA and multi-fidelity comparison of section 4 was not run; it needs CARLA time that was not available.
3. **Surrogate pad for free policies:** the obstacle clearance pad of the `brake` family (1.4 m) is used when the free policy never leaves its lane (lateral offset < 1.75 m), the lane-change pad (0.3 m) otherwise. This is an assumption; the CARLA stage measures its false-safe rate.
4. **Repeat seeds and videos** for near-boundary candidates are not done yet (CARLA stage). Hidden-world replays are part of the candidate list.
5. Follower headway is not varied (no law uses it, see `manifest.json`).

## CARLA results (reference labels)

All numbers from `rc04/summary.md`; CARLA label PASS = supported, anything else = not supported. The structured side has fewer candidates per scene (best and boundary per family, about 4 to 5 policies), the free side up to 5 to 6 selected for behavioural diversity, so counts are not a like-for-like budget comparison.

1. **Surrogate transfer fails the gate for free policies.** False-safe (surrogate PASS, CARLA not PASS): free iCEM-selected, nominal world 9 of 24 (38%, 95% CI 21 to 57%); free random-prior 6 of 9 (67%, 35 to 88%); free replays in other worlds 32 of 73 (44%). Structured families on the same scenes: 1 of 17 (6%) nominal, 1 of 50 (2%) replays. The gate for production use is < 10%, so the current surrogate must not be used to approve free trajectories; CARLA verification is required for each. False-negatives are fewer (free 1 of 17 selected, 7 of 13 in replays, small denominators).
2. **No added scene-level coverage.** In the three all-families-refuted scenes no free candidate passed in CARLA (best, mitigation, refuted and 4 random per scene; the surrogate found none either, also at 6.7x budget), and in every other scene a structured policy also passed. A free-form response did not recover a scene that the structured families could not. This does not prove those scenes are unrecoverable (limits: this parameterisation, this budget, the surrogate-guided search, 7 s horizon).
3. **Added behavioural diversity among CARLA-PASS policies** (nominal world, distinct = greedy picks with RMS distance >= 0.5 on speed and lateral offset at t = 1..7 s; `rc04/diversity.json`): `no_follower` 5 free vs 3 structured distinct behaviours (6 pooled), `easy` 3 vs 2 (4 pooled), `truck_far_ahead` 2 vs 4 (5 pooled), `brake_boundary` 2 vs 1 (2 pooled), `lane_change_boundary` 2 vs 2 (3 pooled). Pooling free and structured policies gives more distinct behaviours than the structured ones alone in all five supported scenes (pooled minus structured: +3, +2, +1, +1, +1 in the order above), but see the caveat on unequal candidate selection above.
4. **Robustness across the declared hidden worlds** (CARLA, same complete policy in all worlds, w0 supported only): free supported in all worlds: 13 policies (`no_follower` 4, `easy` 4, `truck_far_ahead` 1, `brake_boundary` 2, `lane_change_boundary` 2); structured: 17 policies (4, 2, 5, 2, 4). Free is not more robust.
5. **Random free policies** pass rarely (3 of 32 in CARLA); iCEM and warm starts are what produce supported ones, and cold iCEM failed in the narrow `brake_boundary` band (see above).

Interpretation for the go / no-go (proposal section 7): the free family adds a few CARLA-verified, behaviourally distinct supported policies in scenes the structured families already solve, adds no new recoverable scene, is not more world-robust, and needs CARLA verification for every candidate because the current surrogate is unreliable on it (38% false-safe). On this pilot the evidence does not justify extending free search to other RCs without a better surrogate or direct-CARLA / multi-fidelity search; that decision is the team's. Caveats: one speed (22 m/s), 8 scenes, 2 seeds, one continuation design, no repeat-seed check yet.
