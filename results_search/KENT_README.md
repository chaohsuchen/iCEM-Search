# results_search (Kent): quick guide for readers

> **In this repository:** Kent's RC03 (`RC03-S1-opt`) and RC07's earlier batches (`baseline`, `validation`, `pilot2`) are not included; RC03 here is Victor's. Their sections below are kept for reference. RC12 and RC19 `on_rc03` were built on Kent's RC03 scene.

**What this is.** For each Recovery Condition (RC) and each variation of its parameters (RC_scenarios_for_policy_search.md) we searched for the actions that recover it, or the best mitigation when none does. A fast surrogate per RC searches the policies; CARLA 0.9.15 then executes **every candidate policy**, and every label and every per-tick file comes from CARLA. The surrogate is scored against CARLA by a safety score (below). Layout and run folders follow `results_search_victor/`.

**How a batch is made**
1. **Variations.** N scene-parameter sets per site, drawn at random from the RC's ranges (`spec.sample` with the batch seed; continuous parameters uniform, discrete ones from weighted lists). Not a grid: N is set by the compute budget (RC07: 150 / 150 / 60 for sites A / B / E1).
2. **Search (surrogate).** Per variation and action family: the family's fixed settings, then a cross-entropy search over its policy parameters (10 per iteration, up to 3 iterations, stopping once a recovery has margin r >= 0.3). About 60 surrogate rollouts per variation and family.
3. **Verdict (surrogate).** Per family, the 3 best distinct recoveries (by margin r) are re-run in 9 perturbed surrogate worlds (below); the family's **best** is the one recovering in the most worlds, then with the largest worst-case r. With no recovery, the best mitigation (no contact first, then the lowest impact speed). The **boundary** is the family's smallest-margin recovery. Each gets PASS / FAIL / UNCERTAIN.
4. **Candidates** = every family's best and boundary policy per variation (`candidates_*.json`; RC07: ~7.6 per variation).
5. **CARLA.** Every candidate's policy runs closed-loop in CARLA; the run folder (per-tick data) and the label are CARLA's. The surrogate contributes only the policy parameters and its prediction (`surrogate` in summary.json), never trajectories or labels.
6. **Score.** Surrogate predictions against CARLA labels (safety score and coverage, below).

**Where to start**
1. `summary/carla_runs_<batch>.json` (per RC) - one row per CARLA run: variation, scene parameters, family, policy parameters, why it was selected, surrogate verdict, CARLA label, path to the run folder.
2. `<rc>/summary/recovery_profiles_<batch>.json` - per variation: status (recoverable / unrecoverable), the recovering families, the mitigation when nothing recovers, and per family the label, its source (carla / surrogate), the policy, which candidate it is (`kind`: a family recovers if its best *or* boundary candidate passed in CARLA) and, for re-run UNCERTAIN candidates, whether the label is `stable`.
3. A run folder `<rc>/carla/<batch>/runs/<NNN>_<variation>_<family>_<kind>/`:

| file | what it is | who may use it |
|---|---|---|
| `summary.json` | RC, variation ID, scene parameters, hidden parameters, family and policy parameters, scene seed and policy source, surrogate verdict, CARLA label + reason, contact, metrics | metadata |
| `ego_observable.csv` | per tick (0.05 s): ego state, observation (`obs_*`), per-actor visibility (`seen_<actor>_*`), requested acceleration and applied throttle / brake / steer | world-model input |
| `decisions.jsonl` | per decision tick: query time, observation, observation history frames, controller state, applied command, pending commands | world-model input |
| `ground_truth.csv` | per tick: true state of every actor, clearances, corridor margins (includes hidden information) | analysis / labelling only |

No `run.mp4`: these batches ran without the camera.

**Directories**

| path | meaning |
|---|---|
| `<rc>/search/<batch>/` | the surrogate search: `scenarios_<site>_s<seed>.json` (variations), `candidates_*.json` (every candidate with its surrogate verdict), `profiles_*.json` (per variation and family: best / boundary candidate), `runs_*.jsonl` (all rollouts) |
| `<rc>/carla/<batch>/` | the CARLA check of the same batch: `selected.json` (the candidates sent, with why), `results_shard*.jsonl` (one line per candidate: surrogate verdict and CARLA result), `runs/` (evidence folders), `surrogate_v*_verdicts.json` (verdicts of a later surrogate version on the same candidates) |
| `<rc>/summary/` | recovery profiles and the CARLA run index of each batch |

**Surrogate verdicts and the safety score.** Each candidate is re-run in the nominal surrogate and in perturbed surrogate worlds (ego start +-0.6 m / +-0.15 m/s, braking limit x0.9 / x1.1, braking tracking bias 0 / 0.8 m/s^2, steering lag x1.3 / x0.7, grip x0.95). PASS = recovers in all of them (lane changes: with r >= 0.18 in each), FAIL = in none, UNCERTAIN = the surrogate abstains and CARLA labels it. Safety score = 1 - (FP + FN) / decided, over the CARLA-checked PASS / FAIL verdicts; FP = surrogate PASS, CARLA not; FN = surrogate FAIL, CARLA PASS. Target 100%. UNCERTAIN candidates are not in the score, so it can rise by abstaining more: read it with the **coverage** = decided / all candidates.

**Batch `vsearch`: the combined pipeline (2026-11-18).** Victor's search (copied to `vsearch/`, `victor/` untouched: one Case per RC, iCEM on the worst-case cost over the case's perturbed scenes, success / mitigate / fail, best / boundary / ref candidates, every evaluation logged) with Kent's additions: stratified random variations at scale plus named presets plus boundary sweeps (bisection along the parameter that best separates a family's PASS from not-PASS), three-way verdicts (nominal + scene perturbations + ego-dynamics worlds; FP and FN scored, coverage reported, target 1.000), event-driven families, and every candidate run in CARLA with per-tick evidence. `python vsearch/search/pipeline.py <rc> --stage random|presets|boundary --batch vsearch`, then `carla_verify.py select / run / rerun / score / summarize --batch vsearch`. RCs: rc07, rc10 and rc11 (one scene, `RC10-S4-opt`; the RC is the reveal-time band), rc14, rc15.

**What each `vsearch` run carries** (`docs/dataset/PIPELINE_HANDOFF.md` section 5; full field list in `KENT_RUN_SCHEMA.md`):

| ID | Item | Where |
|---|---|---|
| R1 | every constraint the label was judged on (name, measured, required, pass) | `summary.json` → `constraints`, `pass_rule`; impact speed in `contact` (Q14) |
| R2 | hidden settings of the scene | `scene.hidden_params` (and `hidden_params`) |
| R3 | scene name apart from the RC, `uid`, `site`, `split` | `scene.name` (`blowout_onramp`, `spinning_lead`, `hidden_motorbike`, `lane_end_merge`), `scene.uid` = `variation_id`, `scene.site`, `scene.split` |
| R4 | a shared set of policies on every scene | each family's first fixed setting, in every scene and hidden world: kind `shared`, or under `best` / `boundary` when it is that policy; `carla/<batch>/shared_runs.json` lists all of them |
| R5 | each candidate in every hidden world of its history | `variation_id` `<history>~<world>`, `scene.history_id` / `hidden_world`: the mirror slide (RC10/11), the other bike speed (RC14), the other two driver responses (RC15), the hidden-behaviour perturbations (RC07) |
| R6 | more than one seed near the boundary | `carla/vsearch/repeats/`: every candidate with a CARLA margin within 0.1 of the threshold, again with the ego's start shifted +-0.6 m / +-0.15 m/s |
| R7 | the whole pre-roll | recorded from `-prefix_s`: 3 s (RC07, RC10/11), 4 s (RC14: the motorbike is seen in it, and the policy's history starts there), RC15 2 s at site A and 1 s at R (the longest each site's geometry allows) |
| R8 | `RUN_SCHEMA.md` | `results_search/KENT_RUN_SCHEMA.md` |

The `vsearch` run folders stay on the CARLA VM (`~/rc/results_search/<rc>/carla/vsearch/runs/`, tens of GB); the results, selections, summaries and scores are here.

**Selective CARLA (from 2026-11-19).** The batches run after the first ones (rc10 / rc11 `vsearch_rear`, rc14 `vsearch_rc14rear`, rc07's boundary stage, and every new RC below) send only chosen histories to CARLA, with every candidate of each one in every hidden world: about 40-50 per RC, picked from the surrogate profile as *defining* (the RC's story holds; about 40%), *contrast* (some family passes, some fails; about 35%), *hard* (nothing recovers; 15%) and *easy* (everything recovers; 10%), spread over presets / random / boundary. Repeats near the boundary (R6) only for those. `carla_verify.py select / rerun --keep <file of history ids>`; the lists are in each batch's `selected.json` (`variation_id`).

**New RCs in `vsearch` (2026-11-19).**

| RC | scene (`scene.name`) | what varies | families | hidden (R2) |
|---|---|---|---|---|
| rc03 | `RC03-S1-opt`, `spun_car`: a car beside the ego spins into its lane; 11 sites (straights, long and tight bends, Town13 held out), dry / wet / heavy | crash side and depth, distance, the escape-lane gap, gap leader braking, rear car | keep, brake, left, right, left_brake, right_brake | intrude, psi_rest, v_post, lead_brake, follow_react, rear_react |
| rc12 | rc03's scenes with the response held back `delay` s after the trigger (lane and speed kept; the delay prefix is scored) | every scene at delay 0, 0.3, 0.6, 0.9, 1.2, 1.5 s: the set of delays that still recover per scene | as rc03 | as rc03 |
| rc16 | `RC16-S1-opt`, `double_parked_van`: Town01 (Town02 held out); a van double-parked beside a traffic island, seen late past the van ahead | passage width 2.6-3.4 m and offset, how close the van ahead follows (reveal distance ~17-35 m), island length, speed 6-12 m/s | keep, brake, stop_end, shift_pass (early shallow / brake then shift / late sharp) | none: the geometry is observable once the van is seen |
| rc08 | `RC08-S1-opt`, `garbage_truck`: Town01 (Town02 held out); a truck stopped in the ego lane of a two-way street, an oncoming stream, a follower | truck distance, oncoming gaps and speed, follower headway / reaction / braking, when the truck leaves | keep, brake, stop_end (protected hold), overtake, wait_overtake (peek, go on a seen gap) | t_go, fol_react, fol_b (and oncoming cars hidden by the truck) |
| rc19 | rc03, rc10 (`vsearch_rear`) and rc07 anchors (12 each) re-run under A anchor / B appearance / C far traffic / D roadside objects / S semantic control (a broken-down car closing the lateral escape); `carla_rc/teachers/rc19_opts.py`, `vsearch/search/rc19_batch.py` | the case | the anchor's candidates (best / boundary / shared) | as the base scene |

**rc19 CARLA (2026-11-19):** B / C / D never change the CARLA label against A (0 of 224 anchor × family × kind groups: rc03 51, rc10 49, rc07 124); S turns PASS into FAIL for rc03 2, rc10 13, rc07 59 (and FAIL into PASS once, rc10). Surrogate vs CARLA: on_rc03 0.982, on_rc10 0.995, on_rc07 0.961 (S is nominal-only, no hidden worlds).

**rc08 / rc16 CARLA fixes (2026-11-19):** sites rc08 B / C and rc16 H named the opposite carriageway's lane (fixed: B lane -1, C lane 1, H lane 1; the opposing lane checked to be on the driver's left); the Town01 / 02 straights end before the horizon at speed, and the frame continues straight past the turn into the map edge (an unnamed collision), so for these two scenes the horizon comes where the real road ends, 10 m ahead (`end_at_road_end`, `info.road_end_t`; the terminal contract is scored there). Real road ahead of the origin (`frame.straight_from`, m): rc08 A 260, B 259, C 177, H 132; rc16 A 267, B 183, C 182, H 137. So at the held-out sites a fast family (keep, overtake) reaches the horizon after ~8-10 s, not at T. The INVALID runs, and every pre-fix run whose ego went past the road end (rc08 39 + 21 repeats, rc16 9 + 6; the map-edge collisions and three "contact with onc4" against a car driving on the straight extension), were dropped and rerun. After the fixes (selective runs + boundary repeats, 0 INVALID): rc08 1434 candidates, safety score 0.993 (FP 0, FN 2), UNCERTAIN 197 -> CARLA 146 PASS / 51 FAIL; rc16 383 candidates, 0.997 (FP 1: a `shift_pass` at B, clearance), UNCERTAIN 53 -> 34 / 19.

**rc03 complete fill (partial, 2026-11-19):** with the non-selected candidates added the surrogate makes 21 FP (score 0.936 on 327 decided), all at the bend sites (C1 / C1b / C2 / C2b / D1 / D2, binding clearance); the straights A and B have none. This is the known bend issue (backlog), not fixed here. The long fill batches also hit a CARLA server memory leak (a server grows to ~10 GB and the OOM killer takes it): `carla_verify.py` now restarts its server when the RSS passes 7 GB.

rc08 stores safety and route completion apart: the label is safety only; `metrics.goal.route_done` / `route_t` say whether the ego was past the truck within 20 s.

**rc18 from Victor's RC03** (`rc18/from_victor_rc03/`): the 11 of his 111 RC03 runs where the requested braking was limited, with requested / applied / measured acceleration per tick (README there).

**RC20: when is a library complete?** Only relative to a reference search: a library is complete for a scene when, whenever the search finds a policy that recovers in every hidden world, the library has one too. Labels: the library recovers → `R_REF`; only the search does → library incomplete there (`U_LIBRARY`); neither → unrecoverable *at that search budget*; missing evaluations → `UNRESOLVED`. Batch measure, with the shared setting per family as the library and CARLA labels (every scene and hidden world of the first `vsearch` batches): rc07 632 / 636, rc14 345 / 353, rc15 538 / 564, rc10 366 / 419, rc11 240 / 405 of the search's recoverable scenes are recovered by the library. A second check is saturation: more search budget or seeds stop adding recoverable scenes.

**Selection** (`why` in selected.json / carla_runs): `recommended` (each family's best), `uncertain` (every UNCERTAIN), `edge` (r < 0.1), `boundary` (the smallest-margin recovery), `handback` (hand-back options), `mitigation` (variations without a recovery), `sample` (20% random of the rest).

---

## rc07 - tyre blowout ahead, then an on-ramp car squeezes in (scene `RC07-S5-opt`)

Story: dense traffic, the lead car's tyre blows out (or it brakes benignly); a fast car may close from behind on the right, an on-ramp car may cut in from the acceleration lane, a pile-up may lie ahead, a rear car may follow close. Families: `keep`, `brake`, `right>return`, `right>stay`, `right>away` (lane change right, then return / stay / pull away). Sites on Town06 straights: A, B (train), E1 (held out); the on-ramp is a cone taper.

**Batch `baseline`** (seeds A 71, B 81, E1 91; 360 variations; 2748 candidates; 1933 run in CARLA, 4 servers, ~73 min)

| | result |
|---|---|
| variations | 360 of 360 recoverable, each by at least one family confirmed **in CARLA** |
| surrogate v1 (as searched) | safety score 0.983: 1330 decided, 22 FP, 1 FN; 603 UNCERTAIN (CARLA 440 PASS / 163 FAIL) |
| surrogate v2 (`surrogate_v2_verdicts.json`) | safety score **1.000**: 854 decided, 0 FP, 0 FN; 1079 UNCERTAIN (CARLA 893 PASS / 186 FAIL) |

What changed v1 -> v2 (fitted on 150 of these CARLA runs, so the 1.000 is in-sample until a fresh batch confirms it):
- **Braking.** CARLA brakes ~0.4 m/s^2 harder than requested (the accel map's brake side is weak and the controller's anti-windup clip drops the correction), so it stopped 3-6 m short: 16 of the 22 FP (rear contact or rear clearance) and the FN. The surrogate now adds that bias, with worlds at 0 and 0.8 m/s^2; brake-family max position error 5.4 -> 0.8 m.
- **Lane changes.** CARLA's lane change starts slower and overshoots by 0.3-0.5 m, which no steering lag or delay of the surrogate reproduces; the 6 remaining FP were lane changes CARLA failed by < 0.2 in r. A lane-change PASS now needs r >= 0.18 in every world, else it goes to CARLA. Cost: more UNCERTAIN candidates.

**Batch `validation`** (fresh seeds A 72, B 82, E1 92; 360 variations; surrogate v2 as above; 2763 candidates, 2104 run in CARLA) - the out-of-sample check of v2:

| | result |
|---|---|
| variations | 360 recoverable, each confirmed in CARLA (first reported 359: `A_s72_0065` only had a CARLA-passing *boundary* candidate, which the summary then ignored; a search run directly in CARLA, `carla/validation/carla_search/`, found recoveries in all three lane-change families, best r 0.60) |
| safety score | **0.994**: 890 decided, 5 FP, 0 FN; 1214 UNCERTAIN (CARLA 1026 PASS / 188 FAIL) |
| by family | `brake` 1.000 (350), `keep` 1.000 (154), `right>away` 0.975 (120, 3 FP), `right>return` 0.992 (130, 1 FP), `right>stay` 0.993 (136, 1 FP) |

**Completion (every candidate in CARLA).** The 815 + 659 candidates left out by the selection were then run too (`why` = `complete`), so both batches have a CARLA label and per-tick evidence for every candidate (2748 and 2763 run folders). Scores over all of them:

| batch | decided | coverage | FP | FN | safety score | by family |
|---|---|---|---|---|---|---|
| `baseline` (surrogate v1) | 2145 | 78% (of 2748) | 29 | 2 | 0.986 | `brake` 0.959 (19 FP, 1 FN), `keep` 1.000, lane changes 10 FP, 1 FN |
| `validation` (surrogate v2) | 1549 | 56% (of 2763) | 13 | 0 | **0.992** | `brake` 1.000 (470), `keep` 1.000 (355), `right>away` 0.971 (7 FP), `right>return` 0.982 (4 FP), `right>stay` 0.992 (2 FP) |

The UNCERTAIN candidates are mostly close calls in CARLA too (validation): of the 1026 CARLA recovers, 784 have r < 0.2 (decided PASS: 351 of 1120); of the 188 CARLA failures, 82 fail narrowly without contact (decided FAIL: 8 of 429). Their labels are CARLA's but borderline; see the stability check below.

**Stability of the UNCERTAIN labels** (`carla/validation/stability/`: `results_shard*.jsonl`, `stability.json`). Each of the 1214 UNCERTAIN candidates ran twice more in CARLA with the ego's start shifted by +0.6 m / +0.15 m/s and -0.6 m / -0.15 m/s (`carla_verify.py rerun --uncertain`; the run folders stay on the VM). CARLA itself is repeatable: two re-runs at zero offset reproduced their margins.

| | candidates | same label 3 of 3 (**stable**) | **borderline** (a re-run flips) |
|---|---|---|---|
| all UNCERTAIN | 1214 | 1039 (85.6%) | 175 |
| CARLA PASS | 1026 | 893 | 133 |
| CARLA FAIL | 188 | 146 | 42 |
| by family | `brake` 80 (26 borderline), `keep` 5 (0), `right>away` 388 (50), `right>return` 402 (38), `right>stay` 339 (61) | | |

Flips are not confined to thin margins: 24 of 242 candidates with CARLA r >= 0.2 flip, and 25 of 106 with r < -0.1. Of the PASS -> FAIL flips, 57 end in contact (mostly the on-ramp car against `right>away` / `right>stay`, 41; the lead car against `brake`, 9), at 0.1-11.5 m/s: a 0.6 m start shift changes when the on-ramp car commits to the gap. Use `stability.json` to tell a stable label from a borderline one. At variation level only `E1_s92_0035` has no stable recovery (its recoveries are all borderline).

With surrogate v2, braking and keep-lane verdicts hold on all 825 decided candidates; lane-change PASS verdicts are wrong 13 times in 724 (1.8%), 8 of them among the clear-cut candidates the selection had skipped. Lane-change labels in the summaries are therefore all CARLA's.

In the selected set, all 5 FP are lane changes (surrogate r_worst 0.18-0.49): CARLA's lateral speed peaks ~10% lower than the surrogate's, which shifts the hand-back time and, in `362`, lets the on-ramp car close in and sideswipe the ego. No single margin fixes it (zero FP would need r_worst >= 0.55, leaving 28 of 360 lateral candidates decided), nor does a yaw-gain world (catches 1 of 5). Braking verdicts held out of sample.

**Contact attribution.** In 26 runs (15 baseline, 11 validation) CARLA's collision sensor fired a tick before the boxes overlapped, and the run folder named the actor `unknown`. These were re-attributed to the nearest actor at the contact tick (`lead` or `onramp` in all but one `pileup`; the results files already had it), checked against a re-run of the two least clear-cut cases that read the actor from the sensor. Runs since then take it from the sensor.

---

## Future work

**Graded UNCERTAIN verdicts in surrogate validation (not implemented).** UNCERTAIN is scored as one bucket today, but a candidate that passes 13 of 14 surrogate worlds and fails in CARLA says more about the surrogate than one that passes 1 of 14. Every result line already stores what is needed (`surrogate.nominal`, `n_pass`, `n_worlds`, `r`, `r_worst`), so this is scoring only (about 1-2 h in `carla_verify.py score`, no CARLA runs):
- **Lean**: per candidate, the share of worlds that pass and the nominal verdict and margin. The worlds are pushed to the edges of plausible variation, so the share is not a probability; calibrate it against CARLA instead.
- **Reliability table**: CARLA's pass rate per lean bin (by `n_pass / n_worlds` and by `r`), per RC and family.
- **Leaning-wrong counts** (soft FP / FN), reported next to the hard ones: e.g. nominal PASS with >= 80% of worlds passing but CARLA FAIL. Not counted as hard FP: when CARLA sits closest to the one failing world, the surrogate flagged the risk correctly.
- **One weighted score**: the Brier score, mean of (lean - CARLA label)^2.
A family where the surrogate leans the wrong way often is the signal to improve (retrain / recalibrate) the surrogate there, not only to widen its UNCERTAIN band.

**Search quality under surrogate bias.** The search optimizes against the surrogate, so where the surrogate is biased (e.g. no lateral overshoot after `brake_free` or a merge) a family's best / boundary candidates can both fail in CARLA while another setting of that family would pass: a missed recovering family. Check: count variations where a family's best and boundary failed in CARLA but another candidate of that family passed. If frequent, improve the surrogate (lateral response, low-speed braking, grip at E1, RC15 target-lane traffic) and rerun the search.
