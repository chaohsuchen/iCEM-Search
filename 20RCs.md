# 20 RCs — scenes and variations

> Cases already built (RC01, RC02, RC05, RC09, RC13) follow the **as-built** scenes of this project; the rest follow the original design. RC03, RC04 and RC06 are also built and CARLA-verified (scene in `rcx/scenarios.py`, laws in `rcx/rc03_laws.py` / `rcx/rc04_laws.py`).

## The cases

Each case lists:
- **Scenario**: the story;
- **Options**: what the ego can do, and when each works;
- **Variations**: what to vary, with ranges;
- **Hidden at the trigger**: what the planner must redraw;
- **Notes**.

Units: speeds in m/s, distances in m, times in s.

### RC01 — Braking with a protected stop (as built; supersedes the roadworks story)

**Scenario (built, Town01). A delivery van stopped in a narrow one-way street.** Both lanes of Town01 road 15 are used as ONE one-way street.
Continuous rows of parked cars line both kerbs, pedestrians stay on the sidewalks, and a van (Sprinter) is stopped in the narrow corridor
between the rows. The ego cannot pass on either side, so it can only brake behind the van. The cue is the first time the van is *seen*.
- **RC01:** the trigger gap is longer than the stopping distance, so braking works.
- **RC02:** the gap is shorter, so nothing works (negative case; generated from the same scene by shrinking `gap`, not a separate scene).
- **RC17 (built on this scene):** the same scene with the ego's observation delayed by `delay_s`. This is *not* the md's RC17 (RC16 + lag); see RC17 below.

**Options.** Brake and hold at different deceleration levels (`brake`, `two_stage`, `surge_brake` families). Too gentle: stops short of the 2 m clearance
or hits the van. Steering is useless (parked rows).

**Variations (theta_s, CARLA names).**

| Parameter | Range used | Effect |
|---|---|---|
| `v_ego` | 6 / 8 / 9.5 | calibration speeds |
| `gap` (trigger gap to the van) | 8–20 | spans the RC01 / RC02 boundary |
| `delay_s` (RC17) | 0 / 0.3 / 0.6 | observation delay |

SCEN grid in `search/cases.py` (`RC01Case.SCEN`). **Hidden at the trigger:** none (fully observable). Terminal contract: stopped and holding 2 s, clearance >= 2 m.

### RC02 — Too late to stop

**Scenario.** The RC01 scene (as built: van in the narrow one-way street) with `gap` below the stopping distance: generated as part of RC01's `gap` range
(e.g. `g8_v9.5`), not as a separate scene. **Use:** negative case; every option fails; world-model data labelled unrecoverable; tests that the planner
reports "no recovering plan". Do **not** disable steering to make it negative: the parked rows make steering useless, which is the point.
Impact-speed reduction is mitigation, not recovery.

### RC03 — Braking fails but a lateral escape remains

**Scenario (built, self-contained): RC03-S1, the spun car.**
- **The crash:** in a lane next to the ego, a car is rear-ended and spins. It comes to rest deep in the ego lane,
  clipping its edge, or stays in its own lane (benign).
- **The ego's choices:** keep, brake in lane, or change into the other adjacent lane, where the gap between two cars
  may be open or closed. A tailgater may follow the ego.
- **Build:** this project defines the scene itself: the CARLA scene is written in `rcx/scenarios.py` and the surrogate in `search/lane_sim.py` from the tables below. No earlier RC03 spec, README or helper scripts are assumed.

**Options.**

| Option | Works when |
|---|---|
| Keep lane and speed | benign: the victim stays in its own lane |
| Brake in lane (optionally shifted away from the intruder inside the lane) | the crash is far enough (`d0` large, `v` low), or the victim only clips the lane edge |
| **Change into the other adjacent lane** | the gap there is open (`esc_gap` large, centred near the ego), its leader doesn't brake hard, and there's grip for the change |
| Change lane, ready to brake | as above, when the victim may keep sliding towards the target lane |
| Nothing | the crash is close and deep, and the gap is closed or the road is wet |

A tailgater (`rear`) makes hard braking risky, so it can close the brake option even when the distance allows a stop.

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| `side` | ±1 (crash on the ego's right / left) | |
| `crash` | 1 (5/6) / 0 (benign) | **hidden** until the victim moves |
| `v` (ego) | 14–30 | |
| `d0` (to the victim at t = 0) | 20–85 | |
| `intrude` (victim's resting depth, in lanes) | 0.45–1.6 | **hidden** |
| `psi_rest` (victim's final yaw) | 0.2–1.5 rad | **hidden** |
| `T_slide` (slide duration) | 1.0–3.2 s | **hidden** |
| `v_post` (victim speed after impact) | 9–26 | |
| victim vehicle | 5 models (car / van) | |
| escape lane: speed `v_esc` | 12–27 | |
| escape lane: gap `esc_gap`, its centre `gap_center` | 10–60, −10 to +5 | |
| gap leader's braking on seeing the crash `lead_brake` | 0–5 m/s² × grip | **hidden** until seen braking |
| gap follower: headway `follow_T`, reaction `follow_react` | 0.8–1.6 s, 0.5–1.1 s | **hidden** |
| tailgater `rear`: present, gap | 0 / 1, 10–60 | |
| tailgater: reaction `rear_react` | 0.4–1.2 s | **hidden** |
| road condition `wx` | dry / wet / heavy (tyre friction × 1 / 0.25 / 0.15) | |
| sites | 2 straight sections (Town06); 3-4 bends (Town04 / Town05, chosen when the scene is built); **one bend held out** for evaluation | |
| late-slide subset | `d0` 45–85, `intrude` 1.1–1.6, `T_slide` 2.0–3.2 (evaluation-only grid; the slide ends late, so waiting is tempting) | |

**Hidden at the trigger** (what the planner redraws):
- the spun car's final depth `intrude`, final angle `psi_rest` and slide duration `T_slide`, conditioned on its
  currently observed pose and lateral speed;
- the gap leader's braking `lead_brake`, until it has been seen;
- the gap follower's headway and reaction (`follow_T`, `follow_react`);
- the rear car's reaction (`rear_react`).

### RC04 — Staged slowing before a feasible lane change

**Scenario. A stopped car ahead, with a truck alongside in the escape lane.**
- **The hazard:** on a 2-lane motorway, the ego (right lane) approaches a stopped car in its lane, an earlier crash
  with hazard lights on.
- **Why each single option fails:**
  - a truck is driving in the left lane, roughly **alongside** the ego, so an immediate lane change hits it;
  - braking in lane cannot stop in time.
- **What works:** brake in lane first, so the truck pulls ahead, then change lane **behind** the truck before reaching
  the stopped car.
- **The trap:** a car following the truck may close that gap.

**Options.**

| Option | Works when |
|---|---|
| Brake in lane | the stopped car is far enough |
| Immediate lane change | the truck is well ahead or well behind |
| Accelerate and change ahead of the truck | the truck is slower than the ego and the room ahead is enough |
| **Staged: brake, then change behind the truck** | the truck is alongside and there's enough distance |
| Nothing | the truck is alongside, the stopped car is close, and the truck's follower is close behind it |

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| `v` (ego) | 22–30 | |
| `d_obs` (to the stopped car) | 40–100 | |
| truck speed − ego speed | −4 to +2 | |
| truck position relative to the ego | −20 to +20 (behind to ahead) | |
| truck length | 8–17 (rigid / articulated) | |
| truck's follower: gap | 10–50 | |
| truck's follower: headway, reaction | 0.8–1.6 s, 0.5–1.2 s | **hidden** |
| truck's response to the hazard | none / brakes 0–3 m/s² (rubbernecking) | **hidden** |
| benign share | the stopped car is in the left lane, or on the shoulder | |
| `side` | ±1 | |
| road condition, sites | as in RC03 | |

**Hidden at the trigger:** the truck's braking response, and its follower's headway and reaction.

**Notes.**
- The staged plan must run as one controller, not stitched segments.
- Lane change and braking share the grip budget. On wet roads that is what makes the immediate change fail, so this
  case also exercises RC18.

### RC05 — Aborting an unsafe lane change (as built)

**Scenario (built, Town04 road 40). Overtaking a slow car; a rear-left car accelerates when the ego is mid-change.**
- **Setup:** the ego (10 m/s, right lane) follows a slow lead (6 m/s, 35 m ahead) and starts a lane change left (`t_lc` 4 s).
- **The threat:** when the lane-change lateral progress reaches `hazard_progress` (the hazard cue), a car in the target lane, behind-left (`rear_gap` 25 m,
  closing +2 m/s), **accelerates** (`rear_acc` 2.5 m/s² up to 20 m/s).
- **Outcomes:** commit is unsafe; abort and return to the right lane works if the source gap behind the lead is kept.

**Families (surrogate / CARLA):** `abort` (ttc_abort, gap_abort, t_ret), `abort_surge` (abort + forward surge), `commit_surge` (commit + forward surge).

**Variations (theta_s).**

| Parameter | Range used |
|---|---|
| `hazard_progress` | 0.5 (0.8) |
| `rear_gap` | 12 / 25 / 40 |
| `closing` (rear car speed excess) | 2 / 4.5 |
| `rear_acc` | 2.5 (+0.5 in perturbation) |

**Hidden at the trigger:** the rear car's acceleration. Clearance rules: rear-left >= 1 m, lead >= 2 m. Lane-change prefix is physically executed.

### RC06 — Yield, then use a time-varying gap

**Scenario. A broken-down car ahead, with a car passing alongside.**
- **Setup:** on a 2-lane motorway, a car has broken down in the ego's lane 80–200 m ahead, so the ego must change
  lane.
- **The conflict:** a car in the other lane is passing the ego right now, so moving over immediately conflicts with
  it.
- **What works:** slow down, let it pass, and move in **behind** it, before the broken-down car.
- **What can close the gap:** a second car behind the passer.

Unlike RC15, the passing car does **not** react to the ego. Its motion follows an explicit law.

**Options.**

| Option | Works when |
|---|---|
| Immediate entry | the passer is well ahead or behind |
| Accelerate past it | the passer is slower and there's enough room |
| **Wait, then enter behind the passer** | the second car leaves a gap |
| Protected stop behind the broken-down car | the fallback: always safe if the distance allows, but costs the route |
| None | there's no room to wait, and the second car is close |

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| `d_block` | 60–200 | |
| `v` (ego) | 20–28 | |
| passer speed − ego speed | +1 to +8 | |
| passer position relative to the ego | −10 to +10 | |
| passer law | constant / accelerates / brakes on seeing the blockage | **hidden** |
| second car: present | yes / no | |
| second car: gap behind the passer | 10–60 | |
| second car: headway, reaction | | **hidden** |
| benign share | the broken-down car is on the shoulder | |
| `side`, road condition, sites | as in RC03 | |

**Hidden at the trigger:** the passer's and the second car's acceleration behaviour.

**Notes.** "Time to feasibility" is not monotonic in the parameters: sweep the passer's phase finely.

### RC07 — Primary avoidance with continued secondary-threat management

**Scenario (RC07-S5). A tyre blowout ahead, then an on-ramp car squeezed in accelerates into the ego.**
- **Setup:** the ego drives in the **left lane** of a two-lane motorway, about 1 s behind another car. On the right
  there is an **on-ramp acceleration lane**.
- **The primary hazard:** the car ahead **blows a tyre** and skids to a stop in the left lane. Braking in lane is too
  late, so the ego swerves into the right lane.
- **The secondary threat:**
  - a **fast car** is tailgating up the right lane, leaving room for just one car behind the ego;
  - the **on-ramp car**, about level with the ego, must merge before its ramp ends. About 2 s after the trigger it
    **cuts into that slot** between the ego and the fast car;
  - the fast car closes in on it, so the on-ramp driver **floors it** to avoid being rear-ended, straight at the ego,
    whom they haven't registered.
- **The real decision:** once past the stopped car, the ego must **return to the left lane** (or pull away) before the
  squeeze develops, rather than stay in the right lane.
- **Why the 3 s test is misleading:** staying in the right lane passes it, but the on-ramp car hits the ego about
  4–5 s after the trigger.

**Options** (`stage 1 > hand-back`; the hand-back is decided once the ego has passed the stopped car):

| Option | Works when |
|---|---|
| Keep | benign: the car ahead controls the blowout and stays mobile |
| Brake in lane | the headway to the car ahead is long, or the road is dry and the ego slow |
| Right > stay | the on-ramp car merges behind the fast car, or doesn't accelerate, or there is no fast car |
| **Right > return to the left lane** | the left lane past the stopped car is clear (no pile-up) and the return starts in time |
| Right > accelerate away | the ego can stay ahead of the on-ramp car's acceleration (enough speed margin and road ahead) |
| Nothing | a pile-up closes the left lane, the ego can't outrun the on-ramp car, and braking is too late |

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| `v` (ego) | 24–30 | |
| headway to the car ahead | 0.8–1.6 s | |
| blowout: deceleration, rest offset in its lane | 7–9 m/s² × grip, ±0.5 | |
| blowout: benign share | the car keeps control and slows gently (about 1 in 6) | **hidden** until it skids |
| fast car: present | yes (3/4) / no | |
| fast car: speed | ego + 3 to + 8 | |
| fast car: gap behind the ego at the trigger | 25–60 | |
| fast car: notice distance, reaction, braking | 15–40, 0.8–1.5 s, 3–6 m/s² × grip | **hidden** |
| on-ramp car: position relative to the ego at the trigger | −15 to +10 | |
| on-ramp car: speed | 25–29 | |
| on-ramp car: cuts in or waits behind the fast car | cut in (3/4) / wait | **hidden** until it moves |
| on-ramp car: cut-in time after the trigger | 1.5–3.0 s | **hidden** |
| on-ramp car: acceleration after the cut-in | 2.5–3.5 m/s² towards 35 | **hidden** |
| remaining ramp length | 80–200 | |
| pile-up: a second stopped car in the left lane | none (2/3) / 30–80 beyond the first | |
| left-lane follower behind the ego: present, gap, reaction | 0 / 1, 15–50, 0.6–1.2 s (reaction hidden) | |
| road condition | dry / wet / heavy | |
| sites | motorway on-ramps (Town06 and Town04 ramps, picked when the scene is built); hold one ramp out | |

**Hidden at the trigger:** whether, when and how hard the on-ramp car cuts in and accelerates; the fast car's notice
distance, reaction and braking; the left-lane follower's reaction.

**Notes.**
- **Paired contrast:** the pile-up closes the way back. Then only "accelerate away" or nothing remains.
- **Build:** write the CARLA scene as a new `RC07` class in `rcx/scenarios.py` and the surrogate on top of `search/lane_sim.py` (shared with RC03/RC04/RC06).
- **No mirroring:** on-ramps are on the right in right-hand traffic. Left-side on-ramps exist but are rare.
- **Horizon:** the planner's horizon **must cover the secondary threat**, so use 12 s with a terminal contract. With
  a short horizon the planner accepts the failing "stay", which is exactly what this case tests.
- **The on-ramp car's law:** it reacts to the fast car, not to the ego. The planner's worlds must keep that law; they
  must not assume the on-ramp driver will see the ego and yield.

### RC08 — Protected waiting despite route non-completion

**Scenario. A garbage truck stopped on a two-way street.**
- **Setup:** on a two-way, two-lane urban street (14 m/s), a garbage truck has stopped in the ego lane to load. It
  hides part of the oncoming lane.
- **Oncoming traffic** comes in a stream with gaps. A car follows the ego.
- **Safe behaviour:** stop behind the truck with a protected gap, so the follower can stop too. Then either:
  - overtake through the oncoming lane once a large enough gap is *seen*; or
  - wait until the truck moves on.
- **The route deadline** (20 s) may not be met.

**Options.**
- **Protected hold:** always safe when started in time.
- **Overtake into an oncoming gap:** safe only with enough gap *and* sight past the truck.
- **Overtake on a short gap:** fails.
- **Brake late:** the follower hits the ego.

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| truck stop duration | 5–60 s | **hidden** |
| oncoming headways | 2–10 s | |
| oncoming speed | 10–15 | |
| sight past the truck | truck size, ego's lateral position | |
| follower: headway | 0.8–2.0 s | |
| follower: reaction, braking | | **hidden** |
| `v` (ego) | 10–15 | |
| road condition, sites | town streets (Town03 / 05 / 10HD); hold one town out | |

**Hidden at the trigger:** how long the truck stays, the follower's reaction, and oncoming cars not yet visible.

**Notes.**
- **Store safety (`r`) and route completion as separate fields.**
- The planner's score must not trade safety for progress. A high progress weight rewards the risky overtake. 

### RC09 — Front/rear conflict with no supported escape (as built)

**Scenario (built, Town04 road 40). Single-lane tunnel: stopped queue ahead, faster truck behind that brakes late.**
- Tunnel walls (containers) close both sides. A **stopped queue** (3 cars) is `front_gap` ahead.
- A **truck** (Carlacola) approaches from behind at ego speed + `closing`, and brakes (`rear_decel` 5 m/s²) only after `rear_delay_s`.
- **Brake hard:** the truck hits the ego. **Brake too little:** the ego hits the queue. An intermediate, time-varying braking profile can work; in some cells nothing does.

**Families:** `brake` (d0, a1), `two_stage` (d0, a1, t1, a2), `surge_brake` (a0, d0, a1). Steering is useless (walls).

**Variations (theta_s).** Defaults: `v_ego` 8, `front_gap` 24, `rear_gap` 18, `closing` 4, `rear_delay_s` 1.4, `rear_decel` 5.

| SCEN | change |
|---|---|
| baseline | defaults |
| closing6 | `closing` 6 |
| delay2.0 | `rear_delay_s` 2.0 |
| rear12 | `rear_gap` 12 |
| front18 | `front_gap` 18 |
| closing6_front18 | both |

**Hidden at the trigger:** the truck's reaction delay and braking. The negative label holds only for the evaluated family class.

### RC10 — Early evidence enables a contingent escape

**Scenario (to build, reusing RC03's crash-path code). The car ahead spins: which way will it slide?**
- **Setup:** the ego follows a car closely (0.4–0.8 s headway) in the middle lane of a 3-lane road.
- **The hazard:** that car blows a rear tyre. It skids along its own lane at first, identically in both worlds, then
  slides to rest across the middle lane and into **either** the left or the right lane.
- **Why it's RC10:** the slide direction becomes visible **early** (`t_reveal` small). A shared first step (brake)
  keeps time to see the direction, then the ego escapes to the free side.

**Options.**
- Brake first, then escape left or right once the direction is seen. This is the contingent plan.
- Commit early to one side: fails in one world.
- Brake in lane: too close to stop.

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| `t_reveal` | 0.2–0.8 s (RC10 band) | |
| slide direction | L / R | **hidden until revealed** |
| headway | 0.4–0.8 s | |
| `v` (ego) | 20–28 | |
| adjacent traffic | per side: one side may be closed too, so a single escape remains | |
| benign share | the car regains control in its lane | |
| road condition, sites | as RC03 | |

**Hidden at the trigger:** the slide direction, until `t_reveal`.

**Notes.**
- Reuses RC03's crash-path code and sites.
- The planner must redraw the direction (L / R) in its worlds until it has been observed. A planner that knows it would
  "solve" RC11 too, which is wrong.

### RC11 — Late evidence prevents robust causal recovery

**Scenario.** The RC10 scene with the direction shown **late** (`t_reveal` 1.0–1.6 s):
- each world alone is solvable;
- no plan decided on what's observable works in both.

The same parameters as RC10 (same scene as RC10), with `t_reveal` sweeping across the RC10 / RC11
boundary.

**Use.**
- A **negative / information** case: it stays out of teacher training.
- The **best test that a planner is not clairvoyant.**
  - **Correct:** a planner that only uses observable information fails here, in some world.
  - **Wrong:** a planner that succeeds in every world is cheating.

### RC12 — Recoverability lost by delaying intervention

**Scenario.** RC03-S1, with the ego keeping its lane and speed for `Δ` after the trigger before the recovery starts.

**Variations.** `Δ` = 0–1.5 s in 0.1 s steps, on the same scenes.

**Status.** Largely covered already:
- For each scene and each decision time, record which options still recover (computed by the search and verified in CARLA).
- To add: run the delayed hand-over with the **planner**, scoring the delay prefix too.

**Notes.**
- Keep all delays of one scene in the same split, grouped by scene id.
- Record the full set of delays that work. Moving traffic can make it disjoint, not a single threshold.

### RC13 — Pedestrians on the sidewalk: stay fast unless someone darts (as built)

**Scenario (built, Town03 road 3). Pedestrians walk along the sidewalks; one may step into the lane.**
- The ego drives at ~8 m/s in its lane. Three pedestrians walk on the sidewalk (lateral ~5.9–6.2 m, speeds 1.1–1.4 m/s).
- **Benign:** nobody enters the lane. A good policy keeps speed (>= 0.9 × target at the pass); needless slowing or stopping is a failure ("unnecessary slowing").
- **Dart:** the target pedestrian steps toward the lane centre at `dart_speed` once the ego front is within `dart_dist` of it, and stands there.
  Needed: clearance >= 0.5 m, no contact. Pedestrian lateral speed ramps up with a ~0.18 s time constant (fit on CARLA runs, used by the surrogate).
- A policy must satisfy both worlds: robust cost = worst over benign and dart.

**Families:** `reachable_set` (v_ped_max, horizon, pad, a_max), `pre_slow` (v_slow, zone, a_react), `react` (thr, a_react, d_react).

**Variations (theta_s).** `dart2@20` (2 m/s, 20 m), `dart3@15`, `dart4@12`, `dart5@9`; each paired with its benign companion.

**Hidden at the trigger:** whether and how fast a pedestrian darts. Physics envelope: filtered braking <= 1 g.

*(The earlier md RC13, over-approximated predictions on the RC06 scene, is dropped as a numbered case; the idea can return as a planner-side wrapper later.)*

### RC14 — Current-view aliasing resolved by past observations

**Scenario. A motorbike seen earlier, now hidden behind a van.**
- **The hazard:** on a 2-lane motorway, the ego must move left to avoid a stopped car ahead in its lane.
- **The earlier sighting:** 1–3 s before the decision, a **motorbike** was visible in the left lane, either approaching
  fast or cruising slowly. Then it went out of view behind a van in the left lane.
- **At the decision:** the current view is identical in both histories.
  - **Bike approaching fast:** it will arrive in the gap, so the ego must brake and wait.
  - **Bike cruising slowly:** an immediate lane change is safe.
- **What decides it:** only the memory of the earlier sighting.

**Options.**
- **Immediate lane change:** works in the slow world.
- **Brake, then change after the bike passes:** works in the fast world, if there is room.
- **Brake in lane:** works only when the stopped car is far.

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| bike's past speed | fast 30–40 / slow 18–24 | **observed earlier, hidden now** |
| time since last seen | 0.5–3 s | |
| van size and position (the occluder) | | |
| `d_obs` | 40–100 | |
| `v` (ego) | 22–28 | |
| benign share | the bike takes an exit | |
| `side`, road condition, sites | as RC03 | |

**Hidden at the trigger:** the bike's current position and speed, while occluded.

**Notes.**
- The policy needs the **observation history**. Give the network a history window, and give the planner the full
  observed history.
- The planner's redraw is exactly RC03's: condition on everything observed so far, here on the bike's earlier speed.
- **Diagnostic:** a final-frame-only policy should fail where the histories differ.

### RC15 — Yielding must be observed, not assumed

**Scenario (to build). The motorway on-ramp merge.**
- **Setup:** the ego joins from an acceleration lane, with a car alongside in the motorway's right lane.
- **How that driver responds** to the ego's signal and nudge:
  - **yield:** slows, opening a gap ahead;
  - **hold:** keeps its speed;
  - **match:** speeds up to stay level.
- **The witness:** watch the response, then merge in front, or drop back and merge behind.
- **The trap:** `always_merge` assumes a yield and crashes in the match world.
- **Classes** (as chosen for this project): merge-left and merge-right only. Each is split into four classes, each
  with a fixed response and outcome:
  - hold → the ego speeds up and merges in front;
  - hold → the ego can't speed up and merges behind;
  - match → merges behind;
  - yield → merges in front.

**Options.**
- Merge in front.
- Speed up, pass, merge in front.
- Drop back and merge behind.
- Protective stop before the lane end.
- (Assumed yield: fails.)

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| response mode | yield / hold / match | **hidden until seen** |
| reaction delay | 0.5–1.0 s | **hidden** |
| response magnitude | subtle / pronounced | |
| acceleration-lane length | 60–250 | |
| adjacent car: speed | | |
| cars ahead of / behind the adjacent car | | |
| tailgater behind the ego | | |
| road condition, sites | | |

**Hidden at the trigger:** the mode and reaction delay. The planner must redraw them until the response is observed.

**Notes.** A replayed yielding trajectory is never evidence for a successful merge. The same exogenous mode must apply
to every candidate plan.

### RC16 — Centre-point clearance but body-footprint conflict

**Scenario. Squeezing past a double-parked van.**
- **Setup:** on a one-way city street with one traffic lane, a delivery van is double-parked in the right part of the
  lane. Beside it, a traffic island with bollards leaves a **passage of 2.2–3.0 m**, offset 1.0–1.5 m from the ego's
  lane centre.
- **The hazard:** the ego follows a box truck, which hides the van until the truck swerves around it, so the van is
  seen only `d_see` ahead.
- **What the ego must do:** shift sideways into the passage and be **straight** while passing the van, which is about
  6 m long. The passage is a corridor, not a point gate.

**Why the centre point is not enough.** A car of length L and width B, rotated by a heading ψ, covers
`L·sin ψ + B·cos ψ` across the road. For a Lincoln MKZ (4.9 × 1.85 m):

| Heading ψ | Width covered |
|---|---|
| 0 | 1.85 m |
| 0.1 rad (6°) | 2.33 m |
| 0.2 rad (11°) | 2.79 m |
| 0.3 rad (17°) | 3.22 m |

- An S-shaped shift of Δy over a distance D reaches a peak heading of about `π·Δy / (2D)`.
- **Late swerve:** Δy = 1.2 m over D = 10 m peaks at about 0.19 rad. The car is still angled when its front enters
  the passage. The **centre** stays inside a 2.6 m passage with room to spare, but the **corners** clip the van or a
  bollard. This is the RC16 refutation: the point path clears, the body doesn't.
- **Early shift:** the same shift over 25 m peaks at 0.075 rad, and the car is straight before the van.

**Options.**

| Option | Works when |
|---|---|
| Early, shallow shift | `d_see` is long enough to complete the S and straighten before the van |
| **Brake, then shift** | `d_see` is short. At low speed the grip allows a tighter S, so the car still straightens before the van |
| Late, sharp swerve | never when the passage is narrow (the body clips); only when it is wide |
| Stop behind the van | always safe; costs progress (the van stays 20–90 s) |
| Nothing | — (stopping always remains, if `d_see` allows it) |

**Variations.**

| Parameter | Range | Hidden? |
|---|---|---|
| passage width | 2.1–3.2 | |
| passage offset from the lane centre | 0.8–1.6 | |
| van position: side of the street | right / left (`side` ±1) | |
| van length | 5–7 (van / small truck) | |
| `d_see` (van first visible) | 12–50 (set by the box truck's headway and swerve) | |
| `v` (ego) | 6–12 | |
| bollard spacing | continuous kerb / bollards every 1.5–3 m | |
| van departs | 20–90 s | **hidden** |
| ego vehicle | sedan vs van blueprint, as a **separate condition** (length changes the swing) | |
| road condition | dry / wet | |
| sites | narrow one-way city streets (Town01 / 02 / 10HD); hold one out | |

**Hidden at the trigger:** essentially none: the geometry is observable once the van is seen. The van's departure
time only matters for the stop option.

**Notes.**
- The planner already checks clearance with the **oriented body footprint** (the surrogate's footprint distance uses the rotated box).
  The case's contrast is a planner that checks only the centre point, or a circle of radius B/2: that planner should
  pick the late swerve and fail in CARLA.
- Also test the network: it must learn to shift or brake **early**, from `d_see` and the passage width.
- Inspect front and rear swing in the trace, and corroborate contacts with the collision sensor.
- One failing manoeuvre does not make the scene negative. The label comes from searching all options.

### RC17 — Response delay eliminates an earlier recovery option

> **Status.** An RC17 delay variant is already built on the *RC01 van scene* (`delay_s`, see RC01). The scenario below (RC16 + lag) is the planned one and is not built yet.

**Scenario: RC16's squeeze past the double-parked van, with a lag added to the ego's control loop.** The scene,
actors and parameters are RC16's. What changes is the ego's **observation delay** and **command delay**: everything it
sees is late by `δ_obs`, and everything it commands is applied `δ_cmd` later, on every tick of the episode.

- **Without lag:** at a short `d_see`, "brake, then shift" threads the passage. The car straightens before the van
  thanks to quick steering corrections.
- **With 0.3 s of lag:** each correction arrives late. The car overshoots the shift or is still angled at the van, and
  a corner clips it.
- **Stopping behind the van still works.** So the lag **removes an option** but the scene stays recoverable (success
  through the stop).

**How it differs from RC12 and RC11.**

| Case | What delays the ego | When it acts | What is lost |
|---|---|---|---|
| RC12 | the ego **chooses** to carry on for Δ (0–1.5 s) before recovering | once, at the start | options shrink with Δ; possibly all of them |
| RC17 | the **car's sensing and actuation** lag δ (0–0.3 s) | on every tick, through the whole manoeuvre | an option that needs fast corrections; others remain |
| RC11 | **nothing**: the evidence itself comes too late | — | no causal plan works in both worlds |

RC16 is fully observable (no hidden mode to wait for). So a failure under lag is a timing effect, not missing
information. On a scene with a hidden mode, like RC10's, an observation delay would just reveal the mode later and
look like RC11.

**Variations.**

| Parameter | Range |
|---|---|
| `δ_obs` (observation delay) | 0 / 0.15 / 0.30 s, on the 0.05 s tick grid; then refine around the crossing |
| `δ_cmd` (command delay) | 0 / 0.15 / 0.30 s, varied separately from `δ_obs` |
| scenes | RC16 scenes with the same `params` at every delay; mainly short `d_see` and narrow passages, where "brake, then shift" is the option at stake |
| everything else | as RC16 |

**Use.**
- **The required contrast:** the same policy passes at the short measured delay and fails at the long one, on the
  same physical scene.
- **For the planner:**
  - plan with the lag modelled, and check it either still threads the passage, or correctly falls back to the stop;
  - plan as if there were no lag: it should pick "brake, then shift" and clip the van in CARLA.

**Notes.**
- **Model the lag in the planner:** its surrogate must include the **same delays**, as an observation buffer and a
  command queue. Otherwise it plans as if the car responded instantly.
- **Don't freeze the world:** during the delay the previous command continues.
- **Log the realised delays**, per tick, not only the configured ones. Keep capture, delivery, decision, command and
  response times distinct. No backdated sensor packets.
- **Keep all delays of one scene in the same split**, grouped by scene id.
- **Paired contrast:** a wider passage, as a separate physical variant. There the lag no longer matters.

### RC18 — Requested control exceeds available authority

**Scenario.** RC03-S1 on the **bend sites** (the RC03 bends) in **wet and heavy rain**. An evasive manoeuvre that is
fine geometrically needs more grip than the road has: lateral, or braking plus the bend's own lateral demand.
- **Fails:** a quick lane change at bend speed.
- **Works:** a staged or slower one.

**Variations.** Already in the RC03 data: sites × road condition × speed. To add:
- per tick, log the **requested** and the **applied** acceleration;
- log when the envelope governor or limiter intervened.

**Notes.**
- This is the hard case for the surrogate: on the held-out bend in heavy rain, compare the surrogate's verdicts with CARLA for lane changes and for plain braking.
- The open question is how the surrogate splits grip between braking and following the bend. Treat RC18 as the test of
  that.

### RC19 — Benign and label-preserving nuisance controls

**Scenario: RC03-S1, re-rendered with nuisance changes.** Take a subset of RC03-S1 scenes (for example 150 scenes
from the RC03 evaluation set: recoverable and unrecoverable, all sites). Run each scene, with the **same `params`**, under four benign cases
plus one semantic control. Using RC03-S1 rather than a new scene makes the comparison direct: same planner, same
network, same labels. Benign objects are spawned with a `bg_` name prefix so they are never counted as relevant actors, and weather and colours are set by a single helper.

**Variations** (one scene × five cases):

| Case | What changes | Ranges |
|---|---|---|
| A anchor | nothing | the RC03-S1 scene as stored (dry, clear, noon) |
| B appearance | paint colours of every car (ego, victim, striker, gap cars, tailgater); sun position and clouds; CARLA map layers (buildings, foliage, props, parked vehicles) on / off | sun altitude 25–80°, never low ahead of the ego; cloudiness 0–80 %; no rain, no wet road |
| C distant traffic | cars that never interact: on the opposite carriageway, or ≥ 2 lanes away from the ego, ≥ 80 m ahead or behind | 0–8 cars, speeds 0.8–1.1 × traffic |
| D roadside objects | cones, parked cars or signs on the verge or shoulder | ≥ 2 m from any lane used; not on any sight line to a relevant actor (check with CARLA ray casts from the ego camera and sensor positions) |
| S semantic control | something that **should** change the label | a broken-down car in the escape lane beyond the gap (closes the lane change); or a second car inside the gap |

**Never benign:** rain, a wet road, fog, night, low sun ahead, a different size of a relevant actor (e.g. the victim
as a van instead of a car), anything on a sight line to a relevant actor. These change physics or information, so
they are variations of RC03 itself, not RC19.

**Use.**
- **Check the labels:** run every option in CARLA in all five cases. A–D must give the same recovery profile as A,
  within tolerance. S must change it.
- **World model and camera-based policies:** their predictions must agree across A–D, and change in S.
- **Planner:** on ground-truth state, the A–D actors don't enter its worlds, so it is invariant by construction. If
  the distant cars of C *are* in its observation (realistic), it must still pick the same option and not intervene
  more often. That catches an over-cautious planner.
- **Paired data:** keep all five cases of one scene in the same split, grouped by scene id.

### RC20 — Incomplete library must remain unresolved

**Scenario.** The RC04 scene: brake, then change behind the truck. What varies is **search coverage**, compared three
ways:
- a restricted family library: immediate lane changes and straight braking only;
- the full family library, including the staged option;
- the planner's search.

**Use.**
- **The label:**
  - **restricted library alone:** "unrecoverable within the library" at most, or `UNRESOLVED` when some evaluations are missing;
  - **once the staged plan is found:** recoverable (success).
- **The argument for policy search:** the search finds the staged plan without being told it exists.

**Notes.**
- Store the library version with every label.
- Missing evaluations are NaN, **never** failure.

---