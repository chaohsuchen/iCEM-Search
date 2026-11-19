"""Gymnasium environment: one scenario class on its real CARLA site, run as a ScenarioRunner scenario.

Each episode samples a variation (or takes one via reset(options={"params": ...})), spawns the ego and builds a
`TeacherScenario` (a ScenarioRunner BasicScenario): its actors are spawned through CarlaDataProvider and moved by the
`StepScenarioLaws` atomic behaviour along the notebook's laws (kinematic, physics off, reacting to the CARLA ego); its
test criterion is ScenarioRunner's CollisionTest. The scenario tree is ticked by this environment, Leaderboard style
(world.tick -> CarlaDataProvider.on_carla_tick -> scenario_tree.tick_once), so no scenario_runner.py process and no
world reload per episode are needed.

Ego: CARLA physics. Action (10 Hz) = [u_acc, u_yaw] in [-1, 1]^2 -> commanded acceleration a in [-8, 3] m/s^2 and yaw
rate r in [-r_max(v), r_max(v)], r_max = min(0.6, 7 / v) rad/s (lateral acceleration <= 7 m/s^2); executed at 20 Hz
by the calibrated throttle/brake map and yaw-rate steering controller from carla_rc/bridge.py.
Specs with `envelope = True` use the vehicle envelope instead (envelope.py): z -> (a_x, a_y) inside the 1 g contract ∩
the qualified capability, rate-limited, yaw rate = a_y / v; the measured acceleration is monitored and a contract
violation fails the episode (info["envelope"]). With spec.governor the request is also scaled down while the measured
acceleration is near the contract (_govern): the low-level controller can deliver more than was asked.

Reward (per 0.1 s step) mirrors the notebook's r-margin terms: alive bonus, clearance, road-edge and friction-circle
penalties, smoothness and speed-keeping; -20 and termination on contact or leaving the corridor; at the horizon +10 if
the terminal contract (lane-centred, straight) holds, else -5. info["r_nb"] = the notebook's episode r.

Every episode is logged (per 0.05 s tick: ego and actor states in the scenario frame and the world, controls,
actions, margins; per 0.1 s step: observations and rewards) as a compressed .npz in log_dir.
"""
import json
import math
import os
import sys
import time

import gymnasium as gym
import numpy as np
import py_trees

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE),
                os.environ.get("SCENARIO_RUNNER_ROOT", os.path.expanduser("~/scenario_runner")),
                os.path.join(os.environ.get("CARLA_ROOT", os.path.expanduser("~/carla0915")), "PythonAPI", "carla")]

import carla                                                                    # noqa: E402
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration  # noqa: E402
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider        # noqa: E402
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior  # noqa: E402
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest  # noqa: E402
from srunner.scenariomanager.timer import GameTime                               # noqa: E402
from srunner.scenarios.basic_scenario import BasicScenario                      # noqa: E402

from bridge import EGO_BP, WARMUP_TICKS, YAW_KP, AccelMap, LongCtrl, gear_for     # noqa: E402
from envelope import (G0, GOV_AX_G, GOV_G, GOV_MIN, GOV_RECOVER, V_MIN_YAW, Envelope, Monitor,  # noqa: E402
                      RateLimiter)
from frame import LaneFrame                                                     # noqa: E402
from laws import DT, EGO_L, EGO_W, Body, obb, signed_distance, track_y          # noqa: E402
from scenarios import SPECS                                                     # noqa: E402

C0, T_H = 0.5, 0.3                 # notebook clearance reserve [m] and closing-speed look-ahead [s]
MU_G = 7.0                         # friction-circle authority used by the notebook's dyn term [m/s^2]
A_MAX, B_MAX = 3.0, 8.0
POLICY_TICKS = 2                   # 10 Hz policy on the 20 Hz simulation
LARGE_MAPS = ("Town12", "Town13", "Town15")   # tiled maps (streamed around the hero / spectator)


def z_off(b):
    """Height of the actor's origin above the road: walkers are spawned at their centre."""
    return 0.95 if (b.bp or "").startswith("walker.") else 0.0


def write_card(out, spec):
    """<out>/spec.json: the scenario card (site, nominal, variation ranges, observation names)."""
    os.makedirs(out, exist_ok=True)
    json.dump(spec.card(), open(os.path.join(out, "spec.json"), "w"), indent=1, default=float)


class StepScenarioLaws(AtomicBehavior):
    """Moves every scenario actor one tick along its law / scripted path (always RUNNING)."""
    def __init__(self, env, name="StepScenarioLaws"):
        super().__init__(name)
        self.env = env

    def update(self):
        self.env.step_bodies()
        return py_trees.common.Status.RUNNING


class EndAfterHorizon(AtomicBehavior):
    def __init__(self, env, name="EndAfterHorizon"):
        super().__init__(name)
        self.env = env

    def update(self):
        return py_trees.common.Status.SUCCESS if self.env.t >= self.env.spec.T - 1e-6 else py_trees.common.Status.RUNNING


class TeacherScenario(BasicScenario):
    """ScenarioRunner scenario for one variation of a scenario class (actors and laws from the env's scene)."""
    def __init__(self, env, world, ego):
        self.env = env
        self.timeout = env.spec.T + env.spec.prefix_T + 10.0
        cfg = ScenarioConfiguration()
        cfg.name, cfg.type, cfg.town = env.spec.sid, "TeacherScenario", env.spec.town
        cfg.weather = carla.WeatherParameters(**env.spec.weather)
        super().__init__(env.spec.sid, [ego], cfg, world, debug_mode=False, terminate_on_failure=False,
                         criteria_enable=True)

    def _initialize_actors(self, config):
        env = self.env
        for b in env.body_list:
            if b.bp is None:
                continue
            a = env.spawn(b.bp, env.frame.to_world(*b.position, b.heading, z_off(b)), b.name, color=b.color)
            a.set_simulate_physics(False)
            self.other_actors.append(a)
            env.actors[b.name] = a
        for bp, x, y in env.scene["props"]:
            a = env.spawn(bp, env.frame.to_world(x, y), "prop", required=False)
            if a is not None:
                self.other_actors.append(a)

    def _create_behavior(self):
        root = py_trees.composites.Parallel("ScenarioLaws", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(StepScenarioLaws(self.env))
        root.add_child(EndAfterHorizon(self.env))
        return root

    def _create_test_criteria(self):
        self.collision = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)
        return [self.collision]

    def __del__(self):
        pass


class TeacherEnv(gym.Env):
    metadata = {"render_modes": []}
    start = (0.0, 0.0)             # extra ego start offset (x [m], v [m/s]): CARLA label stability (carla_verify rerun)

    def __init__(self, sid, host="localhost", port=2000, seed=0, log_dir=None, render=False, timeout=60.0):
        self.spec = SPECS[sid]
        self.rng = np.random.default_rng(seed)
        self.seed_, self.log_dir, self.render_ = seed, log_dir, render
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        world = self.client.get_world()
        if world.get_map().name.split("/")[-1] != self.spec.town:
            s = world.get_settings()
            if s.synchronous_mode:            # a map load in synchronous mode waits for a tick nobody sends
                s.synchronous_mode = False
                world.apply_settings(s)
            self.client.load_world(self.spec.town)
            for _ in range(60):
                world = self.client.get_world()
                if world.get_map().name.split("/")[-1] == self.spec.town:
                    break
                time.sleep(2)
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, DT
        s.substepping, s.max_substep_delta_time, s.max_substeps = True, 0.01, 10
        s.no_rendering_mode = not render
        world.apply_settings(s)
        world.tick()
        # this server is dedicated to one environment: remove what an earlier (killed) process left in the world
        stale = [a.id for a in world.get_actors()
                 if a.type_id.startswith(("vehicle.", "walker.", "sensor.", "static.prop.", "controller."))]
        if stale:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in stale], False)
            world.tick()
        self.world = world
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(world)
        site = self.spec.site
        self.frame = LaneFrame(world.get_map(), site["start_xy"], site["ref_lane"], site["ref_idx"], site["s0"],
                               curve=site.get("curve", False))
        self.W = self.frame.W
        k = np.gradient(self.frame.H, self.frame.S)
        self._kappa = np.convolve(k, np.ones(9) / 9, "same")      # road curvature along the frame [1/m]
        self.spec.on_frame(self.frame)
        if self.spec.town in LARGE_MAPS:          # stream the site's tiles in before the first synchronous tick
            s = world.get_settings()               # (otherwise that tick waits on the streaming and times out)
            s.synchronous_mode, s.no_rendering_mode = False, False
            world.apply_settings(s)
            tf = self.frame.to_world(0.0, self.frame.y_ego)
            world.get_spectator().set_transform(carla.Transform(tf.location + carla.Location(z=60.0),
                                                                carla.Rotation(pitch=-60.0, yaw=tf.rotation.yaw)))
            time.sleep(45.0)
            s.synchronous_mode, s.no_rendering_mode = True, not render
            world.apply_settings(s)
            world.tick()
        self._weather = None
        n = len(self.spec.obs_names)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.scenario, self.ego_actor, self.episode = None, None, 0
        self._init_envelope()

    def _apply_road(self):
        """Per episode: the road condition's envelope (spec.road_mu(params), dry when the spec has none)."""
        mu = self.spec.road_mu(self.params) if hasattr(self.spec, "road_mu") else 1.0
        if mu != self.mu:
            self._init_envelope(mu)

    def _init_envelope(self, mu=1.0):
        """The envelope of the road condition mu (envelope.ROAD) and the controller's map recalibrated to it."""
        self.mu = mu
        self.amap = AccelMap(EGO_BP)
        self.envelope = Envelope(EGO_BP, mu) if getattr(self.spec, "envelope", False) else None
        if self.envelope:
            self.limiter, self.monitor = RateLimiter(self.envelope, DT * POLICY_TICKS), Monitor(DT)
            lv = getattr(self.envelope, "brake_levels", None)
            if lv and all(lv.values()):
                # brake half of the controller's map from the qualified fixed-pedal runs (sustained decel per level)
                A, P = self.amap.A, self.amap.p
                vs = sorted(lv)
                p_min = min(p for v in vs for p, _ in lv[v])
                for j, pj in enumerate(P):
                    if pj > -p_min + 1e-6:
                        continue
                    meas = [float(np.interp(-pj, [p for p, _ in sorted(lv[v])], [b for _, b in sorted(lv[v])]))
                            for v in vs]
                    A[:, j] = -np.interp(self.amap.v, vs, meas)
                self.amap.A = np.maximum.accumulate(A, axis=1)                 # monotone in pedal

    # ---------------- CARLA helpers ----------------
    def spawn(self, bp, tf, role, color=None, required=True):
        if not self.world.get_blueprint_library().filter(bp):
            assert not required, f"no blueprint {bp}"
            return None
        a = CarlaDataProvider.request_new_actor(bp, carla.Transform(tf.location + carla.Location(z=0.3), tf.rotation),
                                                rolename=role, color=color, tick=False)
        if a is None:                         # overlapping spawn: spawn below the road, then teleport
            low = carla.Transform(tf.location + carla.Location(z=-60.0 - 5.0 * len(self.actors)), tf.rotation)
            a = CarlaDataProvider.request_new_actor(bp, low, rolename=role, color=color, tick=False)
        assert a is not None or not required, f"could not spawn {bp} for {role}"
        return a

    def _teardown(self):
        if self.scenario is not None:
            self.scenario.terminate()
            self.scenario.remove_all_actors()
            self.scenario = None
        if self.ego_actor is not None:
            CarlaDataProvider.remove_actor_by_id(self.ego_actor.id)
            self.ego_actor = None
        self.world.tick()

    # ---------------- simulation ----------------
    def trel(self):
        return self.t

    def step_bodies(self):
        """One tick of every non-ego body (called by the StepScenarioLaws behaviour before world.tick)."""
        self.move_bodies()
        batch = [carla.command.ApplyTransform(self.actors[b.name].id, self.frame.to_world(*b.position, b.heading, z_off(b)))
                 for b in self.body_list if b.name in self.actors and b.kind != "static"]
        self.client.apply_batch(batch)

    def move_bodies(self):
        """Advance every non-ego body one tick along its law / scripted path (scenario frame)."""
        cmds = []
        for b in self.body_list:
            if b.kind == "law":
                out = b.law(self.t, b, self)
                acc, ty = out if isinstance(out, tuple) else (out, b.target_y)
                b.target_y = ty
                if b.speed + acc * DT < 0:
                    acc = -b.speed / DT
                cmds.append((b, acc, track_y(*b.position, b.heading, b.speed, b.target_y, half_l=b.LENGTH / 2)))
        for b, acc, steer in cmds:
            b.bicycle_step(acc, steer)
        for b in self.body_list:
            if b.kind == "scripted":
                b.place(self.t + DT)

    def _read_ego(self):
        a = self.ego_actor
        tf, vel, acc = a.get_transform(), a.get_velocity(), a.get_acceleration()
        x, y, psi = self.frame.from_world(tf.location, tf.rotation.yaw)
        vx, vy = self.frame.vec_from_world((vel.x, vel.y), x)
        ax, ay = self.frame.vec_from_world((acc.x, acc.y), x)
        e = self.ego
        if self.prev_psi is not None:
            self.ego_yaw_rate = math.atan2(math.sin(psi - self.prev_psi), math.cos(psi - self.prev_psi)) / DT
        self.prev_psi = psi
        e.position = np.array([x, y])
        e.heading = psi
        e.speed = max(0.0, vx * math.cos(psi) + vy * math.sin(psi))
        e._vel = np.array([vx, vy])
        self.ego_acc = 0.7 * self.ego_acc + 0.3 * (ax * math.cos(psi) + ay * math.sin(psi))
        self.ego_world = (tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw)
        if self.envelope:
            self.meas = self.monitor.update(vel.x, vel.y, math.radians(tf.rotation.yaw))

    def _snapshot(self):
        """Scenario actors as notebook state dicts, as the expert sees them: background / benign objects (bg_*)
        excluded, and only actors the spec delivers (sight line)."""
        return {n: dict(x=float(b.position[0]), y=float(b.position[1]), v=float(b.speed), psi=float(b.heading),
                        length=b.LENGTH, width=b.WIDTH) for n, b in self.bodies.items()
                if not n.startswith("bg_") and self.spec.deliver(self, n)}

    def curvature(self, x):
        """Road curvature [1/m] at frame coordinate x (0 on a straight site)."""
        return float(np.interp(x, self.frame.S, self._kappa)) if self.frame is not None else 0.0

    def _ego_control(self, a_cmd, r_cmd):
        v = self.ego.speed
        wb = self.wheelbase
        # r_cmd is relative to the lane frame; the steering geometry needs the world yaw rate: + the road's turn rate
        r_w = r_cmd + v * math.cos(self.ego.heading) * self.curvature(self.ego.position[0])
        sb = max(-0.99, min(0.99, r_w * (wb / 2) / max(v, 1.0)))
        wheel = math.atan(2.0 * math.tan(math.asin(sb)))
        if v > 1.0:
            wheel += YAW_KP * (r_cmd - self.ego_yaw_rate) * wb / v
        lim = math.radians(self.max_steer) * (np.interp(v * 3.6, *self.steer_curve) if self.steer_curve else 1.0)
        thr, brk = self.long(a_cmd, v)
        steer = float(np.clip(wheel / lim, -1.0, 1.0))
        self.ctrl = (float(thr), float(brk), steer)
        return carla.VehicleControl(throttle=float(thr), brake=float(brk), steer=steer)

    def _margins(self, a_cmd, r_cmd):
        """Notebook margins at this tick: clearance (with the closing-speed reserve), road edge, and dyn = the
        commanded friction-circle demand hypot(a, v * r) / mu_g (requests beyond the accel bounds count in full)."""
        e = self.ego
        ep = obb(*e.position, e.heading, EGO_L, EGO_W)
        m_clear, near, contact = 1e3, "", False
        for b in self.body_list:
            dx, dy = b.position[0] - e.position[0], b.position[1] - e.position[1]
            rv = b.velocity - e.velocity
            reach = 0.5 * (math.hypot(EGO_L, EGO_W) + math.hypot(b.LENGTH, b.WIDTH)) + math.hypot(*rv) * T_H
            lower = math.hypot(dx, dy) - reach - C0          # lower bound of this body's margin
            if lower > 2.0:
                continue                                      # > 2 m: cannot bind; skip the exact polygon distances
            op = obb(*b.position, b.heading, b.LENGTH, b.WIDTH)
            dist = signed_distance(ep, op)
            d_pred = min(signed_distance(ep, [(px + rv[0] * h, py + rv[1] * h) for px, py in op])
                         for h in (0.5 * T_H, T_H))
            m = min(dist, d_pred) - C0
            if m < m_clear:
                m_clear, near = m, b.name
            contact = contact or dist < 0
        m_clear = min(m_clear, 2.0)                           # beyond 2 m the exact value does not matter
        y_lo, y_hi = self.scene["bounds"](e.position[0])
        ys = [p[1] for p in ep]
        m_road = min(y_hi - max(ys), min(ys) - y_lo)
        if self.envelope:                                     # demand on the usable combined budget
            dyn = math.hypot(a_cmd, max(e.speed, V_MIN_YAW) * r_cmd) / self.envelope.M
        else:
            dyn = math.hypot(a_cmd, e.speed * r_cmd) / MU_G
        return m_clear, m_road, dyn, near, contact

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._teardown()
        options = options or {}
        self.params = dict(options.get("params") or self.spec.sample(self.rng))
        self.scene = self.spec.build(self.params, self.W)
        self.body_list = self.scene["bodies"]
        self.init_scene = [dict(name=b.name, kind=b.kind, bp=b.bp, color=b.color, x=float(b.position[0]),
                                y=float(b.position[1]), psi=float(b.heading), v=float(b.speed), length=b.LENGTH,
                                width=b.WIDTH) for b in self.body_list]
        self.bodies = self.objs = {b.name: b for b in self.body_list}
        self.actors, self.mem = {}, {}
        self.t = -self.spec.prefix_T
        for b in self.body_list:
            if b.kind == "scripted":
                b.place(self.t)
        e0 = self.scene["ego"]
        self.ego = Body("ego", e0["x"], e0["y"], v=e0["v"], length=EGO_L, width=EGO_W, kind="ego")
        self.ego_yaw_rate, self.ego_acc, self.prev_psi = 0.0, 0.0, None
        self.meas = (0.0, 0.0)
        self._apply_road()
        if self.envelope:
            self.monitor.reset()
            self.limiter.reset()
            self.n_shaped = 0
        self._spawn_episode(e0)
        self.prev_psi = self.ego.heading
        self.snaps = [(self.t, self._snapshot())]
        self.speed_hist = {n: [self.bodies[n].speed] for n in self.bodies}
        self.speed_hist["ego"] = [self.ego.speed]
        self.prev_action = np.zeros(2)
        self.ctrl = (0.0, 0.0, 0.0)
        self.log = dict(tick=[], step=[])
        self.min_m = dict(clear=1e3, road=1e3, dyn=1e3)
        self.episode += 1
        self.ep_return = 0.0
        self.note = None                                    # a driver's per-step note (e.g. its chosen family), logged
        obs = self.spec.features(self)
        self.last_obs = obs
        return obs, {"params": self.params}

    def _spawn_episode(self, e0):
        """CARLA side of reset: weather, the ego (physics, controllers), the ScenarioRunner scenario, warm-up."""
        e0 = dict(e0, x=e0["x"] + self.start[0], v=e0["v"] + self.start[1])
        weather = self.spec.weather_for(self.params)
        if weather != self._weather:
            self.world.set_weather(carla.WeatherParameters(**weather))
            self._weather = dict(weather)
        GameTime.restart()
        tf0 = self.frame.to_world(e0["x"], e0["y"])
        self.ego_actor = self.spawn(EGO_BP, tf0, "hero", color=self.spec.ego_color(self.params))
        pc = self.ego_actor.get_physics_control()
        if hasattr(self.spec, "ego_physics"):              # e.g. lower tyre friction (generalize.py)
            pc = self.spec.ego_physics(pc, self.params)
            self.ego_actor.apply_physics_control(pc)
        self.wheelbase = abs(pc.wheels[0].position.x - pc.wheels[-1].position.x) / 100.0 or 2.8
        self.max_steer = pc.wheels[0].max_steer_angle
        curve = list(getattr(pc, "steering_curve", []) or [])
        self.steer_curve = ([c.x for c in curve], [c.y for c in curve]) if curve else None
        n_gears = max(1, len(pc.forward_gears))
        self.long = LongCtrl(self.amap, e0["v"])
        if self.envelope:                                   # never ask the pedal for more than the admissible braking
            self.long.a_floor = lambda v: -float(np.interp(v, self.envelope.v_tab, self.envelope.b_tab))
            if getattr(self.spec, "governor", False):              # feedback included, inside the friction circle
                self.long.a_floor, self.long.a_ceil = (lambda v: -self._long_budget(v)[1]), (lambda v: self._long_budget(v)[0])
        self.scenario = TeacherScenario(self, self.world, self.ego_actor)
        # warm-up: hold everything at its start pose; the ego at speed in a matching gear (bridge.py)
        for k in range(WARMUP_TICKS):
            self.ego_actor.set_transform(carla.Transform(tf0.location + carla.Location(z=0.02), tf0.rotation))
            h = math.radians(tf0.rotation.yaw)
            self.ego_actor.set_target_velocity(carla.Vector3D(e0["v"] * math.cos(h), e0["v"] * math.sin(h), 0.0))
            self.ego_actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            if k == 0:
                self.ego_actor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=gear_for(e0["v"], n_gears)))
            else:
                p = self.amap.pedal(0.0, e0["v"])
                self.ego_actor.apply_control(carla.VehicleControl(throttle=max(p, 0.0), brake=max(-p, 0.0)))
            self.client.apply_batch([carla.command.ApplyTransform(self.actors[b.name].id,
                                                                  self.frame.to_world(*b.position, b.heading, z_off(b)))
                                     for b in self.body_list if b.name in self.actors])
            self.world.tick()
        CarlaDataProvider.on_carla_tick()
        self._read_ego()
        if self.envelope:                                   # the warm-up's set velocities are not vehicle motion
            self.monitor.reset()
        self.gov_k, self.n_gov, self.app, self.a_curve = 1.0, 0, (0.0, 0.0), 0.0

    def _long_budget(self, v):
        """(propulsion, braking) the low-level speed controller may ask for, its feedback included (spec.governor): what
        the combined budget min(M, GOV_G) leaves after the lateral acceleration - the larger of the applied request (its
        bend part included) and the measured one; at least 1 m/s^2 of braking. Without it, a brake request the car is
        slow to follow gets up to ~3 m/s^2 of speed-tracking feedback on top (measured: 0.65 g asked, 0.9 g delivered
        while the bend took 0.45 g sideways). The pedal map can also deliver more than it predicts: a measured
        acceleration beyond the allowance is taken off it (trim)."""
        A, B, _, M = self.envelope.caps(v)
        al, at = self.monitor.last
        lat = max(abs(self.app[1] + self.a_curve), abs(at))
        left = math.sqrt(max(min(M, GOV_G * G0) ** 2 - lat ** 2, 0.0))
        up, down = min(A, left), min(B, left)
        return max(up - max(al - up, 0.0), 0.0), max(down - max(-al - down, 0.0), 1.0)

    def _govern(self, ax, ay):
        """Envelope governor (spec.governor): while the measured planar acceleration (the monitor's filtered signal,
        bend included) is above GOV_G, the policy's request (a_x, its own a_y; not the bend's v^2 kappa) is scaled down
        in proportion, recovering by GOV_RECOVER per step below it; measured propulsion above GOV_AX_G cuts the
        throttle request."""
        al, at = self.monitor.last
        g = math.hypot(al, at) / G0
        k = max(GOV_MIN, self.gov_k * GOV_G / g) if g > GOV_G else min(1.0, self.gov_k + GOV_RECOVER)
        self.gov_k = k
        bx, by = k * ax, k * ay
        if al / G0 > GOV_AX_G and bx > 0.0:
            bx = 0.0
        self.n_gov += (bx, by) != (ax, ay)
        return bx, by

    def _action_to_cmd(self, u):
        u = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
        if self.envelope:
            v = self.ego.speed
            ax, ay = self.envelope.map(u, v)
            # spec.comb_frac: the policy's request is kept inside that fraction of the combined budget (the experts'
            # own cap), so a saturated learned action cannot spend the dyn margin the demonstrations kept
            frac = getattr(self.spec, "comb_frac", None)
            if frac and math.hypot(ax, ay) > frac * self.envelope.M:
                k = frac * self.envelope.M / math.hypot(ax, ay)
                ax, ay = k * ax, k * ay
            # in a bend the road itself needs v^2 kappa of lateral acceleration: the total stays inside the envelope
            a_curve = v * v * self.curvature(self.ego.position[0])
            if a_curve:
                lim = min(self.envelope.AY, math.sqrt(max(self.envelope.M ** 2 - ax * ax, 0.0)))
                ay = float(np.clip(ay + a_curve, -lim, lim)) - a_curve
            if getattr(self.spec, "governor", False):
                ax, ay = self._govern(ax, ay)
            self.req = (ax, ay)
            ax, ay = self.limiter(*self.req)
            self.app, self.a_curve = (ax, ay), a_curve
            self.n_shaped += abs(ax - self.req[0]) + abs(ay - self.req[1]) > 0.05
            return float(ax), float(ay / max(v, V_MIN_YAW))
        a = A_MAX * u[0] if u[0] >= 0 else B_MAX * u[0]
        r = u[1] * min(0.6, MU_G / max(self.ego.speed, 5.0))
        return float(a), float(r)

    def _advance(self, a_cmd, r_cmd):
        """One 0.05 s tick: ego control, scenario actors (behaviour tree), world; -> collision sensor fired."""
        self.ego_actor.apply_control(self._ego_control(a_cmd, r_cmd))
        self.scenario.scenario_tree.tick_once()
        self.world.tick()
        snap = self.world.get_snapshot()
        self.frame_id = snap.frame                    # CARLA's frame number (the per-tick evidence's key)
        if snap.frame % 4 == 0:                      # CarlaDataProvider caches (not used by our atomics/criteria)
            CarlaDataProvider.on_carla_tick()
        GameTime.on_carla_tick(snap.timestamp)
        self.t += DT
        self._read_ego()
        return self.scenario.collision.test_status == "FAILURE"

    def _hit_actor(self):
        """The body the collision sensor fired on (it can fire a tick before our boxes overlap); None if not ours."""
        cid = getattr(self.scenario.collision, "_collision_id", None)
        return next((n for n, a in self.actors.items() if a.id == cid), None)

    def step(self, action):
        a_cmd, r_cmd = self._action_to_cmd(action)
        fail, why = False, ""
        ms = []
        for _ in range(POLICY_TICKS):
            crashed = self._advance(a_cmd, r_cmd)
            self.snaps.append((self.t, self._snapshot()))
            for n, b in self.bodies.items():
                self.speed_hist[n].append(b.speed)
            self.speed_hist["ego"].append(self.ego.speed)
            m_clear, m_road, dyn, near, contact = self._margins(a_cmd, r_cmd)
            ms.append((m_clear, m_road, dyn))
            self._log_tick(a_cmd, r_cmd, m_clear, m_road, dyn, near)
            if self.t >= -1e-6:                                    # the notebook scores from t0 on
                self.min_m["clear"] = min(self.min_m["clear"], m_clear)
                self.min_m["road"] = min(self.min_m["road"], m_road)
                self.min_m["dyn"] = min(self.min_m["dyn"], 1.0 - dyn)
            if contact or crashed:
                fail, why = True, f"contact:{near}" if contact else f"collision_sensor:{self._hit_actor() or near}"
                break
            if m_road < -1.0:
                fail, why = True, "left_corridor"
                break
        m_clear = min(m[0] for m in ms)
        m_road = min(m[1] for m in ms)
        dyn = max(m[2] for m in ms)
        u = np.array([a_cmd / (A_MAX if a_cmd >= 0 else B_MAX), r_cmd / 0.3])
        rew = (0.1 - 1.0 * max(0.0, 1.0 - m_clear / 1.5) - 1.0 * max(0.0, (0.3 - m_road) / 0.3)
               - 0.5 * max(0.0, (dyn - 0.85) / 0.15) - 0.05 * float(np.sum((u - self.prev_action) ** 2))
               - 0.1 * abs(self.ego.speed - self.scene["v_ref"]) / self.scene["v_ref"])
        self.prev_action = u
        terminated, truncated, info = fail, False, {}
        if fail:
            rew -= 20.0
        elif self.t >= self.spec.T - 1e-6 or self.spec.done(self) or self._road_ends():
            truncated = True
            term = self._terminal_margin()
            rew += 10.0 if term >= 0.0 else -5.0
            info["term"] = term
            if "road_end_t" in self.mem:
                info["road_end_t"] = self.mem["road_end_t"]
        obs = self.spec.features(self)
        self.ep_return += rew
        self.log["step"].append(dict(t=self.t, obs=self.last_obs, a_cmd=a_cmd, r_cmd=r_cmd, u=np.asarray(action, float),
                                     reward=rew, req=list(getattr(self, "req", (a_cmd, 0.0))),
                                     note=getattr(self, "note", None)))
        self.last_obs = obs
        if terminated or truncated:
            r_nb = min(self.min_m["clear"], self.min_m["road"], self.min_m["dyn"], info.get("term", -1.0))
            goal_ok, goal = self.spec.goal(self) if not fail else (False, {})
            env_ok = True
            if self.envelope:
                info["envelope"] = dict(self.monitor.summary(), shaped_frac=self.n_shaped / max(len(self.log["step"]) + 1, 1),
                                        governed_frac=self.n_gov / max(len(self.log["step"]) + 1, 1))
                env_ok = info["envelope"]["viol_ticks"] == 0
                if not fail and not env_ok:
                    why = "envelope"
            info.update(r_nb=r_nb, success=bool(not fail and r_nb >= 0.05 and goal_ok and env_ok), goal_ok=bool(goal_ok),
                        goal=goal, fail=why or ("" if goal_ok else "goal"), params=self.params,
                        min_margins=dict(self.min_m), ep_return=self.ep_return)
            self._save_episode(info)
        return obs, float(rew), terminated, truncated, info

    def _road_ends(self):
        """spec.end_at_road_end: the real road ends (the frame continues straight past a turn, into the map edge) 10 m
        ahead of the ego: the horizon comes early, the terminal contract is scored there."""
        x_end = getattr(self.frame, "straight_from", None)
        if not getattr(self.spec, "end_at_road_end", False) or x_end is None or self.ego.position[0] < x_end - 10.0:
            return False
        self.mem["road_end_t"] = self.t
        return True

    def _terminal_margin(self):
        """Notebook 'track' contract over the last T_term s of the episode: lane-centred (|ey| <= 0.6 m) and straight (|psi| <= 0.06)."""
        rows = [r for r in self.log["tick"] if r["t"] >= self.t - self.spec.T_term - 1e-6]     # last T_term s
        ey = max(abs(r["ego"][1] - round(r["ego"][1] / self.W) * self.W) for r in rows)
        psi = max(abs(r["ego"][2]) for r in rows)
        return float(min(0.6 - ey, (0.06 - psi) * 10))

    def _log_tick(self, a_cmd, r_cmd, m_clear, m_road, dyn, near):
        e = self.ego
        self.log["tick"].append(dict(
            frame=getattr(self, "frame_id", None), t=round(self.t, 4), ego=[e.position[0], e.position[1], e.heading, e.speed, self.ego_acc, self.ego_yaw_rate],
            ego_world=list(self.ego_world), ctrl=list(self.ctrl), cmd=[a_cmd, r_cmd],
            bodies={n: [b.position[0], b.position[1], b.heading, b.speed, b.velocity[0], b.velocity[1]]
                    for n, b in self.bodies.items()},
            m=[m_clear, m_road, dyn], near=near, meas=list(self.meas)))

    def _save_episode(self, info):
        if not self.log_dir:
            return
        T, S = self.log["tick"], self.log["step"]
        names = list(self.bodies)
        out = dict(
            t=np.array([r["t"] for r in T], np.float32),
            ego=np.array([r["ego"] for r in T], np.float32),                 # x, y, psi, v, a_long, yaw_rate (frame)
            ego_world=np.array([r["ego_world"] for r in T], np.float32),     # x, y, z, yaw_deg (CARLA world)
            ctrl=np.array([r["ctrl"] for r in T], np.float32),               # throttle, brake, steer
            cmd=np.array([r["cmd"] for r in T], np.float32),                 # a_cmd, yaw-rate cmd (held per step)
            bodies=np.array([[r["bodies"][n] for n in names] for r in T], np.float32),  # [tick, body, x y psi v vx vy]
            margins=np.array([r["m"] for r in T], np.float32),               # m_clear, m_road, dyn
            step_t=np.array([s["t"] for s in S], np.float32),
            obs=np.array([s["obs"] for s in S], np.float32),
            action=np.array([s["u"] for s in S], np.float32),
            cmd_step=np.array([[s["a_cmd"], s["r_cmd"]] for s in S], np.float32),
            reward=np.array([s["reward"] for s in S], np.float32))
        if any(s.get("note") is not None for s in S):
            out["notes"] = np.array(json.dumps([s.get("note") for s in S], default=float))
        if self.envelope:
            out["req_step"] = np.array([s["req"] for s in S], np.float32)           # envelope request a_x, a_y (pre rate limit)
            out["meas"] = np.array([r["meas"] for r in T], np.float32)              # measured body accel a_long, a_lat (filtered)
        meta = dict(sid=self.spec.sid, town=self.spec.town, site=self.spec.site, lane_width=self.W, seed=self.seed_,
                    episode=self.episode, params=self.params, body_names=names, obs_names=self.spec.obs_names,
                    body_sizes={n: [b.LENGTH, b.WIDTH, b.bp] for n, b in self.bodies.items()},
                    init_scene=self.init_scene, ego_init=self.scene["ego"], ego_bp=EGO_BP,
                    ego_color=self.spec.ego_color(self.params), props=self.scene["props"], weather=self._weather,
                    outcome={k: v for k, v in info.items() if k != "params"}, dt=DT, policy_dt=DT * POLICY_TICKS,
                    policy=getattr(self, "policy_tag", "train"), modes=list(getattr(self.spec, "modes", ())),
                    envelope=self.envelope.card() if self.envelope else None)
        out["meta"] = np.array(json.dumps(meta, default=float))
        name = f"{self.spec.sid}_s{self.seed_}_{self.episode:06d}.npz"
        np.savez_compressed(os.path.join(self.log_dir, name), **out)
        # registry: one line per episode = the variation's config (rebuild with spec.build(params) / replay.py)
        with open(os.path.join(self.log_dir, "variations.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(sid=self.spec.sid, npz=name, policy=meta["policy"], params=self.params,
                                     success=info["success"], r_nb=info["r_nb"], fail=info["fail"], goal=info["goal"]),
                                default=float) + "\n")

    def load_variations(self, path):
        """Variation configs from a file: variations.jsonl (one {"params": ...} per line, as the env writes it),
        search.json ({"variations": [...]}) or a JSON list of params dicts."""
        txt = open(path).read().strip()
        if path.endswith(".jsonl"):
            return [json.loads(l)["params"] for l in txt.splitlines() if l.strip()]
        d = json.loads(txt)
        return d["variations"] if isinstance(d, dict) else [x.get("params", x) for x in d]

    def close(self):
        try:
            self._teardown()
            s = self.world.get_settings()
            s.synchronous_mode = False
            self.world.apply_settings(s)
        except Exception:
            pass
