"""Kinematic stand-in for TeacherEnv (no CARLA server): the same scenes, laws, observations, margins, scoring and
envelope, with the ego as a rate-limited point-mass bicycle (first-order lag on the acceleration and yaw rate, the
measured accel-map limits). The report's "cheap calibrated model" for screening variation ranges and testing the
pipeline offline; CARLA stays the reference for every label used for training and evaluation.

    from surrogate import SurrogateEnv
    env = SurrogateEnv("RC03-S1-opt", seed=0)
"""
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _stub_missing():
    """Offline: CARLA's PythonAPI from tests/fake_carla; ScenarioRunner / py_trees as empty stubs (never called)."""
    try:
        import carla  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests", "fake_carla"))
    try:
        import py_trees  # noqa: F401
        import srunner  # noqa: F401
    except ImportError:
        class _Any:
            def __init__(self, *a, **k):
                pass
        names = {"py_trees": {}, "srunner": {}, "srunner.scenarioconfigs": {}, "srunner.scenariomanager": {},
                 "srunner.scenariomanager.scenarioatomics": {}, "srunner.scenarios": {},
                 "srunner.scenarioconfigs.scenario_configuration": {"ScenarioConfiguration": _Any},
                 "srunner.scenariomanager.carla_data_provider": {"CarlaDataProvider": _Any},
                 "srunner.scenariomanager.scenarioatomics.atomic_behaviors": {"AtomicBehavior": _Any},
                 "srunner.scenariomanager.scenarioatomics.atomic_criteria": {"CollisionTest": _Any},
                 "srunner.scenariomanager.timer": {"GameTime": _Any},
                 "srunner.scenarios.basic_scenario": {"BasicScenario": _Any}}
        for n, attrs in names.items():
            m = sys.modules.setdefault(n, types.ModuleType(n))
            for k, v in attrs.items():
                setattr(m, k, v)


_stub_missing()
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import gymnasium as gym  # noqa: E402

from bridge import EGO_BP, AccelMap  # noqa: E402
from env import TeacherEnv  # noqa: E402
from envelope import Envelope  # noqa: E402
from laws import DT, EGO_L, EGO_W, Body  # noqa: E402
from scenarios import SPECS  # noqa: E402

TAU_A, TAU_R = 0.15, 0.12          # ego longitudinal / yaw-rate response lags [s] (brake buildup, steering)
K_YAW = 0.25                       # kinematic limit |yaw rate| <= v K_YAW [1/m] (4 m turning radius): a stopped car
                                   # cannot turn (without it a fork straightened up at standstill, CARLA cannot)
MU_ACT = 8.5                       # tyre limit on the planar acceleration before qualification (then the measured one)
# TeacherEnv's start, measured (RC07-S5 pilot, 55 CARLA runs, 2026-11-17): the warm-up's last world tick moves the ego
# one tick on, it leaves 0.12 m/s slow, and the cruise controller then drifts up ~0.4 m/s^2 over the prefix (at t0:
# +1.25 m, +0.28 m/s). The surrogate starts the same way.
START_DV, PREFIX_DRIFT = -0.12, 0.4
# ... and that drift is a transient after the warm-up: in longer pre-rolls (3-4 s, vsearch) CARLA gains ~0.26 m/s over
# the first second, then holds (RC07 / RC10 / RC14 nominal, keep lane, 2026-11-18)
T_DRIFT = 0.9                      # [s] from the start of the pre-roll
# CARLA brakes harder than asked: the accel map's brake side is ~0.4 m/s^2 weak and LongCtrl's anti-windup clip drops
# the PI correction, so the overshoot accumulates (RC07-S5 baseline, 150 runs: brake-family max |dx| 5.4 -> 0.8 m).
# A yaw-command delay was fitted too and is best at 0 ticks (the lateral residual, ~0.3 m, is left to the verdict).
BRAKE_BIAS = 0.4                   # extra deceleration CARLA adds to a braking request [m/s^2] (full from -2 m/s^2)
YAW_DELAY = 0                      # ticks between a yaw-rate command and its effect
# CARLA's low-speed stop: under hard braking below ~6 m/s the car stops within a tick or two (RC07-S5 validation brake
# runs: median 24 m/s^2 from 5-6 m/s, 11 from 6-7; RC10 smoke: 5.8 -> 1.3 m/s in 0.2 s). Scenes opt in (spec.snap_stop),
# so a batch searched before it keeps one surrogate version.
V_SNAP, SNAP_DEC, A_SNAP = 6.0, 22.0, -4.8   # [m/s], [m/s^2], braking request that counts as hard (~0.6 pedal)


class SurrogateEnv(TeacherEnv):
    dyn = (1.0, 1.0, 1.0, 1.0)     # scales of the brake / throttle limits, TAU_A, TAU_R, grip (planner's robust worlds)
    start = (0.0, 0.0)             # extra ego start offset (x [m], v [m/s]) on top of CARLA's (robust verdicts)
    bias = None                    # brake bias / yaw delay of this world (None: the fitted BRAKE_BIAS / YAW_DELAY)
    delay = None

    def __init__(self, sid, seed=0, log_dir=None, W=3.5, **_):
        self.spec = SPECS[sid]
        self.rng = np.random.default_rng(seed)
        self.seed_, self.log_dir, self.render_ = seed, log_dir, False
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.frame, self.W, self._weather = None, getattr(self.spec, "lane_w", W), None   # Town01 / 02: 4 m
        n = len(self.spec.obs_names)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.scenario, self.ego_actor, self.episode = None, None, 0
        self._init_envelope()

    def curvature(self, x):
        return 0.0

    def _teardown(self):
        pass

    def close(self):
        pass

    def _init_envelope(self, mu=1.0):
        super()._init_envelope(mu)
        self.cap = Envelope(EGO_BP, mu)          # measured full-brake / full-throttle accel vs speed (gaps filled)

    def _spawn_episode(self, e0):
        dx, dv = self.start
        self.x = dict(x=e0["x"] + e0["v"] * DT + dx, y=e0["y"], psi=0.0, v=e0["v"] + START_DV + dv, a=0.0, r=0.0)
        self.ctrl = (0.0, 0.0, 0.0)
        self.r_buf = [0.0] * (YAW_DELAY if self.delay is None else self.delay)
        self.gov_k, self.n_gov, self.app, self.a_curve = 1.0, 0, (0.0, 0.0), 0.0     # as TeacherEnv's spawn
        self._read_ego()
        if self.envelope:
            self.monitor.reset()

    def _advance(self, a_cmd, r_cmd):
        self.move_bodies()
        s = self.x
        v = s["v"]
        c = self.cap
        kb, ta, tr, km = self.dyn
        if a_cmd < 0:
            a_cmd -= (BRAKE_BIAS if self.bias is None else self.bias) * min(-a_cmd / 2.0, 1.0)
        if self.r_buf:
            self.r_buf.append(r_cmd)
            r_cmd = self.r_buf.pop(0)
        a_t = float(np.clip(a_cmd, -kb * np.interp(v, c.v_tab, c.b_tab), kb * np.interp(v, c.v_tab, c.a_tab)))
        if v < 0.3 and a_t <= 0:
            a_t = 0.0
        if self.t < -1e-6 and self.t < -self.spec.prefix_T + T_DRIFT:
            a_t += PREFIX_DRIFT
        s["a"] += (a_t - s["a"]) * DT / (ta * TAU_A)
        s["r"] += (r_cmd - s["r"]) * DT / (tr * TAU_R)
        s["r"] = float(np.clip(s["r"], -K_YAW * v, K_YAW * v))
        a_lat = v * s["r"]
        mu = km * (c.comb_measured if c.qualified else MU_ACT)
        k = mu / max(math.hypot(s["a"], a_lat), mu)                 # tyre limit: scale the planar demand
        s["a"], s["r"] = s["a"] * k, s["r"] * k
        if getattr(self.spec, "snap_stop", False) and a_cmd <= A_SNAP and v < V_SNAP:
            s["a"] = -SNAP_DEC
        s["v"] = max(0.0, v + s["a"] * DT)
        s["psi"] += s["r"] * DT
        s["x"] += s["v"] * math.cos(s["psi"]) * DT
        s["y"] += s["v"] * math.sin(s["psi"]) * DT
        self.ctrl = (max(s["a"], 0.0) / 3.0, max(-s["a"], 0.0) / 8.0, s["r"])
        self.t += DT
        self.frame_id = int(round((self.t + self.spec.prefix_T) / DT))   # ticks since the episode start
        self._read_ego()
        return False

    def _read_ego(self):
        s, e = self.x, self.ego
        e.position = np.array([s["x"], s["y"]])
        e.heading, e.speed = s["psi"], s["v"]
        e._vel = s["v"] * np.array([math.cos(s["psi"]), math.sin(s["psi"])])
        self.ego_yaw_rate, self.ego_acc = s["r"], s["a"]
        self.ego_world = (s["x"], s["y"], 0.0, math.degrees(s["psi"]))
        if self.envelope:
            self.meas = self.monitor.update(e._vel[0], e._vel[1], s["psi"])


__all__ = ["SurrogateEnv", "Body", "EGO_L", "EGO_W"]
