"""RC11 scenarios (hidden mode, unrecoverable for a shared policy) split by mode, as RC15 is split by the adjacent
driver's response: each class fixes the mode, so its teacher assumes it and commits to that mode's right pick (the
notebook's "right pick survives" clip); the variations decide how early and how hard.

    RC11-S1-{L,R}  Town04 road 49 at night, lanes (4, 5, 6): crashed car 1 (hazard lights) stands in the ego lane;
                   unlit car 2 lies in the left (L) / right (R) lane beyond it, visible only within the low-beam range
                   -> L: escape right, R: escape left. Also covers RC11-S2 (fog) and RC11-S3 (tunnel glare): the same
                   scene with another reason for the late sight of car 2 - identical in state space once the mode is known.
    RC11-S4-{L,R}  Town06 road 35, lanes (-5, -6, -7): the car ahead (10 m) blows a tyre and spins to rest across the
                   middle and the left (L) / right (R) lane -> L: escape right, R: escape left.
    RC11-S5-{W,D}  Town07 road 11 (rural 1+1): a drowsy oncoming driver drifts over the centre line; W: wakes and steers
                   back -> brake hard in lane; D: keeps drifting to the ego lane's edge -> swerve into the vacated
                   opposing lane.
    RC11-S6-{F,C}  Town05 road 51 (one-way, 2 lanes + parking lane): a child runs out between parked cars; F: freezes in
                   the ego lane -> swerve left; C: sprints across -> brake in lane.
"""
import math

import numpy as np

from laws import HALF_L, Body, child_path, const_path, drift_path, spin_path
from rc03 import in_lane_goal, seen_rel
from scenarios import CAR, Spec, car, platoon, rel

NIGHT = dict(cloudiness=10.0, precipitation=0.0, sun_altitude_angle=-40.0, sun_azimuth_angle=150.0, fog_density=0.0)


class ModeSpec(Spec):
    """A hidden-mode scenario with the mode fixed. options: name -> lateral target y in scenario lanes (None = stay,
    'stop' = brake in lane); answer[mode] = the right pick (the expert's SharedBranch default)."""
    T, T_term = 6.0, 2.0
    MODE_NAMES = {}
    options, answer = {}, {}

    def __init__(self, mode):
        self.mode = mode
        self.sid = f"{self.base}-{mode}"
        self.pick = self.answer[mode]
        self.description = f"{self.story} [{self.MODE_NAMES[mode]} -> {self.pick}]"

    def card(self):
        return dict(super().card(), mode=self.mode, pick=self.pick)

    def options_y(self, W):
        return {k: (None if v is None else v if isinstance(v, str) else v * W) for k, v in self.options.items()}

    def goal(self, env):
        o = self.options_y(env.W)[self.pick]
        if o == "stop":
            ok = env.ego.speed <= 1.0 and abs(env.ego.position[1] - env.scene["home_y"]) <= 1.0
            return ok, dict(v_end=round(env.ego.speed, 2))
        return in_lane_goal(env, env.scene["home_y"] if o is None else o)


class RC11S1(ModeSpec):
    base, town = "RC11-S1", "Town04"
    site = dict(start_xy=(-9.9, -202.8), ref_lane=4, ref_idx=0, s0=30.0)
    story = "Town04 motorway at night: crashed car ahead, unlit second car beyond it"
    MODE_NAMES = {"L": "car 2 in the left lane", "R": "car 2 in the right lane"}
    options, answer = {"left": 0.0, "right": 2.0, "stop": "stop"}, {"L": "right", "R": "left"}
    weather = NIGHT
    nominal = dict(v=25.0, D=38.0, gap2=8.0, d_vis=25.0, y1=0.0, y2=0.0)
    ranges = dict(v=(24.0, 26.0), D=(35.0, 41.0), gap2=(6.0, 10.0), d_vis=(22.0, 28.0), y1=(-0.2, 0.2), y2=(-0.3, 0.3))
    obs_names = Spec.EGO_NAMES + ["car1_dx", "car1_dy", "car2_seen", "car2_dx", "car2_dy"]

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        L1 = CAR["vehicle.lincoln.mkz_2020"][0]
        x1 = HALF_L + p["D"] + L1 / 2
        c1 = Body("car1", x1, W + p["y1"], psi=0.15, length=L1, width=CAR["vehicle.lincoln.mkz_2020"][1], kind="static",
                  bp="vehicle.lincoln.mkz_2020")
        y_c2 = (0.0 if self.mode == "L" else 2 * W) + p["y2"]
        c2 = Body("car2", x1 + p["gap2"], y_c2, psi=-0.3 if self.mode == "L" else 0.3, length=CAR["vehicle.audi.a2"][0],
                  width=CAR["vehicle.audi.a2"][1], kind="static", bp="vehicle.audi.a2")
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[c1, c2], home_y=W, bounds=lambda x: (-W / 2, 2.5 * W),
                    v_ref=v, props=[])

    def deliver(self, env, name):
        if name != "car2":
            return True
        b, e = env.bodies["car2"], env.ego
        return (b.position[0] - b.LENGTH / 2) - (e.position[0] + HALF_L) <= env.params["d_vis"]

    def features(self, env):
        dx, dy, _, _ = rel(env, "car1")
        s = seen_rel(env, "car2", sentinel=(60.0, 0.0, 0.0, 0.0))
        return np.array(self.ego_features(env) + [dx, dy] + s[:3], dtype=np.float32)


class RC11S4(ModeSpec):
    base, town = "RC11-S4", "Town06"
    site = dict(start_xy=(21.7, 143.0), ref_lane=-6, ref_idx=1, s0=60.0)
    story = "Town06 motorway: the car 10 m ahead blows a tyre and spins across a lane"
    MODE_NAMES = {"L": "slides into the left lane", "R": "slides into the right lane"}
    options, answer = {"left": 0.0, "right": 2.0, "stop": "stop"}, {"L": "right", "R": "left"}
    nominal = dict(v=25.0, gap=10.0, b_spin=10.0, t_div=1.4, slide=2.3)
    ranges = dict(v=(24.0, 26.0), gap=(9.0, 11.0), b_spin=(9.0, 11.0), t_div=(1.3, 1.6), slide=(2.1, 2.5))
    obs_names = Spec.EGO_NAMES + ["sc_dx", "sc_dy", "sc_dvx", "sc_vy", "sc_psi"]

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        L = CAR["vehicle.mini.cooper_s_2021"][0]
        x0 = HALF_L + p["gap"] + L / 2
        sgn = -1.0 if self.mode == "L" else 1.0
        spin = car("spin_car", "vehicle.mini.cooper_s_2021", 0, W, v, kind="scripted",
                   path=spin_path(x0, W, v, p["b_spin"], p["t_div"], 1.2, W + sgn * p["slide"], 0.25, -sgn * 1.3))
        bg = platoon("l1_", -W, v + 2.0, [-60.0, -15.0, 35.0]) + platoon("l2_", -2 * W, v + 4.0, [-40.0, 20.0])
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[spin] + bg, home_y=W, bounds=lambda x: (-W / 2, 2.5 * W),
                    v_ref=v, props=[])

    def features(self, env):
        return np.array(self.ego_features(env) + list(rel(env, "spin_car")) + [env.bodies["spin_car"].heading],
                        dtype=np.float32)


class RC11S5(ModeSpec):
    base, town = "RC11-S5", "Town07"
    site = dict(start_xy=(-3.6, -228.6), ref_lane=-1, ref_idx=1, s0=60.0)
    story = "Town07 rural two-way road: a drowsy oncoming driver drifts over the centre line"
    MODE_NAMES = {"W": "the driver wakes and steers back", "D": "the driver keeps drifting"}
    options, answer = {"keep": None, "left": 0.0, "stop": "stop"}, {"W": "stop", "D": "left"}
    nominal = dict(v=22.0, v_onc=22.0, x_onc=90.0, y0=1.0, vy=1.6, t_wake=1.7)
    ranges = dict(v=(21.0, 23.0), v_onc=(21.0, 23.0), x_onc=(85.0, 95.0), y0=(0.8, 1.2), vy=(1.4, 1.8),
                  t_wake=(1.5, 1.8))
    obs_names = Spec.EGO_NAMES + ["onc_dx", "onc_dy", "onc_dvx", "onc_vy"]

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        onc = car("oncoming", "vehicle.toyota.prius", 0, 0.0, p["v_onc"], kind="scripted",
                  path=drift_path(p["x_onc"], p["v_onc"], p["y0"] * W / 4.0, p["vy"], p["t_wake"], self.mode,
                                  y_end=1.5 * W, y_own=0.0))
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[onc], home_y=W, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[])

    def features(self, env):
        return np.array(self.ego_features(env) + list(rel(env, "oncoming")), dtype=np.float32)

    def goal(self, env):
        if self.mode == "D":                               # swerved into the vacated opposing lane and got past
            y_min = min(r["ego"][1] for r in env.log["tick"] if r["t"] >= 0.0)
            return y_min <= 0.25 * env.W, dict(min_y=round(y_min, 2))
        return super().goal(env)


class RC11S6(ModeSpec):
    base, town = "RC11-S6", "Town05"
    site = dict(start_xy=(-123.5, 136.6), ref_lane=-1, ref_idx=0, s0=65.0)
    story = "Town05 one-way street: a child runs out between parked cars"
    MODE_NAMES = {"F": "the child freezes in the ego lane", "C": "the child sprints across"}
    options, answer = {"left": 0.0, "stop": "stop", "keep": None}, {"F": "left", "C": "stop"}
    PARK = 3.0                                             # parking lane width right of the ego lane
    nominal = dict(v=14.0, d_c=17.0, v_run=3.5, y_slow=1.0)     # d_c >= the 13-15 m/s braking distance (~15-17.5 m)
    ranges = dict(v=(13.0, 15.0), d_c=(16.0, 18.5), v_run=(3.2, 3.8), y_slow=(0.8, 1.2))
    obs_names = Spec.EGO_NAMES + ["child_seen", "child_dx", "child_dy", "child_vy"]

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        x_c = HALF_L + p["d_c"] + 0.25
        y0 = 1.5 * W + self.PARK / 2
        child = Body("child", x_c, y0, psi=-math.pi / 2, length=0.5, width=0.5, kind="scripted",
                     bp="walker.pedestrian.0011",
                     path=child_path(x_c, y0, p["v_run"], self.mode, y_slow=W + p["y_slow"], y_end=-W / 2 - 1.5))
        y_p = 1.5 * W + self.PARK / 2
        bps = ("vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.mini.cooper_s_2021", "vehicle.ford.mustang")
        parked = [Body(f"bg_park{i}", x_c + dx, y_p, length=CAR[bps[i % 4]][0], width=CAR[bps[i % 4]][1], kind="static",
                       bp=bps[i % 4]) for i, dx in enumerate((-33.0, -27.0, -21.0, -15.0, -9.0, -3.3, 3.3, 9.0, 15.0, 21.0, 27.0))]
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[child] + parked, home_y=W, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[])

    def deliver(self, env, name):
        return name != "child" or env.t >= -1e-6               # steps out between the parked cars at t0

    def features(self, env):
        s = seen_rel(env, "child")
        return np.array(self.ego_features(env) + [s[0], s[1], s[2], s[4]], dtype=np.float32)

    def goal(self, env):
        if self.mode == "F":                               # passed the frozen child in the left lane
            c = env.bodies["child"]
            ok = env.ego.position[0] - HALF_L > c.position[0] + 1.0
            return ok, dict(passed=bool(ok))
        return super().goal(env)


SPECS11 = {s.sid: s for s in [cls(m) for cls in (RC11S1, RC11S4) for m in ("L", "R")]
           + [RC11S5(m) for m in ("W", "D")] + [RC11S6(m) for m in ("F", "C")]}
