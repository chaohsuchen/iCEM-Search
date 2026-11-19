"""Pilot scenarios on real CARLA sites, with narrow, feasible variations and per-scenario observation features.

Each spec builds the notebook scene (same actors, laws and timing) in the site's scenario frame with the real lane
width W, plus background traffic in the real lanes outside the scenario's corridor (realism; never observed).
Variation ranges stay close to the notebook's nominal world: speeds within the posted limit, human reaction times,
decelerations within tyre limits. Only quantities that differ between variations enter the observation.
"""
import math

import numpy as np

import carla

from laws import (HALF_L, STYLES, Body, crash_path, crossing_path, ego_ahead_in_lane, law_adjacent, law_idm,
                  law_trigger)

CAR = {  # blueprint -> (length, width)
    "vehicle.lincoln.mkz_2020": (4.9, 2.1), "vehicle.audi.a2": (3.7, 1.8), "vehicle.mini.cooper_s_2021": (4.0, 2.0),
    "vehicle.dodge.charger_2020": (5.0, 2.1), "vehicle.nissan.patrol_2021": (5.6, 2.2),
    "vehicle.mercedes.sprinter": (5.9, 2.0), "vehicle.ford.mustang": (4.8, 1.9), "vehicle.toyota.prius": (4.5, 2.0),
    "vehicle.carlamotors.european_hgv": (7.9, 2.9), "vehicle.volkswagen.t2_2021": (4.4, 1.8),
    "vehicle.ford.ambulance": (6.4, 2.35)}


def car(name, bp, x, y, v, **kw):
    L, Wd = CAR[bp]
    return Body(name, x, y, v=v, length=L, width=Wd, bp=bp, **kw)


def platoon(prefix, lane_y, v, xs, bps=("vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.ford.mustang")):
    """Background traffic in a lane outside the corridor (IDM, never observed)."""
    return [car(f"bg_{prefix}{i}", bps[i % len(bps)], x, lane_y, v, law=law_idm(v, T=1.4, react=0.6))
            for i, x in enumerate(xs)]


class Spec:
    sid = ""
    town = ""
    site = {}                 # start_xy, ref_lane (CARLA lane id of the frame's reference lane), ref_idx, s0
    T, prefix_T, T_term, terminal = 8.0, 1.0, 2.0, "track"
    ranges, nominal = {}, {}
    weather = dict(cloudiness=10.0, precipitation=0.0, sun_altitude_angle=55.0, sun_azimuth_angle=150.0, fog_density=0.0)
    obs_names = []
    description = ""

    def sample(self, rng):
        p = {}
        for k, r in self.ranges.items():
            p[k] = r[int(rng.integers(len(r)))] if isinstance(r, list) else float(rng.uniform(*r))
        return p

    def build(self, p, W):
        raise NotImplementedError

    def features(self, env):
        raise NotImplementedError

    # ---- hooks (defaults: fixed weather, default colours, no extra goal, every actor delivered) ----
    def weather_for(self, p):
        return self.weather

    def ego_color(self, p):
        return None

    def goal(self, env):
        """Scenario-specific completion check besides r >= 0.05 -> (ok, details)."""
        return True, {}

    def deliver(self, env, name):
        """Is this actor observable now (sight line)? Gates the expert's view and the observation."""
        return True

    def done(self, env):
        """Scenario complete before the horizon (the episode then ends and is scored as at the horizon)."""
        return False

    def on_frame(self, frame):
        """Called once the site's LaneFrame exists (site constants in frame coordinates)."""

    def card(self):
        """Everything needed to rebuild the scenario's variations (written to <out>/<sid>/spec.json)."""
        return dict(sid=self.sid, town=self.town, site=self.site, description=self.description, T=self.T,
                    prefix_T=self.prefix_T, nominal=self.nominal,
                    ranges={k: list(v) for k, v in self.ranges.items()}, obs_names=self.obs_names,
                    sampling="uniform on each range, uniform choice on each list")

    @staticmethod
    def ego_features(env):
        e = env.ego
        return [e.speed, e.position[1] - env.scene["home_y"], e.heading, env.ego_yaw_rate, env.ego_acc,
                env.t / env.spec.T]

    EGO_NAMES = ["ego_v", "ego_dy_home", "ego_psi", "ego_yaw_rate", "ego_a", "t_frac"]


def rel(env, name):
    e, b = env.ego, env.bodies[name]
    return b.position[0] - e.position[0], b.position[1] - e.position[1], b.velocity[0] - e.velocity[0], b.velocity[1]


class RC03S1(Spec):
    """Highway secondary collision: a spun car slides into the ego lane; escape left into a gap in moving traffic.
    Town06 road 35 (5-lane motorway); scenario lanes = the three right lanes (-5, -6, -7), ego in -6."""
    sid, town = "RC03-S1", "Town06"
    site = dict(start_xy=(21.7, 143.0), ref_lane=-6, ref_idx=1, s0=60.0)
    T = 8.0
    nominal = dict(v=27.0, d0=25.0, v_left=24.0, left_gap=40.0, gap_center=-2.5, lead_brake=3.0, follow_T=1.5,
                   follow_react=0.8)
    ranges = dict(v=(25.0, 29.0), d0=(22.0, 28.0), v_left=(22.0, 26.0), left_gap=(36.0, 46.0),
                  gap_center=(-5.0, 0.0), lead_brake=(0.0, 5.0), follow_T=(1.0, 1.5), follow_react=(0.6, 1.0))
    obs_names = Spec.EGO_NAMES + ["victim_dx", "victim_dy", "victim_dvx", "victim_vy", "victim_psi",
                                  "striker_dx", "striker_dy", "striker_dvx",
                                  "lead_dx", "lead_dvx", "follow_dx", "follow_dvx"]

    def build(self, p, W):
        pT = self.prefix_T
        x_v = HALF_L + p["d0"] + 2.5
        vic = car("victim", "vehicle.dodge.charger_2020", 0, 2 * W, 20.0, kind="scripted",
                  path=crash_path(x_v, 2 * W, v_pre=20.0, v_post=22.0, a_slide=8.0, y_rest=1.1 * W, psi_rest=-0.6,
                                  T_slide=1.5))
        stk = car("striker", "vehicle.nissan.patrol_2021", 0, 2 * W, 26.0, kind="scripted",
                  path=crash_path(x_v - 5.3, 2 * W, v_pre=26.0, v_post=18.0, a_slide=8.0, y_rest=2 * W + 0.4,
                                  psi_rest=0.15, T_slide=1.0))
        vl = p["v_left"]
        lead_dx = p["gap_center"] + p["left_gap"] / 2 + 2.0
        follow_dx = p["gap_center"] - p["left_gap"] / 2 - 2.45
        lead = car("left_lead", "vehicle.mini.cooper_s_2021", lead_dx - vl * pT, 0.0, vl,
                   law=law_idm(vl, T=1.5, react=0.8, rubberneck=(1.0, p["lead_brake"], 12.0) if p["lead_brake"] > 0.2 else None))
        fol = car("left_follow", "vehicle.lincoln.mkz_2020", follow_dx - vl * pT, 0.0, vl,
                  law=law_idm(vl, T=p["follow_T"], react=p["follow_react"]))
        bg = platoon("l1_", -W, vl + 2.0, [-60.0, -20.0, 25.0, 70.0]) + platoon("l2_", -2 * W, vl + 5.0, [-40.0, 10.0, 60.0])
        return dict(ego=dict(x=-p["v"] * pT, y=W, v=p["v"]), bodies=[vic, stk, lead, fol] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=p["v"], props=[])

    def features(self, env):
        f = self.ego_features(env)
        dx, dy, dvx, vy = rel(env, "victim")
        f += [dx, dy, dvx, vy, env.bodies["victim"].heading]
        dx, dy, dvx, _ = rel(env, "striker")
        f += [dx, dy, dvx]
        for n in ("left_lead", "left_follow"):
            dx, _, dvx, _ = rel(env, n)
            f += [dx, dvx]
        return np.array(f, dtype=np.float32)


class RC07S1(Spec):
    """Moose on the highway, then a speeding car in the escape lane: swerve left, pass, return right in time.
    Town06 road 35; scenario lanes = (-6, -7), ego in -7 (right lane); the moose leaps from the right shoulder."""
    sid, town = "RC07-S1", "Town06"
    site = dict(start_xy=(21.7, 143.0), ref_lane=-7, ref_idx=1, s0=140.0)
    T = 12.0
    nominal = dict(v=25.0, d_moose=45.0, rear_v=38.0, rear_gap=80.0, rear_notice=25.0, rear_react=1.0, rear_b=5.0)
    ranges = dict(v=(23.0, 27.0), d_moose=(40.0, 50.0), rear_v=(36.0, 40.0), rear_gap=(65.0, 95.0),
                  rear_notice=(20.0, 30.0), rear_react=(0.8, 1.2), rear_b=(4.0, 6.0))
    obs_names = Spec.EGO_NAMES + ["moose_dx", "moose_dy", "moose_vy", "fast_dx", "fast_dy", "fast_dvx", "fast_a"]

    def build(self, p, W):
        pT = self.prefix_T
        x_m = HALF_L + p["d_moose"] + 0.45
        moose = Body("moose", x_m, 1.5 * W + 1.5, length=2.6, width=0.9, kind="scripted", bp="static.prop.box03",
                     path=crossing_path(x_m, 1.5 * W + 1.5, W, 0.0, 1.0))
        x_r = -HALF_L - p["rear_gap"] - 2.5
        fast = car("speeding_car", "vehicle.dodge.charger_2020", x_r - p["rear_v"] * pT, 0.0, p["rear_v"],
                   law=law_trigger(ego_ahead_in_lane(p["rear_notice"], half_w=0.5 * W), -p["rear_b"],
                                   reaction=p["rear_react"]))
        bg = (platoon("l1_", -W, 28.0, [-50.0, 0.0, 45.0, 95.0]) + platoon("l2_", -2 * W, 31.0, [-70.0, -10.0, 60.0])
              + platoon("l3_", -3 * W, 33.0, [-30.0, 40.0]))
        return dict(ego=dict(x=-p["v"] * pT, y=W, v=p["v"]), bodies=[moose, fast] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 1.5 * W), v_ref=p["v"], props=[])

    def features(self, env):
        f = self.ego_features(env)
        dx, dy, _, vy = rel(env, "moose")
        f += [dx, dy, vy]
        dx, dy, dvx, _ = rel(env, "speeding_car")
        f += [dx, dy, dvx, env.bodies["speeding_car"].acc]
        return np.array(f, dtype=np.float32)


class RC15S1(Spec):
    """Motorway on-ramp: the adjacent driver in the right lane yields, holds or matches; observe, then merge in
    front or behind. Town06 entrance ramp (road 58 -> junction 511): the ramp joins as an added right lane at CARLA
    x~18 (y 52.4). Town06's added ramp lanes never end (they continue as through lanes; only the median lane drops), so
    the end of the acceleration lane is marked with a cone taper END_X m after the ego's t0 position, as on a coned-off
    ramp lane. Frame reference = lane at y 48.9 (straight), ego lane = scenario lane 1. The ramp curve before it is
    tight, so the ego joins at 16-20 m/s and the motorway's right lane runs at 20-24 m/s."""
    sid, town = "RC15-S1", "Town06"
    site = dict(start_xy=(40.0, 48.9), ref_lane=-6, ref_idx=0, s0=0.0)
    T = 14.0
    END_X, TAPER = 180.0, 24.0        # cones close the acceleration lane from frame x = 180 (CARLA x ~220) over 24 m
    nominal = dict(mode="hold", v=18.0, v_adj=22.0, adj_dx=-4.0, react=0.5, lead_dx=45.0, follow_dx=-48.0)
    ranges = dict(mode=["yield", "hold", "match"], v=(16.0, 20.0), v_adj=(20.0, 24.0), adj_dx=(-7.0, -1.0),
                  react=(0.3, 0.9), lead_dx=(40.0, 50.0), follow_dx=(-53.0, -43.0))
    obs_names = Spec.EGO_NAMES + ["adj_dx", "adj_dvx", "adj_a", "adj_dvx_0.5s", "adj_dvx_1s", "lead_dx", "lead_dvx",
                                  "follow_dx", "follow_dvx", "end_dx"]

    def build(self, p, W):
        pT, va = self.prefix_T, p["v_adj"]
        ka = STYLES["pronounced"]
        gap0, s_eq = (p["adj_dx"] - p["follow_dx"]) - 4.8, 2.0 + 1.2 * va
        v_follow = va / (1.0 - (s_eq / gap0) ** 2) ** 0.25 if gap0 > 1.05 * s_eq else va + 3.0
        adj = car("adj", "vehicle.ford.mustang", p["adj_dx"] - va * pT, 0.0, va,
                  law=law_adjacent(p["mode"], p["react"], W, -1.0, va, lane_w=W, **ka))
        lead = car("tgt_lead", "vehicle.toyota.prius", p["lead_dx"] - va * pT, 0.0, va, law=law_idm(va, T=1.2, react=0.5))
        fol = car("tgt_follow", "vehicle.nissan.patrol_2021", p["follow_dx"] - va * pT, 0.0, va,
                  law=law_idm(v_follow, T=1.2, react=0.8))
        end = Body("lane_end", self.END_X + 40.0, W, length=80.0, width=W - 0.6, kind="static", bp=None)
        bg = platoon("l1_", -W, va + 3.0, [-80.0, -30.0, 20.0, 75.0]) + platoon("l2_", -2 * W, va + 6.0, [-55.0, 5.0, 65.0])
        e0, tp = self.END_X, self.TAPER

        def bounds(x):
            u = min(max((x - e0) / tp, 0.0), 1.0)
            return -W / 2, 1.5 * W - u * W
        cones = [("static.prop.constructioncone", e0 + 2.0 * i, 1.5 * W - W * min(2.0 * i / tp, 1.0) - 0.3)
                 for i in range(int(tp / 2) + 1)]
        return dict(ego=dict(x=-p["v"] * pT, y=W, v=p["v"]), bodies=[adj, lead, fol, end] + bg, home_y=W,
                    bounds=bounds, v_ref=p["v"], props=cones)

    def features(self, env):
        f = self.ego_features(env)
        dx, _, dvx, _ = rel(env, "adj")
        h = env.speed_hist["adj"]
        ev = env.speed_hist["ego"]
        a_adj = (h[-1] - h[-11]) / 0.5 if len(h) > 10 else 0.0
        past = lambda k: (h[-1 - k] - ev[-1 - k]) if len(h) > k else dvx
        f += [dx, dvx, a_adj, past(10), past(20)]
        for n in ("tgt_lead", "tgt_follow"):
            dx, _, dvx, _ = rel(env, n)
            f += [dx, dvx]
        f += [self.END_X - env.ego.position[0]]
        return np.array(f, dtype=np.float32)


SPECS = {s.sid: s for s in (RC03S1(), RC07S1(), RC15S1())}

from rc15 import SPECS15  # noqa: E402  (they import this module's helpers, defined above)
from rc19 import SPECS19  # noqa: E402
from rc03 import SPECS03  # noqa: E402
from rc07 import SPECS07  # noqa: E402
from rc11 import SPECS11  # noqa: E402
from rc03_opt import SPECS03O  # noqa: E402
from rc03_opts import SPECS03X  # noqa: E402
from rc07_opts import SPECS07X  # noqa: E402
from rc01_opts import SPECS01X  # noqa: E402
from rc10_opts import SPECS10X  # noqa: E402
from rc15_opts import SPECS15X  # noqa: E402
from rc14_opts import SPECS14X  # noqa: E402
from rc16_opts import SPECS16X  # noqa: E402
from rc08_opts import SPECS08X  # noqa: E402

for _d in (SPECS15, SPECS19, SPECS03, SPECS07, SPECS11, SPECS03O, SPECS03X, SPECS07X, SPECS01X, SPECS10X, SPECS15X, SPECS14X, SPECS16X, SPECS08X):
    SPECS.update(_d)

from rc19_opts import specs19  # noqa: E402
SPECS.update(specs19(SPECS))
