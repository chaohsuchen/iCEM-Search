"""RC03 scenarios (braking fails, a lateral escape remains) on real CARLA sites, one teacher per scenario.

    RC03-S2  Town06 road 35, lanes (-5, -6, -7): the car ahead swerves left at the last moment and uncovers a stalled
             van; the left lane is taken by the car the lead squeezed in front of -> escape right.
    RC03-S4  Town07 road 11 (rural 1+1, lane 1 = opposing): a moose steps out of the right verge and freezes in the ego
             lane -> swerve through the opposing lane in a gap before an oncoming car and return. Also covers RC07-S4
             (cyclist, the same swerve-and-return through the opposing lane).
    RC03-S5  Town04 road 49, lanes (3, 4) next to the median, at night: a wrong-way driver in the ego's lane; the truck
             being overtaken blocks the right lane -> brake hard to drop behind it, then steer into the slot.
    RC03-S6  Town04 road 49, lanes (5, 6) and the real 3.5 m hard shoulder: a crate falls off the pickup ahead into the
             left part of the ego lane, a car sits in the blind spot -> partial shift right onto the shoulder and back.
    RC03-S8  Town03 road 3 (2 lanes northbound + centre lane), side street road 62 from the right at junction 861:
             the side-street car pulls out and stops half in the ego lane -> change to the left lane.
Not built: RC03-S3 (no stock map has a crest sharp enough: the Town04 road 40 crest shows the queue from >= 125 m,
where braking from 30 m/s stops in time) and RC03-S7 (= RC03-S1's evasive lane change into a gap; RC03-S1's teacher).
"""
import math

import numpy as np

import carla

from laws import (HALF_L, Body, crash_path, crossing_path, const_path, lane_change_path, law_idm)
from scenarios import CAR, Spec, car, platoon, rel


def seen_rel(env, name, sentinel=(60.0, 0.0, 0.0, 0.0)):
    """(seen, dx, dy, dvx, vy) of an actor if the spec delivers it now, else (0, sentinel)."""
    if env.spec.deliver(env, name):
        return [1.0, *rel(env, name)]
    return [0.0, *sentinel]


def in_lane_goal(env, y_lane, tol=1.0):
    y = env.ego.position[1]
    return abs(y - y_lane) <= tol, dict(final_lane_offset=round(y - y_lane, 2))


class RC03S2(Spec):
    """Late cut-out: the lead swerves left round a stalled van it was hiding; the left lane is taken -> escape right."""
    sid, town = "RC03-S2", "Town06"
    site = dict(start_xy=(21.7, 143.0), ref_lane=-6, ref_idx=1, s0=60.0)
    T = 8.0
    nominal = dict(v=25.0, headway=1.0, cut_ttc=1.2, T_cut=1.2, v_right=22.0, right_dx=35.0, left_dx=-1.0)
    ranges = dict(v=(24.0, 26.0), headway=(0.9, 1.1), cut_ttc=(0.9, 1.4), T_cut=(1.0, 1.4), v_right=(20.0, 24.0),
                  right_dx=(31.0, 39.0), left_dx=(-3.0, 1.0))
    obs_names = Spec.EGO_NAMES + ["stalled_seen", "stalled_dx", "stalled_dy", "lead_dx", "lead_dy", "lead_vy",
                                  "left_dx", "left_dvx", "right_dx", "right_dvx"]
    description = "Town06 motorway: the lead swerves left round a stalled van; left lane taken: escape right"

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        Ll, Ls = CAR["vehicle.toyota.prius"][0], CAR["vehicle.volkswagen.t2_2021"][0]
        x_lead = HALF_L + v * p["headway"] + Ll / 2
        x_st = x_lead + Ll / 2 + v * p["cut_ttc"] + Ls / 2
        path = lane_change_path(x_lead, W, v, 0.0, 0.0, p["T_cut"])
        t_rev = next(t for t in np.arange(0.0, p["T_cut"], 0.05) if W - path(t)[1] > 2.0)   # lead body clears the view
        lead = car("lead", "vehicle.toyota.prius", 0, W, v, kind="scripted", path=path)
        stalled = Body("stalled", x_st, W, length=Ls, width=CAR["vehicle.volkswagen.t2_2021"][1], kind="static",
                       bp="vehicle.volkswagen.t2_2021")
        left = car("left_car", "vehicle.lincoln.mkz_2020", p["left_dx"] - v * pT, 0.0, v, law=law_idm(v, T=1.2, react=0.8))
        vr = p["v_right"]
        right = car("right_car", "vehicle.mini.cooper_s_2021", p["right_dx"] - vr * pT, 2 * W, vr,
                    law=law_idm(vr, T=1.5, react=0.8))
        bg = platoon("l1_", -W, v + 2.0, [-60.0, -15.0, 30.0, 75.0]) + platoon("l2_", -2 * W, v + 4.0, [-40.0, 15.0, 65.0])
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[lead, stalled, left, right] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[], t_rev=float(t_rev))

    def deliver(self, env, name):
        return name != "stalled" or env.t >= env.scene["t_rev"]

    def features(self, env):
        s = seen_rel(env, "stalled")
        f = self.ego_features(env) + s[:3]
        dx, dy, _, vy = rel(env, "lead")
        f += [dx, dy, vy]
        for n in ("left_car", "right_car"):
            dx, _, dvx, _ = rel(env, n)
            f += [dx, dvx]
        return np.array(f, dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, 2 * env.W)


class RC03S4(Spec):
    """Moose on a rural two-way road: swerve through the opposing lane before the oncoming car, and return.
    Town07 road 11 northbound (lane -1 = ego at y W, lane 1 = opposing at y 0), village junctions every ~60 m."""
    sid, town = "RC03-S4", "Town07"
    site = dict(start_xy=(-3.6, -228.6), ref_lane=-1, ref_idx=1, s0=40.0)   # ego spawns 20 m before s0: past the junction exit
    T = 8.0
    nominal = dict(v=22.0, d0=38.0, moose_dy=0.0, onc_dist=200.0, onc_v=22.0)
    ranges = dict(v=(20.0, 24.0), d0=(35.0, 41.0), moose_dy=(-0.6, 0.0), onc_dist=(175.0, 225.0), onc_v=(20.0, 24.0))
    obs_names = Spec.EGO_NAMES + ["moose_dx", "moose_dy", "moose_vy", "onc_dx", "onc_dvx"]
    description = "Town07 rural two-way road: moose in the lane, swerve through the opposing lane before the oncoming car"

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        x_m = HALF_L + p["d0"] + 0.45
        moose = Body("moose", x_m, 1.5 * W + 1.5, length=2.6, width=0.9, kind="scripted", bp="static.prop.box03",
                     path=crossing_path(x_m, 1.5 * W + 1.5, W + p["moose_dy"], 0.0, 1.0))
        Lo = CAR["vehicle.ford.mustang"][0]
        onc = car("oncoming", "vehicle.ford.mustang", 0, 0.0, p["onc_v"], kind="scripted",
                  path=const_path(HALF_L + p["onc_dist"] + Lo / 2, 0.0, p["onc_v"], psi=math.pi))
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[moose, onc], home_y=W, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[])

    def features(self, env):
        dx, dy, _, vy = rel(env, "moose")
        f = self.ego_features(env) + [dx, dy, vy]
        dx, _, dvx, _ = rel(env, "oncoming")
        return np.array(f + [dx, dvx], dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, env.W)


class RC03S5(Spec):
    """Night, wrong-way driver in the fast lane; brake hard to drop behind the truck being overtaken, then slot in.
    Town04 road 49 northbound: lane 3 (next to the median) = y 0 = ego, lane 4 = y W (truck); lanes 5, 6 carry
    background traffic (the scenario's carriageway is the two inner lanes, as in the notebook)."""
    sid, town = "RC03-S5", "Town04"
    site = dict(start_xy=(-9.9, -202.8), ref_lane=3, ref_idx=0, s0=30.0)
    T = 8.0
    weather = dict(cloudiness=10.0, precipitation=0.0, sun_altitude_angle=-40.0, sun_azimuth_angle=150.0, fog_density=0.0)
    TRUCK = "vehicle.carlamotors.european_hgv"
    nominal = dict(v=30.0, d_ww=200.0, v_ww=30.0, v_truck=23.0, truck_gap=5.0, follow_gap=30.0)
    ranges = dict(v=(29.0, 31.0), d_ww=(190.0, 230.0), v_ww=(26.0, 34.0), v_truck=(22.5, 25.0), truck_gap=(4.0, 6.0),
                  follow_gap=(27.0, 35.0))
    obs_names = Spec.EGO_NAMES + ["ww_seen", "ww_dx", "ww_dvx", "truck_dx", "truck_dvx", "tf_dx", "tf_dvx"]
    description = "Town04 motorway at night: wrong-way driver ahead; drop behind the truck and slot in"

    def build(self, p, W):
        pT, v, vt = self.prefix_T, p["v"], p["v_truck"]
        Lt = CAR[self.TRUCK][0]
        x_truck = HALF_L + p["truck_gap"] + Lt / 2
        Lw = CAR["vehicle.dodge.charger_2020"][0]
        ww = car("wrong_way", "vehicle.dodge.charger_2020", 0, 0.0, p["v_ww"], kind="scripted",
                 path=const_path(HALF_L + p["d_ww"] + Lw / 2, 0.0, p["v_ww"], psi=math.pi))
        truck = car("truck", self.TRUCK, x_truck - vt * pT, W, vt, law=law_idm(vt, T=1.5, react=1.0))
        Lf = CAR["vehicle.audi.a2"][0]
        tf = car("truck_follower", "vehicle.audi.a2", x_truck - Lt / 2 - p["follow_gap"] - Lf / 2 - vt * pT, W, vt,
                 law=law_idm(vt, T=1.2, react=0.8))
        bg = (platoon("l5_", 2 * W, vt - 1.0, [-70.0, -35.0, 0.0, 35.0, 70.0, 105.0])
              + platoon("l6_", 3 * W, vt - 3.0, [-55.0, -15.0, 25.0, 65.0]))
        return dict(ego=dict(x=-v * pT, y=0.0, v=v), bodies=[ww, truck, tf] + bg, home_y=0.0,
                    bounds=lambda x: (-W / 2, 1.5 * W), v_ref=v, props=[])

    def deliver(self, env, name):
        return name != "wrong_way" or env.t >= -1e-6           # headlights first seen at t0 (d_ww)

    def features(self, env):
        s = seen_rel(env, "wrong_way", sentinel=(250.0, 0.0, 0.0, 0.0))
        f = self.ego_features(env) + [s[0], s[1], s[3]]
        for n in ("truck", "truck_follower"):
            dx, _, dvx, _ = rel(env, n)
            f += [dx, dvx]
        return np.array(f, dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, env.W)


class RC03S6(Spec):
    """A crate falls off the pickup ahead into the left part of the ego lane; a car sits in the blind spot -> partial
    shift right onto the hard shoulder, then back. Town04 road 49: lane 5 = y 0 (blind-spot car), lane 6 = y W (ego),
    then the real 3.5 m hard shoulder."""
    sid, town = "RC03-S6", "Town04"
    site = dict(start_xy=(-9.9, -202.8), ref_lane=5, ref_idx=0, s0=30.0)
    T = 8.0
    SHOULDER = 3.5
    PICKUP = "vehicle.nissan.patrol_2021"
    nominal = dict(v=28.0, headway=1.0, debris_y=-0.5, a_debris=15.0, blind_dx=-3.0)
    ranges = dict(v=(26.0, 30.0), headway=(0.9, 1.1), debris_y=(-0.6, -0.2), a_debris=(12.0, 16.0), blind_dx=(-4.5, 1.5))
    obs_names = Spec.EGO_NAMES + ["crate_dx", "crate_dy", "crate_dvx", "crate_vy", "pickup_dx", "blind_dx", "blind_dvx"]
    description = "Town04 motorway: crate falls into the lane, blind-spot car on the left: shift onto the shoulder and back"

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        Lp = CAR[self.PICKUP][0]
        x_p = HALF_L + v * p["headway"] + Lp / 2
        pickup = car("pickup", self.PICKUP, 0, W, v, kind="scripted", path=const_path(x_p, W, v))
        crate = Body("crate", x_p - 1.6, W, length=1.5, width=1.0, kind="scripted", bp="static.prop.box01",
                     path=crash_path(x_p - 1.6, W, v, v, p["a_debris"], W + p["debris_y"], 0.9, 0.8))
        blind = car("blind_spot_car", "vehicle.lincoln.mkz_2020", p["blind_dx"] - v * pT, 0.0, v,
                    law=law_idm(v, T=1.2, react=0.8))
        bg = platoon("l4_", -W, v + 1.0, [-60.0, -20.0, 25.0, 70.0]) + platoon("l3_", -2 * W, v + 3.0, [-45.0, 10.0, 60.0])
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[pickup, crate, blind] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 1.5 * W + self.SHOULDER), v_ref=v, props=[])

    def features(self, env):
        f = self.ego_features(env) + list(rel(env, "crate")) + [rel(env, "pickup")[0]]
        dx, _, dvx, _ = rel(env, "blind_spot_car")
        return np.array(f + [dx, dvx], dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, env.W)


class CityStop(Spec):
    """Town03 road 3 northbound (lanes -1, -2 = y 0, W; a centre lane on the left, 2.5 m shoulder + kerb on the right);
    side street road 62 joins from the right at junction 861 (its inbound lane meets the arterial at world (-98.2, 0.4)).
    The side-street car, already crept up to the shoulder edge, pulls out at t0 and stops with its front `protrude` m
    into the ego lane, d_x m (bumper gap) ahead of the ego."""
    town = "Town03"
    site = dict(start_xy=(-88.7, -122.9), ref_lane=-1, ref_idx=0, s0=0.0)
    SIDE_WORLD = (-98.2, 0.4)
    KERB = 2.5 + 0.63                                     # shoulder + kerb strip right of the ego lane
    CROSS = "vehicle.lincoln.mkz_2020"
    x_side = 123.3

    def on_frame(self, frame):
        self.x_side = frame.from_world(carla.Location(x=self.SIDE_WORLD[0], y=self.SIDE_WORLD[1]), 90.0)[0]

    def crossing(self, p, W):
        L, Wc = CAR[self.CROSS]
        y0 = 1.5 * W + self.KERB + L / 2
        y1 = 1.5 * W + L / 2 - p["protrude"]
        return Body("crossing_car", self.x_side, y0, psi=-math.pi / 2, length=L, width=Wc, kind="scripted", bp=self.CROSS,
                    path=crossing_path(self.x_side, y0, y1, 0.0, p["T_creep"]))

    def ego_x0(self, p):
        return self.x_side - CAR[self.CROSS][1] / 2 - p["d_x"] - HALF_L          # ego centre at t0

    def left_lead(self, p, x_e):
        v = p["v"] - 1.0
        return car("left_lead", "vehicle.audi.a2", x_e + p["lead_dx"] - v * self.prefix_T, 0.0, v,
                   law=law_idm(v, T=1.5, react=0.8))


class RC03S8(CityStop):
    sid = "RC03-S8"
    T = 8.0
    nominal = dict(v=16.7, d_x=26.0, protrude=2.4, T_creep=1.5, lead_dx=45.0)
    ranges = dict(v=(15.5, 17.5), d_x=(23.0, 29.0), protrude=(2.0, 2.8), T_creep=(1.2, 1.8), lead_dx=(40.0, 50.0))
    obs_names = Spec.EGO_NAMES + ["cc_dx", "cc_dy", "cc_vy", "lead_dx", "lead_dvx"]
    description = "Town03 arterial: a side-street car pulls out half into the ego lane: change to the left lane"

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        x_e = self.ego_x0(p)
        return dict(ego=dict(x=x_e - v * pT, y=W, v=v), bodies=[self.crossing(p, W), self.left_lead(p, x_e)], home_y=W,
                    bounds=lambda x: (-W / 2, 1.5 * W), v_ref=v, props=[])

    def features(self, env):
        dx, dy, _, vy = rel(env, "crossing_car")
        f = self.ego_features(env) + [dx, dy, vy]
        dx, _, dvx, _ = rel(env, "left_lead")
        return np.array(f + [dx, dvx], dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, 0.0)


SPECS03 = {s.sid: s for s in (RC03S2(), RC03S4(), RC03S5(), RC03S6(), RC03S8())}
