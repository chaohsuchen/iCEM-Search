"""RC07 scenarios (the primary hazard is avoidable; the hand-back after it decides the outcome) on real CARLA sites.

    RC07-S2  Town06 road 35, lanes (-5, -6, -7): a ladder falls off a van into the ego lane; the left lane is a dense
             platoon -> swerve into a gap in the faster right lane and accelerate to its speed.
    RC07-S3  Town04 road 49, lanes (3, 4) in congestion: a shed tyre tread in the right half of the ego lane -> shift
             left inside the lane, then move back past the centre to its right half: a motorcycle is lane-splitting
             along the left lane line.
    RC07-S6  Town03 road 3 / side street 62 (as RC03-S8): the side-street car pulls out half into the ego lane ->
             change left, then return right once past it, before the speeding car in the left lane arrives.
    RC07-S7  Town06 road 35: a wheel comes off the truck ahead into the middle lane; broken-down car A stands in the
             right lane and car B has planned its stop behind A -> swerve right past the wheel, then cut back to the
             middle (stopping behind A would take B's stopping space). Also covers RC07-S5 (swerve right, return left).
Not built: RC07-S4 (= RC03-S4's swerve through the opposing lane and back), RC07-S5 (= RC07-S7's swerve and cut back).
"""
import numpy as np

from laws import (HALF_L, Body, crash_path, const_path, ego_ahead_in_lane, law_idm, law_planned_stop, law_trigger)
from rc03 import CityStop, in_lane_goal
from scenarios import CAR, Spec, car, platoon, rel


class RC07S2(Spec):
    """Ladder in the lane; dense platoon on the left -> swerve into the faster right lane and match its speed."""
    sid, town = "RC07-S2", "Town06"
    site = dict(start_xy=(21.7, 143.0), ref_lane=-6, ref_idx=1, s0=40.0)
    T = 12.0
    VAN = "vehicle.mercedes.sprinter"
    nominal = dict(v=25.0, headway=1.5, v_right=35.0, f_gap=45.0, dv_f=0.0, f_notice=20.0, f_react=1.2)
    ranges = dict(v=(24.0, 26.0), headway=(1.3, 1.7), v_right=(33.0, 36.0), f_gap=(36.0, 60.0), dv_f=(-1.0, 1.0),
                  f_notice=(18.0, 22.0), f_react=(1.0, 1.4))
    obs_names = Spec.EGO_NAMES + ["ladder_dx", "ladder_dy", "ladder_dvx", "ladder_vy", "rf_dx", "rf_dvx", "rf_a",
                                  "rl_dx", "rl_dvx"]
    description = "Town06 motorway: ladder in the lane, dense left lane: swerve into the faster right lane and match it"

    def build(self, p, W):
        pT, v, vr = self.prefix_T, p["v"], p["v_right"]
        Lv = CAR[self.VAN][0]
        x_van = HALF_L + v * p["headway"] + Lv / 2
        van = car("van", self.VAN, 0, W, v, kind="scripted", path=const_path(x_van, W, v))
        ladder = Body("ladder", x_van - 1.0, W, length=3.6, width=0.5, kind="scripted", bp="static.prop.box02",
                      path=crash_path(x_van - 1.0, W, v, v, 12.0, W + 0.3, 0.35, 1.0))
        fv = vr + p["dv_f"]
        Lf = CAR["vehicle.dodge.charger_2020"][0]
        rf = car("right_follower", "vehicle.dodge.charger_2020", -HALF_L - p["f_gap"] - Lf / 2 - fv * pT, 2 * W, fv,
                 law=law_trigger(ego_ahead_in_lane(p["f_notice"], half_w=0.5 * W), -5.0, reaction=p["f_react"]))
        rl = car("right_lead", "vehicle.toyota.prius", 45.0 - vr * pT, 2 * W, vr, law=law_idm(vr, T=1.5, react=0.8))
        left = [car(f"bg_left_{i}", ("vehicle.audi.a2", "vehicle.ford.mustang", "vehicle.toyota.prius")[i % 3],
                    dx - v * pT, 0.0, v, law=law_idm(v, T=1.0, react=0.8)) for i, dx in enumerate((-24.0, -12.0, 0.0, 12.0, 24.0))]
        bg = (left + platoon("r_", 2 * W, vr, [-95.0, 95.0, 140.0]) + platoon("l1_", -W, v + 3.0, [-50.0, 0.0, 50.0])
              + platoon("l2_", -2 * W, v + 6.0, [-30.0, 40.0]))
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[van, ladder, rf, rl] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def features(self, env):
        f = self.ego_features(env) + list(rel(env, "ladder"))
        dx, _, dvx, _ = rel(env, "right_follower")
        f += [dx, dvx, env.bodies["right_follower"].acc]
        dx, _, dvx, _ = rel(env, "right_lead")
        return np.array(f + [dx, dvx], dtype=np.float32)

    def goal(self, env):
        ok, d = in_lane_goal(env, 2 * env.W)
        need = env.params["v_right"] - 4.0
        d.update(v_end=round(env.ego.speed, 2), v_needed=round(need, 2))
        return ok and env.ego.speed >= need, d


class RC07S3(Spec):
    """Tyre tread in the right half of the lane, dense queue on the left -> shift left inside the lane, then move back
    past the centre to the right half before the motorcycle lane-splitting along the left line arrives. Town04 road 49: lane 3 = y 0 (queue), lane 4 = y W (ego);
    lanes 5, 6 carry background traffic (the corridor ends at the ego lane's right line, as the notebook's guard rail).
    The queue keeps 0.25 m left of its lane centre, leaving room for the motorcycle filtering along the line."""
    sid, town = "RC07-S3", "Town04"
    site = dict(start_xy=(-9.9, -202.8), ref_lane=3, ref_idx=0, s0=30.0)
    T = 12.0
    MOTO = "vehicle.yamaha.yzf"
    nominal = dict(v=20.0, d_tread=30.0, tread_y=1.1, moto_v=30.0, moto_gap=52.0, moto_dy=0.0, moto_notice=15.0)
    ranges = dict(v=(19.0, 21.0), d_tread=(28.0, 32.0), tread_y=(1.0, 1.2), moto_v=(28.0, 32.0), moto_gap=(47.0, 60.0),
                  moto_dy=(-0.2, 0.0), moto_notice=(13.0, 17.0))
    obs_names = Spec.EGO_NAMES + ["tread_seen", "tread_dx", "tread_dy", "moto_dx", "moto_dy", "moto_dvx", "lead_dx"]
    description = "Town04 congested motorway: tyre tread in the lane: shift left in lane, then right of centre for the motorcycle"

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        x_t = HALF_L + p["d_tread"] + 0.6
        tread = Body("tread", x_t, W + p["tread_y"], length=1.2, width=0.6, kind="static", bp="static.prop.box01")
        lead = car("lead", "vehicle.audi.a2", 0, W, v, kind="scripted", path=const_path(x_t + 8.0, W, v))
        mv = p["moto_v"]
        moto = Body("motorcycle", -HALF_L - p["moto_gap"] - 1.1 - mv * pT, W / 2 + p["moto_dy"], v=mv, length=2.2,
                    width=0.8, bp=self.MOTO,
                    law=law_trigger(ego_ahead_in_lane(p["moto_notice"], half_w=1.3), -5.0, reaction=0.8))
        queue = [car(f"bg_q{i}", ("vehicle.audi.a2", "vehicle.mini.cooper_s_2021")[i % 2], dx - v * pT, -0.25, v,
                     law=law_idm(v, T=0.8, react=0.8)) for i, dx in enumerate(np.arange(-48.0, 61.0, 12.0))]
        bg = queue + platoon("l5_", 2 * W, v, [-50.0, -10.0, 30.0, 70.0]) + platoon("l6_", 3 * W, v - 2.0, [-30.0, 20.0, 60.0])
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[tread, lead, moto] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 1.5 * W), v_ref=v, props=[])

    def deliver(self, env, name):
        return name != "tread" or env.t >= -1e-6               # revealed by the car ahead at t0

    def features(self, env):
        if self.deliver(env, "tread"):
            dx, dy, _, _ = rel(env, "tread")
            f = self.ego_features(env) + [1.0, dx, dy]
        else:
            f = self.ego_features(env) + [0.0, 60.0, 0.0]
        dx, dy, dvx, _ = rel(env, "motorcycle")
        return np.array(f + [dx, dy, dvx, rel(env, "lead")[0]], dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, env.W, tol=0.6)


class RC07S6(CityStop):
    """As RC03-S8, plus a speeding car coming up the left lane: change left, pass, return right before it arrives."""
    sid = "RC07-S6"
    T = 11.0
    nominal = dict(v=16.7, d_x=26.0, protrude=2.4, T_creep=1.5, lead_dx=45.0, rear_v=26.0, rear_gap=65.0,
                   rear_react=1.0)
    ranges = dict(v=(15.5, 17.5), d_x=(23.0, 29.0), protrude=(2.0, 2.8), T_creep=(1.2, 1.8), lead_dx=(40.0, 50.0),
                  rear_v=(24.0, 28.0), rear_gap=(55.0, 85.0), rear_react=(0.8, 1.2))
    obs_names = Spec.EGO_NAMES + ["cc_dx", "cc_dy", "cc_vy", "lead_dx", "lead_dvx", "sp_dx", "sp_dvx", "sp_a"]
    description = "Town03 arterial: side-street car pulls out; change left, return right before the speeding car"

    def build(self, p, W):
        pT, v, vr = self.prefix_T, p["v"], p["rear_v"]
        x_e = self.ego_x0(p)
        Ls = CAR["vehicle.dodge.charger_2020"][0]
        sp = car("speeding_car", "vehicle.dodge.charger_2020", x_e - HALF_L - p["rear_gap"] - Ls / 2 - vr * pT, 0.0, vr,
                 law=law_trigger(ego_ahead_in_lane(20.0, half_w=0.5 * W), -5.0, reaction=p["rear_react"]))
        return dict(ego=dict(x=x_e - v * pT, y=W, v=v), bodies=[self.crossing(p, W), self.left_lead(p, x_e), sp],
                    home_y=W, bounds=lambda x: (-W / 2, 1.5 * W), v_ref=v, props=[])

    def features(self, env):
        dx, dy, _, vy = rel(env, "crossing_car")
        f = self.ego_features(env) + [dx, dy, vy]
        dx, _, dvx, _ = rel(env, "left_lead")
        f += [dx, dvx]
        dx, _, dvx, _ = rel(env, "speeding_car")
        return np.array(f + [dx, dvx, env.bodies["speeding_car"].acc], dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, env.W)


class RC07S7(Spec):
    """Wheel off the truck ahead into the middle lane; left lane full; broken-down car A in the right lane with car B
    planning its stop behind it -> swerve right past the wheel, then cut back to the middle lane."""
    sid, town = "RC07-S7", "Town06"
    site = dict(start_xy=(21.7, 143.0), ref_lane=-6, ref_idx=1, s0=60.0)
    T = 12.0
    TRUCK = "vehicle.carlamotors.european_hgv"
    nominal = dict(v=25.0, d_obs=22.0, x_a=95.0, b_x0=-26.0, b_v0=31.0, b_t=0.5, b_react=1.2, b_max=6.0)
    ranges = dict(v=(24.0, 26.0), d_obs=(20.0, 24.0), x_a=(90.0, 100.0), b_x0=(-34.0, -20.0), b_v0=(30.0, 32.0),
                  b_t=(0.4, 0.6), b_react=(1.0, 1.4), b_max=(6.0, 6.5))
    obs_names = Spec.EGO_NAMES + ["wheel_dx", "wheel_dy", "wheel_dvx", "a_dx", "b_dx", "b_dvx", "b_a"]
    description = "Town06 motorway: wheel in the lane, swerve right past it, cut back before car B's planned stop"

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        Lt = CAR[self.TRUCK][0]
        x_truck = HALF_L + v * 1.5 + Lt / 2
        x_w = HALF_L + p["d_obs"] + 0.4
        truck = car("truck", self.TRUCK, 0, W, v, kind="scripted", path=const_path(x_truck, W, v))
        wheel = Body("wheel", x_w + 12.0, W, length=0.8, width=0.8, kind="scripted", bp="static.prop.box01",
                     path=crash_path(x_w + 12.0, W, v, v - 3.0, 14.0, W - 0.2, 0.0, 0.8))
        La = CAR["vehicle.toyota.prius"][0]
        car_a = Body("car_A", p["x_a"] + La / 2, 2 * W, length=La, width=CAR["vehicle.toyota.prius"][1], kind="static",
                     bp="vehicle.toyota.prius")
        bv, Lb = p["b_v0"], CAR["vehicle.lincoln.mkz_2020"][0]
        b_plan = bv ** 2 / (2.0 * ((p["x_a"] - 2.0) - (p["b_x0"] + Lb / 2) - bv * p["b_t"]))
        car_b = car("car_B", "vehicle.lincoln.mkz_2020", p["b_x0"] - bv * pT, 2 * W, bv,
                    law=law_planned_stop(p["b_t"], b_plan, 25.0, p["b_react"], p["b_max"], half_w=0.5 * W))
        left = [car(f"bg_left_{i}", ("vehicle.audi.a2", "vehicle.ford.mustang", "vehicle.toyota.prius")[i % 3],
                    dx - v * pT, 0.0, v, law=law_idm(v, T=1.0, react=0.8))
                for i, dx in enumerate((-36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0))]
        bg = left + platoon("l1_", -W, v + 3.0, [-60.0, -10.0, 40.0]) + platoon("l2_", -2 * W, v + 6.0, [-35.0, 30.0])
        return dict(ego=dict(x=-v * pT, y=W, v=v), bodies=[truck, wheel, car_a, car_b] + bg, home_y=W,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def features(self, env):
        dx, dy, dvx, _ = rel(env, "wheel")
        f = self.ego_features(env) + [dx, dy, dvx, rel(env, "car_A")[0]]
        dx, _, dvx, _ = rel(env, "car_B")
        return np.array(f + [dx, dvx, env.bodies["car_B"].acc], dtype=np.float32)

    def goal(self, env):
        return in_lane_goal(env, env.W)


SPECS07 = {s.sid: s for s in (RC07S2(), RC07S3(), RC07S6(), RC07S7())}
