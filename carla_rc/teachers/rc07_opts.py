"""RC07 with the options left open (as rc03_opts) and the hand-back as a second decision (2026-11-12).

RC07: the primary hazard is avoidable; the hand-back after the avoidance decides the outcome. Families are composites
"stage1>handback" (options.composite): stage 1 is picked at the trigger (the observed intrusion), the hand-back at the
hand-back point (options.handback_seen: the ego has passed the actor that fired the trigger, or settled in another lane,
or T_HB_MAX after the trigger), when what decides it - a car closing in the lane the ego moved to, a car planning its
stop behind a broken-down one, a motorcycle filtering along the line - is observable. Every variation is executed with
every family; per stage-1 group the best member's stage 1 is run again with every other hand-back (cross runs), so the
stage-2 choice is learned from measured outcomes at the same state. Envelope: every policy's action z is mapped into
the qualified vehicle envelope (envelope = True) and scaled down near the contract (governor = True); the experts'
commands are capped by its friction circle.

    RC07-S1-opt  a moose runs out of the verge (left or right) across the near lane into the ego lane, stands there,
                 walks on into the far lane, or stops short (benign). A speeding, inattentive car may come up one of
                 the adjacent lanes; the other carries a gap pair; a car may follow the ego. Families keep / brake /
                 {left, right}>{return, stay}.
    RC07-S2-opt  a ladder falls off the van ahead into the ego lane. One adjacent lane is a dense platoon at the ego's
                 speed (a gap in it now and then), the other a faster lane with a car closing from behind (it brakes
                 once it notices the ego, late). Stay in the fast lane matching its speed, or return once past the
                 ladder. Families keep / brake / {left, right}>{return, stay}.
    RC07-S7-opt  a wheel comes off the truck ahead into the ego lane; a broken-down car A stands in one adjacent lane
                 further on and car B (in that lane, behind the ego) has planned its stop behind A; the other lane is a
                 platoon. Swerve into A's lane past the wheel, then cut back before A or stop behind A (B's stopping
                 space). Families keep / brake / {left, right}>{return, stop}. (RC07-S5 is the mirrored case: a_side.)
    RC07-S5-opt  the car ahead blows a tyre and skids to a stop in the ego lane; the right lane carries a fast car
                 coming up behind, and an on-ramp car in the acceleration lane beyond it cuts into the slot between
                 the ego and the fast car, then floors it once the fast car closes on it (it has not registered the
                 ego). Swerve right past the stopped car, then return left (unless a pile-up blocks it), stay, or pull
                 away. Families keep / brake / right>{return, stay, away}.
    RC07-S6-opt  RC03-S8-opt's side-street car (Town03, line of sight past the corner buildings, conflict trigger),
                 plus a speeding car that may come up the left lane. Families keep / brake / left>{return, stay}.
    RC07-S3-opt  congestion: a tyre tread in the right half of the ego lane (seen when the lead passes it); the left
                 lane's queue keeps left; a motorcycle may filter along the line between them. Shift left inside the
                 lane past the tread, then recentre (return) or come back right of centre (edge), or brake.
                 Families keep / brake / shift_left>{return, edge}.
"""
import math

import numpy as np

from laws import (HALF_L, Body, const_path, crash_path, crossing_path, ego_ahead_in_lane, law_fast, law_idm, law_onramp,
                  law_planned_stop, law_trigger)
from options import OBS_HB, OBS_NAMES, composite, handback_features
from rc03_opt import SITES as SITES03
from rc03_opts import OptBase, RC03S8Opt, SITES as SITES_X, gap_pair, rear_car, speed_path
from scenarios import CAR, car, platoon

CARS = ["vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.ford.mustang", "vehicle.lincoln.mkz_2020",
        "vehicle.mini.cooper_s_2021"]


class HBBase(OptBase):
    """Shared: role-free observation + the hand-back point, envelope with its governor, composite families."""
    governor = True
    comb_frac = 0.9                                                     # learned actions: the experts' 0.9 M cap
    obs_names = OBS_NAMES + OBS_HB

    def __init__(self):
        for m in self.modes:
            composite(m)

    def features(self, env):
        return handback_features(env)

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        return " / ".join(f"{m} {'ok' if ok(m) else 'X'}" for m in self.modes if m != "keep")

    def card(self):
        d = super().card()
        d.update(handback="options.handback_seen (passed the trigger's actor / settled in another lane / T_HB_MAX)")
        return d


def speeding(name, lane_y, x0, v, p, grip, pT):
    """An inattentive driver coming up a lane fast: constant speed until the ego is ahead of it in its lane within
    sp_notice m, then braking at sp_b (scaled by the grip) after sp_react s."""
    return car(name, "vehicle.dodge.charger_2020", x0 - v * pT, lane_y, v,
               law=law_trigger(ego_ahead_in_lane(p["sp_notice"], half_w=1.6), -p["sp_b"] * grip,
                               reaction=p["sp_react"]))


# ---------------------------------------------------------------- RC07-S1
class RC07S1Opt(HBBase):
    sid = "RC07-S1-opt"
    T = 11.0
    modes = ("keep", "brake", "left>return", "left>stay", "right>return", "right>stay")
    SPEEDS = ("v", "v_adj")
    nominal = dict(side=1, behav="stand", v=27.0, ttc_in=2.5, t_go=0.3, T_cross=1.8, moose_dy=0.0, sp_lane="away",
                   sp_dv=9.0, sp_gap=60.0, sp_notice=25.0, sp_react=1.0, sp_b=5.0, v_adj=26.0, gap=30.0, gap_center=-3.0,
                   rear=0, rear_gap=25.0, rear_react=0.9, wx="dry")
    ranges = dict(side=[1, -1], behav=["stand", "stand", "stand", "through", "short"], v=(24.0, 30.0),
                  ttc_in=(1.0, 4.0), t_go=(0.0, 1.0), T_cross=(1.3, 2.6), moose_dy=(-0.8, 0.8),
                  sp_lane=["away", "away", "near", "none"], sp_dv=(7.0, 14.0), sp_gap=(20.0, 80.0),
                  sp_notice=(8.0, 25.0), sp_react=(1.0, 2.0), sp_b=(3.0, 6.0), v_adj=(23.0, 28.0), gap=(8.0, 60.0),
                  gap_center=(-15.0, 10.0), rear=[0, 1], rear_gap=(8.0, 45.0), rear_react=(0.4, 1.2),
                  wx=["dry", "wet", "heavy"])
    description = ("A moose runs out of the verge across the near lane into the ego lane (stands, walks on into the far "
                   "lane, or stops short). A speeding car may come up an adjacent lane. Brake, or change lane past it and "
                   "then return or stay - decided once past it, when the closing car is in view.")

    def build(self, p, W):
        pT, v, s = self.prefix_T, p["v"], p["side"]
        grip = self.road(p)["grip"]
        y_e, y_near, y_far = W, W + s * W, W - s * W
        y0 = W + s * (1.5 * W + 2.0)                                    # in the verge beyond the near lane
        y1 = {"stand": y_e + p["moose_dy"], "through": y_far, "short": y_near + s * 0.3 * W}[p["behav"]]
        T_c = p["T_cross"] * abs(y1 - y0) / (2 * W + 2.0)               # the same running pace whatever the end point
        # placed so that the ego is ttc_in s from it when its footprint (2.6 m along y) enters the ego lane (a moose
        # that stops short: when it would have); then the observed intrusion leaves that much time
        path = crossing_path(0.0, y0, y1, p["t_go"], T_c, psi=-s * math.pi / 2)
        y_in = y_e + s * (W / 2 + 1.3)
        ts = np.arange(0.0, 10.0, 0.01)
        t_in = next((t for t in ts if s * (path(t)[1] - y_in) <= 0), p["t_go"] + 0.7 * T_c)
        x_m = HALF_L + v * (t_in + p["ttc_in"]) + 0.45
        moose = Body("moose", x_m, y0, length=2.6, width=0.9, kind="scripted", bp="static.prop.box03",
                     path=crossing_path(x_m, y0, y1, p["t_go"], T_c, psi=-s * math.pi / 2))
        bodies = [moose]
        va = p["v_adj"]
        lanes = {"away": y_far, "near": y_near}
        sp_y = lanes.get(p["sp_lane"])
        for y in (y_near, y_far):
            if y == sp_y:                                               # the speeding car's lane: it alone
                vs = v + p["sp_dv"]
                bodies.append(speeding("speeding_car", y, -HALF_L - p["sp_gap"] - 2.5, vs, p, grip, pT))
            else:
                bodies += gap_pair("near" if y == y_near else "far", y, va, p["gap_center"], p["gap"],
                                   CARS[0], CARS[3], pT, grip=grip)
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        # unobserved lanes further left: only when the moose comes from the right (from the left it ran across them)
        bg = platoon("l1_", -W, va + 2.0, [-60.0, -20.0, 25.0, 70.0]) if self.n_left >= 1 and s == 1 else []
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def cell(self, r):
        return f"{r['params']['behav']}: " + super().cell(r)


# ---------------------------------------------------------------- RC07-S2
class RC07S2Opt(HBBase):
    sid = "RC07-S2-opt"
    T = 12.0
    modes = ("keep", "brake", "left>return", "left>stay", "right>return", "right>stay")
    SPEEDS = ("v",)
    VAN = "vehicle.mercedes.sprinter"
    nominal = dict(fall=1, v=25.0, headway=1.5, ladder_dy=0.0, a_ladder=10.0, fast_side=1, f_dv=9.0, f_gap=45.0,
                   f_notice=20.0, f_react=1.2, f_b=5.0, f_lead_dx=60.0, lead_brake=0.0, lead_t=2.5, dense_sp=14.0, dense_off=0.0, rear=0,
                   rear_gap=25.0, rear_react=0.9, wx="dry")
    ranges = dict(fall=[1, 1, 1, 1, 0], v=(22.0, 28.0), headway=(1.0, 2.2), ladder_dy=(-0.8, 0.8), a_ladder=(6.0, 14.0),
                  fast_side=[1, -1], f_dv=(5.0, 12.0), f_gap=(10.0, 80.0), f_notice=(15.0, 30.0), f_react=(0.8, 1.6),
                  f_b=(3.0, 6.0), f_lead_dx=(25.0, 80.0), lead_brake=(-3.0, 5.0), lead_t=(1.5, 4.0),
                  dense_sp=(11.0, 32.0), dense_off=(-6.0, 6.0), rear=[0, 1],
                  rear_gap=(8.0, 45.0), rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("A ladder falls off the van ahead into the ego lane; one adjacent lane is a dense platoon, the other a "
                   "faster lane with a car closing from behind. Brake, or change lane past the ladder, then stay (matching "
                   "the lane's speed) or return - decided once past it.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        sp = self.road(p)["speed"] * self.site_speed                    # the road's speed level (v is scaled already)
        y_e, f = W, p["fast_side"]
        y_fast, y_dense = W + f * W, W - f * W
        Lv = CAR[self.VAN][0]
        x_van = HALF_L + v * p["headway"] + Lv / 2
        van = car("van", self.VAN, 0, y_e, v, kind="scripted", path=const_path(x_van, y_e, v))
        x_l = x_van - Lv / 2 - 1.0
        y_rest = y_e + p["ladder_dy"] if p["fall"] else y_e
        ladder = Body("ladder", x_l, y_e, length=3.6, width=0.5, kind="scripted", bp="static.prop.box02",
                      path=crash_path(x_l, y_e, v, v, p["a_ladder"] * grip, y_rest, 0.35, 1.0)
                      if p["fall"] else const_path(x_l, y_e, v))
        bodies = [van, ladder]
        vf = v + p["f_dv"] * sp
        Lf = CAR["vehicle.dodge.charger_2020"][0]
        bodies.append(car("fast_follow", "vehicle.dodge.charger_2020", -HALF_L - p["f_gap"] - Lf / 2 - vf * pT, y_fast, vf,
                          law=law_trigger(ego_ahead_in_lane(p["f_notice"], half_w=1.6), -p["f_b"] * grip,
                                          reaction=p["f_react"])))
        lb = max(p["lead_brake"], 0.0) * grip              # the fast lane's lead may slow down (it closes "stay")
        bodies.append(car("fast_lead", "vehicle.toyota.prius", p["f_lead_dx"] - vf * pT, y_fast, vf,
                          law=law_idm(vf, T=1.5, react=0.8, b_max=6 * grip,
                                      rubberneck=(p["lead_t"], lb, vf - 10.0) if lb > 0.2 else None)))
        xs = np.arange(-4, 5) * p["dense_sp"] + p["dense_off"]          # the dense lane around the ego
        bodies += [car(f"dense{i}", CARS[i % len(CARS)], x - v * pT, y_dense, v,
                       law=law_idm(v, T=0.9, react=0.8, b_max=6 * grip)) for i, x in enumerate(xs)]
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        bg = platoon("l1_", -W, v + 2.0, [-60.0, -20.0, 25.0, 70.0]) if self.n_left >= 1 else []
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def cell(self, r):
        return ("fall: " if r["params"]["fall"] else "benign: ") + super().cell(r)


# ---------------------------------------------------------------- RC07-S7 (and S5)
class RC07S7Opt(HBBase):
    sid = "RC07-S7-opt"
    T = 12.0
    modes = ("keep", "brake", "left>return", "left>stop", "right>return", "right>stop")
    SPEEDS = ("v",)
    TRUCK = "vehicle.carlamotors.european_hgv"
    nominal = dict(wheel=1, v=25.0, headway=1.5, d_obs=22.0, wheel_dy=-0.2, a_side=1, x_a=95.0, b=1, b_notice=25.0, b_x0=-26.0, b_dv=6.0,
                   b_t=0.5, b_react=1.2, b_max=6.0, other_sp=13.0, other_off=0.0, rear=0, rear_gap=25.0, rear_react=0.9,
                   wx="dry")
    ranges = dict(wheel=[1, 1, 1, 1, 0], v=(22.0, 28.0), headway=(1.2, 2.2), d_obs=(20.0, 45.0), wheel_dy=(-0.8, 0.8),
                  a_side=[1, -1], x_a=(70.0, 160.0), b=[1, 1, 1, 0], b_notice=(20.0, 60.0), b_x0=(-80.0, -12.0), b_dv=(3.0, 9.0), b_t=(0.3, 1.2),
                  b_react=(0.6, 1.6), b_max=(5.0, 7.0), other_sp=(11.0, 32.0), other_off=(-6.0, 6.0), rear=[0, 1],
                  rear_gap=(8.0, 45.0), rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("A wheel comes off the truck ahead into the ego lane; broken-down car A stands in an adjacent lane "
                   "further on, car B there has planned its stop behind A. Brake, or swerve into A's lane past the "
                   "wheel and then cut back before A or stop behind A (B's space) - decided once past the wheel.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        sp = self.road(p)["speed"] * self.site_speed
        y_e, s = W, p["a_side"]
        y_a, y_o = W + s * W, W - s * W
        Lt = CAR[self.TRUCK][0]
        x_truck = HALF_L + v * p["headway"] + Lt / 2
        truck = car("truck", self.TRUCK, 0, y_e, v, kind="scripted", path=const_path(x_truck, y_e, v))
        x_w = HALF_L + p["d_obs"] + 0.4
        if p["wheel"]:
            wheel = Body("wheel", x_w + 12.0, y_e, length=0.8, width=0.8, kind="scripted", bp="static.prop.box01",
                         path=crash_path(x_w + 12.0, y_e, v, v - 3.0, 14.0 * grip, y_e + p["wheel_dy"], 0.0, 0.8))
        else:                                       # benign: it bounces off onto the far verge
            wheel = Body("wheel", x_w + 12.0, y_e, length=0.8, width=0.8, kind="scripted", bp="static.prop.box01",
                         path=crash_path(x_w + 12.0, y_e, v, v - 3.0, 14.0 * grip, y_o - s * (W + 1.5), 0.0, 0.8))
        La = CAR["vehicle.toyota.prius"][0]
        x_a = p["x_a"] * sp
        car_a = Body("car_A", x_a + La / 2, y_a, length=La, width=CAR["vehicle.toyota.prius"][1], kind="static",
                     bp="vehicle.toyota.prius")
        bv, Lb = v + p["b_dv"] * sp, CAR["vehicle.lincoln.mkz_2020"][0]
        room = (x_a - 2.0) - (p["b_x0"] + Lb / 2) - bv * p["b_t"]
        b_plan = bv ** 2 / (2.0 * max(room, 5.0))
        car_b = car("car_B", "vehicle.lincoln.mkz_2020", p["b_x0"] - bv * pT, y_a, bv,
                    law=law_planned_stop(p["b_t"], b_plan, p["b_notice"], p["b_react"], p["b_max"] * grip, half_w=1.6))
        xs = np.arange(-4, 5) * p["other_sp"] + p["other_off"]
        other = [car(f"other{i}", CARS[i % len(CARS)], x - v * pT, y_o, v, law=law_idm(v, T=0.9, react=0.8, b_max=6 * grip))
                 for i, x in enumerate(xs)]
        bodies = [truck, wheel, car_a] + ([car_b] if p["b"] else []) + other     # no B: stopping behind A is safe
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        bg = platoon("l1_", -W, v + 2.0, [-60.0, -20.0, 25.0, 70.0]) if self.n_left >= 1 else []
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def cell(self, r):
        return ("wheel: " if r["params"]["wheel"] else "benign: ") + super().cell(r)


# ---------------------------------------------------------------- RC07-S5
TAPER = 24.0


class RC07S5Opt(HBBase):
    """Ego in the left lane (y W) of the motorway's right pair; the right lane (y 2W), and beyond it the acceleration
    lane (y 3W) that a cone taper closes ramp_left m ahead of the ego at t0. The lane left of the ego (y 0) is a dense
    platoon at the ego's speed (no escape left; outside the corridor). Trigger: the lead is seen braking hard."""
    sid = "RC07-S5-opt"
    scene_name = "blowout_onramp"                      # R3 (PIPELINE_HANDOFF): the scene, apart from the RC number
    T, prefix_T = 12.0, 3.0                            # R7: 3 s pre-roll recorded before t0 (batches before vsearch: 1 s)
    trigger = "braking"
    modes = ("keep", "brake", "right>return", "right>stay", "right>away")
    SPEEDS = ("v", "ramp_v")
    LEAD, FAST, RAMP, PILE = "vehicle.toyota.prius", "vehicle.dodge.charger_2020", "vehicle.mini.cooper_s_2021", \
        "vehicle.nissan.patrol_2021"
    nominal = dict(blowout=1, v=27.0, headway=1.2, blow_dec=8.0, blow_dy=0.0, benign_dec=2.0, fast=1, fast_dv=5.0,
                   fast_gap=40.0, fast_notice=25.0, fast_react=1.2, fast_b=4.5, ramp_dx=-3.0, ramp_v=27.0, ramp_cut=1,
                   t_cut=2.0, ramp_acc=3.0, ramp_left=140.0, pile=0, pile_dx=50.0, rear=0, rear_gap=30.0,
                   rear_react=0.9, dense_sp=16.0, dense_off=0.0, wx="dry")
    ranges = dict(blowout=[1, 1, 1, 1, 1, 0], v=(24.0, 30.0), headway=(0.8, 1.6), blow_dec=(7.0, 9.0),
                  blow_dy=(-0.5, 0.5), benign_dec=(1.0, 2.5), fast=[1, 1, 1, 0], fast_dv=(3.0, 8.0),
                  fast_gap=(25.0, 60.0), fast_notice=(15.0, 40.0), fast_react=(0.8, 1.5), fast_b=(3.0, 6.0),
                  ramp_dx=(-15.0, 10.0), ramp_v=(25.0, 29.0), ramp_cut=[1, 1, 1, 0], t_cut=(1.5, 3.0),
                  ramp_acc=(2.5, 3.5), ramp_left=(80.0, 200.0), pile=[0, 0, 1], pile_dx=(30.0, 80.0), rear=[0, 1],
                  rear_gap=(15.0, 50.0), rear_react=(0.6, 1.2), dense_sp=(12.0, 20.0), dense_off=(-6.0, 6.0),
                  wx=["dry", "wet", "heavy"])
    hidden = ("blowout", "fast_notice", "fast_react", "fast_b", "ramp_cut", "t_cut", "ramp_acc", "rear_react")
    description = ("A tyre blowout: the car ahead skids to a stop in the ego lane. The right lane carries a fast car "
                   "coming up behind; an on-ramp car cuts into the slot between the ego and the fast car and floors "
                   "it once the fast car closes on it, straight at the ego. Brake, or swerve right past the stopped "
                   "car, then return left, stay or pull away - decided once past it.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        rd = self.road(p)
        grip, sp = rd["grip"], rd["speed"] * self.site_speed
        y_d, y_e, y_r, y_a = 0.0, W, 2 * W, 3 * W
        Ll, Lp = CAR[self.LEAD][0], CAR[self.PILE][0]
        x_l = HALF_L + v * p["headway"] + Ll / 2
        if p["blowout"]:                                # skids to a stop, ending off-centre and slightly yawed
            dec = p["blow_dec"] * grip
            path = crash_path(x_l, y_e, v, v, dec, y_e + p["blow_dy"], 0.15 * math.copysign(1.0, p["blow_dy"]), 1.5)
            x_rest = x_l + v * v / (2 * dec)
        else:                                           # keeps control: slows gently by 4 m/s, then back up
            path = speed_path(x_l - 5.0 * v, y_e, 0.0, v, [(0.0, 0.0, v), (2.5, -p["benign_dec"] * grip, v - 4.0),
                                                           (20.0, 1.0, v)])
            x_rest = None
        bodies = [car("lead", self.LEAD, 0, y_e, v, kind="scripted", path=path)]
        if p["pile"] and x_rest is not None:            # a car further ahead also brakes to a stop pile_dx beyond it
            x0 = x_l + Ll / 2 + v * 1.0 + Lp / 2
            x_stop = x_rest + Ll / 2 + p["pile_dx"] + Lp / 2
            bodies.append(car("pileup", self.PILE, 0, y_e - 0.3, v, kind="scripted",
                              path=const_path(x0, y_e - 0.3, v, t_brake=0.0, b=v * v / (2 * (x_stop - x0)))))
        if p["fast"]:
            vf = v + p["fast_dv"] * sp
            Lf = CAR[self.FAST][0]
            bodies.append(car("fast", self.FAST, -HALF_L - p["fast_gap"] - Lf / 2 - vf * pT, y_r, vf,
                              law=law_fast(vf, p["fast_notice"], p["fast_react"], p["fast_b"] * grip)))
        vr = p["ramp_v"]
        bodies.append(car("onramp", self.RAMP, p["ramp_dx"] - vr * pT, y_a, vr,
                          law=law_onramp(vr, y_r, p["ramp_cut"], p["t_cut"], p["ramp_acc"], 35.0 * sp, p["ramp_left"],
                                         lane_w=W)))
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        xs = np.arange(-4, 6) * p["dense_sp"] + p["dense_off"]
        bodies += [car(f"dense{i}", CARS[i % len(CARS)], x - v * pT, y_d, v, law=law_idm(v, T=0.9, react=0.8, b_max=6 * grip))
                   for i, x in enumerate(xs)]
        bg = platoon("l1_", -W, v + 2.0, [-60.0, -20.0, 25.0, 70.0]) if self.n_left >= 1 else []
        x_end, y_out, y_in = p["ramp_left"], 3.5 * W - 0.3, 2.5 * W + 0.3
        cones = [("static.prop.constructioncone", x_end - TAPER + 2.0 * i, y_out + (y_in - y_out) * min(2.0 * i / TAPER, 1.0))
                 for i in range(int(TAPER / 2) + 1)]
        cones += [("static.prop.constructioncone", x_end + 8.0 * i, y_in) for i in range(1, 6)]
        # corridor: the ego lane and the right lane, plus the acceleration lane up to the taper (closed along it)
        hi = lambda x: 2.5 * W + W * min(max((x_end - x) / TAPER, 0.0), 1.0)
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (W / 2, hi(x)), v_ref=v, props=cones)

    def cell(self, r):
        p = r["params"]
        pre = ("blowout" if p["blowout"] else "benign") + (", pile-up" if p["pile"] else "") + \
            ("" if p["fast"] else ", no fast car")
        return f"{pre}: " + super().cell(r)


# ---------------------------------------------------------------- RC07-S6
class RC07S6Opt(HBBase, RC03S8Opt):
    """RC03-S8-opt's scene (Town03, side street hidden by corner buildings) + a speeding car that may come up the left
    lane (then it replaces the left lane's follower)."""
    sid = "RC07-S6-opt"
    T = 11.0
    trigger = "conflict"
    modes = ("keep", "brake", "left>return", "left>stay")
    nominal = dict(RC03S8Opt.nominal, sp=1, sp_dv=9.0, sp_gap=40.0, sp_notice=20.0, sp_react=1.0, sp_b=5.0)
    ranges = dict(RC03S8Opt.ranges, sp=[0, 1, 1], sp_dv=(5.0, 12.0), sp_gap=(10.0, 90.0), sp_notice=(12.0, 30.0),
                  sp_react=(0.8, 1.6), sp_b=(3.0, 6.0))
    description = ("City arterial: a side-street car behind corner buildings stops, runs the sign or pulls out late; a "
                   "speeding car may come up the left lane. Brake, or change left past it, then return or stay - "
                   "decided once past it.")

    def build(self, p, W):
        sc = RC03S8Opt.build(self, p, W)
        if p["sp"]:
            pT, v = self.prefix_T, p["v"]
            x_e = sc["ego"]["x"] + v * pT
            sc["bodies"] = [b for b in sc["bodies"] if b.name != "left_follow"]
            vs = v + p["sp_dv"]
            sc["bodies"].append(speeding("speeding_car", 0.0, x_e - HALF_L - p["sp_gap"] - 2.5, vs, p,
                                         self.road(p)["grip"], pT))
        return sc

    def cell(self, r):
        return f"{r['params']['behav']}: " + HBBase.cell(self, r)


# ---------------------------------------------------------------- RC07-S3
class RC07S3Opt(HBBase):
    """Ego in the right lane (y W) of a motorway with a hard shoulder, the queue in the left lane (y 0) keeping
    0.25 m left of its centre; a motorcycle may filter along the line between them."""
    sid = "RC07-S3-opt"
    T = 12.0
    modes = ("keep", "brake", "shift_left>return", "shift_left>edge")
    SPEEDS = ("v",)
    MOTO = "vehicle.yamaha.yzf"
    nominal = dict(tread=1, v=20.0, d_tread=30.0, tread_y=1.1, tread_w=0.6, moto=1, moto_dv=10.0, moto_gap=52.0,
                   moto_dy=0.0, moto_notice=15.0, q_off=-0.25, rear=0, rear_gap=15.0, rear_react=0.9, wx="dry")
    ranges = dict(tread=[1, 1, 1, 1, 0], v=(16.0, 22.0), d_tread=(25.0, 45.0), tread_y=(0.9, 1.4), tread_w=(0.4, 0.7),
                  moto=[1, 1, 0], moto_dv=(6.0, 13.0), moto_gap=(25.0, 75.0), moto_dy=(-0.25, 0.1),
                  moto_notice=(10.0, 20.0), q_off=(-0.5, -0.2), rear=[0, 1], rear_gap=(6.0, 30.0),
                  rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("Congestion: a tyre tread in the right half of the ego lane, the left lane's queue keeping left, maybe "
                   "a motorcycle filtering along the line. Brake, or shift left in lane past it and then recentre or "
                   "come back right of centre - decided once past it, when the motorcycle is in view.")

    def deliver(self, env, name):
        return (name != "tread" or env.t >= -1e-6) and super().deliver(env, name)   # revealed by the lead at t0

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e = W
        x_t = HALF_L + p["d_tread"] + p["tread_w"] / 2
        y_t = y_e + p["tread_y"] if p["tread"] else y_e + W / 2 + 0.9     # benign: on the shoulder line
        tread = Body("tread", x_t, y_t, length=1.2, width=p["tread_w"], kind="static", bp="static.prop.box01")
        lead = car("lead", "vehicle.audi.a2", 0, y_e, v, kind="scripted", path=const_path(x_t + 8.0, y_e, v))
        bodies = [tread, lead]
        if p["moto"]:
            mv = v + p["moto_dv"]
            bodies.append(Body("motorcycle", -HALF_L - p["moto_gap"] - 1.1 - mv * pT, W / 2 + p["moto_dy"], v=mv,
                               length=2.2, width=0.8, bp=self.MOTO,
                               law=law_trigger(ego_ahead_in_lane(p["moto_notice"], half_w=1.3), -5.0 * grip,
                                               reaction=0.8)))
        bodies += [car(f"q{i}", CARS[i % 2], dx - v * pT, p["q_off"], v, law=law_idm(v, T=0.8, react=0.8, b_max=6 * grip))
                   for i, dx in enumerate(np.arange(-48.0, 61.0, 12.0))]
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e,
                    bounds=lambda x: (-W / 2, 1.5 * W + 1.0), v_ref=v, props=[])

    def cell(self, r):
        pre = ("moto, " if r["params"]["moto"] else "no moto, ") + ("tread" if r["params"]["tread"] else "benign")
        return f"{pre}: " + super().cell(r)


# ---------------------------------------------------------------- sites
# the motorway classes use RC03-S1-opt's middle-lane sites (a lane on each side): straight Town06 A, B; Town04 bends
# C1, C2b; Town05 ring D2 (training); Town05 ring D1 held out. S6 uses RC03-S8-opt's Town03 junctions (C held out); S3
# RC03-S6+7's right-lane-with-shoulder sites (D held out).
_MW = {k: SITES03[k] for k in ("A", "B", "C1", "C2b", "D2", "D1")}
# S5: straight Town06 motorways with two lanes right of the ego lane (the right lane and the one that becomes the
# acceleration lane) and one left of it (the dense lane); frame lane 0 = the dense lane, the ego in lane 1
_S5 = {"A": ("Town06", dict(start_xy=(21.7, 143.0), ref_lane=-5, ref_idx=1, s0=120.0), 1, "train", 1.0),   # road 35-38
       "B": ("Town06", dict(start_xy=(128.5, 244.6), ref_lane=-5, ref_idx=1, s0=90.0), 1, "train", 1.0),   # road 40
       "E1": ("Town06", dict(start_xy=(20.0, 45.4), ref_lane=-4, ref_idx=1, s0=150.0), 0, "heldout", 1.0)}  # roads 14-30
SITES = {"RC07-S1-opt": _MW, "RC07-S2-opt": _MW, "RC07-S7-opt": _MW, "RC07-S6-opt": SITES_X["RC03-S8-opt"],
         "RC07-S3-opt": SITES_X["RC03-S67-opt"], "RC07-S5-opt": _S5}
CLASSES = {"RC07-S1-opt": RC07S1Opt, "RC07-S2-opt": RC07S2Opt, "RC07-S7-opt": RC07S7Opt, "RC07-S6-opt": RC07S6Opt,
           "RC07-S3-opt": RC07S3Opt, "RC07-S5-opt": RC07S5Opt}


def site_spec(sid, key):
    town, site, n_left, split, speed = SITES[sid][key]
    sp = CLASSES[sid]()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.T = site.get("T", sp.T)
    sp.sid = sid if key == "A" else f"{sid}@{key}"
    return sp


SPECS07X = {sp.sid: sp for sid in SITES for sp in (site_spec(sid, k) for k in SITES[sid])}
