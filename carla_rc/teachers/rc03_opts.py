"""RC03-S2 / S4 / S5 / S6+7 / S8 with the options left open (as RC03-S1-opt): every variation is executed with every
maneuver family of the class (options.py) and its recovery profile is kept; the variations are designed so that each
family is open in some and closed in others, including scenes where nothing recovers (kept for the world-model data,
left out of the teacher's training). What the ego sees is a line-of-sight model (sight.py): a hidden actor is not in
the observation, the experts' view or the trigger.

    RC03-S2-opt   stalled car ahead hidden by the lead, which swerves late (left or right). Prior knowledge: a stalled
                  car is somewhere ahead. Straight sites (Town06): it shows only when the lead moves out; curved sites
                  (Town04 / Town05): it can come into view past the lead's side earlier. Families keep / brake / left /
                  right; trigger: the stalled car seen in the ego lane needing > 3 m/s^2 of braking.
    RC03-S67-opt  debris (a crate off the pickup ahead) in the ego lane of a 2-lane motorway with a hard shoulder (S6 and
                  S7 together): brake, change into the open left lane, or a partial shift onto the shoulder and back;
                  the shoulder may be blocked by a broken-down car, the left lane may be closed, a follower may make
                  braking fail.
    RC03-S8-opt   city arterial, a side street from the right with a stop sign, corner buildings hiding it: the side-
                  street car stops at its line (benign), runs the sign, or stops and then pulls out late. The ego has to
                  read the car's speed from when it comes into view; trigger: a predicted conflict (options.conflict_seen,
                  observed speed and acceleration). Families keep / brake / left.
    RC03-S4-opt   rural two-way road (Town07): a pedestrian steps out from the right verge, possibly from behind a parked
                  van, and stops at the kerb (benign), freezes in the lane, or keeps crossing; an oncoming car may close
                  the opposing lane. Families keep / brake / swerve_left (through the opposing lane and back).
    RC03-S3-opt   a traffic jam hidden beyond a crest (Town04); queue lengths per lane unknown (see the class).
    RC03-S5-opt   night, a wrong-way driver in the ego's (fast) lane; the truck in the right lane is ahead of, beside or
                  behind the ego. Braking cannot recover (the closing speed stays); merge_front (speed up, change in front
                  of the truck) or merge_back (brake, change behind it), depending on where the truck and its follower
                  are. Families keep / brake / merge_front / merge_back.
"""
import math

import numpy as np

import carla

from laws import HALF_L, Body, crash_path, const_path, lane_change_path, law_idm
from options import MODES, OBS_NAMES
from rc03_opt import GAP_CARS, SITES as SITES03, RC03S1Opt
from scenarios import CAR, car, platoon
from sight import SightMixin

LEADS = ["vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.mercedes.sprinter", "vehicle.volkswagen.t2_2021",
         "vehicle.carlamotors.european_hgv", "vehicle.ford.ambulance"]
STALLED = ["vehicle.volkswagen.t2_2021", "vehicle.nissan.patrol_2021", "vehicle.toyota.prius", "vehicle.audi.a2"]
WALKERS = ["walker.pedestrian.0001", "walker.pedestrian.0005", "walker.pedestrian.0012", "walker.pedestrian.0026"]


def speed_path(x0, y0, psi, v0, phases, t0=-5.0, t1=20.0, dt=0.01):
    """A body moving along a fixed heading psi with a piecewise speed profile from (x0, y0) at t = t0:
    phases = [(t_end, accel, v_target)]: until t_end accelerate at `accel` towards v_target (never below 0)."""
    ts = np.arange(t0, t1 + dt, dt)
    v, d, V, D = v0, 0.0, [], []
    for t in ts:
        a, vt = next(((a, vt) for te, a, vt in phases if t < te), (0.0, v))
        if (a > 0 and v < vt) or (a < 0 and v > vt):
            v = min(v + a * dt, vt) if a > 0 else max(v + a * dt, vt)
        v = max(v, 0.0)
        V.append(v), D.append(d)
        d += v * dt
    V, D = np.array(V), np.array(D)
    c, s = math.cos(psi), math.sin(psi)

    def f(t):
        dd, vv = float(np.interp(t, ts, D)), float(np.interp(t, ts, V))
        return x0 + c * dd, y0 + s * dd, psi, c * vv, s * vv
    return f


def stop_at(d_stop, v0, a_dec=None):
    """Deceleration that stops a body moving at v0 within d_stop."""
    return v0 * v0 / (2 * max(d_stop, 0.5)) if a_dec is None else a_dec


class OptBase(SightMixin, RC03S1Opt):
    """Shared: sampling with overrides and road-condition speed scaling, rain, friction, role-free observation,
    envelope; deliver = line of sight. Subclasses: sites, ranges, build, modes, trigger."""
    SPEEDS = ("v",)
    modes = MODES
    governor = False                 # RC03-S1's envelope governor is not part of these classes' collections
    trigger = "intrusion"
    obs_names = OBS_NAMES

    def sample(self, rng):
        from envelope import ROAD
        p = {}
        for k, r in {**self.ranges, **self.overrides}.items():
            p[k] = r[int(rng.integers(len(r)))] if isinstance(r, list) else float(rng.uniform(*r))
        for k in self.SPEEDS:
            p[k] *= ROAD[p.get("wx", "dry")]["speed"] * self.site_speed
        return p

    def card(self):
        d = super().card()
        d.update(trigger=self.trigger, sight="line of sight (sight.py)")
        return d


def gap_pair(prefix, lane_y, v, center, gap, lead_bp, fol_bp, pT, react=0.8, T_f=1.2, grip=1.0):
    """Two cars in an adjacent lane leaving a gap of `gap` m centred `center` m from the ego (at t0)."""
    lead_dx = center + gap / 2 + CAR[lead_bp][0] / 2
    fol_dx = center - gap / 2 - CAR[fol_bp][0] / 2
    return [car(f"{prefix}_lead", lead_bp, lead_dx - v * pT, lane_y, v, law=law_idm(v, T=1.5, react=0.8, b_max=6 * grip)),
            car(f"{prefix}_follow", fol_bp, fol_dx - v * pT, lane_y, v,
                law=law_idm(v, T=T_f, react=react, b_max=6 * grip))]


def rear_car(p, y, grip, pT):
    Lr = CAR["vehicle.audi.a2"][0]
    return car("rear", "vehicle.audi.a2", -HALF_L - p["rear_gap"] - Lr / 2 - p["v"] * pT, y, p["v"],
               law=law_idm(p["v"], T=max((p["rear_gap"] - 2.0) / p["v"], 0.15), react=p["rear_react"], b_max=7 * grip))


# ---------------------------------------------------------------- RC03-S2
class RC03S2Opt(OptBase):
    sid = "RC03-S2-opt"
    T = 10.0
    SPEEDS = ("v", "v_adj")
    nominal = dict(v=27.0, headway=1.4, lead_bp="vehicle.mercedes.sprinter", stalled_bp="vehicle.volkswagen.t2_2021",
                   stalled_dy=0.0, crash=1, cut_side=-1, t_cut=2.0, cut_ttc=1.5, T_cut=1.3, v_adj=26.0,
                   lead_lane_gap=25.0, other_center=-2.0, other_gap=35.0, rear=0, rear_gap=25.0, rear_react=0.9,
                   wx="dry")
    ranges = dict(v=(24.0, 30.0), headway=(0.8, 2.2), lead_bp=LEADS, stalled_bp=STALLED, stalled_dy=(-0.5, 0.5),
                  crash=[1, 1, 1, 1, 1, 0], cut_side=[-1, 1], t_cut=(0.5, 3.5), cut_ttc=(0.7, 2.6), T_cut=(1.0, 1.8),
                  v_adj=(22.0, 28.0), lead_lane_gap=(4.0, 45.0), other_center=(-15.0, 10.0), other_gap=(8.0, 50.0),
                  rear=[0, 1], rear_gap=(10.0, 60.0), rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("A stalled car ahead, hidden by the lead, which swerves late into an adjacent lane; on a bend it may "
                   "come into view earlier. Keep, brake, or change left / right into the adjacent traffic.")

    def build(self, p, W):
        pT, v, s = self.prefix_T, p["v"], p["cut_side"]
        rd = self.road(p)
        grip = rd["grip"]
        y_e = W
        Ll, Ls = CAR[p["lead_bp"]][0], CAR[p["stalled_bp"]][0]
        x_l = HALF_L + v * p["headway"] + Ll / 2                          # lead centre at t0
        x_st = x_l + v * p["t_cut"] + Ll / 2 + v * p["cut_ttc"] + Ls / 2   # stalled centre
        va = p["v_adj"]
        bodies = []
        if p["crash"]:
            y_st, y_lead_to = y_e + p["stalled_dy"], y_e + s * W
            lead = car("lead", p["lead_bp"], 0, y_e, v, kind="scripted",
                       path=lane_change_path(x_l, y_e, v, y_lead_to, p["t_cut"], p["T_cut"]))
        else:                                       # benign: the stalled car stands in the other lane, the lead stays
            y_st = y_e - s * W
            lead = car("lead", p["lead_bp"], 0, y_e, v, kind="scripted", path=const_path(x_l, y_e, v))
        stalled = Body("stalled", x_st, y_st, length=Ls, width=CAR[p["stalled_bp"]][1], kind="static",
                       bp=p["stalled_bp"])
        bodies += [lead, stalled]
        # the lane the lead swerves into: a follower `lead_lane_gap` m behind the lead's rear when it starts moving over
        xl_cut = x_l + v * p["t_cut"]
        fb = "vehicle.lincoln.mkz_2020"
        xf = xl_cut - Ll / 2 - p["lead_lane_gap"] - CAR[fb][0] / 2 - va * p["t_cut"]
        bodies.append(car("cut_follow", fb, xf - va * pT, y_e + s * W, va, law=law_idm(va, T=1.2, react=0.8,
                                                                                         b_max=6 * grip)))
        # the other adjacent lane: a gap pair (the escape the lead did not take)
        bodies += gap_pair("other", y_e - s * W, va, p["other_center"], p["other_gap"], "vehicle.ford.mustang",
                           "vehicle.toyota.prius", pT, grip=grip)
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        bg = []
        if self.n_left >= 1:
            bg += platoon("l1_", -W, va + 2.0, [-60.0, -20.0, 25.0, 70.0])
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        if not r["params"].get("crash", 1):
            return "benign"
        return f"brake {'ok' if ok('brake') else 'X'} / lane change {'ok' if ok('left') or ok('right') else 'X'}"


# ---------------------------------------------------------------- RC03-S6+7
class RC03S67Opt(OptBase):
    """Town04 road 49: lane 5 = y 0 (left lane), lane 6 = y W (ego), then the real 3.5 m hard shoulder."""
    sid = "RC03-S67-opt"
    T = 9.0
    modes = ("keep", "brake", "left", "shoulder")
    SPEEDS = ("v", "v_left")
    SHOULDER = 3.5
    PICKUP = "vehicle.nissan.patrol_2021"
    nominal = dict(v=28.0, headway=1.2, crate=1, debris_y=-0.3, debris_w=1.2, a_debris=14.0, left_center=-3.0,
                   left_gap=30.0, v_left=28.0, shoulder_blocked=0, block_dx=0.0, rear=0, rear_gap=25.0, rear_react=0.9,
                   wx="dry")
    ranges = dict(v=(25.0, 31.0), headway=(0.8, 3.5), crate=[1, 1, 1, 1, 1, 0], debris_y=(-1.0, 1.0),
                  debris_w=(0.8, 2.0), a_debris=(10.0, 18.0), left_center=(-15.0, 10.0), left_gap=(6.0, 50.0),
                  v_left=(24.0, 32.0), shoulder_blocked=[0, 0, 1], block_dx=(-10.0, 25.0), rear=[0, 1],
                  rear_gap=(8.0, 50.0), rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("A crate falls off the pickup ahead into the ego lane (2-lane motorway, hard shoulder): brake, change "
                   "into the left lane, or shift onto the shoulder past it and back. The shoulder may be blocked, the "
                   "left lane closed, a follower close behind.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e = W
        Lp = CAR[self.PICKUP][0]
        x_p = HALF_L + v * p["headway"] + Lp / 2
        pickup = car("pickup", self.PICKUP, 0, y_e, v, kind="scripted", path=const_path(x_p, y_e, v))
        y_rest = y_e + p["debris_y"] if p["crate"] else y_e + W / 2 + 0.6 + p["debris_w"] / 2   # benign: on the line
        a_sl = p["a_debris"] * grip
        crate = Body("crate", x_p - Lp / 2 - 0.8, y_e, length=p["debris_w"], width=p["debris_w"], kind="scripted",
                     bp="static.prop.box03" if p["debris_w"] > 1.3 else "static.prop.box01",
                     path=crash_path(x_p - Lp / 2 - 0.8, y_e, v, v, a_sl, y_rest, 0.9, 0.8))
        bodies = [pickup, crate]
        bodies += gap_pair("left", 0.0, p["v_left"], p["left_center"], p["left_gap"], "vehicle.lincoln.mkz_2020",
                           "vehicle.audi.a2", pT, grip=grip)
        if p["shoulder_blocked"]:                      # a broken-down car on the shoulder near where the crate rests
            x_rest = x_p - Lp / 2 - 0.8 + v * v / (2 * a_sl)
            bodies.append(Body("shoulder_car", x_rest + p["block_dx"], 1.5 * W + self.SHOULDER / 2, length=4.6,
                               width=2.0, kind="static", bp="vehicle.ford.mustang"))
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        bg = platoon("l4_", -W, p["v_left"] + 1.0, [-60.0, -20.0, 25.0, 70.0])
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (-W / 2, 1.5 * W + self.SHOULDER), v_ref=v, props=[])

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        if not r["params"].get("crate", 1):
            return "benign"
        return " / ".join(f"{m} {'ok' if ok(m) else 'X'}" for m in ("brake", "left", "shoulder"))


# ---------------------------------------------------------------- RC03-S8
class RC03S8Opt(OptBase):
    """Town03 (loaded as Town03_Opt: plain Town03 segfaults this server build on load) road 3 northbound (lanes -1, -2 = y 0, W; ego in the right / kerb lane y = W), side street road 62 from
    the right at junction 861 with a stop sign; buildings on the near corner hide it (an occluder set back `setback` m
    from the kerb)."""
    sid = "RC03-S8-opt"
    T = 9.0
    trigger = "conflict"
    modes = ("keep", "brake", "left")
    SPEEDS = ("v",)
    SIDE_WORLD = (-98.2, 0.4)
    KERB = 2.5 + 0.63
    CROSS = ["vehicle.lincoln.mkz_2020", "vehicle.toyota.prius", "vehicle.nissan.patrol_2021", "vehicle.audi.a2"]
    x_side = 123.3
    T_IN = 2.5                       # [s] latest the side car's front enters the ego lane after t0 (see build)
    nominal = dict(v=15.0, lag=1.0, cross_bp="vehicle.lincoln.mkz_2020", behav="run", v_app=7.0, d_app=20.0,
                   t_wait=1.5, a_go=2.5, go_far=1, protrude=2.0, setback=3.0, left_lead_dx=40.0, left_fol_dx=-15.0,
                   rear=0, rear_gap=15.0, rear_react=0.9, wx="dry")
    ranges = dict(v=(12.5, 17.5), lag=(-1.0, 2.5), cross_bp=CROSS, behav=["stop", "run", "run", "stop_go", "stop_go"],
                  v_app=(3.0, 10.0), d_app=(5.0, 30.0), t_wait=(0.5, 3.5), a_go=(1.5, 3.5), go_far=[0, 1],
                  protrude=(1.2, 2.8), setback=(1.0, 8.0), left_lead_dx=(10.0, 70.0), left_fol_dx=(-35.0, -3.0),
                  rear=[0, 1], rear_gap=(6.0, 30.0), rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("City arterial, a side-street car behind corner buildings: it stops at its line, runs the stop sign, "
                   "or stops and then pulls out late. Keep, brake, or change to the left lane.")

    def on_frame(self, frame):
        sx, sy = self.site.get("side_xy", self.SIDE_WORLD)       # the side street's entry into the junction (world)
        self.x_side = frame.from_world(carla.Location(x=sx, y=sy), 90.0)[0]

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e = W
        L, Wc = CAR[p["cross_bp"]]
        y_line = 1.5 * W + self.KERB + L / 2 + 0.8                 # the side car's centre at its stop line
        y_start = y_line + p["d_app"]
        va = p["v_app"]
        dec = va * va / (2 * p["d_app"])                             # stops on its line
        t_line = 2 * p["d_app"] / va                                 # when it is there (stopping)
        y_end = (1.5 * W + L / 2 - p["protrude"]) if not p["go_far"] else -3.0 * W
        if p["behav"] == "stop":
            phases = [(0.0, 0.0, va), (t_line, -dec, 0.0)]
        elif p["behav"] == "run":                                    # slows a little, then crosses
            phases = [(0.0, 0.0, va), (t_line * 0.6, -0.8, va * 0.7), (99.0, 2.0, max(va, 6.0))]
        else:                                                        # stop, wait, pull out
            phases = [(0.0, 0.0, va), (t_line, -dec, 0.0), (t_line + p["t_wait"], 0.0, 0.0),
                      (99.0, p["a_go"] * grip, 7.0)]
        free = speed_path(self.x_side, y_start, -math.pi / 2, va, phases, t0=0.0, t1=40.0)

        def raw(t):                          # approaching at v_app before its clock starts; holds at y_end once there
            if t < 0:
                return self.x_side, y_start - va * t, -math.pi / 2, 0.0, -va
            x, y, psi, vx, vy = free(t)
            return (x, y_end, psi, 0.0, 0.0) if y <= y_end else (x, y, psi, vx, vy)
        # the side car's clock is shifted so that its front enters the ego lane at most T_IN s after t0: a long approach
        # or wait plays out before t0, behind the corner buildings (the site has ~120 m of road before the junction)
        ts = np.arange(0.0, 40.0, 0.05)
        t_raw = next((t for t in ts if raw(t)[1] - L / 2 <= 1.5 * W), t_line)
        shift = max(t_raw - self.T_IN, 0.0)
        path = lambda t: raw(t + shift)
        cross = Body("crossing_car", *path(-pT)[:2], psi=-math.pi / 2, length=L, width=Wc, kind="scripted",
                     bp=p["cross_bp"], path=path)
        # the ego is placed so that it would reach the side car's path `lag` s after the car's front enters the ego
        # lane (lag < 0: the car enters behind the ego; a car that stops at its line: measured from it stopping there)
        t_in = t_raw - shift
        D0 = max(v * (t_in + p["lag"]), 15.0)
        x_e = self.x_side - Wc / 2 - D0 - HALF_L                     # ego centre at t0
        bodies = [cross]
        vl = v
        Ll = CAR["vehicle.audi.a2"][0]
        bodies.append(car("left_lead", "vehicle.audi.a2", x_e + p["left_lead_dx"] - vl * pT, 0.0, vl,
                          law=law_idm(vl, T=1.5, react=0.8, b_max=6 * grip)))
        bodies.append(car("left_follow", "vehicle.toyota.prius", x_e + p["left_fol_dx"] - Ll / 2 - vl * pT, 0.0, vl,
                          law=law_idm(vl, T=1.2, react=0.8, b_max=6 * grip)))
        if p["rear"]:
            Lr = CAR["vehicle.audi.a2"][0]
            bodies.append(car("rear", "vehicle.audi.a2", x_e - HALF_L - p["rear_gap"] - Lr / 2 - v * pT, y_e, v,
                              law=law_idm(v, T=max((p["rear_gap"] - 2.0) / v, 0.15), react=p["rear_react"],
                                          b_max=7 * grip)))
        xs = self.x_side - 5.0                                       # the near corner: a building block
        yb = 1.5 * W + self.KERB + p["setback"]
        occ = [[(xs - 60.0, yb), (xs, yb), (xs, yb + 30.0), (xs - 60.0, yb + 30.0)]]
        return dict(ego=dict(x=x_e - v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e,
                    bounds=lambda x: (-W / 2, 1.5 * W), v_ref=v, props=[], occluders=occ, D0=D0)

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        b = r["params"]["behav"]
        return f"{b}: " + " / ".join(f"{m} {'ok' if ok(m) else 'X'}" for m in ("keep", "brake", "left"))


# ---------------------------------------------------------------- RC03-S4
class RC03S4Opt(OptBase):
    """Town07 road 11 northbound (rural 1+1): lane -1 = ego at y W, lane 1 = opposing at y 0. A pedestrian steps out of
    the right verge, possibly from behind a parked van."""
    sid = "RC03-S4-opt"
    T = 9.0
    trigger = "conflict"
    modes = ("keep", "brake", "swerve_left")
    SPEEDS = ("v", "onc_v")
    nominal = dict(v=20.0, d0=50.0, walker_bp=WALKERS[0], behav="freeze", v_ped=1.4, t_step=0.5, y_stop=0.0,
                   parked=1, onc=1, onc_dist=150.0, onc_v=20.0, rear=0, rear_gap=15.0, rear_react=0.9, wx="dry")
    ranges = dict(v=(16.0, 23.0), d0=(30.0, 75.0), walker_bp=WALKERS, behav=["kerb", "freeze", "freeze", "cross", "cross"],
                  v_ped=(1.0, 2.6), t_step=(0.0, 2.0), y_stop=(-0.8, 0.8), parked=[0, 1], onc=[1, 1, 0],
                  onc_dist=(60.0, 260.0), onc_v=(15.0, 23.0), rear=[0, 1], rear_gap=(8.0, 40.0),
                  rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    description = ("Rural two-way road: a pedestrian steps out of the right verge (maybe from behind a parked van) and "
                   "stops at the kerb, freezes in the lane, or keeps crossing. Keep, brake, or swerve through the "
                   "opposing lane before the oncoming car and back.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e = W
        x_ped = HALF_L + p["d0"] + v * p["t_step"]                   # the ego's front reaches it ~d0 after it steps out
        y0 = 1.5 * W + 1.2
        y1 = {"kerb": 1.5 * W + 0.4, "freeze": y_e + p["y_stop"], "cross": -0.5 * W - 1.5}[p["behav"]]
        d = y0 - y1
        T_walk = d / p["v_ped"]

        def path(t):
            u = min(max(t - p["t_step"], 0.0), T_walk)
            moving = 0.0 < t - p["t_step"] < T_walk
            return x_ped, y0 - p["v_ped"] * u, -math.pi / 2, 0.0, -p["v_ped"] if moving else 0.0
        ped = Body("pedestrian", x_ped, y0, psi=-math.pi / 2, length=0.5, width=0.5, kind="scripted",
                   bp=p["walker_bp"], path=path)
        bodies = [ped]
        if p["parked"]:                                              # hides the pedestrian until they step out
            bodies.append(Body("parked_van", x_ped - 4.5, 1.5 * W + 1.1, length=CAR["vehicle.volkswagen.t2_2021"][0],
                               width=1.8, kind="static", bp="vehicle.volkswagen.t2_2021"))
        if p["onc"]:
            Lo = CAR["vehicle.ford.mustang"][0]
            bodies.append(car("oncoming", "vehicle.ford.mustang", 0, 0.0, p["onc_v"], kind="scripted",
                              path=const_path(HALF_L + p["onc_dist"] + Lo / 2, 0.0, p["onc_v"], psi=math.pi)))
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[])

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        return f"{r['params']['behav']}: " + " / ".join(f"{m} {'ok' if ok(m) else 'X'}"
                                                        for m in ("keep", "brake", "swerve_left"))


# ---------------------------------------------------------------- RC03-S5
class RC03S5Opt(OptBase):
    """Town04 road 49 northbound at night: lane 3 (next to the median) = y 0 = ego, lane 4 = y W (truck and its
    follower); lanes 5, 6 background. A wrong-way driver comes down the ego's lane; its headlights are seen from far."""
    sid = "RC03-S5-opt"
    T = 16.0                                  # cap; the episode ends once the wrong-way car is past the ego (done)
    modes = ("keep", "brake", "merge_front", "merge_back")
    SPEEDS = ("v", "v_truck", "v_ww")
    TRUCK = "vehicle.carlamotors.european_hgv"
    sight_range = {"dry": 400.0, "wet": 300.0, "heavy": 200.0}     # oncoming headlights at night
    weather = dict(cloudiness=10.0, precipitation=0.0, sun_altitude_angle=-40.0, sun_azimuth_angle=150.0,
                   fog_density=0.0)
    nominal = dict(v=29.0, d_ww=220.0, v_ww=28.0, truck_dx=0.0, v_truck=24.0, fol_gap=20.0, lead_gap=40.0, wx="dry")
    ranges = dict(v=(26.0, 32.0), d_ww=(160.0, 380.0), v_ww=(20.0, 34.0), truck_dx=(-30.0, 30.0),
                  v_truck=(20.0, 26.0), fol_gap=(3.0, 40.0), lead_gap=(8.0, 70.0), wx=["dry", "wet", "heavy"])
    description = ("Night: a wrong-way driver in the ego's lane; braking cannot recover. Speed up and change in front of "
                   "the truck in the right lane, or brake and change behind it - whichever its position and follower "
                   "leave open.")

    def done(self, env):
        """Scored once the wrong-way car is behind the ego (so stopping in its path is judged when it arrives)."""
        return env.bodies["wrong_way"].position[0] < env.ego.position[0] - 10.0

    def weather_for(self, p):
        from rc03_opt import WEATHER
        w = dict(WEATHER[p.get("wx", "dry")])
        w.update(sun_altitude_angle=-40.0)
        return w

    def build(self, p, W):
        pT, v, vt = self.prefix_T, p["v"], p["v_truck"]
        grip = self.road(p)["grip"]
        Lt = CAR[self.TRUCK][0]
        Lw = CAR["vehicle.dodge.charger_2020"][0]
        ww = car("wrong_way", "vehicle.dodge.charger_2020", 0, 0.0, p["v_ww"], kind="scripted",
                 path=const_path(HALF_L + p["d_ww"] + Lw / 2, 0.0, p["v_ww"], psi=math.pi))
        x_t = p["truck_dx"]
        truck = car("truck", self.TRUCK, x_t - vt * pT, W, vt, law=law_idm(vt, T=1.5, react=1.0, b_max=5 * grip))
        Lf = CAR["vehicle.audi.a2"][0]
        fol = car("truck_follower", "vehicle.audi.a2", x_t - Lt / 2 - p["fol_gap"] - Lf / 2 - vt * pT, W, vt,
                  law=law_idm(vt, T=max(p["fol_gap"] / vt, 0.3), react=0.8, b_max=6 * grip))
        Ll = CAR["vehicle.toyota.prius"][0]
        lead = car("truck_lead", "vehicle.toyota.prius", x_t + Lt / 2 + p["lead_gap"] + Ll / 2 - vt * pT, W, vt,
                   law=law_idm(vt, T=1.5, react=0.8, b_max=6 * grip))
        n_right = self.site.get("n_right", 3)                         # lanes right of the ego's at this site
        bg = (platoon("l5_", 2 * W, vt - 1.0, [-70.0, -35.0, 0.0, 35.0, 70.0, 105.0]) if n_right >= 2 else [])
        bg += platoon("l6_", 3 * W, vt - 3.0, [-55.0, -15.0, 25.0, 65.0]) if n_right >= 3 else []
        return dict(ego=dict(x=-v * pT, y=0.0, v=v), bodies=[ww, truck, fol, lead] + bg, home_y=0.0,
                    bounds=lambda x: (-W / 2, 1.5 * W), v_ref=v, props=[])

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        return " / ".join(f"{m} {'ok' if ok(m) else 'X'}" for m in ("brake", "merge_front", "merge_back"))


# ---------------------------------------------------------------- RC03-S3
class RC03S3Opt(OptBase):
    """Town04 road 39 (lanes 3..6 one way): a plateau, then the road drops at ~5 % past its edge (frame x ~ X_CREST;
    other sites give theirs as x_crest).
    Corridor lanes 4, 5, 6 (ego in lane 5); lane 3 background. Prior knowledge: a traffic jam ahead; the queue length
    of each lane is unknown. Each lane's queue (stopped cars) ends q m past the crest; the crest hides it until the
    sight line over the real road profile clears it (sight.py). case picks the queue pattern:
        all_short  every tail far past the crest (brake and lane change feasible)
        mid_long   the ego lane's tail just past the crest, the side lanes' far (must change lane)
        side_long  the side lanes' tails just past the crest, the ego lane's far (must brake)
        all_long   every tail just past the crest (nothing recovers: world-model data only)
    Families keep / brake / left / right (a lane change, then stop behind the queue in that lane)."""
    sid = "RC03-S3-opt"
    T = 18.0                                  # cap; the episode ends once the ego has stopped (done)
    SPEEDS = ("v",)
    X_CREST = 220.0
    QBP = ["vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.mercedes.sprinter", "vehicle.lincoln.mkz_2020",
           "vehicle.carlamotors.european_hgv"]
    nominal = dict(v=33.0, x0=0.0, case="mid_long", q_short=(110.0, 110.0, 110.0), q_long=(20.0, 20.0, 20.0), rear=0,
                   rear_gap=30.0, rear_react=0.9, wx="dry")
    ranges = dict(v=(28.0, 36.0), x0=(-40.0, 30.0), case=["all_short", "mid_long", "side_long", "all_long"],
                  qs_l=(80.0, 170.0), qs_m=(80.0, 170.0), qs_r=(80.0, 170.0), ql_l=(0.0, 45.0), ql_m=(0.0, 45.0),
                  ql_r=(0.0, 45.0), rear=[0, 1], rear_gap=(15.0, 60.0), rear_react=(0.4, 1.2),
                  wx=["dry", "wet", "heavy"])
    description = ("A traffic jam hidden beyond a crest (Town04, ~5 % downhill past it); each lane's queue length is "
                   "unknown. Brake in lane, or change into a lane whose queue ends further on, then stop.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e = W
        long_lanes = dict(all_short=(), mid_long=("m",), side_long=("l", "r"), all_long=("l", "m", "r"))[p["case"]]
        bodies = []
        for lane, y in (("l", 0.0), ("m", W), ("r", 2 * W)):
            q = p[f"ql_{lane}"] if lane in long_lanes else p[f"qs_{lane}"]
            x = self.site.get("x_crest", self.X_CREST) + q
            for j in range(5):                                         # the stopped queue, tail first
                bp = self.QBP[(j + "lmr".index(lane)) % len(self.QBP)]
                L, Wd = CAR[bp]
                bodies.append(Body(f"q{lane}{j}", x + L / 2, y, length=L, width=Wd, kind="static", bp=bp))
                x += L + 2.5
        if p["rear"]:
            bodies.append(rear_car(p, y_e, grip, pT))
        return dict(ego=dict(x=p["x0"] - v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=v, props=[])

    def done(self, env):
        """Scored once the ego has been stopped for 1 s (behind a queue) - the queue is always reached."""
        m = env.mem.setdefault("stopped", [0])
        m[0] = m[0] + 1 if env.ego.speed < 0.3 else 0
        return m[0] >= 10

    def cell(self, r):
        ok = lambda m: r["profile"][m]["success"]
        return f"{r['params']['case']}: brake {'ok' if ok('brake') else 'X'} / lane change " \
               f"{'ok' if ok('left') or ok('right') else 'X'}"


# ---------------------------------------------------------------- sites
# (town, LaneFrame site, lanes left of the frame's lane 0 with background traffic, split, speed level)
SITES = {
    "RC03-S2-opt": SITES03,               # straight Town06 (A, B), bends Town04 / Town05 (C1, C2, D2); held out D1, Town13
    # B..: real roads found by map scans (sitescan.py / s8probe.py / crest_find.py, radius and height checked along the
    # frame); the last of each class is held out (never trained on). Speed level: the class's speeds scaled for the
    # road (motorway bends of R ~100 m at 0.75 as RC03-S1's D sites, Town07's hill roads at 0.55)
    "RC03-S67-opt": {
        "A": ("Town04", dict(start_xy=(-9.9, -202.8), ref_lane=5, ref_idx=0, s0=30.0), 0, "train", 1.0),
        "B": ("Town04", dict(start_xy=(-515.2, 241.0), ref_lane=-3, ref_idx=0, s0=0.0, curve=True), 0, "train", 0.9),  # ring bend R 192
        "C": ("Town05", dict(start_xy=(-132.9, -207.6), ref_lane=-2, ref_idx=0, s0=0.0, curve=True), 0, "train", 0.75),  # elevated, R 104
        "D": ("Town05", dict(start_xy=(190.0, 13.6), ref_lane=3, ref_idx=0, s0=0.0, curve=True), 0, "heldout", 0.75)},  # straight into R 92
    "RC03-S8-opt": {
        "A": ("Town03_Opt", dict(start_xy=(-88.7, -122.9), ref_lane=-1, ref_idx=0, s0=0.0), 0, "train", 1.0),
        "B": ("Town03_Opt", dict(start_xy=(224.3, 156.7), ref_lane=-1, ref_idx=0, s0=118.0, side_xy=(-5.7, 180.9)), 0,
              "train", 1.0),                                                                  # junction 82, STOP, downhill approach
        "C": ("Town03_Opt", dict(start_xy=(227.6, -173.7), ref_lane=-1, ref_idx=0, s0=48.0, side_xy=(220.9, 6.3)), 0,
              "heldout", 1.0)},                                                               # junction 1135, STOP
    "RC03-S4-opt": {
        "A": ("Town07", dict(start_xy=(-3.6, -228.6), ref_lane=-1, ref_idx=1, s0=40.0), 0, "train", 1.0),
        "B": ("Town01", dict(start_xy=(396.3, 278.5), ref_lane=-1, ref_idx=1, s0=0.0), 0, "train", 1.0),   # straight, 4 m lanes
        "C": ("Town07", dict(start_xy=(59.0, -61.1), ref_lane=-1, ref_idx=1, s0=0.0, curve=True), 0, "train", 0.55),  # hill, R 30-80
        "D": ("Town07", dict(start_xy=(41.5, -174.5), ref_lane=1, ref_idx=1, s0=0.0, curve=True), 0, "heldout", 0.55)},
    "RC03-S5-opt": {
        "A": ("Town04", dict(start_xy=(-9.9, -202.8), ref_lane=3, ref_idx=0, s0=30.0), 0, "train", 1.0),
        "B": ("Town06", dict(start_xy=(128.5, 248.1), ref_lane=-3, ref_idx=0, s0=40.0, n_right=4), 0, "train", 1.0),
        "C": ("Town04", dict(start_xy=(-504.8, 240.9), ref_lane=-1, ref_idx=0, s0=0.0, curve=True), 0, "train", 0.9),  # ring bend R 185
        "D": ("Town05", dict(start_xy=(207.5, 13.8), ref_lane=-1, ref_idx=0, s0=110.0, curve=True, n_right=2), 0,
              "train", 0.75),                                                                  # R 103
        "E": ("Town05", dict(start_xy=(-143.1, -204.1), ref_lane=-1, ref_idx=0, s0=0.0, curve=True, n_right=2), 0,
              "heldout", 0.75)},                                                               # elevated, R 101
    "RC03-S3-opt": {
        "A": ("Town04", dict(start_xy=(-153.8, 33.7), ref_lane=5, ref_idx=1, s0=0.0), 1, "train", 1.0),
        "B": ("Town04", dict(start_xy=(168.2, 11.2), ref_lane=-3, ref_idx=1, s0=0.0, curve=True, x_crest=195.0), 1,
              "train", 1.0),                                                                   # plateau, then -5 %
        "C": ("Town05", dict(start_xy=(159.9, -185.9), ref_lane=-2, ref_idx=1, s0=0.0, curve=True, x_crest=270.0, T=26.0), 0,
              "train", 0.75),                                                                  # 12 % ramp onto the ring
        "D": ("Town05", dict(start_xy=(150.5, 176.6), ref_lane=3, ref_idx=1, s0=120.0, curve=True, x_crest=135.0, T=26.0), 0,
              "heldout", 0.75)},                                                                # origin 120 m on: the pre-roll left the frame at s0 0
}
CLASSES = {"RC03-S2-opt": RC03S2Opt, "RC03-S67-opt": RC03S67Opt, "RC03-S8-opt": RC03S8Opt, "RC03-S4-opt": RC03S4Opt,
           "RC03-S5-opt": RC03S5Opt, "RC03-S3-opt": RC03S3Opt}


def site_spec(sid, key):
    town, site, n_left, split, speed = SITES[sid][key]
    sp = CLASSES[sid]()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    # bend sites need RC03-S1's envelope governor (without it the controller's steering oscillates on wet bends and
    # even lane keeping leaves the road); the straight A sites stay as they were collected
    sp.governor = sp.governor or bool(site.get("curve"))
    sp.T = site.get("T", sp.T)                                # a site whose hazard is further away needs a longer cap
    sp.sid = sid if key == "A" else f"{sid}@{key}"
    return sp


SPECS03X = {sp.sid: sp for sid in SITES for sp in (site_spec(sid, k) for k in SITES[sid])}
