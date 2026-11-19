"""RC01 / RC02 / RC09 (and the RC12, RC18, RC19 axes) as one scene with the options left open: a single-lane
roadworks corridor (2026-11-14).

Motorway roadworks close the lanes on both sides of the ego's lane with barriers and 2 m screens, so there is no
shoulder and no lane to escape into: braking is the only response, and how the ego brakes decides the outcome.

    hazard = stalled   a car has broken down in the corridor `d_stall` m ahead. On a bend the inside screen hides it
                       until the ego is close (line of sight, sight.py), on a straight it is seen from far.
    hazard = lead      the car ahead (time headway lead_T) stops: an emergency stop for a broken-down car hidden
                       beyond it (lead_mode brake, lead_dec x grip), or it clips the barrier and stops abruptly (crash,
                       10-14 m/s^2 on any road).
    benign (1 in 6)    the broken-down car stands in the closed lane behind the barrier, or the lead only slows down
                       (below the trigger's 3 m/s^2) and drives on.
    rear               none / a car / a truck following the ego at time headway rear_T, reacting rear_react s late,
                       braking at most rear_b (x grip; a truck brakes less than a car).

Which RC a variation belongs to is decided by executing every family (export_corridor.py), not by its parameters:
recoverable without the rear binding -> RC01; nothing recovers and only the front binds -> RC02; the rear binds (some
response is rear-ended) -> RC09 (negative when nothing recovers; its positive contrast when an intermediate braking
profile still does).

Families: keep / brake (constant fraction of the usable braking B(v)) / brake2 (b1 for t1 s, then b2: intermediate
and "brake, then ease off" profiles, RC09's witness). Trigger "braking": a stopped object seen in the ego lane ahead
(the broken-down car comes into view), or a car ahead seen decelerating harder than 3 m/s^2 (its brake lights;
options.braking_actor).

RC19 (case): A anchor, B appearance (paint, sun and clouds), C distant traffic beyond the barrier (sites with lanes
there), D roadside objects behind the barrier, S semantic control (a second broken-down car in the corridor, inside the stopping distance: braking can no longer recover).
The CARLA prop library has no 2 m screen: the barriers are drawn as street barriers, and the screens exist as sight
occluders only (a camera-based model would see over them).

Sites: RC03-S1-opt's middle-lane roads (straights A, B; Town04 bends C1, C2, C1b, C2b; Town05 tight bends D2, D2b;
D1 held out), so the corridor runs straight, through long bends and through tight ones.
"""
import math

import numpy as np

from envelope import ROAD
from laws import HALF_L, Body, idm_accel, leader_of
from rc03_opt import SITES as SITES03
from rc03_opts import OptBase, speed_path
from scenarios import CAR, car, platoon

STALLED = ["vehicle.volkswagen.t2_2021", "vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.nissan.patrol_2021"]
LEADS = ["vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.mercedes.sprinter", "vehicle.lincoln.mkz_2020",
         "vehicle.carlamotors.european_hgv"]
REAR = {"car": ("vehicle.audi.a2", (6.0, 8.0)), "truck": ("vehicle.carlamotors.european_hgv", (3.5, 5.5))}
COLOURS = ["200,30,30", "20,60,160", "230,230,230", "30,30,30", "240,180,20", "40,140,60"]
W_B = dict(cloudiness=60.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0, sun_altitude_angle=30.0,
           sun_azimuth_angle=300.0, fog_density=0.0)
BARRIER, SIGN, CONE = "static.prop.streetbarrier", "static.prop.warningconstruction", "static.prop.constructioncone"
PROP_DX = 6.0          # [m] barrier spacing along each corridor edge (drawn; the bounds and screens are exact)
SCREEN_STEP = 2.0      # [m] screen polygon resolution along x (sight.py maps vertices through the curved frame)


def law_rear(v0, T, react, b_max, s0=2.0, a=1.5, b=2.0):
    """A follower that is late only in *noticing*: IDM on the current gap, its braking held to at most 1 m/s^2 until
    `react` s after the car ahead's brake lights come on (decelerating > 1 m/s^2); from then on it brakes as hard as
    the closing needs (to stop s0 behind), up to b_max. (law_idm's perception delay applies to every reading, so it
    keeps reacting to where the leader was and runs into a stopped one even from a long headway.)"""
    def f(t, me, sim):
        m = sim.mem.setdefault(("rear", id(me)), {})
        s, ld = leader_of(me, sim)
        if ld is None:
            return float(np.clip(idm_accel(me.speed, v0, 1e3, 0.0, T, s0, a, b), -b_max, a))
        dv = me.speed - ld.velocity[0]
        a_ld = sim.ego_acc if ld is sim.ego else ld.acc
        if "t_on" not in m and a_ld < -1.0:
            m["t_on"] = t
        acc = idm_accel(me.speed, v0, s, dv, T, s0, a, b)
        if "t_on" not in m or t < m["t_on"] + react:
            return float(np.clip(acc, -1.0, a))
        need = dv * dv / (2 * max(s - s0, 0.1)) if dv > 0 else 0.0
        return float(np.clip(min(acc, -need), -b_max, a))
    return f


class RC01Opt(OptBase):
    sid = "RC01-opt"
    T = 12.0
    trigger = "braking"
    modes = ("keep", "brake", "brake2")
    SPEEDS = ("v",)
    nominal = dict(hazard="stalled", benign=0, v=22.0, d_stall=70.0, stall_bp="vehicle.volkswagen.t2_2021",
                   stall_dy=0.0, scr_off=0.6, screen=1, lead_T=1.4, lead_bp="vehicle.toyota.prius", t_lead=1.0,
                   lead_mode="brake", lead_dec=7.0, crash_dec=12.0, rear="none", rear_T=1.2, rear_react=0.9,
                   rear_bq=0.5, benign_dec=1.8, wx="dry", case="A")
    ranges = dict(hazard=["stalled", "lead"], benign=[0, 0, 0, 0, 0, 1], v=(14.0, 25.0), d_stall=(30.0, 110.0),
                  stall_bp=STALLED, stall_dy=(-0.4, 0.4), scr_off=(0.2, 1.5), screen=[1, 1, 0], lead_T=(0.8, 2.5),
                  lead_bp=LEADS, t_lead=(0.3, 2.0), lead_mode=["brake", "brake", "crash"], lead_dec=(6.0, 9.0),
                  crash_dec=(10.0, 14.0), rear=["none", "car", "truck"], rear_T=(0.6, 2.0), rear_react=(0.6, 1.6),
                  rear_bq=(0.0, 1.0), benign_dec=(1.0, 2.5), wx=["dry", "wet", "heavy"],
                  case=["A"])
    overrides = {}
    description = ("Single-lane roadworks corridor (barriers and screens both sides, no shoulder): a broken-down car "
                   "around the bend, or the car ahead stopping hard; maybe a car or truck close behind. Brake at a "
                   "constant level, or in two stages; which profile (if any) avoids both ends is measured.")

    # ---- geometry ----
    def lanes(self, W):
        y_e = W
        lo, hi = y_e - W / 2 - 0.3, y_e + W / 2 + 0.3          # barrier faces: 0.3 m outside the lane lines
        return y_e, lo, hi

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        rd = self.road(p)
        grip = rd["grip"]
        y_e, lo, hi = self.lanes(W)
        bodies, x_end = [], 60.0
        hz, benign = p["hazard"], p["benign"]
        if hz == "stalled":
            bp = p["stall_bp"]
            Ls, Ws = CAR[bp]
            x_st = HALF_L + p["d_stall"] + Ls / 2
            y_st = y_e + p["stall_dy"] if not benign else y_e - W - 0.5     # benign: in the closed lane, behind the barrier
            bodies.append(Body("stalled", x_st, y_st, length=Ls, width=Ws, kind="static", bp=bp))
            x_end = x_st + 40.0
        else:
            bp = p["lead_bp"]
            Ll = CAR[bp][0]
            x_l = HALF_L + v * p["lead_T"] + Ll / 2                         # lead centre at t0
            t_b = p["t_lead"]
            if benign:                                                      # slows below the trigger, then drives on
                a = p["benign_dec"]
                phases = [(t_b, 0.0, v), (t_b + 1.5, -a, 0.0), (30.0, 1.0, v)]
            else:
                a = p["crash_dec"] if p["lead_mode"] == "crash" else p["lead_dec"] * grip
                phases = [(t_b, 0.0, v), (30.0, -a, 0.0)]
            t0 = -pT - 0.5
            path = speed_path(x_l + v * t0, y_e, 0.0, v, phases, t0=t0, t1=40.0)
            bodies.append(car("lead", bp, 0, y_e, v, kind="scripted", path=path))
            x_stop = x_l + v * t_b + v * v / (2 * a)
            if not benign and p["lead_mode"] == "brake":                    # what the lead stops for (hidden by it)
                sb = p["stall_bp"]
                Ls, Ws = CAR[sb]
                bodies.append(Body("stalled", x_stop + Ll / 2 + 4.0 + Ls / 2, y_e + p["stall_dy"], length=Ls,
                                   width=Ws, kind="static", bp=sb))
            x_end = max(x_stop, x_l + v * (t_b + 1.5)) + 50.0
        if p["case"] == "S" and hz == "stalled":                # semantic: a second broken-down car, in view but
            x2 = HALF_L + max(0.6 * v * v / 15.0, 6.0) + 2.2     # inside the stopping distance at t0
            bodies.append(Body("stalled_2", x2, y_e, length=4.4, width=1.8, kind="static",
                               bp="vehicle.volkswagen.t2_2021"))
        if p["rear"] != "none":
            rbp, (b_lo, b_hi) = REAR[p["rear"]]
            Lr = CAR[rbp][0]
            gap = max(v * p["rear_T"], 3.0)
            bodies.append(car("rear", rbp, -HALF_L - gap - Lr / 2 - v * pT, y_e, v,
                              law=law_rear(v, max((gap - 2.0) / v, 0.15), p["rear_react"],
                                           (b_lo + p["rear_bq"] * (b_hi - b_lo)) * grip)))
        x0 = -v * pT - 20.0
        x1 = x_end + 20.0
        props = self.edge_props(x0, x1, lo, hi)
        occ = self.screens(x0, x1, lo, hi, p) if p["screen"] else []
        bg = self.nuisance(p, W, x0, x1, lo, hi, v, props)
        if p["case"] == "B":
            for i, b in enumerate(bodies):
                b.color = COLOURS[i % len(COLOURS)]
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (lo, hi), v_ref=v, props=props, occluders=occ)

    def edge_props(self, x0, x1, lo, hi):
        xs = np.arange(x0, x1, PROP_DX)
        out = [(BARRIER, float(x), lo - 0.4) for x in xs] + [(BARRIER, float(x), hi + 0.4) for x in xs]
        return out + [(SIGN, float(x0 + 5.0), lo - 1.2)]

    def screens(self, x0, x1, lo, hi, p):
        """2 m screens scr_off m beyond each barrier face: thin polygons sampled every SCREEN_STEP m, so they
        follow the road on a bend (sight.py maps each vertex through the curved frame)."""
        xs = np.arange(x0, x1 + 1e-6, SCREEN_STEP)
        out = []
        for y in (lo - p["scr_off"], hi + p["scr_off"]):
            out.append([(float(x), y) for x in xs] + [(float(x), y - 0.2 * np.sign(y - (lo + hi) / 2))
                                                       for x in xs[::-1]])
        return out

    def nuisance(self, p, W, x0, x1, lo, hi, v, props):
        """RC19 C / D: traffic beyond the barrier (never interacting, never observed: bg_*) and roadside objects
        (props: never occlude, never collide with the corridor)."""
        bg = []
        if p["case"] == "C" and self.n_left >= 1:
            bg += platoon("far", -W, v * 0.9, [-70.0, -30.0, 40.0, 90.0])
        if p["case"] in ("C", "D"):
            for x in np.arange(x0 + 15.0, x1 - 10.0, 25.0):
                props.append((CONE, float(x), lo - 2.5))
                props.append(("static.prop.barrel", float(x) + 8.0, hi + 2.5))
        return bg

    def weather_for(self, p):
        return W_B if p.get("case") == "B" and p.get("wx", "dry") == "dry" else super().weather_for(p)

    def ego_color(self, p):
        return "240,240,240" if p.get("case") == "B" else None

    def goal(self, env):
        """Stopped and holding (RC01's protected stop) once the trigger has fired: the ego and the car behind it are
        at rest at the end. Without a trigger (benign) nothing more is required."""
        if env.mem.get("hazard", {}).get("t_det") is None:
            return True, {}
        rear = env.bodies.get("rear")
        v_rear = 0.0 if rear is None else rear.speed
        ok = env.ego.speed < 0.5 and v_rear < 1.0
        return ok, dict(v_end=round(env.ego.speed, 2), v_rear_end=round(v_rear, 2))

    def cell(self, r):
        p = r["params"]
        what = "benign" if p["benign"] else p["hazard"] + ("" if p["hazard"] == "stalled" else "/" + p["lead_mode"])
        return f"{what}, rear {p['rear']}"


SITES = {"RC01-opt": {k: SITES03[k] for k in ("A", "B", "C1", "C2", "C1b", "C2b", "D2", "D2b", "D1")}}


def site_spec(sid, key):
    town, site, n_left, split, speed = SITES[sid][key]
    sp = RC01Opt()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.governor = bool(site.get("curve"))
    sp.sid = sid if key == "A" else f"{sid}@{key}"
    return sp


SPECS01X = {sp.sid: sp for sid in SITES for sp in (site_spec(sid, k) for k in SITES[sid])}
