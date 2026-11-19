"""RC14 with options (RC14-S1-opt): current-view aliasing resolved by past observations - a motorbike seen earlier,
now hidden behind a van.

The ego (right lane of the two-lane corridor, y W) must move left (y 0) around a stopped car d_obs m ahead. A van
(the occluder) drives in the left lane just behind the ego. A motorbike was in view in the left lane, approaching the
van, in the 4 s pre-roll; t_since s before the decision (t0) it went into the van's sight shadow (computed from the
line-of-sight model at build time, so it is hidden in both worlds):
    fast  (bike_fast, hidden): it closes up behind the van and, just after t0, pulls out on its far side, passes it and
          cuts back into the left lane ahead of the van - into the gap the ego would take
    slow  it settles behind the van
At t0 the current view is the same in both worlds; only the remembered speed tells them apart. The policy's history
starts at the pre-roll (history_from), so a family can remember it.

A heavy vehicle (rear, on by default) tailgates the ego in its own lane with limited braking (rear_b, loaded van or truck,
late to react): braking in lane to a stop harder than it can follow ends in a rear-end, so waiting must be gentle.

Families: keep, brake (in lane, hard: only when the rear vehicle can follow), stop_end (in lane, the gentlest stop
behind the stopped car: the memory-free alternative when it is far), left (change now: fails in the fast world),
merge_back (slow down, change once the lane looks free: memory-less, it does not wait for the hidden bike),
memory_left (change now unless an actor remembered closing in the left lane is out of sight: then wait for it to pass).
Sites: RC07-S5's Town06 straights (A, B train, E1 held out); the lane right of the ego carries background traffic.
"""
import math

import numpy as np

from laws import HALF_L, Body, law_idm, quintic
from rc03_opts import OptBase
from rc07_opts import _S5
from scenarios import CAR, car, platoon
from sight import EYE, _blocked_by_boxes, _corners

BIKE, BIKE_L, BIKE_W = "vehicle.kawasaki.ninja", 2.2, 0.8
REAR = ["vehicle.mercedes.sprinter", "vehicle.carlamotors.european_hgv"]
STALLED = ["vehicle.toyota.prius", "vehicle.nissan.patrol_2021", "vehicle.volkswagen.t2_2021"]


def shadow_gap(rel_occ, L_o, w_o, y_ego, y_lane):
    """Largest gap [m] between the bike's front and the occluder's rear at which the whole bike (centre and corners) is
    hidden from the ego's eye, the occluder rel_occ m from the ego (centres) in the lane at y_lane (straight road)."""
    eye = (EYE[0], y_ego + EYE[1])
    occ = np.array([[rel_occ, y_lane, 1.0, 0.0, L_o / 2, w_o / 2]])
    best = 0.0
    for g in np.arange(0.25, 60.0, 0.25):
        bx = rel_occ - L_o / 2 - g - BIKE_L / 2
        if all(_blocked_by_boxes(eye, pt, occ) for pt in _corners((bx, y_lane, 1.0, 0.0, BIKE_L / 2, BIKE_W / 2))):
            best = g
        elif best > 0.0:
            break
    return best


def bike_path(x_o0, v_o, L_o, w_o, y_lane, v_b, fast, t_h, gap_h, t_go=0.3, a_b=4.0):
    """The bike: in the left lane at v_b, gap_h m behind the van's rear at t_h (just inside its sight shadow), closing
    in to 1.5 m behind it at the van's speed (both worlds: hidden at t0). fast: t_go s after t0 it pulls out to the
    van's far side, accelerates back to v_b, passes it and cuts back into the lane 3 m ahead of it; slow: it stays.
    x_o0, v_o: the van's centre at t = 0 and its speed. Integrated on a grid (kinematic, scripted)."""
    dt = 0.01
    ts = np.arange(-6.0, 20.0 + dt, dt)
    dv0 = v_b - v_o
    gap, rel_v, out = gap_h + dv0 * (t_h - ts[0]), dv0, False
    G, Y, VY = [], [], []
    y_pass = y_lane - (w_o / 2 + 0.9)
    y, t_back = y_lane, None
    for t in ts:
        if t > t_h and not (fast and t >= t_go):            # in the shadow: ease to 1.5 m behind, the van's speed
            rel_v = min(rel_v, max(gap - 1.5, 0.0) / 0.5)
        elif fast and t >= t_go:                            # pull out and pass at up to v_b
            rel_v = min(rel_v + a_b * dt, dv0)
            if t_back is None and gap < -(L_o + BIKE_L + 3.0):
                t_back = t
        u_out = min(max((t - t_go) / 0.8, 0.0), 1.0) if fast else 0.0
        u_in = min(max((t - t_back) / 1.0, 0.0), 1.0) if t_back is not None else 0.0
        y_new = y_lane + (y_pass - y_lane) * (quintic(u_out) - quintic(u_in))
        VY.append((y_new - y) / dt if Y else 0.0)
        y = y_new
        G.append(gap), Y.append(y)
        gap -= rel_v * dt
    G, Y, VY = np.array(G), np.array(Y), np.array(VY)
    VX = v_o + np.gradient(-G, dt)

    def f(t):
        g = float(np.interp(t, ts, G))
        x = x_o0 + v_o * t - L_o / 2 - g - BIKE_L / 2
        vx, vy = float(np.interp(t, ts, VX)), float(np.interp(t, ts, VY))
        return x, float(np.interp(t, ts, Y)), math.atan2(vy, max(vx, 0.1)), vx, vy
    return f


def heavy_rear(p, y, grip, pT):
    """A loaded van or truck tailgating the ego in its lane (RC14, RC10/11): rear_gap m behind, late to react
    (rear_react) and braking at most rear_b m/s^2 (x grip), below the ego's limit: a hard stop ends in a rear-end."""
    L_r = CAR[p["rear_bp"]][0]
    return car("rear", p["rear_bp"], -HALF_L - p["rear_gap"] - L_r / 2 - p["v"] * pT, y, p["v"],
               law=law_idm(p["v"], T=max((p["rear_gap"] - 2.0) / p["v"], 0.15), react=p["rear_react"],
                           b_max=p["rear_b"] * grip))


class RC14Opt(OptBase):
    sid = "RC14-S1-opt"
    scene_name = "hidden_motorbike"
    T, prefix_T = 10.0, 4.0
    history_from = -4.0 + 0.05                         # the policy remembers the pre-roll (options.hist_start)
    trigger = "braking"                                # the stopped car, in view at t0
    snap_stop = True
    merge_dir = -1                                     # merge_back changes left
    modes = ("keep", "brake", "stop_end", "left", "merge_back", "memory_left")
    SPEEDS = ("v",)
    OCC = ["vehicle.mercedes.sprinter", "vehicle.mercedes.sprinter", "vehicle.carlamotors.european_hgv"]
    nominal = dict(v=25.0, d_obs=60.0, stalled_bp=STALLED[0], occ_bp=OCC[0], occ_dx=-9.0, dv_occ=1.0, bike=1,
                   bike_fast=1, dv_fast=10.0, dv_slow=1.5, t_since=0.5, rear=1, rear_bp=REAR[0], rear_gap=9.0,
                   rear_react=1.0, rear_b=4.0, wx="dry")
    ranges = dict(v=(22.0, 28.0), d_obs=(55.0, 110.0), stalled_bp=STALLED, occ_bp=OCC, occ_dx=(-13.0, -6.0),
                  dv_occ=(0.0, 2.0), bike=[1, 1, 1, 1, 1, 0], bike_fast=[0, 1], dv_fast=(6.0, 14.0), dv_slow=(0.5, 2.5),
                  t_since=(0.3, 0.8), rear=[1, 1, 1, 0], rear_bp=REAR, rear_gap=(6.0, 14.0), rear_react=(0.8, 1.3),
                  rear_b=(3.5, 5.0), wx=["dry", "wet", "heavy"])
    hidden = ("bike_fast", "dv_fast", "dv_slow", "rear_react", "rear_b")
    description = ("A stopped car ahead in the ego's lane; the way around it is the left lane, where a van drives just "
                   "behind the ego. A motorbike seen approaching in the left lane went out of view behind the van "
                   "moments ago: fast, it will pass the van and arrive in the gap; slow, it stays behind. Change now, "
                   "brake, or remember the bike and wait for it; a heavy vehicle close behind cannot follow a hard stop.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e, y_l = W, 0.0
        L_st = CAR[p["stalled_bp"]][0]
        bodies = [Body("stalled", HALF_L + p["d_obs"] + L_st / 2, y_e, length=L_st, width=CAR[p["stalled_bp"]][1],
                       kind="static", bp=p["stalled_bp"])]
        L_o, w_o = CAR[p["occ_bp"]]
        v_o = v - p["dv_occ"]
        bodies.append(car("van", p["occ_bp"], p["occ_dx"] - v_o * pT, y_l, v_o,
                          law=law_idm(v_o, T=1.0, react=0.8, b_max=6 * grip)))
        if p["bike"]:
            t_h = -p["t_since"]
            rel_h = p["occ_dx"] + p["dv_occ"] * p["t_since"]           # the van relative to the ego at t_h
            g = shadow_gap(rel_h, L_o, w_o, y_e, y_l)
            gap_h = max(g - 0.5, 1.0)
            v_b = v_o + (p["dv_fast"] if p["bike_fast"] else p["dv_slow"])
            bodies.append(Body("bike", 0.0, y_l, v=v_b, length=BIKE_L, width=BIKE_W, bp=BIKE, kind="scripted",
                               path=bike_path(p["occ_dx"], v_o, L_o, w_o, y_l, v_b, p["bike_fast"], t_h, gap_h)))
        if p.get("rear", 0):                                         # older batches have no rear vehicle
            bodies.append(heavy_rear(p, y_e, grip, pT))
        bodies += platoon("r1_", 2 * W, v, [-55.0, -20.0, 15.0, 50.0])     # the lane on the right: no way out
        if self.n_left >= 1:
            bodies += platoon("l1_", -W, v + 3.0, [-70.0, 30.0])
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[], end_x=HALF_L + p["d_obs"])                   # stop_end: behind the stopped car

    def cell(self, r):
        p = r["params"]
        return ("no bike" if not p["bike"] else f"bike {'fast' if p['bike_fast'] else 'slow'}") + f", d_obs {p['d_obs']:.0f}"


def site_spec(key):
    town, site, n_left, split, speed = _S5[key]
    sp = RC14Opt()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.sid = RC14Opt.sid if key == "A" else f"{RC14Opt.sid}@{key}"
    return sp


SPECS14X = {sp.sid: sp for sp in (site_spec(k) for k in _S5)}
