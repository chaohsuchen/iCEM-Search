"""Multi-option teachers: a declared library of maneuver families, their recovery profile, and role-free observations.

A scenario class with `modes` (e.g. RC03-S1-opt) is not taught one maneuver. Every variation is executed with every
family of the library (mode-separated expert search, report §9/§11), giving its recovery profile
R(h) = [r_keep, r_brake, r_left, r_right] at the decision point h. The learner (profile_bc.py) has one policy per
family and a selector that picks the family from the observation at h, so it has to read from the scene which options
exist instead of learning one fixed escape.

Decision point h = the observable intrusion (`intrusion_seen`): the first tick at which a delivered actor's footprint
has crossed into the ego lane while still moving towards it, or an object in the ego lane ahead needs > 3 m/s^2 of
braking. A crash next to the ego (sudden deceleration) is not a trigger by itself: nothing in it yet shows whether the
car will end in the ego lane, so no family may react before the intrusion is seen.

Families (each behind the same trigger, so all share the same detection latency; ModeExpert can instead start a
family at a given time, for the world-model data's other decision points):
    keep   cruise in lane, no reaction (the right answer in benign scenes)
    brake  emergency braking in lane at b_frac of the usable braking B(v) (friction circle shared with the lane
           tracking), optionally shifted `off` m inside the lane away from the intruding side; to a stop
    left / right  EscapeAdjacent to the adjacent lane, then lane tracking + IDM on its traffic (A_lat, b_evasive;
           y_off: aim point shifted back towards the original lane; t_start: hold the lane that long after the trigger)
    left_brake / right_brake  the same lane change, ready to brake: emergency braking in the target lane once an
           intrusion into it is observed (then_brake); b_evasive = how much speed is shed during the change
SETTINGS are the fixed first candidates of each family; PARAM_SPACE is the continuous space the search samples from.
Commands are converted to the envelope action z with Envelope.inverse (the experts' braking is capped by 0.9 of the
usable combined budget), so every demonstration is inside the vehicle envelope.

Preference among supported families (report §7, fixed before any comparison): J = 0.6 progress + 0.4 min(r, 1), with
progress = distance driven / (v0 T); refuted families get J = -1.
"""
import math

import numpy as np


def notebook_ns():
    from expert import notebook_ns as ns     # lazy: expert imports scenarios, which imports this module
    return ns()

MODES = ("keep", "brake", "left", "right")
SETTINGS = {
    "keep": [dict()],
    "brake": [dict(b_frac=0.9, off=0.0), dict(b_frac=0.9, off=0.45)],
    "left": [dict(A_lat=6.0, b_evasive=3.0, y_off=0.0), dict(A_lat=4.5, b_evasive=3.0, y_off=0.0),
             dict(A_lat=6.0, b_evasive=5.0, y_off=0.35)],
}
SETTINGS["right"] = SETTINGS["left"]
SETTINGS.update({
    "shoulder": [dict(dy=2.3, A_lat=5.0, margin=3.0, acc=0.0), dict(dy=1.5, A_lat=4.0, margin=3.0, acc=-2.0)],
    "swerve_left": [dict(dy=3.0, A_lat=5.0, margin=3.0, acc=0.0), dict(dy=2.5, A_lat=6.0, margin=2.0, acc=-2.0)],
    "merge_front": [dict(a=2.0, g_lead=4.0, g_lag=4.0, A_lat=5.0, y_off=0.35, b_ev=3.0),
                    dict(a=3.0, g_lead=3.0, g_lag=2.0, A_lat=6.0, y_off=0.5, b_ev=1.0)],
    "merge_back": [dict(a=-6.0, g_lead=3.0, g_lag=4.0, A_lat=5.0, y_off=0.35, b_ev=3.0),
                   dict(a=-4.0, g_lead=5.0, g_lag=4.0, A_lat=6.0, y_off=0.5, b_ev=5.0)],
})
PARAM_SPACE = {
    "brake": dict(b_frac=(0.5, 1.0), off=(-0.6, 0.6)),
    "left": dict(A_lat=(3.0, 8.0), b_evasive=(0.0, 6.0), y_off=(-0.5, 0.6), t_start=(0.0, 0.4)),
}
PARAM_SPACE["right"] = PARAM_SPACE["left"]
PARAM_SPACE.update({
    "shoulder": dict(dy=(1.0, 3.5), A_lat=(3.0, 7.0), margin=(1.0, 5.0), acc=(-4.0, 0.0)),
    "swerve_left": dict(dy=(1.5, 4.0), A_lat=(3.0, 7.0), margin=(1.0, 5.0), acc=(-4.0, 0.0)),
    "merge_front": dict(a=(0.5, 3.5), g_lead=(1.0, 8.0), g_lag=(1.0, 8.0), A_lat=(3.0, 8.0), y_off=(-0.3, 0.9),
                        b_ev=(0.0, 6.0)),
    "merge_back": dict(a=(-8.0, -2.0), g_lead=(1.0, 8.0), g_lag=(1.0, 8.0), A_lat=(3.0, 8.0), y_off=(-0.3, 0.9),
                       b_ev=(0.0, 6.0)),
})
PARAM_DEFAULT = {"brake": dict(b_frac=0.9, off=0.0), "left": dict(A_lat=6.0, b_evasive=3.0, y_off=0.0, t_start=0.0)}
PARAM_DEFAULT["right"] = PARAM_DEFAULT["left"]
PARAM_DEFAULT.update({m: dict(v[0]) for m, v in SETTINGS.items() if m in ("shoulder", "swerve_left", "merge_front",
                                                                          "merge_back")})
# lane change ready to brake (RC03-S1, 2026-11-08): the escape's settings, and the fixed candidates span holding the speed
# through the lane change (b_evasive 0) to slowing down in it, so the library shows when only the cautious one recovers
for m in ("left_brake", "right_brake"):
    SETTINGS[m] = [dict(A_lat=6.0, b_evasive=0.0, y_off=0.0), dict(A_lat=6.0, b_evasive=3.0, y_off=0.0),
                   dict(A_lat=6.0, b_evasive=5.0, y_off=0.35)]
    PARAM_SPACE[m] = PARAM_SPACE["left"]
    PARAM_DEFAULT[m] = PARAM_DEFAULT["left"]
MODES_LB = MODES + ("left_brake", "right_brake")
# shift_left: a partial move inside the own lane away from an obstacle in its right half, held until the hand-back
SETTINGS["shift_left"] = [dict(dy=0.9, A_lat=3.0, acc=0.0), dict(dy=1.1, A_lat=3.0, acc=-1.0)]
PARAM_SPACE["shift_left"] = dict(dy=(0.5, 1.3), A_lat=(1.5, 4.0), acc=(-2.0, 0.0))
PARAM_DEFAULT["shift_left"] = dict(SETTINGS["shift_left"][0])
# brake2 (RC01-opt): b1 of the usable braking for t1 s, then b2, to a stop - intermediate levels and "brake, then ease
# off" (RC09: use the room ahead without being rear-ended) next to brake's constant level
SETTINGS["brake2"] = [dict(b1=1.0, t1=0.6, b2=0.5), dict(b1=0.5, t1=1.0, b2=1.0), dict(b1=0.7, t1=1.5, b2=0.35),
                      dict(b1=0.35, t1=1.5, b2=0.8)]
PARAM_SPACE["brake2"] = dict(b1=(0.15, 1.0), t1=(0.2, 3.0), b2=(0.15, 1.0))
PARAM_DEFAULT["brake2"] = dict(SETTINGS["brake2"][0])
# brake_free (RC10, contingent escape): brake in lane at b_frac while watching the actor that fired the trigger; once
# it is seen d_seen m off the ego lane's centre (the side it slides to), change into the adjacent lane on the OTHER side
# (A_lat, b_evasive, y_off as left / right); never seen leaving the lane: keep braking
# (an escape started from hard braking overshoots the target lane: gentle A_lat and an aim short of its centre)
SETTINGS["brake_free"] = [dict(b_frac=0.6, d_seen=0.1, A_lat=4.5, b_evasive=2.0, y_off=0.3),
                          dict(b_frac=0.9, d_seen=0.1, A_lat=3.0, b_evasive=2.0, y_off=0.3),
                          dict(b_frac=0.3, d_seen=0.2, A_lat=4.5, b_evasive=2.0, y_off=0.6)]
PARAM_SPACE["brake_free"] = dict(b_frac=(0.3, 1.0), d_seen=(0.1, 1.0), A_lat=(3.0, 8.0), b_evasive=(0.0, 6.0),
                                 y_off=(-0.5, 0.6))
PARAM_DEFAULT["brake_free"] = dict(SETTINGS["brake_free"][0])
# RC15 (merge where the ego's lane ends; env.spec.merge_dir = the target lane's side, env.scene["end_x"] = the end):
#   watch_merge  nudge nudge_y m towards the target lane holding the speed, watching the adjacent car's speed change
#                (over the last 0.5 s, as observed); it slows by more than a_thr m/s^2: merge in front (merge_front with
#                a_f), it speeds up: drop back and merge behind (merge_back with a_b); after t_watch s with neither: in
#                front if its front is behind the ego's rear, else behind. The decision is the observed response.
#   stop_end     protective stop: brake to a stop d_gap m before the lane end (no merge: a mitigation, never a success)
SETTINGS["watch_merge"] = [dict(nudge_y=0.7, a_thr=0.8, t_watch=1.5, a_f=2.0, a_b=-4.0, g_lead=4.0, g_lag=4.0, A_lat=5.0,
                                y_off=0.35, b_ev=0.0),
                           dict(nudge_y=0.9, a_thr=0.5, t_watch=2.5, a_f=2.5, a_b=-3.0, g_lead=3.0, g_lag=6.0, A_lat=5.0,
                                y_off=0.35, b_ev=0.0)]
PARAM_SPACE["watch_merge"] = dict(nudge_y=(0.5, 1.2), a_thr=(0.3, 1.5), t_watch=(0.8, 3.5), a_f=(0.5, 3.5),
                                  a_b=(-6.0, -1.5), g_lead=(1.0, 8.0), g_lag=(1.0, 8.0), A_lat=(3.0, 7.0),
                                  y_off=(-0.3, 0.9), b_ev=(0.0, 3.0))
PARAM_DEFAULT["watch_merge"] = dict(SETTINGS["watch_merge"][0])
# memory_left (RC14): change into the lane on the left now, unless an actor seen earlier in that lane behind the ego,
# seen closing faster than dv_thr (its fastest in the last 3 s), is out of sight now (it may arrive in the gap): then slow at b_wait of the
# usable braking until it is seen again and its rear is g_pass m past the ego's front, or t_wait s have passed
SETTINGS["memory_left"] = [dict(dv_thr=3.0, b_wait=0.6, t_wait=3.0, g_pass=3.0, A_lat=5.0, b_evasive=2.0, y_off=0.3),
                           dict(dv_thr=1.5, b_wait=0.9, t_wait=2.0, g_pass=6.0, A_lat=6.0, b_evasive=0.0, y_off=0.3)]
PARAM_SPACE["memory_left"] = dict(dv_thr=(0.5, 6.0), b_wait=(0.2, 1.0), t_wait=(0.5, 4.0), g_pass=(0.0, 10.0),
                                  A_lat=(3.0, 7.0), b_evasive=(0.0, 4.0), y_off=(-0.3, 0.6))
PARAM_DEFAULT["memory_left"] = dict(SETTINGS["memory_left"][0])
SETTINGS["stop_end"] = [dict(d_gap=8.0, b_max=0.9), dict(d_gap=20.0, b_max=0.7)]
PARAM_SPACE["stop_end"] = dict(d_gap=(3.0, 30.0), b_max=(0.4, 1.0))
PARAM_DEFAULT["stop_end"] = dict(SETTINGS["stop_end"][0])
# shift_pass (RC16): past a stopped obstacle in the own lane through a narrow passage beside it (spec.pass_dir). Slow to
#   v_pass at b, shift dy m along a quintic over D m that ends x_ready m before the obstacle (at once if already
#   closer), hold the line until the obstacle is passed by 3 m, shift back over D m. D sets the peak heading (early and
#   shallow vs late and sharp); v_pass < v is "brake, then shift".
SETTINGS["shift_pass"] = [dict(dy=1.2, D=25.0, x_ready=6.0, v_pass=12.0, b=2.0), dict(dy=1.2, D=10.0, x_ready=3.0,
                                                                                      v_pass=5.0, b=4.0),
                          dict(dy=1.2, D=8.0, x_ready=0.0, v_pass=12.0, b=2.0)]
PARAM_SPACE["shift_pass"] = dict(dy=(0.6, 1.8), D=(6.0, 35.0), x_ready=(0.0, 15.0), v_pass=(2.0, 12.0), b=(1.0, 5.0))
PARAM_DEFAULT["shift_pass"] = dict(SETTINGS["shift_pass"][0])
# overtake (RC08): out into the opposing lane (dy m to the left) at once, past the stopped obstacle by margin m, back;
#   acceleration acc (capped at the speed at the trigger)
SETTINGS["overtake"] = [dict(dy=4.0, A_lat=4.0, margin=4.0, acc=1.0), dict(dy=3.6, A_lat=5.0, margin=2.0, acc=-1.0)]
PARAM_SPACE["overtake"] = dict(dy=(3.0, 4.4), A_lat=(2.5, 6.0), margin=(1.0, 8.0), acc=(-3.0, 2.0))
PARAM_DEFAULT["overtake"] = dict(SETTINGS["overtake"][0])
# wait_overtake (RC08): stop d_gap m behind the stopped obstacle (as stop_end), edging y_peek m left to see past it
#   (at most 0.55 m: still lane-centred for the terminal check);
#   while waiting, go (overtake with A_lat, margin, a_go) once every delivered oncoming car in the opposing lane needs
#   more than the time to pass plus T_need to reach the point where the ego is back in its lane; follow the obstacle if
#   it drives off. An oncoming car out of sight behind the obstacle is not seen: peeking further sees more.
SETTINGS["wait_overtake"] = [dict(d_gap=6.0, b_max=0.7, y_peek=0.3, T_need=3.0, a_go=1.5, A_lat=4.0, margin=4.0),
                             dict(d_gap=10.0, b_max=0.7, y_peek=0.5, T_need=5.0, a_go=1.0, A_lat=3.5, margin=4.0)]
PARAM_SPACE["wait_overtake"] = dict(d_gap=(3.0, 15.0), b_max=(0.4, 1.0), y_peek=(0.0, 0.55), T_need=(0.0, 8.0),
                                    a_go=(0.8, 2.5), A_lat=(2.5, 6.0), margin=(1.0, 8.0))
PARAM_DEFAULT["wait_overtake"] = dict(SETTINGS["wait_overtake"][0])
# hand-back behaviours (h_ parameters), built at the hand-back point from the state there:
#   return  back to the own lane (h_y m off its centre, away from the side the ego is on) once every slow obstacle in
#           it is h_margin m behind the ego's rear and no car is alongside there; lateral cap h_A; speed back towards the
#           speed at the trigger (IDM, acceleration h_a)
#   edge    return, ending h_y m off-centre on the far side (RC07-S3: room for a motorcycle filtering along the line)
#   stay    hold the lane (or in-lane offset) the ego is in; speed towards the lane's traffic (h_match: 0 = hold the
#           speed, 1 = the speed of the nearest car in that lane, behind first), IDM acceleration h_a, time gap h_T
#   stop    brake to a stop h_gap m behind the nearest stopped obstacle ahead in the ego's lane (e.g. a broken-down
#           car: the deceleration that stops it there, up to h_b of the usable braking); none ahead: at h_b at once
#   away    hold the lane the ego is in and pull away: speed towards the speed at the trigger + h_dv (IDM on the car
#           ahead, acceleration h_a; RC07-S5: stay ahead of a car accelerating up behind)
HB_SPACE = {"return": dict(h_A=(2.0, 6.0), h_margin=(0.5, 6.0), h_a=(0.3, 2.5), h_y=(-0.2, 0.2)),
            "edge": dict(h_A=(1.5, 4.0), h_margin=(0.5, 6.0), h_a=(0.3, 2.5), h_y=(0.2, 0.5)),     # 0.55: the track contract
            "stay": dict(h_match=(0.0, 1.0), h_a=(0.5, 3.0), h_T=(0.8, 2.0)),
            "stop": dict(h_b=(0.5, 1.0), h_gap=(2.0, 10.0)),
            "away": dict(h_dv=(1.0, 6.0), h_a=(1.0, 3.5))}             # at most ~20 km/h over the trigger speed
HB_SETTINGS = {"return": [dict(h_A=3.0, h_margin=3.0, h_a=1.0, h_y=0.0), dict(h_A=5.0, h_margin=1.0, h_a=2.0, h_y=0.0)],
               "edge": [dict(h_A=3.0, h_margin=2.0, h_a=1.0, h_y=0.35), dict(h_A=3.0, h_margin=1.0, h_a=1.0, h_y=0.45)],
               "stay": [dict(h_match=1.0, h_a=2.5, h_T=1.2), dict(h_match=0.0, h_a=1.0, h_T=1.5)],
               "stop": [dict(h_b=0.9, h_gap=4.0), dict(h_b=0.6, h_gap=8.0)],
               "away": [dict(h_dv=6.0, h_a=3.0), dict(h_dv=3.0, h_a=2.0)]}


def composite(mode):
    """Register "stage1>handback": fixed settings (stage 1's last - for a lane change the one aiming short of the
    target lane's centre, against CARLA's overshoot - with each hand-back's first two, and stage 1's first with the
    hand-back's first), the joint search space."""
    if mode in SETTINGS or ">" not in mode:
        return mode
    s1, hb = mode.split(">")
    S1, H = SETTINGS[s1], HB_SETTINGS[hb]
    SETTINGS[mode] = [{**S1[-1], **H[0]}, {**S1[-1], **H[1]}, {**S1[0], **H[0]}]
    PARAM_SPACE[mode] = {**PARAM_SPACE[s1], **HB_SPACE[hb]}
    PARAM_DEFAULT[mode] = {**full_cfg(s1, S1[-1]), **H[0]}
    return mode


def group_of(mode):
    """The stage-1 group a family belongs to (its own name without a hand-back)."""
    return mode.split(">")[0]


def full_cfg(mode, cfg):
    """A family setting with every searched parameter (defaults filled in)."""
    return {**PARAM_DEFAULT.get(mode, {}), **cfg}
W_PROGRESS = 0.6
VY_INTRUDE = 0.3                  # [m/s] lateral speed towards the ego lane that makes a lane-line crossing an intrusion


def preference(success, r_nb, progress):
    return W_PROGRESS * progress + (1 - W_PROGRESS) * min(r_nb, 1.0) if success else -1.0


# ---------------- observations ----------------
SLOT_F = ["present", "dx", "dy_lane", "dvx", "dvy", "psi", "a", "length", "width"]
SLOTS = [(lane, kind, k) for lane in ("L", "E", "R") for kind, k in (("ahead", 0), ("ahead", 1), ("behind", 0))]
OBS_BASE = (["ego_v", "ego_dy_lane", "ego_psi", "ego_yaw_rate", "ego_a", "room_left", "room_right", "hazard", "t_hazard"]
            + [f"{lane}{kind[0]}{k}_{f}" for lane, kind, k in SLOTS for f in SLOT_F])
# what the ego knows of the road and itself: rain level (0 dry, 0.5 wet, 1 heavy), its usable braking and lateral
# acceleration at its speed under that condition (the qualified envelope), road curvature here and 50 m ahead, lane width,
# and the braking the combined budget leaves while following the bend here (cap_B_bend: on a wet bend the cornering
# takes much of the grip, which the separate inputs leave the selector to combine on few scenes); last, so a model
# trained before it reads the observation without it
OBS_ROAD = ["road_wet", "cap_B", "cap_AY", "curv0", "curv50", "lane_w", "cap_B_bend"]
OBS_NAMES = OBS_BASE + OBS_ROAD
WET_LEVEL = {"dry": 0.0, "wet": 0.5, "heavy": 1.0}
D_FAR = 80.0


def intrusion_seen(ns, o, lane_y):
    """Causal trigger on the delivered history (last two snapshots): some actor's rotated footprint overlaps the ego
    lane and moves towards it faster than VY_INTRUDE, or an object ahead in the ego lane needs more than 3 m/s^2 of
    braking (the notebook hazard_seen's in-lane test)."""
    return intrusion_actor(ns, o, lane_y) is not None


def intrusion_actor(ns, o, lane_y):
    """intrusion_seen's test -> the name of the actor that fires it (None: no intrusion)."""
    if len(o.history) < 2:
        return None
    e, W = o.ego, ns["LANE_W"]
    (t1, s1), (t2, s2) = o.history[-2], o.history[-1]
    for n, b in s2.items():
        ex, ey = ns["extent"](b)
        if b["x"] + ex / 2 < e["x"] - ns["HALF_L"]:                 # entirely behind the ego
            continue
        s = b["x"] - ex / 2 - (e["x"] + ns["HALF_L"])
        closing = e["v"] - b["v"] * math.cos(b["psi"])
        if abs(b["y"] - lane_y) < (ey + ns["EGO_W"]) / 2 and s > 0 and closing > 0 and closing ** 2 / (2 * s) > 3.0:
            return n
        a = s1.get(n)
        if a is None or abs(b["y"] - lane_y) - ey / 2 >= W / 2:    # footprint still outside the ego lane
            continue
        if (b["y"] - a["y"]) / (t2 - t1) * math.copysign(1, lane_y - b["y"]) > VY_INTRUDE:
            return n
    return None


H_CONFLICT = 4.0                  # [s] prediction horizon of the conflict trigger


def conflict_seen(ns, o, lane_y):
    """intrusion_seen, or a predicted conflict: a delivered actor that crosses, turns or comes the other way, projected
    with its observed speed and acceleration along its heading (to a stop at most, never reversing), overlaps the ego
    held at its speed in its lane within H_CONFLICT s. A car seen braking to its stop line is not a conflict; one that
    keeps going, or starts again, is - only what the ego has observed is used."""
    return conflict_actor(ns, o, lane_y) is not None


def conflict_actor(ns, o, lane_y):
    """conflict_seen's test -> the name of the actor that fires it (None: no conflict)."""
    n = intrusion_actor(ns, o, lane_y)
    if n is not None:
        return n
    if len(o.history) < 3:
        return None
    e = o.ego
    (t0, s0), (t2, s2) = o.history[-3], o.history[-1]
    for n, b in s2.items():
        a0 = s0.get(n)
        if a0 is None:
            continue
        crossing = abs(math.sin(b["psi"])) > 0.35 or math.cos(b["psi"]) < 0
        if not crossing or b["x"] + b["length"] / 2 < e["x"] - ns["HALF_L"]:
            continue
        acc = max(min((b["v"] - a0["v"]) / (t2 - t0), 3.0), -9.0)
        ex, ey = ns["extent"](b)
        for tau in np.arange(0.0, H_CONFLICT + 1e-6, 0.1):
            t_stop = b["v"] / -acc if acc < 0 else np.inf
            tt = min(tau, t_stop)
            d = b["v"] * tt + 0.5 * acc * tt ** 2
            bx, by = b["x"] + d * math.cos(b["psi"]), b["y"] + d * math.sin(b["psi"])
            if abs(bx - (e["x"] + e["v"] * tau)) < (2 * ns["HALF_L"] + ex) / 2 + 0.5 and \
                    abs(by - lane_y) < (ns["EGO_W"] + ey) / 2 + 0.2:
                return n
    return None


A_LEAD = 3.0                      # [m/s^2] a car ahead in the ego lane seen decelerating harder fires the braking trigger


def braking_seen(ns, o, lane_y):
    return braking_actor(ns, o, lane_y) is not None


def braking_actor(ns, o, lane_y):
    """intrusion_actor, a stopped object seen in the ego lane ahead (a driver starts slowing for a broken-down car
    once it is in view, not once it needs 3 m/s^2), or a delivered car ahead in the ego lane seen decelerating harder
    than A_LEAD over the last 0.1 s (its brake lights and closing; RC01-opt's lead stopping hard)."""
    n = intrusion_actor(ns, o, lane_y)
    if n is not None or len(o.history) < 3:
        return n
    e = o.ego
    for n, b in o.history[-1][1].items():
        if b["x"] > e["x"] and abs(b["y"] - lane_y) < ns["LANE_W"] / 2 and b["v"] < 0.5:
            return n
    (t0, s0), (t2, s2) = o.history[-3], o.history[-1]
    for n, b in s2.items():
        a = s0.get(n)
        if a is None or b["x"] <= e["x"] or abs(b["y"] - lane_y) >= ns["LANE_W"] / 2:
            continue
        if (a["v"] - b["v"]) / (t2 - t0) > A_LEAD:
            return n
    return None


def start_actor(ns, o, lane_y):
    """RC15's merge: the hazard (the ego's lane ending) is known from the start (road geometry): fires at t0."""
    return "lane_end"


TRIGGERS = {"intrusion": intrusion_seen, "conflict": conflict_seen, "braking": braking_seen,
            "start": lambda ns, o, lane_y: True}
ACTOR_TRIGGERS = {"intrusion": intrusion_actor, "conflict": conflict_actor, "braking": braking_actor,
                  "start": start_actor}


def trigger_of(env):
    return TRIGGERS[getattr(env.spec, "trigger", "intrusion")]


def hist_start(env):
    """Where the observed history starts: t0, or earlier for a spec whose decision needs what was seen in its pre-roll
    (spec.history_from < 0; RC14: a motorbike seen before the decision, hidden at it)."""
    return getattr(env.spec, "history_from", 0.0) - 1e-6


def env_obs(env, ns):
    """The notebook Obs the experts get at this tick (delivered history from t0 on, or from spec.history_from)."""
    e = env.ego
    hist = [s for s in env.snaps if s[0] >= hist_start(env)] or env.snaps[-1:]
    ego = dict(x=float(e.position[0]), y=float(e.position[1]), v=float(e.speed), psi=float(e.heading),
               length=5.0, width=2.0, steer=0.0, acc=float(env.ego_acc))
    return ns["Obs"](t=env.t, ego=ego, actors=hist[-1][1], history=hist, lanes=3)


def hazard_state(env):
    """The trigger (intrusion_seen) on the delivered history, latched: identical to what every family's wrapper sees,
    so the learner and the experts decide at the same moment. The actor that fired it is kept (the hand-back point)."""
    m = env.mem.setdefault("hazard", {})
    if env.t >= -1e-6 and m.get("t_det") is None:
        ns = notebook_ns()
        ns["LANE_W"] = env.W
        if m.get("lane_y") is None:
            m["lane_y"] = round(env.ego.position[1] / env.W) * env.W
        n = ACTOR_TRIGGERS[getattr(env.spec, "trigger", "intrusion")](ns, env_obs(env, ns), m["lane_y"])
        if n is not None:
            m["t_det"], m["actor"] = env.t, n
    t_det = m.get("t_det")
    return (1.0, min(env.t - t_det, 3.0)) if t_det is not None else (0.0, 0.0)


# ---------------- the hand-back (RC07: the second, observable decision) ----------------
# After the first maneuver has avoided the primary hazard, how the ego hands back decides the outcome (return to its
# lane, stay and match the lane's speed, stop). That choice is made at the hand-back point, when what decides it (a car
# closing in the lane the ego moved to, a car planning its stop, a filtering motorcycle) is observable - not at the
# trigger. Hand-back point: the first tick after the trigger at which the ego has passed the actor that fired the
# trigger (its rear beyond that actor's front), or has settled in another lane than its own, or T_HB_MAX s after the
# trigger. Families "stage1>handback" share their stage 1 up to that point.
T_HB_MAX = 3.0
OBS_HB = ["handback", "t_handback"]


def handback_seen(ns, o, lane_y, actor, W):
    e = o.ego
    k = round(e["y"] / W) * W
    if abs(k - lane_y) > 0.5 * W and abs(e["y"] - k) < 0.5 and abs(e["psi"]) < 0.05:
        return True
    b = next((snap[actor] for _, snap in reversed(o.history) if actor in snap), None) if actor else None
    return b is not None and e["x"] - ns["HALF_L"] > b["x"] + ns["extent"](b)[0] / 2


def handback_state(env):
    """The hand-back point on the delivered history, latched (as hazard_state; the families' HandBack sees the same)."""
    m = env.mem.setdefault("handback", {})
    hz = env.mem.get("hazard", {})
    if hz.get("t_det") is not None and m.get("t_hb") is None:
        ns = notebook_ns()
        ns["LANE_W"] = env.W
        if env.t - hz["t_det"] >= T_HB_MAX - 1e-6 or handback_seen(ns, env_obs(env, ns), hz["lane_y"], hz.get("actor"),
                                                                      env.W):
            m["t_hb"] = env.t
    t = m.get("t_hb")
    return (1.0, min(env.t - t, 3.0)) if t is not None else (0.0, 0.0)


def handback_features(env):
    """lane_slot_features + the hand-back point (OBS_NAMES + OBS_HB)."""
    f = lane_slot_features(env)
    return np.concatenate([f, np.array(handback_state(env), np.float32)])


def cap_b_bend(v, kappa, B, M):
    """Braking left at speed v on curvature kappa: the combined budget M less the bend's lateral demand v^2 kappa."""
    return min(B, math.sqrt(max(M * M - (v * v * kappa) ** 2, 0.0)))


def road_features(env):
    e = env.ego
    _, B, AY, M = env.envelope.caps(e.speed)
    k0 = env.curvature(e.position[0])
    return [WET_LEVEL[env.params.get("wx", "dry")], B, AY, k0, env.curvature(e.position[0] + 50.0), env.W,
            cap_b_bend(e.speed, k0, B, M)]


_M = {}


def augment_obs(O, wx, envelope, W):
    """Observations recorded before OBS_ROAD existed (site A: straight) -> with the road features appended; recorded
    before cap_B_bend -> with it computed from the stored speed, curvature, braking and the road condition's budget."""
    n = len(O)
    if O.shape[1] == len(OBS_BASE):
        caps = np.array([envelope.caps(v)[1:3] for v in O[:, 0]], np.float32)
        O = np.concatenate([O, np.full((n, 1), WET_LEVEL[wx], np.float32), caps, np.zeros((n, 2), np.float32),
                            np.full((n, 1), W, np.float32)], 1)
    if O.shape[1] == len(OBS_NAMES) - 1:
        if not _M:
            from envelope import ROAD, Envelope
            _M.update({WET_LEVEL[k]: Envelope(mu=r["mu"]).M for k, r in ROAD.items()})
        i = {k: OBS_NAMES.index(k) for k in ("ego_v", "curv0", "cap_B", "road_wet")}
        col = [cap_b_bend(o[i["ego_v"]], o[i["curv0"]], o[i["cap_B"]], _M[float(o[i["road_wet"]])]) for o in O]
        O = np.concatenate([O, np.array(col, np.float32)[:, None]], 1)
    return O


def lane_slot_features(env):
    """Ego-centric, role-free: for the left / ego / right lane, the 2 nearest delivered actors ahead and the nearest
    behind ([present, dx, dy from that lane's centre, dvx, dvy, heading, accel over 0.5 s, length, width]); the room
    the permitted corridor leaves on each side of the ego lane; the trigger and the time since it fired."""
    e, W = env.ego, env.W
    k = round(e.position[1] / W)
    lo, hi = env.scene["bounds"](e.position[0])
    f = [e.speed, e.position[1] - k * W, e.heading, env.ego_yaw_rate, env.ego_acc,
         (k * W - W / 2) - lo, hi - (k * W + W / 2), *hazard_state(env)]
    lanes = {"L": [], "E": [], "R": []}
    for n, b in env.bodies.items():
        if not env.spec.deliver(env, n):
            continue
        off = round(b.position[1] / W) - k
        if abs(off) <= 1:
            lanes["LER"[off + 1]].append((b.position[0] - e.position[0], n, b))
    for lane in "LER":
        ahead = sorted((x for x in lanes[lane] if x[0] >= 0), key=lambda x: x[0])
        behind = sorted((x for x in lanes[lane] if x[0] < 0), key=lambda x: -x[0])
        for kind, items, j in (("ahead", ahead, 0), ("ahead", ahead, 1), ("behind", behind, 0)):
            if j < len(items):
                dx, n, b = items[j]
                h = env.speed_hist[n]
                a = (h[-1] - h[-11]) / 0.5 if len(h) > 10 else 0.0
                v = b.velocity - e.velocity
                f += [1.0, dx, b.position[1] - (k + "LER".index(lane) - 1) * W, v[0], v[1], b.heading, a,
                      b.LENGTH, b.WIDTH]
            else:
                f += [0.0, D_FAR if kind == "ahead" else -D_FAR, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return np.array(f + road_features(env), dtype=np.float32)


# ---------------- the family experts ----------------
def build_mode(ns, env, mode, cfg, start_at=None, choose=None):
    W = env.W
    home = int(round(env.scene["home_y"] / W))
    M = env.envelope.M
    trig = trigger_of(env)
    if ">" in mode:                                     # stage 1, then the hand-back chosen at the hand-back point
        cfg = full_cfg(composite(mode), cfg)
        s1, hb = mode.split(">")
        stage1 = build_mode(ns, env, s1, {k: v for k, v in cfg.items() if not k.startswith("h_")}, start_at)
        return HandBack(ns, env, stage1, choose or (lambda o: (hb, cfg)))
    if mode == "keep":
        return on_intrusion(ns, ns["cruise"](name="keep"), trig=trig)
    cfg = full_cfg(mode, cfg)
    if mode == "brake":
        inner = brake_in_lane(ns, env, cfg.get("b_frac", 0.9), cfg.get("off", 0.0))
        delay = 0.0
    elif mode == "brake2":
        inner = brake_in_lane(ns, env, cfg["b1"], 0.0, b_then=(cfg["t1"], cfg["b2"]))
        delay = 0.0
    elif mode in ("left", "right"):
        lane = home - 1 if mode == "left" else home + 1
        y_off, delay = cfg["y_off"], cfg["t_start"]
        inner = ns["EscapeAdjacent"](lane, y=lane * W + y_off * np.sign(home - lane), mu_g=M,
                                     name=f"escape_{mode}", A_lat=cfg["A_lat"], b_evasive=cfg["b_evasive"])
    elif mode in ("left_brake", "right_brake"):
        lane = home - 1 if mode == "left_brake" else home + 1
        y_off, delay = cfg["y_off"], cfg["t_start"]
        esc = ns["EscapeAdjacent"](lane, y=lane * W + y_off * np.sign(home - lane), mu_g=M,
                                   name=f"escape_{mode}", A_lat=cfg["A_lat"], b_evasive=cfg["b_evasive"])
        inner = then_brake(ns, env, esc, lane * W, trig)
    elif mode in ("shoulder", "swerve_left"):            # partial shift right onto the shoulder / through the
        dy = cfg["dy"] if mode == "shoulder" else -cfg["dy"]      # opposing lane, past the obstacle, and back
        inner = ns["SwerveReturn"](dy, A_lat=cfg["A_lat"], margin=cfg["margin"], acc=cfg["acc"], mu_g=M, name=mode)
        delay = 0.0
    elif mode in ("merge_front", "merge_back"):
        inner = merge_slot(ns, env, home + getattr(env.spec, "merge_dir", 1), cfg, M, mode)
        delay = 0.0
    elif mode == "watch_merge":
        inner = watch_merge(ns, env, home + getattr(env.spec, "merge_dir", 1), cfg, M)
        delay = 0.0
    elif mode == "memory_left":
        inner = memory_change(ns, env, home - 1, cfg, M)
        delay = 0.0
    elif mode == "stop_end":
        inner = stop_before(ns, env, env.scene["end_x"], cfg)
        delay = 0.0
    elif mode == "brake_free":
        inner = brake_then_free(ns, env, cfg, trig)
        delay = 0.0
    elif mode == "shift_pass":
        inner, delay = shift_pass(ns, env, cfg), 0.0
    elif mode == "overtake":
        inner, delay = overtake(ns, env, cfg, M), 0.0
    elif mode == "wait_overtake":
        inner, delay = wait_overtake(ns, env, cfg, M), 0.0
    elif mode == "shift_left":                           # held until the hand-back takes over (margin: never back)
        inner = ns["SwerveReturn"](-cfg["dy"], A_lat=cfg["A_lat"], margin=1e9, acc=cfg["acc"], mu_g=M, name=mode)
        delay = 0.0
    else:
        raise ValueError(mode)
    # RC12: the scene's `delay` (s) after the trigger, lane and speed held, before any family starts
    return on_intrusion(ns, inner, delay + getattr(env, "params", {}).get("delay", 0.0), start_at, trig)


def HandBack(ns, env, stage1, choose):
    """stage1 (a family behind the trigger) until the hand-back point (handback_seen / T_HB_MAX, the same test as
    handback_state on the same delivered history), then the hand-back that choose(o) -> (name, cfg) returns there,
    built from the state at that point, with its clock starting there."""
    W = env.W
    atrig = ACTOR_TRIGGERS[getattr(env.spec, "trigger", "intrusion")]

    class _HandBack(ns["Policy"]):
        name = stage1.name

        def reset(self):
            stage1.reset()
            self.lane_y, self.t_det, self.actor, self.v_det, self.t_hb, self.hb, self.choice = (None,) * 7

        def act(self, o):
            if self.lane_y is None:
                self.lane_y = round(o.ego["y"] / W) * W
            if self.t_det is None:
                n = atrig(ns, o, self.lane_y)
                if n is not None:
                    self.t_det, self.actor, self.v_det = o.t, n, o.ego["v"]
            if self.t_det is not None and self.hb is None and (
                    o.t - self.t_det >= T_HB_MAX - 1e-6 or handback_seen(ns, o, self.lane_y, self.actor, W)):
                self.t_hb, self.choice = o.t, choose(o)
                self.hb = handback_policy(ns, env, self.choice[0], self.choice[1], o, self.lane_y, self.v_det,
                                          actor=self.actor)
                self.hb.reset()
            if self.hb is None:
                return stage1.act(o)
            return self.hb.act(ns["Obs"](o.t - self.t_hb, o.ego, o.actors, o.history, o.lanes))
    return _HandBack()


def handback_policy(ns, env, hb, cfg, o0, lane_y, v_ref, actor=None):
    """The hand-back behaviour `hb` (HB_SPACE), built at the hand-back point o0. Commands are capped by the friction
    circle (0.9 of the envelope's combined budget), so the demonstrations stay inside the envelope. actor = the one
    that fired the trigger: return / edge go back only once past it (whatever its speed at the hand-back point)."""
    W, e0 = env.W, o0.ego
    side = math.copysign(1.0, e0["y"] - lane_y) if abs(e0["y"] - lane_y) > 0.05 else 1.0
    # where the ego holds before it returns / while it stays: an in-lane offset is kept, otherwise its lane's centre
    y_hold = e0["y"] if abs(e0["y"] - lane_y) < 0.5 * W else round(e0["y"] / W) * W
    def cap(a, v, steer):
        _, B, _, Mc = env.envelope.caps(v)
        lim = math.sqrt(max((0.9 * Mc) ** 2 - ns["lat_accel"](v, steer) ** 2, 0.0))
        return float(np.clip(a, -min(B, lim), lim))

    def lead_gap(o, y_t):
        e, s, dv = o.ego, 1e3, 0.0
        for a in o.actors.values():
            if abs(a["y"] - y_t) < 2.3 and a["x"] > e["x"] and math.cos(a["psi"]) > 0:
                gap = a["x"] - e["x"] - (ns["extent"](a)[0] + 2 * ns["HALF_L"]) / 2
                if gap < s:
                    s, dv = gap, e["v"] - a["v"] * math.cos(a["psi"])
        return s, dv

    def steer_to(e, y_t, A):
        if abs(y_t - e["y"]) > 1.5:
            return ns["evade_y"](e["y"], e["psi"], e["v"], y_t, A_lat=A)
        return ns["track_y"](e["x"], e["y"], e["psi"], e["v"], y_t)

    if hb == "stop":
        class Stop(ns["Policy"]):
            name = "stop"

            def act(self, o):
                e = o.ego
                steer = steer_to(e, y_hold, 3.0)
                _, B, _, _ = env.envelope.caps(e["v"])
                ahead = [a["x"] - ns["extent"](a)[0] / 2 - (e["x"] + ns["HALF_L"]) for a in o.actors.values()
                         if abs(a["y"] - y_hold) < 0.5 * W and a["x"] > e["x"] and a["v"] < 1.0]
                d = min(ahead) - cfg["h_gap"] if ahead else 0.0
                b = cfg["h_b"] * B if d <= 0.5 else min(e["v"] ** 2 / (2 * d), cfg["h_b"] * B)
                return cap(-b if e["v"] > 0.05 else 0.0, e["v"], steer), steer
        return Stop()

    if hb == "stay":
        lane = [a for a in o0.actors.values() if abs(a["y"] - y_hold) < 0.5 * W and math.cos(a["psi"]) > 0]
        behind = [a for a in lane if a["x"] < e0["x"]]
        ref = max(behind, key=lambda a: a["x"]) if behind else (min(lane, key=lambda a: a["x"]) if lane else None)
        v_lane = ref["v"] * math.cos(ref["psi"]) if ref else e0["v"]
        v_set = max(e0["v"] + cfg["h_match"] * (v_lane - e0["v"]), 1.0)

        class Stay(ns["Policy"]):
            name = "stay"

            def act(self, o):
                e = o.ego
                steer = steer_to(e, y_hold, 3.0)
                s, dv = lead_gap(o, y_hold)
                a = ns["idm_accel"](e["v"], v_set, s, dv, T=cfg["h_T"], s0=3.0, a=cfg["h_a"], b=3.0)
                return cap(a, e["v"], steer), steer
        return Stay()

    if hb == "away":
        v_set = v_ref + cfg["h_dv"]

        class Away(ns["Policy"]):
            name = "away"

            def act(self, o):
                e = o.ego
                steer = steer_to(e, y_hold, 3.0)
                s, dv = lead_gap(o, y_hold)
                a = ns["idm_accel"](e["v"], v_set, s, dv, T=1.0, s0=3.0, a=cfg["h_a"], b=3.0)
                return cap(a, e["v"], steer), steer
        return Away()

    y_ret = lane_y - side * cfg["h_y"]                   # return / edge

    class Return(ns["Policy"]):
        name = hb

        def reset(self):
            self.back = False

        def act(self, o):
            e = o.ego
            if not self.back:
                home = [a for a in o.actors.values() if abs(a["y"] - lane_y) < 0.5 * W + 0.3]
                slow = [a for n, a in o.actors.items() if abs(a["y"] - lane_y) < 0.5 * W + 0.3
                        and (n == actor or a["v"] * math.cos(a["psi"]) < 0.5 * e["v"])]
                past = all(e["x"] - ns["HALF_L"] - (a["x"] + ns["extent"](a)[0] / 2) > cfg["h_margin"] for a in slow)
                beside = any(abs(a["x"] - e["x"]) < ns["extent"](a)[0] / 2 + ns["HALF_L"] + 4.0 for a in home
                             if a not in slow) and abs(e["y"] - lane_y) > 0.5 * W
                self.back = past and not beside
            y_t = y_ret if self.back else y_hold
            steer = steer_to(e, y_t, cfg["h_A"])
            s, dv = lead_gap(o, y_t)
            a = ns["idm_accel"](e["v"], max(v_ref, 1.0), s, dv, T=1.2, s0=3.0, a=cfg["h_a"], b=3.0)
            return cap(a, e["v"], steer), steer
    return Return()


def brake_in_lane(ns, env, b_frac, off, y_lane=None, b_then=None):
    """Emergency braking at b_frac of the usable braking B(v) (friction circle shared with the lane tracking), to a
    stop, tracking the lane the ego is nearest to when it starts (or y_lane), `off` m away from the actor moving
    fastest towards that lane. b_then = (t1, b2): b_frac for the first t1 s, b2 from then on."""
    W = env.W

    class BrakeInLane(ns["Policy"]):
        name = f"brake(b={b_frac},off={off})" if b_then is None else f"brake2(b1={b_frac},t1={b_then[0]},b2={b_then[1]})"

        def reset(self):
            self.y_t = None

        def act(self, o):
            e = o.ego
            if self.y_t is None:                       # shift away from the actor moving fastest towards the lane
                lane = round(e["y"] / W) * W if y_lane is None else y_lane
                side = 0.0
                if len(o.history) >= 2:
                    (t1, s1), (t2, s2) = o.history[-2:]
                    vy = {n: (b["y"] - s1[n]["y"]) / (t2 - t1) for n, b in s2.items() if n in s1}
                    movers = [(v * math.copysign(1, lane - s2[n]["y"]), n) for n, v in vy.items()
                              if abs(s2[n]["y"] - lane) < 1.5 * W]
                    if movers and max(movers)[0] > 0.3:
                        side = math.copysign(1, s2[max(movers)[1]]["y"] - lane)
                self.y_t = lane - side * off
            steer = ns["track_y"](e["x"], e["y"], e["psi"], e["v"], self.y_t)
            _, B, _, Mc = env.envelope.caps(e["v"])
            lat = ns["lat_accel"](e["v"], steer)
            f = b_frac if b_then is None or o.t < b_then[0] else b_then[1]
            b = min(f * B, math.sqrt(max((0.9 * Mc) ** 2 - lat ** 2, 0.0)))
            return -b, steer
    return BrakeInLane()


def then_brake(ns, env, esc, y_lane, trig=None):
    """Lane change ready to brake: `esc` (EscapeAdjacent; its b_evasive is how much speed the ego sheds on the way)
    until an intrusion into the target lane is observed (the same trigger as the family's, on that lane: e.g. the
    spun car sliding on into it), then emergency braking in the target lane. The switch is the ego's own observable
    decision, inside the family: nothing about the scene's future is used."""
    seen = trig or intrusion_seen

    class ThenBrake(ns["Policy"]):
        name = f"{esc.name}+brake"

        def reset(self):
            esc.reset()
            self.brk, self.t_sw = brake_in_lane(ns, env, 0.9, 0.0, y_lane), None
            self.brk.reset()

        def act(self, o):
            if self.t_sw is None and seen(ns, o, y_lane):
                self.t_sw = o.t
            if self.t_sw is None:
                return esc.act(o)
            return self.brk.act(ns["Obs"](o.t - self.t_sw, o.ego, o.actors, o.history, o.lanes))
    return ThenBrake()


def brake_then_free(ns, env, cfg, trig=None):
    """Contingent escape (RC10): emergency braking in lane at cfg b_frac, watching the actor that fired the trigger;
    the first time it is observed more than d_seen m off the ego lane's centre (the side it is sliding to), change into
    the adjacent lane on the other side (EscapeAdjacent: A_lat, b_evasive, aim y_off m short of that lane's centre).
    The decision point is that observation, nothing about the scene's future is used; never seen leaving the lane, the
    ego keeps braking."""
    W, M = env.W, env.envelope.M
    atrig = ACTOR_TRIGGERS[getattr(env.spec, "trigger", "intrusion")]

    class BrakeFree(ns["Policy"]):
        name = "brake_free"

        def reset(self):
            self.brk = brake_in_lane(ns, env, cfg["b_frac"], 0.0)
            self.brk.reset()
            self.esc, self.actor, self.lane_y, self.t_sw = None, None, None, None

        def act(self, o):
            if self.lane_y is None:
                self.lane_y = round(o.ego["y"] / W) * W
            if self.actor is None:
                self.actor = atrig(ns, o, self.lane_y)
            if self.esc is None and self.actor is not None:
                b = o.history[-1][1].get(self.actor)
                if b is not None and abs(b["y"] - self.lane_y) > cfg["d_seen"]:
                    home = round(self.lane_y / W)
                    lane = home - int(math.copysign(1, b["y"] - self.lane_y))
                    self.esc = ns["EscapeAdjacent"](lane, y=lane * W + cfg["y_off"] * np.sign(home - lane), mu_g=M,
                                                    name="brake_free", A_lat=cfg["A_lat"], b_evasive=cfg["b_evasive"])
                    self.esc.reset()
                    self.t_sw = o.t
            if self.esc is None:
                return self.brk.act(o)
            return self.esc.act(ns["Obs"](o.t - self.t_sw, o.ego, o.actors, o.history, o.lanes))
    return BrakeFree()


def watch_merge(ns, env, lane, cfg, M):
    """RC15's contingent merge (SETTINGS note): nudge, watch the adjacent car's response, then merge in front of or
    behind it. The adjacent car = the delivered actor in the target lane nearest the ego when the watch starts."""
    W = env.W

    class WatchMerge(ns["Policy"]):
        name = "watch_merge"

        def reset(self):
            self.v0, self.adj, self.go, self.t_go, self.choice = None, None, None, None, None

        def act(self, o):
            e = o.ego
            if self.v0 is None:
                self.v0 = e["v"]
                tgt = {n: b for n, b in o.actors.items() if abs(b["y"] - lane * W) < W / 2}
                self.adj = min(tgt, key=lambda n: abs(tgt[n]["x"] - e["x"])) if tgt else None
            if self.go is None:
                a_adj = 0.0
                if self.adj is not None and len(o.history) > 10:
                    (t1, s1), (t2, s2) = o.history[-11], o.history[-1]
                    if self.adj in s1 and self.adj in s2:
                        a_adj = (s2[self.adj]["v"] - s1[self.adj]["v"]) / (t2 - t1)
                choice = None
                if a_adj < -cfg["a_thr"]:
                    choice = "front"                            # it yields
                elif a_adj > cfg["a_thr"]:
                    choice = "back"                             # it matches
                elif o.t >= cfg["t_watch"]:
                    b = o.history[-1][1].get(self.adj) if self.adj else None
                    choice = "front" if b is None or b["x"] + b["length"] / 2 < e["x"] - ns["HALF_L"] else "back"
                if choice is not None:
                    c = dict(cfg, a=cfg["a_f"] if choice == "front" else cfg["a_b"])
                    self.choice, self.go = choice, merge_slot(ns, env, lane, c, M, f"watch_merge_{choice}")
                    self.go.reset()
                    self.t_go = o.t
                else:                                           # the nudge: towards the target lane, holding the speed
                    home = round(e["y"] / W)
                    steer = ns["track_y"](e["x"], e["y"], e["psi"], e["v"],
                                          home * W + cfg["nudge_y"] * np.sign(lane - home))
                    return 0.5 * (self.v0 - e["v"]), steer
            return self.go.act(ns["Obs"](o.t - self.t_go, o.ego, o.actors, o.history, o.lanes))
    return WatchMerge()


def memory_change(ns, env, lane, cfg, M):
    """RC14's memory-aware lane change (SETTINGS note). The memory is the observed history (from spec.history_from):
    the last snapshot in which each actor was delivered."""
    W = env.W

    class MemoryChange(ns["Policy"]):
        name = "memory_left"

        def reset(self):
            self.esc, self.wait_for, self.t0, self.t_go = None, False, None, None

        def act(self, o):
            e = o.ego
            if self.t0 is None:                                     # the decision: what is remembered, not in view
                self.t0 = o.t
                last, v_max = {}, {}                                # last sighting; fastest seen in the last 3 s
                for t, snap in o.history:
                    last.update(snap)
                    for n, b in snap.items():
                        if o.t - t <= 3.0:
                            v_max[n] = max(v_max.get(n, 0.0), b["v"])
                now = o.history[-1][1]
                hid = [n for n, b in last.items() if n not in now and abs(b["y"] - lane * W) < 0.75 * W
                       and b["x"] < e["x"] and v_max.get(n, b["v"]) - e["v"] > cfg["dv_thr"]]
                self.wait_for = hid[0] if hid else None
            if self.esc is None:
                go = self.wait_for is None or o.t - self.t0 >= cfg["t_wait"]
                b = o.history[-1][1].get(self.wait_for) if self.wait_for else None
                if b is not None and b["x"] - b["length"] / 2 > e["x"] + ns["HALF_L"] + cfg["g_pass"]:
                    go = True                                       # seen again, and past
                if not go:
                    steer = ns["track_y"](e["x"], e["y"], e["psi"], e["v"], round(e["y"] / W) * W)
                    _, B, _, Mc = env.envelope.caps(e["v"])
                    lat = ns["lat_accel"](e["v"], steer)
                    return -min(cfg["b_wait"] * B, math.sqrt(max((0.9 * Mc) ** 2 - lat ** 2, 0.0))), steer
                home = round(e["y"] / W)
                self.esc = ns["EscapeAdjacent"](lane, y=lane * W + cfg["y_off"] * np.sign(home - lane), mu_g=M,
                                                name="memory_left", A_lat=cfg["A_lat"], b_evasive=cfg["b_evasive"],
                                                v_set=max(e["v"], 5.0))
                self.esc.reset()
                self.t_go = o.t
            return self.esc.act(ns["Obs"](o.t - self.t_go, o.ego, o.actors, o.history, o.lanes))
    return MemoryChange()


def stop_before(ns, env, x_end, cfg):
    """Protective stop d_gap m before x_end (the lane end) in the own lane: the deceleration that stops there, capped
    at b_max of the usable braking."""
    W = env.W

    class StopBefore(ns["Policy"]):
        name = "stop_end"

        def act(self, o):
            e = o.ego
            steer = ns["track_y"](e["x"], e["y"], e["psi"], e["v"], round(e["y"] / W) * W)
            d = x_end - cfg["d_gap"] - (e["x"] + ns["HALF_L"])
            _, B, _, _ = env.envelope.caps(e["v"])
            b = B * cfg["b_max"] if d <= 0.5 else min(e["v"] ** 2 / (2 * d), B * cfg["b_max"])
            return (-b if e["v"] > 0.1 else -1.0), steer
    return StopBefore()


def stopped_ahead(ns, o, home_y, W):
    """The nearest stopped object seen in the home lane ahead (now or in the remembered history): its dict or None."""
    e = o.ego
    for snap in [o.actors] + [s for _, s in reversed(o.history)]:
        obst = [b for b in snap.values() if b["x"] > e["x"] and abs(b["y"] - home_y) < W / 2 and b["v"] < 0.5]
        if obst:
            return min(obst, key=lambda b: b["x"])
    return None


def shift_pass(ns, env, cfg):
    """RC16's passage (SETTINGS note): a quintic shift of dy towards spec.pass_dir ending x_ready m before the
    obstacle, held past it, then back; speed down to v_pass first."""
    W, sgn = env.W, getattr(env.spec, "pass_dir", -1)
    HL = ns["HALF_L"]

    class ShiftPass(ns["Policy"]):
        name = "shift_pass"

        def reset(self):
            self.home, self.v0, self.plan = None, None, None

        def act(self, o):
            e = o.ego
            if self.home is None:
                self.home, self.v0 = round(e["y"] / W) * W, e["v"]
            ob = stopped_ahead(ns, o, self.home, W)
            if self.plan is None and ob is not None:
                x_rear, x_front = ob["x"] - ob["length"] / 2, ob["x"] + ob["length"] / 2
                x_done = x_rear - cfg["x_ready"]
                x_s = min(x_done - cfg["D"], e["x"] + HL)
                self.plan = (x_s, max(x_done - x_s, 3.0), x_front + 3.0)
            y_t, v_t = self.home, self.v0
            if self.plan is not None:
                x_s, D, x_back = self.plan
                look = e["x"] + HL + max(e["v"], 2.0) * 0.8
                u_out = ns["quintic"]((look - x_s) / D)
                u_back = ns["quintic"]((e["x"] - HL - x_back) / cfg["D"])
                y_t = self.home + sgn * cfg["dy"] * (u_out - u_back)
                v_t = min(self.v0, cfg["v_pass"]) if e["x"] - HL < x_back else self.v0
            a = -cfg["b"] if e["v"] > v_t + 0.3 else float(np.clip(0.8 * (v_t - e["v"]), -cfg["b"], 1.0))
            return a, ns["track_y"](e["x"], e["y"], e["psi"], e["v"], y_t, Tp=0.5, a_cap=6.0)
    return ShiftPass()


def overtake(ns, env, cfg, M, v_cap=None):
    """RC08's overtake (SETTINGS note): SwerveReturn into the opposing lane, speed capped at v_cap (default: the speed
    at the start)."""
    sw = ns["SwerveReturn"](-cfg["dy"], A_lat=cfg["A_lat"], margin=cfg["margin"], acc=cfg["acc"], mu_g=M,
                            name="overtake")

    class Overtake(ns["Policy"]):
        name = "overtake"

        def reset(self):
            sw.reset()
            self.v_cap = v_cap

        def act(self, o):
            if self.v_cap is None:
                self.v_cap = max(o.ego["v"], 8.0)
            a, steer = sw.act(o)
            return min(a, 0.8 * (self.v_cap - o.ego["v"])), steer
    return Overtake()


def wait_overtake(ns, env, cfg, M):
    """RC08's protected wait, then the overtake once a big enough gap is seen (SETTINGS note)."""
    W = env.W
    HL = ns["HALF_L"]

    class WaitOvertake(ns["Policy"]):
        name = "wait_overtake"

        def reset(self):
            self.home, self.v0, self.go, self.t_go = None, None, None, None

        def act(self, o):
            e = o.ego
            if self.home is None:
                self.home, self.v0 = round(e["y"] / W) * W, e["v"]
            if self.go is not None:
                return self.go.act(ns["Obs"](o.t - self.t_go, o.ego, o.actors, o.history, o.lanes))
            ob = stopped_ahead(ns, o, self.home, W)
            moving = [b for b in o.actors.values() if b["x"] > e["x"] and abs(b["y"] - self.home) < W / 2]
            if ob is None:                                       # it drove off: follow whatever is ahead in the lane
                lead = min(moving, key=lambda b: b["x"]) if moving else None
                a = 0.8 * (self.v0 - e["v"])
                if lead is not None:
                    gap = lead["x"] - lead["length"] / 2 - (e["x"] + HL)
                    a = min(a, 0.4 * (gap - 5.0 - 1.5 * e["v"]) + 0.8 * (lead["v"] - e["v"]))
                return float(np.clip(a, -4.0, 1.5)), ns["track_y"](e["x"], e["y"], e["psi"], e["v"], self.home)
            x_rear, x_front = ob["x"] - ob["length"] / 2, ob["x"] + ob["length"] / 2
            if e["v"] < 2.0:                                     # waiting: is the opposing lane free long enough?
                d_need = x_front + cfg["margin"] + 2 * HL - (e["x"] - HL)
                t_pass = math.sqrt(2.0 * max(d_need, 0.0) / cfg["a_go"]) + cfg["T_need"]
                y_opp = self.home - W
                onc = [b for b in o.actors.values() if abs(b["y"] - y_opp) < W / 2 and math.cos(b["psi"]) < 0
                       and b["x"] + b["length"] / 2 > e["x"] - HL]
                if all((b["x"] - b["length"] / 2 - (x_front + cfg["margin"])) / max(b["v"], 0.5) > t_pass for b in onc):
                    self.go = overtake(ns, env, dict(dy=W, A_lat=cfg["A_lat"], margin=cfg["margin"], acc=cfg["a_go"]),
                                       M, v_cap=self.v0)
                    self.go.reset()
                    self.t_go = o.t
                    return self.go.act(ns["Obs"](0.0, o.ego, o.actors, o.history, o.lanes))
            d = x_rear - cfg["d_gap"] - (e["x"] + HL)
            _, B, _, _ = env.envelope.caps(e["v"])
            b = B * cfg["b_max"] if d <= 0.5 else min(e["v"] ** 2 / (2 * d), B * cfg["b_max"])
            steer = ns["track_y"](e["x"], e["y"], e["psi"], e["v"], self.home - cfg["y_peek"])
            return (-b if e["v"] > 0.1 else -1.0), steer
    return WaitOvertake()


def merge_slot(ns, env, lane, cfg, M, name):
    """Change into `lane` through a slot made by speed: accelerate (a > 0, in front of the traffic there) or brake
    (a < 0, behind it) in the own lane until no delivered actor of the target lane is within g_lead m ahead of the
    ego's front or g_lag m behind its rear, then the evasive move (EscapeAdjacent, keeping the speed reached; aim
    y_off m short of the target lane's centre against the overshoot, braking up to b_ev during it)."""
    W = env.W

    class MergeSlot(ns["Policy"]):
        def reset(self):
            self.esc, self.t_go = None, None

        def act(self, o):
            e = o.ego
            if self.esc is None:
                hl = ns["HALF_L"]
                tgt = [b for b in o.actors.values() if abs(b["y"] - lane * W) < W / 2 and math.cos(b["psi"]) > 0]
                free = all(b["x"] - b["length"] / 2 >= e["x"] + hl + cfg["g_lead"] or
                           b["x"] + b["length"] / 2 <= e["x"] - hl - cfg["g_lag"] for b in tgt)
                if free or e["v"] < 1.0:
                    home = round(e["y"] / W)
                    self.esc = ns["EscapeAdjacent"](lane, y=lane * W + cfg["y_off"] * np.sign(home - lane), mu_g=M,
                                                    name=name, A_lat=cfg["A_lat"], b_evasive=cfg["b_ev"],
                                                    v_set=max(e["v"], 5.0))
                    self.esc.reset()
                    self.t_go = o.t
                else:
                    steer = ns["track_y"](e["x"], e["y"], e["psi"], e["v"], round(e["y"] / W) * W)
                    _, B, _, Mc = env.envelope.caps(e["v"])
                    a = cfg["a"] if cfg["a"] > 0 else -min(-cfg["a"], B)
                    lat = ns["lat_accel"](e["v"], steer)
                    cap = math.sqrt(max((0.9 * Mc) ** 2 - lat ** 2, 0.0))
                    return max(min(a, cap), -cap), steer
            return self.esc.act(ns["Obs"](o.t - self.t_go, o.ego, o.actors, o.history, o.lanes))
    return MergeSlot()


def on_intrusion(ns, inner, delay=0.0, start_at=None, trig=None):
    """The notebook's OnHazard wrapper (hold lane and speed, then run `inner` with its clock at its start), with
    intrusion_seen as the trigger; the family starts `delay` s after the trigger, or at the episode time `start_at`
    (a fixed decision point, trigger or not)."""
    class OnIntrusion(ns["OnHazard"]):
        def act(self, o):
            if self.lane_y is None:
                self.lane_y = round(o.ego["y"] / ns["LANE_W"]) * ns["LANE_W"]
            if self.t_det is None and (trig or intrusion_seen)(ns, o, self.lane_y):
                self.t_det = o.t
            t0 = start_at if start_at is not None else (None if self.t_det is None else self.t_det + delay)
            if t0 is None or o.t < t0 - 1e-6:
                return self.hold.act(o)
            return self.inner.act(ns["Obs"](o.t - t0, o.ego, o.actors, o.history, o.lanes))
    return OnIntrusion(inner)


class ModeExpert:
    """One setting of one family as an env policy (before t0 the notebook's cruise prefix); act() -> envelope z.
    choose: for a composite family, choose(o) -> (hand-back, cfg) at the hand-back point (default: its own)."""
    def __init__(self, env, mode, cfg=None, start_at=None, choose=None):
        self.ns, self.env, self.mode, self.cfg, self.start_at = notebook_ns(), env, mode, dict(cfg or {}), start_at
        self.choose = choose
        self.ns["LANE_W"] = env.W
        self.prefix = self.ns["cruise"]()

    def reset(self):
        self.ns["LANE_W"] = self.env.W
        self.pol = build_mode(self.ns, self.env, self.mode, self.cfg, self.start_at, self.choose)
        self.pol.reset()
        self.prefix.reset()

    @property
    def t_det(self):
        return getattr(self.pol, "t_det", None)

    @property
    def t_hb(self):
        return getattr(self.pol, "t_hb", None)

    def act(self, obs=None):
        env, ns = self.env, self.ns
        e = env.ego
        ego = dict(x=float(e.position[0]), y=float(e.position[1]), v=float(e.speed), psi=float(e.heading),
                   length=5.0, width=2.0, steer=0.0, acc=float(env.ego_acc))
        if env.t < -1e-6:
            a, d = self.prefix.act(ns["Obs"](t=env.t + env.spec.prefix_T, ego=ego, actors=env.snaps[-1][1],
                                             history=env.snaps, lanes=3))
        else:
            hist = [s for s in env.snaps if s[0] >= hist_start(env)] or env.snaps[-1:]
            a, d = self.pol.act(ns["Obs"](t=env.t, ego=ego, actors=hist[-1][1], history=hist, lanes=3))
        r = e.speed * math.sin(math.atan(0.5 * math.tan(d))) / 2.5           # HighwayEnv bicycle yaw rate
        return env.envelope.inverse(a, max(e.speed, 3.0) * r, e.speed)
