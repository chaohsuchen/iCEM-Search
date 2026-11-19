"""RC04 (staged slowing before a feasible lane change) and RC06 (yield, then use a time-varying gap): scripted actor laws shared verbatim by the CARLA scene and the surrogate
(pure Python, no CARLA). One scene serves both RCs: a stopped / broken-down obstacle ahead in the ego lane, a `side` vehicle (RC04: truck, RC06: passing car) in the neighbouring
lane around the ego, and a follower behind it that can close the gap.
Frame: s forward from the ego centre at the cue (rt = 0), t lateral (right positive, ego lane centre = 0). The other lane is at t = -side*lane_w (side = +1: it is on the ego's left),
the same convention as rc03_laws. No law reacts to the ego: actors are scripted in time, so CARLA and the surrogate see exactly the same traffic."""
import math

HL_OBS, HW_OBS, HL_FOL, HW_FOL = 2.3, 0.95, 2.3, 0.95
DEFAULTS = dict(side=1, v_ego=22.0, d_obs=55.0, obs_lane=0, v_side=22.0, side_pos=0.0, side_hl=2.4, side_hw=0.95, side_acc=0.0, side_react=0.8,
                fol=1, fol_gap=20.0, fol_react=0.8, fol_headway=1.2, lane_w=3.5)


def obstacle(p, rt):
    """Static obstacle: centre (s, t, speed). obs_lane 0 = in the ego lane (the hazard), 1 = in the other lane, 2 = on the shoulder (benign: nothing blocks the ego)."""
    t = {0: 0.0, 1: -p["side"] * p["lane_w"], 2: p["side"] * 3.9}[int(p["obs_lane"])]
    return p["d_obs"], t, 0.0


def side_vehicle(p, rt):
    """(s, t, speed) of the vehicle in the other lane: constant speed v_side, then from side_react after the cue a constant acceleration side_acc (negative = braking, i.e. it brakes
    on seeing the blockage; positive = it speeds up). Position side_pos = its centre relative to the ego at the cue."""
    v0, a, tr = p["v_side"], p["side_acc"], p["side_react"]; t = -p["side"] * p["lane_w"]
    if rt <= tr or a == 0.0: return p["side_pos"] + v0 * rt, t, v0
    td = rt - tr
    if a < 0.0: td = min(td, v0 / -a)
    return p["side_pos"] + v0 * rt + 0.5 * a * td * td, t, max(0.0, v0 + a * td)


def follower(p, rt):
    """(s, t, speed) of the car following the side vehicle at fol_gap (bumper to bumper at the cue). It keeps its speed until it sees the leader braking: after fol_react it
    decelerates at the same rate (headway keeping)."""
    s_l, _, _ = side_vehicle(p, 0.0)
    s0 = s_l - p["side_hl"] - p["fol_gap"] - HL_FOL; v0 = p["v_side"]; t = -p["side"] * p["lane_w"]
    a, tr = p["side_acc"], p["side_react"] + p["fol_react"]
    if a >= 0.0 or rt <= tr: return s0 + v0 * rt, t, v0
    td = min(rt - tr, v0 / -a)
    return s0 + v0 * rt + 0.5 * a * td * td, t, max(0.0, v0 + a * td)


def still_closing(gap, v, dt_lat, min_ttc=2.0):
    """The ego is still moving and, in the lateral overlap of the obstacle, would reach it within min_ttc seconds (gap = obstacle rear s - ego front s)."""
    return v > 0.3 and dt_lat < 0.3 and 0.0 < gap and gap / v < min_ttc


def rear_closing(gap, closing, dt_lat, min_ttc=2.0):
    """The same for a car behind the ego: gap = ego rear s - its front s, closing = its speed - ego speed."""
    return closing > 0.3 and dt_lat < 0.3 and 0.0 < gap and gap / closing < min_ttc
