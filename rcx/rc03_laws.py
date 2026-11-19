"""RC03 (spun car, lateral escape): scripted actor laws shared verbatim by the CARLA scene and the surrogate (pure Python, no CARLA).
Frame: s forward from the ego centre at the cue (rt = 0), t lateral (right positive, lane centre of the ego lane = 0). `side` = +1: the crash is in the lane to the ego's right.
No law reacts to the ego: actors are scripted in time, so CARLA and the surrogate see exactly the same traffic."""
import math

HL_VICTIM, HW_VICTIM, HL_ESC, HW_ESC, HL_REAR, HW_REAR = 2.4, 1.0, 2.3, 0.95, 2.2, 0.95
DEFAULTS = dict(side=1, crash=1, v_ego=18.0, d0=45.0, intrude=1.0, psi_rest=0.8, T_slide=2.0, v_post=16.0,
                v_esc=18.0, esc_gap=35.0, gap_center=0.0, lead_brake=0.0, lead_react=0.7, rear=0, rear_gap=20.0, lane_w=3.5)


def _sm(x): x = min(max(x, 0.0), 1.0); return 0.5 * (1.0 - math.cos(math.pi * x))


def victim(p, rt):
    """(s, t, yaw, speed) of the victim's centre. Before the cue it drives straight in its own lane; after it slides to rest at depth `intrude` (crash=1) or drives on (crash=0)."""
    W, sd = p["lane_w"], p["side"]
    if not p["crash"] or rt <= 0.0:
        v = p["v_post"]; return p["d0"] + v * rt, sd * W, 0.0, v
    T = p["T_slide"]; x = min(rt / T, 1.0); v = p["v_post"] * (1.0 - x)
    s = p["d0"] + p["v_post"] * T * (x - 0.5 * x * x)
    t = sd * W + (sd * W * (1.0 - p["intrude"]) - sd * W) * _sm(x)
    return s, t, -sd * p["psi_rest"] * _sm(x), v


def esc_leader(p, rt):
    """(s, t, speed) of the escape-lane car ahead of the gap; brakes at `lead_brake` m/s^2 from `lead_react` after the cue (hidden)."""
    s0 = p["gap_center"] + 0.5 * p["esc_gap"] + HL_ESC; ve = p["v_esc"]; tb = p["lead_react"]; a = p["lead_brake"]
    if rt <= tb or a <= 0.0: return s0 + ve * rt, -p["side"] * p["lane_w"], ve
    td = min(rt - tb, ve / a); return s0 + ve * rt - 0.5 * a * td * td, -p["side"] * p["lane_w"], max(0.0, ve - a * (rt - tb))


def esc_follower(p, rt):
    return p["gap_center"] - 0.5 * p["esc_gap"] - HL_ESC + p["v_esc"] * rt, -p["side"] * p["lane_w"], p["v_esc"]


def rear_car(p, rt):
    """Tailgater in the ego lane at constant speed (present only if rear=1)."""
    return -p["rear_gap"] - HL_REAR + p["v_ego"] * rt, 0.0, p["v_ego"]


def still_closing(gap, v, dt_lat, min_ttc=2.0):
    """End-of-run check shared by the surrogate and the CARLA scene: the ego is still moving and, in the lateral overlap of the victim, would reach it within min_ttc seconds.
    gap = victim rear s - ego front s; dt_lat = |ego t - victim t| minus the half widths (<= 0.3 counts as overlapping)."""
    return v > 0.3 and dt_lat < 0.3 and 0.0 < gap and gap / v < min_ttc


def rear_closing(gap, closing, dt_lat, min_ttc=2.0):
    """Same end-of-run check for a car behind the ego: gap = ego rear s - that car's front s, closing = its speed - ego speed (> 0 means catching up)."""
    return closing > 0.3 and dt_lat < 0.3 and 0.0 < gap and gap / closing < min_ttc
