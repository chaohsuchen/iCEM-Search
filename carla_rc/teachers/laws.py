"""Actor laws and kinematics ported from RC_scenarios_HighwayEnv_CARLA.ipynb, without HighwayEnv.

Laws keep the notebook's signature law(t, me, sim) -> accel | (accel, target_y), with t relative to t0, and read the
same attributes (me.position, me.speed, me.heading, me.velocity, me.LENGTH, me.WIDTH, sim.ego, sim.objs, sim.mem), so
the bodies below are drop-in stand-ins for HighwayEnv vehicles. Scenario frame: x along the ego lane, y to the right,
lane k centred at y = k * W (W = real lane width of the site).
"""
import math

import numpy as np
from shapely.geometry import Polygon as ShPolygon
from shapely.ops import nearest_points

DT = 0.05
EGO_L, EGO_W = 4.9, 2.1            # vehicle.lincoln.mkz_2020 footprint
HALF_L = EGO_L / 2


class Body:
    """A road user in the scenario frame. kind: 'law' (kinematic bicycle driven by a law), 'scripted' (pose from
    path(t)), 'static', 'ego' (state written from CARLA)."""
    def __init__(self, name, x, y, v=0.0, psi=0.0, length=4.8, width=2.0, kind="law", law=None, path=None, bp=None,
                 note="", color=None):
        self.name, self.kind, self.law, self.path, self.bp, self.note = name, kind, law, path, bp, note
        self.color = color                        # CARLA 'color' attribute ("r,g,b"); None = blueprint default
        self.position = np.array([x, y], dtype=float)
        self.speed, self.heading = float(v), float(psi)
        self.LENGTH, self.WIDTH = float(length), float(width)
        self.target_y = float(y)
        self._vel = None
        self.acc = 0.0

    @property
    def velocity(self):
        if self._vel is not None:
            return self._vel
        return self.speed * np.array([math.cos(self.heading), math.sin(self.heading)])

    def state(self):
        return dict(x=float(self.position[0]), y=float(self.position[1]), psi=float(self.heading), v=float(self.speed),
                    vx=float(self.velocity[0]), vy=float(self.velocity[1]), length=self.LENGTH, width=self.WIDTH)

    def place(self, t):
        """Scripted bodies: pose at time t (relative to t0)."""
        x, y, psi, vx, vy = self.path(t)
        self.position = np.array([x, y], dtype=float)
        self.heading, self.speed, self._vel = float(psi), float(math.hypot(vx, vy)), np.array([vx, vy], dtype=float)

    def bicycle_step(self, acc, steer, dt=DT):
        """HighwayEnv's kinematic bicycle (Vehicle.step) with the body's own length."""
        beta = math.atan(0.5 * math.tan(steer))
        v = self.speed
        self.position = self.position + v * np.array([math.cos(self.heading + beta), math.sin(self.heading + beta)]) * dt
        self.heading += v * math.sin(beta) / (self.LENGTH / 2) * dt
        self.speed = max(0.0, v + acc * dt)
        self.acc = acc


# ---------------- geometry ----------------
def quintic(s):
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def obb(x, y, psi, length, width):
    c, s = math.cos(psi), math.sin(psi)
    pts = [(-length / 2, -width / 2), (-length / 2, width / 2), (length / 2, width / 2), (length / 2, -width / 2)]
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in pts]


def signed_distance(p1, p2):
    a, b = ShPolygon(p1), ShPolygon(p2)
    if a.intersects(b):
        return -0.01 - math.sqrt(max(a.intersection(b).area, 0.0))
    pa, pb = nearest_points(a, b)
    return float(math.hypot(pb.x - pa.x, pb.y - pa.y))


def steer_for_yaw_rate(yaw_rate, v, half_l, steer_max=0.5):
    v = max(v, 0.5)
    beta = math.asin(float(np.clip(yaw_rate * half_l / v, -0.99, 0.99)))
    return float(np.clip(math.atan(2 * math.tan(beta)), -steer_max, steer_max))


def track_y(x, y, psi, v, y_target, half_l=2.5, Tp=0.8, k=2.5, steer_max=0.5, a_cap=5.0):
    """The notebook's look-ahead lateral controller towards y_target."""
    v_ = max(v, 0.5)
    psi_d = math.atan2(y_target - y, max(v, 2.0) * Tp)
    r = float(np.clip(k * (psi_d - psi), -a_cap / v_, a_cap / v_))
    return steer_for_yaw_rate(r, v, half_l, steer_max)


def speed_cmd(v, v_ref, k=1.0, a_lo=-3.0, a_hi=2.0):
    return float(np.clip(k * (v_ref - v), a_lo, a_hi))


# ---------------- scripted paths ----------------
def crash_path(x_c, y0, v_pre, v_post, a_slide, y_rest, psi_rest, T_slide):
    """Vehicle in a crash at t = 0: straight at v_pre before; after, slides from v_post to rest at a_slide while
    shifting y0 -> y_rest and yawing 0 -> psi_rest along a quintic over T_slide."""
    t_stop = v_post / a_slide

    def f(t):
        if t < 0:
            return x_c + v_pre * t, y0, 0.0, v_pre, 0.0
        s = min(t, t_stop)
        x = x_c + v_post * s - 0.5 * a_slide * s * s
        vx = v_post - a_slide * s
        u = min(t / T_slide, 1.0)
        y = y0 + (y_rest - y0) * quintic(u)
        vy = (y_rest - y0) * 30 * u * u * (1 - u) ** 2 / T_slide
        return x, y, psi_rest * quintic(u), vx, vy
    return f


def const_path(x0, y0, v, psi=0.0, t_brake=None, b=0.0):
    """Constant-velocity path along heading psi (psi = pi: oncoming traffic); optionally brakes at b from t_brake to
    rest. (x0, y0) = position at t = 0 (notebook const_path)."""
    c, s = math.cos(psi), math.sin(psi)

    def f(t):
        if t_brake is None or t <= t_brake:
            d, vv = v * t, v
        else:
            tb = min(t - t_brake, v / b)
            d, vv = v * t_brake + v * tb - 0.5 * b * tb * tb, v - b * tb
        return x0 + c * d, y0 + s * d, psi, c * vv, s * vv
    return f


def lane_change_path(x0, y0, v, y1, t_start, T_lc):
    """Constant speed v; lateral quintic y0 -> y1 over T_lc from t_start (x0 at t = 0; notebook lane_change_path)."""
    def f(t):
        u = min(max((t - t_start) / T_lc, 0.0), 1.0)
        vy = (y1 - y0) * 30 * u * u * (1 - u) ** 2 / T_lc
        return x0 + v * t, y0 + (y1 - y0) * quintic(u), math.atan2(vy, v), v, vy
    return f


def spin_path(x0, y0, v, b, t_div, T_slide, y_rest, psi_common, psi_rest):
    """RC11-S4 car losing control at t = 0 (notebook spin_path): skids at b to rest, yawing to psi_common along its lane
    until t_div (identical for both outcomes), then slides to y_rest rotating to psi_rest over T_slide. Before t = 0 it
    drives straight at v."""
    t_s = v / b

    def f(t):
        if t < 0:
            return x0 + v * t, y0, 0.0, v, 0.0
        tt = min(t, t_s)
        x, vx = x0 + v * tt - 0.5 * b * tt * tt, max(v - b * t, 0.0)
        u = min(max((t - t_div) / T_slide, 0.0), 1.0)
        y = y0 + (y_rest - y0) * quintic(u)
        vy = (y_rest - y0) * 30 * u * u * (1 - u) ** 2 / T_slide
        psi = psi_common * min(t / t_div, 1.0) + (psi_rest - psi_common) * quintic(u)
        return x, y, psi, vx, vy
    return f


def drift_path(x0, v, y0, vy, t_wake, mode, T_corr=0.9, vy_sleep=2.6, y_end=6.0, y_own=0.0):
    """RC11-S5 oncoming car (heading -x) drifting across the centre line at vy (notebook drift_path). 'W': the driver
    wakes at t_wake and steers back to its own lane centre y_own over T_corr; 'D': the drift steepens to vy_sleep until
    y_end. Identical until t_wake."""
    y_w = y0 + vy * t_wake

    def f(t):
        x = x0 - v * t
        if t <= t_wake:
            y, vyy = y0 + vy * t, vy
        elif mode == "W":
            u = min((t - t_wake) / T_corr, 1.0)
            y, vyy = y_w + (y_own - y_w) * quintic(u), (y_own - y_w) * 30 * u * u * (1 - u) ** 2 / T_corr
        else:
            y = min(y_w + vy_sleep * (t - t_wake), y_end)
            vyy = vy_sleep if y < y_end else 0.0
        return x, y, math.atan2(vyy, -v), -v, vyy
    return f


def child_path(x, y0, v_run, mode, y_slow, y_end, b=12.0, v_sprint=5.5, a_sprint=8.0):
    """RC11-S6 child running out toward -y at v_run from t = 0 (notebook child_path). At y_slow it reacts to the car:
    'F' stops hard and freezes in the ego lane, 'C' sprints on to y_end. Identical until y_slow."""
    t_slow = (y0 - y_slow) / v_run
    t_stop, t_acc = v_run / b, (v_sprint - v_run) / a_sprint

    def f(t):
        if t <= t_slow:
            y, vy = y0 - v_run * max(t, 0.0), (-v_run if t >= 0 else 0.0)
        elif mode == "F":
            tt = min(t - t_slow, t_stop)
            y, vy = y_slow - v_run * tt + 0.5 * b * tt * tt, -(v_run - b * tt) if tt < t_stop else 0.0
        else:
            tt = t - t_slow
            ta = min(tt, t_acc)
            y = y_slow - v_run * ta - 0.5 * a_sprint * ta * ta - v_sprint * max(tt - t_acc, 0.0)
            vy = -(v_run + a_sprint * ta)
            if y <= y_end:
                y, vy = y_end, 0.0
        return x, y, -math.pi / 2, 0.0, vy
    return f


def walk_across(x, y0, y1, t_start, v=1.4):
    """Pedestrian (notebook walk_across): stands at (x, y0), from t_start walks straight to y1 at v and stops there."""
    sgn = 1.0 if y1 > y0 else -1.0

    def f(t):
        d = min(max(t - t_start, 0.0) * v, abs(y1 - y0))
        moving = 0.0 < t - t_start and d < abs(y1 - y0)
        return x, y0 + sgn * d, sgn * math.pi / 2, 0.0, sgn * v if moving else 0.0
    return f


def crossing_path(x, y0, y1, t_start, T_cross, psi=-math.pi / 2):
    """Animal/pedestrian: stands at y0, then moves to y1 along a quintic over T_cross and stops there."""
    def f(t):
        u = min(max((t - t_start) / T_cross, 0.0), 1.0)
        return x, y0 + (y1 - y0) * quintic(u), psi, 0.0, (y1 - y0) * 30 * u * u * (1 - u) ** 2 / T_cross
    return f


# ---------------- traffic laws (notebook code, unchanged apart from the ego size) ----------------
def leader_of(me, sim):
    best = (None, None)
    for ob in [sim.ego] + list(sim.objs.values()):
        if ob is me:
            continue
        dx = ob.position[0] - me.position[0]
        if dx > 0 and abs(ob.position[1] - me.position[1]) < (ob.WIDTH + me.WIDTH) / 2 + 0.3:
            s = dx - (ob.LENGTH + me.LENGTH) / 2
            if best[0] is None or s < best[0]:
                best = (s, ob)
    return best


def idm_accel(v, v0, s, dv, T=1.5, s0=2.0, a=1.5, b=2.0):
    s_star = s0 + max(0.0, v * T + v * dv / (2 * math.sqrt(a * b)))
    return a * (1 - (v / max(v0, 0.1)) ** 4 - (s_star / max(s, 0.1)) ** 2)


def law_idm(v0, T=1.5, s0=2.0, a=1.5, b=2.0, react=0.0, b_max=6.0, rubberneck=None):
    """IDM on the leader perceived `react` s ago; rubberneck=(t_start, decel, v_floor) adds a braking episode."""
    def f(t, me, sim):
        buf = sim.mem.setdefault(("idm", id(me)), [])
        s, ld = leader_of(me, sim)
        buf.append((1e3, 0.0) if ld is None else (s, me.speed - ld.velocity[0]))
        s_p, dv_p = buf[max(0, len(buf) - 1 - int(round(react / DT)))]
        acc = idm_accel(me.speed, v0, s_p, dv_p, T, s0, a, b)
        if rubberneck is not None and t >= rubberneck[0] and me.speed > rubberneck[2]:
            acc = min(acc, -rubberneck[1])
        return float(np.clip(acc, -b_max, a))
    return f


def law_trigger(cond, a_after, v_cap=None, reaction=0.0, v_floor=0.0):
    def f(t, me, sim):
        key = ("trig", id(me))
        if key not in sim.mem and cond(t, me, sim):
            sim.mem[key] = t
        if key in sim.mem and t >= sim.mem[key] + reaction:
            if v_cap is not None and a_after > 0 and me.speed >= v_cap:
                return 0.0
            if a_after < 0 and me.speed <= v_floor:
                return 0.0
            return a_after
        return 0.0
    return f


def ego_ahead_in_lane(dist, half_w=2.0):
    def cond(t, me, sim):
        e = sim.ego
        dx = (e.position[0] - me.position[0]) * math.cos(me.heading)
        return abs(e.position[1] - me.position[1]) < half_w and 0.0 < dx < dist
    return cond


def law_planned_stop(t_plan, b_plan, notice, react, b_max, half_w=2.0):
    """RC07-S7 driver who has planned a stop behind a stopped car (notebook law_planned_stop): constant speed, then
    b_plan from t_plan; once the ego is in its lane ahead within `notice` m it brakes at b_max after `react` s."""
    seen = ego_ahead_in_lane(notice, half_w)

    def f(t, me, sim):
        if me.speed <= 0.05:
            return 0.0
        key = ("seen", id(me))
        if key not in sim.mem and seen(t, me, sim):
            sim.mem[key] = t
        if key in sim.mem and t >= sim.mem[key] + react:
            return -b_max
        return -b_plan if t >= t_plan else 0.0
    return f


def law_adjacent(mode, react, y_ego_lane, side, v_cruise, b_yield=2.5, dv_yield=3.0, dv_floor=6.0, a_match=2.5,
                 d_match=-2.0, T_match=1.2, nudge=0.4, lane_w=4.0):
    """RC15 adjacent driver (yield / hold / match), reacting to the actual ego state (notebook law_adjacent)."""
    v_floor = max(v_cruise - dv_floor, 0.0)

    def f(t, me, sim):
        e = sim.ego
        key = ("intent", id(me))
        if key not in sim.mem and side * (e.position[1] - y_ego_lane) > nudge:
            sim.mem[key] = t
        seen = key in sim.mem and t >= sim.mem[key] + react
        ego_in_lane = abs(e.position[1] - me.position[1]) < 0.5 * lane_w
        acc = speed_cmd(me.speed, v_cruise, a_lo=-1.0, a_hi=2.0 if sim.mem.get(("gave_up", id(me))) else 1.0)
        if seen and not ego_in_lane:
            if mode == "yield":
                rear_gap = (e.position[0] - HALF_L) - (me.position[0] + me.LENGTH / 2)
                if rear_gap < -3.0:
                    sim.mem[("gave_up", id(me))] = True
                if not sim.mem.get(("gave_up", id(me))):
                    acc = speed_cmd(me.speed, min(v_cruise, max(e.speed - dv_yield, v_floor)), a_lo=-b_yield, a_hi=0.5)
            elif mode == "match":
                acc = float(np.clip(0.8 * ((e.position[0] - me.position[0]) + d_match) + 1.2 * (e.speed - me.speed),
                                    0.0, a_match))
        s, ld = leader_of(me, sim)
        buf = sim.mem.setdefault(("idm_buf", id(me)), [])
        buf.append((1e3, 0.0) if ld is None else (s, me.speed - ld.velocity[0]))
        s_p, dv_p = buf[max(0, len(buf) - 1 - int(round(react / DT)))]
        if s_p < 1e3:
            T_f = T_match if (mode == "match" and seen) else 1.2
            cap = idm_accel(me.speed, 1e3, s_p, dv_p, T=T_f, s0=2.0, a=2.5, b=2.0)
            if dv_p < 0.0 and s_p > 2.0:
                cap = max(cap, -0.8)
            acc = min(acc, cap)
        return float(np.clip(acc, -6.0, a_match))
    return f


def law_lead_zipper(v0, lane_y, tgt_y, x_end, react=0.5, T=1.2, t_look=4.0, g_front=6.0, g_rear=8.0, T_lc=3.0,
                    lane_w=4.0):
    """RC15 vehicle ahead of the ego in the ending lane (notebook law_lead_zipper): IDM on vehicles (the lane end only
    within its last 30 m) and, once within t_look s of the end, it merges into the first gap next to it."""
    def f(t, me, sim):
        key = ("zip", id(me))
        x = me.position[0]
        if key not in sim.mem and x > x_end - t_look * max(me.speed, 5.0):
            ahead, behind = 1e3, 1e3
            for ob in [sim.ego] + list(sim.objs.values()):
                if ob is me or abs(ob.position[1] - tgt_y) > 0.45 * lane_w:
                    continue
                dx = ob.position[0] - x
                if dx >= 0:
                    ahead = min(ahead, dx - (ob.LENGTH + me.LENGTH) / 2)
                else:
                    behind = min(behind, -dx - (ob.LENGTH + me.LENGTH) / 2 - 1.5 * max(0.0, ob.speed - me.speed))
            if ahead >= g_front and behind >= g_rear:
                sim.mem[key] = (t, me.position[1])
        if key in sim.mem:
            t0, y0 = sim.mem[key]
            y_t = y0 + (tgt_y - y0) * quintic((t - t0) / T_lc)
        else:
            y_t = lane_y
        buf = sim.mem.setdefault(("zip_buf", id(me)), [])
        s, ld = leader_of(me, sim)
        if ld is not None and ld.LENGTH > 20 and (key in sim.mem or s > 30.0):     # the static lane end: only near it
            s, ld = None, None
        buf.append((1e3, 0.0) if ld is None else (s, me.speed - ld.velocity[0]))
        s_p, dv_p = buf[max(0, len(buf) - 1 - int(round(react / DT)))]
        return float(np.clip(idm_accel(me.speed, v0, s_p, dv_p, T=T, s0=2.0, a=1.5, b=2.0), -6.0, 1.5)), y_t
    return f


def law_event(v0, t_ev=None, b=0.0, v_end=0.0, y_to=None, T_lc=2.5, T=1.2):
    """RC19 fixed traffic (notebook law_event): IDM on its own lane (desired v0) plus a scripted event from t_ev:
    brake at b down to v_end and/or change lane to y_to over T_lc (a quintic)."""
    def f(t, me, sim):
        acc = 0.0
        if t_ev is not None and t >= t_ev and me.speed > v_end:
            acc = -b
        s, ld = leader_of(me, sim)
        if ld is not None:
            acc = min(acc, idm_accel(me.speed, v0, s, me.speed - ld.velocity[0], T=T, s0=2.0, a=1.5, b=2.0))
        if y_to is None or t_ev is None:
            return float(np.clip(acc, -6.0, 1.5))
        y0 = sim.mem.setdefault(("y_from", id(me)), me.position[1])
        return float(np.clip(acc, -6.0, 1.5)), y0 + (y_to - y0) * quintic((t - t_ev) / T_lc)
    return f


def law_hold(v0):
    """Constant speed in its own lane, never reacting (distant benign traffic)."""
    return lambda t, me, sim: speed_cmd(me.speed, v0, a_lo=-1.0, a_hi=1.0)


def law_fast(v0, notice, react, b_max, T=0.9, s0=2.0, a=1.5):
    """RC07-S5 fast car coming up a lane, inattentive: holds v0 until a body in its lane ahead (the ego or another
    car) is within `notice` m (bumper gap); `react` s later it follows that body (IDM, time gap T), braking up to
    b_max."""
    def f(t, me, sim):
        key = ("fast", id(me))
        s, ld = leader_of(me, sim)
        if key not in sim.mem and ld is not None and s < notice:
            sim.mem[key] = t
        if key not in sim.mem or t < sim.mem[key] + react:
            return 0.0
        dv = me.speed - ld.velocity[0] if ld is not None else 0.0
        return float(np.clip(idm_accel(me.speed, v0, s if ld is not None else 1e3, dv, T, s0, a, b=3.0), -b_max, a))
    return f


def _in_lane(sim, me, y, lane_w):
    """Bodies (the ego included) centred within the lane at y, as (dx to me, body)."""
    return [(ob.position[0] - me.position[0], ob) for ob in [sim.ego] + list(sim.objs.values())
            if ob is not me and abs(ob.position[1] - y) < 0.45 * lane_w]


def law_onramp(v0, y_tgt, cut, t_cut, a_floor, v_top, x_end, panic=25.0, T_lc=2.5, T=1.2, b_max=6.0, lane_w=3.5):
    """RC07-S5 on-ramp car in the acceleration lane (ends at x_end), merging into the lane at y_tgt:
        cut   at t_cut (or 40 m before the ramp ends) into the slot beside it; a car alongside then (overlapping
              when it would cross the line, T_lc / 2 ahead): it eases off (-2 m/s^2) and squeezes in behind it
        wait  once the cars behind it in the target lane have passed (60 m clear behind and alongside), or 40 m
              before the ramp ends; until then it eases off (-1.5 m/s^2) to let them by
    The lane change is a quintic over T_lc. Once in the target lane it watches only its mirror: when the car behind
    it there is within `panic` m (bumper gap) and closing, it floors it (a_floor) towards v_top for good, ignoring
    what is ahead (it has not registered the ego); otherwise IDM on its leader. Not merged at the ramp's end: it
    brakes to a stop there."""
    def f(t, me, sim):
        m = sim.mem.setdefault(("onramp", id(me)), {})
        x = me.position[0]
        if "t_lc" not in m:
            near_end = x > x_end - 40.0
            # alongside = overlapping (1 m to spare) when it would cross the line, T_lc / 2 from now
            h = T_lc / 2
            beside = [ob for dx, ob in _in_lane(sim, me, y_tgt, lane_w)
                      if abs(dx + (ob.velocity[0] - me.speed) * h) < (ob.LENGTH + me.LENGTH) / 2 + 1.0]
            if cut:
                go = (t >= t_cut or near_end) and not beside
            else:
                behind = [ob for dx, ob in _in_lane(sim, me, y_tgt, lane_w) if -60.0 < dx < (ob.LENGTH + me.LENGTH) / 2]
                go = (near_end and not beside) or (t >= t_cut and not behind)
            if go:
                m["t_lc"], m["y0"] = t, me.position[1]
        if "t_lc" not in m:
            y_t = m.setdefault("y_ramp", me.position[1])
            acc = speed_cmd(me.speed, v0, a_lo=-3.0, a_hi=1.5)
            if not cut and t >= t_cut:
                acc = min(acc, -1.5)
            elif cut and t >= t_cut:                           # a car alongside: ease off to slot in behind it
                acc = min(acc, -2.0)
            d_end = x_end - x - me.LENGTH / 2
            if d_end < 40.0:                                   # the cones: stop before them
                acc = min(acc, -me.speed ** 2 / (2 * max(d_end, 0.5)))
            return float(np.clip(acc, -b_max, 1.5)), y_t
        y_t = m["y0"] + (y_tgt - m["y0"]) * quintic((t - m["t_lc"]) / T_lc)
        if abs(me.position[1] - y_tgt) < 0.5 and "panic" not in m:
            back = [(-dx - (ob.LENGTH + me.LENGTH) / 2, ob) for dx, ob in _in_lane(sim, me, y_tgt, lane_w) if dx < 0]
            if back:
                gap, ob = min(back, key=lambda g: g[0])
                if gap < panic and ob.speed > me.speed:
                    m["panic"] = t
        if "panic" in m:
            return (a_floor if me.speed < v_top else 0.0), y_t
        s, ld = leader_of(me, sim)
        dv = me.speed - ld.velocity[0] if ld is not None else 0.0
        return float(np.clip(idm_accel(me.speed, v0, s if ld is not None else 1e3, dv, T, 2.0, 1.5, 2.0), -b_max, 1.5)), y_t
    return f


STYLES = {   # RC15 response strength (notebook STYLES, adjacent-driver part)
    "subtle": dict(b_yield=2.5, dv_yield=3.0, dv_floor=6.0, a_match=2.5, d_match=-2.0, T_match=1.2),
    "pronounced": dict(b_yield=3.5, dv_yield=7.0, dv_floor=9.0, a_match=3.0, d_match=3.0, T_match=0.6),
}
