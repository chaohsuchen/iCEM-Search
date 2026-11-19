"""The vehicle maneuvering envelope (Recoverability_Vehicle_Envelope_and_Policy_Search_v2, §3-6).

Contract (flat road, g0 = 9.80665): -1.0 g <= a_x <= +0.5 g, |a_y| <= 1.0 g, a_x^2 + a_y^2 <= (1.0 g)^2.
Usable envelope = contract ∩ the qualified capability of the ego blueprint:
    A(v)  propulsion: the measured full-throttle acceleration at speed v (carla_rc/accel_map.json)
    B(v)  braking:    the sustained full-brake deceleration at speed v measured by qualify.py (envelope_qual.json);
                      before qualification the accel map's (window-averaged, lower) full-brake value
    AY    lateral:    the measured stable lateral acceleration (qualify.py)
    M     combined:   the measured combined planar acceleration (qualify.py)
Until qualify.py has run, AY / M fall back to declared, unqualified values (UNQUALIFIED) and the card says so.
Road condition: the ego's tyre friction scale mu (1 = dry; CARLA's grip only drops below ~x0.3, see ROAD) has its own
qualification, keyed "<bp>@mu<mu>" (dry keeps the plain "<bp>" key), so every condition's envelope is measured. The
request rate limits are declared actuator limits (RATES); the measured end-to-end build-up (which already contains the
vehicle's own lag) is recorded in the card, not applied a second time.

Policy action z in [-1, 1]^2 -> request (§6, extended by the usable caps):
    a_x = A(v) max(z_x, 0) + B(v) min(z_x, 0)
    a_y = z_y min(AY, sqrt(M^2 - a_x^2))
which is inside the usable set for every z (with A = 0.5 g, B = M = AY = 1 g it is the report's mapping). The
request is then rate-limited (brake / propulsion buildup, lateral build-up) by moving along the segment towards it,
which keeps it inside the (convex) set. The yaw-rate command is a_y / v.
Monitor: the measured planar acceleration (world-velocity differences projected on the body axes, declared 0.25 s
moving mean) must stay inside the contract (+ TOL_G) while the ego moves faster than V_MON; a violation fails the
episode. Below V_MON = 10 m/s CARLA's MKZ jolts at every automatic downshift (1.1-1.3 g for ~0.1 s at any brake pedal
from 0.6 to 1.0, measured by qualify.py) and again when it comes to rest (~2 g): drivetrain artefacts of the simulator,
not manoeuvring acceleration, so they are not checked (known limitation; above 10 m/s full braking stays <= 0.94 g).
"""
import json
import math
import os
from collections import deque

import numpy as np

G0 = 9.80665
CONTRACT = dict(ax_min=-1.0 * G0, ax_max=0.5 * G0, ay_max=1.0 * G0, comb=1.0 * G0)
HERE = os.path.dirname(os.path.abspath(__file__))
QUAL_FILE = os.path.join(HERE, "envelope_qual.json")
ACCEL_MAP_FILE = os.path.join(os.path.dirname(HERE), "accel_map.json")
UNQUALIFIED = dict(ay=7.0, comb=7.7)       # notebook mu_g, accel-map straight braking
RATES = dict(jerk_x=40.0, jerk_y=30.0)      # declared request rate limits [m/s^3] (brake / steering build-up)
TOL_G = 0.05                       # monitor tolerance on the filtered measurement [g]
FILT_TICKS = 5                     # 0.25 s moving mean at 20 Hz
V_MIN_YAW = 3.0                    # yaw rate = a_y / max(v, V_MIN_YAW)
V_MON = 10.0                       # the monitor checks while the ego is faster than this [m/s]
# governor (specs with governor = True): the low-level controller can deliver more than was requested (the pedal on
# hard braking, steering oscillation on tight bends), so the request is scaled down while the monitor's measurement
# is above these levels [g], and recovers by GOV_RECOVER per policy step once it is below
GOV_G, GOV_AX_G, GOV_MIN, GOV_RECOVER = 0.97, 0.47, 0.3, 0.05
# road conditions: the ego's CARLA tyre-friction scale mu (measured: x1 / x0.6 / x0.4 brake alike, ~7.6 m/s^2 at 27 m/s;
# x0.25 ~5 m/s^2 stable, x0.15 ~4.3 m/s^2), the speed level drivers keep (scales the scene's speeds) and the grip
# left to the other cars (scales their braking / sliding deceleration, ~ the ego's measured braking ratio)
ROAD = {"dry": dict(mu=1.0, speed=1.0, grip=1.0), "wet": dict(mu=0.25, speed=0.9, grip=0.7),
        "heavy": dict(mu=0.15, speed=0.78, grip=0.55)}


def qual_key(bp, mu=1.0):
    return bp if mu == 1.0 else f"{bp}@mu{mu:g}"


class Envelope:
    def __init__(self, bp="vehicle.lincoln.mkz_2020", mu=1.0):
        m = json.load(open(ACCEL_MAP_FILE))[bp]
        A = np.array(m["accel"], dtype=float)
        self.v_tab = np.array(m["speeds"], dtype=float)
        ok = ~np.isnan(A[:, 0])                                    # full brake is not measured below ~14 m/s
        self.b_tab = np.interp(self.v_tab, self.v_tab[ok], -A[ok, 0])
        self.a_tab = A[:, -1]
        q = json.load(open(QUAL_FILE)).get(qual_key(bp, mu)) if os.path.exists(QUAL_FILE) else None
        assert q is not None or mu == 1.0, f"no qualification for {qual_key(bp, mu)}: run qualify.py --friction {mu}"
        self.qualified = q is not None
        q = q or UNQUALIFIED
        if self.qualified:                                         # sustained braking measured per speed
            br = sorted((r["v0"], r["peak"]) for r in q["tests"]["brake"])
            self.b_tab = np.interp(self.v_tab, [v for v, _ in br], [b for _, b in br])
            self.brake_levels = {r["v0"]: [(x["pedal"], x["sustained"]) for x in r.get("levels", [])]
                                 for r in q["tests"]["brake"]}
            self.build_up = dict(jerk_x=q["jerk_x"], jerk_y=q["jerk_y"])
        self.AY = min(CONTRACT["ay_max"], q["ay"])
        self.M = min(CONTRACT["comb"], q["comb"])
        self.comb_measured = q["comb"]
        self.jerk_x, self.jerk_y = RATES["jerk_x"], RATES["jerk_y"]
        self.bp, self.mu = bp, mu

    def caps(self, v):
        """(A, B, AY, M) at speed v: usable propulsion, braking, lateral and combined acceleration [m/s^2]."""
        A = min(CONTRACT["ax_max"], max(float(np.interp(v, self.v_tab, self.a_tab)), 0.3))
        B = min(-CONTRACT["ax_min"], self.M, float(np.interp(v, self.v_tab, self.b_tab)))
        return A, B, self.AY, self.M

    def map(self, z, v):
        A, B, AY, M = self.caps(v)
        zx, zy = float(np.clip(z[0], -1, 1)), float(np.clip(z[1], -1, 1))
        ax = A * max(zx, 0.0) + B * min(zx, 0.0)
        return ax, zy * min(AY, math.sqrt(max(M * M - ax * ax, 0.0)))

    def inverse(self, ax, ay, v):
        """z for a desired (a_x, a_y): a_x is kept if possible, a_y gets what the combined budget leaves."""
        A, B, AY, M = self.caps(v)
        zx = float(np.clip(ax / A if ax >= 0 else ax / B, -1, 1))
        ax = A * max(zx, 0.0) + B * min(zx, 0.0)
        lim = min(AY, math.sqrt(max(M * M - ax * ax, 0.0)))
        return np.array([zx, float(np.clip(ay / lim, -1, 1)) if lim > 1e-6 else 0.0], np.float32)

    def card(self):
        return dict(contract_g=dict(ax_min=-1.0, ax_max=0.5, ay_max=1.0, comb=1.0), bp=self.bp, mu=self.mu,
                    qualified=self.qualified,
                    AY=self.AY, M=self.M, comb_measured=self.comb_measured, jerk_x=self.jerk_x, jerk_y=self.jerk_y,
                    measured_build_up=getattr(self, "build_up", None),
                    A_of_v=dict(zip(self.v_tab.tolist(), [self.caps(v)[0] for v in self.v_tab])),
                    B_of_v=dict(zip(self.v_tab.tolist(), [self.caps(v)[1] for v in self.v_tab])),
                    mapping="a_x = A(v) max(z_x,0) + B(v) min(z_x,0); a_y = z_y min(AY, sqrt(M^2 - a_x^2))",
                    monitor=f"measured planar accel, {FILT_TICKS}-tick mean, inside the contract + {TOL_G} g above {V_MON} m/s")


class RateLimiter:
    """Moves the applied request towards the new one along the segment: |da_x| <= jerk_x dt, |da_y| <= jerk_y dt."""
    def __init__(self, env_, dt):
        self.e, self.dt, self.cur = env_, dt, None

    def reset(self, ax=0.0, ay=0.0):
        self.cur = (ax, ay)

    def __call__(self, ax, ay):
        cx, cy = self.cur
        dx, dy = ax - cx, ay - cy
        s = min(1.0, self.e.jerk_x * self.dt / max(abs(dx), 1e-9), self.e.jerk_y * self.dt / max(abs(dy), 1e-9))
        self.cur = (cx + s * dx, cy + s * dy)
        return self.cur


class Monitor:
    """Independent check of what the vehicle actually did (body-frame planar acceleration from world velocities)."""
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.prev, self.buf = None, deque(maxlen=FILT_TICKS)
        self.max_g, self.max_ax_g, self.min_ax_g, self.viol, self.n = 0.0, 0.0, 0.0, 0, 0
        self.last = (0.0, 0.0)

    def update(self, vx, vy, yaw):
        """World velocity [m/s] and yaw [rad] at this tick -> filtered (a_long, a_lat) [m/s^2]."""
        if self.prev is not None:
            ax_w, ay_w = (vx - self.prev[0]) / self.dt, (vy - self.prev[1]) / self.dt
            c, s = math.cos(yaw), math.sin(yaw)
            self.buf.append((ax_w * c + ay_w * s, -ax_w * s + ay_w * c))
        self.prev = (vx, vy)
        if len(self.buf) < FILT_TICKS:
            return self.last
        al, at = np.mean(self.buf, axis=0)
        self.last = (float(al), float(at))
        if math.hypot(vx, vy) < V_MON:
            return self.last
        g, gx = math.hypot(al, at) / G0, al / G0
        self.n += 1
        self.max_g, self.max_ax_g, self.min_ax_g = max(self.max_g, g), max(self.max_ax_g, gx), min(self.min_ax_g, gx)
        if g > 1.0 + TOL_G or gx > 0.5 + TOL_G or abs(at) / G0 > 1.0 + TOL_G:
            self.viol += 1
        return self.last

    def summary(self):
        return dict(max_g=self.max_g, max_ax_g=self.max_ax_g, min_ax_g=self.min_ax_g, viol_ticks=self.viol)
