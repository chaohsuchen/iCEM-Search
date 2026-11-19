"""Fast surrogate of RC01-01 / RC17-01 (street blocked by a van, brake in lane): same policy class + LowLevelController as CARLA, fitted plant, static van.
Starts at the instant the observation is captured (gap <= trigger) with the measured speed/gap; the cue reaches the policy after delay_s (RC17).
simulate(theta_s, policy_kw, plant, calib) -> dict(min_gap, stopping_gap, contact, rel_speed, stopped, verdict, ...)."""
import numpy as np
from rc01.config import PolicyConfig
from rc01.controller import LowLevelController
from .plant import LongPlant
from .families import ProfilePolicy

SCN = dict(v_ego=8.1, gap=20.0, delay_s=0.0, min_clear=2.7, time_limit=8.0)


def simulate(theta_s, policy_kw, plant: LongPlant, calib, pcfg=None, dt=0.05, policy_cls=ProfilePolicy):
    p = dict(SCN); p.update(theta_s); pcfg = pcfg or PolicyConfig(target_speed=8.0)
    kw = dict(policy_kw); kw.setdefault("v_target", pcfg.target_speed)
    pol = policy_cls(dt, **kw); pol.reset()
    ctrl = LowLevelController(pcfg, calib, 0.0, 0.6, dt); plant.reset()
    v, g, a_prev = p["v_ego"], p["gap"], 0.0
    min_g, contact, rows, t_stop = g, None, [], None; t_slow = None
    for n in range(int(p["time_limit"] / dt)):
        t = n * dt
        out = pol.step(dict(v=v, cue=t >= p["delay_s"], frame=n, t=t, gap=g, delivered=t >= p["delay_s"]))
        c = ctrl.step(out, 0.0, 0.0, v, a_prev)
        v2, a_prev = plant.step(v, c["throttle"], c["brake"], c["hand_brake"])
        g -= 0.5 * (v + v2) * dt; v = v2; min_g = min(min_g, g)
        rows.append((t, v, g, 0.0, 0.0, out.a_req, a_prev))
        if g <= 0.02: contact = ("van", t, v); break
        if v < 0.1 and t_slow is None: t_slow = t
        if v < 0.05: t_stop = t; break
    stopped = bool(v < 0.1 and contact is None)
    fail = []
    if contact: fail.append("contact: van")
    else:
        if not stopped or t_slow is None or t_slow > p["time_limit"] - 2.0: fail.append("did not stop and hold 2 s within the time limit")
        elif g < p["min_clear"]: fail.append("insufficient stopping clearance")
    return dict(min_front_gap=min_g, min_rear_gap=99.0, stopping_gap=g if stopped else None, contact=contact is not None, rel_speed=contact[2] if contact else 0.0,
                stopped=stopped, verdict="FAIL" if fail else "PASS", reason="; ".join(fail), rows=rows)
