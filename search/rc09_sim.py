"""Fast surrogate of RC09-01 (sandwiched, no exit): same policy class + same LowLevelController as the CARLA run, fitted plant, kinematic truck/queue.
simulate(theta_s, policy_kw, plant, pcfg, calib) -> metrics dict (same names as the CARLA summary)."""
import numpy as np
from rc01.config import PolicyConfig
from rc01.controller import LowLevelController
from rcx.policies import StopPolicy
from .plant import LongPlant

SCN = dict(v_ego=8.0, front_gap=24.0, rear_gap=18.0, closing=4.0, rear_delay_s=1.4, rear_decel=5.0, min_clear=1.0, duration_s=16.0)


def simulate(theta_s, policy_kw, plant: LongPlant, calib, pcfg=None, dt=0.05, policy_cls=StopPolicy):
    p = dict(SCN); p.update(theta_s); pcfg = pcfg or PolicyConfig()
    pol = policy_cls(dt, **policy_kw); pol.reset()
    ctrl = LowLevelController(pcfg, calib, 0.0, 0.6, dt); plant.reset()
    v, fg, rg = p["v_ego"], p["front_gap"], p["rear_gap"]
    vt0 = v + p["closing"]; a_prev = 0.0
    min_f, min_r = fg, rg; contact = None; rows = []
    for n in range(int(p["duration_s"] / dt)):
        rt = n * dt
        vt = vt0 if rt < p["rear_delay_s"] else max(0.0, vt0 - p["rear_decel"] * (rt - p["rear_delay_s"]))
        out = pol.step(dict(v=v, cue=True, front_gap=fg, frame=n, t=rt))
        c = ctrl.step(out, 0.0, 0.0, v, a_prev)
        v2, a_prev = plant.step(v, c["throttle"], c["brake"], c["hand_brake"])
        v_mid = 0.5 * (v + v2)
        vt2 = vt0 if rt + dt < p["rear_delay_s"] else max(0.0, vt0 - p["rear_decel"] * (rt + dt - p["rear_delay_s"]))
        fg -= v_mid * dt; rg += (v_mid - 0.5 * (vt + vt2)) * dt; v = v2
        min_f, min_r = min(min_f, fg), min(min_r, rg)
        rows.append((rt, v, fg, rg, vt, out.a_req, a_prev))
        if fg <= 0.02: contact = ("queue0", rt, v); break
        if rg <= 0.02: contact = ("rear_truck", rt, vt2 - v); break
        if v < 0.1 and vt2 < 0.1: break
    fail = []
    if contact: fail.append("contact: " + contact[0])
    else:
        if min_f < p["min_clear"]: fail.append("front clearance")
        if min_r < p["min_clear"]: fail.append("rear clearance")
        if v >= 0.1: fail.append("did not stop")
    return dict(min_front_gap=min_f, min_rear_gap=min_r, contact=contact is not None, contact_actor=contact[0] if contact else None,
                rel_speed=contact[2] if contact else 0.0, stopped=bool(v < 0.1), verdict="FAIL" if fail else "PASS", reason="; ".join(fail), rows=rows)
