"""Fast surrogate of RC05-01 (lane change with abort): same policy class + LowLevelController (incl. its steering law) as CARLA, fitted longitudinal plant,
a fitted lateral plant (yaw rate = LAT_K * v * steer, first-order steering lag), kinematic lead car and rear-left car, true convex footprint distance.
Starts at the lane-change start (t_since_cue = 0)."""
import math
import numpy as np
from rc01.config import PolicyConfig
from rc01.controller import LowLevelController
from rc01.policy import PolicyOutput
from rcx.policies import LaneChangeAbortPolicy
from .plant import LongPlant
from .lane_core import MAX_STEER, LAT_K, LAT_TAU, EGO_HL, EGO_HW, _rect, _pt_seg, quad_box_dist

LEAD_HL, LEAD_HW, REAR_HL, REAR_HW = 1.95, 0.9, 2.14, 0.9
SCN = dict(v_ego=10.0, v_lead=6.0, lead_gap=35.0, closing=2.0, rear_gap=25.0, hazard_progress=0.5, rear_acc=2.5, rear_vmax=20.0, t_lc=4.0, lane_w=3.5,
           duration_s=9.0, min_rear_clear=1.15, min_lead_clear=2.0)


class LCFamilyPolicy(LaneChangeAbortPolicy):
    """LaneChangeAbortPolicy + an optional forward surge after the hazard (surge_acc m/s^2 for surge_dur s, still bounded by the ACC safety brake)."""
    name = "lane_change_family"; version = "rc05-fam-v1"
    def __init__(self, dt=0.05, surge_acc=0.0, surge_dur=0.0, **kw):
        super().__init__(dt, **kw); self.kw.update(surge_acc=surge_acc, surge_dur=surge_dur); self.t_hz = None
    def reset(self): super().reset(); self.t_hz = None
    def step(self, o):
        out = super().step(o)
        if o.get("hazard") and self.t_hz is None: self.t_hz = o["t"]
        k = self.kw
        if k["surge_acc"] > 0 and self.t_hz is not None and o["t"] - self.t_hz <= k["surge_dur"]:
            out.a_req = self.acc(o, k["surge_acc"])
        return out


def simulate(theta_s, policy_kw, plant: LongPlant, calib, pcfg=None, dt=0.05, policy_cls=LCFamilyPolicy):
    p = dict(SCN); p.update(theta_s); pcfg = pcfg or PolicyConfig(target_speed=p["v_ego"])
    kw = dict(policy_kw); kw.setdefault("v_target", p["v_ego"]); kw.setdefault("lane_w", p["lane_w"]); kw.setdefault("t_lc", p["t_lc"])
    pol = policy_cls(dt, **kw); pol.reset()
    ctrl = LowLevelController(pcfg, calib, 0.0, MAX_STEER, dt); plant.reset()
    W = p["lane_w"]; tp = p["t_lc"] / math.pi * math.acos(1 - 2 * p["hazard_progress"])         # hazard timing is defined by the scenario nominal lane change
    s, t, yaw, v, a_prev, st_f = 0.0, 0.0, 0.0, p["v_ego"], 0.0, 0.0
    sl = EGO_HL + p["lead_gap"] + LEAD_HL
    rl_gap0 = p["rear_gap"] + p["closing"] * tp                                   # gap at the lane-change start (the car is placed so the gap at the cue = rear_gap)
    sr = -EGO_HL - rl_gap0 - REAR_HL; v0r = p["v_ego"] + p["closing"]; vr = v0r
    min_rl = min_lead = 99.0; contact = None; rows = []; peak_lat = 0.0; peak_dec = 0.0; phases = []
    for n in range(int(p["duration_s"] / dt)):
        rt = n * dt
        vr = v0r if rt < tp else min(p["rear_vmax"], v0r + p["rear_acc"] * (rt - tp))
        o = dict(t=rt, v=v, s=s, ego_t=t, cue=True, hazard=rt >= tp, lead_gap=(sl - LEAD_HL) - (s + EGO_HL), lead_v=p["v_lead"],
                 rl_gap=(s - EGO_HL) - (sr + REAR_HL), rl_v=vr)
        out = pol.step(o)
        c = ctrl.step(out, t, math.degrees(yaw), v, a_prev)
        st_f += dt / (LAT_TAU + dt) * (c["steer"] - st_f)
        v2, a_prev = plant.step(v, c["throttle"], c["brake"], c["hand_brake"])
        yr = LAT_K * 0.5 * (v + v2) * st_f
        yaw += yr * dt; t += 0.5 * (v + v2) * math.sin(yaw) * dt; s += 0.5 * (v + v2) * math.cos(yaw) * dt
        peak_lat = max(peak_lat, abs(0.5 * (v + v2) * yr)); peak_dec = max(peak_dec, -a_prev); v = v2
        sl += p["v_lead"] * dt; sr += 0.5 * (vr + (v0r if rt + dt < tp else min(p["rear_vmax"], v0r + p["rear_acc"] * (rt + dt - tp)))) * dt
        fe = _rect(s, t, yaw, EGO_HL, EGO_HW)
        d_rl = quad_box_dist(fe, (sr - REAR_HL, sr + REAR_HL, -W - REAR_HW, -W + REAR_HW)); d_ld = quad_box_dist(fe, (sl - LEAD_HL, sl + LEAD_HL, -LEAD_HW, LEAD_HW))
        min_rl, min_lead = min(min_rl, d_rl), min(min_lead, d_ld)
        rows.append((rt, v, t, math.degrees(yaw), d_rl, d_ld, out.a_req, a_prev)); phases.append(pol.phase)
        if d_rl <= 0.02 or d_ld <= 0.02:
            contact = ("rear_left" if d_rl <= 0.02 else "lead", rt, abs(vr - v) if d_rl <= 0.02 else abs(v - p["v_lead"])); break
    R = np.array(rows); aborted = any(q in ("abort", "follow_lead") for q in phases)
    et = R[:, 2]; yw = R[:, 3]
    fin_ok = bool(np.all(np.abs(et[-20:]) < 0.5) or np.all(np.abs(et[-20:] + W) < 0.5)) if len(et) >= 20 else False
    stable = None
    if aborted and "follow_lead" in phases:
        j = phases.index("follow_lead") + int(1.0 / dt)
        if j < len(et): stable = bool(np.abs(et[j:]).max() < 0.42 and np.abs(yw[j:]).max() < 3.0)     # CARLA limit 0.3 m; the surrogate overshoots ~0.12 m more
    fail = []
    if contact: fail.append("contact with " + contact[0])
    else:
        if min_rl < p["min_rear_clear"]: fail.append("rear-left clearance")
        if min_lead < p["min_lead_clear"]: fail.append("lead clearance")
        if not fin_ok: fail.append("not settled")
        if stable is False: fail.append("unstable after return")
    if peak_lat > 9.81: fail.append("lateral envelope")
    return dict(min_front_gap=min_lead, min_rear_gap=min_rl, min_rl=min_rl, min_lead=min_lead, contact=contact is not None, contact_actor=contact[0] if contact else None,
                rel_speed=contact[2] if contact else 0.0, aborted=aborted, settled=fin_ok, stable=stable, peak_lat=peak_lat, peak_decel=peak_dec,
                final_t=float(et[-1]), stopped=False, verdict="FAIL" if fail else "PASS", reason="; ".join(fail), rows=rows)
