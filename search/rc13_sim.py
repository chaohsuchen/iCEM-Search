"""Fast surrogate of RC13-01 (pedestrians on the sidewalk, ego keeps lane at ~8 m/s): same policy class + LowLevelController as CARLA, fitted longitudinal plant,
kinematic pedestrians. Variation: a 'dart' pedestrian that steps into the lane (dart_speed m/s toward the lane centre once the ego is within dart_dist m of it).
simulate(theta_s, policy_kw, plant, calib, policy_cls) -> dict like the other surrogates."""
import math
import numpy as np
from rc01.config import PolicyConfig
from rc01.controller import LowLevelController
from rc01.policy import PolicyOutput
from rcx.policies import _Base, ReachableSetPolicy
from .plant import LongPlant

# state at the cue measured in CARLA (results_v3/rc13): ego s, speed; pedestrians (s, t, direction, speed) at cue
SCN = dict(v_ego=7.94, ego_s=26.42, dart_speed=0.0, dart_dist=15.0, dart_lag=0.18, max_brake=9.81, duration_s=11.0, min_speed_ratio=0.9, speed_bias=0.05, v_target=8.0, clear_req=0.9,
           peds=[(76.34, 5.85, -1.0, 1.4), (109.4, 6.05, 1.0, 1.2), (51.7, 6.15, -1.0, 1.1)])
EGO_HL, EGO_HW, PED_R, LANE_HALF = 2.4, 1.08, 0.3, 1.75


def _out(phase, a):
    return PolicyOutput(phase, a, info=dict(conflict=int(phase in ("yield", "hold")), pred_lane_overlap=0))


class PreSlowPolicy(_Base):
    """Slow to v_slow when a pedestrian is within `zone` m ahead (any lateral position); brake a_react if a pedestrian is already in / entering the lane."""
    name = "pre_slow"; version = "rc13-preslow-v1"
    def __init__(self, dt=0.05, v_target=8.0, v_slow=6.0, zone=30.0, a_react=5.0, lane_half=1.75):
        super().__init__(dt, v_target=v_target, v_slow=v_slow, zone=zone, a_react=a_react, lane_half=lane_half); self.k = self.kw
    def step(self, o):
        k, v, s = self.k, o["v"], o["s"]
        if not o.get("cue", True): self.phase = "follow"; return _out("follow", self.speed_hold(v, k["v_target"]))
        ahead = [(ps, pt) for ps, pt in o["peds"] if ps >= s - 1.0]
        inlane = [ps for ps, pt in ahead if abs(pt) <= k["lane_half"] + 0.6]
        if inlane and v > 0.05:
            if min(inlane) - (s + 2.4) < 2.5 and v < 2.0: self.phase = "hold"; return _out("hold", 0.0)
            d = max(min(inlane) - (s + 2.4) - 2.0, 0.5); self.phase = "yield"
            return _out("yield", -min(k["a_react"], max(1.0, v * v / (2 * d))))
        if inlane: self.phase = "hold"; return _out("hold", 0.0)
        near = any(ps - s <= k["zone"] for ps, pt in ahead); self.phase = "follow"
        return _out("follow", self.speed_hold(v, k["v_slow"] if near else k["v_target"]))


class ReactPolicy(_Base):
    """Keep speed; react to what a pedestrian DOES: if its lateral speed towards the lane exceeds `thr` (estimated over 0.2 s) and it is within `d_react` m ahead,
    brake (up to a_react) to stop before it; also brake for any pedestrian already in the lane. No assumption about a maximum pedestrian speed."""
    name = "react_to_motion"; version = "rc13-react-v1"
    def __init__(self, dt=0.05, v_target=8.0, thr=0.8, a_react=5.0, d_react=25.0, lane_half=1.75):
        super().__init__(dt, v_target=v_target, thr=thr, a_react=a_react, d_react=d_react, lane_half=lane_half); self.k = self.kw; self.hist = []; self.alarm = set()
    def reset(self): super().reset(); self.hist = []; self.alarm = set()
    def step(self, o):
        k, v, s = self.k, o["v"], o["s"]
        if not o.get("cue", True): self.phase = "follow"; return _out("follow", self.speed_hold(v, k["v_target"]))
        self.hist = (self.hist + [list(o["peds"])])[-5:]
        ahead = [(i, ps, pt) for i, (ps, pt) in enumerate(o["peds"]) if ps >= s - 1.0]
        inlane = [ps for i, ps, pt in ahead if abs(pt) <= k["lane_half"] + 0.6]
        if len(self.hist) >= 5:
            old = self.hist[0]
            for i, ps, pt in ahead:
                if i < len(old) and ps - s <= k["d_react"] and (abs(old[i][1]) - abs(pt)) / (4 * self.dt) > k["thr"]: self.alarm.add(i)
        al = [ps for i, ps, pt in ahead if i in self.alarm]
        if (al or inlane) and v > 0.05:
            tgt = min(inlane + al)
            if tgt - (s + 2.4) < 2.5 and v < 2.0: self.phase = "hold"; return _out("hold", 0.0)       # close enough and slow: stop dead (no creeping)
            d = max(tgt - (s + 2.4) - 2.0, 0.5); self.phase = "yield"
            return _out("yield", -min(k["a_react"], max(1.0, v * v / (2 * d))))
        if (al or inlane) and v <= 0.05: self.phase = "hold"; return _out("hold", 0.0)
        self.phase = "follow"; return _out("follow", self.speed_hold(v, k["v_target"]))


def _rect_circle_dist(es, ps, pt):
    dx = max(abs(ps - es) - EGO_HL, 0.0); dy = max(abs(pt) - EGO_HW, 0.0); return math.hypot(dx, dy) - PED_R


def simulate(theta_s, policy_kw, plant: LongPlant, calib, pcfg=None, dt=0.05, policy_cls=ReachableSetPolicy):
    p = dict(SCN); p.update(theta_s); pcfg = pcfg or PolicyConfig(target_speed=8.0)
    pol = policy_cls(dt, **policy_kw); pol.reset()
    ctrl = LowLevelController(pcfg, calib, 0.0, 0.6, dt); plant.reset()
    s, v, a_prev = p["ego_s"], p["v_ego"], 0.0
    peds = [[ps, pt, d, vs] for ps, pt, d, vs in p["peds"]]; dart_on = False; vlat = 0.0; k_lag = 1.0 - math.exp(-dt / max(p["dart_lag"], 1e-6))
    contact = None; rows = []; min_d = 99.0; v_pass = None; stopped = False; entered = False
    for n in range(int(p["duration_s"] / dt)):
        t = n * dt
        out = pol.step(dict(v=v, s=s, cue=True, t=t, peds=[(q[0], q[1]) for q in peds]))
        c = ctrl.step(out, 0.0, 0.0, v, a_prev)
        v2, a_prev = plant.step(v, c["throttle"], c["brake"], c["hand_brake"])
        s += 0.5 * (v + v2) * dt; v = v2
        tgt = peds[0]
        if p["dart_speed"] > 0 and not dart_on and tgt[0] - (s + EGO_HL) <= p["dart_dist"]: dart_on = True
        for i, q in enumerate(peds):
            if i == 0 and dart_on:
                vlat += ((p["dart_speed"] if q[1] > 0.0 else 0.0) - vlat) * k_lag         # CARLA walker control ramps up with tau ~0.18 s (fit on 34 CARLA dart runs, rmse < 4 cm)
                q[1] = max(0.0, q[1] - vlat * dt)                                         # steps into the lane and stands at the lane centre
            else:
                q[0] += q[2] * q[3] * dt
        if abs(tgt[1]) <= LANE_HALF: entered = True
        d = _rect_circle_dist(s, tgt[0], tgt[1]); min_d = min(min_d, d)
        if v_pass is None and tgt[0] - (s + EGO_HL) <= 0: v_pass = v
        stopped = stopped or v < 0.1
        rows.append((t, v, s, tgt[0], tgt[1], out.a_req, a_prev))
        if d <= 0.0: contact = ("ped", t, v); break
        if tgt[0] - s < -12.0: break
        if v < 0.05 and dart_on: break
    al = np.array([r[6] for r in rows]); al_f = np.array([al[max(0, i - 3):i + 1].mean() for i in range(len(al))]) if len(al) else al      # same 0.2 s trailing mean as the CARLA envelope check
    phys = len(al) > 0 and float(-al_f.min()) > p["max_brake"]
    fail = []
    slowing = bool(stopped or (v_pass is not None and v_pass < p["min_speed_ratio"] * p["v_target"] + p["speed_bias"]) or (v_pass is None and not dart_on))
    if contact: fail.append("contact: pedestrian")
    elif not entered and slowing: fail.append("unnecessary slowing")
    elif entered and min_d < p["clear_req"]: fail.append("pedestrian clearance")
    if phys: fail.append("physics: braking > cap")
    return dict(min_front_gap=min_d, min_rear_gap=99.0, contact=contact is not None, rel_speed=contact[2] if contact else 0.0, stopped=stopped, v_pass=v_pass,
                min_speed=float(min(r[1] for r in rows)), entered_lane=entered, slowing=slowing, verdict="FAIL" if fail else "PASS", reason="; ".join(fail), rows=rows)
