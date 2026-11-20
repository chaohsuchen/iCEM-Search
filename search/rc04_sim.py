"""Fast surrogate of RC04 / RC06 (blocked lane, side vehicle alongside, its follower): same EscapePolicy + LowLevelController + fitted plants as rc03_sim, scripted actors from
rcx/rc04_laws.py (shared with the CARLA scene), true convex footprint distances. Starts at the cue (rt = 0), ego at s = 0 in its lane centre."""
import math
import numpy as np
from rc01.config import PolicyConfig
from rc01.controller import LowLevelController
from rcx.policies import EscapePolicy
from rcx import rc04_laws as L
from .plant import LongPlant
from .lane_core import MAX_STEER, LAT_K, LAT_TAU, EGO_HL, EGO_HW, _rect, quad_quad_dist

SCN = dict(L.DEFAULTS, duration_s=7.0, min_clear=0.5, settle_tol=0.6, v_bias=-0.5, it0=0.25, obs_pad=1.4, obs_pad_lc=0.3, side_pad=0.3, fol_pad=0.3, sur_ttc=3.0)   # pads start at the rc03 values; re-measured against CARLA


def simulate(theta_s, policy_kw, plant: LongPlant, calib, pcfg=None, dt=0.05, policy_cls=EscapePolicy):
    p = dict(SCN); p.update(theta_s); pcfg = pcfg or PolicyConfig(target_speed=p["v_ego"])
    kw = dict(policy_kw); kw.setdefault("v_target", p["v_ego"]); kw.setdefault("lane_w", p["lane_w"]); kw.setdefault("esc_dir", -float(p["side"]))
    pol = policy_cls(dt, **kw); pol.reset()
    ctrl = LowLevelController(pcfg, calib, 0.0, MAX_STEER, dt); plant.reset(); ctrl._it = p["it0"]
    W = p["lane_w"]; s, t, yaw, v, a_prev, st_f = 0.0, 0.0, 0.0, p["v_ego"] + p["v_bias"], 0.0, 0.0
    names = ["obstacle", "side"] + (["fol"] if p["fol"] else [])
    mn = {n: 99.0 for n in names}; contact = None; rows = []; peak_lat = 0.0; peak_dec = 0.0
    for n in range(int(p["duration_s"] / dt)):
        rt = n * dt
        o = dict(t=rt, v=v, s=s, ego_t=t, cue=True, hazard=True)
        out = pol.step(o)
        if out.a_req > 0.0: out.a_req *= p.get("acc_gain", 1.0)
        c = ctrl.step(out, t, math.degrees(yaw), v, a_prev)
        st_f += dt / (LAT_TAU + dt) * (c["steer"] - st_f)
        v2, a_prev = plant.step(v, c["throttle"], c["brake"], c["hand_brake"])
        yr = LAT_K * 0.5 * (v + v2) * st_f
        yaw += yr * dt; t += 0.5 * (v + v2) * math.sin(yaw) * dt; s += 0.5 * (v + v2) * math.cos(yaw) * dt
        peak_lat = max(peak_lat, abs(0.5 * (v + v2) * yr)); peak_dec = max(peak_dec, -a_prev); v = v2
        rn = rt + dt; fe = _rect(s, t, yaw, EGO_HL, EGO_HW)
        os_, ot, _ = L.obstacle(p, rn); ss, st_, sv = L.side_vehicle(p, rn)
        acts = {"obstacle": _rect(os_, ot, 0.0, L.HL_OBS, L.HW_OBS), "side": _rect(ss, st_, 0.0, p["side_hl"], p["side_hw"])}
        if p["fol"]: fs, ft, fv = L.follower(p, rn); acts["fol"] = _rect(fs, ft, 0.0, L.HL_FOL, L.HW_FOL)
        d = {k: quad_quad_dist(fe, q) for k, q in acts.items()}
        for k, x in d.items(): mn[k] = min(mn[k], x)
        rows.append((rn, v, t, math.degrees(yaw), min(d.values()), d["obstacle"], 0.0, a_prev))
        hit = [k for k, x in d.items() if x <= 0.02]
        if hit:
            rel = {"obstacle": 0.0, "side": sv, "fol": fv if p["fol"] else 0.0}[hit[0]]
            contact = (hit[0], rn, abs(v - rel)); break
    R = np.array(rows); et = R[:, 2]
    oq = acts["obstacle"]; closing = (not contact) and L.still_closing(min(q[0] for q in oq) - (s + EGO_HL), v, abs(t - float(np.mean([q[1] for q in oq]))) - EGO_HW - L.HW_OBS, p["sur_ttc"])
    stopped = bool(R[-1, 1] < 0.3)
    settled = stopped or bool(np.all(np.abs(et[-10:]) < p["settle_tol"]) or np.all(np.abs(et[-10:] + p["side"] * W) < p["settle_tol"]))
    off_road = bool(np.abs(et).max() > 1.5 * W)
    mc = min(mn.values()); fail = []
    brake_like = kw.get("mode") == "brake" or (kw.get("mode") == "free" and float(np.abs(et).max()) < 0.5 * W)      # free policy that never leaves its lane: stopping-in-lane pad (rc04 families: brake)
    if contact: fail.append("contact with " + contact[0])
    else:
        for k, x in mn.items():
            if x < p["min_clear"] + {"obstacle": p["obs_pad"] if brake_like else p["obs_pad_lc"], "side": p["side_pad"], "fol": p["fol_pad"]}.get(k, 0.0): fail.append(f"{k} clearance")
        if not settled: fail.append("not settled")
        if closing: fail.append("still closing on the obstacle")
        if p["fol"] and L.rear_closing((s - EGO_HL) - (fs + L.HL_FOL), fv - v, abs(t - ft) - EGO_HW - L.HW_FOL, p["sur_ttc"]): fail.append("still closing from fol")
    if off_road: fail.append("ego left the road")
    if peak_lat > 9.81: fail.append("lateral envelope")
    af = R[:, 7]; pf = max([-af[max(0, i - 3):i + 1].mean() for i in range(len(af)) if R[i, 1] >= 1.6] or [0.0])
    if pf > 9.81: fail.append("braking envelope")
    return dict(min_clear=mc, min_clear_by=mn, min_front_gap=mn["obstacle"], min_rear_gap=mn.get("fol", 99.0), contact=contact is not None, contact_actor=contact[0] if contact else None,
                rel_speed=contact[2] if contact else 0.0, settled=settled, stopped=stopped, peak_lat=peak_lat, peak_decel=peak_dec, peak_filt=float(pf), final_t=float(et[-1]),
                verdict="FAIL" if fail else "PASS", reason="; ".join(fail), rows=rows)
