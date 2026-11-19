"""Fast surrogate of RC03 (spun car, lateral escape): same EscapePolicy + LowLevelController + fitted longitudinal/lateral plants as RC05, scripted actors from
rcx/rc03_laws.py (shared with the CARLA scene), true convex footprint distances. Starts at the cue (rt = 0), ego at s = 0 in its lane centre."""
import math
import numpy as np
from rc01.config import PolicyConfig
from rc01.controller import LowLevelController
from rcx.policies import EscapePolicy
from rcx import rc03_laws as L
from .plant import LongPlant
from .lane_core import MAX_STEER, LAT_K, LAT_TAU, EGO_HL, EGO_HW, _rect, quad_quad_dist

SCN = dict(L.DEFAULTS, duration_s=5.0, min_clear=0.5, settle_tol=0.6, v_bias=-0.5, it0=0.25, clear_pad=1.2, clear_pad_lc=0.3, lead_pad=3.5, sur_ttc=3.0)   # v_bias: measured CARLA ego speed at the cue is 0.5 m/s BELOW the target at 18 and 22 m/s (17.43, 21.53; it then catches up); clear_pad: extra surrogate margin on stopping


def _quad(s, t, yaw, hl, hw): return _rect(s, t, yaw, hl, hw)


def simulate(theta_s, policy_kw, plant: LongPlant, calib, pcfg=None, dt=0.05, policy_cls=EscapePolicy):
    p = dict(SCN); p.update(theta_s); pcfg = pcfg or PolicyConfig(target_speed=p["v_ego"])
    kw = dict(policy_kw); kw.setdefault("v_target", p["v_ego"]); kw.setdefault("lane_w", p["lane_w"]); kw.setdefault("esc_dir", -float(p["side"]))
    pol = policy_cls(dt, **kw); pol.reset()
    ctrl = LowLevelController(pcfg, calib, 0.0, MAX_STEER, dt); plant.reset(); ctrl._it = p["it0"]      # in CARLA the measured accel is -4 m/s2 for the first ticks after the cue (start transient), which saturates the controller accel-tracking integrator at +0.25 within 0.1 s; mimicked here
    W = p["lane_w"]; s, t, yaw, v, a_prev, st_f = 0.0, 0.0, 0.0, p["v_ego"] + p["v_bias"], 0.0, 0.0
    names = ["victim", "esc_lead", "esc_follow"] + (["rear"] if p["rear"] else [])
    mn = {n: 99.0 for n in names}; contact = None; rows = []; peak_lat = 0.0; peak_dec = 0.0; stop_t = None
    for n in range(int(p["duration_s"] / dt)):
        rt = n * dt
        o = dict(t=rt, v=v, s=s, ego_t=t, cue=True, hazard=True)
        out = pol.step(o); c = ctrl.step(out, t, math.degrees(yaw), v, a_prev)
        st_f += dt / (LAT_TAU + dt) * (c["steer"] - st_f)
        v2, a_prev = plant.step(v, c["throttle"], c["brake"], c["hand_brake"])
        yr = LAT_K * 0.5 * (v + v2) * st_f
        yaw += yr * dt; t += 0.5 * (v + v2) * math.sin(yaw) * dt; s += 0.5 * (v + v2) * math.cos(yaw) * dt
        peak_lat = max(peak_lat, abs(0.5 * (v + v2) * yr)); peak_dec = max(peak_dec, -a_prev); v = v2
        rn = rt + dt; fe = _rect(s, t, yaw, EGO_HL, EGO_HW)
        vs, vt, vy, vv = L.victim(p, rn)
        acts = {"victim": _quad(vs, vt, vy, L.HL_VICTIM, L.HW_VICTIM), "esc_lead": _quad(*L.esc_leader(p, rn)[:2], 0.0, L.HL_ESC, L.HW_ESC),
                "esc_follow": _quad(*L.esc_follower(p, rn)[:2], 0.0, L.HL_ESC, L.HW_ESC)}
        if p["rear"]: acts["rear"] = _quad(*L.rear_car(p, rn)[:2], 0.0, L.HL_REAR, L.HW_REAR)
        d = {k: quad_quad_dist(fe, q) for k, q in acts.items()}
        for k, x in d.items(): mn[k] = min(mn[k], x)
        rows.append((rn, v, t, math.degrees(yaw), min(d.values()), d["victim"], 0.0, a_prev))
        if v < 0.05 and stop_t is None: stop_t = rn
        hit = [k for k, x in d.items() if x <= 0.02]
        if hit:
            rel = {"victim": vv, "esc_lead": L.esc_leader(p, rn)[2], "esc_follow": p["v_esc"], "rear": p["v_ego"]}[hit[0]]
            contact = (hit[0], rn, abs(v - rel) if hit[0] != "victim" else math.hypot(v - vv * math.cos(vy), vv * math.sin(vy))); break
    R = np.array(rows); et = R[:, 2]
    vq = acts["victim"]; closing = (not contact) and L.still_closing(min(q[0] for q in vq) - (s + EGO_HL), v, abs(t - float(np.mean([q[1] for q in vq]))) - EGO_HW - L.HW_VICTIM, p["sur_ttc"])
    stopped = bool(R[-1, 1] < 0.3)
    settled = stopped or bool(np.all(np.abs(et[-10:]) < p["settle_tol"]) or np.all(np.abs(et[-10:] + p["side"] * W) < p["settle_tol"]))
    off_road = bool(np.abs(et).max() > 1.5 * W)
    mc = min(mn.values())
    fail = []
    if contact: fail.append("contact with " + contact[0])
    else:
        for k, x in mn.items():
            if x < p["min_clear"] + {"victim": p["clear_pad"] if kw.get("mode") == "brake" else p["clear_pad_lc"], "esc_lead": p["lead_pad"]}.get(k, 0.0): fail.append(f"{k} clearance")      # pad on the actors ahead (victim: stopping bias; esc_lead: residual acceleration-model error while closing on it), none on the one behind
        if not settled: fail.append("not settled")
        if closing: fail.append("still closing on the victim")
        for k, fn in (("esc_follow", L.esc_follower), ("rear", L.rear_car)):
            if k in acts and L.rear_closing((s - EGO_HL) - (fn(p, rn)[0] + (L.HL_ESC if k == "esc_follow" else L.HL_REAR)), fn(p, rn)[2] - v,
                                            abs(t - fn(p, rn)[1]) - EGO_HW - (L.HW_ESC if k == "esc_follow" else L.HW_REAR), p["sur_ttc"]): fail.append(f"still closing from {k}")
    if off_road: fail.append("ego left the road")
    if peak_lat > 9.81: fail.append("lateral envelope")
    af = R[:, 7]; pf = max([-af[max(0, i - 3):i + 1].mean() for i in range(len(af)) if R[i, 1] >= 1.6] or [0.0])      # ignore the low-speed stop snap (known artefact, parked)
    if pf > 9.81: fail.append("braking envelope")
    return dict(min_clear=mc, min_clear_by=mn, min_front_gap=mn["victim"], min_rear_gap=mn.get("rear", 99.0), contact=contact is not None, contact_actor=contact[0] if contact else None,
                rel_speed=contact[2] if contact else 0.0, settled=settled, stopped=stopped, peak_lat=peak_lat, peak_decel=peak_dec, peak_filt=float(pf), final_t=float(et[-1]),
                verdict="FAIL" if fail else "PASS", reason="; ".join(fail), rows=rows)
