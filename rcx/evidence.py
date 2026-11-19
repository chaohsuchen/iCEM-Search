"""Per-run evidence writer shared by all scenes (see docs/RUN_SCHEMA.md).
Separates what the ego can observe from simulator ground truth:
  ego_observable.csv  ego proprioception + what the ego perceives (visible actors only, ego-relative) + requested / applied controls
  ground_truth.csv    true states of every actor and the scene-specific measurements (hidden information, NOT for the world model input)
  decisions.jsonl     one record per decision tick: query time, observation, observation-history window, policy output, controller state, pending commands
(The legacy mixed telemetry.csv is no longer written: ego_observable.csv + ground_truth.csv together contain every column.)"""
import json, math
import numpy as np

SCHEMA = "rcx-run-1"
VIS_RANGE = 80.0          # m, euclidean; an actor beyond it is not perceived
HIST_TICKS = 20           # observation-history window (ticks) referenced by each decision record


def _seg_hits_poly(p, q, poly):
    """Does segment p->q cross the convex polygon poly (rows = vertices)? (Liang-Barsky on the hull edges via separating test.)"""
    from .common import _hull
    H = _hull(poly[:, :2]); d = q - p; t0, t1 = 0.0, 1.0
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]; e = b - a; n = np.array([e[1], -e[0]])        # outward normal for CCW hull
        num = n @ (a - p); den = n @ d
        if abs(den) < 1e-12:
            if num < 0: return False
        else:
            t = num / den
            if den > 0: t1 = min(t1, t)
            else: t0 = max(t0, t)
            if t0 > t1: return False
    return True


def actor_truth(scn, ego_st):
    """Ground truth of every non-ego actor: lane-frame centre, yaw, velocity components, footprint, clearance to the ego."""
    from .common import fp_local, poly_dist
    out = {}; seen = {}
    for a in list(scn.actors) + list(scn.walkers):
        if a.id == scn.ego.id: continue
        name = scn.roles.get(a.id, a.type_id.split(".")[-1]); k = seen.get(name, 0); seen[name] = k + 1
        if k: name = f"{name}{k}"
        fp = fp_local(a, scn.frame); s, t = float(fp[:, 0].mean()), float(fp[:, 1].mean())
        v = a.get_velocity(); f = scn.frame.fwd(s); n = np.array([-f[1], f[0]])
        vs, vt = float(v.x * f[0] + v.y * f[1]), float(-(v.x * n[0] + v.y * n[1]))        # t is to the right of travel
        yaw = scn.frame.yaw_rel(a.get_transform().rotation.yaw, s)
        out[name] = dict(s=s, t=t, yaw_rel=float(yaw), vs=vs, vt=vt, fp=fp, clear=float(poly_dist(ego_st["fp"], fp)),
                         length=float(2 * a.bounding_box.extent.x), width=float(2 * a.bounding_box.extent.y))
    return out


def perceive(ego_st, truth):
    """Ego-relative view of the actors the ego can see: within VIS_RANGE and not hidden behind another tracked actor (line of sight centre -> centre)."""
    ec = np.array([float(ego_st["fp"][:, 0].mean()), float(ego_st["fp"][:, 1].mean())]); view = {}
    for n, g in truth.items():
        c = np.array([g["s"], g["t"]]); vis = float(np.linalg.norm(c - ec)) <= VIS_RANGE
        if vis:
            for m, o in truth.items():
                if m != n and _seg_hits_poly(ec, c, o["fp"]): vis = False; break
        view[n] = dict(visible=int(vis), rel_s=g["s"] - ec[0] if vis else "", rel_t=g["t"] - ec[1] if vis else "", rel_vs=g["vs"] - ego_st["v"] * math.cos(math.radians(ego_st["yaw_rel"])) if vis else "",
                       rel_vt=g["vt"] if vis else "", yaw_rel=g["yaw_rel"] if vis else "", length=g["length"] if vis else "", width=g["width"] if vis else "")
    return view


OBS_KEYS = ("frame", "sim_time", "t_since_cue", "phase", "ego_s", "ego_t", "ego_yaw_rel", "speed", "accel_long", "accel_lat", "accel_long_f", "accel_lat_f", "accel_planar_f",
            "a_req", "a_req_clipped", "t_des", "throttle", "brake", "steer", "hand_brake", "contact")


def split_rows(rows, views, truths, obs_dicts):
    """-> (ego_observable rows, ground_truth rows). A row's key is observable only if listed in OBS_KEYS; every other key of the legacy row is scene ground truth."""
    eo, gt = [], []
    for r, vw, tr, od in zip(rows, views, truths, obs_dicts):
        o = {k: r[k] for k in OBS_KEYS if k in r}
        for k, x in (od or {}).items():
            if isinstance(x, (int, float, bool, str)) and not isinstance(x, np.generic) or isinstance(x, np.generic): o["obs_" + k] = float(x) if isinstance(x, np.generic) else x
        for n, d in vw.items():
            for k, x in d.items(): o[f"seen_{n}_{k}"] = x
        g = {"frame": r["frame"], "sim_time": r["sim_time"], "contact_actor": r.get("contact_actor", "")}
        for k, x in r.items():
            if k not in OBS_KEYS and k not in g: g[k] = x
        for n, d in tr.items():
            for k in ("s", "t", "yaw_rel", "vs", "vt", "clear", "length", "width"): g[f"actor_{n}_{k}"] = d[k]
        eo.append(o); gt.append(g)
    return eo, gt


def decision_record(frame, t, t_cue, obs, out, ctrl, applied, pending, hist_frames):
    """One decision tick. pending = commands issued within the actuator delay window (issued, maybe not yet effective)."""
    cs = {k: getattr(ctrl, k) for k in ("_ib", "_brk_frames", "_it", "_thr_prev") if hasattr(ctrl, k)}; cs["_hist"] = list(getattr(ctrl, "_hist", []))
    return dict(frame=frame, query_time=t, t_since_cue=t_cue, observation=obs, observation_history_frames=hist_frames,
                policy_output=dict(phase=out.phase, a_req=out.a_req, t_des=out.t_des, t_rate=getattr(out, "t_rate", None), lane_keep=getattr(out, "lane_keep", None), info=out.info or {}),
                controller_state=cs, applied_command=applied, pending_commands=pending)


def write_all(out_dir, rows, views, truths, obs_dicts, decisions):
    import csv
    eo, gt = split_rows(rows, views, truths, obs_dicts)
    for name, rr in (("ego_observable.csv", eo), ("ground_truth.csv", gt)):
        cols = list(dict.fromkeys(k for r in rr for k in r))                       # union of keys (actors may be missing on some ticks)
        with open(f"{out_dir}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, cols, restval=""); w.writeheader(); w.writerows(rr)
    with open(f"{out_dir}/decisions.jsonl", "w") as f:
        for d in decisions: f.write(json.dumps(d, default=lambda x: float(x) if isinstance(x, np.generic) else str(x)) + "\n")
