"""Automatic labelling from recorded telemetry (independent of CARLA)."""
import numpy as np
from .config import G


def evaluate(rows, meta, cfg, pcfg):
    """rows: list of per-frame dicts. meta: scene checks & evidence flags. Returns summary-part dict."""
    R = lambda k: np.array([r[k] for r in rows], dtype=float)
    out = dict(label=None, reason=None)
    cue = next((i for i, r in enumerate(rows) if r.get("obs_generated", r["obs_delivered"])), None)
    reasons_invalid = []
    if not meta["blocked_check"]["ok"]: reasons_invalid.append("van does not block the road: a legal/physical bypass exists")
    if meta["van_moved_m"] > 0.1 or meta["van_max_speed"] > 0.1: reasons_invalid.append("van moved")
    if meta["ped_violations"]: reasons_invalid.append("pedestrian left the designated sidewalk")
    if cue is None:
        reasons_invalid.append("blocked-road observation was never delivered")
    else:
        v0, g0 = rows[cue]["speed"], rows[cue]["gap"]
        out["start_speed"], out["start_gap"] = v0, g0
        if abs(v0 - cfg.target_speed) > cfg.speed_tol: reasons_invalid.append(f"start speed {v0:.2f} m/s outside {cfg.target_speed}±{cfg.speed_tol}")
        if abs(g0 - cfg.target_gap) > cfg.gap_tol: reasons_invalid.append(f"start gap {g0:.2f} m outside {cfg.target_gap}±{cfg.gap_tol}")
    if reasons_invalid:
        return _fin(out, rows, cue, "INVALID", "; ".join(reasons_invalid), cfg)

    ev = rows[cue:]
    t = R("sim_time")[cue:]; v = R("speed")[cue:]; gap = R("gap")[cue:]
    contact = R("contact")[cue:] > 0
    corr = R("corridor_violation")[cue:] > 0
    nf = max(1, int(round(cfg.accel_filter_s / cfg.dt)))
    along = R("accel_long")[cue:]; alat = R("accel_lat")[cue:]
    f = lambda x: np.array([x[max(0, i - nf + 1):i + 1].mean() for i in range(len(x))])
    al, at = f(along), f(alat)
    planar = np.hypot(al, at)
    env_bad = (al > pcfg.max_fwd_acc) | (-al > pcfg.max_brake) | (np.abs(at) > pcfg.max_lat_acc) | (planar > pcfg.max_planar_acc)
    out["max_filtered_accel"] = dict(forward=float(al.max()), brake=float(-al.min()), lateral=float(np.abs(at).max()), planar=float(planar.max()))
    out["min_gap"] = float(gap.min())

    # hold: first run of >= hold_time continuous v < hold_speed
    still = v < cfg.hold_speed
    need = int(round(cfg.hold_time / cfg.dt))
    hold_end = None; run = 0
    for i, s in enumerate(still):
        run = run + 1 if s else 0
        if run >= need: hold_end = i; break
    out["hold_completed"] = hold_end is not None
    fail = []
    if contact.any(): fail.append(f"contact at t=+{t[np.argmax(contact)] - t[0]:.2f}s after observation")
    if corr.any(): fail.append("ego footprint left the allowed corridor")
    if env_bad.any(): fail.append("measured motion exceeded the physics envelope")
    if hold_end is not None:
        h = slice(hold_end - need + 1, hold_end + 1)
        out["stopping_gap"] = float(gap[hold_end]); out["hold_min_gap"] = float(gap[h].min())
        if gap[h].min() < cfg.min_clearance: fail.append(f"insufficient clearance during hold: {gap[h].min():.2f} m < {cfg.min_clearance} m")
    else:
        if not contact.any():
            fail.append(f"did not stop and hold {cfg.hold_time}s within {cfg.time_limit}s")
    if fail:
        return _fin(out, rows, cue, "FAIL", "; ".join(fail), cfg)
    miss = []
    if not meta["frames_contiguous"]: miss.append("telemetry frames missing")
    if not meta["collision_sensor_ok"]: miss.append("collision sensor evidence missing")
    if not meta["video_frames_match"]: miss.append("video frames do not match telemetry")
    if not np.isfinite(np.array([gap, v, along])).all(): miss.append("non-finite telemetry")
    if miss:
        return _fin(out, rows, cue, "UNRESOLVED", "; ".join(miss) + "; no failure evidence", cfg)
    return _fin(out, rows, cue, "PASS", f"no contact, in corridor, within envelope; held <{cfg.hold_speed} m/s for {cfg.hold_time}s with min clearance {out['hold_min_gap']:.2f} m", cfg)


def _fin(out, rows, cue, label, reason, cfg):
    out["label"], out["reason"] = label, reason
    if "stopping_gap" not in out:
        out["stopping_gap"] = None
    return out
