"""Straight-line braking calibration on the same vehicle and road surface (empty street, before the scored run)."""
import math, json
import numpy as np
import carla
from .scene import footprint


import os
SUBSTEP = float(os.environ.get("RC_SUBSTEP", "0.01"))                 # physics substep (s); 0.002 removes the wheel-lock cliff
BRAKE_CAP_MAX = float(os.environ.get("RC_BRAKE_CAP_MAX", "1.0"))      # upper bound on the usable brake command
BRAKE_SCALE = float(os.environ.get("RC_BRAKE_SCALE", "1.0"))          # scales every wheel's max brake torque (vehicle physics tweak, identical for calibration and scored runs)


def physics_settings(s, dt):
    s.substepping, s.max_substep_delta_time, s.max_substeps = True, SUBSTEP, int(dt / SUBSTEP + 0.999) + 2
    return s


def apply_brake_scale(ego):
    if BRAKE_SCALE != 1.0:
        pc = ego.get_physics_control(); ws = pc.wheels
        for w in ws: w.max_brake_torque *= BRAKE_SCALE
        pc.wheels = ws; ego.apply_physics_control(pc)


def spawn_ego(world, cfg, x, speed=0.0, frame=None, t=None):
    bp = world.get_blueprint_library().find(cfg.ego_bp)
    bp.set_attribute("role_name", "ego")
    if bp.has_attribute("color"): bp.set_attribute("color", "200,30,30")
    if frame is None:
        tf = carla.Transform(carla.Location(x, cfg.lane_center, 0.4))
    else:
        tf = frame.transform(x, 0.0 if t is None else t, 0.4)
    ego = world.spawn_actor(bp, tf)
    apply_brake_scale(ego)
    for _ in range(15):
        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True)); world.tick()
    if speed:
        f = (1.0, 0.0) if frame is None else frame.fwd(x)
        ego.set_target_velocity(carla.Vector3D(speed * f[0], speed * f[1], 0))
    return ego


def max_steer_rad(ego):
    return math.radians(max(w.max_steer_angle for w in ego.get_physics_control().wheels))


def _speed(a):
    v = a.get_velocity(); return math.hypot(v.x, v.y)


def _lane_steer(ego, frame, max_steer):
    if frame is None: return 0.0
    l = ego.get_location(); s, t = frame.to_local(l.x, l.y)
    yaw = math.radians(frame.yaw_rel(ego.get_transform().rotation.yaw, s)); v = _speed(ego)
    return float(np.clip((-yaw - math.atan2(t, v + 2.0)) / max_steer, -1, 1))


def _trial(world, cfg, brake, x0=20.0, frame=None):
    ego = spawn_ego(world, cfg, x0, cfg.target_speed, frame)
    ms = max_steer_rad(ego)
    thr_hist, I = [], 0.0
    for _ in range(int(5.0 / cfg.dt)):            # settle at cruise speed with a simple PI
        v = _speed(ego)
        I += (cfg.target_speed - v) * cfg.dt * 0.6
        thr = float(np.clip(0.25 + 0.5 * (cfg.target_speed - v) + I, 0, 1)); thr_hist.append(thr)
        ego.apply_control(carla.VehicleControl(throttle=thr, steer=_lane_steer(ego, frame, ms))); world.tick()
    cruise = float(np.mean(thr_hist[-40:]))
    v_prev = _speed(ego)
    ts, vs = [0.0], [v_prev]
    for k in range(1, int(6.0 / cfg.dt)):
        ego.apply_control(carla.VehicleControl(brake=brake, steer=_lane_steer(ego, frame, ms))); world.tick()
        v = _speed(ego); ts.append(k * cfg.dt); vs.append(v)
        if v < 0.2: break
    ego.destroy(); world.tick()
    ts, vs = np.array(ts), np.array(vs)
    a = -np.diff(vs) / cfg.dt                      # decel per tick
    i0 = np.argmax(vs <= 6.5); i1 = np.argmax(vs <= 1.5) if (vs <= 1.5).any() else len(vs) - 1
    steady = float((vs[i0] - vs[i1]) / max(ts[i1] - ts[i0], 1e-6))
    onset = int(np.argmax(a >= 0.1 * steady)) if steady > 0 else 0
    t90 = int(np.argmax(a >= 0.9 * steady))
    tick_dec = -np.diff(vs) / cfg.dt
    hi = vs[:-1] > 2.5                                   # lock test above the low-speed region
    lock = bool(hi.any() and tick_dec[hi].max() > 9.81)
    big = int(np.argmax(tick_dec)); snap_v = float(vs[big]) if tick_dec[big] > 9.81 else None
    return dict(brake=brake, decel=steady, locks=lock, snap_from_speed=snap_v, max_tick_decel=float(tick_dec.max()),
                mean_decel_hi=float((vs[0] - vs[hi.sum()]) / (hi.sum() * cfg.dt)) if hi.sum() else 0.0, onset_s=(onset + 1) * cfg.dt, t90_s=(t90 + 1) * cfg.dt,
                peak_decel=float(a.max()), v_series=[round(float(x),3) for x in vs], cruise_throttle=cruise, v0=float(vs[0]))


def calibrate(world, cfg, path=None, frame=None, x0=20.0):
    trials = [_trial(world, cfg, b, x0, frame) for b in (0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0)]
    ok = [t for t in trials if not t["locks"] and t["brake"] <= BRAKE_CAP_MAX + 1e-9]
    # smooth (non-locking) brake range; decel taken over the speed range above the low-speed snap region
    cal = dict(brake_cmd=[t["brake"] for t in ok], brake_decel=[t["mean_decel_hi"] for t in ok],
               brake_cap=float(min(BRAKE_CAP_MAX, max(t["brake"] for t in ok))), max_decel=float(max(t["mean_decel_hi"] for t in ok)),
               locking_brakes=[t["brake"] for t in trials if t["locks"]],
               low_speed_snap_from=[t["snap_from_speed"] for t in trials],
               brake_delay_s=float(np.median([t["onset_s"] for t in trials])),
               brake_buildup_90_s=float(np.median([t["t90_s"] for t in trials])),
               cruise_throttle=float(np.median([t["cruise_throttle"] for t in trials])), coast_decel=float(trials[0]["mean_decel_hi"]), ego_bp=cfg.ego_bp, cruise_speed=float(cfg.target_speed), trials=trials)
    if path:
        json.dump(cal, open(path, "w"), indent=2)
    return cal
