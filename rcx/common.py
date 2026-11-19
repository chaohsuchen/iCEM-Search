"""Shared engine for RC05 / RC09 / RC13: frames, scripted NPCs, run loop, outputs."""
import csv, json, math, os, queue, time
import numpy as np
import carla
from rc01.config import PolicyConfig, G
from rc01.controller import LowLevelController
from rc01.calibrate import max_steer_rad
from rc01 import video
from . import evidence


class Frame:
    """Curvilinear (Frenet) frame along a lane centre-line: s = arc length along travel, t = lateral offset to the RIGHT of travel.
    Follows the real lane geometry, so scripted vehicles stay in their lane on gently curving roads."""
    EXT = 300.0

    def __init__(self, xy, z):
        xy = np.asarray(xy, float); z = np.asarray(z, float)
        d = np.diff(xy, axis=0); seg = np.hypot(d[:, 0], d[:, 1]); keep = np.concatenate([[True], seg > 1e-6])
        xy, z = xy[keep], z[keep]; d = np.diff(xy, axis=0); seg = np.hypot(d[:, 0], d[:, 1])
        h0, h1 = d[0] / seg[0], d[-1] / seg[-1]
        self.P = np.vstack([xy[0] - h0 * self.EXT, xy, xy[-1] + h1 * self.EXT])
        self.Z = np.concatenate([[z[0]], z, [z[-1]]])
        dd = np.diff(self.P, axis=0); self.L = np.hypot(dd[:, 0], dd[:, 1]); self.H = dd / self.L[:, None]
        self.S = np.concatenate([[0.0], np.cumsum(self.L)]) - self.EXT           # s = 0 at the first real sample
        self.YAW = np.degrees(np.arctan2(self.H[:, 1], self.H[:, 0]))
        self.length = float(self.S[-2])

    @classmethod
    def from_lane(cls, cmap, road, lane, s0, ds=1.0, max_len=500.0, off=0.0):
        pts, zs, s = [], [], s0
        step = ds if lane < 0 else -ds                     # negative lane ids travel with increasing OpenDRIVE s
        for _ in range(int(max_len / ds)):
            try: wp = cmap.get_waypoint_xodr(road, lane, s)
            except RuntimeError: wp = None
            if wp is None or wp.road_id != road: break
            l = wp.transform.location; pts.append((l.x, l.y)); zs.append(l.z); s += step
        if off:                                            # shift the reference line `off` metres to the RIGHT of the lane centre (negative = left)
            P = np.asarray(pts, float); H = np.gradient(P, axis=0); H /= np.hypot(H[:, 0], H[:, 1])[:, None]
            P = P + off * np.stack([-H[:, 1], H[:, 0]], 1); pts = [tuple(q) for q in P]
        return cls(pts, zs)

    def _idx(self, s):
        return int(np.clip(np.searchsorted(self.S, s, side="right") - 1, 0, len(self.L) - 1))

    def heading(self, s): return float(self.YAW[self._idx(s)])

    def fwd(self, s):
        h = self.H[self._idx(s)]; return (float(h[0]), float(h[1]))

    def z(self, s): return float(np.interp(s, self.S, self.Z))

    def to_world(self, s, t):
        i = self._idx(s); h = self.H[i]; p = self.P[i] + h * (s - self.S[i])
        return float(p[0] - t * h[1]), float(p[1] + t * h[0])

    def to_local(self, x, y):
        q = np.array([x, y]); A = self.P[:-1]; AB = self.P[1:] - A
        u = np.clip(((q - A) * AB).sum(1) / (self.L ** 2), 0.0, 1.0)
        proj = A + AB * u[:, None]; dist = np.hypot(*(q - proj).T); i = int(np.argmin(dist))
        h = self.H[i]; r = np.array([-h[1], h[0]])
        return float(self.S[i] + u[i] * self.L[i]), float((q - proj[i]) @ r)

    def transform(self, s, t, dz=0.4, dyaw=0.0):
        x, y = self.to_world(s, t)
        return carla.Transform(carla.Location(x, y, self.z(s) + dz), carla.Rotation(yaw=self.heading(s) + dyaw))

    def yaw_rel(self, yaw_deg, s=0.0):
        return ((yaw_deg - self.heading(s) + 180) % 360) - 180

    @property
    def o(self):
        x, y = self.to_world(0.0, 0.0); return (x, y, self.z(0.0))

    @property
    def yaw(self): return self.heading(0.0)

    @property
    def f(self): return self.fwd(0.0)


def _hull(p):
    p = sorted(map(tuple, p))
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cr(up[-2], up[-1], q) <= 0: up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def poly_dist(fa, fb):
    """Distance between the convex footprints of two actors (0 if they overlap). Inputs: (n,2+) arrays of (s,t)."""
    A, B = _hull(np.asarray(fa)[:, :2]), _hull(np.asarray(fb)[:, :2])
    sep = False
    for P in (A, B):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]; n = np.array([-e[1], e[0]]); n = n / (np.linalg.norm(n) + 1e-12)
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min(): sep = True
    if not sep: return 0.0
    def pt_seg(p, a, b):
        ab = b - a; u = np.clip(((p - a) @ ab) / (ab @ ab + 1e-12), 0, 1); return np.linalg.norm(p - (a + u * ab))
    best = 1e9
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for j in range(len(Q)): best = min(best, pt_seg(p, Q[j], Q[(j + 1) % len(Q)]))
    return float(best)


def fp_local(actor, frame):
    v = actor.bounding_box.get_world_vertices(actor.get_transform())
    return np.array([frame.to_local(p.x, p.y) for p in v])      # (8,2) columns s,t


def speed_of(actor):
    v = actor.get_velocity(); return math.hypot(v.x, v.y)


class NPC:
    """Scripted vehicle: velocity imposed along the lane by a speed profile of scenario-relative time."""
    def __init__(self, actor, frame, speed_fn=None, role="npc"):
        self.a, self.frame, self.speed_fn, self.role = actor, frame, speed_fn, role

    def init_pose(self, dt=0.05):
        loc = self.a.get_location(); self.s0, self.t_lat = self.frame.to_local(loc.x, loc.y); self.z0 = loc.z
        self.s_cur, self.dt = self.s0, dt

    def teleport_ahead(self, dist):
        pass                                             # motion is integrated kinematically in update()

    def update(self, rel_t, warm=False):
        if self.speed_fn is None:
            self.a.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True)); return
        v = max(0.0, self.speed_fn(rel_t))
        self.s_cur += v * self.dt                        # scripted behaviour = exact kinematic trajectory along the lane
        x, y = self.frame.to_world(self.s_cur, self.t_lat); f = self.frame.fwd(self.s_cur)
        z = self.z0 + self.frame.z(self.s_cur) - self.frame.z(self.s0) + 0.02
        self.a.set_transform(carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=self.frame.heading(self.s_cur))))
        self.a.set_target_velocity(carla.Vector3D(f[0] * v, f[1] * v, 0.0))
        self.a.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self.a.apply_control(carla.VehicleControl(throttle=0.6 if warm else 0.0, brake=0.0, steer=0.0))


def envelope(al, at, pcfg, dt, win_s=0.2):
    nf = max(1, int(round(win_s / dt)))
    f = lambda x: np.array([x[max(0, i - nf + 1):i + 1].mean() for i in range(len(x))])
    al, at = f(np.asarray(al, float)), f(np.asarray(at, float))
    pl = np.hypot(al, at)
    bad = (al > pcfg.max_fwd_acc) | (-al > pcfg.max_brake) | (np.abs(at) > pcfg.max_lat_acc) | (pl > pcfg.max_planar_acc)
    return bad, dict(forward=float(al.max()), brake=float(-al.min()), lateral=float(np.abs(at).max()), planar=float(pl.max()))


def run_engine(world, scn, policy, pcfg, calib, run_id, out_dir, meta_extra=None):
    """scn provides: build(), start(), step(st) -> (obs, extras), finished(rows, contacts), evaluate(rows, contacts, meta),
    overlay(row) -> (lines, banners), destroy(), params(), roles, ego, frame, dt, cfg(video size...)."""
    os.makedirs(out_dir, exist_ok=True)
    dt = scn.dt
    scn.build()
    for _ in range(50):
        scn.hold(); world.tick()
    scn.start()
    bl = world.get_blueprint_library()
    cb = bl.find("sensor.camera.rgb")
    cb.set_attribute("image_size_x", "1280"); cb.set_attribute("image_size_y", "720"); cb.set_attribute("fov", str(getattr(scn, "cam_fov", 100)))
    ego0, fr0 = scn.ego, scn.frame
    cam_tf = lambda: (scn.camera_pose() or _cam(ego0, fr0, scn))
    cam = world.spawn_actor(cb, cam_tf()); imgq = queue.Queue(); cam.listen(imgq.put)     # spawned at its final pose
    col = world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=scn.ego)
    colq = queue.Queue(); col.listen(colq.put)
    for _ in range(int(scn.warmup_s / dt)):          # spin wheels up and let the camera settle at its pose before recording
        scn.warm(calib["cruise_throttle"]); cam.set_transform(cam_tf()); world.tick()
        while not imgq.empty(): imgq.get()
    scn.warm_end()
    ctrl = LowLevelController(pcfg, calib, 0.0, max_steer_rad(scn.ego), dt)
    ego, fr = scn.ego, scn.frame
    policy.reset()
    images, rows, contacts = {}, [], {}
    views, truths, obs_dicts, decisions, applied_hist = [], [], [], [], []
    prev_v = ego.get_velocity(); hist_l, hist_t = [], []
    end_reason, end_after = None, None
    cam.set_transform(cam_tf())
    for n in range(scn.max_frames):
        frame = world.tick()
        im = imgq.get(timeout=30)
        while im.frame < frame: im = imgq.get(timeout=30)
        images[frame] = np.frombuffer(im.raw_data, np.uint8).reshape(720, 1280, 4)[:, :, 2::-1].copy()
        time.sleep(0.003)
        while not colq.empty():
            e = colq.get(); contacts[e.frame] = scn.roles.get(e.other_actor.id, e.other_actor.type_id)
        t = world.get_snapshot().timestamp.elapsed_seconds
        tf, vel = ego.get_transform(), ego.get_velocity()
        yaw = math.radians(tf.rotation.yaw)
        fwd, right = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        dv = np.array([vel.x - prev_v.x, vel.y - prev_v.y]) / dt; prev_v = vel
        a_l, a_t = float(dv @ fwd), float(dv @ right)
        s, tl = fr.to_local(tf.location.x, tf.location.y)
        st = dict(frame=frame, t=t, s=s, tl=tl, yaw_rel=fr.yaw_rel(tf.rotation.yaw, s), v=math.hypot(vel.x, vel.y),
                  a_long=a_l, a_lat=a_t, fp=fp_local(ego, fr), z=tf.location.z)
        obs, extras = scn.step(st)
        out = policy.step(obs)
        c = ctrl.step(out, tl, st["yaw_rel"], st["v"], a_l)
        ego.apply_control(carla.VehicleControl(throttle=c["throttle"], brake=c["brake"], steer=c["steer"], hand_brake=c["hand_brake"]))
        hist_l.append(a_l); hist_t.append(a_t)
        truth = evidence.actor_truth(scn, st); views.append(evidence.perceive(st, truth)); truths.append(truth)
        od = dict(obs) if isinstance(obs, dict) else {}; obs_dicts.append(od)
        applied = dict(frame=frame, throttle=c["throttle"], brake=c["brake"], steer=c["steer"], hand_brake=int(c["hand_brake"])); applied_hist.append(applied)
        decisions.append(evidence.decision_record(frame, t, (t - scn.t0) if scn.t0 is not None else None, od, out, ctrl, applied,
                                                  applied_hist[-ctrl.delay_frames:], [d["frame"] for d in decisions[-evidence.HIST_TICKS:]]))
        k = max(1, int(round(0.2 / dt)))
        af, atf = float(np.mean(hist_l[-k:])), float(np.mean(hist_t[-k:]))
        row = dict(frame=frame, sim_time=t, t_since_cue=(t - scn.t0) if scn.t0 is not None else "", phase=out.phase,
                   ego_s=s, ego_t=tl, ego_yaw_rel=st["yaw_rel"], speed=st["v"], a_req=out.a_req, a_req_clipped=c["a_req_clipped"],
                   t_des=out.t_des if out.t_des is not None else "", throttle=c["throttle"], brake=c["brake"], steer=c["steer"],
                   hand_brake=int(c["hand_brake"]), accel_long=a_l, accel_lat=a_t, accel_long_f=af, accel_lat_f=atf,
                   accel_planar_f=math.hypot(af, atf), contact=0, contact_actor="")
        row.update(extras); row.update(out.info or {})
        g = scn.geom_contact(row)                        # footprint overlap with a scripted (kinematic) vehicle
        if g and frame not in contacts: contacts[frame] = g + " (footprint overlap)"
        rows.append(row)
        cam.set_transform(cam_tf())
        done, why = scn.finished(rows, contacts)
        if contacts and end_after is None:
            end_after = n + int(1.0 / dt); end_reason = "contact"
        if end_after is not None and n >= end_after: break
        if end_after is None and done: end_reason = why; break
    time.sleep(0.3)
    while not colq.empty():
        e = colq.get(); contacts[e.frame] = scn.roles.get(e.other_actor.id, e.other_actor.type_id)
    sensor_ok = col.is_alive and cam.is_alive
    cam.stop(); col.stop(); cam.destroy(); col.destroy()
    scn.destroy(); world.tick()
    for r in rows:
        r["contact"] = int(r["frame"] in contacts); r["contact_actor"] = contacts.get(r["frame"], "")
    frames = [r["frame"] for r in rows]
    meta = dict(frames_contiguous=all(b - a == 1 for a, b in zip(frames, frames[1:])), collision_sensor_ok=bool(sensor_ok),
                video_frames_match=all(f in images for f in frames))
    meta.update(scn.meta())
    res = scn.evaluate(rows, contacts, meta, pcfg)
    evidence.write_all(out_dir, rows, views, truths, obs_dicts, decisions)
    first_c = min(contacts) if contacts else None
    fr_list = []
    for r in rows:
        lines, banners = scn.overlay(r)
        if first_c is not None and r["frame"] >= first_c: banners = list(banners) + [("CONTACT: " + contacts[first_c], (255, 70, 70))]
        fr_list.append(video.overlay_lines(images[r["frame"]], lines, r["phase"], banners))
    last = video.card2(fr_list[-1], res["label"], res["reason"], res.get("detail", ""), run_id)
    fr_list += [last] * int(3.0 / dt)
    video.export(os.path.join(out_dir, "run.mp4"), fr_list, int(round(1 / dt)))
    summary = dict(run_id=run_id, rc=scn.rc_id, label=res["label"], reason=res["reason"], end_reason=end_reason,
                   scenario=scn.params(), policy=dict(name=policy.name, version=policy.version, params=policy.params()),
                   calibration={k: calib[k] for k in ("brake_cmd", "brake_decel", "brake_cap", "coast_decel", "brake_delay_s", "cruise_throttle", "max_decel")},
                   metrics=res.get("metrics", {}), times=res.get("times", {}), contact=dict(frames=sorted(contacts), actors=sorted(set(contacts.values()))),
                   evidence_meta=meta, schema=evidence.SCHEMA, scene_seed=scn.p.get("seed", 7),
                   files=["run.mp4", "ego_observable.csv", "ground_truth.csv", "decisions.jsonl", "summary.json"])
    summary.update(meta_extra or {})
    if hasattr(scn, "extra_summary"): summary.update(scn.extra_summary(rows, res, contacts))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=float)
    return summary


def _cam(ego, fr, scn):
    tf = ego.get_transform()
    s, t = fr.to_local(tf.location.x, tf.location.y)
    x, y = fr.to_world(s - scn.cam_back, scn.cam_t)
    return carla.Transform(carla.Location(x, y, fr.z(s) + scn.cam_up), carla.Rotation(pitch=-scn.cam_pitch, yaw=fr.heading(s - scn.cam_back)))
