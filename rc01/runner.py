"""One scored RC01-01 run: scene -> policy loop -> telemetry -> label -> mp4/csv/json."""
import copy, csv, json, math, os, queue, time
import numpy as np
import carla
from .config import G
from .policy import SafeStopPolicy, Observation
from .controller import LowLevelController
from .scene import Scene, footprint, rear_front_x, lanes
from .calibrate import spawn_ego, max_steer_rad
from .evaluate import evaluate
from . import video

COLS = ["frame", "sim_time", "t_since_obs", "phase", "x", "y", "yaw", "speed", "gap", "obs_generated", "obs_delivered", "a_req", "a_req_clipped",
        "throttle", "brake", "steer", "hand_brake", "accel_long", "accel_lat", "accel_long_f", "accel_lat_f", "accel_planar_f",
        "contact", "contact_geom", "contact_actor", "corridor_violation", "lateral_offset", "van_moved_m", "ped_violation"]


def _img(im, h, w):
    return np.frombuffer(im.raw_data, np.uint8).reshape(h, w, 4)[:, :, 2::-1].copy()


def run_episode(world, cfg0, pcfg, calib, run_id, out_dir, gap, changes="none", obs_delay_s=0.0):
    cfg = copy.deepcopy(cfg0); cfg.target_gap = gap
    os.makedirs(out_dir, exist_ok=True)
    dt, ln = cfg.dt, lanes(cfg)
    scene = Scene(world, cfg); scene.build()
    ego = spawn_ego(world, cfg, 60.0)
    van_rear = rear_front_x(scene.van)[0]
    off = rear_front_x(ego)[1] - ego.get_location().x
    ego_w = 2 * ego.bounding_box.extent.y
    block = scene.check_blocked(ego_w)
    if cfg.start_from_rest:                       # natural approach: from standstill, comfortable acceleration, then cruise
        gap0 = gap + cfg.target_speed ** 2 / (2 * 1.1) + 4.0 + cfg.target_speed * cfg.cruise_before_cue_s
    else:
        gap0 = gap + cfg.target_speed * cfg.approach_time
    ego.set_transform(carla.Transform(carla.Location(van_rear - gap0 - off, cfg.lane_center, 0.4)))
    ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True)); scene.hold_static(); world.tick()
    for _ in range(10): scene.hold_static(); ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True)); world.tick()
    if not cfg.start_from_rest:
        ego.set_target_velocity(carla.Vector3D(cfg.target_speed, 0, 0))

    bl = world.get_blueprint_library()
    cb = bl.find("sensor.camera.rgb")
    cb.set_attribute("image_size_x", str(cfg.video_w)); cb.set_attribute("image_size_y", str(cfg.video_h))
    cb.set_attribute("fov", "95"); cb.set_attribute("sensor_tick", "0.0")
    cam_tf = lambda x: carla.Transform(carla.Location(x - 11, cfg.lane_center, 6.5), carla.Rotation(pitch=-20))
    cam = world.spawn_actor(cb, cam_tf(ego.get_location().x))
    imgq = queue.Queue(); cam.listen(imgq.put)
    col = world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    colq = queue.Queue(); col.listen(colq.put)

    policy = SafeStopPolicy(pcfg, dt)
    ctrl = LowLevelController(pcfg, calib, cfg.lane_center, max_steer_rad(ego), dt)
    images, rows, contacts = {}, [], {}
    prev_v = ego.get_velocity(); hist_l, hist_t = [], []
    delivered_frame = t_cue = None
    gen_frame = None
    delay_frames = int(round(obs_delay_s / dt))
    still_run, need = 0, int(round(cfg.hold_time / dt))
    end_reason, end_after = None, None
    ped_viol = 0; ped_viol_info = None
    van_speed_max = 0.0
    nmax = int((cfg.approach_time + 14 + cfg.time_limit + cfg.post_contact_s + 5) / dt)
    cam.set_transform(cam_tf(ego.get_location().x))
    for _ in range(3):                                   # let the camera settle at its pose (ego is still at rest)
        scene.hold_static(); ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True)); world.tick()
        while not imgq.empty(): imgq.get()
    for n in range(nmax):
        scene.hold_static(); scene.move_walkers()
        frame = world.tick()
        im = imgq.get(timeout=20)
        while im.frame < frame: im = imgq.get(timeout=20)
        images[frame] = _img(im, cfg.video_h, cfg.video_w)
        time.sleep(0.003)
        while not colq.empty():
            e = colq.get(); contacts[e.frame] = e.other_actor.type_id
        snap = world.get_snapshot(); t = snap.timestamp.elapsed_seconds

        tf, vel = ego.get_transform(), ego.get_velocity()
        yaw = math.radians(tf.rotation.yaw)
        fwd, right = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        dv = np.array([vel.x - prev_v.x, vel.y - prev_v.y]) / dt; prev_v = vel
        a_l, a_t = float(dv @ fwd), float(dv @ right)
        speed = math.hypot(vel.x, vel.y)
        fp = footprint(ego)
        gap_now = float(van_rear - fp[:, 0].max())
        corr = bool(fp[:, 1].min() < ln["drive"][0] or fp[:, 1].max() > ln["drive"][1])
        for wi, w in enumerate(scene.walkers):
            wy = w.get_location().y
            if not (ln["walk_r"][0] <= wy <= ln["walk_r"][1] or ln["walk_l"][0] <= wy <= ln["walk_l"][1]):
                ped_viol += 1
                if ped_viol_info is None: ped_viol_info = dict(walker=wi, y=wy, x=w.get_location().x, frame=frame, sidewalks=[ln["walk_l"], ln["walk_r"]])
        vm = scene.van.get_location().distance(scene.van_pose0)
        vv = scene.van.get_velocity(); van_speed_max = max(van_speed_max, math.hypot(vv.x, vv.y))

        if gen_frame is None and gap_now <= cfg.target_gap:
            gen_frame, t_cue = frame, t                      # obstacle information generated (scored window starts)
        delivered = gen_frame is not None and (frame - gen_frame) >= delay_frames
        if delivered and delivered_frame is None: delivered_frame = frame
        obs = Observation(frame, t, speed, delivered, gap_now)
        out = policy.step(obs)
        hist_l.append(a_l); hist_t.append(a_t)
        k = max(1, int(round(cfg.accel_filter_s / dt)))
        c = ctrl.step(out, tf.location.y, tf.rotation.yaw, speed, a_l)
        ego.apply_control(carla.VehicleControl(throttle=c["throttle"], brake=c["brake"], steer=c["steer"], hand_brake=c["hand_brake"]))
        af, atf = float(np.mean(hist_l[-k:])), float(np.mean(hist_t[-k:]))
        geom = gap_now <= 0
        rows.append(dict(frame=frame, sim_time=t, t_since_obs=(t - t_cue) if gen_frame is not None else "", phase=out.phase,
                         x=tf.location.x, y=tf.location.y, yaw=tf.rotation.yaw, speed=speed, gap=gap_now,
                         obs_generated=int(gen_frame is not None), obs_delivered=int(delivered), a_req=out.a_req, a_req_clipped=c["a_req_clipped"],
                         throttle=c["throttle"], brake=c["brake"], steer=c["steer"], hand_brake=int(c["hand_brake"]),
                         accel_long=a_l, accel_lat=a_t, accel_long_f=af, accel_lat_f=atf, accel_planar_f=math.hypot(af, atf),
                         contact=0, contact_geom=int(geom), contact_actor="", corridor_violation=int(corr),
                         lateral_offset=tf.location.y - cfg.lane_center, van_moved_m=vm, ped_violation=int(ped_viol > 0)))
        cam.set_transform(cam_tf(tf.location.x))
        # termination
        if gen_frame is not None:
            still_run = still_run + 1 if speed < cfg.hold_speed else 0
            if end_after is None and (contacts or geom):
                end_after = n + int(cfg.post_contact_s / dt); end_reason = "contact"
            if end_after is None and still_run >= need: end_reason = "hold_complete"; break
            if end_after is None and t - t_cue >= cfg.time_limit: end_reason = "time_limit"; break
            if end_after is not None and n >= end_after: break
    time.sleep(0.3)
    while not colq.empty():
        e = colq.get(); contacts[e.frame] = e.other_actor.type_id
    sensor_ok = col.is_alive and cam.is_alive
    cam.stop(); col.stop(); cam.destroy(); col.destroy(); ego.destroy(); scene.destroy(); world.tick()

    first_c = min(contacts) if contacts else None
    for r in rows:
        r["contact"] = int(r["frame"] in contacts); r["contact_actor"] = contacts.get(r["frame"], "")
    contact_any = False
    frames = [r["frame"] for r in rows]
    meta = dict(blocked_check=block, van_moved_m=max(r["van_moved_m"] for r in rows), van_max_speed=van_speed_max,
                ped_violations=ped_viol, frames_contiguous=all(b - a == 1 for a, b in zip(frames, frames[1:])),
                collision_sensor_ok=bool(sensor_ok), video_frames_match=all(f in images for f in frames))
    # a geometric overlap counts as contact too
    for r in rows:
        if r["contact_geom"] and not contacts: r["contact"] = 1 if r is next(x for x in rows if x["contact_geom"]) else 0
    res = evaluate(rows, meta, cfg, pcfg)

    with open(os.path.join(out_dir, "telemetry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)

    cue_row = next((r for r in rows if r["obs_delivered"]), None)
    pre_cue = (t_cue - rows[0]["sim_time"]) if t_cue is not None else None
    brake_row = next((r for r in rows if r["obs_delivered"] and r["brake"] > 0), None)
    onset_row = next((r for r in rows if r["obs_delivered"] and r["accel_long"] < -0.5), None)
    summary = dict(
        run_id=run_id, label=res["label"], reason=res["reason"], end_reason=end_reason,
        scene=dict(config=cfg.to_dict(), blocked_road_check=block, ego_width=ego_w,
                   van_footprint_rear_x=float(van_rear), pedestrians=len(scene.walkers), walkers_per_side=cfg.n_walkers_per_side,
                   clear_width_measured=block["clear_width_measured"], lane_width=cfg.lane_width, parked_cars=len(scene.parked)),
        policy=dict(version=pcfg.version, params=pcfg.to_dict(), low_level="calibrated feedforward + integral brake, lane-keeping steer"),
        calibration=dict(brake_cmd=calib["brake_cmd"], brake_decel=calib["brake_decel"], brake_delay_s=calib["brake_delay_s"],
                         brake_buildup_90_s=calib["brake_buildup_90_s"], max_decel=calib["max_decel"], brake_cap=calib["brake_cap"], locking_brakes=calib["locking_brakes"], low_speed_snap_from=calib["low_speed_snap_from"], cruise_throttle=calib["cruise_throttle"]),
        observation=dict(source="idealized: bumper gap from CARLA ground-truth footprints; generated when gap <= target",
                         generated_frame=gen_frame, delivered_to_controller_frame=delivered_frame, delay_s=obs_delay_s,
                         frame=gen_frame, sim_time=t_cue, recording_before_cue_s=pre_cue),
        start_speed=res.get("start_speed"), start_gap=res.get("start_gap"),
        target_speed=cfg.target_speed, target_gap=cfg.target_gap,
        min_gap=res.get("min_gap"), stopping_gap=res.get("stopping_gap"), hold_min_gap=res.get("hold_min_gap"),
        max_filtered_accel=res.get("max_filtered_accel"),
        times=dict(observation_delivery=t_cue, observation_at_controller=(next((x["sim_time"] for x in rows if x["obs_delivered"]), None)), first_brake_application=brake_row["sim_time"] if brake_row else None,
                   onset_measured_decel=onset_row["sim_time"] if onset_row else None,
                   brake_to_decel_lag_s=(onset_row["sim_time"] - brake_row["sim_time"]) if (onset_row and brake_row) else None),
        contact=dict(frames=sorted(contacts), actors=sorted(set(contacts.values()))),
        controller_changes=changes, carla_version=world.get_client_version() if hasattr(world, "get_client_version") else "0.9.15",
        files=["run.mp4", "telemetry.csv", "summary.json"])
    summary["evidence_meta"] = {k: v for k, v in meta.items() if k != "blocked_check"}; summary["ped_violation_info"] = ped_viol_info

    fr = []
    for r in rows:
        rr = dict(r); rr["contact_any"] = first_c is not None and r["frame"] >= first_c
        fr.append(video.overlay(images[r["frame"]], rr, t_cue))
    ref = res["min_gap"] if res.get("min_gap") is not None else float(min(r["gap"] for r in rows))
    last = video.card(fr[-1], res["label"], res["reason"], res.get("stopping_gap"), ref, run_id)
    fr += [last] * int(cfg.card_s / dt)
    video.export(os.path.join(out_dir, "run.mp4"), fr, int(round(1 / dt)))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=float)
    return summary
