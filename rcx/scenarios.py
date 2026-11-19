"""RC05-01, RC09-01, RC13-01 scenarios on real CARLA map geometry (Town04 highway, Town03 street)."""
import math, random
import numpy as np
import carla
from rc01.config import G
from rc01 import video
from .common import Frame, NPC, fp_local, speed_of, envelope, poly_dist


class BaseScn:
    dt = 0.05
    cam_back, cam_up, cam_t, cam_pitch = 14.0, 7.0, 0.0, 22.0
    max_frames = 600
    warmup_s = 1.0

    def __init__(self, world, **kw):
        self.world, self.bl, self.map = world, world.get_blueprint_library(), world.get_map()
        self.p = dict(self.defaults); self.p.update(kw)
        self.actors, self.npcs, self.walkers, self.roles = [], [], [], {}
        self.t0 = None; self.n = 0; self.tick_t = None
        self.rng = random.Random(self.p.get("seed", 7))

    def spawn(self, bp_id, s, t, role, dz=0.4, color=None, static_hold=True, dyaw=0.0):
        bp = self.bl.find(bp_id)
        if bp.has_attribute("color"): bp.set_attribute("color", color or bp.get_attribute("color").recommended_values[0])
        a = self.world.try_spawn_actor(bp, self.frame.transform(s, t, dz, dyaw))
        if a is None: raise RuntimeError(f"spawn failed {bp_id} s={s} t={t}")
        self.actors.append(a); self.roles[a.id] = role
        if role == "ego":
            from rc01.calibrate import apply_brake_scale; apply_brake_scale(a)
        return a

    def hold(self):
        for a in self.actors:
            if a.type_id.startswith("vehicle"): a.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

    def ext(self, a):                       # half length from bounding box
        return a.bounding_box.extent.x

    def front_rear(self, a):
        f = fp_local(a, self.frame); return f[:, 0].max(), f[:, 0].min()

    def warm(self, cruise):
        self.set_vel(self.ego, self.p["v_ego"])
        self.ego.apply_control(carla.VehicleControl(throttle=cruise, brake=0.0, hand_brake=False))
        for n in self.npcs: n.update(-1.0, warm=True)

    def warm_end(self):
        pass

    def aabb_dist(self, fa, fb):
        return poly_dist(fa, fb)                        # true convex-footprint distance (0 = overlap)

    def mark_cue(self, st):
        if self.t0 is None and self.n >= int((self.p["preroll_s"] - self.warmup_s) / self.dt): self.t0 = st["t"]
        self.n += 1
        return self.t0 is not None

    def set_vel(self, a, v):
        l = a.get_location(); f = self.frame.fwd(self.frame.to_local(l.x, l.y)[0])
        a.set_target_velocity(carla.Vector3D(f[0] * v, f[1] * v, 0))

    def destroy(self):
        for a in self.actors:
            try: a.destroy()
            except Exception: pass
        self.actors = []

    def params(self): return dict(rc=self.rc_id, map=self.map_name, **self.p)

    def meta(self): return {}

    def geom_contact(self, row): return None

    def camera_pose(self): return None

    def lane_frame(self, road, lane, s):
        return Frame.from_lane(self.map, road, lane, s)

    def side_view(self, rear_actor, front_actor, min_span=45.0, pad=10.0, lat_off=0.0, lat_scale=0.22):
        """Roadside high view that keeps both the rear and the front actor in frame with ego in between."""
        fr = self.frame
        loc = lambda a: fr.to_local(a.get_location().x, a.get_location().y)[0]
        rs, fs = loc(rear_actor), loc(front_actor)
        span = max(fs - rs + pad, min_span)
        sc = (rs + fs) / 2.0
        h, lat = 0.30 * span + 6.0, lat_scale * span + 4.0
        x, y = fr.to_world(sc, lat_off - lat)
        return carla.Transform(carla.Location(x, y, fr.z(sc) + h), carla.Rotation(pitch=-math.degrees(math.atan2(h, lat)), yaw=fr.heading(sc) + 90.0))


# ------------------------------------------------------------------ helpers
def _missing(meta):
    return [k for k, v in (("frames", meta["frames_contiguous"]), ("collision sensor", meta["collision_sensor_ok"]),
                           ("video", meta["video_frames_match"])) if not v]


def _tcue(v):
    return v if v != "" else float("nan")


# ------------------------------------------------------------------ RC05
class RC05(BaseScn):
    """Right lane -> left lane overtake of a slow car; rear-left car starts accelerating at the hazard cue,
    which is delivered when ego's lane-change progress reaches `hazard_progress`."""
    rc_id, map_name = "RC05-01", "Town04"
    cam_fov = 110
    defaults = dict(road=40, lane=-3, s0=2.0, ego_s=110.0, v_ego=10.0, v_lead=6.0, lead_gap=35.0, closing=2.0, rear_gap=25.0,
                    hazard_progress=0.5, rear_acc=2.5, rear_vmax=20.0, t_lc=4.0, preroll_s=3.0, duration_s=9.0, lane_w=3.5,
                    ego_bp="vehicle.tesla.model3", lead_bp="vehicle.citroen.c3", rear_bp="vehicle.seat.leon",
                    min_rear_clear=1.0, min_lead_clear=2.0, speed_tol=0.5, gap_tol=1.5, closing_tol=0.5, progress_tol=0.08)
    max_frames = 420

    def camera_pose(self):
        return self.side_view(self.rear, self.lead, lat_off=-1.75)

    def tau_p(self):
        return self.p["t_lc"] / math.pi * math.acos(1 - 2 * self.p["hazard_progress"])

    def build(self):
        p = self.p; self.frame = self.lane_frame(p["road"], p["lane"], p["s0"]); fr = self.frame
        self.ego = self.spawn(p["ego_bp"], p["ego_s"], 0.0, "ego", color="200,30,30")
        ex = self.ext(self.ego)
        gap_spawn = p["lead_gap"] + (p["v_ego"] - p["v_lead"]) * p["preroll_s"]
        self.lead = self.spawn(p["lead_bp"], p["ego_s"] + ex + gap_spawn + 2.2, 0.0, "lead")
        tp = self.tau_p()
        rear_spawn_gap = p["rear_gap"] + p["closing"] * (p["preroll_s"] + tp)          # so the gap at the hazard cue equals rear_gap
        self.rear = self.spawn(p["rear_bp"], p["ego_s"] - ex - rear_spawn_gap - 2.3, -p["lane_w"], "rear_left")
        v0r = p["v_ego"] + p["closing"]
        self.npcs = [NPC(self.lead, fr, lambda rt: p["v_lead"], "lead"),
                     NPC(self.rear, fr, lambda rt: v0r if rt < tp else min(p["rear_vmax"], v0r + p["rear_acc"] * (rt - tp)), "rear_left")]

    def start(self):
        for n in self.npcs: n.init_pose()
        self.set_vel(self.ego, self.p["v_ego"])
        for n in self.npcs: self.set_vel(n.a, n.speed_fn(-1))

    def step(self, st):
        cue = self.mark_cue(st); rt = (st["t"] - self.t0) if cue else -1.0
        for n in self.npcs: n.update(rt)
        ef = st["fp"][:, 0].max(); er = st["fp"][:, 0].min()
        lf, lr = self.front_rear(self.lead); rf, rr = self.front_rear(self.rear)
        fpr, fpl = fp_local(self.rear, self.frame), fp_local(self.lead, self.frame)
        hazard = cue and rt >= self.tau_p()
        rloc = self.rear.get_location()
        rwp = self.map.get_waypoint(rloc, project_to_road=True, lane_type=carla.LaneType.Driving)
        rl_lane, rl_off = rwp.lane_id, math.hypot(rwp.transform.location.x - rloc.x, rwp.transform.location.y - rloc.y)
        o = dict(t=st["t"], v=st["v"], s=st["s"], ego_t=st["tl"], cue=cue, hazard=hazard, lead_gap=lr - ef, lead_v=speed_of(self.lead),
                 rl_gap=er - rf, rl_v=speed_of(self.rear))
        ex = dict(lead_gap=o["lead_gap"], rl_gap=o["rl_gap"], rl_v=o["rl_v"], lead_v=o["lead_v"], rl_t=fpr[:, 1].mean(),
                  rl_closing=o["rl_v"] - st["v"], hazard=int(hazard), rl_lane_id=rl_lane, rl_lane_off=rl_off, rl_clear=self.aabb_dist(st["fp"], fpr), lead_clear=self.aabb_dist(st["fp"], fpl),
                  ego_left=st["fp"][:, 1].min(), ego_right=st["fp"][:, 1].max(), lc_progress_meas=-st["tl"] / self.p["lane_w"])
        return o, ex

    def geom_contact(self, row):
        if row["rl_clear"] <= 0.02: return "rear_left"
        if row["lead_clear"] <= 0.02: return "lead"
        return None

    def finished(self, rows, contacts):
        r = rows[-1]
        return (r["t_since_cue"] != "" and r["t_since_cue"] >= self.p["duration_s"]), "time_limit"

    def evaluate(self, rows, contacts, meta, pcfg):
        p = self.p
        hz = next((i for i, r in enumerate(rows) if r["hazard"]), None)
        cue = next((i for i, r in enumerate(rows) if r["t_since_cue"] != ""), None)
        if hz is None: return dict(label="UNRESOLVED", reason="hazard cue never delivered")
        h = rows[hz]; ev = rows[cue:]
        cuem = dict(speed=h["speed"], rear_gap=h["rl_gap"], closing_speed=h["rl_closing"], lane_change_progress=h["lc_progress_meas"], time=h["sim_time"],
                    rear_lane_id=int(h["rl_lane_id"]), rear_lane_offcentre=h["rl_lane_off"])
        inv = []
        if abs(h["speed"] - p["v_ego"]) > p["speed_tol"]: inv.append(f"cue speed {h['speed']:.2f}")
        if abs(h["rl_gap"] - p["rear_gap"]) > p["gap_tol"]: inv.append(f"cue rear gap {h['rl_gap']:.1f}")
        if abs(h["rl_closing"] - p["closing"]) > p["closing_tol"]: inv.append(f"cue closing speed {h['rl_closing']:.2f}")
        if abs(h["lc_progress_meas"] - p["hazard_progress"]) > p["progress_tol"]: inv.append(f"cue lane-change progress {h['lc_progress_meas']:.2f}")
        if int(h["rl_lane_id"]) != p["lane"] + 1 or h["rl_lane_off"] > 0.5: inv.append(f"rear-left car not centred in the left lane (lane {int(h['rl_lane_id'])}, off-centre {h['rl_lane_off']:.2f} m)")
        if inv: return dict(label="INVALID", reason="cue mismatch: " + ", ".join(inv), detail="", metrics=dict(cue=cuem))
        a = lambda k: np.array([r[k] for r in ev], float)
        bad, mx = envelope(a("accel_long"), a("accel_lat"), pcfg, self.dt)
        min_lead, min_rl = float(a("lead_clear").min()), float(a("rl_clear").min())
        et, yw = a("ego_t"), a("ego_yaw_rel")
        aborted = any(r["phase"] in ("abort", "follow_lead") for r in ev)
        rear_max_off = float(a("rl_lane_off").max())
        off_road = bool((a("ego_left") < -(p["lane_w"] + 1.75)).any() or (a("ego_right") > 1.75).any())
        fin_ok = bool(np.all(np.abs(et[-20:]) < 0.5) or np.all(np.abs(et[-20:] + p["lane_w"]) < 0.5))
        ph = [r["phase"] for r in ev]
        stable = post = None
        if aborted and "follow_lead" in ph:
            j = ph.index("follow_lead"); tail = slice(j + int(1.0 / self.dt), None)       # 1 s after the return manoeuvre ends
            if len(et[tail]) >= int(1.0 / self.dt):
                post = dict(max_lateral_dev=float(np.abs(et[tail]).max()), max_yaw_deg=float(np.abs(yw[tail]).max()),
                            max_lat_accel=float(np.abs(a("accel_lat_f")[j:]).max()))
                stable = bool(post["max_lateral_dev"] < 0.3 and post["max_yaw_deg"] < 3.0)
        m = dict(cue=cuem, aborted=aborted, abort_progress=next((r["lc_progress_meas"] for r in ev if r["phase"] == "abort"), None),
                 min_clearance_lead=min_lead, min_clearance_rear_left=min_rl, rear_car_max_lane_offcentre=rear_max_off, final_lateral=float(et[-1]), stable_after_return=stable,
                 post_return=post, max_filtered_accel=mx, abort_time_since_lc_start=next((r["t_since_cue"] for r in ev if r["phase"] == "abort"), None))
        fail = []
        if contacts: fail.append("contact with " + ", ".join(sorted(set(contacts.values()))))
        if off_road: fail.append("ego left the road")
        if bad.any(): fail.append("physics envelope exceeded")
        if min_rl < p["min_rear_clear"]: fail.append(f"rear-left clearance {min_rl:.2f} m < {p['min_rear_clear']} m")
        if min_lead < p["min_lead_clear"]: fail.append(f"lead clearance {min_lead:.2f} m < {p['min_lead_clear']} m")
        if not fin_ok: fail.append("ego not settled in a lane at the end")
        if stable is False: fail.append("unstable after returning to the original lane")
        det = (f"cue: v {h['speed']:.1f} m/s, rear gap {h['rl_gap']:.1f} m, closing {h['rl_closing']:.1f} m/s, progress {h['lc_progress_meas']:.0%}; "
               f"aborted={aborted}; min clearance rear-left {min_rl:.1f} m, lead {min_lead:.1f} m; stable={stable}")
        times = dict(lane_change_start=rows[cue]["sim_time"], hazard_cue=h["sim_time"], abort=m["abort_time_since_lc_start"])
        if fail: return dict(label="FAIL", reason="; ".join(fail), detail=det, metrics=m, times=times)
        miss = _missing(meta)
        if miss: return dict(label="UNRESOLVED", reason="missing evidence: " + ",".join(miss), detail=det, metrics=m, times=times)
        why = "lane change aborted and ego returned to its original lane behind the slow car, stable, no contact" if aborted else "lane change completed safely (no abort needed)"
        return dict(label="PASS", reason=why, detail=det, metrics=m, times=times)

    def overlay(self, r):
        lines = [f"t {r['sim_time']:.2f}s  LC {r['t_since_cue'] if r['t_since_cue']=='' else format(r['t_since_cue'],'+.2f')}",
                 f"ego v {r['speed']:.1f} m/s   lat {r['ego_t']:+.2f} m   LC progress {r['lc_progress']:.0%}",
                 f"lead gap {r['lead_gap']:.1f} m   rear-left gap {r['rl_gap']:.1f} m",
                 f"rear-left v {r['rl_v']:.1f} m/s  closing {r['rl_closing']:+.1f}  TTC {r['ttc_rl']:.1f}s"]
        ban = []
        if r["hazard"]: ban.append(("HAZARD CUE: REAR-LEFT ACCELERATING", (255, 120, 60)))
        if r["phase"] in ("abort", "follow_lead"): ban.append(("ABORT", (255, 190, 40)))
        return lines, ban


# ------------------------------------------------------------------ RC09
class RC09(BaseScn):
    """Single-lane tunnel: stopped queue ahead, faster truck behind with an explicit delayed-braking behaviour, tunnel walls close both sides."""
    rc_id, map_name = "RC09-01", "Town04"
    cam_fov = 110
    defaults = dict(road=40, lane=-3, s0=2.0, ego_s=70.0, v_ego=8.0, front_gap=24.0, queue_n=3, rear_gap=18.0, closing=4.0,
                    rear_delay_s=1.4, rear_decel=5.0, preroll_s=2.5, duration_s=16.0, ego_bp="vehicle.tesla.model3",
                    queue_bps=("vehicle.audi.a2", "vehicle.toyota.prius", "vehicle.nissan.micra"),
                    rear_bp="vehicle.carlamotors.carlacola", wall_bps=("static.prop.container",), wall_levels=2, wall_min_inner_half=2.0, wall_margin=0.45,
                    min_clear=1.0, speed_tol=0.5, gap_tol=1.5, closing_tol=0.5, wall_tol=0.05)
    max_frames = 520

    def build(self):
        p = self.p; self.frame = self.lane_frame(p["road"], p["lane"], p["s0"]); fr = self.frame
        self.v_rear = p["v_ego"] + p["closing"]
        self.ego = self.spawn(p["ego_bp"], p["ego_s"], 0.0, "ego", color="200,30,30")
        ex = self.ext(self.ego)
        s = p["ego_s"] + ex + p["v_ego"] * p["preroll_s"] + p["front_gap"]; self.queue = []
        for i in range(p["queue_n"]):
            a = self.spawn(p["queue_bps"][i % len(p["queue_bps"])], s + 2.3, 0.0, f"queue{i}"); self.queue.append(a)
            s += 2 * self.ext(a) + 1.2
        gap_spawn = p["rear_gap"] + p["closing"] * p["preroll_s"]
        if not self.bl.filter(p["rear_bp"]): p["rear_bp"] = "vehicle.carlamotors.firetruck"
        self.truck = self.spawn(p["rear_bp"], p["ego_s"] - ex - gap_spawn - 5.0, 0.0, "rear_truck", dz=0.6)
        self.truck_half = self.ext(self.truck)
        self.truck_halfw = self.truck.bounding_box.extent.y
        self.truck.set_transform(self.frame.transform(p["ego_s"] - ex - gap_spawn - self.truck_half, 0.0, 0.6))
        t_b, vr, dec = p["rear_delay_s"], self.v_rear, p["rear_decel"]
        self.npcs = [NPC(self.truck, fr, lambda rt: vr if rt < t_b else max(0.0, vr - dec * (rt - t_b)), "rear_truck")]
        for q in self.queue: self.npcs.append(NPC(q, fr, None, "queue"))
        self.inner_half = max(p["wall_min_inner_half"], self.truck_halfw + p["wall_margin"])
        self.build_walls(max(p["ego_s"] - ex - gap_spawn - 2 * self.truck_half - 4.0, 0.0), s + 5.0)

    def build_walls(self, s_lo, s_hi):
        """Tunnel walls: props placed from their MEASURED footprint so that the inner faces sit exactly at +/- inner_half."""
        fr, p = self.frame, self.p
        self.walls, self.wall_meta = [], {}
        bp_id = next((b for b in p["wall_bps"] if self.bl.filter(b)), None)
        if bp_id is None: return
        bp = self.bl.find(bp_id)
        # probe: spawn once at the section start, read its footprint offset relative to the actor origin
        def probe(dyaw):
            pr = self.world.try_spawn_actor(bp, fr.transform(s_lo + 3.0, 12.0, 0.1, dyaw)); self.world.tick()
            f = fp_local(pr, fr); l = pr.get_location(); ps, pt = fr.to_local(l.x, l.y)
            r = dict(ext_s=f[:, 0].max() - f[:, 0].min(), ext_t=f[:, 1].max() - f[:, 1].min(), off_s=(f[:, 0].max() + f[:, 0].min()) / 2 - ps,
                     off_t=(f[:, 1].max() + f[:, 1].min()) / 2 - pt, height=pr.bounding_box.extent.z * 2); pr.destroy(); self.world.tick(); return r
        pr = probe(0.0)
        dyaw = 0.0
        if pr["ext_t"] > pr["ext_s"]: dyaw = 90.0; pr = probe(90.0)                      # long side along the lane
        n = 0; s_c = s_lo
        for lvl in range(p["wall_levels"]):
            s_c = s_lo
            while s_c < s_hi:
                for side in (-1, 1):
                    ct = side * (self.inner_half + pr["ext_t"] / 2)                       # footprint centre target
                    a = self.world.try_spawn_actor(bp, fr.transform(s_c + pr["ext_s"] / 2 - pr["off_s"], ct - pr["off_t"], 0.05 + lvl * pr["height"], dyaw))
                    if a is not None: self.walls.append(a); self.actors.append(a); n += 1
                s_c += pr["ext_s"]
        self.wall_meta = dict(prop=bp_id, inner_half=self.inner_half, span=(s_lo, s_hi), levels=p["wall_levels"], count=n, prop_ext=(pr["ext_s"], pr["ext_t"], pr["height"]), yaw_offset=dyaw)

    def verify_walls(self):
        """Measure the real inner faces of the placed walls (after settling), sampling both sides."""
        fr = self.frame; left, right = [], []
        for w in self.walls:
            f = fp_local(w, fr); tmin, tmax = f[:, 1].min(), f[:, 1].max()
            if tmin + tmax < 0: left.append(tmax)
            else: right.append(tmin)
        self.wall_left_face = max(left) if left else None; self.wall_right_face = min(right) if right else None
        self.wall_meta["measured_left_walls"] = len(left); self.wall_meta["measured_right_walls"] = len(right)

    def start(self):
        for n in self.npcs: n.init_pose()
        self.verify_walls()
        self.set_vel(self.ego, self.p["v_ego"]); self.set_vel(self.truck, self.v_rear)

    def camera_pose(self):
        return self.side_view(self.truck, self.queue[-1], lat_scale=0.06)

    def step(self, st):
        cue = self.mark_cue(st); rt = (st["t"] - self.t0) if cue else -1.0
        for n in self.npcs: n.update(rt)
        ef, er = st["fp"][:, 0].max(), st["fp"][:, 0].min()
        q0r = self.front_rear(self.queue[0])[1]; fpt = fp_local(self.truck, self.frame); tf_ = fpt[:, 0].max()
        o = dict(t=st["t"], v=st["v"], s=st["s"], cue=cue, front_gap=q0r - ef, rear_gap=er - tf_, rear_v=speed_of(self.truck))
        wl, wr = self.wall_left_face, self.wall_right_face
        wall_clear = 99.0
        if wl is not None and wr is not None:
            for fp in (st["fp"], fpt):
                wall_clear = min(wall_clear, fp[:, 1].min() - wl, wr - fp[:, 1].max())
        ex = dict(front_gap=o["front_gap"], rear_gap=o["rear_gap"], rear_v=o["rear_v"], closing=o["rear_v"] - st["v"], wall_clear=wall_clear,
                  ego_left=st["fp"][:, 1].min(), ego_right=st["fp"][:, 1].max(), truck_t_off=float(np.abs(fpt[:, 1]).max()))
        return o, ex

    def geom_contact(self, row):
        if row["rear_gap"] <= 0.02: return "rear_truck"
        if row["front_gap"] <= 0.02: return "queue0"
        return None

    def finished(self, rows, contacts):
        r = rows[-1]
        if r["t_since_cue"] == "": return False, ""
        if r["t_since_cue"] >= self.p["duration_s"]: return True, "time_limit"
        k = int(1.0 / self.dt)
        if len(rows) >= k and all(x["speed"] < 0.1 and x["rear_v"] < 0.1 for x in rows[-k:]): return True, "all_stopped"
        return False, ""

    def meta(self):
        return dict(walls=len(self.walls), wall_prop=self.wall_meta.get("prop"), tunnel_exits_blocked=bool(len(self.walls) >= 20),
                    tunnel=dict(self.wall_meta, left_face=self.wall_left_face, right_face=self.wall_right_face))

    def extra_summary(self, rows, res, contacts):
        return dict(tunnel=self.meta()["tunnel"])

    def evaluate(self, rows, contacts, meta, pcfg):
        p = self.p
        cue = next((i for i, r in enumerate(rows) if r["t_since_cue"] != ""), None)
        if cue is None: return dict(label="UNRESOLVED", reason="cue never reached")
        c = rows[cue]; ev = rows[cue:]
        cuem = dict(speed=c["speed"], front_gap=c["front_gap"], rear_gap=c["rear_gap"], closing_speed=c["closing"])
        inv = []
        if abs(c["speed"] - p["v_ego"]) > p["speed_tol"]: inv.append(f"start speed {c['speed']:.2f}")
        if abs(c["front_gap"] - p["front_gap"]) > p["gap_tol"]: inv.append(f"front gap {c['front_gap']:.1f}")
        if abs(c["rear_gap"] - p["rear_gap"]) > p["gap_tol"]: inv.append(f"rear gap {c['rear_gap']:.1f}")
        if abs(c["closing"] - p["closing"]) > p["closing_tol"]: inv.append(f"closing speed {c['closing']:.2f}")
        if not meta["tunnel_exits_blocked"]: inv.append("tunnel walls missing: side exits not blocked")
        wc = float(np.array([r["wall_clear"] for r in rows]).min())
        if wc < p["wall_tol"]: inv.append(f"a vehicle overlaps the tunnel wall (min wall clearance {wc:.2f} m)")
        if inv: return dict(label="INVALID", reason="setup mismatch: " + ", ".join(inv), detail="", metrics=dict(cue=cuem, min_wall_clearance=wc))
        a = lambda k: np.array([r[k] for r in ev], float)
        bad, mx = envelope(a("accel_long"), a("accel_lat"), pcfg, self.dt)
        fg, rg = a("front_gap"), a("rear_gap")
        stopped = bool(rows[-1]["speed"] < 0.1)
        rv = a("rear_v"); ts = a("t_since_cue")
        t_brake_rear = next((float(t) for t, v in zip(ts, rv) if v < self.v_rear - 0.2), None)
        m = dict(cue=cuem, min_front_gap=float(fg.min()), min_rear_gap=float(rg.min()), final_front_gap=float(fg[-1]), final_rear_gap=float(rg[-1]),
                 ego_stopped=stopped, rear_trigger_to_brake_delay_s=t_brake_rear, max_filtered_accel=mx, min_wall_clearance=wc,
                 ego_brake_start=next((r["t_since_cue"] for r in ev if r["phase"] == "brake"), None))
        roles = sorted(set(contacts.values()))
        fail = []
        if contacts: fail.append("contact: " + ", ".join(("rear-end by truck" if x.startswith("rear_truck") else "front impact with " + x) for x in roles))
        if bad.any() and not contacts: fail.append("physics envelope exceeded")
        if not contacts and fg.min() < p["min_clear"]: fail.append(f"front clearance {fg.min():.2f} m < {p['min_clear']} m")
        if not contacts and rg.min() < p["min_clear"]: fail.append(f"rear clearance {rg.min():.2f} m < {p['min_clear']} m")
        if not contacts and not stopped: fail.append("ego did not stop before time limit")
        det = (f"cue: gaps {c['front_gap']:.1f}/{c['rear_gap']:.1f} m, closing {c['closing']:.1f} m/s; rear brake delay {t_brake_rear}; "
               f"min front {fg.min():.2f} m, min rear {rg.min():.2f} m; wall clearance {wc:.2f} m")
        times = dict(cue=c["sim_time"], ego_brake=m["ego_brake_start"], rear_brake=(c["sim_time"] + t_brake_rear) if t_brake_rear is not None else None)
        if fail: return dict(label="FAIL", reason="; ".join(fail), detail=det, metrics=m, times=times)
        miss = _missing(meta)
        if miss: return dict(label="UNRESOLVED", reason="missing evidence: " + ",".join(miss), detail=det, metrics=m, times=times)
        return dict(label="PASS", reason="stopped between queue and truck with no contact and >= 1 m clearance both sides", detail=det, metrics=m, times=times)

    def overlay(self, r):
        lines = [f"t {r['sim_time']:.2f}s  cue {r['t_since_cue'] if r['t_since_cue']=='' else format(r['t_since_cue'],'+.2f')}",
                 f"ego v {r['speed']:.1f} m/s  a_req {r['a_req']:+.1f}  accel {r['accel_long']:+.1f}",
                 f"FRONT gap {r['front_gap']:.1f} m", f"REAR gap {r['rear_gap']:.1f} m   truck {r['rear_v']:.1f} m/s (closing {r['closing']:+.1f})",
                 f"min wall clearance {r['wall_clear']:.2f} m"]
        ban = []
        if r["t_since_cue"] != "" and r["t_since_cue"] >= self.p["rear_delay_s"] and r["rear_v"] > 0.5: ban.append(("TRUCK BRAKING", (255, 190, 40)))
        return lines, ban


# ------------------------------------------------------------------ RC13
class RC13(BaseScn):
    rc_id, map_name = "RC13-01", "Town03"
    cam_back, cam_up, cam_t, cam_pitch = 12.0, 5.5, 2.5, 18.0
    defaults = dict(road=3, lane=-2, s0=3.0, ego_s=6.0, v_ego=8.0, ped_s=78.0, ped_speed=1.4, ped_t=5.85, preroll_s=2.5, duration_s=11.0,
                    true_v_ped_max=1.8, true_horizon=2.0, ego_bp="vehicle.tesla.model3", speed_tol=0.5,
                    min_speed_ratio=0.9, sidewalk=(4.85, 6.85), lane_half=1.75, dart_speed=0.0, dart_dist=15.0)   # dart: the target pedestrian steps into the lane
    max_frames = 330

    def build(self):
        p = self.p; self.frame = self.lane_frame(p["road"], p["lane"], p["s0"]); fr = self.frame
        self.ego = self.spawn(p["ego_bp"], p["ego_s"], 0.0, "ego", color="200,30,30")
        specs = [(p["ped_s"], p["ped_t"], -1.0, p["ped_speed"], "ped_target"), (p["ped_s"] + 30, p["ped_t"] + 0.2, 1.0, 1.2, "ped_b"),
                 (p["ped_s"] - 25, p["ped_t"] + 0.3, -1.0, 1.1, "ped_c")]
        self.walk = []
        for i, (s, t, d, v, role) in enumerate(specs):
            w = self.spawn(f"walker.pedestrian.{i + 2:04d}", s, t, role, dz=1.0)
            self.walkers.append(w); self.walk.append((w, d, v))
        self.ped_violation = 0; self.ped_in_lane = 0; self.dart_on = False

    def start(self):
        self.set_vel(self.ego, self.p["v_ego"])

    def step(self, st):
        cue = self.mark_cue(st); ef_ = st["fp"][:, 0].max()
        for k, (w, d, v) in enumerate(self.walk):
            l = w.get_location(); sw, tw = self.frame.to_local(l.x, l.y); f = self.frame.fwd(sw)
            if k == 0 and self.p["dart_speed"] > 0:
                if not self.dart_on and cue and sw - ef_ <= self.p["dart_dist"]: self.dart_on = True
                if self.dart_on:                                       # step towards the lane centre (t decreasing = to the left), stand there
                    sp = self.p["dart_speed"] if tw > 0.0 else 0.0
                    w.apply_control(carla.WalkerControl(direction=carla.Vector3D(f[1], -f[0], 0), speed=sp)); continue
            w.apply_control(carla.WalkerControl(direction=carla.Vector3D(f[0] * d, f[1] * d, 0), speed=v))
        peds = []
        for w, d, v in self.walk:
            loc = w.get_location(); ps, pt = self.frame.to_local(loc.x, loc.y); peds.append((ps, pt))
            if not (self.p["sidewalk"][0] - 0.3 <= pt <= self.p["sidewalk"][1] + 0.3): self.ped_violation += 1
            if abs(pt) <= self.p["lane_half"]: self.ped_in_lane += 1
        ps0, pt0 = peds[0]
        ef = st["fp"][:, 0].max()
        o = dict(t=st["t"], v=st["v"], s=st["s"], cue=cue, peds=peds)
        ex = dict(ped_s=ps0, ped_t=pt0, ped_gap_s=ps0 - ef, ped_lat=pt0 - st["fp"][:, 1].max(), ped_violation=int(self.ped_violation > 0),
                  ped_in_lane=int(self.ped_in_lane > 0), ego_left=st["fp"][:, 1].min(), ego_right=st["fp"][:, 1].max())
        return o, ex

    def finished(self, rows, contacts):
        r = rows[-1]
        if r["t_since_cue"] == "": return False, ""
        if r["ped_gap_s"] < -12.0: return True, "passed_pedestrian"
        if r["t_since_cue"] >= self.p["duration_s"]: return True, "time_limit"
        k = int(2.0 / self.dt)
        if len(rows) > k and all(x["speed"] < 0.1 for x in rows[-k:]): return True, "stopped"
        return False, ""

    def meta(self): return dict(ped_violations=self.ped_violation, ped_frames_in_lane=self.ped_in_lane)

    def evaluate(self, rows, contacts, meta, pcfg):
        p = self.p
        cue = next((i for i, r in enumerate(rows) if r["t_since_cue"] != ""), None)
        if cue is None: return dict(label="UNRESOLVED", reason="cue never reached")
        c = rows[cue]; ev = rows[cue:]
        inv = []
        if abs(c["speed"] - p["v_ego"]) > p["speed_tol"]: inv.append(f"start speed {c['speed']:.2f}")
        dart = p["dart_speed"] > 0
        if meta["ped_violations"] and not dart: inv.append("pedestrian left the sidewalk")
        a = lambda k: np.array([r[k] for r in ev], float)
        min_lat = float(a("ped_lat").min())
        if min_lat < 2.0 and not dart: inv.append(f"safe passing space missing (ped lateral distance {min_lat:.2f} m)")
        if inv: return dict(label="INVALID", reason="setup mismatch: " + ", ".join(inv), detail="")
        # ---- physics envelope, evaluated independently of the slowing verdict
        al_raw = a("accel_long"); bad, mx = envelope(al_raw, a("accel_lat"), pcfg, self.dt)
        al_f, ts = a("accel_long_f"), a("t_since_cue")
        phys = []
        if (al_f > pcfg.max_fwd_acc).any():
            i = int(np.argmax(al_f)); phys.append(dict(kind="forward_accel", value=float(al_f[i]), cap=pcfg.max_fwd_acc, t_since_cue=float(ts[i]), phase=ev[i]["phase"], throttle=float(ev[i]["throttle"]), a_req=float(ev[i]["a_req"])))
        if (-al_f > pcfg.max_brake).any():
            i = int(np.argmin(al_f)); phys.append(dict(kind="braking", value=float(-al_f[i]), cap=pcfg.max_brake, t_since_cue=float(ts[i])))
        if (np.abs(a("accel_lat_f")) > pcfg.max_lat_acc).any(): phys.append(dict(kind="lateral", value=float(np.abs(a("accel_lat_f")).max()), cap=pcfg.max_lat_acc))
        v = a("speed"); gp = a("ped_gap_s")
        i_pass = int(np.argmax(gp <= 0)) if (gp <= 0).any() else None
        v_pass = float(v[i_pass]) if i_pass is not None else None
        stopped = bool((v < 0.1).any())
        slowing = bool(stopped or (v_pass is not None and v_pass < p["min_speed_ratio"] * p["v_ego"]) or (i_pass is None))
        entered = bool(meta["ped_frames_in_lane"] > 0)
        gs = a("ped_gap_s"); dxs = np.where(gs > 0, gs, np.where(gs > -4.8, 0.0, -gs - 4.8)); dys = np.maximum(np.abs(a("ped_t")) - 1.08, 0.0)
        min_ped_dist = float((np.hypot(dxs, dys) - 0.3).min())
        pred = a("pred_lane_overlap")
        first_pred = float(ts[int(np.argmax(pred > 0))]) if (pred > 0).any() else None
        confl = a("conflict"); first_conf = float(ts[int(np.argmax(confl > 0))]) if (confl > 0).any() else None
        yield_on = next((float(r["t_since_cue"]) for r in ev if r["phase"] in ("yield", "hold")), None)
        m = dict(first_predicted_lane_overlap_t=first_pred, first_predicted_ego_conflict_t=first_conf, pedestrian_entered_lane=bool(meta["ped_frames_in_lane"] > 0),
                 min_lateral_distance_to_ped=min_lat, yield_onset_t=yield_on, speed_at_pass=v_pass, min_speed=float(v.min()), stopped=stopped,
                 unnecessary_slowing=slowing and not entered, min_ped_dist=min_ped_dist, dart=dart, physics_violations=phys, physics_ok=not phys and not bad.any(), max_filtered_accel=mx,
                 peak_decel_filtered=float(-al_f.min()), peak_fwd_accel_filtered=float(al_f.max()))
        fail = []
        if contacts: fail.append("[CONTACT] " + ", ".join(sorted(set(contacts.values()))))
        if dart and entered and not contacts and min_ped_dist < 0.5: fail.append(f"[CLEARANCE] pedestrian clearance {min_ped_dist:.2f} m < 0.5 m")
        if slowing and not entered:
            fail.append(f"[SLOWING] ego slowed/stopped for a pedestrian who never entered the lane (speed at pass {v_pass if v_pass is None else round(v_pass,1)} m/s vs {p['v_ego']:.0f} target, min speed {v.min():.1f})")
        if phys or bad.any():
            fail.append("[PHYSICS] " + "; ".join(f"{q['kind']} {q['value']:.2f} > cap {q['cap']:.2f} m/s2" for q in phys) if phys else "[PHYSICS] envelope exceeded")
        det = (f"pred. lane overlap at {first_pred}; ped entered lane: {m['pedestrian_entered_lane']}; yield onset {yield_on}; "
               f"pass speed {v_pass if v_pass is None else round(v_pass,1)} m/s; physics ok: {m['physics_ok']}")
        times = dict(cue=c["sim_time"], first_pred_lane_overlap=first_pred, yield_onset=yield_on)
        if fail: return dict(label="FAIL", reason=" | ".join(fail), detail=det, metrics=m, times=times)
        miss = _missing(meta)
        if miss: return dict(label="UNRESOLVED", reason="missing evidence: " + ",".join(miss), detail=det, metrics=m, times=times)
        return dict(label="PASS", reason="passed the pedestrian at near-target speed; no contact, no unnecessary slowing, physics within envelope", detail=det, metrics=m, times=times)

    def overlay(self, r):
        lines = [f"t {r['sim_time']:.2f}s  cue {r['t_since_cue'] if r['t_since_cue']=='' else format(r['t_since_cue'],'+.2f')}",
                 f"ego v {r['speed']:.1f} m/s  a_req {r['a_req']:+.1f}  accel {r['accel_long']:+.1f}",
                 f"pedestrian: {r['ped_gap_s']:.1f} m ahead, {r['ped_lat']:.1f} m lateral",
                 f"reach set overlaps lane: {'YES' if r.get('pred_lane_overlap') else 'no'}   ego conflict: {'YES' if r.get('conflict') else 'no'}"]
        ban = [("PREDICTED CONFLICT - BRAKING", (255, 190, 40))] if r.get("conflict") else []
        return lines, ban


# ------------------------------------------------------------------ RC01 / RC17 (Town01 narrow one-way street)
class RC01(BaseScn):
    """Narrow one-way street (Town01, both lanes of the road used as ONE one-way street, reference line = road centre): pedestrians on both
    sidewalks, continuous rows of parked cars along both kerbs, a delivery van stopped in the narrow corridor between the rows.
    The ego cannot pass on either side (parked rows + van), it can only brake behind the van. Pedestrians stay on the sidewalks.
    delay_s > 0 gives the RC17 information-delay variant."""
    rc_id, map_name = "RC01-01", "Town01"
    cam_back, cam_up, cam_pitch = 13.0, 6.5, 18.0
    cam_fov = 100
    defaults = dict(road=15, lane=-1, s0=3.0, van_s=100.0, gap=20.0, v_ego=8.0, clear=1.35, n_per_side=3, delay_s=0.0, seed=7,
                    ego_bp="vehicle.tesla.model3", van_bp="vehicle.mercedes.sprinter", lane_w=4.0, shoulder=0.3, brake_torque_scale=1.0,
                    parked_bps=("vehicle.audi.a2", "vehicle.toyota.prius", "vehicle.seat.leon", "vehicle.mini.cooper_s", "vehicle.citroen.c3", "vehicle.nissan.micra"),
                    row_range=(6.0, 114.0), walker_range=(10.0, 108.0), cruise_before_cue_s=2.5, accel_est=1.1, row_gap=(0.35, 0.7),
                    speed_tol=0.2, gap_tol=0.5, accel_filter_s=0.2, time_limit=8.0, hold_time=2.0, hold_speed=0.1, min_clearance=2.0)
    max_frames = 700

    def lane_frame(self, road, lane, s):          # reference line = road centre line (the street is treated as one 8 m wide one-way street)
        return Frame.from_lane(self.map, road, lane, s, off=-self.p["lane_w"] / 2)

    def build(self):
        p = self.p; self.rng = random.Random(p["seed"]); self.frame = self.lane_frame(p["road"], p["lane"], p["s0"]); fr = self.frame
        self.road_half = p["lane_w"] + p["shoulder"]
        # sidewalks from the map (right side measured, left side mirrored)
        mid = p["s0"] + 40.0
        wp = self.map.get_waypoint_xodr(p["road"], p["lane"], mid); edge = wp.lane_width / 2; cur = wp; sw = None
        while True:
            cur = cur.get_right_lane()
            if cur is None: break
            if cur.lane_type == carla.LaneType.Sidewalk: sw = (edge, edge + cur.lane_width); break
            edge += cur.lane_width
        sw = sw or (self.road_half - p["lane_w"] / 2, self.road_half - p["lane_w"] / 2 + 4.0)
        self.sw_r = (sw[0] + p["lane_w"] / 2, sw[1] + p["lane_w"] / 2); self.sw_l = (-self.sw_r[1], -self.sw_r[0])
        # ego (from rest) and van
        self.ego = self.spawn(p["ego_bp"], 20.0, 0.0, "ego", color="200,30,30")
        import rc01.calibrate as _cal
        ex = self.ext(self.ego); self.ego_w = 2 * self.ego.bounding_box.extent.y
        # parked rows first (continuous, bumper to bumper) so the corridor width is measured before the van is placed
        self.rows = {"l": [], "r": []}; self.holes = 0; self.hole_pos = []; i = 0
        for side in ("r", "l"):
            s_c = p["row_range"][0] + (3.0 if side == "l" else 0.0)
            while s_c < p["row_range"][1]:
                placed = None
                for k in range(len(p["parked_bps"])):
                    bpid = p["parked_bps"][(i * 5 + k + (2 if side == "l" else 0)) % len(p["parked_bps"])]
                    try: a = self.spawn(bpid, s_c, (self.road_half - 0.9) * (1 if side == "r" else -1), "parked", dz=4.0, dyaw=0.0); placed = a; break
                    except RuntimeError: continue
                i += 1
                if placed is None:
                    self.holes += 1; self.hole_pos.append((side, round(s_c, 1))); s_c += 0.6; continue      # retry a little further on (keeps any hole small)
                hw = placed.bounding_box.extent.y; sg = 1 if side == "r" else -1
                s_c += self.ext(placed)                                           # origin at the car centre
                placed.set_transform(fr.transform(s_c, sg * (self.road_half - 0.1 - hw) + self.rng.uniform(-0.04, 0.04) * (-sg), 0.4, self.rng.uniform(-1.0, 1.0)))
                self.rows[side].append(placed)
                s_c += self.ext(placed) + self.rng.uniform(*p["row_gap"])
        self.parked = self.rows["l"] + self.rows["r"]
        self.inner_nom = 2.15            # nominal inner face (verified from real footprints at start)
        self.van = self.spawn(p["van_bp"], p["van_s"], 0.0, "van", dyaw=1.0)
        vw = 2 * self.van.bounding_box.extent.y
        t_vl = p["clear"] - self.inner_nom                        # van left edge: 'clear' = free width between the left parked row and the van
        self.van.set_transform(fr.transform(p["van_s"], t_vl + vw / 2, 0.4, 1.0))
        self.van_rear_s = p["van_s"] - self.ext(self.van)
        g0 = p["gap"] + p["v_ego"] ** 2 / (2 * p["accel_est"]) + 4.0 + p["v_ego"] * p["cruise_before_cue_s"]
        self.ego_s0 = self.van_rear_s - g0 - ex
        self.ego.set_transform(fr.transform(self.ego_s0, 0.0, 0.4))
        # pedestrians (scripted, kept on the sidewalks) + delivery people at the van
        self.walk = []; k = 0
        for side, (a0, a1) in (("r", self.sw_r), ("l", self.sw_l)):
            for j in range(p["n_per_side"]):
                s_w = p["walker_range"][0] + (j + self.rng.random() * 0.8) * (p["walker_range"][1] - p["walker_range"][0]) / max(1, p["n_per_side"])
                d = 1.0 if j % 2 == 0 else -1.0
                t = a0 + (a1 - a0) * (0.32 if d > 0 else 0.62); w = None
                for ds in (0.0, 1.5, -1.5, 3.0, -3.0, 5.0):
                    try: w = self.spawn(f"walker.pedestrian.{(k * 3) % 40 + 1:04d}", s_w + ds, t, "ped", dz=1.0); break
                    except RuntimeError: continue
                if w is None: continue
                self.walkers.append(w); self.walk.append([w, d, self.rng.uniform(0.9, 1.4), (a0, a1)]); k += 1
        t_dp = self.sw_r[0] + 0.7
        for dx in (-2.5, 2.0):
            try: w = self.spawn(f"walker.pedestrian.{(31 + int(dx) % 7) % 40 + 1:04d}", p["van_s"] + dx, t_dp, "delivery", dz=1.0)
            except RuntimeError: continue
            self.walkers.append(w); self.walk.append([w, 0.0, 0.0, self.sw_r])
        for dx, bpid in ((-1.2, "static.prop.box01"), (0.0, "static.prop.shoppingtrolley"), (1.0, "static.prop.box02")):
            if self.bl.filter(bpid):
                try: self.spawn(bpid, p["van_s"] + dx, t_dp + 1.1, "prop", dz=0.3, dyaw=self.rng.uniform(-25, 25))
                except RuntimeError: pass
        self.ped_violation = 0; self.van_max_speed = 0.0

    def warm(self, cruise):                      # ego starts from rest: just keep it parked while the camera settles
        self.ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

    def start(self):
        self.van_pose0 = self.van.get_location(); self.van_fp0 = fp_local(self.van, self.frame)
        self.row_fp = [(fp_local(a, self.frame), 1 if a in self.rows["r"] else -1) for a in self.parked]
        self.blocked = self.check_blocked()
        self.generated = None; self.delivered = None

    def check_blocked(self):
        """Physical check from real footprints: free width between each parked row and the van must be smaller than the ego width."""
        p = self.p; vf = fp_local(self.van, self.frame); vl, vr, v0, v1 = vf[:, 1].min(), vf[:, 1].max(), vf[:, 0].min(), vf[:, 0].max()
        near = [(f, sg) for f, sg in self.row_fp if f[:, 0].max() > v0 - 3.0 and f[:, 0].min() < v1 + 3.0]
        lf = max([f[:, 1].max() for f, sg in near if sg < 0], default=-self.inner_nom); rf = min([f[:, 1].min() for f, sg in near if sg > 0], default=self.inner_nom)
        # worst corridor across the whole approach (smallest inner-face spacing) -- the ego must fit
        free_l, free_r = vl - lf, rf - vr
        ok = free_l < self.ego_w and free_r < self.ego_w
        # continuity of the rows: largest gap between consecutive parked cars (must be < ego width)
        def maxgap(sg):
            L = sorted([(f[:, 0].min(), f[:, 0].max()) for f, g in self.row_fp if g == sg]); return max([b[0] - a[1] for a, b in zip(L, L[1:])], default=0.0)
        mg = max(maxgap(1), maxgap(-1))
        return dict(ok=bool(ok and mg < self.ego_w), ego_width=self.ego_w, van_t=[float(vl), float(vr)], row_inner_faces=[float(lf), float(rf)],
                    free_width_each_side=[float(free_l), float(free_r)], clear_width_measured=float(free_l), right_free_width=float(free_r),
                    parked_left=len(self.rows["l"]), parked_right=len(self.rows["r"]), row_max_gap_m=float(mg), row_slot_failures=self.holes, row_slot_failure_positions=self.hole_pos,
                    pedestrians=len(self.walkers), note="parked cars on both kerbs + van block the street; the only available action is braking in lane")

    def step(self, st):
        p = self.p
        for wk in self.walk:
            w, d, v, (a0, a1) = wk
            l = w.get_location(); sw, tw = self.frame.to_local(l.x, l.y)
            if not (a0 - 0.3 <= tw <= a1 + 0.3):
                self.ped_violation += 1
                if self.ped_violation <= 3: print(f"[ped violation] walker {w.id} role={self.roles.get(w.id)} s={sw:.1f} t={tw:.2f} sidewalk=({a0:.2f},{a1:.2f}) d={d} t_sim={st['t']:.2f}", flush=True)
            if d != 0.0:
                if sw > p["walker_range"][1]: wk[1] = d = -1.0
                if sw < p["walker_range"][0]: wk[1] = d = 1.0
            f = self.frame.fwd(sw)
            w.apply_control(carla.WalkerControl(direction=carla.Vector3D(f[0] * (d or 1.0), f[1] * (d or 1.0), 0), speed=v))
        vf = fp_local(self.van, self.frame); gap = float(vf[:, 0].min() - st["fp"][:, 0].max())
        self.n_frames += 1
        if self.generated is None and gap <= p["gap"]:
            self.generated = self.n_frames; self.t0 = st["t"]
        delivered = self.generated is not None and (self.n_frames - self.generated) >= int(round(p["delay_s"] / self.dt))
        if delivered and self.delivered is None: self.delivered = self.n_frames
        vv = self.van.get_velocity(); self.van_max_speed = max(self.van_max_speed, math.hypot(vv.x, vv.y))
        vl = self.van.get_location(); moved = vl.distance(self.van_pose0)
        lf, rf = self.blocked["row_inner_faces"]; corr = int(st["fp"][:, 1].min() < lf or st["fp"][:, 1].max() > rf)    # corridor = between the parked rows
        sx = st["fp"][:, 0]
        wall_clear = min([poly_dist(st["fp"], f) for f, sg in self.row_fp if f[:, 0].max() > sx.min() - 3.0 and f[:, 0].min() < sx.max() + 3.0], default=99.0)
        o = dict(frame=st["frame"], t=st["t"], v=st["v"], delivered=delivered, gap=gap)
        ex = dict(gap=gap, obs_generated=int(self.generated is not None), obs_delivered=int(delivered), corridor_violation=corr,
                  van_moved_m=moved, ped_violation=int(self.ped_violation > 0), ego_left=st["fp"][:, 1].min(), ego_right=st["fp"][:, 1].max(), wall_clear=wall_clear)
        return o, ex

    n_frames = 0
    t0 = None

    def geom_contact(self, row):
        return "van" if row["gap"] <= 0.02 else None

    def finished(self, rows, contacts):
        if self.generated is None: return False, ""
        r = rows[-1]; k = int(round(self.p["hold_time"] / self.dt))
        if len(rows) >= k and all(x["speed"] < self.p["hold_speed"] for x in rows[-k:]): return True, "hold_complete"
        if r["t_since_cue"] != "" and r["t_since_cue"] >= self.p["time_limit"]: return True, "time_limit"
        return False, ""

    def meta(self):
        return dict(blocked_check=self.blocked, van_moved_m=0.0, van_max_speed=self.van_max_speed, ped_violations=self.ped_violation)

    def evaluate(self, rows, contacts, meta, pcfg):
        import types
        from rc01.evaluate import evaluate as ev_rc01
        p = self.p
        cfg = types.SimpleNamespace(target_speed=p["v_ego"], speed_tol=p["speed_tol"], target_gap=p["gap"], gap_tol=p["gap_tol"], accel_filter_s=p["accel_filter_s"],
                                    dt=self.dt, time_limit=p["time_limit"], hold_time=p["hold_time"], hold_speed=p["hold_speed"], min_clearance=p["min_clearance"])
        m2 = dict(meta); m2["van_moved_m"] = max(r["van_moved_m"] for r in rows)
        res = ev_rc01(rows, m2, cfg, pcfg)
        cue = next((i for i, r in enumerate(rows) if r["obs_generated"]), None)
        first_brake = next((r for r in rows if r["obs_delivered"] and r["brake"] > 0), None)
        onset = next((r for r in rows if r["obs_delivered"] and r["accel_long"] < -0.5), None)
        recv = next((r for r in rows if r["obs_delivered"]), None)
        res["times"] = dict(observation_captured=rows[cue]["sim_time"] if cue is not None else None, observation_delivered_to_controller=recv["sim_time"] if recv else None,
                            first_brake_application=first_brake["sim_time"] if first_brake else None, onset_measured_decel=onset["sim_time"] if onset else None,
                            recording_before_cue_s=(rows[cue]["sim_time"] - rows[0]["sim_time"]) if cue is not None else None)
        res["metrics"] = {k: res.get(k) for k in ("start_speed", "start_gap", "min_gap", "stopping_gap", "hold_min_gap", "max_filtered_accel")}
        res["metrics"].update(clear_width_measured=self.blocked["clear_width_measured"], van_blocks_route=self.blocked["ok"], pedestrians=self.blocked["pedestrians"],
                              ego_width=self.ego_w, delay_s=p["delay_s"])
        wc = min(r["wall_clear"] for r in rows)
        res["metrics"].update(min_wall_clearance=float(wc), max_lateral_excursion_m=float(max(max(abs(r["ego_left"]), abs(r["ego_right"])) for r in rows)))
        if wc < 0.05 and res["label"] != "INVALID":
            res["label"] = "INVALID"; res["reason"] = f"setup mismatch: ego overlapped a parked car (clearance {wc:.2f} m)"
        sg, mg = res.get("stopping_gap"), res.get("min_gap")
        res["detail"] = (f"Stopping gap {sg:.2f} m (min gap {mg:.2f} m)" if sg is not None else f"Did not stop - minimum gap {mg if mg is None else round(mg,2)} m") + f"; cue {res.get('start_speed') and round(res['start_speed'],2)} m/s at {res.get('start_gap') and round(res['start_gap'],2)} m"
        return res

    def extra_summary(self, rows, res, contacts):
        return dict(start_speed=res.get("start_speed"), start_gap=res.get("start_gap"), min_gap=res.get("min_gap"), stopping_gap=res.get("stopping_gap"),
                    blocked_road_check=self.blocked, observation=dict(source="idealized: bumper gap from CARLA footprints; captured when gap <= target, delivered after delay_s",
                                                                       delay_s=self.p["delay_s"], captured_frame=self.generated, delivered_frame=self.delivered))

    def overlay(self, r):
        lines = [f"sim time {r['sim_time']:.2f} s   (cue {r['t_since_cue'] if r['t_since_cue']=='' else format(r['t_since_cue'],'+.2f')} s)",
                 f"speed {r['speed']:.2f} m/s    gap {r['gap']:.2f} m", f"a_req {r['a_req']:+.2f}  accel {r['accel_long']:+.2f} m/s2  brake {r['brake']:.2f}"]
        ban = []
        if r["obs_generated"] and not r["obs_delivered"]: ban.append(("OBSERVATION CAPTURED - IN TRANSIT", (255, 190, 40)))
        if r["obs_delivered"] and r["t_since_cue"] != "" and r["t_since_cue"] < self.p["delay_s"] + 1.0: ban.append(("OBSERVATION DELIVERED TO CONTROLLER", (255, 120, 60)))
        return lines, ban


# ------------------------------------------------------------------ RC03
class RC03(BaseScn):
    """Spun car beside the ego (Town04 highway, 3 lanes). The victim drives in the neighbouring lane, then slides toward the ego lane (crash=1) or drives on (crash=0).
    Escape lane = the opposite neighbour, with a leader and a follower around a gap. Optional tailgater. All actors are scripted by rcx/rc03_laws (shared with the surrogate)."""
    rc_id, map_name = "RC03-01", "Town04"
    cam_fov = 110
    defaults = dict(road=40, lane=-3, s0=2.0, ego_s=40.0, preroll_s=1.5, duration_s=5.0, ego_bp="vehicle.tesla.model3", victim_bp="vehicle.citroen.c3",
                    esc_bp="vehicle.seat.leon", rear_bp="vehicle.audi.a2", min_clear=0.5, settle_tol=0.6, **__import__("rcx.rc03_laws", fromlist=["x"]).DEFAULTS)
    max_frames = 400

    def camera_pose(self): return self.side_view(self.follow, self.lead, lat_off=-self.p["side"] * 1.75)

    def build(self):
        from . import rc03_laws as L
        p = self.p; self.L = L; self.frame = self.lane_frame(p["road"], p["lane"], p["s0"]); fr = self.frame
        self.ego = self.spawn(p["ego_bp"], p["ego_s"], 0.0, "ego", color="200,30,30")
        e0, pre = p["ego_s"], p["preroll_s"]; ve = p["v_ego"]
        sv, tv, _, vv = L.victim(p, 0.0); sl, tl, vl = L.esc_leader(p, 0.0); sf, tf, vf = L.esc_follower(p, 0.0)
        at = lambda s, v: e0 + s + (ve - v) * pre                       # spawn so that the actor is at its cue position when the ego is at e0 + ve*pre
        self.victim = self.spawn(p["victim_bp"], at(sv, vv), tv, "victim")
        self.lead = self.spawn(p["esc_bp"], at(sl, vl), tl, "esc_lead"); self.follow = self.spawn(p["esc_bp"], at(sf, vf), tf, "esc_follow")
        self.npcs = [NPC(self.lead, fr, lambda rt: L.esc_leader(p, max(rt, 0.0))[2] if rt >= 0 else vl, "esc_lead"),
                     NPC(self.follow, fr, lambda rt: vf, "esc_follow")]
        self.rear = None
        if p["rear"]:
            sr, tr, vr = L.rear_car(p, 0.0); self.rear = self.spawn(p["rear_bp"], at(sr, vr), tr, "rear"); self.npcs.append(NPC(self.rear, fr, lambda rt: vr, "rear"))
        self.vs0, self.vt0 = sv, tv

    def start(self):
        for n in self.npcs: n.init_pose()
        self.set_vel(self.ego, self.p["v_ego"])
        for n in self.npcs: self.set_vel(n.a, n.speed_fn(-1))
        self.vs = self.p["ego_s"] + self.vs0 + (self.p["v_ego"] - self.p["v_post"]) * self.p["preroll_s"]; self._vz = self.victim.get_location().z
        self.set_vel(self.victim, self.p["v_post"])

    def _move_victim(self, rt):
        p, L = self.p, self.L; fr = self.frame
        v0 = p["v_post"]
        if rt < 0.0: self.vs += v0 * self.dt; s, t, yaw, v = self.vs, p["side"] * p["lane_w"], 0.0, v0
        else:
            s0, t, yaw, v = L.victim(p, rt); s = self.p["ego_s"] + p["v_ego"] * p["preroll_s"] + s0
        x, y = fr.to_world(s, t); f = fr.fwd(s)
        self.victim.set_transform(carla.Transform(carla.Location(x, y, self._vz + 0.02), carla.Rotation(yaw=fr.heading(s) + math.degrees(yaw))))
        c, sn = math.cos(math.radians(fr.heading(s) + math.degrees(yaw))), math.sin(math.radians(fr.heading(s) + math.degrees(yaw)))
        self.victim.set_target_velocity(carla.Vector3D(c * v, sn * v, 0.0)); self.victim.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self.victim.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

    def step(self, st):
        cue = self.mark_cue(st); rt = (st["t"] - self.t0) if cue else -1.0
        for n in self.npcs: n.update(rt)
        self._move_victim(rt)
        ex = {}
        for k, a in (("victim", self.victim), ("esc_lead", self.lead), ("esc_follow", self.follow)) + ((("rear", self.rear),) if self.rear else ()):
            fpa = fp_local(a, self.frame); ex[k + "_clear"] = self.aabb_dist(st["fp"], fpa)
            if k == "victim": ex["victim_rear_s"], ex["victim_t"] = fpa[:, 0].min(), fpa[:, 1].mean()
            else: ex[k + "_front_s"], ex[k + "_t"], ex[k + "_v"] = fpa[:, 0].max(), fpa[:, 1].mean(), speed_of(a)
        ex["ego_front_s"] = st["fp"][:, 0].max(); ex["ego_rear_s"] = st["fp"][:, 0].min(); ex["ego_t_c"] = st["fp"][:, 1].mean()
        ex["ego_left"], ex["ego_right"] = st["fp"][:, 1].min(), st["fp"][:, 1].max()
        o = dict(t=st["t"], v=st["v"], s=st["s"], ego_t=st["tl"], cue=cue, hazard=cue)
        return o, ex

    def geom_contact(self, row):
        for k in ("victim", "esc_lead", "esc_follow", "rear"):
            if row.get(k + "_clear", 99.0) <= 0.02: return k
        return None

    def finished(self, rows, contacts):
        r = rows[-1]; return (r["t_since_cue"] != "" and r["t_since_cue"] >= self.p["duration_s"]), "time_limit"

    def evaluate(self, rows, contacts, meta, pcfg):
        p = self.p; cue = next((i for i, r in enumerate(rows) if r["t_since_cue"] != ""), None)
        if cue is None: return dict(label="UNRESOLVED", reason="cue never delivered")
        ev = rows[cue:]; a = lambda k: np.array([r[k] for r in ev], float)
        bad, mx = envelope(a("accel_long"), a("accel_lat"), pcfg, self.dt)
        names = [k for k in ("victim", "esc_lead", "esc_follow", "rear") if k + "_clear" in ev[0]]
        mn = {k: float(a(k + "_clear").min()) for k in names}
        et = a("ego_t"); W = p["lane_w"]; stopped = bool(a("speed")[-1] < 0.3)
        settled = stopped or bool(np.all(np.abs(et[-10:]) < p["settle_tol"]) or np.all(np.abs(et[-10:] + p["side"] * W) < p["settle_tol"]))
        off_road = bool(np.abs(et).max() > 1.5 * W)
        m = dict(min_front_gap=mn["victim"], min_rear_gap=mn.get("rear", 99.0), min_clear_by=mn, final_lateral=float(et[-1]), stopped=stopped, settled=settled, max_filtered_accel=mx)
        fail = []
        if contacts: fail.append("contact with " + ", ".join(sorted(set(contacts.values()))))
        else:
            for k, x in mn.items():
                if x < p["min_clear"]: fail.append(f"{k} clearance {x:.2f} m")
            if not settled: fail.append("ego not settled")
            l = ev[-1]
            if self.L.still_closing(l["victim_rear_s"] - l["ego_front_s"], l["speed"], abs(l["ego_t_c"] - l["victim_t"]) - 2.08, 2.0): fail.append("still closing on the victim")
            for k, hw in (("esc_follow", 0.95), ("rear", 0.95)):
                if k + "_front_s" in l and self.L.rear_closing(l["ego_rear_s"] - l[k + "_front_s"], l[k + "_v"] - l["speed"], abs(l["ego_t_c"] - l[k + "_t"]) - 1.08 - hw, 2.0): fail.append(f"still closing from {k}")
        if off_road: fail.append("ego left the road")
        if bad.any(): fail.append("physics envelope exceeded")
        det = f"min clearance {min(mn.values()):.2f} m; stopped={stopped}; final lateral {et[-1]:+.2f}"
        if fail: return dict(label="FAIL", reason="; ".join(fail), detail=det, metrics=m)
        miss = _missing(meta)
        if miss: return dict(label="UNRESOLVED", reason="missing evidence: " + ",".join(miss), detail=det, metrics=m)
        return dict(label="PASS", reason="no contact, clearances kept, ego settled", detail=det, metrics=m)

    def overlay(self, r):
        return [f"t {r['sim_time']:.2f}s  cue {r['t_since_cue'] if r['t_since_cue']=='' else format(r['t_since_cue'],'+.2f')}",
                f"ego v {r['speed']:.1f} m/s  lat {r['ego_t']:+.2f} m", f"victim clear {r['victim_clear']:.1f} m"], []


# ------------------------------------------------------------------ RC04 / RC06
SIDE_BPS = [(2.3, "vehicle.seat.leon"), (2.6, "vehicle.carlamotors.carlacola"), (4.2, "vehicle.carlamotors.european_hgv"), (5.1, "vehicle.mitsubishi.fusorosa")]   # (half length m, blueprint); re-measured by tools/probe_extents.py


class RC04(BaseScn):
    """Blocked lane with a side vehicle alongside (RC04 truck / RC06 passing car), Town04 highway. A stopped car ahead in the ego lane, a vehicle in the neighbouring lane around the
    ego (constant speed, optional acceleration / braking law) and its follower. All actors are scripted by rcx/rc04_laws (shared with the surrogate)."""
    rc_id, map_name = "RC04-01", "Town04"
    cam_fov = 110
    defaults = dict(road=40, lane=-3, s0=2.0, ego_s=40.0, preroll_s=1.5, duration_s=7.0, ego_bp="vehicle.tesla.model3", obs_bp="vehicle.citroen.c3", fol_bp="vehicle.audi.a2",
                    min_clear=0.5, settle_tol=0.6, **__import__("rcx.rc04_laws", fromlist=["x"]).DEFAULTS)
    max_frames = 460

    def camera_pose(self): return self.side_view(self.fol if self.fol is not None else self.side, self.obstacle, lat_off=-self.p["side"] * 1.75)

    def build(self):
        from . import rc04_laws as L
        p = self.p; self.L = L; self.frame = self.lane_frame(p["road"], p["lane"], p["s0"]); fr = self.frame
        self.ego = self.spawn(p["ego_bp"], p["ego_s"], 0.0, "ego", color="200,30,30")
        side_bp = p.get("side_bp") or min(SIDE_BPS, key=lambda x: abs(x[0] - p["side_hl"]))[1]
        if not self.bl.filter(side_bp): side_bp = SIDE_BPS[0][1]
        e0, pre, ve = p["ego_s"], p["preroll_s"], p["v_ego"]
        os_, ot, _ = L.obstacle(p, 0.0); ss, st_, vs = L.side_vehicle(p, 0.0)
        at = lambda s, v: e0 + s + (ve - v) * pre                       # spawn so that the actor is at its cue position when the ego is at e0 + ve*pre
        self.obstacle = self.spawn(p["obs_bp"], at(os_, 0.0), ot, "obstacle")
        self.side = self.spawn(side_bp, at(ss, vs), st_, "side", dz=0.6)
        self.side_hl_meas = self.ext(self.side)
        self.npcs = [NPC(self.obstacle, fr, None, "obstacle"), NPC(self.side, fr, lambda rt: L.side_vehicle(p, max(rt, 0.0))[2] if rt >= 0 else vs, "side")]
        self.fol = None
        if p["fol"]:
            fs, ft, fv = L.follower(p, 0.0); self.fol = self.spawn(p["fol_bp"], at(fs, fv), ft, "fol")
            self.npcs.append(NPC(self.fol, fr, lambda rt: L.follower(p, max(rt, 0.0))[2] if rt >= 0 else fv, "fol"))

    def start(self):
        for n in self.npcs: n.init_pose()
        self.set_vel(self.ego, self.p["v_ego"])
        for n in self.npcs:
            if n.speed_fn is not None: self.set_vel(n.a, n.speed_fn(-1))

    def step(self, st):
        cue = self.mark_cue(st); rt = (st["t"] - self.t0) if cue else -1.0
        for n in self.npcs: n.update(rt)
        ex = {}
        for k, a in (("obstacle", self.obstacle), ("side", self.side)) + ((("fol", self.fol),) if self.fol is not None else ()):
            fpa = fp_local(a, self.frame); ex[k + "_clear"] = self.aabb_dist(st["fp"], fpa)
            if k == "obstacle": ex["obstacle_rear_s"], ex["obstacle_t"] = fpa[:, 0].min(), fpa[:, 1].mean()
            else: ex[k + "_front_s"], ex[k + "_t"], ex[k + "_v"] = fpa[:, 0].max(), fpa[:, 1].mean(), speed_of(a)
        ex["ego_front_s"] = st["fp"][:, 0].max(); ex["ego_rear_s"] = st["fp"][:, 0].min(); ex["ego_t_c"] = st["fp"][:, 1].mean()
        ex["ego_left"], ex["ego_right"] = st["fp"][:, 1].min(), st["fp"][:, 1].max()
        o = dict(t=st["t"], v=st["v"], s=st["s"], ego_t=st["tl"], cue=cue, hazard=cue)
        return o, ex

    def geom_contact(self, row):
        for k in ("obstacle", "side", "fol"):
            if row.get(k + "_clear", 99.0) <= 0.02: return k
        return None

    def finished(self, rows, contacts):
        r = rows[-1]; return (r["t_since_cue"] != "" and r["t_since_cue"] >= self.p["duration_s"]), "time_limit"

    def evaluate(self, rows, contacts, meta, pcfg):
        p = self.p; cue = next((i for i, r in enumerate(rows) if r["t_since_cue"] != ""), None)
        if cue is None: return dict(label="UNRESOLVED", reason="cue never delivered")
        ev = rows[cue:]; a = lambda k: np.array([r[k] for r in ev], float)
        if abs(self.side_hl_meas - p["side_hl"]) > 0.4:
            return dict(label="INVALID", reason=f"setup mismatch: side vehicle half length {self.side_hl_meas:.2f} m != {p['side_hl']:.2f} m", detail="", metrics={})
        bad, mx = envelope(a("accel_long"), a("accel_lat"), pcfg, self.dt)
        names = [k for k in ("obstacle", "side", "fol") if k + "_clear" in ev[0]]
        mn = {k: float(a(k + "_clear").min()) for k in names}
        et = a("ego_t"); W = p["lane_w"]; stopped = bool(a("speed")[-1] < 0.3)
        settled = stopped or bool(np.all(np.abs(et[-10:]) < p["settle_tol"]) or np.all(np.abs(et[-10:] + p["side"] * W) < p["settle_tol"]))
        off_road = bool(np.abs(et).max() > 1.5 * W)
        m = dict(min_front_gap=mn["obstacle"], min_rear_gap=mn.get("fol", 99.0), min_clear_by=mn, final_lateral=float(et[-1]), stopped=stopped, settled=settled, max_filtered_accel=mx)
        fail = []
        if contacts: fail.append("contact with " + ", ".join(sorted(set(contacts.values()))))
        else:
            for k, x in mn.items():
                if x < p["min_clear"]: fail.append(f"{k} clearance {x:.2f} m")
            if not settled: fail.append("ego not settled")
            l = ev[-1]
            if self.L.still_closing(l["obstacle_rear_s"] - l["ego_front_s"], l["speed"], abs(l["ego_t_c"] - l["obstacle_t"]) - 2.08, 2.0): fail.append("still closing on the obstacle")
            if "fol_front_s" in l and self.L.rear_closing(l["ego_rear_s"] - l["fol_front_s"], l["fol_v"] - l["speed"], abs(l["ego_t_c"] - l["fol_t"]) - 1.08 - 0.95, 2.0): fail.append("still closing from fol")
        if off_road: fail.append("ego left the road")
        if bad.any(): fail.append("physics envelope exceeded")
        det = f"min clearance {min(mn.values()):.2f} m; stopped={stopped}; final lateral {et[-1]:+.2f}"
        if fail: return dict(label="FAIL", reason="; ".join(fail), detail=det, metrics=m)
        miss = _missing(meta)
        if miss: return dict(label="UNRESOLVED", reason="missing evidence: " + ",".join(miss), detail=det, metrics=m)
        return dict(label="PASS", reason="no contact, clearances kept, ego settled", detail=det, metrics=m)

    def overlay(self, r):
        return [f"t {r['sim_time']:.2f}s  cue {r['t_since_cue'] if r['t_since_cue']=='' else format(r['t_since_cue'],'+.2f')}",
                f"ego v {r['speed']:.1f} m/s  lat {r['ego_t']:+.2f} m", f"obstacle clear {r['obstacle_clear']:.1f} m  side clear {r['side_clear']:.1f} m"], []
