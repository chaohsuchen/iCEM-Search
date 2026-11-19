"""Closed-loop CARLA co-simulation of the notebook's scenarios.

The notebook's `Sim` loop is kept unchanged: observations, visibility gates, policies, command delay, the
actuator envelope, reactive actor laws and the r-margin scoring all run in Python every 0.05 s tick.
Only the physics step is replaced:

* ego and reactive actors (`kind="vehicle"`) are CARLA vehicles with full physics, driven by
  VehicleAckermannControl built from the commanded acceleration and steering;
* scripted actors (crash slides, pedestrians, cut-in paths) and static objects are moved kinematically
  along their prescribed paths (set_transform, physics off);
* after every world.tick() the CARLA poses are written back into the notebook's vehicle objects, so
  margins, laws and policies see the CARLA state.

The road is generated from OpenDRIVE with the notebook's geometry (4 m lanes, shoulders, crest profile),
so scenario coordinates map 1:1 to CARLA coordinates (x forward, y to the right).
"""
import json
import math
import os
import queue
import re

import numpy as np

import carla

LANE_W = 4.0
DT = 0.05
X_START, ROAD_LEN = -400.0, 2600.0          # generated road covers x in [-400, 2200] m
SIDE_MAX = 25.0                             # widest off-road margin generated beside the road [m]
VERGE = 12.0                                # minimum ground (raised verge) on each side, for the cameras [m]

# ---------------------------------------------------------------- blueprints (visual stand-ins)
# approximate (length, width) of CARLA 0.9.15 blueprints; the scoring always uses the scenario footprint
CARS = {"vehicle.lincoln.mkz_2020": (4.9, 2.1), "vehicle.audi.a2": (3.7, 1.8),
        "vehicle.mini.cooper_s_2021": (4.0, 2.0), "vehicle.dodge.charger_2020": (5.0, 2.1),
        "vehicle.nissan.patrol_2021": (5.6, 2.2), "vehicle.mercedes.sprinter": (5.9, 2.0),
        "vehicle.ford.ambulance": (6.4, 2.4), "vehicle.carlamotors.carlacola": (5.2, 2.6),
        "vehicle.carlamotors.firetruck": (8.6, 2.9), "vehicle.carlamotors.european_hgv": (10.5, 2.9),
        "vehicle.mitsubishi.fusorosa": (10.3, 3.9)}
EGO_BP = "vehicle.lincoln.mkz_2020"
KEYWORD_BP = [  # (regex on the actor name, blueprint candidates, mode)
    (r"^child", ["walker.pedestrian.0011", "walker.pedestrian.0012", "walker.pedestrian.0009"], "walker"),
    (r"^(ped|walker)", ["walker.pedestrian.0001", "walker.pedestrian.0005", "walker.pedestrian.0015"], "walker"),
    (r"^cyclist", ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"], "vehicle"),
    (r"^motorcycle", ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja"], "vehicle"),
    (r"^bus", ["vehicle.mitsubishi.fusorosa"], "vehicle"),
    (r"truck", ["vehicle.carlamotors.european_hgv", "vehicle.carlamotors.firetruck"], "vehicle"),
    (r"^van", ["vehicle.mercedes.sprinter", "vehicle.ford.ambulance"], "vehicle"),
    (r"^moose", ["static.prop.box03", "static.prop.box01"], "prop"),             # CARLA has no animals
    (r"^(lane_end|ramp_end)$", ["static.prop.constructioncone", "static.prop.trafficcone01"], "cones"),
    (r"^(crate|ladder|wheel|tread)$", ["static.prop.box01", "static.prop.box02", "static.prop.box03"], "prop"),
]
EGO_RGB = (40, 160, 40)
PALETTE = [(30, 90, 200), (200, 120, 20), (120, 120, 130), (150, 30, 30), (230, 230, 230), (20, 20, 20), (90, 60, 160)]
UNLIT = re.compile(r"unlit|no lights|lights off", re.I)
PARKED = re.compile(r"parked|stopped|broken|bay|lay-by|driveway|shoulder", re.I)   # zero-speed vehicles that never move

# story weather for scenarios whose visibility gate models a physical condition (visual only: the
# visibility gates of the notebook still decide what the policy observes)
STORY_WEATHER = {
    "RC11-S1": dict(sun_altitude_angle=-40.0, cloudiness=10.0),                     # night
    "RC11-S2": dict(fog_density=65.0, fog_distance=8.0, fog_falloff=1.0, cloudiness=90.0, sun_altitude_angle=30.0),
    "RC11-S3": dict(sun_altitude_angle=6.0, sun_azimuth_angle=0.0, cloudiness=0.0),  # low sun ahead
}
CLEAR_NOON = dict(cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0,
                  sun_altitude_angle=70.0, sun_azimuth_angle=0.0, fog_density=0.0, mie_scattering_scale=0.0)


# ---------------------------------------------------------------- OpenDRIVE road for a scene
def scene_lateral_extent(sc):
    ys = [sc.bounds[0], sc.bounds[1]]
    for a in sc.actors:
        ys += [a.y0 - a.width / 2, a.y0 + a.width / 2]
    return min(ys), max(ys)


def lane_strips(sc):
    """Left-to-right list of (y_left, y_right, type, mark_on_right_edge)."""
    n = sc.lanes
    lo, hi = sc.bounds
    y_min, y_max = scene_lateral_extent(sc)
    strips = []
    left_edge = -LANE_W / 2
    inner = min(lo, left_edge)
    strips.append((min(max(y_min - 2.0, inner - SIDE_MAX), inner - VERGE), inner, "sidewalk", "none"))
    if lo < left_edge - 0.05:
        strips.append((lo, left_edge, "shoulder", "solid"))
    for k in range(n):
        strips.append((k * LANE_W - LANE_W / 2, k * LANE_W + LANE_W / 2, "driving",
                       "broken" if k < n - 1 else "solid"))
    right_edge = (n - 0.5) * LANE_W
    if hi > right_edge + 0.05:
        strips.append((right_edge, hi, "shoulder", "none"))
    outer = max(hi, right_edge)
    strips.append((outer, max(min(y_max + 2.0, outer + SIDE_MAX), outer + VERGE), "sidewalk", "none"))
    return strips


def make_xodr(sc):
    """Straight road along +x. Reference line = left edge of the leftmost strip; every strip is a right lane, so
    CARLA y (OpenDRIVE y negated) equals the scenario y."""
    strips = lane_strips(sc)
    y_ref = strips[0][0]
    prof = sc.meta.get("profile")
    if prof is not None and hasattr(prof, "opendrive"):
        elev = prof.opendrive(X_START, X_START + ROAD_LEN)
    else:
        elev = [dict(s=0.0, a=0.0, b=0.0, c=0.0, d=0.0)]
    elev_xml = "".join(f'<elevation s="{e["s"]}" a="{e["a"]}" b="{e["b"]}" c="{e["c"]}" d="{e.get("d", 0.0)}"/>'
                       for e in elev)
    left_mark = "solid" if strips[0][2] == "driving" else "none"

    def mark(t):
        if t == "none":
            return '<roadMark sOffset="0" type="none" weight="standard" color="standard" width="0"/>'
        return f'<roadMark sOffset="0" type="{t}" weight="standard" color="standard" width="0.15" laneChange="both"/>'

    lanes = "".join(f'<lane id="{-i - 1}" type="{t}" level="false"><link/>'
                    f'<width sOffset="0" a="{yr - yl:.3f}" b="0" c="0" d="0"/>{mark(m)}</lane>'
                    for i, (yl, yr, t, m) in enumerate(strips))
    return (f'<?xml version="1.0" standalone="yes"?><OpenDRIVE>'
            f'<header revMajor="1" revMinor="4" name="rc_{sc.name}" version="1.0" north="0" south="0" east="0" west="0"/>'
            f'<road name="rc" length="{ROAD_LEN}" id="1" junction="-1"><link/>'
            f'<planView><geometry s="0" x="{X_START}" y="{-y_ref:.3f}" hdg="0" length="{ROAD_LEN}"><line/></geometry></planView>'
            f'<elevationProfile>{elev_xml}</elevationProfile><lateralProfile/>'
            f'<lanes><laneSection s="0"><center><lane id="0" type="none" level="false"><link/>{mark(left_mark)}</lane></center>'
            f'<right>{lanes}</right></laneSection></lanes></road></OpenDRIVE>')


# ---------------------------------------------------------------- helpers
def _grade(sc, x):
    prof = sc.meta.get("profile")
    return float(prof.grade(x)) if prof is not None else 0.0


def _z(sc, x):
    prof = sc.meta.get("profile")
    return float(prof.z(x)) if prof is not None else 0.0


def _plain_camera(bp):
    """No bloom, lens flare or motion blur (debug lane lines would glow; the generated world has no backdrop)."""
    for k, v in (("bloom_intensity", "0"), ("lens_flare_intensity", "0"), ("motion_blur_intensity", "0")):
        if bp.has_attribute(k):
            bp.set_attribute(k, v)


def weather_for(clip_name, handoff):
    sid = clip_name[:7]
    w = dict(CLEAR_NOON)
    w.update(STORY_WEATHER.get(sid, {}))
    rec = handoff.get(sid, {})
    for case, spec in rec.get("benign_cases", {}).items():                 # RC19: the case's declared weather
        if clip_name[8:].startswith(case):
            w.update(spec.get("carla", {}).get("weather", {}))
    return w


def pick_blueprint(lib, name, spec):
    for rx, cands, mode in KEYWORD_BP:
        if re.search(rx, name, re.I):
            for c in cands:
                if lib.filter(c):
                    return c, mode
    if spec.width < 1.2:                                     # small, unnamed object
        return "static.prop.box01", "prop"
    best = min(CARS, key=lambda b: abs(CARS[b][0] - spec.length) + 2 * abs(CARS[b][1] - spec.width))
    return best, "vehicle"


def gear_for(v, n_gears):
    """Gear matching a spawn speed (a vehicle spawned at speed in 1st gear engine-brakes at ~20 m/s^2)."""
    g = 1 if v < 5 else 2 if v < 10 else 3 if v < 16 else 4 if v < 23 else 5 if v < 30 else 6
    return max(1, min(g, n_gears))


WARMUP_TICKS = 6
YAW_KP = 3.0           # yaw-rate feedback gain of the ego's low-level lateral controller
ACCEL_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accel_map.json")


class AccelMap:
    """Measured mean acceleration a(v, pedal) of a blueprint (tests/diag_accel_map.py); pedal < 0 = brake."""
    _maps = None

    def __init__(self, bp_id):
        if AccelMap._maps is None:
            AccelMap._maps = json.load(open(ACCEL_MAP_FILE)) if os.path.exists(ACCEL_MAP_FILE) else {}
        m = AccelMap._maps.get(bp_id) or AccelMap._maps.get(EGO_BP)
        if m is None:                                           # crude default until a map is measured
            m = dict(speeds=[0.0, 40.0], pedals=[-1.0, 0.0, 1.0], accel=[[-8.0, -0.3, 4.0], [-8.0, -0.8, 2.0]])
        self.v = np.array(m["speeds"], dtype=float)
        self.p = np.array(m["pedals"], dtype=float)
        A = np.array(m["accel"], dtype=float)
        for i in range(A.shape[0]):                             # fill unmeasured cells (car stopped) from neighbours
            row, ok = A[i], ~np.isnan(A[i])
            A[i] = np.interp(self.p, self.p[ok], row[ok]) if ok.any() else np.nan
        for j in range(A.shape[1]):
            col, ok = A[:, j], ~np.isnan(A[:, j])
            A[:, j] = np.interp(self.v, self.v[ok], col[ok])
        self.A = np.maximum.accumulate(A, axis=1)               # monotone in pedal

    def pedal(self, a_des, v):
        row = np.array([np.interp(v, self.v, self.A[:, j]) for j in range(len(self.p))])
        if a_des <= row[0]:
            return -1.0
        if a_des >= row[-1]:
            return 1.0
        k = int(np.searchsorted(row, a_des))
        a0, a1 = row[k - 1], row[k]
        return float(self.p[k - 1] + (self.p[k] - self.p[k - 1]) * (a_des - a0) / max(a1 - a0, 1e-6))


class LongCtrl:
    """Track the notebook's speed profile: v_ref integrates the commanded acceleration (exactly as HighwayEnv
    integrates it); pedal = feedforward(a_cmd + kp * (v_ref - v) + ki * integral) through the measured map.
    a_floor(v) (optional, vehicle envelope): the strongest admissible braking; the pedal never asks for more and v_ref
    stays within 1.5 m/s of v (anti-windup: a request the vehicle cannot follow must not turn into a full-pedal jolt).
    a_ceil(v) (optional): the same bound on propulsion."""
    def __init__(self, amap, v0, kp=1.5, ki=0.4, a_floor=None, a_ceil=None):
        self.map, self.v_ref, self.kp, self.ki, self.i = amap, float(v0), kp, ki, 0.0
        self.a_floor, self.a_ceil = a_floor, a_ceil

    def __call__(self, a_cmd, v):
        self.v_ref = max(0.0, self.v_ref + a_cmd * DT)
        if self.a_floor is not None:
            self.v_ref = float(np.clip(self.v_ref, v - 1.5, v + 1.5))
        e = self.v_ref - v
        self.i = float(np.clip(self.i + e * DT, -2.0, 2.0))
        if self.v_ref <= 0.0 and v < 0.3:
            return 0.0, 1.0                                     # hold at standstill
        a = a_cmd + self.kp * e + self.ki * self.i
        if self.a_floor is not None:
            a = max(a, self.a_floor(v))
        if self.a_ceil is not None:
            a = min(a, self.a_ceil(v))
        p = self.map.pedal(a, v)
        return max(p, 0.0), max(-p, 0.0)


class Handle:
    def __init__(self, name, actor, mode, bp_id, length, width, wheelbase=None, z_off=0.0, extra=()):
        self.name, self.actor, self.mode, self.bp_id = name, actor, mode, bp_id
        self.length, self.width, self.wheelbase, self.z_off = length, width, wheelbase, z_off
        self.extra = list(extra)              # additional visual actors (cone rows)
        self.long, self.max_steer, self.steer_curve, self.prev_heading = None, None, None, None

    def steer_limit(self, v):
        """Max front-wheel angle [rad] at speed v (CARLA scales it by the blueprint's steering curve, km/h)."""
        f = np.interp(v * 3.6, self.steer_curve[0], self.steer_curve[1]) if self.steer_curve else 1.0
        return math.radians(self.max_steer) * f


# ---------------------------------------------------------------- the bridge
class CarlaBridge:
    def __init__(self, host="localhost", port=2000, video=True, res=(960, 540), bev_h=300, debug_marks=True,
                 tau=0.6, timeout=120.0, yaw_kp=YAW_KP):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.video, self.res, self.bev_h, self.debug_marks, self.tau = video, res, bev_h, debug_marks, tau
        self.yaw_kp = yaw_kp
        self.world, self.xodr = None, None

    # ---- world ----
    def load_road(self, sc):
        xodr = make_xodr(sc)
        if xodr == self.xodr:
            return
        params = carla.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=50.0, wall_height=0.0,
                                                     additional_width=0.0, smooth_junctions=True,
                                                     enable_mesh_visibility=True)
        self.world = self.client.generate_opendrive_world(xodr, params)
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, DT
        s.substepping, s.max_substep_delta_time, s.max_substeps = True, 0.01, 10
        s.no_rendering_mode = not self.video
        self.world.apply_settings(s)
        self.world.tick()
        self.map, self.lib = self.world.get_map(), self.world.get_blueprint_library()
        self.xodr = xodr
        self._check_geometry(sc)
        if self.debug_marks:
            self._draw_marks(sc)

    def _check_geometry(self, sc):
        for k in range(sc.lanes):
            wp = self.map.get_waypoint(carla.Location(100.0, k * LANE_W, _z(sc, 100.0)), project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            loc = wp.transform.location
            assert abs(loc.y - k * LANE_W) < 0.1 and abs(wp.lane_width - LANE_W) < 0.05, \
                f"lane {k}: CARLA centre y={loc.y:.2f} width={wp.lane_width:.2f} (expected {k * LANE_W}, {LANE_W})"

    def _draw_marks(self, sc):
        """Lane lines as persistent debug lines (the generated mesh has no painted markings)."""
        dbg, white = self.world.debug, carla.Color(150, 150, 150)
        n = sc.lanes
        lines = [(-LANE_W / 2, False), ((n - 0.5) * LANE_W, False)] + [((k - 0.5) * LANE_W, True) for k in range(1, n)]
        for y, dashed in lines:
            x = X_START + 50
            while x < X_START + ROAD_LEN - 50:
                L = 3.0 if dashed else 10.0
                a = carla.Location(x, y, _z(sc, x) + 0.03)
                b = carla.Location(x + L, y, _z(sc, x + L) + 0.03)
                dbg.draw_line(a, b, thickness=0.12, color=white, life_time=0.0)
                x += 12.0 if dashed else 10.0

    def transform(self, sc, x, y, psi, z_off=0.0):
        g = _grade(sc, x)
        return carla.Transform(carla.Location(float(x), float(y), _z(sc, x) + z_off),
                               carla.Rotation(pitch=math.degrees(math.atan(g)), yaw=math.degrees(psi), roll=0.0))

    # ---- actors ----
    def spawn(self, sim, clip_name, ego_mode="physics", actor_physics=False):
        sc = sim.sc
        self.load_road(sc)
        self.world.set_weather(carla.WeatherParameters(**weather_for(clip_name, sim.ns.get("HANDOFF", {}))))
        colours = sc.meta.get("colours", {})
        night = clip_name.startswith("RC11-S1")
        self.handles = {}
        specs = [("ego", sim.ego, None)] + [(n, ob, ob.spec) for n, ob in sim.objs.items()]
        for name, ob, spec in specs:
            if spec is None:
                bp_id, mode = EGO_BP, ego_mode
                L, W = ob.LENGTH, ob.WIDTH
            else:
                bp_id, mode = pick_blueprint(self.lib, name, spec)
                L, W = spec.length, spec.width
                if mode == "vehicle":
                    parked = spec.speed == 0 and PARKED.search(spec.note)
                    # other road users follow their scenario laws exactly by default (still reacting to the CARLA ego);
                    # CARLA blueprints cannot always realise them (e.g. the HGV brakes at ~3 m/s^2)
                    phys = actor_physics and spec.kind == "vehicle" and ego_mode == "physics" and not parked
                    mode = "physics" if phys and not re.match(r"^(cyclist|motorcycle)", name) else "kinematic"
            bp = self.lib.find(bp_id)
            if bp.has_attribute("color"):
                rgb = colours.get(name) or (None if clip_name.startswith("RC19") else
                                            (EGO_RGB if name == "ego" else PALETTE[len(self.handles) % len(PALETTE)]))
                if rgb is not None:                    # RC19: only the case's declared colours (appearance is the test)
                    bp.set_attribute("color", ",".join(str(int(c)) for c in rgb))
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero" if name == "ego" else name)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            z_off = 1.0 if mode == "walker" else (0.25 if mode == "physics" else 0.0)
            extra = []
            if mode == "cones":
                cones = self._cone_row(sc, ob, bp_id)
                actor, extra = (cones[0] if cones else None), cones[1:]
            else:
                actor = self.world.try_spawn_actor(bp, self.transform(sc, *ob.position, ob.heading, z_off))
                if actor is None:                                 # overlapping spawn: lift and retry
                    actor = self.world.spawn_actor(bp, self.transform(sc, *ob.position, ob.heading, z_off + 1.0))
            wb, pc = None, None
            if mode == "physics":
                pc = actor.get_physics_control()
                wb = abs(pc.wheels[0].position.x - pc.wheels[-1].position.x) / 100.0 or 2.8
            elif mode != "cones":
                actor.set_simulate_physics(False)
            if night and name != "ego" and spec is not None and not UNLIT.search(spec.note) and mode in ("physics", "kinematic"):
                actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position))
            if night and name == "ego":
                actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position))
            h = self.handles[name] = Handle(name, actor, mode, bp_id, L, W, wb, z_off, extra)
            if pc is not None:
                h.max_steer = pc.wheels[0].max_steer_angle
                curve = list(getattr(pc, "steering_curve", []) or [])
                h.steer_curve = ([c.x for c in curve], [c.y for c in curve]) if curve else None
                h.n_gears = max(1, len(pc.forward_gears))
                h.long = LongCtrl(AccelMap(bp_id), ob.speed)
        self.collisions = []
        bp = self.lib.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.handles["ego"].actor)
        self.col_sensor.listen(lambda ev: self.collisions.append(
            (ev.frame, ev.other_actor.type_id if ev.other_actor else "?",
             next((h.name for h in self.handles.values() if ev.other_actor and h.actor.id == ev.other_actor.id), "?"))))
        if self.video:
            self._spawn_cameras(sc)
        # settle: a few warm-up ticks holding every actor at its exact start pose and speed, so that physics
        # vehicles start in a matching gear with the engine at speed (a cold start engine-brakes at ~20 m/s^2)
        self.world.tick()
        for k in range(WARMUP_TICKS):
            for name, ob, _ in specs:
                h = self.handles[name]
                if h.mode == "cones":
                    continue
                h.actor.set_transform(self.transform(sc, *ob.position, ob.heading, h.z_off if h.mode != "physics" else 0.02))
                if h.mode == "physics":
                    v = ob.speed
                    h.actor.set_target_velocity(carla.Vector3D(v * math.cos(ob.heading), v * math.sin(ob.heading), 0.0))
                    h.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    if k == 0:
                        h.actor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=gear_for(v, h.n_gears)))
                    else:
                        thr, brk = h.long.map.pedal(0.0, v), 0.0
                        h.actor.apply_control(carla.VehicleControl(throttle=max(thr, 0.0), brake=max(-thr, 0.0)))
            self.world.tick()
        self._drain_cameras()

    def _cone_row(self, sc, ob, bp_id):
        """A blocked lane (lane_end-type obstacle) drawn as cones every 6 m along its first 60 m, plus its taper."""
        x0, y0, L, W = ob.position[0], ob.position[1], ob.LENGTH, ob.WIDTH
        bp = self.lib.find(bp_id)
        actors = []
        xs = np.arange(x0 - L / 2, min(x0 + L / 2, x0 - L / 2 + 60.0) + 1e-6, 6.0)
        for i, x in enumerate(xs):
            for y in ((y0,) if i else np.linspace(y0 - W / 2 + 0.3, y0 + W / 2 - 0.3, 4)):
                a = self.world.try_spawn_actor(bp, self.transform(sc, x, y, 0.0, 0.0))
                if a is not None:
                    a.set_simulate_physics(False)
                    actors.append(a)
        return actors

    def _spawn_cameras(self, sc):
        bp = self.lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.res[0]))
        bp.set_attribute("image_size_y", str(self.res[1]))
        bp.set_attribute("fov", "90")
        _plain_camera(bp)
        self.cam_q = {"chase": queue.Queue(), "bev": queue.Queue()}
        ego = self.handles["ego"].actor
        self.chase = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=-9.0, z=3.6), carla.Rotation(pitch=-11.0)),
                                            attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        bp2 = self.lib.find("sensor.camera.rgb")
        bp2.set_attribute("image_size_x", str(self.res[0]))
        bp2.set_attribute("image_size_y", str(self.bev_h))
        bp2.set_attribute("fov", "70")
        _plain_camera(bp2)
        self.bev_sc = sc
        self.bev = self.world.spawn_actor(bp2, self._bev_tf(sc, 0.0))
        self.chase.listen(lambda im: self.cam_q["chase"].put(im))
        self.bev.listen(lambda im: self.cam_q["bev"].put(im))

    def _bev_tf(self, sc, x_ego):
        lo, hi = sc.bounds
        xc = x_ego + 18.0
        return carla.Transform(carla.Location(xc, 0.5 * (lo + hi), _z(sc, xc) + 55.0), carla.Rotation(pitch=-90.0, yaw=-90.0))

    def _drain_cameras(self):
        if self.video:
            for q in self.cam_q.values():
                while not q.empty():
                    q.get()

    def _grab(self, frame):
        out = {}
        for k, q in self.cam_q.items():
            while True:
                im = q.get(timeout=10.0)
                if im.frame >= frame:
                    break
            a = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(im.height, im.width, 4)[:, :, 2::-1]
            out[k] = a.copy()
        return out

    # ---- one physics step (replaces HighwayEnv's road.step) ----
    def control(self, ob, h):
        a, d, v = float(ob.action["acceleration"]), float(ob.action["steering"]), float(ob.speed)
        beta = math.atan(0.5 * math.tan(d))
        yaw_rate = v * math.sin(beta) / (ob.LENGTH / 2)          # HighwayEnv bicycle yaw rate of this vehicle
        # same yaw rate on this vehicle's wheelbase (CG bicycle: yaw_rate = v sin(beta) / (wb/2), beta = atan(tan(d)/2))
        sb = max(-0.99, min(0.99, yaw_rate * (h.wheelbase / 2) / max(v, 1.0)))
        wheel = math.atan(2.0 * math.tan(math.asin(sb)))
        # low-level yaw-rate tracking: feedforward wheel angle + feedback on the measured yaw rate (CARLA's tyres
        # respond with ~0.3 s lag; HighwayEnv's kinematic model has none)
        if h.prev_heading is not None and v > 1.0:
            r_meas = math.atan2(math.sin(ob.heading - h.prev_heading), math.cos(ob.heading - h.prev_heading)) / DT
            wheel += self.yaw_kp * (yaw_rate - r_meas) * h.wheelbase / v
        h.prev_heading = float(ob.heading)
        throttle, brake = h.long(a, v)
        return carla.VehicleControl(throttle=float(throttle), brake=float(brake),
                                    steer=float(np.clip(wheel / h.steer_limit(v), -1.0, 1.0)))

    def step(self, sim):
        sc = sim.sc
        items = [("ego", sim.ego)] + list(sim.objs.items())
        replay = self.handles["ego"].mode != "physics"
        if replay:                                                 # pure replay: HighwayEnv's own step (incl. its
            sim.hw_step(DT)                                        # collision handling), CARLA follows
        for name, ob in items:
            h = self.handles[name]
            if h.mode == "physics":
                h.actor.apply_control(self.control(ob, h))
            elif h.mode == "cones" or (name != "ego" and ob.spec.kind == "static"):
                continue
            else:                                                  # kinematic: HighwayEnv integrates, CARLA follows
                if not replay:
                    ob.step(DT)
                h.actor.set_transform(self.transform(sc, *ob.position, ob.heading, h.z_off))
        if self.video:
            self.bev.set_transform(self._bev_tf(sc, float(sim.ego.position[0])))
        frame = self.world.tick()
        for name, ob in items:
            h = self.handles[name]
            if h.mode != "physics":
                continue
            tf, vel = h.actor.get_transform(), h.actor.get_velocity()
            yaw = math.radians(tf.rotation.yaw)
            ob.position = np.array([tf.location.x, tf.location.y], dtype=float)
            ob.heading = float(math.atan2(math.sin(yaw), math.cos(yaw)))
            ob.speed = max(0.0, vel.x * math.cos(yaw) + vel.y * math.sin(yaw))
            ob.on_state_update()
        if self.video:
            self.frames.append(self._grab(frame))

    def cleanup(self):
        actors = [self.col_sensor] + ([self.chase, self.bev] if self.video else [])
        for s in actors:
            s.stop()
        for h in self.handles.values():
            actors += [h.actor] + h.extra
        self.client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in actors if a is not None], True)
        self.world.tick()


def make_carla_sim(ns):
    """Subclass of the notebook's Sim whose physics step runs in CARLA."""
    Sim = ns["Sim"]

    class CarlaSim(Sim):
        def __init__(self, scene, policy, bridge, clip_name, ego_mode="physics", actor_physics=False):
            super().__init__(scene, policy, render=False)
            self.ns, self.bridge = ns, bridge
            bridge.frames = []
            bridge.spawn(self, clip_name, ego_mode, actor_physics)
            self.hw_step = self.road.step
            self.road.step = lambda dt: bridge.step(self)      # the only change to the notebook's loop

        def run(self):
            try:
                tr = super().run()
            finally:
                self.bridge.cleanup()
            tr.carla_collisions = list(self.bridge.collisions)
            tr.carla_frames = self.bridge.frames
            tr.blueprints = {n: (h.bp_id, h.mode) for n, h in self.bridge.handles.items()}
            return tr

    return CarlaSim
