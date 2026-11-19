"""Scene construction: generated one-way street, parked cars, van, sidewalk pedestrians, geometry helpers."""
import random
import carla
import numpy as np


def xodr(cfg):
    L, lw, pw, sw = cfg.road_length, cfg.lane_width, cfg.parking_width, cfg.sidewalk_width

    def lane(i, t, w):
        return (f'<lane id="{i}" type="{t}" level="false"><width sOffset="0" a="{w}" b="0" c="0" d="0"/>'
                f'<roadMark sOffset="0" type="none" weight="standard" color="standard" width="0.1"/></lane>')
    return f'''<?xml version="1.0"?>
<OpenDRIVE><header revMajor="1" revMinor="4" name="{cfg.scene_id}" version="1" date="2026" north="0" south="0" east="0" west="0"/>
<road name="street" length="{L}" id="1" junction="-1"><link/><type s="0" type="town"/>
<planView><geometry s="0" x="0" y="0" hdg="0" length="{L}"><line/></geometry></planView>
<elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
<lateralProfile/>
<lanes><laneSection s="0">
<left>{lane(2,'sidewalk',sw)}{lane(1,'parking',pw)}</left>
<center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="none" weight="standard" color="standard" width="0.1"/></lane></center>
<right>{lane(-1,'driving',lw)}{lane(-2,'parking',pw)}{lane(-3,'sidewalk',sw)}</right>
</laneSection></lanes></road></OpenDRIVE>'''


def load_world(client, cfg):
    params = carla.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=50.0, wall_height=0.0,
                                                 additional_width=0.0, smooth_junctions=False,
                                                 enable_mesh_visibility=True)
    client.generate_opendrive_world(xodr(cfg), params)
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, cfg.dt
    from .calibrate import physics_settings; physics_settings(s, cfg.dt)
    world.apply_settings(s)
    wp = carla.WeatherParameters.ClearNoon
    wp.fog_density, wp.fog_distance, wp.fog_falloff = 30.0, 20.0, 0.8
    world.set_weather(wp)
    for _ in range(5):
        world.tick()
    return world


def lanes(cfg):
    """Lateral bounds (y) of each strip."""
    lw, pw, sw = cfg.lane_width, cfg.parking_width, cfg.sidewalk_width
    return dict(drive=(0, lw), park_r=(lw, lw + pw), walk_r=(lw + pw, lw + pw + sw),
                park_l=(-pw, 0), walk_l=(-pw - sw, -pw))


def footprint(actor):
    v = actor.bounding_box.get_world_vertices(actor.get_transform())
    return np.array([[p.x, p.y] for p in v])      # 8 corners, planar use


def rear_front_x(actor):
    f = footprint(actor)
    return f[:, 0].min(), f[:, 0].max()


class Scene:
    def __init__(self, world, cfg):
        self.world, self.cfg = world, cfg
        self.bl = world.get_blueprint_library()
        self.actors, self.parked, self.walkers, self.walker_dir, self.walker_speed = [], [], [], [], []
        self.rng = random.Random(cfg.seed)

    def _spawn(self, bp_id, x, y, yaw=0.0, z=0.5, color=None):
        bp = self.bl.find(bp_id)
        if bp.has_attribute("color") and color is None:
            color = bp.get_attribute("color").recommended_values[0]
        if color and bp.has_attribute("color"):
            bp.set_attribute("color", color)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "rc01")
        a = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw)))
        if a is None:
            raise RuntimeError(f"spawn failed: {bp_id} at x={x} y={y}")
        self.actors.append(a)
        return a

    def build(self):
        c, ln = self.cfg, lanes(self.cfg)
        self.van = self._spawn(c.van_bp, c.van_x, c.lane_width - c.van_width / 2, yaw=2.5)   # pulled to the right curb, slightly angled
        # parked cars on both curbs: irregular spacing, small yaw/lateral jitter, mixed models
        ys = {"r": sum(ln["park_r"]) / 2, "l": sum(ln["park_l"]) / 2}
        i = 0
        for side in "rl":
            x = c.parked_x_range[0] + (3.0 if side == "l" else 0.0)
            while x < c.parked_x_range[1]:
                bp = c.parked_bps[(i * 5 + (2 if side == "l" else 0)) % len(c.parked_bps)]; i += 1
                if side == "l" and abs(x - c.van_x) < 6.0:                 # a left-curb car right beside the van: no bypass
                    x = c.van_x - 0.5 + self.rng.uniform(-0.5, 0.5)
                if side == "r" and abs(x - c.van_x) < 3.5:                 # van occupies this stretch of the right curb
                    x += 6.0; continue
                self.parked.append(self._spawn(bp, float(x), ys[side] + self.rng.uniform(-0.12, 0.12), yaw=self.rng.uniform(-2.5, 2.5)))
                x += c.parked_spacing + self.rng.uniform(-0.6, 2.2)
                if self.rng.random() < 0.18: x += 5.5                      # a free parking slot
        # pedestrians on both sidewalks: alternating directions, each direction in its own lateral lane so they pass each other
        inner = {"r": (ln["walk_r"][0], 1.0), "l": (ln["walk_l"][1], -1.0)}     # inner sidewalk edge, sign towards the outer edge
        k = 0
        for side in "rl":
            e0, sg = inner[side]
            for j in range(c.n_walkers_per_side):
                x = c.walker_x_range[0] + (j + 0.5) * (c.walker_x_range[1] - c.walker_x_range[0]) / max(1, c.n_walkers_per_side) + self.rng.uniform(-3, 3)
                d = 1.0 if j % 2 == 0 else -1.0
                y = e0 + sg * (0.5 if d > 0 else 1.2)
                w = self._spawn(f"walker.pedestrian.{(k * 3) % 40 + 1:04d}", x, y, z=1.0)
                self.walkers.append(w); self.walker_dir.append(d); self.walker_speed.append(self.rng.uniform(0.9, 1.4)); k += 1
        if c.dressing: self._dress(ln)
        self.world.tick()
        for _ in range(60):
            self.hold_static()
            self.world.tick()
        self.van_pose0 = self.van.get_location()

    def _dress(self, ln):
        """Sidewalk furniture and a delivery scene next to the van (props only exist if the blueprint is available)."""
        c = self.cfg
        furn = ["static.prop.trashcan01", "static.prop.trashcan03", "static.prop.plantpot05", "static.prop.mailbox",
                "static.prop.bin", "static.prop.streetsign", "static.prop.trashcan04", "static.prop.plantpot08"]
        furn = [f for f in furn if self.bl.filter(f)]
        n = 0
        for side, y0, face in (("r", ln["walk_r"][1] - 0.4, 0.0), ("l", ln["walk_l"][0] + 0.4, 180.0)):
            x = c.parked_x_range[0] - 10
            while x < c.parked_x_range[1] + 10:
                if furn:
                    try: self._spawn(furn[n % len(furn)], x, y0, yaw=face, z=0.3); n += 1
                    except RuntimeError: pass
                x += self.rng.uniform(11, 19)
        # delivery scene by the van: boxes + trolley on the right sidewalk, two standing delivery people
        yw = (ln["walk_r"][0] + 1.7)
        for k, (dx, bp) in enumerate([(-1.5, "static.prop.box01"), (-0.5, "static.prop.box02"), (0.8, "static.prop.shoppingtrolley"), (2.0, "static.prop.box03")]):
            if self.bl.filter(bp):
                try: self._spawn(bp, c.van_x + dx, yw + 0.2 * (k % 2), yaw=self.rng.uniform(-20, 20), z=0.3)
                except RuntimeError: pass
        for dx in (-2.6, 3.0):
            w = self._spawn(f"walker.pedestrian.{(30 + int(dx) % 7) % 40 + 1:04d}", c.van_x + dx, ln["walk_r"][0] + 0.7, z=1.0)
            self.walkers.append(w); self.walker_dir.append(0.0); self.walker_speed.append(0.0)

    def hold_static(self):
        for a in [self.van] + self.parked:
            a.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

    def move_walkers(self):
        lo, hi = self.cfg.walker_x_range
        for i, w in enumerate(self.walkers):
            x = w.get_location().x
            if self.walker_dir[i] != 0.0:
                if x > hi: self.walker_dir[i] = -1.0
                if x < lo: self.walker_dir[i] = 1.0
            w.apply_control(carla.WalkerControl(direction=carla.Vector3D(self.walker_dir[i] or 1.0, 0, 0),
                                                speed=self.walker_speed[i]))

    def check_blocked(self, ego_width):
        """Neither side of the van offers a legal route wide enough for ego (full footprints)."""
        c, ln = self.cfg, lanes(self.cfg)
        vf = footprint(self.van)
        vy0, vy1 = vf[:, 1].min(), vf[:, 1].max()
        lo, hi = c.corridor
        legal_free = [vy0 - lo, hi - vy1]           # free width left/right of van inside the lane
        # physical (illegal) bypass: gap between van and nearest parked car overlapping the van's x-span
        vx0, vx1 = vf[:, 0].min() - 4.7, vf[:, 0].max() + 4.7
        phys = {}
        for name, edge, sign in (("left", vy0, -1), ("right", vy1, 1)):
            nearest = ln["walk_l"][1] if sign < 0 else ln["walk_r"][0]
            for p in self.parked:
                pf = footprint(p)
                if pf[:, 0].max() < vx0 or pf[:, 0].min() > vx1: continue
                if sign > 0 and pf[:, 1].min() >= edge: nearest = min(nearest, pf[:, 1].min())
                if sign < 0 and pf[:, 1].max() <= edge: nearest = max(nearest, pf[:, 1].max())
            phys[name] = float(abs(nearest - edge))
        ok = all(w < ego_width for w in legal_free) and all(w < ego_width for w in phys.values())
        return dict(ok=bool(ok), ego_width=ego_width, clear_width_measured=float(max(legal_free)),
                    lane_width=float(hi - lo), pedestrians=len(self.walkers), van_y=[float(vy0), float(vy1)],
                    legal_free_width_each_side=[float(w) for w in legal_free],
                    physical_gap_to_parked_cars=phys,
                    total_free_in_lane=float(legal_free[0] + legal_free[1]))

    def destroy(self):
        for a in self.actors:
            try: a.destroy()
            except Exception: pass
        self.actors = []
