"""Minimal kinematic stand-in for the CARLA client API, for testing carla_rc without a server.

Vehicles under VehicleAckermannControl follow a kinematic bicycle (rear-axle reference) with ideal
acceleration tracking; everything else stays where it is put. No rendering.
"""
import math
import types


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


Location = Vector3D


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location, self.rotation = location or Location(), rotation or Rotation()


class Color:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r, self.g, self.b, self.a = r, g, b, a


class VehicleAckermannControl:
    def __init__(self, steer=0.0, steer_speed=0.0, speed=0.0, acceleration=0.0, jerk=0.0):
        self.steer, self.steer_speed, self.speed, self.acceleration, self.jerk = steer, steer_speed, speed, acceleration, jerk


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0):
        self.throttle, self.steer, self.brake, self.manual_gear_shift, self.gear = throttle, steer, brake, manual_gear_shift, gear


class VehicleLightState(int):
    NONE, Position, LowBeam = 0, 1, 2


class LaneType:
    Driving = 2


class AttachmentType:
    Rigid = 0


class WeatherParameters:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class OpendriveGenerationParameters:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class WorldSettings:
    synchronous_mode = False
    fixed_delta_seconds = None
    substepping = True
    max_substep_delta_time = 0.01
    max_substeps = 10
    no_rendering_mode = False


command = types.SimpleNamespace(DestroyActor=lambda aid: ("destroy", aid))


class _Blueprint:
    def __init__(self, bp_id):
        self.id, self.attrs = bp_id, {"color": "0,0,0", "role_name": "", "is_invincible": "true"} if bp_id.startswith("vehicle") else {}

    def has_attribute(self, k):
        return k in self.attrs or self.id.startswith("sensor")

    def set_attribute(self, k, v):
        self.attrs[k] = v


class _Library:
    IDS = ["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.mini.cooper_s_2021",
           "vehicle.dodge.charger_2020", "vehicle.nissan.patrol_2021", "vehicle.mercedes.sprinter", "vehicle.ford.ambulance",
           "vehicle.carlamotors.carlacola", "vehicle.carlamotors.firetruck", "vehicle.carlamotors.european_hgv",
           "vehicle.mitsubishi.fusorosa", "vehicle.bh.crossbike", "vehicle.yamaha.yzf", "walker.pedestrian.0001",
           "walker.pedestrian.0011", "static.prop.box01", "static.prop.box03", "static.prop.constructioncone",
           "sensor.other.collision", "sensor.camera.rgb"]

    def filter(self, pat):
        return [i for i in self.IDS if i == pat]

    def find(self, bp_id):
        return _Blueprint(bp_id)


class _Wheel:
    def __init__(self, x):
        self.position, self.max_steer_angle = Vector3D(x * 100.0, 0, 0), 70.0


class _Physics:
    wheels = [_Wheel(1.4), _Wheel(1.4), _Wheel(-1.4), _Wheel(-1.4)]
    forward_gears = [0] * 6
    steering_curve = []


class Actor:
    _next = 1

    def __init__(self, world, bp, tf):
        self.id, Actor._next = Actor._next, Actor._next + 1
        self.world, self.type_id, self.tf = world, bp.id, Transform(Location(tf.location.x, tf.location.y, tf.location.z),
                                                                     Rotation(tf.rotation.pitch, tf.rotation.yaw, 0.0))
        self.v, self.physics, self.ctrl, self.cb = 0.0, True, None, None

    def get_transform(self):
        return self.tf

    def set_transform(self, tf):
        self.tf = tf

    def get_velocity(self):
        y = math.radians(self.tf.rotation.yaw)
        return Vector3D(self.v * math.cos(y), self.v * math.sin(y), 0.0)

    def set_target_velocity(self, vel):
        self.v = math.hypot(vel.x, vel.y)

    def set_target_angular_velocity(self, w):
        pass

    def set_simulate_physics(self, on):
        self.physics = on

    def get_physics_control(self):
        return _Physics()

    def apply_ackermann_control(self, c):
        self.ctrl = c

    def apply_control(self, c):
        if c.manual_gear_shift and c.throttle == 0 and c.brake == 0 and c.steer == 0:
            return                                               # gear selection only
        # ideal drivetrain matching AccelMap's default map: a = 4 * throttle - 8 * brake - 0.3 (rolling)
        a = 4.0 * c.throttle - 8.0 * c.brake - 0.3
        self.ctrl = VehicleAckermannControl(steer=math.radians(70.0) * c.steer, speed=1e3 if a > 0 else 0.0,
                                            acceleration=abs(a))

    def set_light_state(self, s):
        pass

    def listen(self, cb):
        self.cb = cb

    def stop(self):
        pass

    def _step(self, dt):
        if not (self.physics and self.ctrl is not None):
            return
        c, wb = self.ctrl, 2.8                                       # centre-of-gravity bicycle, CG mid-wheelbase
        dv = max(-c.acceleration * dt, min(c.acceleration * dt, c.speed - self.v))
        y = math.radians(self.tf.rotation.yaw)
        beta = math.atan(0.5 * math.tan(c.steer))
        self.tf.location.x += self.v * math.cos(y + beta) * dt
        self.tf.location.y += self.v * math.sin(y + beta) * dt
        self.tf.rotation.yaw = math.degrees(y + self.v * math.sin(beta) / (wb / 2) * dt)
        self.v = max(0.0, self.v + dv)


class _Waypoint:
    def __init__(self, x, y, z):
        self.transform, self.lane_width = Transform(Location(x, round(y / 4.0) * 4.0, z), Rotation()), 4.0


class _Map:
    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        return _Waypoint(loc.x, loc.y, loc.z)


class World:
    def __init__(self):
        self.actors, self.frame, self.settings = [], 0, WorldSettings()
        self.debug = types.SimpleNamespace(draw_line=lambda *a, **k: None)

    def get_settings(self):
        return self.settings

    def apply_settings(self, s):
        self.settings = s

    def get_map(self):
        return _Map()

    def get_blueprint_library(self):
        return _Library()

    def set_weather(self, w):
        self.weather = w

    def try_spawn_actor(self, bp, tf, attach_to=None, attachment_type=None):
        a = Actor(self, bp, tf)
        self.actors.append(a)
        return a

    spawn_actor = try_spawn_actor

    def tick(self):
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in self.actors:
            a._step(dt)
        self.frame += 1
        return self.frame


class Client:
    def __init__(self, host="localhost", port=2000):
        self.world = World()

    def set_timeout(self, t):
        pass

    def generate_opendrive_world(self, xodr, params):
        import xml.dom.minidom
        xml.dom.minidom.parseString(xodr)                  # must be well-formed
        self.world = World()
        self.last_xodr = xodr
        return self.world

    def get_world(self):
        return self.world

    def apply_batch_sync(self, cmds, do_tick=False):
        ids = {c[1] for c in cmds}
        self.world.actors = [a for a in self.world.actors if a.id not in ids]
        return []
