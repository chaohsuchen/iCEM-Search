"""RC19 scenarios: a fixed safety-relevant scene whose required ego decision never changes (one static decision per
scenario: brake and follow in lane / stop for the pedestrian and go on / drop back and merge behind).

The notebook's benign cases are categorical variations, on top of narrow numeric ones (speeds, gaps, timing):
    A anchor (plain)  B appearance (weather, colours)  C distant traffic / people  D parked / stopped objects beside
    the road  S the notebook's semantic control (an object that matters; the ego still makes the same decision).
Benign objects are named bg_* and never enter the observation or the expert's view, so the policy is invariant to
A-D by construction; the semantic object (sem_*) is a real obstacle and enters through the corridor features.

    RC19-S1  Town04 road 49 (4-lane motorway, 3.5 m hard shoulder): the car ahead brakes hard, the lane beside is
             occupied -> brake in lane.  S: a broken-down car stands 1.4 m into the ego lane 200 m ahead (the lead
             and the ego stop behind it; at the notebook's 70 m the hard-braking lead runs into it).
    RC19-S2  Town05 road 51 (one-way, 2 lanes + parking lane): a car cuts in and slows to look for parking -> brake
             and follow.  S: a van double-parked 1.2 m into the ego lane.
    RC19-S3  Town10HD road 21 (2 lanes + shoulder, crosswalk at x ~ -91): a pedestrian crosses -> stop, wait, go on.
             S: a van parked just before the crossing hides the pedestrian until it steps out. The crosswalk is at
             the mouth of a T-junction, so the episode ends once the ego's rear is 2 m past it.
    RC19-S4  = RC15-L-match-behind (Town06 ramp): the adjacent driver matches -> drop back and merge behind.
             S: the adjacent driver is a truck.
"""
import math

import numpy as np

import carla

from laws import HALF_L, Body, const_path, law_event, law_hold, obb, walk_across
from rc15 import MergeSpec
from scenarios import CAR, Spec, car, rel
from shapely.geometry import LineString, Polygon as ShPolygon

CASES = ["A", "B", "C", "D", "S"]
WEATHER_FIXED = dict(precipitation=0.0, precipitation_deposits=0.0, wetness=0.0, fog_density=0.0)
WALKER = "walker.pedestrian.0001"


def walker(name, x0, y, v, bp=WALKER):
    """Benign pedestrian walking along the sidewalk (v > 0: with the traffic)."""
    return Body(name, x0, y, length=0.6, width=0.6, kind="scripted", bp=bp,
                path=const_path(x0, y, abs(v), psi=0.0 if v >= 0 else math.pi))


def parked(name, bp, x, y):
    L, Wd = CAR[bp]
    return Body(name, x, y, length=L, width=Wd, kind="static", bp=bp)


def corridor_threat(env, y_c, t_pred=2.0, margin=0.3, cap=60.0):
    """Nearest delivered object ahead that is in the ego corridor (lane +- margin) or, at its lateral speed, enters it
    within t_pred s (the notebook CorridorDriver's notion of a threat): (bumper gap, its speed along x, its accel).
    No threat -> (cap, ego speed, 0)."""
    e, half = env.ego, env.W / 2 + margin
    best = (cap, e.speed, 0.0)
    for n, b in env.bodies.items():
        if n.startswith("bg_") or b.kind == "static" and b.bp is None or not env.spec.deliver(env, n):
            continue
        c, s = abs(math.cos(b.heading)), abs(math.sin(b.heading))
        ex, ey = b.LENGTH * c + b.WIDTH * s, b.LENGTH * s + b.WIDTH * c
        if b.position[0] + ex / 2 <= e.position[0] + HALF_L:
            continue
        lo, hi = sorted((b.position[1], b.position[1] + b.velocity[1] * t_pred))
        if hi + ey / 2 > y_c - half and lo - ey / 2 < y_c + half:
            gap = b.position[0] - ex / 2 - (e.position[0] + HALF_L)
            if gap < best[0]:
                best = (gap, float(b.velocity[0]), float(b.acc))
    return best


class Benign:
    """Mixin: case-dependent weather and colours (B), and the follow goal (not stopped behind a moving leader)."""
    W_A = dict(cloudiness=5.0, sun_altitude_angle=70.0, sun_azimuth_angle=0.0, **WEATHER_FIXED)
    W_B = dict(cloudiness=70.0, sun_altitude_angle=45.0, sun_azimuth_angle=120.0, **WEATHER_FIXED)
    COLOURS = {}            # case B: actor (or 'ego') -> "r,g,b"

    def weather_for(self, p):
        return self.W_B if p.get("case") == "B" else self.W_A

    def ego_color(self, p):
        return self.COLOURS.get("ego") if p.get("case") == "B" else None

    def paint(self, p, bodies):
        if p.get("case") == "B":
            for b in bodies:
                b.color = self.COLOURS.get(b.name, b.color)
        return bodies

    def goal_follow(self, env, y_c):
        """Keeps up with traffic: at the end the ego is not much slower than the object it follows (always-stop fails)."""
        gap, v_l, _ = corridor_threat(env, y_c)
        need = 0.5 * (v_l if gap < 60.0 else env.params["v"]) - 0.5
        return env.ego.speed >= need, dict(v_end=round(env.ego.speed, 2), v_needed=round(need, 2))


class RC19S1(Benign, Spec):
    """Motorway: the car ahead brakes hard for a jam; the lane beside is occupied -> brake in lane.
    Town04 road 49 northbound; scenario lanes (4, 5, 6) = y 0, W, 2W; ego in 6 (right), 3.5 m shoulder beyond, lane 3
    (y -W) is the far lane."""
    sid, town = "RC19-S1", "Town04"
    site = dict(start_xy=(-9.9, -202.8), ref_lane=4, ref_idx=0, s0=30.0)     # the road bends just before y = -203
    T = 14.0
    nominal = dict(v=30.0, lead_gap=30.0, lead_b=5.0, lead_v_end=12.0, t_ev=1.0, left_dx=-3.0, case="A")
    ranges = dict(v=(28.0, 32.0), lead_gap=(27.0, 33.0), lead_b=(4.0, 6.0), lead_v_end=(10.0, 14.0), t_ev=(0.8, 1.2),
                  left_dx=(-5.0, -1.0), case=CASES)
    COLOURS = {"ego": "235,235,235", "lead": "150,20,30", "left_1": "30,30,30", "left_2": "200,170,40"}
    obs_names = Spec.EGO_NAMES + ["thr_gap", "thr_v", "thr_a", "left_dx", "left_dvx"]
    description = "Town04 4-lane motorway: the car ahead brakes hard, the lane beside is occupied: brake in lane"

    def build(self, p, W):
        pT, v, y_e = self.prefix_T, p["v"], 2 * W
        lead_dx = HALF_L + p["lead_gap"] + CAR["vehicle.toyota.prius"][0] / 2
        b = [car("lead", "vehicle.toyota.prius", lead_dx - v * pT, y_e, v,
                 law=law_event(v, t_ev=p["t_ev"], b=p["lead_b"], v_end=p["lead_v_end"])),
             car("left_1", "vehicle.lincoln.mkz_2020", p["left_dx"] - v * pT, W, v, law=law_event(v)),
             car("left_2", "vehicle.audi.a2", 40.0 - v * pT, W, v,
                 law=law_event(v, t_ev=p["t_ev"] + 0.6, b=4.0, v_end=p["lead_v_end"] + 2.0))]
        props = []
        y_sh = 2.5 * W                                       # right edge of the ego lane = start of the 3.5 m shoulder
        if p["case"] == "C":                                 # far traffic: a car 170 m ahead in lane -2, two in lane -1
            b += [car("bg_c_far", "vehicle.ford.mustang", 170.0 - v * pT, 0.0, v, law=law_hold(v)),
                  car("bg_c_l1a", "vehicle.dodge.charger_2020", 120.0 - v * pT, -W, v + 2.0, law=law_hold(v + 2.0)),
                  car("bg_c_l1b", "vehicle.mini.cooper_s_2021", -110.0 - v * pT, -W, v + 1.0, law=law_hold(v + 1.0))]
        elif p["case"] == "D":                               # on the hard shoulder, 1.6 m clear of the lane
            b += [parked("bg_d_car", "vehicle.lincoln.mkz_2020", 70.0, y_sh + 1.6 + 1.05),
                  parked("bg_d_van", "vehicle.mercedes.sprinter", 160.0, y_sh + 1.5 + 1.0)]
            props = [("static.prop.warningaccident", 52.0, y_sh + 1.8)]
        elif p["case"] == "S":   # SEMANTIC: a broken-down car stands 1.4 m into the lane, far enough ahead that the lead
            b += [parked("sem_car", "vehicle.lincoln.mkz_2020", 200.0, y_sh + 1.05 - 1.4)]   # and the ego stop behind it
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=self.paint(p, b), home_y=y_e, bounds=lambda x: (-W / 2, 2.5 * W),
                    v_ref=v, props=props)

    def features(self, env):
        f = self.ego_features(env) + list(corridor_threat(env, env.scene["home_y"]))
        dx, _, dvx, _ = rel(env, "left_1")
        return np.array(f + [dx, dvx], dtype=np.float32)

    def goal(self, env):
        return self.goal_follow(env, env.scene["home_y"])


class RC19S2(Benign, Spec):
    """City street: a car from the left lane cuts in close ahead and slows to look for parking -> brake and follow.
    Town05 road 51 southbound (one-way, lanes -1, -2 = y 0, W; a 3 m parking lane and the sidewalk on the right)."""
    sid, town = "RC19-S2", "Town05"
    site = dict(start_xy=(-123.5, 136.6), ref_lane=-1, ref_idx=0, s0=15.0)
    T = 12.0
    nominal = dict(v=14.0, cut_dx=16.0, cut_v=12.0, t_ev=0.3, T_lc=2.5, cut_b=1.5, v_end=5.0, follow_dx=-14.0, case="A")
    ranges = dict(v=(13.0, 15.0), cut_dx=(14.0, 18.0), cut_v=(11.0, 13.0), t_ev=(0.2, 0.5), T_lc=(2.2, 2.8),
                  cut_b=(1.2, 1.8), v_end=(4.0, 6.0), follow_dx=(-16.0, -12.0), case=CASES)
    W_A = dict(cloudiness=20.0, sun_altitude_angle=60.0, sun_azimuth_angle=30.0, **WEATHER_FIXED)
    W_B = dict(cloudiness=35.0, sun_altitude_angle=50.0, sun_azimuth_angle=210.0, **WEATHER_FIXED)
    COLOURS = {"ego": "40,60,160", "cutter": "230,230,230", "left_follow": "190,90,20"}
    obs_names = Spec.EGO_NAMES + ["thr_gap", "thr_v", "thr_a", "cut_dx", "cut_dy", "cut_dvx", "cut_vy"]
    description = "Town05 one-way city street: a car cuts in and slows to look for parking: brake and follow"

    def build(self, p, W):
        pT, v, y_e = self.prefix_T, p["v"], W
        b = [car("cutter", "vehicle.mini.cooper_s_2021", p["cut_dx"] - p["cut_v"] * pT, 0.0, p["cut_v"],
                 law=law_event(p["cut_v"], t_ev=p["t_ev"], b=p["cut_b"], v_end=p["v_end"], y_to=y_e, T_lc=p["T_lc"])),
             car("left_follow", "vehicle.lincoln.mkz_2020", p["follow_dx"] - v * pT, 0.0, v, law=law_event(v))]
        y_park = 1.5 * W + 3.0 - 1.0                         # outer side of the parking lane
        y_walk = 1.5 * W + 3.0 + 0.63 + 1.5                  # on the sidewalk
        if p["case"] == "C":
            b += [walker("bg_c_w1", 25.0, y_walk, 1.3), walker("bg_c_w2", 75.0, y_walk + 1.0, -1.2),
                  walker("bg_c_w3", 40.0, y_walk + 0.5, 1.4, bp="walker.pedestrian.0015"),
                  walker("bg_c_w4", 95.0, y_walk + 1.5, -1.3, bp="walker.pedestrian.0015")]
        elif p["case"] == "D":
            b += [parked(f"bg_d_car{i}", bp, x, y_park) for i, (x, bp) in enumerate(
                ((30.0, "vehicle.toyota.prius"), (36.5, "vehicle.audi.a2"), (43.0, "vehicle.ford.mustang"),
                 (100.0, "vehicle.mini.cooper_s_2021"), (106.5, "vehicle.toyota.prius")))]
            b += [parked("bg_d_van", "vehicle.mercedes.sprinter", 70.0, y_park)]
        elif p["case"] == "S":                               # SEMANTIC: a van double-parked 1.2 m into the ego lane
            b += [parked("sem_van", "vehicle.mercedes.sprinter", 60.0, 1.5 * W + 1.0 - 1.2)]
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=self.paint(p, b), home_y=y_e, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[])

    def features(self, env):
        f = self.ego_features(env) + list(corridor_threat(env, env.scene["home_y"]))
        dx, dy, dvx, vy = rel(env, "cutter")
        return np.array(f + [dx, dy, dvx, vy], dtype=np.float32)

    def goal(self, env):
        return self.goal_follow(env, env.scene["home_y"])


class RC19S3(Benign, Spec):
    """Zebra crossing: a pedestrian steps onto the crosswalk ahead and walks across to the median -> stop, wait until
    it has left the ego lane, drive on. Town10HD road 21 westbound (lanes 4, 5 = y 0, W; 2.5 m shoulder and the
    sidewalk on the right, a median on the left), real crosswalk at CARLA x ~ -91.1."""
    sid, town = "RC19-S3", "Town10HD"
    site = dict(start_xy=(77.4, 16.9), ref_lane=4, ref_idx=0, s0=110.0)
    ZEBRA_WORLD = (-91.1, 12.0)
    T = 14.0
    nominal = dict(v=11.0, d=45.0, ped_t=0.5, ped_v=1.4, left_dx=14.0, follow_dx=-25.0, case="A")
    ranges = dict(v=(10.0, 12.0), d=(42.0, 48.0), ped_t=(0.3, 0.8), ped_v=(1.2, 1.6), left_dx=(10.0, 18.0),
                  follow_dx=(-28.0, -22.0), case=CASES)
    W_A = dict(cloudiness=10.0, sun_altitude_angle=55.0, sun_azimuth_angle=90.0, **WEATHER_FIXED)
    W_B = dict(cloudiness=40.0, sun_altitude_angle=55.0, sun_azimuth_angle=250.0, **WEATHER_FIXED)
    COLOURS = {"ego": "200,30,30", "left_car": "240,240,240", "follow": "20,20,20"}
    obs_names = Spec.EGO_NAMES + ["thr_gap", "thr_v", "thr_a", "ped_seen", "ped_dx", "ped_dy", "ped_vy"]
    description = "Town10HD avenue: a pedestrian crosses on the crosswalk ahead: stop, wait, drive on"
    zebra_x = 45.0

    def on_frame(self, frame):
        self.zebra_x = frame.from_world(carla.Location(x=self.ZEBRA_WORLD[0], y=self.ZEBRA_WORLD[1]), 180.0)[0]

    def build(self, p, W):
        pT, v, y_e, xz = self.prefix_T, p["v"], W, self.zebra_x
        x_e = xz - p["d"]                                    # ego centre at t0
        y_walk0 = 1.5 * W + 2.5 + 0.63 + 0.5                 # kerb edge of the right sidewalk
        y_med = -W / 2 - 0.5 - 1.75                          # median refuge
        ped_bp = "walker.pedestrian.0015" if p["case"] == "B" else WALKER
        b = [Body("ped", xz, y_walk0, length=0.6, width=0.6, kind="scripted", bp=ped_bp,
                  path=walk_across(xz, y_walk0, y_med, p["ped_t"], p["ped_v"])),
             car("left_car", "vehicle.toyota.prius", x_e + p["left_dx"] - v * pT, 0.0, v, law=law_hold(v)),
             car("follow", "vehicle.lincoln.mkz_2020", x_e + p["follow_dx"] - v * pT, y_e, v, law=law_event(v))]
        y_sh = 1.5 * W + 2.5 + 0.63 - 1.05                   # parked on the shoulder, outer side
        y_side = 1.5 * W + 2.5 + 0.63 + 3.0                  # sidewalk centre
        if p["case"] == "C":                                  # walkers along the sidewalk, moving away from the crossing
            b += [walker("bg_c_w1", xz - 20.0, y_side, -1.3), walker("bg_c_w2", xz + 12.0, y_side + 1.0, 1.4),
                  walker("bg_c_w3", xz - 45.0, y_side - 1.0, -1.2, bp="walker.pedestrian.0015"),
                  walker("bg_c_w4", xz + 30.0, y_side + 0.5, 1.3, bp="walker.pedestrian.0015")]
        elif p["case"] == "D":                                # parked AFTER the crossing (sight line stays clear)
            b += [parked("bg_d_car0", "vehicle.audi.a2", xz + 12.0, y_sh), parked("bg_d_car1", "vehicle.toyota.prius", xz + 18.5, y_sh),
                  parked("bg_d_van", "vehicle.mercedes.sprinter", xz + 28.0, y_sh), parked("bg_d_car2", "vehicle.ford.mustang", xz + 40.0, y_sh)]
        elif p["case"] == "S":                                # SEMANTIC: the same van just BEFORE the crossing hides the ped
            b += [parked("sem_van", "vehicle.mercedes.sprinter", xz - 6.5, 1.5 * W + 0.3 + 1.0)]
        self.paint(p, b)
        return dict(ego=dict(x=x_e - v * pT, y=y_e, v=v), bodies=b, home_y=y_e, bounds=lambda x: (-W / 2, 1.5 * W),
                    v_ref=v, props=[], zebra_x=xz)

    def deliver(self, env, name):
        """The pedestrian is seen only with a clear sight line from the driver's eye to its centre or a corner (any
        body wider than 1 m blocks, benign ones included: the D case keeps them after the crossing)."""
        if name != "ped":
            return True
        e, o = env.ego, env.bodies["ped"]
        eye = (e.position[0] + HALF_L - 0.5, e.position[1])
        polys = [ShPolygon(obb(*b.position, b.heading, b.LENGTH, b.WIDTH)) for n, b in env.bodies.items()
                 if n != "ped" and b.WIDTH > 1.0 and b.LENGTH < 20.0
                 and min(eye[0], o.position[0]) - 10.0 < b.position[0] < max(eye[0], o.position[0]) + 10.0]
        pts = [tuple(o.position)] + obb(*o.position, o.heading, o.LENGTH, o.WIDTH)
        return any(not any(LineString([eye, q]).intersects(P) for P in polys) for q in pts)

    def features(self, env):
        f = self.ego_features(env) + list(corridor_threat(env, env.scene["home_y"]))
        if self.deliver(env, "ped"):
            dx, dy, _, vy = rel(env, "ped")
            f += [1.0, dx, dy, vy]
        else:
            f += [0.0, 60.0, 10.0, 0.0]
        return np.array(f, dtype=np.float32)

    def done(self, env):
        """Ends once the ego has driven on past the crossing (the road ends in a T-junction right after it)."""
        return env.t >= 2.0 and env.ego.position[0] - HALF_L > env.scene["zebra_x"] + 2.0

    def goal(self, env):
        """Drove on: the ego's rear is past the crosswalk at the end."""
        x_rear = env.ego.position[0] - HALF_L
        return x_rear > env.scene["zebra_x"] + 2.0, dict(rear_past_crossing_m=round(x_rear - env.scene["zebra_x"], 2))


class RC19S4(Benign, MergeSpec):
    """RC15-L-match-behind with the benign cases. C: far traffic in the motorway lanes; D: road-works vans parked in
    the coned-off lane well after the taper; S (semantic): the adjacent driver is a truck."""
    W_A = dict(cloudiness=15.0, sun_altitude_angle=60.0, sun_azimuth_angle=180.0, **WEATHER_FIXED)
    W_B = dict(cloudiness=75.0, sun_altitude_angle=40.0, sun_azimuth_angle=180.0, **WEATHER_FIXED)
    COLOURS = {"ego": "240,240,240", "adj": "20,20,120", "tgt_lead": "150,150,150", "tgt_follow": "120,20,20"}

    def __init__(self):
        super().__init__("L", "match-behind")
        self.sid = "RC19-S4"
        self.ranges = dict(self.ranges, case=CASES)
        self.nominal = dict(self.nominal, case="A")
        self.description = "RC15-L-match-behind (Town06 ramp) with benign cases A-D and the semantic control S"

    def adj_bp(self, p):
        return "vehicle.carlamotors.european_hgv" if p.get("case") == "S" else "vehicle.ford.mustang"

    def build(self, p, W):
        sc = super().build(p, W)
        b, pT = sc["bodies"], self.prefix_T
        if p["case"] == "S":                                 # the truck's real size
            adj = next(x for x in b if x.name == "adj")
            adj.LENGTH, adj.WIDTH = CAR["vehicle.carlamotors.european_hgv"]
        va = p["v"] + p["dv_adj"]
        if p["case"] == "C":
            b += [car("bg_c_far", "vehicle.toyota.prius", 140.0 - va * pT, sc["tgt_y"], va, law=law_hold(va)),
                  car("bg_c_truck", "vehicle.carlamotors.european_hgv", -150.0 - va * pT, -W, va - 2.0, law=law_hold(va - 2.0))]
        elif p["case"] == "D":                               # in the closed lane, behind the cone line, outer side
            y_out = sc["home_y"] + W / 2 - 1.0
            b += [parked("bg_d_van1", "vehicle.mercedes.sprinter", p["end_x"] + 70.0, y_out),
                  parked("bg_d_van2", "vehicle.ford.ambulance", p["end_x"] + 90.0, y_out)]
        self.paint(p, b)
        return sc


SPECS19 = {s.sid: s for s in (RC19S1(), RC19S2(), RC19S3(), RC19S4())}
