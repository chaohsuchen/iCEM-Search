"""RC03-S1-opt: RC03-S1's story with the options left open, for a teacher that must read which responses exist.

A car in a lane next to the ego is rear-ended and spins; depending on the variation it comes to rest deep in the ego
lane, clips its edge, or stays in its own lane (benign). The crash happens on the right (side = +1, as RC03-S1) or
mirrored on the left (side = -1); the other adjacent lane carries traffic with a gap that may be open or closed, and a
car may follow the ego in its lane. Which of keep / brake / left / right / left_brake / right_brake (the lane change
ready to brake) recovers is decided by executing each family (options.py) - never by the parameters - and every
family's outcome is kept (the recovery profile).

What makes each option feasible or not (all varied, none observed directly):
    braking   distance to the crash d0, ego speed, how fast the spun car slides on (v_post), a follower behind (rear,
              its gap and reaction), the road's grip
    escape    the gap in the other adjacent lane (esc_gap, its position, the gap leader braking), crash side, grip
    keep      only when the spun car stays out of the ego lane (crash = 0, or a shallow intrusion)
    escape, then brake   the spun car sliding on into the escape lane, and the speed the ego shed before it brakes
              there (collections with --set intrude=1.1:1.6 ... make the spin reach that lane late; see profile_bc)

Sites (one spec each; the ego in a middle lane with a lane on both sides, permitted corridor = those three lanes):
    A   Town06 road 35, straight 5+ lanes (RC03-S1's site; sid RC03-S1-opt)        training + evaluation
    B   Town06 road 40, straight 5 lanes, other scenery                             training + evaluation
    C1  Town04 road 45, 4-lane ring in a long bend (R ~ 200-300 m)                  training + evaluation
    C2  Town04 road 45, the other carriageway (bend the other way)                  training + evaluation
    D1  Town05 road 37, 3-lane ring in a tight bend (R ~ 100 m)                    evaluation only (unseen site)
    D2  Town05 road 12, another bend of the ring (R ~ 100 m)                        training + evaluation
    E1  Town13 road 1411, straight 3-lane motorway (CARLA's large evaluation town)   evaluation only (unseen town)
    E2  Town13 road 1409, the same motorway in a gentle bend (~19 deg over 320 m)    evaluation only (unseen town)
    C1b C2b D2b  C1 / C2 / D2 moved on into their bends (s0): C1, C2 and D2 meet the crash on a straight stretch, only
                D1 is in its bend over the whole recovery (2026-11-09)                  training + evaluation
D2 joined training (2026-11-08) for coverage of tight bends (Town04's are R ~ 180 m); D1 stays held out, a new site
on a ring road of a town seen in training.
Town05's bends are tight, so traffic there drives at 0.75 of the motorway speeds (the site's speed level; 18-23 m/s
dry, inside the training speeds of the wet / heavy conditions). Road condition `wx` (envelope.ROAD): dry / wet / heavy rain. It sets the rain the ego sees, the ego's tyre friction
(and so its qualified envelope), the speed level of all traffic (drivers slow down in rain) and how hard the other
cars can brake or slide (grip). Variations of the first collection (no `wx`) are dry, site A.
"""
import numpy as np

from envelope import ROAD
from laws import HALF_L, crash_path, law_idm
from options import MODES_LB, OBS_NAMES, lane_slot_features
from scenarios import CAR, RC03S1, Spec, car, platoon

VICTIMS = ["vehicle.dodge.charger_2020", "vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.ford.mustang",
           "vehicle.mercedes.sprinter"]
GAP_CARS = ["vehicle.mini.cooper_s_2021", "vehicle.lincoln.mkz_2020", "vehicle.toyota.prius", "vehicle.ford.mustang"]
SITES = {   # town, LaneFrame site, lanes left of the frame's lane 0 (background traffic there), split, speed level
    "A": ("Town06", RC03S1.site, 2, "train", 1.0),
    "B": ("Town06", dict(start_xy=(128.5, 248.1), ref_lane=-6, ref_idx=1, s0=40.0), 2, "train", 1.0),
    "C1": ("Town04", dict(start_xy=(-363.3, 423.9), ref_lane=-2, ref_idx=1, s0=40.0, curve=True), 0, "train", 1.0),
    "C2": ("Town04", dict(start_xy=(-221.7, 411.2), ref_lane=4, ref_idx=1, s0=40.0, curve=True), 0, "train", 1.0),
    "D1": ("Town05", dict(start_xy=(-143.1, -204.1), ref_lane=-2, ref_idx=1, s0=0.0, curve=True), 0, "heldout", 0.75),
    "D2": ("Town05", dict(start_xy=(207.5, 13.8), ref_lane=-2, ref_idx=1, s0=0.0, curve=True), 0, "train", 0.75),
    # the same roads moved on into their bends: C1 / C2 / D2 above meet the crash on a straight stretch (measured
    # curvature over x 0-150), these keep the whole recovery inside the bend
    "C1b": ("Town04", dict(start_xy=(-363.3, 423.9), ref_lane=-2, ref_idx=1, s0=180.0, curve=True), 0, "train", 1.0),
    "C2b": ("Town04", dict(start_xy=(-221.7, 411.2), ref_lane=4, ref_idx=1, s0=120.0, curve=True), 0, "train", 1.0),
    "D2b": ("Town05", dict(start_xy=(207.5, 13.8), ref_lane=-2, ref_idx=1, s0=110.0, curve=True), 0, "train", 0.75),
    "E1": ("Town13", dict(start_xy=(-3519.7, -2854.8), ref_lane=-3, ref_idx=1, s0=0.0, curve=True), 0, "heldout", 1.0),
    "E2": ("Town13", dict(start_xy=(-5111.6, -2125.2), ref_lane=-3, ref_idx=1, s0=0.0, curve=True), 0, "heldout", 1.0),
}
WEATHER = {
    "dry": RC03S1.weather,
    "wet": dict(cloudiness=80.0, precipitation=60.0, precipitation_deposits=60.0, wetness=80.0, sun_altitude_angle=40.0,
                sun_azimuth_angle=150.0, fog_density=5.0),
    "heavy": dict(cloudiness=100.0, precipitation=100.0, precipitation_deposits=90.0, wetness=100.0,
                  sun_altitude_angle=30.0, sun_azimuth_angle=150.0, fog_density=15.0),
}
SPEEDS = ("v", "v_esc", "v_post")                                  # scaled by the road condition's speed level


class RC03S1Opt(Spec):
    sid, town, site = "RC03-S1-opt", "Town06", RC03S1.site
    site_key, n_left, site_speed = "A", 2, 1.0
    scene_name = "spun_car"                # R3 (PIPELINE_HANDOFF): the scene, apart from the RC number
    # R2: at the trigger (the intrusion seen) how deep the car ends up, how hard the gap leader brakes and how late
    # the followers react are not observable yet
    hidden = ("intrude", "psi_rest", "v_post", "lead_brake", "follow_react", "rear_react")
    envelope, modes = True, MODES_LB       # + lane change ready to brake (2026-11-08)
    governor = True                        # request scaled down near the contract (env._govern, 2026-11-09)
    T = 8.0
    nominal = dict(side=1, crash=1, v=27.0, d0=25.0, intrude=0.9, psi_rest=0.6, T_slide=1.5, v_post=22.0,
                   vic_bp="vehicle.dodge.charger_2020", v_esc=24.0, esc_gap=40.0, gap_center=-2.5, lead_brake=3.0,
                   follow_T=1.5, follow_react=0.8, lead_bp="vehicle.mini.cooper_s_2021",
                   fol_bp="vehicle.lincoln.mkz_2020", rear=0, rear_gap=25.0, rear_react=0.9, wx="dry")
    ranges = dict(side=[1, -1], crash=[1, 1, 1, 1, 1, 0], v=(24.0, 30.0), d0=(20.0, 65.0), intrude=(0.45, 1.1),
                  psi_rest=(0.2, 1.2), T_slide=(1.0, 2.2), v_post=(16.0, 26.0), vic_bp=VICTIMS,
                  v_esc=(21.0, 27.0), esc_gap=(10.0, 50.0), gap_center=(-10.0, 5.0), lead_brake=(0.0, 5.0),
                  follow_T=(0.8, 1.6), follow_react=(0.5, 1.1), lead_bp=GAP_CARS, fol_bp=GAP_CARS,
                  rear=[0, 1], rear_gap=(10.0, 60.0), rear_react=(0.4, 1.2), wx=["dry", "wet", "heavy"])
    overrides = {}             # per collection: a range replaced (e.g. wx=["wet", "heavy"], rear=[1])
    obs_names = OBS_NAMES
    description = ("A car next to the ego is rear-ended and spins (right or left side); it ends deep in the ego lane, "
                   "clipping it, or in its own lane. Keep, brake, or escape into the other adjacent lane's traffic: the "
                   "recovery profile over these families is measured for every variation, on several roads and in "
                   "dry / wet / heavy-rain conditions.")

    def sample(self, rng):
        """Ranges (with this collection's overrides); the speeds are then scaled to the road condition."""
        p = {}
        for k, r in {**self.ranges, **self.overrides}.items():
            p[k] = r[int(rng.integers(len(r)))] if isinstance(r, list) else float(rng.uniform(*r))
        for k in SPEEDS:
            p[k] *= ROAD[p["wx"]]["speed"] * self.site_speed
        return p

    def road(self, p):
        return ROAD[p.get("wx", "dry")]

    def road_mu(self, p):
        return self.road(p)["mu"]

    def weather_for(self, p):
        return WEATHER[p.get("wx", "dry")]

    def ego_physics(self, pc, p):
        mu = self.road_mu(p)
        if mu != 1.0:
            wheels = pc.wheels
            for w in wheels:
                w.tire_friction *= mu
            pc.wheels = wheels
        return pc

    def build(self, p, W):
        pT, s = self.prefix_T, p["side"]
        rd = self.road(p)
        sp, grip = rd["speed"] * self.site_speed, rd["grip"]
        y_e, y_c, y_s = W, W + s * W, W - s * W                      # ego, crash and escape lane centres
        Lv = CAR[p["vic_bp"]][0]
        x_v = HALF_L + p["d0"] + Lv / 2
        a_slide = 8.0 * grip
        if p["crash"]:
            vic_path = crash_path(x_v, y_c, v_pre=20.0 * sp, v_post=p["v_post"], a_slide=a_slide,
                                  y_rest=y_c - s * p["intrude"] * W, psi_rest=-s * p["psi_rest"], T_slide=p["T_slide"])
        else:                                                         # benign: it brakes and stays in its lane
            vic_path = crash_path(x_v, y_c, v_pre=20.0 * sp, v_post=p["v_post"], a_slide=a_slide,
                                  y_rest=y_c - s * 0.12 * W, psi_rest=-s * 0.06, T_slide=p["T_slide"])
        vic = car("victim", p["vic_bp"], 0, y_c, 20.0 * sp, kind="scripted", path=vic_path)
        x_s = x_v - Lv / 2 - CAR["vehicle.nissan.patrol_2021"][0] / 2
        stk = car("striker", "vehicle.nissan.patrol_2021", 0, y_c, 26.0 * sp, kind="scripted",
                  path=crash_path(x_s, y_c, v_pre=26.0 * sp, v_post=18.0 * sp, a_slide=a_slide, y_rest=y_c + s * 0.4,
                                  psi_rest=s * 0.15, T_slide=1.0))
        vl = p["v_esc"]
        lead_dx = p["gap_center"] + p["esc_gap"] / 2 + CAR[p["lead_bp"]][0] / 2
        follow_dx = p["gap_center"] - p["esc_gap"] / 2 - CAR[p["fol_bp"]][0] / 2
        lb = p["lead_brake"] * grip
        lead = car("gap_lead", p["lead_bp"], lead_dx - vl * pT, y_s, vl,
                   law=law_idm(vl, T=1.5, react=0.8, b_max=6.0 * grip,
                               rubberneck=(1.0, lb, 12.0 * sp) if lb > 0.2 else None))
        fol = car("gap_follow", p["fol_bp"], follow_dx - vl * pT, y_s, vl,
                  law=law_idm(vl, T=p["follow_T"], react=p["follow_react"], b_max=6.0 * grip))
        bodies = [vic, stk, lead, fol]
        if p["rear"]:                                                 # a follower settled at its own time gap
            Lr = CAR["vehicle.audi.a2"][0]
            bodies.append(car("rear", "vehicle.audi.a2", -HALF_L - p["rear_gap"] - Lr / 2 - p["v"] * pT, y_e, p["v"],
                              law=law_idm(p["v"], T=max((p["rear_gap"] - 2.0) / p["v"], 0.15), react=p["rear_react"],
                                            b_max=7.0 * grip)))
        bg = []                                                       # unobserved traffic in lanes left of the corridor
        if self.n_left >= 1:
            bg += platoon("l1_", -W, vl + 2.0, [-60.0, -20.0, 25.0, 70.0])
        if self.n_left >= 2:
            bg += platoon("l2_", -2 * W, vl + 5.0, [-40.0, 10.0, 60.0])
        return dict(ego=dict(x=-p["v"] * pT, y=y_e, v=p["v"]), bodies=bodies + bg, home_y=y_e,
                    bounds=lambda x: (-W / 2, 2.5 * W), v_ref=p["v"], props=[])

    def features(self, env):
        return lane_slot_features(env)

    def card(self):
        return dict(super().card(), modes=list(self.modes), site_key=self.site_key, road=ROAD, weather=WEATHER)


def site_spec(key):
    town, site, n_left, split, speed = SITES[key]
    sp = RC03S1Opt()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.sid = RC03S1Opt.sid if key == "A" else f"{RC03S1Opt.sid}@{key}"
    return sp


SPECS03O = {sp.sid: sp for sp in map(site_spec, SITES)}
