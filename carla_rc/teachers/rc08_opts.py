"""RC08 with options (RC08-S1-opt): protected waiting behind a garbage truck despite missing the route deadline.

A garbage truck has stopped in the ego's lane to load on a two-way town street (Town01, one 4 m lane each way); it hides
part of the oncoming lane. Oncoming traffic comes in a stream of five cars with gaps (time headways onc_h1..onc_h4), a
car follows the ego (time headway fol_T; an attentive driver: IDM with 3 m/s^2
comfortable braking, so a timely protected stop is followed and a late hard one is not). The truck leaves after t_go s (hidden: 5-60 s, mostly beyond the horizon).

    protected hold   stop behind the truck far enough back for the follower to stop too: safe when started in time
    overtake         through the oncoming lane at once: safe only on a large enough gap
    wait, overtake   hold, edge left to see past the truck, go once the visible gap is large enough (an oncoming car
                     still hidden by the truck is not seen; peeking further sees more)
    brake late       the follower hits the ego

Safety (r, the label) and route completion are separate: the goal is always met (no deadline in the label), and
`route_done` (the ego is past the truck within ROUTE_T = 20 s) is stored next to it (goal details in the metrics).

Families: keep / brake / stop_end (protected hold) / overtake / wait_overtake (options.py). Trigger "braking": the truck
seen standing in the ego lane. Hidden at the trigger: when the truck leaves, how late and how hard the follower brakes,
and the oncoming cars not yet in view (line of sight past the truck, sight.py).

Sites: Town01 straights (roads 20, 15, 12: training); Town02 road 12 held out (unseen town).
"""
from laws import HALF_L, Body, const_path, law_idm
from rc03_opts import OptBase, speed_path
from scenarios import CAR, car

TRUCK = "vehicle.carlamotors.european_hgv"
ONC = ["vehicle.toyota.prius", "vehicle.audi.a2", "vehicle.ford.mustang", "vehicle.mini.cooper_s_2021",
       "vehicle.lincoln.mkz_2020"]
ROUTE_T = 20.0
T_MAX = 30.0
SITES08 = {
    "A": ("Town01", dict(start_xy=(2.0, 318.0), ref_lane=-1, ref_idx=0, s0=50.0), 0, "train", 1.0),
    "B": ("Town01", dict(start_xy=(-2.0, 10.0), ref_lane=-1, ref_idx=0, s0=50.0), 0, "train", 1.0),
    "C": ("Town01", dict(start_xy=(101.4, 199.1), ref_lane=1, ref_idx=0, s0=50.0), 0, "train", 1.0),
    "H": ("Town02", dict(start_xy=(5.5, 109.5), ref_lane=1, ref_idx=0, s0=45.0), 0, "heldout", 1.0),
}


class RC08Opt(OptBase):
    sid = "RC08-S1-opt"
    scene_name = "garbage_truck"
    T, prefix_T = 22.0, 3.0
    goal_text = "no deadline in the label; route_done stored apart"
    trigger = "braking"
    lane_w = 4.0                                       # Town01 / Town02 lanes (the surrogate's W)
    end_at_road_end = True                             # CARLA: the Town01 / 02 straights end before 22 s at speed
    snap_stop = True
    modes = ("keep", "brake", "stop_end", "overtake", "wait_overtake")
    SPEEDS = ("v", "onc_v")
    nominal = dict(v=13.0, d_truck=45.0, truck_dy=0.3, t_go=40.0, onc_v=13.0, onc_x0=40.0, onc_h1=4.0, onc_h2=6.0,
                   onc_h3=3.0, onc_h4=8.0, fol=1, fol_T=1.2, fol_react=0.6, fol_b=6.0, wx="dry")
    ranges = dict(v=(10.0, 15.0), d_truck=(30.0, 70.0), truck_dy=(0.0, 0.6), t_go=(5.0, 60.0), onc_v=(10.0, 15.0),
                  onc_x0=(-10.0, 150.0), onc_h1=(2.0, 12.0), onc_h2=(2.0, 12.0), onc_h3=(2.0, 12.0),
                  onc_h4=(2.0, 12.0), fol=[1, 1, 1, 0], fol_T=(0.8, 2.0), fol_react=(0.3, 0.9), fol_b=(4.5, 7.5),
                  wx=["dry", "dry", "wet"])
    hidden = ("t_go", "fol_react", "fol_b")
    description = ("A garbage truck stopped in the ego lane on a two-way town street, oncoming traffic in a stream "
                   "with gaps, a car behind. Stop protected behind it and wait, or overtake through the oncoming lane "
                   "now or once a large enough gap is seen past the truck; the route deadline may be missed.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e, y_o = 0.0, -W
        Lt, Wt = CAR[TRUCK]
        x_tr = HALF_L + p["d_truck"] + Lt / 2
        bodies = [Body("truck", x_tr, y_e + p["truck_dy"], length=Lt, width=Wt, kind="scripted", bp=TRUCK,
                       path=speed_path(x_tr, y_e + p["truck_dy"], 0.0, 0.0, [(p["t_go"], 0.0, 0.0), (99.0, 1.0, 8.0)],
                                       t0=-pT - 1.0, t1=T_MAX))]
        vo = p["onc_v"]                                          # the oncoming stream (constant speed, scripted)
        x = x_tr + Lt / 2 + p["onc_x0"]
        for i in range(5):
            bp = ONC[i]
            bodies.append(car(f"onc{i}", bp, 0, y_o, vo, kind="scripted", path=const_path(x, y_o, vo, psi=3.14159265)))
            if i < 4:
                x += vo * p[f"onc_h{i + 1}"] + CAR[bp][0] / 2 + CAR[ONC[i + 1]][0] / 2
        if p["fol"]:
            Lf = CAR["vehicle.lincoln.mkz_2020"][0]
            gap = max(v * p["fol_T"], 4.0)
            bodies.append(car("follower", "vehicle.lincoln.mkz_2020", -HALF_L - gap - Lf / 2 - v * pT, y_e, v,
                              law=law_idm(v, T=max((gap - 3.0) / v, 0.3), s0=3.0, b=3.0, react=p["fol_react"],
                                          b_max=p["fol_b"] * grip)))
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e,
                    bounds=lambda x: (y_o - W / 2 + 0.2, y_e + W / 2 + 0.3), v_ref=v, props=[],
                    end_x=x_tr - Lt / 2)

    def done(self, env):
        """Never ends early; records when the ego is past the truck (its rear beyond the truck's front)."""
        tr = env.bodies.get("truck")
        if "route_t" not in env.mem and tr is not None and \
                env.ego.position[0] - HALF_L > tr.position[0] + tr.LENGTH / 2:
            env.mem["route_t"] = env.t
        return False

    def goal(self, env):
        """Safety only (the label); route_done: past the truck by ROUTE_T."""
        rt = env.mem.get("route_t")
        return True, dict(route_done=bool(rt is not None and rt <= ROUTE_T), route_t=None if rt is None else round(rt, 2))

    def cell(self, r):
        p = r["params"]
        return f"truck {p['d_truck']:.0f} m, first oncoming {p['onc_x0']:.0f} m past it"



def site_spec(key):
    town, site, n_left, split, speed = SITES08[key]
    sp = RC08Opt()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.sid = RC08Opt.sid if key == "A" else f"{RC08Opt.sid}@{key}"
    return sp


SPECS08X = {sp.sid: sp for sp in (site_spec(k) for k in SITES08)}
