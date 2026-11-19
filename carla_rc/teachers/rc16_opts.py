"""RC16 with options (RC16-S1-opt): squeezing past a double-parked van, where the centre point clears and the body may
not.

A delivery van is double-parked in the right part of the ego's lane on a two-way town street (Town01, one 4 m lane each
way). A traffic island on the centre line, opposite the van, leaves a passage of `w_pass` m whose centre is `off` m
left of the ego lane's centre; the island starts `isl_lead` m before the van (a corridor, not a point gate). The ego
follows a van (`lead`) that hides the parked one until it swerves through the passage itself, so the parked van is seen
only about d_see = isl_lead + lead length + gap + v T_lead / 2 ahead (the lead follows lead_hw s ahead). A car of length L and width B at heading psi covers L sin(psi) + B cos(psi) across the road:
a late, sharp shift is still angled in the passage and its corners clip the van or the island, an early shallow one
(or braking first, then a tighter shift at low speed) is straight by then.

Families: keep / brake / stop_end (stop behind the van, always safe when started in time; the van stays 20-90 s) /
shift_pass (options.py: slow to v_pass, shift dy over D m ending x_ready m before the van, hold, back). Trigger
"braking": the parked van seen standing in the ego lane. Nothing is hidden at the trigger: once the van is in view the
geometry is observable (its departure only matters for waiting, beyond the horizon).

The island is the corridor's left bound over its length (exact, footprint corners) and is drawn with cones; the parked
van is a body (clearance with the 0.5 m reserve). A centre-point or circle planner (the RC16 contrast) would take the
late swerve: env._margins uses the rotated footprint.

Sites: Town01 straights (roads 11, 6, 4: training); Town02 road 12 held out (unseen town).
"""
from laws import HALF_L, Body, quintic
from rc03_opts import OptBase
from scenarios import CAR, car

CONE = "static.prop.constructioncone"
PARKED = ["vehicle.mercedes.sprinter", "vehicle.volkswagen.t2_2021", "vehicle.ford.ambulance"]
LEADS16 = ["vehicle.ford.ambulance", "vehicle.mercedes.sprinter"]      # box-bodied, tall
SITES16 = {   # town, LaneFrame site (one lane each way: ego lane index 0), lanes left for background, split, speed
    "A": ("Town01", dict(start_xy=(392.4, 8.6), ref_lane=-1, ref_idx=0, s0=45.0), 0, "train", 1.0),
    "B": ("Town01", dict(start_xy=(101.6, 330.6), ref_lane=-1, ref_idx=0, s0=45.0), 0, "train", 1.0),
    "C": ("Town01", dict(start_xy=(101.4, 133.4), ref_lane=-1, ref_idx=0, s0=45.0), 0, "train", 1.0),
    "H": ("Town02", dict(start_xy=(181.5, 105.4), ref_lane=1, ref_idx=0, s0=40.0), 0, "heldout", 1.0),
}


def swerve_through(x0, y0, v, y1, x_in, x_out, T_lc):
    """The lead at constant v: a quintic shift y0 -> y1 over T_lc ending when its front reaches x_in (its centre at x0
    at t = 0, length via x_in / x_out already in centre terms), back after its centre passes x_out."""
    t_in, t_out = (x_in - x0) / v, (x_out - x0) / v

    def f(t):
        u = quintic((t - (t_in - T_lc)) / T_lc) - quintic((t - t_out) / T_lc)
        return x0 + v * t, y0 + (y1 - y0) * u, 0.0, v, 0.0
    return f


class RC16Opt(OptBase):
    sid = "RC16-S1-opt"
    scene_name = "double_parked_van"
    T, prefix_T = 15.0, 3.0
    goal_text = "inside the corridor, lane-centred at the horizon"
    trigger = "braking"
    lane_w = 4.0                                       # Town01 / Town02 lanes (the surrogate's W)
    end_at_road_end = True                             # CARLA: the Town01 / 02 straights end before T at speed
    snap_stop = True
    pass_dir = -1                                       # the passage is on the left of the van
    T_S = 1.5                                           # [s] after t0 the lead starts its swerve
    modes = ("keep", "brake", "stop_end", "shift_pass")
    SPEEDS = ("v",)
    nominal = dict(v=9.0, w_pass=2.9, off=1.2, park_bp=PARKED[0], lead_hw=1.0, isl_lead=8.0, lead_bp=LEADS16[0],
                   T_lead=1.0, wx="dry")
    ranges = dict(v=(6.0, 12.0), w_pass=(2.6, 3.4), off=(0.8, 1.6), park_bp=PARKED, lead_hw=(0.6, 1.6),
                  isl_lead=(3.0, 12.0), lead_bp=LEADS16, T_lead=(0.8, 1.4), wx=["dry", "dry", "wet"])
    hidden = ()
    description = ("A van double-parked in the right part of the ego lane, a traffic island opposite it: a passage "
                   "w_pass m wide, off m left of the lane centre. The van ahead hides it until it swerves through. "
                   "Stop behind, brake and shift, or shift early and shallow; a late sharp swerve clips with the body.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        y_e = 0.0
        y_pc = y_e - p["off"]                                    # passage centre
        y_isl, y_van_l = y_pc - p["w_pass"] / 2, y_pc + p["w_pass"] / 2
        Lp, Wp = CAR[p["park_bp"]]
        Ll = CAR[p["lead_bp"]][0]
        # the lead (lead_hw s ahead of the ego, same speed) starts its shift T_S s after t0 and is fully over when its
        # front reaches the island; it hides the parked van until then, so the ego sees it about
        # d_see ~ isl_lead + Ll + g + v T_lead / 2 ahead (12-40 m over the ranges)
        g = max(v * p["lead_hw"], 4.0)
        x_isl0 = HALF_L + v * self.T_S + g + Ll + v * p["T_lead"]
        x_pr = x_isl0 + p["isl_lead"]
        x_isl1 = x_pr + Lp + 2.0
        bodies = [Body("parked", x_pr + Lp / 2, y_van_l + Wp / 2, length=Lp, width=Wp, kind="static", bp=p["park_bp"])]
        bodies.append(car("lead", p["lead_bp"], 0, y_e, v, kind="scripted",
                          path=swerve_through(HALF_L + g + Ll / 2, y_e, v, y_pc, x_isl0 - Ll / 2, x_isl1 + Ll / 2,
                                              p["T_lead"])))
        lo_n, hi = y_e - W / 2 - 0.8, y_e + W / 2 + 0.5            # the lane, a little over the centre line, the kerb
        bounds = lambda x: (y_isl if x_isl0 <= x <= x_isl1 else lo_n, hi)
        props = [(CONE, float(x), y_isl - 0.3) for x in (x_isl0 + k * 1.5 for k in range(int((x_isl1 - x_isl0) / 1.5) + 1))]
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e, bounds=bounds, v_ref=v, props=props,
                    end_x=x_pr)

    def cell(self, r):
        p = r["params"]
        return f"passage {p['w_pass']:.1f} m, lead {p['lead_hw']:.1f} s ahead"


def site_spec(key):
    town, site, n_left, split, speed = SITES16[key]
    sp = RC16Opt()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.sid = RC16Opt.sid if key == "A" else f"{RC16Opt.sid}@{key}"
    return sp


SPECS16X = {sp.sid: sp for sp in (site_spec(k) for k in SITES16)}
