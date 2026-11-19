"""RC10 / RC11 with options (RC10-S4-opt): the car ahead blows a tyre, skids along the ego's lane, then slides to rest
across the middle lane and into the left OR the right lane (slide_dir, hidden), the direction showing t_reveal s after
it loses control (rc11.py's RC11-S4 story, here with the direction as a scene parameter instead of a fixed class).

    RC10  t_reveal early (0.2-0.8 s): brake first, see the side, escape to the other one (brake_free, the contingent
          family); committing early to one side fails in one direction; braking in lane is too close
    RC11  t_reveal late (1.0-1.6 s): no plan decided on what is observable works in both directions
One scene and one surrogate for both: the RC is the t_reveal band (vsearch's rc10 / rc11 cases set it). Either side may
also be closed by a car alongside (adjL / adjR), so a single escape, or none, remains; a sixth of the scenes are benign
(the car regains control in its lane). A loaded van or truck tailgates the ego in most scenes (rc14_opts.heavy_rear:
close, late, braking less than the ego can), so braking hard to wait for the direction ends in a rear-end and the
evidence has to come while braking gently. Ego in the middle lane (y W) of three (0, W, 2W) on the Town06 straights of
RC07-S5 (A, B train; E1 held out).
"""
from laws import HALF_L, law_idm, spin_path
from rc03_opts import OptBase, rear_car, speed_path
from rc07_opts import _S5
from rc14_opts import REAR, heavy_rear
from scenarios import CAR, car, platoon


class RC10Opt(OptBase):
    sid = "RC10-S4-opt"
    scene_name = "spinning_lead"
    T, prefix_T = 8.0, 3.0
    goal_text = "inside the corridor, lane-centred at the horizon"
    trigger = "braking"
    snap_stop = True                                   # surrogate: CARLA's low-speed stop (surrogate.V_SNAP)
    modes = ("keep", "brake", "left", "right", "brake_free")
    SPEEDS = ("v",)
    SPIN, ADJ = "vehicle.mini.cooper_s_2021", "vehicle.audi.a2"
    nominal = dict(spin=1, slide_dir=1, t_reveal=0.9, v=24.0, headway=0.6, b_spin=10.0, slide=2.3, benign_dec=4.0,
                   adjL=0, adjL_dx=0.0, adjR=0, adjR_dx=0.0, rear=1, rear_bp=REAR[0], rear_gap=9.0, rear_react=1.0,
                   rear_b=4.0, wx="dry")
    ranges = dict(spin=[1, 1, 1, 1, 1, 0], slide_dir=[-1, 1], t_reveal=(0.2, 1.6), v=(22.0, 30.0), headway=(0.3, 0.6),
                  b_spin=(9.0, 11.0), slide=(2.1, 2.5), benign_dec=(3.5, 5.0), adjL=[0, 0, 1], adjL_dx=(-12.0, 12.0),
                  adjR=[0, 0, 1], adjR_dx=(-12.0, 12.0), rear=[1, 1, 1, 0], rear_bp=REAR, rear_gap=(6.0, 14.0),
                  rear_react=(0.8, 1.3), rear_b=(3.5, 5.0), wx=["dry", "wet", "heavy"])
    hidden = ("spin", "slide_dir", "t_reveal", "b_spin", "rear_react", "rear_b")
    description = ("The car ahead blows a tyre, skids along the ego's lane, then slides across into the left or the "
                   "right lane; which one shows only t_reveal s after it loses control. Brake in lane, commit early to "
                   "a side, or brake and escape to the side it is not sliding to once that is seen; a heavy vehicle close behind "
                   "cannot follow a hard stop.")

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        y_e = W
        x0 = HALF_L + v * p["headway"] + CAR[self.SPIN][0] / 2
        if p["spin"]:                                  # identical in both directions until t_reveal
            s = p["slide_dir"]
            path = spin_path(x0, y_e, v, p["b_spin"] * grip, p["t_reveal"], 1.2, y_e + s * p["slide"], 0.25, -s * 1.3)
        else:                                          # benign: brakes hard for a moment, keeps control in its lane
            path = speed_path(x0 - 5.0 * v, y_e, 0.0, v, [(0.0, 0.0, v), (1.5, -p["benign_dec"] * grip, v - 5.0),
                                                          (20.0, 1.0, v)])
        bodies = [car("spin_car", self.SPIN, 0, y_e, v, kind="scripted", path=path)]
        for side, y in (("L", 0.0), ("R", 2 * W)):     # a car alongside closes that side
            if p[f"adj{side}"]:
                bodies.append(car(f"adj{side}", self.ADJ, p[f"adj{side}_dx"] - v * pT, y, v,
                                  law=law_idm(v, T=1.2, react=0.8, b_max=6 * grip)))
        if p["rear"]:                                  # older batches: an attentive rear car (no rear_b)
            bodies.append(heavy_rear(p, y_e, grip, pT) if "rear_b" in p else rear_car(p, y_e, grip, pT))
        if self.n_left >= 1:
            bodies += platoon("l1_", -W, v + 2.0, [-60.0, -15.0, 35.0])
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e, bounds=lambda x: (-W / 2, 2.5 * W),
                    v_ref=v, props=[])

    def cell(self, r):
        p = r["params"]
        if not p["spin"]:
            return "benign"
        return f"slides {'left' if p['slide_dir'] < 0 else 'right'} at {p['t_reveal']:.1f} s"


def site_spec(key):
    town, site, n_left, split, speed = _S5[key]
    sp = RC10Opt()
    sp.site_key, sp.town, sp.site, sp.n_left, sp.split, sp.site_speed = key, town, site, n_left, split, speed
    sp.sid = RC10Opt.sid if key == "A" else f"{RC10Opt.sid}@{key}"
    return sp


SPECS10X = {sp.sid: sp for sp in (site_spec(k) for k in _S5)}
