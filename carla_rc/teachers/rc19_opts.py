"""RC19 on the option scenes: RC03-S1-opt (spun car), RC10-S4-opt (spinning lead) and RC07-S5-opt (blowout, on-ramp
car), every site, re-run with the same `params` under the benign cases and one semantic control (RC19 card):

    A anchor        the scene as searched
    B appearance    other paint on every car (and the ego), sun lower and clouds (dry scenes only: never rain)
    C far traffic   cars that never interact: far ahead in the corridor's other lanes and pulling away, and in the
                    lane beyond the corridor where the site has one (bg_*: never in the observation)
    D roadside      cones and barrels 2.5 m beyond the corridor's right edge (props: never occlude, never collide)
    S semantic      something that should change the label: a broken-down car closes the lateral escape
                    (RC03: the escape lane past the spun car; RC10: both adjacent lanes where the spinning car stops; RC07: the right
                    lane past the stopped lead). It is a real, observed obstacle (sem_*).

The wrapper subclasses each site's spec, so everything else (sites, ranges, laws, trigger, families) is the base
scene's; sid = "RC19-" + the base sid. Hidden settings and the scene name are the base scene's (R2 / R3): the five
cases of one anchor share its history (one group for the splits).
"""
from laws import HALF_L, Body
from scenarios import CAR, car

CASES = ["A", "B", "C", "D", "S"]
COLOURS = ["200,30,30", "20,60,160", "230,230,230", "30,30,30", "240,180,20", "40,140,60"]
W_B = dict(cloudiness=60.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0, sun_altitude_angle=30.0,
           sun_azimuth_angle=250.0, fog_density=0.0)
STALLED = "vehicle.volkswagen.t2_2021"
PROPS = ("static.prop.constructioncone", "static.prop.barrel")


def semantic(base_sid, p, W):
    """The S case's obstacles for the base scene (scenario frame of that scene)."""
    L, Wd = CAR[STALLED]
    st = lambda n, x, y: Body(n, x, y, length=L, width=Wd, kind="static", bp=STALLED)
    if base_sid.startswith("RC03"):
        y_s = W - p["side"] * W                                  # the escape lane (other side of the crash)
        return [st("sem_stalled", HALF_L + p["d0"] + 3.0 + L / 2, y_s)]
    if base_sid.startswith("RC10"):
        x = HALF_L + p["v"] * p["headway"] + 0.5 * p["v"] ** 2 / p["b_spin"] + L / 2     # beside where it stops
        return [st("sem_stalled_l", x, 0.0), st("sem_stalled_r", x, 2 * W)]
    if base_sid.startswith("RC07"):
        return [st("sem_stalled", HALF_L + p["v"] * p["headway"] + 12.0, 2 * W)]
    raise ValueError(base_sid)


def far_traffic(base_sid, p, W, n_left):
    """C: never interacting, never observed (bg_*)."""
    v = p["v"]
    lanes = {"RC03": (0.0, 2 * W), "RC10": (0.0, 2 * W), "RC07": (2 * W,)}[base_sid[:4]]
    out = [car(f"bg_far{i}", "vehicle.toyota.prius", 160.0 + 35.0 * i, y, v + 6.0, kind="scripted",
               path=_const(160.0 + 35.0 * i, y, v + 6.0)) for i, y in enumerate(lanes)]
    if n_left >= 1:
        out += [car(f"bg_farl{i}", "vehicle.audi.a2", x, -W, v + 3.0, kind="scripted", path=_const(x, -W, v + 3.0))
                for i, x in enumerate((-110.0, 120.0))]
    return out


def _const(x0, y, v):
    from laws import const_path
    return const_path(x0, y, v)


def wrap(base):
    """The RC19 spec of one base site spec."""
    B = type(base)

    class RC19X(B):
        def build(self, p, W):
            sc = B.build(self, p, W)
            case = p.get("case", "A")
            if case == "B":
                for i, b in enumerate(sc["bodies"]):
                    b.color = COLOURS[i % len(COLOURS)]
            elif case == "C":
                sc["bodies"] = sc["bodies"] + far_traffic(self.base_sid, p, W, self.n_left)
            elif case == "D":
                x0, hi = -p["v"] * self.prefix_T, sc["bounds"](0.0)[1]
                sc["props"] = list(sc.get("props", [])) + [(PROPS[k % 2], x0 + 25.0 * k, hi + 2.5) for k in range(14)]
            elif case == "S":
                sc["bodies"] = sc["bodies"] + semantic(self.base_sid, p, W)
            return sc

        def weather_for(self, p):
            return W_B if p.get("case") == "B" and p.get("wx", "dry") == "dry" else B.weather_for(self, p)

        def ego_color(self, p):
            return "240,240,240" if p.get("case") == "B" else B.ego_color(self, p)

    sp = RC19X()
    sp.__dict__.update(base.__dict__)
    sp.base_sid = base.sid
    sp.sid = "RC19-" + base.sid
    sp.ranges = dict(base.ranges, case=CASES)
    sp.nominal = dict(base.nominal, case="A")
    sp.description = base.description + " RC19: benign case A-D or the semantic control S (rc19_opts.py)."
    return sp


def specs19(SPECS):
    return {s.sid: s for s in (wrap(b) for k, b in list(SPECS.items())
                                if k.split("@")[0] in ("RC03-S1-opt", "RC10-S4-opt", "RC07-S5-opt"))}
