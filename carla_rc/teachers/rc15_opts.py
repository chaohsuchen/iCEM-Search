"""RC15 with options (RC15-M-opt): the motorway merge of rc15.py with the adjacent driver's response as a scene
parameter instead of a fixed class. The ego's lane ends (cone taper); the car alongside in the target lane, once the
ego nudges towards it, yields (slows), holds its speed or matches (speeds up to stay level) - hidden, with its reaction
delay and how strongly it responds (style subtle / pronounced). Yielding must be observed, not assumed: merging in front
on an assumed yield crashes in the match world.

Sites (rc15.SIDES; site keys A = L, R): L = Town06 entrance ramp, the acceleration lane ends -> merge LEFT; R = Town06 road 35, the median
lane is coned off -> merge RIGHT. Families: keep, stop_end (protective stop before the end: a mitigation),
merge_front / merge_back (speed up / drop back until a slot, then change), watch_merge (nudge, watch the response, then
front or back: the contingent family). Success = no contact, r >= 0.05, ending in the target lane.
"""
from laws import STYLES, Body, law_adjacent, law_idm, law_lead_zipper
from rc03_opts import OptBase
from rc15 import EGO_LEAD_BP, SIDES, TAPER
from scenarios import CAR, car, platoon

PREFIX = {"L": 2.0, "R": 1.0}


class RC15Opt(OptBase):
    sid = "RC15-M-opt"
    scene_name = "lane_end_merge"
    goal_text = "ends in the target lane (in front of or behind the adjacent car)"
    trigger = "start"
    snap_stop = True
    modes = ("keep", "stop_end", "merge_front", "merge_back", "watch_merge")
    SPEEDS = ("v",)
    nominal = dict(mode="hold", style="subtle", react=0.6, dv_adj=0.0, adj_dx=-1.0, lead_dx=47.0, follow_dx=-49.0,
                   lead=0, ego_lead_dx=30.0, wx="dry")
    BASE = dict(mode=["yield", "hold", "match"], style=["subtle", "pronounced"], react=(0.5, 1.0), dv_adj=(-3.0, 3.0),
                adj_dx=(-5.0, 3.0), lead_dx=(40.0, 55.0), follow_dx=(-55.0, -43.0), lead=[0, 0, 1],
                ego_lead_dx=(20.0, 42.0), wx=["dry", "wet", "heavy"])
    hidden = ("mode", "react", "style")
    description = ("The ego's lane ends; a car alongside in the target lane yields, holds its speed or speeds up to "
                   "stay level once the ego signals (hidden). Merge in front, drop back and merge behind, watch the "
                   "response and decide, or stop before the end.")

    def __init__(self, side="L"):
        s = SIDES[side]
        self.side, self.merge_dir = side, (s["tgt_lane"] - s["ego_lane"])
        self.town, self.site, self.T = s["town"], s["site"], s["T"]
        self.ego_lane, self.tgt_lane = s["ego_lane"], s["tgt_lane"]
        self.ranges = dict(self.BASE, v=s["v"], end_x=s["end_x"])
        self.nominal = dict(self.nominal, v=sum(s["v"]) / 2, end_x=sum(s["end_x"]) / 2)

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        grip = self.road(p)["grip"]
        va = v + p["dv_adj"]
        y_e, y_t = self.ego_lane * W, self.tgt_lane * W
        side = 1.0 if y_t > y_e else -1.0
        gap0, s_eq = (p["adj_dx"] - p["follow_dx"]) - 4.8, 2.0 + 1.2 * va
        v_follow = va / (1.0 - (s_eq / gap0) ** 2) ** 0.25 if gap0 > 1.05 * s_eq else va + 3.0
        style = {k: v_ for k, v_ in STYLES[p["style"]].items()}
        adj = car("adj", "vehicle.ford.mustang", p["adj_dx"] - va * pT, y_t, va,
                  law=law_adjacent(p["mode"], p["react"], y_e, side, va, lane_w=W, **style))
        tl = car("tgt_lead", "vehicle.toyota.prius", p["lead_dx"] - va * pT, y_t, va, law=law_idm(va, T=1.2, react=0.5))
        tf = car("tgt_follow", "vehicle.nissan.patrol_2021", p["follow_dx"] - va * pT, y_t, va,
                 law=law_idm(v_follow, T=1.2, react=0.8, b_max=6 * grip))
        e0 = p["end_x"]
        end = Body("lane_end", e0 + 40.0, y_e, length=80.0, width=W - 0.6, kind="static", bp=None)
        bodies = [adj, tl, tf, end]
        if p["lead"]:                                   # a vehicle ahead in the ending lane (zips into the first gap)
            bp = EGO_LEAD_BP[self.side]
            dx = p["ego_lead_dx"] + (CAR[bp][0] - 4.9) / 2
            bodies.append(car("ego_lead", bp, dx - v * pT, y_e, v, law=law_lead_zipper(v, y_e, y_t, e0, lane_w=W)))
        for k, (lane, dv, xs) in enumerate(SIDES[self.side]["bg"]):
            bodies += platoon(f"l{k}_", lane * W, va + dv, list(xs))

        def bounds(x):
            u = min(max((x - e0) / TAPER, 0.0), 1.0)
            if side < 0:
                return y_t - W / 2, y_e + W / 2 - u * W
            return y_e - W / 2 + u * W, y_t + W / 2
        edge = y_e - side * W / 2
        inner = y_e + side * (W / 2 - 0.3)
        cones = [("static.prop.constructioncone", e0 + 2.0 * i, edge + side * 0.3 + (inner - edge - side * 0.3)
                  * min(2.0 * i / TAPER, 1.0)) for i in range(int(TAPER / 2) + 1)]
        cones += [("static.prop.constructioncone", e0 + TAPER + 8.0 * i, inner) for i in range(1, 6)]
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e, bounds=bounds, v_ref=v, props=cones,
                    end_x=e0, tgt_y=y_t)

    def goal(self, env):
        """Merged: the ego ends in the target lane (in front of or behind the adjacent car: either is a recovery)."""
        e = env.ego
        ok = abs(e.position[1] - env.scene["tgt_y"]) <= 1.0
        order = ("front" if e.position[0] > env.bodies["adj"].position[0] else "behind") if ok else "not_merged"
        return ok, dict(merge=order)

    def cell(self, r):
        return f"{r['params']['mode']} ({r['params']['style']})"


def site_spec(side):
    """Site keys: A = side L (merge left, the base sid), R = side R (merge right)."""
    sp = RC15Opt(side)
    sp.site_key, sp.n_left, sp.split, sp.site_speed = ("A" if side == "L" else side), 0, "train", 1.0
    # R7: the longest pre-roll each site's geometry allows (CARLA, 2026-11-18): L 2 s (at 3 s the adjacent car starts
    # where the ramp joins); R 1 s (earlier, the ego would start outside the coned corridor)
    sp.prefix_T = PREFIX[side]
    sp.sid = RC15Opt.sid if side == "L" else f"{RC15Opt.sid}@{side}"
    return sp


SPECS15X = {sp.sid: sp for sp in (site_spec(s) for s in SIDES)}
