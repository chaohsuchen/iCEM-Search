"""RC15 merge scenarios: one scenario class per (merge side, adjacent driver's response, ego outcome).

The ego's lane ends (cone taper) and it must merge into the target lane next to the adjacent car `adj`. The policy
knows its class, so it never has to tell the adjacent driver's response apart; the variations change how strongly and
when that response comes (speeds, offsets, reaction time, yield / match strength) and the surrounding gaps.

    side L: Town06 entrance ramp, the acceleration lane (right) ends -> merge LEFT     (notebook RC15-S1, pronounced)
    side R: Town06 road 35, the median lane is coned off -> merge RIGHT                (notebook RC15-S2, subtle)

    kind          adjacent driver             ego                                        intended outcome
    hold-front    holds its speed             speeds up past it (room ahead)             merges in front
    hold-behind   holds its speed             cannot speed up (vehicle ahead in lane)    slows, merges behind
    match-behind  speeds up to stay level     tries to speed up, sees it matched         slows, merges behind
    yield-front   slows to open a gap         waits until the gap ahead of it is enough  merges in front

Success = notebook r >= 0.05 AND the ego ends in the target lane on the intended side of the adjacent car.
"""
import numpy as np

from laws import HALF_L, STYLES, Body, law_adjacent, law_idm, law_lead_zipper
from scenarios import CAR, Spec, car, platoon, rel

SIDES = {
    "L": dict(town="Town06", site=dict(start_xy=(40.0, 48.9), ref_lane=-6, ref_idx=0, s0=0.0), ego_lane=1, tgt_lane=0,
              style="pronounced", T=14.0, end_x=(165.0, 195.0), v=(17.0, 20.0),
              bg=[(-1, 3.0, (-80.0, -30.0, 20.0, 75.0)), (-2, 6.0, (-55.0, 5.0, 65.0))],
              where="Town06 entrance ramp (road 58 -> junction 511); acceleration lane closed by a cone taper"),
    "R": dict(town="Town06", site=dict(start_xy=(21.7, 143.0), ref_lane=-3, ref_idx=0, s0=40.0), ego_lane=0, tgt_lane=1,
              style="subtle", T=15.0, end_x=(200.0, 240.0), v=(23.0, 27.0),
              bg=[(2, 2.0, (-70.0, -20.0, 35.0, 90.0)), (3, 0.0, (-45.0, 15.0, 70.0)), (4, -2.0, (-60.0, 0.0, 55.0))],
              where="Town06 road 35 (5-lane motorway); the median lane (-3) is coned off for road works"),
}

KINDS = {   # adjacent mode, intended order, kind-specific variation ranges (dv_adj = v_adj - v)
    "hold-front": dict(mode="hold", order="front",
                       L=dict(dv_adj=(-3.0, 0.0), adj_dx=(-3.0, 3.0)),
                       R=dict(dv_adj=(-2.5, 0.0), adj_dx=(-4.0, 1.0))),
    "hold-behind": dict(mode="hold", order="behind",
                        L=dict(dv_adj=(0.0, 2.5), adj_dx=(-4.0, 2.0), ego_lead_dx=(30.0, 42.0)),
                        R=dict(dv_adj=(0.0, 2.0), adj_dx=(-3.0, 2.0), ego_lead_dx=(20.0, 30.0))),
    "match-behind": dict(mode="match", order="behind",
                         L=dict(dv_adj=(-2.0, 1.0), adj_dx=(-5.0, 0.0), a_match=(2.5, 3.5), d_match=(2.0, 4.0)),
                         R=dict(dv_adj=(-1.5, 1.0), adj_dx=(-4.0, 0.0), a_match=(2.0, 3.0), d_match=(-2.5, -1.0))),
    "yield-front": dict(mode="yield", order="front",
                        L=dict(dv_adj=(0.0, 3.0), adj_dx=(-4.0, 1.0), dv_yield=(5.0, 8.0), b_yield=(2.8, 4.0)),
                        R=dict(dv_adj=(0.0, 2.0), adj_dx=(-4.0, 0.0), dv_yield=(2.5, 4.0), b_yield=(2.0, 3.0))),
}
EGO_LEAD_BP = {"L": "vehicle.carlamotors.european_hgv", "R": "vehicle.mercedes.sprinter"}   # the vehicle ahead (hold-behind)
TAPER = 24.0


class MergeSpec(Spec):
    def __init__(self, side, kind):
        s, k = SIDES[side], KINDS[kind]
        self.side, self.kind, self.mode, self.order = side, kind, k["mode"], k["order"]
        self.sid = f"RC15-{side}-{kind}"
        self.town, self.site, self.T, self.style = s["town"], s["site"], s["T"], s["style"]
        self.ego_lane, self.tgt_lane = s["ego_lane"], s["tgt_lane"]
        self.ranges = dict(v=s["v"], react=(0.3, 0.9), lead_dx=(40.0, 55.0), follow_dx=(-55.0, -43.0), end_x=s["end_x"],
                           **k[side])
        self.nominal = {n: (float(np.mean(r)) if isinstance(r, tuple) else r[0]) for n, r in self.ranges.items()}
        self.has_lead = "ego_lead_dx" in self.ranges
        self.obs_names = Spec.EGO_NAMES + ["adj_dx", "adj_dvx", "adj_a", "lead_dx", "lead_dvx", "follow_dx",
                                           "follow_dvx", "end_dx"]
        if self.has_lead:
            self.obs_names += ["ego_lead_dx", "ego_lead_dy", "ego_lead_dvx"]
        self.description = (f"{s['where']}; merge {'left' if side == 'L' else 'right'}; adjacent driver {self.mode} "
                            f"({self.style}); ego merges {self.order}")

    def adj_style(self, p):
        return {**STYLES[self.style], **{k: p[k] for k in ("dv_yield", "b_yield", "a_match", "d_match") if k in p}}

    def build(self, p, W):
        pT, v = self.prefix_T, p["v"]
        va = v + p["dv_adj"]
        y_e, y_t = self.ego_lane * W, self.tgt_lane * W
        side = 1.0 if y_t > y_e else -1.0
        gap0, s_eq = (p["adj_dx"] - p["follow_dx"]) - 4.8, 2.0 + 1.2 * va
        v_follow = va / (1.0 - (s_eq / gap0) ** 2) ** 0.25 if gap0 > 1.05 * s_eq else va + 3.0
        adj = car("adj", self.adj_bp(p), p["adj_dx"] - va * pT, y_t, va,
                  law=law_adjacent(self.mode, p["react"], y_e, side, va, lane_w=W, **self.adj_style(p)))
        lead = car("tgt_lead", "vehicle.toyota.prius", p["lead_dx"] - va * pT, y_t, va, law=law_idm(va, T=1.2, react=0.5))
        fol = car("tgt_follow", "vehicle.nissan.patrol_2021", p["follow_dx"] - va * pT, y_t, va,
                  law=law_idm(v_follow, T=1.2, react=0.8))
        e0 = p["end_x"]
        end = Body("lane_end", e0 + 40.0, y_e, length=80.0, width=W - 0.6, kind="static", bp=None)
        bodies = [adj, lead, fol, end]
        if self.has_lead:
            bp = EGO_LEAD_BP[self.side]
            L_l = CAR[bp][0]
            dx = p["ego_lead_dx"] + (L_l - 4.9) / 2
            bodies.append(car("ego_lead", bp, dx - v * pT, y_e, v,
                              law=law_lead_zipper(v, y_e, y_t, e0, lane_w=W)))
        for k, (lane, dv, xs) in enumerate(SIDES[self.side]["bg"]):
            bodies += platoon(f"l{k}_", lane * W, va + dv, list(xs))

        def bounds(x):
            u = min(max((x - e0) / TAPER, 0.0), 1.0)
            if side < 0:                                       # ego lane is the right one: its right edge closes in
                return y_t - W / 2, y_e + W / 2 - u * W
            return y_e - W / 2 + u * W, y_t + W / 2           # ego lane is the left one: its left edge closes in
        edge = y_e - side * W / 2                           # outer edge of the ending lane
        inner = y_e + side * (W / 2 - 0.3)                   # cones end 0.3 m inside the ending lane's line
        cones = [("static.prop.constructioncone", e0 + 2.0 * i, edge + side * 0.3 + (inner - edge - side * 0.3)
                  * min(2.0 * i / TAPER, 1.0)) for i in range(int(TAPER / 2) + 1)]
        cones += [("static.prop.constructioncone", e0 + TAPER + 8.0 * i, inner) for i in range(1, 6)]
        return dict(ego=dict(x=-v * pT, y=y_e, v=v), bodies=bodies, home_y=y_e, bounds=bounds, v_ref=v, props=cones,
                    end_x=e0, tgt_y=y_t)

    def adj_bp(self, p):
        return "vehicle.ford.mustang"

    def features(self, env):
        f = self.ego_features(env)
        dx, _, dvx, _ = rel(env, "adj")
        h = env.speed_hist["adj"]
        f += [dx, dvx, (h[-1] - h[-11]) / 0.5 if len(h) > 10 else 0.0]
        for n in ("tgt_lead", "tgt_follow"):
            dx, _, dvx, _ = rel(env, n)
            f += [dx, dvx]
        f += [env.scene["end_x"] - env.ego.position[0]]
        if self.has_lead:
            dx, dy, dvx, _ = rel(env, "ego_lead")
            f += [dx, dy, dvx]
        return np.array(f, dtype=np.float32)

    def goal(self, env):
        """The ego ends in the target lane, on the intended side of the adjacent car."""
        e = env.ego
        if abs(e.position[1] - env.scene["tgt_y"]) > 1.0:
            order = "not_merged"
        else:
            order = "front" if e.position[0] > env.bodies["adj"].position[0] else "behind"
        return order == self.order, dict(merge=order, intended=self.order)


SPECS15 = {s.sid: s for s in (MergeSpec(side, kind) for side in SIDES for kind in KINDS)}
