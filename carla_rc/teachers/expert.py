"""The notebook's witness policies as experts inside TeacherEnv (for behaviour cloning / DAgger warm starts).

Only the notebook's framework and policy cells are executed (core, simulator types, laws, highway toolkit, RC15
toolkit; no scenario runs), with LANE_W set to the site's real lane width. The expert sees the scenario actors'
true states every tick (no delay; background traffic excluded, as in the notebook scenes) and its (accel, steering)
command is converted to this environment's action: steering -> HighwayEnv bicycle yaw rate -> u_yaw.

    RC03-S1: OnHazard(EscapeAdjacent(0, A_lat=6))        (notebook WITNESS)
    RC07-S1: OnHazard(SwerveReturn(-W, A_ret=3))         ("avoid_continue_return")
    RC15-S1: MergeObserve(0, **STYLES['pronounced'])     ("observe_then_merge")
    RC15-<side>-<kind>: MergeObserve(target lane, **STYLES[side's style]['pol'])  (the class fixes the response, the
             same observe-then-merge logic then produces the class's outcome; the search tunes its timing / gaps)
    RC19-S1..S3: CorridorDriver(v_set = the variation's speed, ego lane)          ("corridor_driver")
    RC19-S4: MergeObserve(0, **STYLES['pronounced'])
    RC03-S2/S8: OnHazard(EscapeAdjacent(2 / 0))      RC03-S5: OnHazard(EscapeAdjacent(1, wait_slot=True))
    RC03-S4: OnHazard(SwerveReturn(-W))  RC03-S6: OnHazard(SwerveReturn(+2.3))  RC07-S3: OnHazard(SwerveReturn(-1.0)) back to 0.3-0.45 m right of centre
    RC07-S2: OnHazard(EscapeAdjacent(2, v_set = the variation's right-lane speed))   ("avoid_then_match_traffic")
    RC07-S6: OnHazard(SwerveReturn(-W, A_ret=3))    RC07-S7: OnHazard(SwerveReturn(+W, A_ret=3, margin=1))
    RC03-S8, RC07-S6: OnHazard also fires on the crossing car's near edge (its centre is ~2.5 m further out)
    RC11-<S>-<mode>: SharedBranch committing to the mode's right pick (the class fixes the mode; "right pick survives")
Before t0 the expert cruises (the notebook's prefix policy).
"""
import json
import math
import os

import numpy as np

NB_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "RC_scenarios_HighwayEnv_CARLA.ipynb")
CELLS = (3, 5, 9, 11, 59, 75, 87)
_NS = None


def notebook_ns(nb_path=NB_DEFAULT):
    global _NS
    if _NS is None:
        nb = json.load(open(nb_path))
        ns = {"__name__": "rc_notebook"}
        for i in CELLS:
            src = "".join(nb["cells"][i]["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        _NS = ns
    return _NS


MERGE_POL = {   # notebook STYLES[...]["pol"] defaults (MergeObserve's own defaults for 'subtle')
    "pronounced": dict(a_probe=2.0, dv_probe=6.0, g_front=8.0, dv_fall=4.0, b_fall=3.5),
    "subtle": dict(a_probe=1.5, dv_probe=3.0, g_front=3.0, dv_fall=1.5, b_fall=3.0)}


def merge_grid(style, order):
    """Settings around the style's defaults: for a front merge its timing and required gap, for a behind merge how
    firmly it drops back and the gap it takes (first = the notebook witness)."""
    d = MERGE_POL[style]
    if order == "front":
        return [dict(T_obs=T, a_probe=d["a_probe"] + da, g_front=d["g_front"] + dg)
                for T in (3.0, 5.0) for da in (0.0, 0.5) for dg in (0.0, 3.0)]
    return [dict(g_behind=g, dv_fall=d["dv_fall"] + dv, b_fall=d["b_fall"] + db)
            for g in (5.0, 8.0) for dv in (0.0, 2.0) for db in (0.0, 1.5)]


def _grids():
    from scenarios import SPECS
    g = {
        "RC03-S1": [dict(A_lat=A, b_evasive=b, y=y) for y in (0.0, 0.35) for A in (6.0, 5.0, 4.0) for b in (3.0, 5.0)],
        "RC07-S1": [dict(dy=dy, A_lat=A, A_ret=Ar) for dy in (0.0, 0.4) for A in (6.0, 4.0) for Ar in (3.0, 2.0)],
        "RC15-S1": [dict(g_behind=g, dv_fall=dv, b_fall=b) for g in (5.0, 8.0) for dv in (4.0, 6.0) for b in (3.5, 5.0)],
        "RC19-S4": merge_grid("pronounced", "behind"),
    }
    for sid, sp in SPECS.items():
        if sid.startswith("RC15-") and sid != "RC15-S1" and hasattr(sp, "order"):   # (RC15-M-opt: no fixed order)
            g[sid] = merge_grid(sp.style, sp.order)
        elif sid in ("RC19-S1", "RC19-S2", "RC19-S3"):
            g[sid] = [dict(T=T, b=b) for T in (1.2, 1.0, 1.5) for b in (2.5, 3.5)]
        elif sid.startswith("RC11-"):
            g[sid] = [dict(b_s=b, t_commit=tc, y_off=yo) for yo in (0.6, 0.85, 0.35) for b in (0.0, 3.0) for tc in (0.0, 0.3)]
    escape = [dict(A_lat=A, b_evasive=b, y_off=yo) for yo in (0.35, 0.6, 0.0) for A in (6.0, 5.0) for b in (3.0,)]
    g.update({
        "RC03-S2": escape, "RC03-S8": escape,
        "RC03-S4": [dict(dy=dy, A_lat=A, A_ret=Ar) for dy in (0.4, 0.7) for A in (6.0, 5.0) for Ar in (3.0, 4.0)],
        "RC03-S5": [dict(A_lat=A, s_slot=s, y_off=yo) for yo in (0.35, 0.6) for A in (5.0, 6.0) for s in (1.5, 3.0)],
        "RC03-S6": [dict(dy=dy, A_lat=A, A_ret=Ar) for dy in (2.3, 2.0) for A in (6.0, 5.0) for Ar in (3.0, 4.0)],
        "RC07-S2": [dict(A_lat=A, a=3.0, dv_set=d, y_off=yo) for yo in (0.35, 0.6) for A in (5.0, 6.0) for d in (0.0, 1.0)],
        "RC07-S3": [dict(dy=dy, A_ret=Ar, y_back=yb) for dy in (-1.0, -0.9) for Ar in (3.0, 4.0) for yb in (0.3, 0.45)],
        "RC07-S6": [dict(dy=dy, A_lat=A, A_ret=Ar) for dy in (0.2, 0.4) for A in (6.0, 5.0) for Ar in (3.0, 4.0)],
        "RC07-S7": [dict(dy=dy, A_ret=Ar, margin=m) for dy in (0.4, 0.7) for Ar in (3.0, 4.0) for m in (1.0, 2.0)],
    })
    return g


def build_policy(ns, env, cfg):
    """The RC03 / RC07 / RC11 experts (built per episode: some read the variation). cfg = a CONFIG_GRID setting;
    dy for full-lane swerves = how much short of the lane centre the swerve stops."""
    sid, W, c = env.spec.sid, env.W, dict(cfg)
    OnH, Esc, Sw = ns["OnHazard"], ns["EscapeAdjacent"], ns["SwerveReturn"]
    home = env.scene["home_y"]
    y_off = c.pop("y_off", 0.0)                          # aim this much inside the target lane (CARLA overshoot)
    if sid in ("RC03-S8", "RC07-S6"):
        class OnH(ns["OnHazard"]):
            """Also fires when an actor moves toward the ego lane (> 0.5 m/s) with its near edge within 1.5 lanes: the
            side-street car is crosswise, so hazard_seen's centre test fires ~0.65 s after its nose starts moving."""
            def act(self, o):
                if self.lane_y is None:
                    self.lane_y = round(o.ego["y"] / W) * W
                if self.t_det is None and len(o.history) >= 2:
                    (t1, s1), (t2, s2) = o.history[-2:]
                    for n, b in s2.items():
                        if n in s1 and ((b["y"] - s1[n]["y"]) / (t2 - t1) * np.sign(self.lane_y - b["y"]) > 0.5 and
                                        abs(b["y"] - self.lane_y) - ns["extent"](b)[1] / 2 < 1.5 * W):
                            self.t_det = o.t
                return super().act(o)

    def esc(lane, **kw):
        y = lane * W + math.copysign(y_off, home - lane * W)
        return OnH(Esc(lane, y=y, **kw))
    if sid in ("RC03-S2", "RC03-S8"):
        return esc(2 if sid == "RC03-S2" else 0, **{"A_lat": 6.0, **c})
    if sid == "RC03-S5":
        return esc(1, wait_slot=True, **{"A_lat": 5.0, **c})
    if sid == "RC07-S2":
        dv = c.pop("dv_set", 0.0)
        return esc(2, v_set=env.params["v_right"] + dv, name="avoid_then_match_traffic", **{"A_lat": 5.0, "a": 3.0, **c})
    if sid in ("RC03-S4", "RC03-S6", "RC07-S3", "RC07-S6", "RC07-S7"):
        dy, y_back = c.pop("dy", 0.0), c.pop("y_back", 0.0)
        side = {"RC03-S4": -(W - dy), "RC07-S6": -(W - dy), "RC07-S7": W - dy, "RC03-S6": dy, "RC07-S3": dy}[sid]
        base = {"RC03-S4": dict(A_lat=6.0, A_ret=6.0), "RC03-S6": dict(A_lat=6.0, A_ret=6.0),
                "RC07-S3": dict(A_lat=3.0, A_ret=3.0, margin=1.0), "RC07-S6": dict(A_lat=6.0, A_ret=3.0),
                "RC07-S7": dict(A_lat=6.0, A_ret=6.0, margin=1.0)}[sid]
        if y_back:                                       # come back off-centre (RC07-S3: room for the motorcycle)
            class SwBack(Sw):
                def act(self, o):
                    if self.home is None:
                        self.home = round(o.ego["y"] / W) * W + y_back
                    return super().act(o)
            return OnH(SwBack(side - y_back, **{**base, **c}))
        return OnH(Sw(side, **{**base, **c}))
    if sid.startswith("RC11-"):
        sp = env.spec
        opts = {k: (v if v is None or isinstance(v, str) else v + math.copysign(y_off, home - v) * (v != home))
                for k, v in sp.options_y(W).items()}
        return ns["SharedBranch"](ns["NEVER"], sp.answer, opts, default=sp.pick, hold=True,
                                  name=f"commit_{sp.pick}", **{"b_s": 0.0, "t_commit": 0.0, **c})
    raise KeyError(sid)


CONFIG_GRID = _grids()


class Expert:
    def __init__(self, env, nb_path=NB_DEFAULT, cfg=None):
        ns = notebook_ns(nb_path)
        ns["LANE_W"] = env.W
        self.ns, self.env = ns, env
        sid = env.spec.sid
        cfg = dict(cfg or {})
        self.cfg = cfg
        if sid == "RC03-S1":         # y: target offset from the left lane's centre (+ = toward the ego lane)
            kw = {"A_lat": 6.0, **{k: v for k, v in cfg.items() if k != "y"}}
            self.pol = ns["OnHazard"](ns["EscapeAdjacent"](0, y=cfg.get("y", 0.0), **kw))
        elif sid == "RC07-S1":       # dy: how much less than a full lane width the swerve goes
            kw = {"A_ret": 3.0, **{k: v for k, v in cfg.items() if k != "dy"}}
            self.pol = ns["OnHazard"](ns["SwerveReturn"](-(env.W - cfg.get("dy", 0.0)), name="avoid_continue_return", **kw))
        elif sid == "RC15-S1":
            self.pol = ns["MergeObserve"](0, **{**ns["STYLES"]["pronounced"]["pol"], **cfg})
        elif sid.startswith("RC15-") or sid == "RC19-S4":
            sp = env.spec
            self.pol = ns["MergeObserve"](sp.tgt_lane, **{**MERGE_POL[sp.style], **cfg})
        elif sid in ("RC19-S1", "RC19-S2", "RC19-S3"):
            self.pol = None                      # CorridorDriver: built per episode with the variation's speed
        else:
            self.pol = None                      # build_policy: built per episode
        self.prefix = ns["cruise"]()
        self.per_episode = self.pol is None
        self.name = "corridor_driver" if sid.startswith("RC19-") else ("" if self.pol is None else self.pol.name)

    def reset(self):
        env = self.env
        if self.per_episode and self.name == "corridor_driver":
            self.pol = self.ns["CorridorDriver"](env.params["v"], int(round(env.scene["home_y"] / env.W)), **self.cfg)
        elif self.per_episode:
            self.pol = build_policy(self.ns, env, self.cfg)
            self.name = self.pol.name
        self.pol.reset()
        self.prefix.reset()

    def act(self, obs=None):
        """Action u in [-1, 1]^2 for the env's current state."""
        env, ns = self.env, self.ns
        e = env.ego
        ego = dict(x=float(e.position[0]), y=float(e.position[1]), v=float(e.speed), psi=float(e.heading),
                   length=5.0, width=2.0, steer=0.0, acc=float(env.ego_acc))
        cur = env.snaps[-1][1]
        if env.t < -1e-6:
            o = ns["Obs"](t=env.t + env.spec.prefix_T, ego=ego, actors=cur, history=env.snaps, lanes=3)
            a, d = self.prefix.act(o)
        else:
            o = ns["Obs"](t=env.t, ego=ego, actors=cur, history=[s for s in env.snaps if s[0] >= -1e-6] or env.snaps[-1:],
                          lanes=3)
            a, d = self.pol.act(o)
        r = e.speed * math.sin(math.atan(0.5 * math.tan(d))) / 2.5           # HighwayEnv bicycle yaw rate
        u0 = a / 3.0 if a >= 0 else a / 8.0
        u1 = r / min(0.6, 7.0 / max(e.speed, 5.0))
        return np.clip(np.array([u0, u1], np.float32), -1.0, 1.0)
