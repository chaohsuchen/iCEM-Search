"""Guarded-stage programs: a conditional, closed-loop action space for iCEM.

A program has K stages. Each stage sets a lateral target (in lane-width units, relative to the start lane), a longitudinal accel, and two exits:
  * a GUARD: "feature f crosses threshold" (f from a fixed menu of OBSERVABLES, never hidden parameters)  -> jump to stage `next_guard`
  * a TIMEOUT: after `max_dur` seconds in the stage                                                      -> jump to stage `next_timeout`
Jumps only go forward (index > current) or to END (stay in the last stage's command), so programs always terminate.
This can express "change lane, then if the rear car closes in go back, otherwise accelerate", and also the trivial programs
(one stage: keep / brake), so the optimizer is never forced to a pre-written answer. The designer only fixes: K, the feature menu, and the parameter ranges.

The unit-box vector is decoded by `decode_spec` like any family: categorical dims are real numbers in [0, n] that the policy floors.

Usage in a Case:
    FAM = {"program": program_family(FEATURES, K=3, lat_range=(-1.0, 1.0), a_range=(-A_MAX, 2.0))}
    policy_cls = GuardedProgram;  observation dict must contain the keys named in FEATURES (value None = not visible -> guard cannot fire).
"""
import math
from rc01.policy import PolicyOutput
from rcx.policies import _Base

# feature menu entry: name -> (obs_key, lo, hi): the guard threshold is searched in [lo, hi]; sign chosen by `dir`
PER_STAGE = ["lat", "acc", "max_dur", "feat", "dir", "thr", "nxt_g", "nxt_t"]


def program_family(features, K=3, lat_range=(-1.0, 1.0), a_range=(-7.2, 2.0), dur_range=(0.3, 6.0), v_lat_range=(0.5, 2.5)):
    """Family spec (names/lo/hi/fixed) for decode_spec. `features` = {name: (obs_key, lo, hi)}; threshold is stored normalised in [0,1] and rescaled by the policy."""
    names, lo, hi = ["v_lat"], [v_lat_range[0]], [v_lat_range[1]]
    for k in range(K):
        names += [f"{k}_{p}" for p in PER_STAGE]
        lo += [lat_range[0], a_range[0], dur_range[0], 0.0, 0.0, 0.0, 0.0, 0.0]
        hi += [lat_range[1], a_range[1], dur_range[1], float(len(features)) + 1.0, 2.0, 1.0, float(K), float(K)]     # feat bin len(features) = "no guard"
    return dict(names=names, lo=lo, hi=hi, fixed=dict(K=K, features=dict(features)))


class GuardedProgram(_Base):
    name = "guarded_program"; version = "prog-v1"

    def __init__(self, dt=0.05, K=3, features=None, lane_w=3.5, v_target=8.0, **kw):
        super().__init__(dt, K=K, features=features or {}, lane_w=lane_w, v_target=v_target, **kw)
        self.k = self.kw; self.reset()

    def reset(self):
        super().reset(); self.stage = 0; self.t_in = 0.0; self.t_des = 0.0; self.log = []

    def _g(self, stage, key): return self.k[f"{stage}_{key}"]

    def _guard_fires(self, o, st):
        feats = list(self.k["features"].items()); fi = int(math.floor(self._g(st, "feat")))
        if fi >= len(feats): return False                                        # "no guard"
        okey, lo, hi = feats[fi][1]
        x = o.get(okey)
        if x is None: return False
        thr = lo + self._g(st, "thr") * (hi - lo)
        return x < thr if int(math.floor(self._g(st, "dir"))) % 2 == 0 else x > thr

    def step(self, o):
        k, K, v, dt = self.k, self.k["K"], o["v"], self.dt
        if not o.get("cue", True):                                                  # before the cue: plain speed hold
            return PolicyOutput("follow", self.speed_hold(v, k["v_target"]), True, 0.0, 0.0)
        st = self.stage; self.t_in += dt
        nxt = None
        if self._guard_fires(o, st): nxt = int(math.floor(self._g(st, "nxt_g")))
        elif self.t_in >= self._g(st, "max_dur"): nxt = int(math.floor(self._g(st, "nxt_t")))
        if nxt is not None and nxt > st and nxt < K: self.stage, self.t_in, st = nxt, 0.0, nxt          # forward jumps only; anything else = END (stay)
        tgt = self._g(st, "lat") * k["lane_w"]; vl = k["v_lat"]
        d = tgt - self.t_des; step = max(-vl * dt, min(vl * dt, d)); self.t_des += step; rate = step / dt
        a = self._g(st, "acc")
        if v <= 0.05 and a < 0: self.phase = "hold"; return PolicyOutput("hold", 0.0, True, self.t_des, 0.0)
        self.phase = "brake" if a < -0.3 else "follow"
        return PolicyOutput(self.phase, a, True, self.t_des, rate, dict(stage=st))
