"""Maneuver families for the RC09 surrogate. A family = (policy class, parameter names, bounds, fixed values).
All families are executed by the same LowLevelController + plant as the scripted policies, so a theta_p found in the surrogate
can be replayed in CARLA with the same class (rcx.policies-style PolicyOutput)."""
import numpy as np
from rc01.policy import PolicyOutput
from rcx.policies import _Base

A_MAX = 7.2        # m/s^2, ~0.95 * measured max decel (controller clips anyway); within the 1 g envelope


class ProfilePolicy(_Base):
    """Open-loop longitudinal profile after the cue:
       stage 0 (d0 s): hold speed (a0 == 0) or accelerate forward at a0;  stage 1 (t1 s): brake at a1;  stage 2: brake at a2 until stopped, then hold."""
    name = "profile_policy"; version = "rc09-prof-v1"
    def __init__(self, dt=0.05, v_target=8.0, a0=0.0, d0=0.0, a1=3.0, t1=99.0, a2=None):
        super().__init__(dt, v_target=v_target, a0=a0, d0=d0, a1=a1, t1=t1, a2=a1 if a2 is None else a2); self.k = self.kw; self.t_cue = None
    def reset(self): super().reset(); self.t_cue = None
    def step(self, o):
        k, v = self.k, o["v"]
        if self.t_cue is None and o.get("cue", o.get("delivered")): self.t_cue = o["t"]
        if self.phase == "hold": return PolicyOutput("hold", 0.0)
        if self.t_cue is None: return PolicyOutput("follow", self.speed_hold(v, k["v_target"]))      # before the cue: plain speed hold (preroll)
        tau = o["t"] - self.t_cue
        if tau < k["d0"]:
            return PolicyOutput("follow", k["a0"] if k["a0"] > 0.05 else self.speed_hold(v, k["v_target"]))
        if v <= 0.05: self.phase = "hold"; return PolicyOutput("hold", 0.0)
        self.phase = "brake"
        return PolicyOutput("brake", -(k["a1"] if tau < k["d0"] + k["t1"] else k["a2"]))


FAMILIES = {
    "brake":        dict(names=["d0", "a1"],               lo=[0.0, 1.0],            hi=[3.0, A_MAX],             fixed=dict(a0=0.0, t1=99.0)),
    "two_stage":    dict(names=["d0", "a1", "t1", "a2"],   lo=[0.0, 2.0, 0.3, 0.5],  hi=[3.0, A_MAX, 4.0, A_MAX],  fixed=dict(a0=0.0)),
    "surge_brake":  dict(names=["a0", "d0", "a1"],         lo=[0.3, 0.2, 2.0],       hi=[4.0, 3.0, A_MAX],        fixed=dict(t1=99.0)),
}


def decode(fam, u):
    """unit-box vector -> ProfilePolicy kwargs"""
    f = FAMILIES[fam]; x = np.asarray(f["lo"]) + np.clip(u, 0, 1) * (np.asarray(f["hi"]) - np.asarray(f["lo"]))
    kw = dict(f["fixed"]); kw.update({n: float(v) for n, v in zip(f["names"], x)}); return kw


def decode_spec(spec, u):
    x = np.asarray(spec["lo"]) + np.clip(u, 0, 1) * (np.asarray(spec["hi"]) - np.asarray(spec["lo"]))
    kw = dict(spec.get("fixed", {})); kw.update({n: float(v) for n, v in zip(spec["names"], x)}); return kw
