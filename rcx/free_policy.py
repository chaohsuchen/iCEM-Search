"""Free-form response policy (RC04 pilot, docs/FREE_SEARCH_PROPOSAL.md): K time-indexed control points of requested longitudinal acceleration and lateral target, followed by a fixed
named continuation. Pure logic (no CARLA). Output: rc01.policy.PolicyOutput, executed by the same LowLevelController as the structured families.

Definition (version free-v1)
  prefix      starts T_START s after the cue (the smallest reaction delay the structured families may use) and lasts (K - 1) * DK s: K = 6 knots, DK = 1.0 s, i.e. 0.1 s .. 5.1 s.
              before the prefix: speed hold in lane (same as the structured families before their reaction time).
  a_req       piecewise-linear between the knots, requested longitudinal acceleration in m/s^2 (negative = braking).
  lateral     knot values are fractions `lam` of the escape-lane offset: t_des = esc_dir * lane_w * lam (lam = 0 own lane centre, lam = 1 centre of the other lane, esc_dir = -side as in
              EscapePolicy, so the sign convention is the scene's lane frame: t positive to the right). Cosine-eased interpolation between knots (C1), t_rate = feed-forward derivative.
  continuation after the last knot: hold the last lateral target; longitudinal = speed regulator holding the speed reached at the hand-over (the same `speed_hold(v, v)` as the
              structured families use for a_post = 0). The prefix is open loop in time; the continuation is feedback on speed and lateral error (through the controller).
  stop latch  phase `hold` (hand brake) as soon as the ego has stopped (v < 0.3 m/s) after the prefix started, as in the structured families.
Nothing here looks at other actors: no conditional / guarded program."""
import numpy as np
from rc01.policy import PolicyOutput
from .policies import _Base, smooth, dsmooth

K = 6
T_START = 0.1                 # s after the cue
DK = 1.0                      # s between knots
A_LO, A_HI = -9.0, 2.0        # requested acceleration bounds (m/s^2); brake bound = largest brake request of the structured families
LAM_LO, LAM_HI = -0.3, 1.2    # lateral target bounds (fraction of the escape-lane offset)
DA_MAX = 8.0                  # hard limit |a_{j+1} - a_j| per knot interval (m/s^2 per DK = jerk 8 m/s^3 on average)
DLAM_MAX = 0.8                # hard limit |lam_{j+1} - lam_j| per knot interval (2.8 m/s lateral target rate for a 3.5 m lane)
SMOOTH_W = 0.2                # weight of the smoothness penalty in the search cost
NAMES = [f"a{j}" for j in range(K)] + [f"l{j}" for j in range(K)]
LO = [A_LO] * K + [LAM_LO] * K
HI = [A_HI] * K + [LAM_HI] * K
SPEC = dict(names=NAMES, lo=LO, hi=HI, fixed=dict(mode="free"))


def project(a, lam):
    """Hard rate limits (applied in order, so every decoded vector satisfies them): the first lateral knot starts within DLAM_MAX of the lane centre."""
    a, lam = [float(x) for x in a], [float(x) for x in lam]
    lam[0] = float(np.clip(lam[0], -DLAM_MAX, DLAM_MAX))
    for j in range(1, K):
        a[j] = float(np.clip(a[j], a[j - 1] - DA_MAX, a[j - 1] + DA_MAX))
        lam[j] = float(np.clip(lam[j], lam[j - 1] - DLAM_MAX, lam[j - 1] + DLAM_MAX))
    return a, lam


def decode(u):
    """unit-box vector (12) -> TrajectoryPolicy kwargs (bounds, then hard rate limits)."""
    x = np.asarray(LO) + np.clip(u, 0, 1) * (np.asarray(HI) - np.asarray(LO))
    a, lam = project(x[:K], x[K:])
    return dict(mode="free", a=a, lam=lam, t_start=T_START, dk=DK)


def smoothness_penalty(kw):
    """Sum of squared knot-to-knot changes, normalised by the hard limits (0 = constant profiles). Added to the search cost with weight SMOOTH_W; reported with the results."""
    a, lam = np.asarray(kw["a"]), np.asarray(kw["lam"])
    return float(np.sum((np.diff(a) / DA_MAX) ** 2) + np.sum((np.diff(lam) / DLAM_MAX) ** 2))


def from_brake(a_brake, t_react=0.1):
    """Control points that reproduce EscapePolicy(mode='brake', shift=0) (mechanical test): constant -a_brake from t_react on, lane centre."""
    return dict(mode="free", a=[-a_brake] * K, lam=[0.0] * K, t_start=t_react, dk=DK)


class TrajectoryPolicy(_Base):
    name = "free_trajectory"; version = "free-v1"

    def __init__(self, dt=0.05, mode="free", v_target=22.0, lane_w=3.5, esc_dir=-1.0, a=(0.0,) * K, lam=(0.0,) * K, t_start=T_START, dk=DK):
        super().__init__(dt, mode=mode, v_target=v_target, lane_w=lane_w, esc_dir=esc_dir, a=[float(x) for x in a], lam=[float(x) for x in lam], t_start=t_start, dk=dk)
        self.k = self.kw; self.reset()

    def reset(self): super().reset(); self.t0 = None; self.v_hand = None

    def step(self, o):
        k, v = self.k, o["v"]
        if self.t0 is None and o["cue"]: self.t0 = o["t"]
        a, t_des, rate = self.speed_hold(v, k["v_target"]), 0.0, 0.0
        x = (o["t"] - self.t0 - k["t_start"]) if self.t0 is not None else -1.0
        n = len(k["a"]); T = (n - 1) * k["dk"]; W = k["esc_dir"] * k["lane_w"]
        if x >= 0.0:
            if x < T:
                j = int(x / k["dk"]); u = (x - j * k["dk"]) / k["dk"]; self.phase = "free"
                a = k["a"][j] + (k["a"][j + 1] - k["a"][j]) * u
                d = k["lam"][j + 1] - k["lam"][j]
                t_des = W * (k["lam"][j] + d * smooth(u)); rate = W * d * dsmooth(u) / k["dk"]
            else:
                if self.v_hand is None: self.v_hand = v
                self.phase = "brake"; t_des = W * k["lam"][-1]; a = self.speed_hold(v, self.v_hand)         # continuation (see module docstring)
            if v < 0.3: self.phase = "hold"
        return PolicyOutput(self.phase, a, True, t_des, rate, dict(x_react=x))
