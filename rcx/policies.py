"""Rule-based ego policies for RC05 / RC09 / RC13. Pure logic (no CARLA). Output: rc01.policy.PolicyOutput."""
import math
from rc01.policy import PolicyOutput


def smooth(x):            # 0..1 cosine ease
    x = min(max(x, 0.0), 1.0)
    return 0.5 * (1 - math.cos(math.pi * x))


def dsmooth(x):
    return 0.5 * math.pi * math.sin(math.pi * min(max(x, 0.0), 1.0)) if 0 < x < 1 else 0.0


class _Base:
    name = "base"; version = "v1"
    def __init__(self, dt=0.05, **kw):
        self.dt = dt; self.kw = dict(kw); self._i = 0.0; self.phase = "follow"
    def params(self): return dict(self.kw)
    def reset(self): self._i = 0.0; self.phase = "follow"
    def speed_hold(self, v, vt, kp=1.5, ki=1.2, a_up=1.5):
        e = vt - v
        a = kp * e + ki * self._i
        a_cl = max(-1.0, min(a, a_up))                      # comfort accel limit (well below the 0.5 g cap)
        if a_cl == a: self._i = max(-3.0, min(3.0, self._i + e * self.dt))   # anti-windup
        return a_cl


class LaneChangeAbortPolicy(_Base):
    """RC05: follow -> lane change left -> (abort when rear-left gap/TTC threatened) -> return to original lane behind slow car.
    mode='commit' disables the abort monitor (reference)."""
    name = "lane_change_with_abort"; version = "rc05-v1"
    def __init__(self, dt=0.05, mode="abort", v_target=10.0, lane_w=3.5, t_lc=4.0, t_ret=2.5, ttc_abort=4.5, gap_abort=8.0, acc_headway=1.0, acc_d0=6.0):
        super().__init__(dt, mode=mode, v_target=v_target, lane_w=lane_w, t_lc=t_lc, t_ret=t_ret, ttc_abort=ttc_abort, gap_abort=gap_abort, acc_headway=acc_headway, acc_d0=acc_d0)
        self.k = self.kw; self.reset()
    def reset(self):
        super().reset(); self.tau0 = self.tau_a = self.t_ab = None; self.abort_progress = None
    def acc(self, o, a_speed):
        if o["lead_gap"] is not None and o["ego_t"] > -2.0:
            if self.phase in ("lane_change", "passing"):            # overtaking: only a time-to-collision safety brake, no gentle following
                closing = o["v"] - o["lead_v"]
                if closing > 0.1 and o["lead_gap"] / closing > 3.0: return a_speed
            d_des = self.k["acc_d0"] + self.k["acc_headway"] * o["v"]
            a_acc = 0.25 * (o["lead_gap"] - d_des) + 0.9 * (o["lead_v"] - o["v"])
            return min(a_speed, a_acc)
        return a_speed
    def step(self, o):
        k, W, v = self.k, self.k["lane_w"], o["v"]
        a_speed = self.speed_hold(v, k["v_target"])
        t_des, rate = 0.0, 0.0
        g, vr = o["rl_gap"], o["rl_v"] - v
        ttc = g / vr if (g is not None and vr > 0.1 and g > 0) else float("inf")
        threat = bool(o.get("hazard", True)) and g is not None and (ttc < k["ttc_abort"] or g < k["gap_abort"])
        if self.phase == "follow" and o["cue"]:
            self.phase, self.tau0 = "lane_change", o["t"]
        if self.phase == "lane_change":
            x = (o["t"] - self.tau0) / k["t_lc"]
            t_des, rate = -W * smooth(x), -W * dsmooth(x) / k["t_lc"]
            if k["mode"] == "abort" and threat:
                self.phase, self.tau_a, self.t_ab = "abort", o["t"], o["ego_t"]
                self.abort_progress = -o["ego_t"] / W
            elif x >= 1.0:
                self.phase = "passing"
        if self.phase == "passing":
            t_des, rate = -W, 0.0
        if self.phase == "abort":
            x = (o["t"] - self.tau_a) / k["t_ret"]
            t_des, rate = self.t_ab * (1 - smooth(x)), -self.t_ab * dsmooth(x) / k["t_ret"]
            if x >= 1.0: self.phase = "follow_lead"
        if self.phase == "follow_lead":
            t_des, rate = 0.0, 0.0
        a = self.acc(o, a_speed) if self.phase != "passing" else a_speed
        return PolicyOutput(self.phase, a, True, t_des, rate, dict(ttc_rl=min(ttc, 99.0), threat=int(threat), lc_progress=-o["ego_t"] / W))


class EscapePolicy(_Base):
    """RC03: open-loop timed response to a spun car. Structure only (no conditions): wait t_react, then
      mode='keep'        : hold speed in lane (reference),
      mode='brake'       : brake at a_brake until stopped, optionally shifted `shift` m away from the intruder inside the lane,
      mode='lane_change' : change into the escape lane (esc_dir = lateral sign of that lane) over t_lc while accelerating at a_lc (negative = braking); then brake a_post (>=0) until stopped.
      mode='staged'      : (RC04 / RC06) brake at a_slow for t_slow, then change into the escape lane as in 'lane_change' (a_lc, t_lc, a_post).
    The optimizer chooses the numbers; nothing here looks at other actors."""
    name = "escape"; version = "rc03-v1"
    def __init__(self, dt=0.05, mode="brake", v_target=18.0, lane_w=3.5, esc_dir=-1.0, t_react=0.3, a_brake=6.0, shift=0.0, t_lc=2.5, a_lc=0.0, a_post=0.0, t_slow=0.0, a_slow=0.0):
        super().__init__(dt, mode=mode, v_target=v_target, lane_w=lane_w, esc_dir=esc_dir, t_react=t_react, a_brake=a_brake, shift=shift, t_lc=t_lc, a_lc=a_lc, a_post=a_post, t_slow=t_slow, a_slow=a_slow)
        self.k = self.kw; self.reset()
    def reset(self): super().reset(); self.t0 = None
    def step(self, o):
        k, v = self.k, o["v"]
        if self.t0 is None and o["cue"]: self.t0 = o["t"]
        a, t_des, rate = self.speed_hold(v, k["v_target"]), 0.0, 0.0
        x = (o["t"] - self.t0 - k["t_react"]) if self.t0 is not None else -1.0
        if x >= 0.0 and k["mode"] != "keep":
            if k["mode"] == "brake":
                self.phase = "brake"; a = -k["a_brake"]; xs = x / 1.0
                t_des, rate = -k["esc_dir"] * k["shift"] * smooth(xs), -k["esc_dir"] * k["shift"] * dsmooth(xs)     # shift away from the intruder (opposite side of the escape lane)
            elif k["mode"] == "staged" and x < k["t_slow"]:
                self.phase = "slow"; a = -k["a_slow"]
            else:
                if k["mode"] == "staged": x -= k["t_slow"]
                u = x / k["t_lc"]; self.phase = "lane_change" if u < 1.0 else "brake"
                t_des, rate = k["esc_dir"] * k["lane_w"] * smooth(u), k["esc_dir"] * k["lane_w"] * dsmooth(u) / k["t_lc"]
                a = k["a_lc"] if u < 1.0 else (-k["a_post"] if k["a_post"] > 0.0 else self.speed_hold(v, v))
            if self.phase == "brake" and v < 0.3: self.phase = "hold"
        return PolicyOutput(self.phase, a, True, t_des, rate, dict(x_react=x))


class StopPolicy(_Base):
    """RC09: speed hold -> brake -> hold. mode 'hard': constant -a_hard request; 'moderate': -a_mod unless more is needed to stop short of the queue."""
    name = "stop_policy"; version = "rc09-v1"
    def __init__(self, dt=0.05, mode="hard", v_target=8.0, a_hard=3.0, a_mod=2.0, stop_margin=2.0):
        super().__init__(dt, mode=mode, v_target=v_target, a_hard=a_hard, a_mod=a_mod, stop_margin=stop_margin); self.k = self.kw
    def step(self, o):
        k, v = self.k, o["v"]
        if self.phase == "follow" and o["cue"]: self.phase = "brake"
        if self.phase == "brake" and v <= 0.05: self.phase = "hold"
        if self.phase == "follow": return PolicyOutput("follow", self.speed_hold(v, k["v_target"]))
        if self.phase == "hold": return PolicyOutput("hold", 0.0)
        if k["mode"] == "hard": a = k["a_hard"]
        else:
            room = max(o["front_gap"] - k["stop_margin"], 0.5)
            a = min(k["a_hard"], max(k["a_mod"], v * v / (2 * room)))
        return PolicyOutput("brake", -a)


class ReachableSetPolicy(_Base):
    """RC13: keep speed unless the pedestrian reachable set (disc of radius v_ped_max*tau) can touch the ego footprint
    swept at constant speed within horizon; then brake to stop before the pedestrian."""
    name = "reachable_set_pedestrian"; version = "rc13-v1"
    def __init__(self, dt=0.05, v_target=8.0, v_ped_max=1.8, horizon=2.0, ped_r=0.3, ego_half_w=1.08, ego_half_l=2.4, pad=0.3, a_max=3.0, lane_half=1.75):
        super().__init__(dt, v_target=v_target, v_ped_max=v_ped_max, horizon=horizon, ped_r=ped_r, ego_half_w=ego_half_w, ego_half_l=ego_half_l, pad=pad, a_max=a_max, lane_half=lane_half); self.k = self.kw
    def conflict(self, o):
        k = self.k; s, v = o["s"], o["v"]; best = None
        for (ps, pt) in o["peds"]:
            if ps < s - 1.0: continue
            for i in range(0, int(k["horizon"] / 0.1) + 1):
                tau = i * 0.1
                es = s + v * tau
                dx = max(es - k["ego_half_l"] - k["pad"] - ps, 0.0, ps - (es + k["ego_half_l"] + k["pad"]))
                dt_ = max(-(k["ego_half_w"] + k["pad"]) - pt, 0.0, pt - (k["ego_half_w"] + k["pad"]))
                if math.hypot(dx, dt_) <= k["v_ped_max"] * tau + k["ped_r"]:
                    if best is None or ps < best[0]: best = (ps, tau)
                    break
        return best
    def lane_overlap(self, o):
        """Does any pedestrian's reachable disc (radius v_max*horizon) overlap the ego lane band, ignoring where the ego is?"""
        k = self.k; r = k["v_ped_max"] * k["horizon"] + k["ped_r"]
        return any(ps >= o["s"] - 1.0 and (pt - r) <= k["lane_half"] and (pt + r) >= -k["lane_half"] for ps, pt in o["peds"])

    def step(self, o):
        k, v = self.k, o["v"]
        c = self.conflict(o); plo = int(self.lane_overlap(o))
        if c is None: self.phase = "follow"
        else: self.phase = "hold" if v <= 0.05 else "yield"
        if self.phase == "follow": return PolicyOutput("follow", self.speed_hold(v, k["v_target"]), info=dict(conflict=0, pred_lane_overlap=plo))
        if self.phase == "hold": return PolicyOutput("hold", 0.0, info=dict(conflict=int(c is not None), pred_lane_overlap=plo))
        ps = c[0] if c else o["s"] + 20
        d = max(ps - (o["s"] + k["ego_half_l"]) - 2.0, 0.5)
        return PolicyOutput("yield", -min(k["a_max"], max(1.0, v * v / (2 * d))), info=dict(conflict=int(c is not None), pred_lane_overlap=plo))


class SafeStopAdapter:
    """RC01 / RC17: the unchanged SafeStopPolicy (approach -> brake at -3.0 -> hold) behind the common engine interface."""
    name = "safe_stop_rule_policy"

    def __init__(self, dt, pcfg):
        from rc01.policy import SafeStopPolicy
        self.pcfg, self.p = pcfg, SafeStopPolicy(pcfg, dt); self.version = pcfg.version

    def params(self): return self.pcfg.to_dict()

    def reset(self): self.p.reset()

    def step(self, o):
        from rc01.policy import Observation
        return self.p.step(Observation(frame=o["frame"], sim_time=o["t"], ego_speed=o["v"], blocked_road=o["delivered"], obstacle_gap=o["gap"]))
