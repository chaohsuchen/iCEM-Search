"""Case registry for the CEM pipeline. A Case bundles: scenario variations (theta_s, in CARLA terms), the surrogate simulate(), robust perturbations,
the cost / outcome class, the margin used to pick boundary candidates, and how to map theta_s to CARLA scenario kwargs."""
import json, os
import numpy as np
from rc01.config import PolicyConfig
from .plant import LongPlant
from . import rc01_sim, rc09_sim
from .families import ProfilePolicy, FAMILIES, decode_spec, A_MAX


def _soft(c, pairs):
    return sum(w * max(0.0, thr - x) for w, thr, x in pairs)


class _Base:
    FAM = FAMILIES
    @property
    def families(self): return tuple(self.FAM)
    def decode(self, fam, u): return decode_spec(self.FAM[fam], u)
    def policy_cls(self, fam): return ProfilePolicy
    def make_policy(self, dt, fam, kw): return self.policy_cls(fam)(dt, **kw)
    def companions(self, th): return []


class RC09Case(_Base):
    name = "rc09"
    calib_path = "results_v3/rc09/calibration.json"; carla_scene = "rc09"
    SCEN = {"baseline": {}, "closing6": dict(closing=6.0), "delay2.0": dict(rear_delay_s=2.0), "rear12": dict(rear_gap=12.0), "front18": dict(front_gap=18.0),
            "closing6_front18": dict(closing=6.0, front_gap=18.0)}

    def __init__(self):
        self.calib = json.load(open(self.calib_path)); self.plant = LongPlant(self.calib)

    def simulate(self, th, kw, fam=None): return rc09_sim.simulate(th, kw, self.plant, self.calib, policy_cls=ProfilePolicy)

    def perturbed(self, th):
        b = dict(rc09_sim.SCN); b.update(th)
        return [th, {**th, "closing": b["closing"] + 0.5}, {**th, "rear_delay_s": b["rear_delay_s"] + 0.2}, {**th, "front_gap": b["front_gap"] - 1.0}, {**th, "rear_gap": b["rear_gap"] - 1.0}]

    def cost(self, r):
        from .rc09_cost import cost; return cost(r)

    def outcome_class(self, r):
        from .rc09_cost import outcome_class; return outcome_class(r)

    def margin(self, r): return min(r["min_front_gap"] - 1.0 - 0.4, r["min_rear_gap"] - 1.0 - 0.1)

    def carla_kw(self, th): return dict(th)


class RC01Case(_Base):
    """RC01 (van blocks the street) + RC17 (same, observation delayed by delay_s). theta_s in CARLA terms: gap (trigger gap), v_ego, delay_s."""
    name = "rc01"; carla_scene = "rc01"
    STOP_MARGIN = 0.7          # surrogate stopping gap is 0.34 m optimistic on average, up to 0.66 m (128 CARLA runs); the surrogate PASS threshold is CARLA 2.0 m + this margin (min_clear in rc01_sim)
    SCEN = {"baseline": {}, "g12": dict(gap=12.0), "v9.5": dict(v_ego=9.5), "v9.5_g14": dict(v_ego=9.5, gap=14.0), "d0.3_g17": dict(gap=17.0, delay_s=0.3),
            "d0.6_g14": dict(gap=14.0, delay_s=0.6), "g8_v9.5": dict(gap=8.0, v_ego=9.5)}

    def __init__(self):
        self.cal = {}
        for v in (6.0, 8.0, 9.5):
            self.cal[v] = json.load(open(f"results_v3/rc01/calibration_v{v:g}.json"))
        self.plants = {v: LongPlant(c) for v, c in self.cal.items()}

    def _sim_th(self, th):
        b = dict(gap=20.0, v_ego=8.0, delay_s=0.0); b.update(th)
        return b, dict(v_ego=b["v_ego"] + 0.11, gap=b["gap"] - 0.01, delay_s=b["delay_s"])

    def simulate(self, th, kw, fam=None):
        b, sth = self._sim_th(th); v = min(self.cal, key=lambda x: abs(x - b["v_ego"]))
        return rc01_sim.simulate(sth, kw, self.plants[v], self.cal[v], pcfg=PolicyConfig(target_speed=b["v_ego"]), policy_cls=ProfilePolicy)

    def perturbed(self, th):
        b = dict(gap=20.0, v_ego=8.0, delay_s=0.0); b.update(th)
        return [th, {**th, "v_ego": b["v_ego"] + 0.3}, {**th, "delay_s": b["delay_s"] + 0.1}, {**th, "gap": b["gap"] - 1.0}]

    def outcome_class(self, r):
        if r["verdict"] == "PASS": return "success"
        if r["contact"]: return "mitigate" if r["rel_speed"] < 2.0 else "fail"
        return "mitigate" if r["stopped"] else "fail"      # stopped short of the 2 m clearance, or never stopped

    def cost(self, r):
        rows = np.array(r["rows"]); a = rows[:, 6]; peak = float(max(0.0, -a.min())) if len(a) else 0.0
        if r["contact"]: return 1000.0 + 50.0 * r["rel_speed"] + 0.1 * peak
        if r["reason"].startswith("did not stop"): return 300.0 + 20.0 * float(rows[-1][1])
        return 0.1 * peak + 50.0 * max(0.0, 2.0 + self.STOP_MARGIN - r["stopping_gap"]) + 0.3 * max(0.0, 4.0 - r["stopping_gap"])

    def margin(self, r): return (r["stopping_gap"] - 2.0 - self.STOP_MARGIN) if r["stopped"] and not r["contact"] else -1.0

    def carla_kw(self, th): return dict(th)


class RC13Case(_Base):
    """RC13: pedestrians on the sidewalk; variation = a 'dart' pedestrian that steps into the lane (dart_speed, dart_dist). A policy must stay fast when nobody
    enters the lane (benign) and avoid contact when someone does (dart): robust cost = worst over both."""
    name = "rc13"; carla_scene = "rc13"
    FAM = {"reachable_set": dict(names=["v_ped_max", "horizon", "pad", "a_max"], lo=[1.0, 0.5, 0.0, 2.0], hi=[5.5, 3.0, 1.0, A_MAX], fixed={}),
           "pre_slow": dict(names=["v_slow", "zone", "a_react"], lo=[2.0, 10.0, 2.0], hi=[8.0, 50.0, A_MAX], fixed={}),
           "react": dict(names=["thr", "a_react", "d_react"], lo=[0.3, 2.0, 10.0], hi=[3.0, A_MAX, 45.0], fixed={})}
    SCEN = {"dart2@20": dict(dart_speed=2.0, dart_dist=20.0), "dart3@15": dict(dart_speed=3.0, dart_dist=15.0), "dart4@12": dict(dart_speed=4.0, dart_dist=12.0),
            "dart5@9": dict(dart_speed=5.0, dart_dist=9.0)}
    BENIGN = dict(dart_speed=0.0)

    def __init__(self):
        from . import rc13_sim
        self.m = rc13_sim; self.calib = json.load(open("results_v3/rc13/calibration.json")); self.plant = LongPlant(self.calib)

    def policy_cls(self, fam):
        from rcx.policies import ReachableSetPolicy
        return {"reachable_set": ReachableSetPolicy, "pre_slow": self.m.PreSlowPolicy, "react": self.m.ReactPolicy}[fam]

    def simulate(self, th, kw, fam="reachable_set"):
        return self.m.simulate(th, kw, self.plant, self.calib, policy_cls=self.policy_cls(fam))

    def perturbed(self, th):
        return [{**th, **self.BENIGN}, th, {**th, "dart_speed": th["dart_speed"] + 0.7, "dart_dist": max(5.0, th["dart_dist"] - 2.0)}]

    def companions(self, th): return [("benign", {**th, **self.BENIGN})]

    def outcome_class(self, r):
        if r["verdict"] == "PASS": return "success"
        if r["contact"]: return "mitigate" if r["rel_speed"] < 2.0 else "fail"
        return "mitigate" if not r["entered_lane"] else "fail"           # unnecessary slowing = mitigate (safe but conservative); clearance miss = fail

    def cost(self, r):
        rows = np.array(r["rows"]); a = rows[:, 6]; peak = float(max(0.0, -a.min())) if len(a) else 0.0
        if r["contact"]: return 1000.0 + 50.0 * r["rel_speed"] + 0.1 * peak
        c = 0.1 * peak + 20.0 * max(0.0, self._peak_filtered(rows) - 8.5)      # CARLA envelope: 0.2 s mean decel <= 9.81 (soft margin from 8.5)
        if not r["entered_lane"]:
            vp = r["v_pass"] if r["v_pass"] is not None else r["min_speed"]
            c += 30.0 * max(0.0, 7.2 - vp) + (100.0 if r["stopped"] else 0.0)             # unnecessary slowing
        else: c += 40.0 * max(0.0, 0.5 + 0.3 - r["min_front_gap"])                       # near miss with the darting pedestrian
        return c

    @staticmethod
    def _peak_filtered(rows):
        a = np.array(rows)[:, 6]; return float(max(0.0, max(-a[max(0, i - 3):i + 1].mean() for i in range(len(a)))))

    def margin(self, r): return (r["min_front_gap"] - 0.5 - 0.3) if r["entered_lane"] else (r["v_pass"] - 7.2 if r["v_pass"] else -1.0)

    def carla_kw(self, th): return dict(th)


class RC05Case(_Base):
    """RC05: lane change with a rear-left car that accelerates at the hazard cue. Families: abort (return to the original lane), abort + forward surge, commit + surge."""
    name = "rc05"; carla_scene = "rc05"
    FAM = {"abort": dict(names=["ttc_abort", "gap_abort", "t_ret"], lo=[1.0, 2.0, 1.0], hi=[9.0, 25.0, 4.0], fixed=dict(mode="abort", surge_acc=0.0, surge_dur=0.0)),
           "abort_surge": dict(names=["ttc_abort", "gap_abort", "t_ret", "surge_acc", "surge_dur"], lo=[1.0, 2.0, 1.0, 0.5, 0.5], hi=[9.0, 25.0, 4.0, 4.9, 4.0], fixed=dict(mode="abort")),
           "commit_surge": dict(names=["surge_acc", "surge_dur"], lo=[0.5, 0.5], hi=[4.9, 5.0], fixed=dict(mode="commit"))}
    SCEN = {"baseline": {}, "p0.8": dict(hazard_progress=0.8), "g12": dict(rear_gap=12.0), "c4.5": dict(closing=4.5), "g40": dict(rear_gap=40.0), "c4.5_g12": dict(closing=4.5, rear_gap=12.0)}
    RL_MARGIN, LEAD_MARGIN = 0.3, 0.3

    def __init__(self):
        from . import rc05_sim
        self.m = rc05_sim; self.calib = json.load(open("results_v3/rc05/calibration.json")); self.plant = LongPlant(self.calib)

    def policy_cls(self, fam): return self.m.LCFamilyPolicy

    def simulate(self, th, kw, fam=None): return self.m.simulate(th, kw, self.plant, self.calib)

    def perturbed(self, th):
        b = dict(self.m.SCN); b.update(th)
        return [th, {**th, "closing": b["closing"] + 0.5}, {**th, "rear_gap": b["rear_gap"] - 2.0}, {**th, "hazard_progress": min(0.9, b["hazard_progress"] + 0.05)}, {**th, "hazard_progress": max(0.1, b["hazard_progress"] - 0.05)}, {**th, "rear_acc": b["rear_acc"] + 0.5}]

    def outcome_class(self, r):
        if r["verdict"] == "PASS": return "success"
        if r["contact"]: return "mitigate" if r["rel_speed"] < 2.0 else "fail"
        return "mitigate" if r["min_rl"] >= 0.3 else "fail"

    def cost(self, r):
        if r["contact"]: return 1000.0 + 50.0 * r["rel_speed"]
        c = 0.05 * r["peak_lat"] + 0.05 * r["peak_decel"] + 0.3 * max(0.0, 3.0 - r["min_rl"])
        c += 50.0 * max(0.0, 1.0 + self.RL_MARGIN - r["min_rl"]) + 50.0 * max(0.0, 2.0 + self.LEAD_MARGIN - r["min_lead"])
        c += (100.0 if not r["settled"] else 0.0) + (50.0 if r["stable"] is False else 0.0) + (100.0 if r["peak_lat"] > 9.81 else 0.0)
        return c

    def margin(self, r): return min(r["min_rl"] - 1.0 - self.RL_MARGIN, r["min_lead"] - 2.0 - self.LEAD_MARGIN)

    def carla_kw(self, th): return dict(th)


class RC03Case(_Base):
    """RC03: a car spins into the ego's neighbouring lane. Families (structure only, open loop): keep (trivial reference), brake (timed in-lane brake + optional shift),
    lane_change (timed change into the opposite lane with a free longitudinal accel, then optional braking). No conditions: the optimizer only picks numbers."""
    name = "rc03"; carla_scene = "rc03"
    FAM = {"keep": dict(names=[], lo=[], hi=[], fixed=dict(mode="keep")),
           "brake": dict(names=["t_react", "a_brake", "shift"], lo=[0.1, 2.0, 0.0], hi=[1.0, 9.0, 0.8], fixed=dict(mode="brake")),
           "lane_change": dict(names=["t_react", "t_lc", "a_lc", "a_post"], lo=[0.1, 1.5, -5.0, 0.0], hi=[1.0, 4.0, 2.0, 6.0], fixed=dict(mode="lane_change"))}
    # scene variations (CARLA terms; see rcx/rc03_laws.DEFAULTS). side=+1: crash on the ego's right.
    SCEN = {"baseline": {}, "near_deep": dict(d0=30.0, intrude=1.4), "fast_far": dict(v_ego=22.0, d0=60.0, v_post=20.0, v_esc=22.0, esc_gap=40.0),
            "tight_gap": dict(esc_gap=18.0), "braking_lead": dict(lead_brake=4.0), "tailgater": dict(rear=1, rear_gap=18.0), "clip": dict(intrude=0.55, d0=35.0),
            "benign": dict(crash=0)}

    def __init__(self):
        from . import rc03_sim
        self.m = rc03_sim; self.cal = {}
        for v in (18.0, 22.0): self.cal[v] = json.load(open(f"results_v3/rc03/calibration_v{v:g}.json"))     # CARLA calibration at each ego speed (brake cap / drag differ a lot with speed)
        self.plants = {v: LongPlant(c) for v, c in self.cal.items()}

    def policy_cls(self, fam):
        from rcx.policies import EscapePolicy; return EscapePolicy

    def simulate(self, th, kw, fam=None):
        v0 = {**self.m.SCN, **th}["v_ego"]; v = min(self.cal, key=lambda x: abs(x - v0))
        return self.m.simulate(th, kw, self.plants[v], self.cal[v], pcfg=PolicyConfig(target_speed=v0))

    def perturbed(self, th):
        b = dict(self.m.SCN); b.update(th)
        return [th, {**th, "d0": b["d0"] - 3.0}, {**th, "intrude": b["intrude"] + 0.1}, {**th, "esc_gap": b["esc_gap"] - 3.0}, {**th, "gap_center": b["gap_center"] + 2.0}]

    def outcome_class(self, r):
        if r["verdict"] == "PASS": return "success"
        if r["contact"]: return "mitigate" if r["rel_speed"] < 2.0 else "fail"
        return "fail"

    def cost(self, r):
        rows = np.array(r["rows"]); a = rows[:, 7]; peak = float(max(0.0, -a.min())) if len(a) else 0.0
        if r["contact"]: return 1000.0 + 50.0 * r["rel_speed"] + 0.1 * peak
        c = 0.1 * peak + 0.1 * r["peak_lat"] + 20.0 * max(0.0, r["peak_filt"] - 8.5) + 20.0 * max(0.0, r["peak_lat"] - 8.5)
        c += 40.0 * max(0.0, 1.0 - r["min_clear"]) + (30.0 if not r["settled"] else 0.0) + (30.0 if r["reason"] else 0.0)
        return c

    def margin(self, r): return r["min_clear"] - 0.5

    def carla_kw(self, th): return dict(th)


class RC04Case(_Base):
    """RC04 (staged slowing before a feasible lane change): obstacle ahead in the ego lane, a truck alongside in the other lane and its follower. Families (open loop, structure only):
    brake (stop in lane), lane_change (immediate change, free longitudinal accel), staged (brake t_slow, then change). The optimizer picks numbers only."""
    name = "rc04"; carla_scene = "rc04"; mod = "rc04_sim"
    FAM = {"brake": dict(names=["t_react", "a_brake"], lo=[0.1, 2.0], hi=[1.0, 9.0], fixed=dict(mode="brake", shift=0.0)),
           "lane_change": dict(names=["t_react", "t_lc", "a_lc", "a_post"], lo=[0.1, 1.5, -5.0, 0.0], hi=[1.0, 4.0, 2.0, 6.0], fixed=dict(mode="lane_change")),
           "staged": dict(names=["t_react", "t_slow", "a_slow", "t_lc", "a_lc", "a_post"], lo=[0.1, 0.3, 1.0, 1.5, -3.0, 0.0], hi=[1.0, 3.0, 6.0, 4.0, 2.0, 6.0], fixed=dict(mode="staged"))}
    SCEN = {"baseline": dict(d_obs=55.0), "truck_far_ahead": dict(d_obs=46.0, side_pos=30.0), "near_obstacle": dict(d_obs=42.0), "slow_truck": dict(d_obs=55.0, v_side=19.0),
            "truck_behind": dict(d_obs=50.0, side_pos=-15.0), "follower_close": dict(d_obs=55.0, fol_gap=10.0), "no_follower": dict(d_obs=55.0, fol=0), "benign": dict(d_obs=55.0, obs_lane=2)}

    def __init__(self):
        import importlib
        self.m = importlib.import_module(f"search.{self.mod}"); self.cal = {}
        for v in (18.0, 22.0): self.cal[v] = json.load(open(f"results_v3/rc03/calibration_v{v:g}.json"))      # ego dynamics only: shared with rc03 (speed-dependent, scene-independent)
        self.plants = {v: LongPlant(c) for v, c in self.cal.items()}

    def policy_cls(self, fam):
        if fam == "free": from rcx.free_policy import TrajectoryPolicy; return TrajectoryPolicy      # pilot family, see docs/FREE_SEARCH_PROPOSAL.md (not in FAM: the structured searches are unchanged)
        from rcx.policies import EscapePolicy; return EscapePolicy

    def simulate(self, th, kw, fam=None):
        v0 = {**self.m.SCN, **th}["v_ego"]; v = min(self.cal, key=lambda x: abs(x - v0))
        return self.m.simulate(th, kw, self.plants[v], self.cal[v], pcfg=PolicyConfig(target_speed=v0), policy_cls=self.policy_cls(fam))

    def perturbed(self, th):
        b = dict(self.m.SCN); b.update(th)
        return [th, {**th, "d_obs": b["d_obs"] - 3.0}, {**th, "side_pos": b["side_pos"] + 2.0}, {**th, "fol_gap": max(4.0, b["fol_gap"] - 3.0)}, {**th, "v_side": b["v_side"] - 0.7}]

    def outcome_class(self, r):
        if r["verdict"] == "PASS": return "success"
        if r["contact"]: return "mitigate" if r["rel_speed"] < 2.0 else "fail"
        return "fail"

    def cost(self, r):
        rows = np.array(r["rows"]); a = rows[:, 7]; peak = float(max(0.0, -a.min())) if len(a) else 0.0
        if r["contact"]: return 1000.0 + 50.0 * r["rel_speed"] + 0.1 * peak
        c = 0.1 * peak + 0.1 * r["peak_lat"] + 20.0 * max(0.0, r["peak_filt"] - 8.5) + 20.0 * max(0.0, r["peak_lat"] - 8.5)
        c += 40.0 * max(0.0, 1.0 - r["min_clear"]) + (30.0 if not r["settled"] else 0.0) + (30.0 if r["reason"] else 0.0)
        return c

    def margin(self, r): return r["min_clear"] - 0.5

    def carla_kw(self, th): return dict(th)


class RC06Case(RC04Case):
    """RC06 (yield, then use a time-varying gap): same scene and surrogate as RC04 (rcx/rc04_laws), different variations: a broken-down car 60-200 m ahead, a car PASSING the ego (faster by
    +1..+8 m/s, around the ego), optionally followed by a second car. Families: brake = protected stop behind the broken-down car (fallback, always safe if the distance allows, costs the
    route), lane_change = immediate entry (free longitudinal accel = accelerate past it), staged = wait (slow down), then enter behind the passer. The passer's law (constant /
    accelerates / brakes on seeing the blockage) and the second car's reaction are hidden."""
    name = "rc06"; carla_scene = "rc06"
    SUR = dict(it0=0.05, acc_gain=0.7, fol_pad=0.8, side_pad=0.5, obs_pad_lc=0.5)     # RC06 surrogate overrides: the free-accel response measured in CARLA (a_lc > 0) is much weaker than RC04's it0 = 0.25 integrator start predicts

    def simulate(self, th, kw, fam=None): return super().simulate({**self.SUR, **th}, kw, fam)
    SCEN = {"baseline": dict(d_obs=90.0, side_pos=0.0, v_side=26.0), "passer_ahead": dict(d_obs=90.0, side_pos=12.0, v_side=26.0), "passer_behind": dict(d_obs=90.0, side_pos=-12.0, v_side=26.0),
            "fast_passer": dict(d_obs=90.0, side_pos=0.0, v_side=29.0), "slow_passer": dict(d_obs=90.0, side_pos=0.0, v_side=23.0), "passer_brakes": dict(d_obs=90.0, side_pos=0.0, v_side=26.0, side_acc=-3.0),
            "second_car_close": dict(d_obs=90.0, side_pos=0.0, v_side=26.0, fol_gap=10.0), "near_block": dict(d_obs=60.0), "benign": dict(d_obs=90.0, obs_lane=2)}

    def perturbed(self, th):
        b = dict(self.m.SCN); b.update(th)
        return [th, {**th, "d_obs": b["d_obs"] - 5.0}, {**th, "side_pos": b["side_pos"] + 2.0}, {**th, "fol_gap": max(4.0, b["fol_gap"] - 3.0)}, {**th, "side_react": b["side_react"] + 0.4}]


CASES = {"rc09": RC09Case, "rc01": RC01Case, "rc13": RC13Case, "rc05": RC05Case, "rc03": RC03Case, "rc04": RC04Case, "rc06": RC06Case}
