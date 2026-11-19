"""(vsearch) Cases on Kent's CARLA environment, in the Case shape of cases.py (variations, simulate, perturbed, cost,
outcome class, margin), with the additions agreed for the combined pipeline:

    variations   stratified random draws (every discrete combination, Latin hypercube in the continuous ranges) at
                 scale, named presets (easy / mid / hard / closed / benign), and boundary sweeps (pipeline.py)
    verdicts     three-way (PASS / FAIL / UNCERTAIN): a candidate is PASS when it recovers in the nominal scene, in every
                 scene perturbation (perturbed: what the other actors may do differently) and in every ego-dynamics
                 world (kent_bridge.WORLDS: where the surrogate may differ from CARLA), FAIL when it recovers in none,
                 UNCERTAIN otherwise (CARLA labels it); scored with FP and FN, coverage reported, target 1.000
    policies     event-driven families (options.py): an observable trigger, then observable decision points

A case names its scene (sid, sites) and fills: SCENE_PERTURB (hidden-behaviour perturbations), PRESETS, effective()
(collapse discrete combinations that are the same scene) and, if needed, cost / outcome_class.
"""
import itertools

import numpy as np

from . import kent_bridge as kb

R_CLEAR = 0.3                 # a clear recovery: margin r at least this (the search's soft target)


class KentCase:
    name = None
    sid = None                # scene id in carla_rc's SPECS (site A); other sites are "<sid>@<key>"
    SCENE_PERTURB = []        # [{param: delta}] applied to theta_s, clipped to the range
    PRESETS = {}              # name -> overrides of the spec's nominal
    SITE_WEIGHTS = None       # {site key: weight} of the random draws; None = equal
    MIN_PER_COMBO = 1
    RANGES = {}               # range overrides of the spec (e.g. RC10 / RC11: the t_reveal band of a shared scene)

    # ---- scene ----
    def sites(self):
        S = kb.specs()
        out = {}
        for k, sp in S.items():
            if k == self.sid or k.startswith(self.sid + "@"):
                out[getattr(sp, "site_key", "A") if "@" in k else "A"] = sp
        return out

    def spec(self, site="A"):
        return self.sites()[site]

    def site_sid(self, site):
        return kb.env_sid(self.sid, site)

    def ranges(self, site="A"):
        return {**self.spec(site).ranges, **self.RANGES}

    def speed_factor(self, site, p):
        """The spec's sample() scaling of its traffic speeds (SPEEDS) for the road condition and the site."""
        from envelope import ROAD
        return ROAD[p.get("wx", "dry")]["speed"] * getattr(self.spec(site), "site_speed", 1.0)

    def finish(self, site, p):
        q = dict(p)
        for k in getattr(self.spec(site), "SPEEDS", ()):
            if k in q:
                q[k] *= self.speed_factor(site, p)
        return q

    def range_of(self, k, site, p):
        """k's range for this scene (a speed scaled as finish() scales it)."""
        lo, hi = self.ranges(site)[k]
        f = self.speed_factor(site, p) if k in getattr(self.spec(site), "SPEEDS", ()) else 1.0
        return lo * f, hi * f

    @property
    def families(self):
        return tuple(self.spec().modes)

    def space(self, fam):
        return kb.PARAM_SPACE.get(fam) or {}

    def settings(self, fam):
        return [kb.full_cfg(fam, c) for c in kb.SETTINGS.get(fam, [{}])]

    def decode(self, fam, u):
        return kb.from_unit(u, self.space(fam))

    def encode(self, fam, cfg):
        return kb.to_unit({**kb.full_cfg(fam, {}), **cfg}, self.space(fam))

    # ---- variations ----
    def effective(self, p):
        """The scene a discrete combination actually builds (combinations that build the same scene collapse)."""
        return p

    def stratified(self, n, rng):
        """n draws: every effective discrete combination at least MIN_PER_COMBO times, the rest by the spec's weights
        (a discrete value listed twice is twice as likely); continuous parameters by a Latin hypercube per combination;
        sites by SITE_WEIGHTS."""
        R = self.ranges()
        disc = {k: r for k, r in R.items() if isinstance(r, list)}
        cont = {k: r for k, r in R.items() if not isinstance(r, list)}
        keys = list(disc)
        vals = {k: list(dict.fromkeys(r)) for k, r in disc.items()}
        W = {}
        for combo in itertools.product(*[vals[k] for k in keys]):
            c = dict(zip(keys, combo))
            w = float(np.prod([disc[k].count(v) / len(disc[k]) for k, v in c.items()]))
            e = tuple(sorted(self.effective(dict(c)).items(), key=lambda kv: kv[0]))
            W[e] = W.get(e, 0.0) + w
        combos, w = list(W), np.array(list(W.values()))
        base = self.MIN_PER_COMBO * len(combos)
        extra = max(0, n - base) * w / w.sum()
        alloc = self.MIN_PER_COMBO + np.floor(extra).astype(int)
        rem = n - alloc.sum()
        for i in np.argsort(-(extra - np.floor(extra)))[:max(0, rem)]:
            alloc[i] += 1
        rows = []                             # continuous values as unit draws, mapped with the site's ranges
        for e, a in zip(combos, alloc):
            if a <= 0:
                continue
            u = {k: (rng.permutation(a) + rng.random(a)) / a for k in cont}
            for j in range(a):
                rows.append({**dict(e), **{k: float(u[k][j]) for k in cont}})
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        S = sorted(self.sites())
        sw = np.array([(self.SITE_WEIGHTS or {}).get(s, 1.0) for s in S], float)
        sites = rng.choice(S, size=len(rows), p=sw / sw.sum())
        out = []
        for s, p in zip(sites, rows):
            Rs = self.ranges(str(s))
            q = {k: (p[k] if k not in cont else float(Rs[k][0] + p[k] * (Rs[k][1] - Rs[k][0]))) for k in R}
            out.append((str(s), self.finish(str(s), q)))
        return out

    def presets(self, site="A"):
        sp = self.spec(site)
        nom = {**sp.nominal, **{k: float(np.mean(r)) for k, r in self.RANGES.items() if isinstance(r, tuple)}}
        return {name: self.finish(site, {**nom, **o}) for name, o in self.PRESETS.items()}

    def clip(self, p, site="A"):
        q = dict(p)
        for k, v in p.items():
            if isinstance(self.ranges(site).get(k), tuple):
                lo, hi = self.range_of(k, site, p)
                q[k] = float(min(max(v, lo), hi))
        return q

    def perturbed(self, th, site="A"):
        """th itself + one scene per hidden-behaviour perturbation (Victor's robust worlds)."""
        out = [th]
        for d in self.SCENE_PERTURB:
            out.append(self.clip({**th, **{k: th[k] + dv for k, dv in d.items() if k in th}}, site))
        return out

    # ---- simulation, cost, classes ----
    def simulate(self, th, cfg, fam, site="A", world=None):
        return kb.sim(self.site_sid(site), th, fam, cfg, world)

    def cost(self, o):
        """Lower is better. Contact: 1000 + 50 x impact speed; leaving the corridor: 500; otherwise a soft margin term
        towards a clear recovery, +100 when not a recovery, and a small reward for margin beyond it."""
        if o["contact"] is not None:
            dv = o["impact_speed"] if np.isfinite(o["impact_speed"]) else 20.0
            return 1000.0 + 50.0 * min(dv, 20.0)
        if o["fail"] == "left_corridor":
            return 500.0
        r = o["r"]
        return 40.0 * max(0.0, R_CLEAR - r) + (0.0 if o["success"] else 100.0) - 2.0 * min(max(r, 0.0), 1.0)

    def outcome_class(self, o):
        """success; mitigate = no contact and still in the corridor (but short of the success margin or goal), or a
        contact below 2 m/s; fail otherwise."""
        if o["success"]:
            return "success"
        if o["contact"] is None:
            return "mitigate" if o["fail"] != "left_corridor" else "fail"
        return "mitigate" if np.isfinite(o["impact_speed"]) and o["impact_speed"] < 2.0 else "fail"

    def margin(self, o):
        return o["r"] - kb.R_OK

    PASS_MARGIN = {}          # family -> margin above R_OK a PASS needs in every world (lane changes: CARLA's overshoot)

    def pass_margin(self, fam):
        return self.PASS_MARGIN.get(fam, kb.PASS_MARGIN.get(fam, 0.0))

    def companions(self, th):
        """[(name, theta_s)] the verified candidates are also run in (not part of the cost): e.g. the mirror world of
        a hidden binary outcome - a policy that recovers both uses only what is observable."""
        return []

    def hidden_worlds(self, th):
        """R5 (PIPELINE_HANDOFF): the other hidden worlds of th's history every candidate is also run in (CARLA):
        [(name, theta_s, key of its surrogate result)]. The companion worlds when the case has them (discrete hidden
        settings), else the scene perturbations that change hidden settings only."""
        comp = self.companions(th)
        if comp:
            return [(n, t, ("companion", n)) for n, t in comp]
        hid = set(getattr(self.spec(), "hidden", ()))
        P = self.perturbed(th)
        return [(f"hidden{i}", P[i], ("world", f"scene{i}")) for i, d in enumerate(self.SCENE_PERTURB, 1)
                if d and set(d) <= hid]

    def boundary_axes(self):
        """Continuous parameters a boundary sweep may move (default: every continuous one)."""
        return [k for k, r in self.ranges().items() if isinstance(r, tuple)]


class RC07Case(KentCase):
    """RC07-S5: tyre blowout ahead, then an on-ramp car squeezes in and accelerates at the ego (rc07_opts.py)."""
    name = "rc07"
    sid = "RC07-S5-opt"
    SITE_WEIGHTS = {"A": 5.0, "B": 5.0, "E1": 2.0}
    # what the other actors may do differently (hidden at the trigger): the on-ramp car cuts in earlier and floors it
    # harder; the fast car notices later and brakes softer; the rear car reacts later; the lead stops shorter
    SCENE_PERTURB = [dict(t_cut=-0.3, ramp_acc=0.4), dict(fast_react=0.3, fast_b=-0.7), dict(rear_react=0.2),
                     dict(blow_dec=0.5)]
    PRESETS = {
        "benign": dict(blowout=0),
        "easy": dict(headway=1.6, fast=0, ramp_cut=0, pile=0, rear=0, wx="dry"),
        "mid": dict(),
        "hard_ramp": dict(ramp_dx=0.0, t_cut=1.5, ramp_acc=3.5, fast_gap=25.0, fast_dv=8.0),
        "hard_rear": dict(rear=1, rear_gap=15.0, rear_react=1.2, headway=0.8),
        "closed": dict(headway=0.8, v=30.0, wx="heavy", fast=1, fast_gap=25.0, fast_dv=8.0, ramp_dx=0.0, t_cut=1.5,
                       ramp_acc=3.5, pile=1, pile_dx=30.0, rear=1, rear_gap=15.0, rear_react=1.2),
    }

    def effective(self, p):
        if "blowout" in p and not p["blowout"]:
            p["pile"] = 0                       # no blowout: the pile-up is never built
        return p


class RC10Case(KentCase):
    """RC10: the car ahead spins and slides left or right (hidden), shown early (t_reveal 0.2-0.8 s): the contingent
    family (brake, see the side, escape to the other) recovers without knowing the direction (rc10_opts.py)."""
    name = "rc10"
    sid = "RC10-S4-opt"
    SITE_WEIGHTS = {"A": 5.0, "B": 5.0, "E1": 2.0}
    RANGES = dict(t_reveal=(0.2, 0.8))
    PASS_MARGIN = {"left": 0.13, "right": 0.13, "brake_free": 0.13}      # as RC07's lane changes
    # the evidence comes later, the car stops shorter, it slides further, the rear vehicle reacts later / brakes less
    SCENE_PERTURB = [dict(t_reveal=0.15), dict(b_spin=0.7), dict(slide=0.2), dict(rear_react=0.2), dict(rear_b=-0.3)]
    PRESETS = {
        "benign": dict(spin=0),
        "mid": dict(),
        "far": dict(headway=0.6),
        "close": dict(headway=0.3, v=30.0),
        "slide_side_closed": dict(slide_dir=1, adjR=1, adjR_dx=0.0),
        "free_side_closed": dict(slide_dir=1, adjL=1, adjL_dx=0.0),
        "no_rear": dict(rear=0),
        "rear_truck_late": dict(rear_bp="vehicle.carlamotors.european_hgv", rear_gap=7.0, rear_react=1.3, rear_b=3.5),
    }

    def effective(self, p):
        if "spin" in p and not p["spin"]:
            p["slide_dir"] = 1                  # benign: the direction is never built
        return p

    def companions(self, th):
        return [("mirror", dict(th, slide_dir=-th["slide_dir"]))] if th.get("spin") else []


class RC11Case(RC10Case):
    """RC11: the same scene and surrogate, the direction shown late (t_reveal 1.0-1.6 s): each direction alone is
    recoverable, no policy using only what is observable recovers both."""
    name = "rc11"
    RANGES = dict(t_reveal=(1.0, 1.6))


class RC15Case(KentCase):
    """RC15: merge where the ego's lane ends; the adjacent driver yields, holds or matches (hidden), so yielding must be
    observed (rc15_opts.py). Companion worlds: the other two responses (a family recovering all three uses only what
    it observed)."""
    name = "rc15"
    sid = "RC15-M-opt"
    SITE_WEIGHTS = {"A": 1.0, "R": 1.0}
    PASS_MARGIN = {"merge_front": 0.13, "merge_back": 0.13, "watch_merge": 0.13}   # lane changes, as RC07's
    # the adjacent driver reacts later, it is faster, further ahead; the target lane's lead is closer
    SCENE_PERTURB = [dict(react=0.2), dict(dv_adj=0.5), dict(adj_dx=1.0), dict(lead_dx=-3.0)]
    PRESETS = {
        "yield": dict(mode="yield"),
        "yield_pronounced": dict(mode="yield", style="pronounced"),
        "hold_front": dict(mode="hold", dv_adj=-2.0, adj_dx=-3.0),
        "hold_behind": dict(mode="hold", dv_adj=1.5, lead=1, ego_lead_dx=25.0),
        "match": dict(mode="match"),
        "match_pronounced": dict(mode="match", style="pronounced", adj_dx=-3.0),
        "short_lane": dict(mode="match", end_x=0.0),          # clipped to the site's shortest
    }

    def presets(self, site="A"):
        P = super().presets(site)
        return {k: self.clip(p, site) for k, p in P.items()}

    def companions(self, th):
        return [(m, dict(th, mode=m)) for m in ("yield", "hold", "match") if m != th["mode"]]


class RC14Case(KentCase):
    """RC14: a motorbike seen earlier, hidden behind a van at the decision; fast it will arrive in the gap, slow it stays
    behind (rc14_opts.py). The policy's history includes the pre-roll. Companion world: the other bike speed with
    the same current view (a family recovering both reads its memory, or does not need it)."""
    name = "rc14"
    sid = "RC14-S1-opt"
    SITE_WEIGHTS = {"A": 5.0, "B": 5.0, "E1": 2.0}
    PASS_MARGIN = {"left": 0.13, "merge_back": 0.13, "memory_left": 0.13}       # lane changes, as RC07's
    # the bike is faster, the van slower, the stopped car closer, the bike went out of view earlier
    # the rear vehicle brakes less and reacts later (only when present: rear_b is hidden)
    SCENE_PERTURB = [dict(dv_fast=2.0), dict(dv_occ=0.5), dict(d_obs=-3.0), dict(t_since=0.1), dict(rear_b=-0.3)]
    PRESETS = {
        **{f"{w}_d{d}": dict(bike_fast=int(w == "fast"), d_obs=float(d)) for w in ("fast", "slow")
           for d in (50, 60, 70, 80, 90, 100)},                       # the d_obs band where only waiting works
        "fast_far": dict(bike_fast=1, d_obs=80.0),
        "fast_mid": dict(bike_fast=1, d_obs=62.0),
        "fast_near": dict(bike_fast=1, d_obs=45.0),
        "slow_near": dict(bike_fast=0, d_obs=45.0),
        "slow_far": dict(bike_fast=0, d_obs=80.0),
        "no_bike": dict(bike=0),
        "truck_occluder": dict(bike_fast=1, occ_bp="vehicle.carlamotors.european_hgv", d_obs=70.0),
    }

    def effective(self, p):
        if "bike" in p and not p["bike"]:
            p["bike_fast"] = 0                  # no bike: its speed is never built
        return p

    def companions(self, th):
        return [("other_bike", dict(th, bike_fast=1 - th["bike_fast"]))] if th.get("bike") else []


class RC03Case(KentCase):
    """RC03: a car next to the ego is rear-ended and spins into the ego lane (or stays out, benign); keep, brake, or
    change into the other adjacent lane's gap, plain or ready to brake (rc03_opt.py, 11 sites, dry / wet / heavy)."""
    name = "rc03"
    sid = "RC03-S1-opt"
    SITE_WEIGHTS = {"A": 2.0, "B": 2.0, "C1": 1.0, "C2": 1.0, "C1b": 1.0, "C2b": 1.0, "D2": 1.0, "D2b": 1.0,
                    "D1": 1.0, "E1": 1.0, "E2": 1.0}
    PRESET_SITES = ("A", "C1b", "D1")             # straight, long bend, tight held-out bend
    PASS_MARGIN = {f: 0.13 for f in ("left", "right", "left_brake", "right_brake")}   # lane changes, as RC07's
    # hidden at the trigger: the car slides deeper, the gap leader brakes harder, the followers react later
    SCENE_PERTURB = [dict(intrude=0.1, psi_rest=0.1), dict(lead_brake=1.5), dict(follow_react=0.2, rear_react=0.2)]
    PRESETS = {
        "benign": dict(crash=0),
        "mid": dict(),
        "near_deep": dict(d0=20.0, intrude=1.1),
        "far": dict(d0=60.0),
        "clip": dict(intrude=0.5),
        "gap_closed": dict(esc_gap=10.0),
        "lead_brakes": dict(lead_brake=5.0),
        "tailgater": dict(rear=1, rear_gap=12.0, rear_react=1.1),
        "wet": dict(wx="wet"),
        "heavy": dict(wx="heavy"),
    }
    BLUEPRINTS = ("vic_bp", "lead_bp", "fol_bp")   # drawn at random, not stratified (5 x 4 x 4 combinations)

    def ranges(self, site="A"):
        R = super().ranges(site)
        return R if getattr(self, "_all_bp", True) else {k: v for k, v in R.items() if k not in self.BLUEPRINTS}

    def stratified(self, n, rng):
        self._all_bp = False
        try:
            out = KentCase.stratified(self, n, rng)
        finally:
            self._all_bp = True
        R = self.ranges()
        return [(s, dict(p, **{k: R[k][int(rng.integers(len(R[k])))] for k in self.BLUEPRINTS})) for s, p in out]


class RC12Case(RC03Case):
    """RC12: RC03's scenes with the response held back `delay` s after the trigger (lane and speed kept, the delay
    prefix scored). Each scene is searched at every delay of DELAYS, so the set of delays that still recover is
    recorded per scene (it may be disjoint)."""
    name = "rc12"
    DELAYS = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5)
    RANGES = dict(delay=(0.0, 1.5))
    PRESETS = {k: RC03Case.PRESETS[k] for k in ("mid", "near_deep", "far", "gap_closed", "tailgater", "heavy")}

    def stratified(self, n, rng):
        base = RC03Case.stratified(self, max(1, n // len(self.DELAYS)), rng)
        return [(s, dict(p, delay=d)) for s, p in base for d in self.DELAYS]

    def presets(self, site="A"):
        return {f"{k}_d{d:g}": dict(p, delay=d) for k, p in super().presets(site).items() for d in self.DELAYS}


class RC16Case(KentCase):
    """RC16: a van double-parked beside a traffic island, seen late past the van ahead; the passage is a corridor the
    body (not only the centre point) must fit through straight (rc16_opts.py). Nothing hidden at the trigger."""
    name = "rc16"
    sid = "RC16-S1-opt"
    SITE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "H": 0.6}
    PASS_MARGIN = {"shift_pass": 0.05}
    # CARLA vs the surrogate, not the scene: the van ahead swerves a little later, the parked van stands 0.1 m further out
    SCENE_PERTURB = [dict(T_lead=0.15), dict(off=-0.1)]
    PRESETS = {
        "wide_far": dict(w_pass=3.3, lead_hw=1.6),
        "mid": dict(),
        "narrow_far": dict(w_pass=2.7, lead_hw=1.6),
        "narrow_near": dict(w_pass=2.7, lead_hw=0.6, isl_lead=3.0),
        "wide_near": dict(w_pass=3.3, lead_hw=0.6, isl_lead=3.0),
        "narrow_fast": dict(w_pass=2.7, v=12.0),
        "long_island": dict(isl_lead=12.0),
        "wet": dict(wx="wet"),
    }


class RC08Case(KentCase):
    """RC08: a garbage truck stopped in the ego lane of a two-way town street, an oncoming stream with gaps, a follower;
    hold protected, overtake now, or wait and overtake on a seen gap (rc08_opts.py). Safety is the label, route
    completion (route_done) is stored apart."""
    name = "rc08"
    sid = "RC08-S1-opt"
    SITE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "H": 0.6}
    PASS_MARGIN = {"overtake": 0.13, "wait_overtake": 0.13}        # lane changes, as RC07's
    # hidden: the truck stays longer, the follower reacts later and brakes softer, the oncoming cars come faster
    SCENE_PERTURB = [dict(t_go=15.0), dict(fol_react=0.3, fol_b=-1.0), dict(onc_v=1.5)]
    PRESETS = {
        "mid": dict(),
        "big_gap": dict(onc_x0=150.0, onc_h1=12.0),
        "short_gap": dict(onc_x0=10.0, onc_h1=3.0, onc_h2=3.0),
        "hidden_close": dict(onc_x0=-5.0, onc_h1=6.0, onc_h2=8.0),
        "leaves_soon": dict(t_go=8.0),
        "tailgater": dict(fol_T=0.8, fol_react=0.9, fol_b=4.5),
        "near_truck": dict(d_truck=30.0, v=15.0),
        "no_follower": dict(fol=0),
    }


CASES = {"rc03": RC03Case, "rc08": RC08Case, "rc16": RC16Case, "rc07": RC07Case, "rc10": RC10Case, "rc11": RC11Case, "rc12": RC12Case, "rc14": RC14Case,
         "rc15": RC15Case}
