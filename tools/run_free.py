"""RC04 free-form pilot, surrogate stage (docs/FREE_SEARCH_PROPOSAL.md). For one scene of the pilot manifest: iCEM over the 12 control-point parameters of the `free` family (robust cost =
worst over the scene and its perturbations, + smoothness penalty), then selection of diverse candidates and a surrogate replay in the declared hidden worlds. No CARLA.
python tools/run_free.py --scene near_obstacle [--pop 160 --iters 9 --seeds 2]       ->  results_free/rc04/search/<batch>/<scene>.json
python tools/run_free.py --merge                                                       ->  results_free/rc04/search/<batch>/candidates.json (input of tools/verify_carla.py)

Strata of the candidates (key `kind`): best | diverse | boundary | refuted | mitigation (iCEM samples, selected on the surrogate) and random (uniform prior samples, never optimised). Diversity is measured
on the executed behaviour (speed and lateral position at t = 1..7 s), not on control-point distance."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.cases import CASES
from search.cem import icem
from rcx import free_policy as FP

ap = argparse.ArgumentParser(); ap.add_argument("--scene"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--merge-struct", action="store_true"); ap.add_argument("--batch", default="free_pilot")
ap.add_argument("--manifest", default="results_free/rc04/manifest.json")
ap.add_argument("--pop", type=int, default=160); ap.add_argument("--iters", type=int, default=9); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--warm", action="store_true", help="also run iCEM warm-started from the control-point projection of each structured family's best policy (results_free/rc04/search/struct_matched)"); ap.add_argument("--warm-pop", type=int, default=60); ap.add_argument("--warm-iters", type=int, default=8); ap.add_argument("--warm-std", type=float, default=0.12)
ap.add_argument("--n-diverse", type=int, default=3); ap.add_argument("--n-random", type=int, default=4); ap.add_argument("--tau", type=float, default=0.5, help="min behaviour distance between diverse candidates")
a = ap.parse_args()
M = json.load(open(a.manifest)); OUT = f"results_free/rc04/search/{a.batch}"; os.makedirs(OUT, exist_ok=True)


def worlds_for(th):
    """Declared compatible hidden worlds (manifest); with no follower the worlds that differ only in the follower reaction are identical and dropped."""
    W = {k: dict(v) for k, v in M["worlds"].items()}
    if th.get("fol", 1) == 0:
        seen, keep = [], {}
        for k, v in W.items():
            key = {x: y for x, y in v.items() if x != "fol_react"}
            if key not in seen: seen.append(key); keep[k] = v
        W = keep
    return W


def descriptor(r):
    R = np.array(r["rows"]); idx = [min(len(R) - 1, int(round(t / 0.05)) - 1) for t in range(1, 8)]
    return np.concatenate([R[idx, 1] / 5.0, R[idx, 2] / 1.0])


def dist(d1, d2): return float(np.linalg.norm(d1 - d2) / np.sqrt(len(d1)))


def sur(C, r, extra=None):
    return dict(verdict=r["verdict"], cls=C.outcome_class(r), min_front=r["min_front_gap"], min_rear=r["min_rear_gap"], margin=float(C.margin(r)), reason=r["reason"], rel_speed=r["rel_speed"], **(extra or {}))


def peak_jerk(r):
    A = np.array(r["rows"])[:, 7]; f = np.convolve(A, np.ones(4) / 4, mode="valid"); return float(np.abs(np.diff(f)).max() / 0.05) if len(f) > 1 else 0.0


if a.merge_struct:       # matched-budget structured families (tools/run_cem.py per scene): add scene / role / hidden-world replays, one candidate file for tools/verify_carla.py
    C = CASES["rc04"](); cands = []
    for sc, spec in M["scenes"].items():
        d = f"results_free/rc04/search/struct_matched/{sc}"
        for c in json.load(open(f"{d}/candidates.json")):
            if c["scenario"] != sc: continue
            c.update(scene=sc, role=spec["role"], stratum="icem_selected", world_id="w0"); c["search"]["seeds"] = 2; cands.append(c)
            if c["surrogate"]["verdict"] == "PASS":
                for w, ov in worlds_for(spec["theta_s"]).items():
                    if w == "w0": continue
                    th2 = {**c["theta_s"], **ov}; r2 = C.simulate(th2, c["theta_p"], c["family"])
                    cands.append(dict(scenario=f"{sc}__{w}", scene=sc, role=spec["role"], theta_s=th2, family=c["family"], kind=c["kind"], stratum="world_replay", theta_p=c["theta_p"], world_id=w, search=dict(note="same policy as the w0 candidate"), surrogate=sur(C, r2)))
    json.dump(cands, open(f"results_free/rc04/search/struct_matched/candidates.json", "w"), indent=1); print(len(cands), "structured candidates"); sys.exit()
if a.merge:
    cands = []
    for sc in M["scenes"]:
        p = f"{OUT}/{sc}.json"
        if os.path.exists(p): cands += json.load(open(p))["candidates"]
    for i, c in enumerate(cands): c["index"] = i
    json.dump(cands, open(f"{OUT}/candidates.json", "w"), indent=1); print(len(cands), "candidates"); sys.exit()

C = CASES["rc04"](); sname = a.scene; th = M["scenes"][sname]["theta_s"]; t0 = time.time()
cost_of = lambda r: C.cost(r)
f = lambda u: max(cost_of(C.simulate(t, FP.decode(u), "free")) for t in C.perturbed(th)) + FP.SMOOTH_W * FP.smoothness_penalty(FP.decode(u))
log, bests = [], []
for seed in range(a.seeds):
    l = []; u, c, hist = icem(f, 12, np.random.default_rng(seed), pop=a.pop, iters=a.iters, log=l); bests.append(dict(seed=seed, cost=c, hist=hist, n=len(l))); log += l
origin = {}
n_cold = len(log)
for u, _ in log: origin[tuple(np.round(np.asarray(FP.decode(u)["a"] + FP.decode(u)["lam"]), 4))] = "cold"
n_warm = 0
if a.warm:
    from rcx.policies import EscapePolicy
    for pr in json.load(open(f"results_free/rc04/search/struct_matched/{sname}/profiles.json")):
        kw = dict(pr["theta_p"]); kw.setdefault("v_target", 22.0); kw.setdefault("esc_dir", -1.0)
        pol = EscapePolicy(0.05, **kw); pol.reset(); mu0 = []; lam0 = []
        for j in range(FP.K):
            tj = FP.T_START + j * FP.DK; pol.reset(); out = None
            for n in range(int(round(tj / 0.05)) + 1): out = pol.step(dict(t=n * 0.05, v=22.0, s=0.0, ego_t=0.0, cue=True, hazard=True))
            mu0.append(out.a_req); lam0.append((out.t_des or 0.0) / (kw["esc_dir"] * 3.5))
        x0 = np.concatenate([np.clip(mu0, FP.A_LO, FP.A_HI), np.clip(lam0, FP.LAM_LO, FP.LAM_HI)]); u0 = (x0 - np.asarray(FP.LO)) / (np.asarray(FP.HI) - np.asarray(FP.LO))
        l = []; icem(f, 12, np.random.default_rng(50 + len(bests)), pop=a.warm_pop, iters=a.warm_iters, init_mu=u0, init_std=a.warm_std, log=l); n_warm += len(l); log += l
        for u, _ in l: origin.setdefault(tuple(np.round(np.asarray(FP.decode(u)["a"] + FP.decode(u)["lam"]), 4)), "warm_" + pr["family"])
n_f = len(log)
# nominal replay of every sampled vector -> pool
pool = {}
for u, c in log:
    kw = FP.decode(u); key = tuple(np.round(kw["a"] + kw["lam"], 4))
    if key in pool: continue
    r = C.simulate(th, kw, "free"); pool[key] = dict(origin=origin.get(key, "cold"), kw=kw, cost=float(c), r=r, d=descriptor(r), pen=FP.smoothness_penalty(kw))
P = list(pool.values())
sup = sorted([x for x in P if x["r"]["verdict"] == "PASS"], key=lambda x: x["cost"])
chosen = []
if sup:
    chosen.append(("best", sup[0]))
    near = [x for x in sup if x["cost"] <= sup[0]["cost"] + 10.0]; sel = [sup[0]]
    for _ in range(a.n_diverse):
        cand = [(min(dist(x["d"], s["d"]) for s in sel), x) for x in near if x not in sel]
        if not cand: break
        dm, x = max(cand, key=lambda t: t[0])
        if dm < a.tau: break
        x["div"] = dm; sel.append(x); chosen.append(("diverse", x))
    bnd = min(sup, key=lambda x: C.margin(x["r"]))
    if all(bnd is not x for _, x in chosen): chosen.append(("boundary", bnd))
else:
    chosen.append(("best", min(P, key=lambda x: x["cost"])))          # nothing supported: the lowest-cost (still failing) sample, labelled by its CARLA outcome
nofail = [x for x in P if x["r"]["verdict"] != "PASS" and not x["r"]["contact"]]
if nofail: chosen.append(("refuted", max(nofail, key=lambda x: C.margin(x["r"]))))
cont = [x for x in P if x["r"]["contact"]]
if cont: chosen.append(("mitigation", min(cont, key=lambda x: x["r"]["rel_speed"])))
rng = np.random.default_rng(1000 + sorted(M["scenes"]).index(sname)); rnd = []
for _ in range(4000):
    kw = FP.decode(rng.random(12)); r = C.simulate(th, kw, "free"); rnd.append(dict(origin="uniform_prior", kw=kw, r=r, cost=float(cost_of(r)), d=descriptor(r), pen=FP.smoothness_penalty(kw)))
rp = [x for x in rnd if x["r"]["verdict"] == "PASS"]; rf = [x for x in rnd if x["r"]["verdict"] != "PASS"]
half = a.n_random // 2
for x in [rp[i] for i in rng.permutation(len(rp))[:half]] + [rf[i] for i in rng.permutation(len(rf))[:a.n_random - min(half, len(rp))]]: chosen.append(("random", x))
cands = []
for kind, x in chosen:
    spec = dict(family="free", version=FP.TrajectoryPolicy.version, K=FP.K, knot_times_s=[round(x["kw"]["t_start"] + j * x["kw"]["dk"], 3) for j in range(FP.K)], a_req_mps2=x["kw"]["a"], lateral_fraction=x["kw"]["lam"],
                lateral_target_m=[x["kw"]["lam"][j] * -float(th.get("side", 1)) * 3.5 for j in range(FP.K)], lateral_reference="scene lane frame, t positive to the right, 0 = ego lane centre; target = esc_dir * lane_w * fraction, esc_dir = -side",
                interpolation="a_req piecewise-linear; lateral cosine-eased", prefix_duration_s=(FP.K - 1) * x["kw"]["dk"], continuation="hold last lateral target; speed regulator at the hand-over speed; hand brake once stopped",
                bounds=dict(a_req=[FP.A_LO, FP.A_HI], lateral_fraction=[FP.LAM_LO, FP.LAM_HI], max_da_per_knot=FP.DA_MAX, max_dlam_per_knot=FP.DLAM_MAX), smoothness_weight=FP.SMOOTH_W)
    ws = {w: sur(C, C.simulate({**th, **ov}, x["kw"], "free")) for w, ov in worlds_for(th).items()}
    cands.append(dict(scenario=sname, scene=sname, role=M["scenes"][sname]["role"], theta_s=th, family="free", kind=kind, stratum="random_prior" if kind == "random" else "icem_selected", theta_p=x["kw"], policy_spec=spec,
                      search=dict(origin=x["origin"], optimizer="icem" if kind != "random" else "uniform_prior", pop=a.pop, iters=a.iters, seeds=a.seeds, robust_cost=x["cost"] if kind != "random" else None, smoothness_penalty=x["pen"], behaviour_distance_to_selected=x.get("div")),
                      surrogate=sur(C, x["r"], dict(peak_jerk_mps3=peak_jerk(x["r"]), worlds={w: s["verdict"] for w, s in ws.items()})), world_id="w0"))
    robust = all(s["verdict"] == "PASS" for s in ws.values())
    cands[-1]["surrogate"]["robust_over_worlds"] = robust
    if x["r"]["verdict"] == "PASS" and kind in ("best", "diverse", "boundary"):                      # same policy in every other declared world (CARLA replay candidates)
        for w, ov in worlds_for(th).items():
            if w == "w0": continue
            cands.append(dict(scenario=f"{sname}__{w}", scene=sname, role=M["scenes"][sname]["role"], theta_s={**th, **ov}, family="free", kind=kind, stratum="world_replay", theta_p=x["kw"], policy_spec=spec, world_id=w,
                              search=dict(optimizer="icem", note="same complete policy as the w0 candidate of this kind"), surrogate=sur(C, C.simulate({**th, **ov}, x["kw"], "free"))))
out = dict(scene=sname, theta_s=th, role=M["scenes"][sname]["role"], budget=dict(pop=a.pop, iters=a.iters, seeds=a.seeds, warm=a.warm, robust_evals_cold=n_cold, robust_evals_warm=n_warm, robust_evals=n_f, rollouts=n_f * 5, unique_samples=len(P), per_seed=bests),
           pool=dict(n_supported_nominal=len(sup), n_refuted_no_contact=len(nofail), n_contact=len(cont)), wall_s=time.time() - t0, candidates=cands)
json.dump(out, open(f"{OUT}/{sname}.json", "w"), indent=1, default=float)
print(sname, "evals", n_f, "supported", len(sup), "of", len(P), "kept", [k for k, _ in chosen], f"{time.time() - t0:.0f}s")
