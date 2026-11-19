"""Generic iCEM search (CEM / MPPI only via --opt) per maneuver family on a case's surrogate (no CARLA). Robust objective = worst cost over theta_s and its perturbations.
python tools/run_cem.py --case rc01 [--pop 48 --iters 8 --seeds 2]  ->  results_search/<case>/search/<batch>/{profiles.json,candidates.json,rollouts.pkl}"""
import argparse, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.cases import CASES
from search.cem import cem, icem, mppi

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--batch", default="baseline", help="batch name -> results_search/<rc>/search/<batch>/"); ap.add_argument("--scen-file", default=None, help="JSON {name: theta_s} replacing the case SCEN list"); ap.add_argument("--out", default=None)
ap.add_argument("--opt", default="icem", help="comma list from cem,icem,mppi (default: icem only)")
ap.add_argument("--pop", type=int, default=48); ap.add_argument("--iters", type=int, default=8); ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()
case = CASES[a.case](); out = a.out or f"results_search/{case.name}/search/{a.batch}"; os.makedirs(out, exist_ok=True)
OPT = {k: v for k, v in {"cem": cem, "icem": icem, "mppi": mppi}.items() if k in a.opt.split(",")}
if a.scen_file: case.SCEN = json.load(open(a.scen_file))
json.dump(case.SCEN, open(f"{out}/scenarios.json", "w"), indent=1)      # the scene list of this batch (batch is self-contained)
prof, cands, roll = [], [], []
t0 = time.time()
for sname, th in case.SCEN.items():
    for fam in case.families:
        dim = len(case.FAM[fam]["names"]); allu = []
        for oname, opt in OPT.items():
            bests = []
            for seed in range(a.seeds):
                log = []; rng = np.random.default_rng(seed)
                f = lambda u: max(case.cost(case.simulate(t, case.decode(fam, u), fam)) for t in case.perturbed(th))
                u, c, hist = opt(f, dim, rng, pop=a.pop, iters=a.iters, log=log)
                bests.append((c, u, hist, len(log))); allu += [x for x, _ in log]
                for uu, cc in log: roll.append((sname, fam, oname, seed, list(map(float, uu)), float(cc)))
            bi = min(range(len(bests)), key=lambda i: bests[i][0]); c, u, hist, n = bests[bi]
            kw = case.decode(fam, u); r = case.simulate(th, kw, fam)
            prof.append(dict(scenario=sname, theta_s=th, family=fam, optimizer=oname, theta_p=kw, robust_cost=c, cost_by_seed=[b[0] for b in bests], evals_per_seed=n,
                             best_seed=bi, hist=[b[2] for b in bests], cls=case.outcome_class(r), verdict=r["verdict"], min_front=r["min_front_gap"], min_rear=r["min_rear_gap"], contact=r["contact"], rel_speed=r["rel_speed"]))
            print(f"{sname:17s} {fam:12s} {oname:5s} robust {c:8.2f} {case.outcome_class(r):8s} gap {r['min_front_gap']:6.2f} {r['verdict']} "
                  f"{ {k: round(v, 2) for k, v in kw.items() if k in case.FAM[fam]['names']} }", flush=True)
        pf = [p for p in prof if p["scenario"] == sname and p["family"] == fam]; bp = min(pf, key=lambda p: p["robust_cost"])
        cands.append(dict(scenario=sname, theta_s=th, family=fam, kind="best", theta_p=bp["theta_p"], search=dict(optimizer=bp["optimizer"], seed=bp["best_seed"], pop=a.pop, iters=a.iters, robust_cost=bp["robust_cost"]), surrogate=dict(verdict=bp["verdict"], cls=bp["cls"], min_front=bp["min_front"], min_rear=bp["min_rear"])))
        bnd = None
        for u in allu[::3]:
            kw = case.decode(fam, u); r = case.simulate(th, kw, fam)
            if r["verdict"] != "PASS": continue
            m = case.margin(r)
            if m >= 0 and (bnd is None or m < bnd[0]): bnd = (m, kw, r)
        for kind, cc in [("best", bp)] + ([("boundary", dict(theta_p=bnd[1]))] if bnd else []):
            for suffix, th2 in case.companions(th):
                r2 = case.simulate(th2, cc["theta_p"], fam)
                cands.append(dict(scenario=f"{sname}__{suffix}", theta_s=th2, family=fam, kind=kind, theta_p=cc["theta_p"], surrogate=dict(verdict=r2["verdict"], cls=case.outcome_class(r2), min_front=r2["min_front_gap"], min_rear=r2["min_rear_gap"])))
        if bnd:
            cands.append(dict(scenario=sname, theta_s=th, family=fam, kind="boundary", theta_p=bnd[1], search=dict(optimizer=bp["optimizer"], pop=a.pop, iters=a.iters, note="smallest positive-margin PASS among all iCEM samples"), surrogate=dict(verdict="PASS", cls="success", min_front=bnd[2]["min_front_gap"], min_rear=bnd[2]["min_rear_gap"], margin=bnd[0])))
json.dump(prof, open(f"{out}/profiles.json", "w"), indent=1); json.dump(cands, open(f"{out}/candidates.json", "w"), indent=1); pickle.dump(roll, open(f"{out}/rollouts.pkl", "wb"))
print(f"done in {time.time() - t0:.0f}s, {len(roll)} rollouts, {len(cands)} candidates")
