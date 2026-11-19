"""Generic recovery-boundary exploration on a case's surrogate (no CARLA). For every axis of a config, per family: sweep one scene parameter from the easy end to the hard end,
re-optimising at each point with iCEM warm-started from the previous point (kind=best), and also evaluating the policy found at the easy end unchanged (kind=ref, a fixed-policy
flip). Where the optimised verdict flips PASS -> not PASS between neighbours, the interval is bisected (--rounds). Output: results_search/<rc>/search/<batch>/
{scenarios.json, candidates.json, table.json}; verify with tools/verify_carla.py (--case <rc> --batch <batch>).
Config: tools/boundary_configs/<rc>.json  = {"axes": [{"axis": name, "key": scene param, "grid": [easy ... hard], "base": {other scene params}, "families": [...] or omitted = all}]}
python tools/run_boundary.py --case rc01 --batch boundary_r1"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.cases import CASES
from search.cem import icem

ap = argparse.ArgumentParser(); ap.add_argument("--case", required=True); ap.add_argument("--batch", required=True); ap.add_argument("--config")
ap.add_argument("--pop", type=int, default=32); ap.add_argument("--iters", type=int, default=6); ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
C = CASES[a.case](); cfg = json.load(open(a.config or f"tools/boundary_configs/{a.case}.json"))
OUT = f"results_search/{a.case}/search/{a.batch}"; os.makedirs(OUT, exist_ok=True)
json.dump(dict(config=cfg, pop=a.pop, iters=a.iters, bisect_rounds=a.rounds), open(f"{OUT}/scenarios.json", "w"), indent=1)
cands, table = [], []


def optimise(th, fam, mu0):
    f = lambda u: max(C.cost(C.simulate(t, C.decode(fam, u), fam)) for t in C.perturbed(th))
    return icem(f, len(C.FAM[fam]["names"]), np.random.default_rng(0), pop=a.pop, iters=a.iters, init_mu=mu0, init_std=0.2 if mu0 is not None else 0.3)[:2]


def sur(r): return dict(verdict=r["verdict"], cls=C.outcome_class(r), min_front=r["min_front_gap"], min_rear=r["min_rear_gap"], margin=float(C.margin(r)))


def record(axis, fam, key, x, th, kw, kind, c=None, warm=None):
    r = C.simulate(th, kw, fam)
    row = dict(axis=axis, family=fam, key=key, x=x, kind=kind, theta_s=th, theta_p=kw, verdict=r["verdict"], cls=C.outcome_class(r), margin=float(C.margin(r)), reason=r["reason"], robust_cost=c, warm=warm)
    table.append(row)
    cands.append(dict(scenario=f"bnd_{axis}_{x:g}", theta_s=th, family=fam, kind=kind, theta_p=kw, search=dict(optimizer="icem", pop=a.pop, iters=a.iters, seed=0, robust_cost=c, warm_start=warm), surrogate=sur(r)))
    if kind == "best":
        for suffix, th2 in C.companions(th):
            r2 = C.simulate(th2, kw, fam)
            cands.append(dict(scenario=f"bnd_{axis}_{x:g}__{suffix}", theta_s=th2, family=fam, kind=kind, theta_p=kw, surrogate=sur(r2)))
    return row


t0 = time.time()
for ax in cfg["axes"]:
    axis, key, base = ax["axis"], ax["key"], ax.get("base", {})
    for fam in ax.get("families") or list(C.families):
        grid = list(ax["grid"]); pts = {}; ref = None; mu = None; th_of = lambda x: {**base, key: x}

        def run_point(x, mu_from):
            th = th_of(x); u, c = optimise(th, fam, mu_from); row = record(axis, fam, key, x, th, C.decode(fam, u), "best", c, None if mu_from is None else "neighbour"); pts[x] = (u, row); return u, row
        for x in grid:
            u, row = run_point(x, mu)
            if mu is None: ref = row["theta_p"]
            else: record(axis, fam, key, x, th_of(x), ref, "ref")
            mu = u; print(axis, fam, x, row["verdict"], row["cls"], f"margin {row['margin']:.2f}", row["reason"][:40], flush=True)
        for rnd in range(a.rounds):
            xs = [g for g in grid if g in pts]
            flip = next(((p, q) for p, q in zip(xs, xs[1:]) if pts[p][1]["verdict"] == "PASS" and pts[q][1]["verdict"] != "PASS"), None)
            if not flip: break
            p, q = flip; x = round(0.5 * (p + q), 3); u, row = run_point(x, pts[p][0]); record(axis, fam, key, x, th_of(x), ref, "ref")
            grid.insert(grid.index(p) + 1, x); print(axis, fam, "bisect", x, row["verdict"], f"margin {row['margin']:.2f}", flush=True)
json.dump(cands, open(f"{OUT}/candidates.json", "w"), indent=1); json.dump(table, open(f"{OUT}/table.json", "w"), indent=1)
print(f"done in {time.time() - t0:.0f}s, {len(cands)} candidates")
