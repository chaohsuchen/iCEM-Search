"""Generic parameter sensitivity scan on a case's surrogate: full grid over the given axes, a short nominal iCEM per family at every grid point.
Prints/saves, per point and family, the best outcome class and margin, so that representative easy / hard / near-boundary presets can be chosen.
python tools/scan_params.py --case rc01 --tag gap_v_delay --axes '{"gap":[8,12,16],"v_ego":[8.1,9.5],"delay_s":[0,0.3]}'  ->  results_search/<rc>/scans/scan_<tag>.json"""
import argparse, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.cases import CASES
from search.cem import icem
ap = argparse.ArgumentParser(); ap.add_argument("--case", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--axes", required=True)
ap.add_argument("--base", default="{}"); ap.add_argument("--pop", type=int, default=24); ap.add_argument("--iters", type=int, default=6); a = ap.parse_args()
C = CASES[a.case](); axes = json.loads(a.axes); base = json.loads(a.base); out = []; t0 = time.time()
for vals in itertools.product(*axes.values()):
    th = {**base, **dict(zip(axes, vals))}; row = dict(theta_s=th)
    for fam in C.families:
        f = lambda u: C.cost(C.simulate(th, C.decode(fam, u), fam))
        u, c, _ = icem(f, len(C.FAM[fam]["names"]), np.random.default_rng(0), pop=a.pop, iters=a.iters)
        r = C.simulate(th, C.decode(fam, u), fam); row[fam] = dict(cls=C.outcome_class(r), verdict=r["verdict"], margin=round(float(C.margin(r)), 2), theta_p={k: (round(float(x), 3) if not isinstance(x, str) else x) for k, x in C.decode(fam, u).items()})
    out.append(row); print(" ".join(f"{k}={v}" for k, v in th.items()), "|", " ".join(f"{fam}:{row[fam]['cls'][:4]}{row[fam]['margin']:+.1f}" for fam in C.families), flush=True)
os.makedirs(f"results_search/{a.case}/scans", exist_ok=True); json.dump(dict(axes=axes, base=base, rows=out), open(f"results_search/{a.case}/scans/scan_{a.tag}.json", "w"), indent=1)
print(f"done in {time.time() - t0:.0f}s")
