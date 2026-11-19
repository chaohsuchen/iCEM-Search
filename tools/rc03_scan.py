"""RC03 parameter sensitivity scan (surrogate only): for each (v_ego, d0, esc_gap) configuration run a short iCEM per family (nominal cost) and report the best outcome class and clearance.
Used to choose representative easy / hard / near-boundary presets. Not a search result."""
import sys, json, numpy as np
sys.path.insert(0, ".")
from search.cases import CASES
from search.cem import icem
C = CASES["rc03"](); out = []
for v in (18.0, 22.0):
    for d0 in (22.0, 28.0, 35.0):
        for eg in (0.3, 0.6, 1.0):
            th = dict(v_ego=v, v_post=v - 2, v_esc=v, d0=d0, intrude=eg, esc_gap=40.0); row = dict(v=v, d0=d0, intrude=eg)
            for fam in ("brake", "lane_change"):
                F = C.FAM[fam]; f = lambda u: C.cost(C.simulate(th, C.decode(fam, u), fam))
                u, c, _ = icem(f, len(F["names"]), np.random.default_rng(0), pop=32, iters=7)
                r = C.simulate(th, C.decode(fam, u), fam); row[fam] = dict(cls=C.outcome_class(r), margin=round(C.margin(r), 2), rel=round(r["rel_speed"], 1), kw={k: round(x, 2) for k, x in C.decode(fam, u).items() if k in F["names"]})
            out.append(row); print(row["v"], row["d0"], row["intrude"], "| brake", row["brake"]["cls"], row["brake"]["margin"], "| lc", row["lane_change"]["cls"], row["lane_change"]["margin"], flush=True)
json.dump(out, open("results_search/rc03/scans/scan_d0_x_intrude.json", "w"), indent=1)
