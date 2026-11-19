"""Assemble the recovery-profile dataset + optimizer comparison from results_search/<rc>/search/baseline/ and results_search/<rc>/carla/baseline/.
Outputs: results_search/summary/recovery_profiles.json  (one row per case x theta_s x family: best theta_p, surrogate class, CARLA-verified outcomes)
         results_search/summary/rollouts.npz            (every surrogate evaluation: case, scenario, family, optimizer, seed, theta_p (unit box), robust cost)
         results_search/summary/OPTIMIZER_COMPARISON.md (CEM vs iCEM vs MPPI)"""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = "results_search"; cases = [c for c in ("rc01", "rc05", "rc09", "rc13", "rc03") if os.path.exists(f"{R}/{c}/search/baseline/profiles.json")]
rows, rolls, cmp_ = [], [], {}
for c in cases:
    P = json.load(open(f"{R}/{c}/search/baseline/profiles.json")); C = json.load(open(f"{R}/{c}/search/baseline/candidates.json"))
    V = {}
    d = f"{R}/{c}/carla/baseline"
    for f in glob.glob(f"{d}/results_shard*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "carla" in r: V[(r["scenario"], r["family"], r["kind"].split("_")[0], json.dumps(r["theta_p"], sort_keys=True))] = r["carla"]
    for sc in dict.fromkeys(p["scenario"] for p in P):
        for fam in dict.fromkeys(p["family"] for p in P):
            q = [p for p in P if p["scenario"] == sc and p["family"] == fam]
            if not q: continue
            b = min(q, key=lambda p: p["robust_cost"])
            ver = []
            for cd in C:
                if cd["family"] == fam and cd["scenario"].split("__")[0] == sc:
                    v = V.get((cd["scenario"], fam, cd["kind"], json.dumps(cd["theta_p"], sort_keys=True)))
                    ver.append(dict(scenario=cd["scenario"], kind=cd["kind"], theta_s=cd["theta_s"], theta_p=cd["theta_p"], surrogate=cd["surrogate"], carla=v))
            rows.append(dict(case=c, scenario=sc, theta_s=b["theta_s"], family=fam, best_optimizer=b["optimizer"], best_theta_p=b["theta_p"], robust_cost=b["robust_cost"],
                             surrogate_class=b["cls"], surrogate_verdict=b["verdict"], verified=ver))
    for p in P:
        k = (c, p["family"], p["optimizer"]); cmp_.setdefault(k, []).append(p)
    for (sn, fam, on, seed, u, cost) in pickle.load(open(f"{R}/{c}/search/baseline/rollouts.pkl", "rb")):
        rolls.append((c, sn, fam, on, seed, u, cost))
json.dump(rows, open(f"{R}/summary/recovery_profiles.json", "w"), indent=1)
mx = max(len(r[5]) for r in rolls)
np.savez_compressed(f"{R}/summary/rollouts.npz", case=np.array([r[0] for r in rolls]), scenario=np.array([r[1] for r in rolls]), family=np.array([r[2] for r in rolls]), optimizer=np.array([r[3] for r in rolls]),
                    seed=np.array([r[4] for r in rolls]), theta_unit=np.array([r[5] + [np.nan] * (mx - len(r[5])) for r in rolls], float), cost=np.array([r[6] for r in rolls], float))
L = ["# Optimizer comparison (surrogate, equal budget)", "", "Per case/family/optimizer, over all theta_s: mean best robust cost (capped at 1000), success-class rate, mean iterations to first feasible (cost < 100), mean evals per seed.", "",
     "| case | family | optimizer | mean best cost | success rate | iters to feasible | evals/seed |", "|---|---|---|---|---|---|---|"]
for (c, fam, on), q in sorted(cmp_.items()):
    costs = [min(p["robust_cost"], 1000.0) for p in q]; succ = np.mean([p["cls"] == "success" for p in q])
    it = [next((i + 1 for i, h in enumerate(hs) if h < 100.0), np.nan) for p in q for hs in p["hist"]]
    L.append(f"| {c} | {fam} | {on} | {np.mean(costs):.1f} | {succ:.0%} | {np.nanmean(it) if not np.all(np.isnan(it)) else float('nan'):.2f} | {int(np.mean([p['evals_per_seed'] for p in q]))} |")
open(f"{R}/summary/OPTIMIZER_COMPARISON.md", "w").write("\n".join(L) + "\n")
print(f"{len(rows)} profile rows, {len(rolls)} rollouts; wrote {R}/summary/ (recovery_profiles.json, rollouts.npz, OPTIMIZER_COMPARISON.md)")
