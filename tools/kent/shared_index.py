"""R4 index: which CARLA runs are the family's shared setting (run under kind best / boundary when it coincides).
writes <rc>/carla/<batch>/shared_runs.json {run_id: family}; prints (scene, family) coverage."""
import glob, json, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from search import kent_bridge as kb
from search.kent_cases import CASES
from search.pipeline import _key
OUT = kb.OUT
for rc, b in [a.split(":") for a in sys.argv[1:]]:
    case = CASES[rc]()
    k0 = {f: _key(case.settings(f)[0]) for f in case.families}
    R = [json.loads(l) for f in glob.glob(f"{OUT}/{rc}/carla/{b}/results_shard*.jsonl") for l in open(f) if l.strip()]
    idx, cov = {}, collections.defaultdict(set)
    for r in R:
        if _key(kb.full_cfg(r["family"], r["theta_p"])) == k0[r["family"]]:
            idx[r["id"]] = r["family"]
            cov[r["variation_id"]].add(r["family"])
    json.dump(idx, open(f"{OUT}/{rc}/carla/{b}/shared_runs.json", "w"), indent=0)
    scenes = {r["variation_id"] for r in R}
    full = sum(cov[v] == set(case.families) for v in scenes)
    miss = collections.Counter(f for v in scenes for f in set(case.families) - cov[v])
    print(f"{rc} {b}: {len(idx)} shared runs; scenes (incl. hidden worlds) with every family's shared setting {full}/{len(scenes)}; missing by family {dict(miss)}")
