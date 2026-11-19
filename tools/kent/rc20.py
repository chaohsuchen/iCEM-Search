"""RC20 (an incomplete library must stay unresolved) on Victor's RC04 scene: a stopped car ahead, a truck alongside in
the escape lane, its follower. On new RC04 variations, search coverage is compared:

    restricted   iCEM over the families brake + lane_change (immediate change)                 (family library v1)
    full         + the staged family (brake t_slow, then change behind the truck)              (family library v2)
    planner      iCEM over one generic two-phase plan `plan`: a longitudinal phase of free length (0-3 s) and sign
                 (a_slow -3..8: accelerate or brake), then a lane change. It is not told to slow first; staging is one
                 corner of its space (t_slow > 0, a_slow > 0), the immediate change another (t_slow ~ 0).
    fixed        the library as fixed settings: one setting per family (brake, lane_change, staged), the candidate of
                 Victor's RC04 batches (baseline, presets_main, boundary_r2) recovering the most of their scenes in
                 the surrogate; no per-scene search.

iCEM as tools/run_cem.py (pop 48, 8 iterations, 2 seeds, worst cost over case.perturbed). The family bounds are
Victor's (search/cases.py RC04Case.FAM); the code here only adds `plan` in a subclass. Every policy then runs in CARLA:
    python tools/verify_carla.py --case rc04 --cands results_search/rc20/search/rc20/candidates.json \\
        --out results_search/rc20/carla/rc20 --port P --shard i --nshards n

    python tools/kent/rc20.py search [--n 40 --n-band 16 --seed 20 --workers 8]  -> results_search/rc20/search/rc20/
    python tools/kent/rc20.py label                                  -> results_search/rc20/labels.json

Labels per scene and library version (the version is stored with every label): R_REF when a policy of that version
recovers in CARLA; U_LIBRARY when it does not but another version does; UNRESOLVED when an evaluation the label needs
is missing (NaN, never a failure); NONE_AT_BUDGET when no version recovers.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                       # the case reads results_v3/ relative to the repository
from search.cases import RC04Case  # noqa: E402
from search.cem import icem  # noqa: E402

OUT = os.path.join(ROOT, "results_search", "rc20")
SD = os.path.join(OUT, "search", "rc20")
CD = os.path.join(OUT, "carla", "rc20")
SRC = ["baseline", "presets_main", "boundary_r2"]
RESTRICTED, STAGED, PLAN = ("brake", "lane_change"), "staged", "plan"
VERSIONS = {"restricted": ("search", RESTRICTED), "full": ("search", RESTRICTED + (STAGED,)), "planner": ("search", (PLAN,)),
            "fixed_restricted": ("fixed", RESTRICTED), "fixed_full": ("fixed", RESTRICTED + (STAGED,))}
RANGES = dict(d_obs=(36.0, 54.0), side_pos=(-10.0, 26.0), v_side=(19.0, 25.0), fol_gap=(6.0, 30.0))
# the band where Victor's CARLA runs needed the staged plan (boundary_r2: only `staged` recovers at d_obs 49-50 m with the
# truck alongside); the surrogate is optimistic about braking there, so the contrast shows in CARLA
BAND = dict(d_obs=(46.0, 54.0), side_pos=(-4.0, 4.0), v_side=(21.0, 23.0), fol_gap=(14.0, 26.0))
POP, ITERS, SEEDS = 48, 8, 2


class RC20Case(RC04Case):
    FAM = dict(RC04Case.FAM, plan=dict(names=["t_react", "t_slow", "a_slow", "t_lc", "a_lc", "a_post"],
                                       lo=[0.1, 0.0, -3.0, 1.5, -3.0, 0.0], hi=[1.0, 3.0, 8.0, 4.0, 2.0, 6.0],
                                       fixed=dict(mode="staged")))


case = RC20Case()


def scenes(n, seed, n_band=0):
    """n scenes over RANGES (seed), then n_band over BAND (seed + 1, always with the follower)."""
    out = {}
    for ranges, k, sd, tag in ((RANGES, n, seed, "s"), (BAND, n_band, seed + 1, "band")):
        rng = np.random.default_rng(sd)
        for i in range(k):
            th = {key: round(float(rng.uniform(*r)), 2) for key, r in ranges.items()}
            th["fol"] = int(rng.random() < 0.75) if tag == "s" else 1
            out[f"rc20_{tag}{sd}_{i:03d}"] = th
    return out


def fixed_library():
    """One setting per family: the candidate of Victor's RC04 batches recovering the most of their hazard scenes."""
    C = [c for b in SRC for c in json.load(open(f"results_search/rc04/search/{b}/candidates.json"))]
    S = list({json.dumps(c["theta_s"], sort_keys=True): c["theta_s"] for c in C}.values())
    S = [s for s in S if int(s.get("obs_lane", 0)) == 0]
    lib = {}
    for fam in RESTRICTED + (STAGED,):
        best = None
        for c in (c for c in C if c["family"] == fam):
            R = [case.simulate(s, c["theta_p"], fam) for s in S]
            k = (sum(r["verdict"] == "PASS" for r in R), float(np.mean([case.margin(r) for r in R])))
            if best is None or k > best[0]:
                best = (k, c)
        (n, m), c = best
        lib[fam] = dict(theta_p=c["theta_p"], source=f"rc04/{c['scenario']}/{c['kind']}", recovers=f"{n}/{len(S)}", mean_margin=m)
    return lib


def version_ids(lib):
    """A short id per library version: the families it offers, the bounds searched, or the fixed settings."""
    ids = {}
    for v, (kind, fams) in VERSIONS.items():
        blob = {f: (lib[f]["theta_p"] if kind == "fixed" else case.FAM[f]) for f in fams}
        ids[v] = f"{v}-" + hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return ids


def sur(th, fam, kw):
    r = case.simulate(th, kw, fam)
    return dict(verdict=r["verdict"], cls=case.outcome_class(r), margin=float(case.margin(r)), contact=r["contact"],
                min_front=r["min_front_gap"], min_rear=r["min_rear_gap"])


def search_one(job):
    name, th = job
    out = []
    for fam in RESTRICTED + (STAGED, PLAN):
        dim = len(case.FAM[fam]["names"])
        f = lambda u: max(case.cost(case.simulate(t, case.decode(fam, u), fam)) for t in case.perturbed(th))  # noqa: E731
        bests = []
        for seed in range(SEEDS):
            u, c, _ = icem(f, dim, np.random.default_rng(seed), pop=POP, iters=ITERS, log=[])
            bests.append((c, u, seed))
        c, u, seed = min(bests, key=lambda b: b[0])
        kw = case.decode(fam, u)
        out.append(dict(scenario=name, theta_s=th, family=fam, kind="best", theta_p=kw,
                        search=dict(optimizer="icem", seed=seed, pop=POP, iters=ITERS, robust_cost=float(c)),
                        surrogate=sur(th, fam, kw), rc20=dict(arm="search")))
    return out


def cmd_search(a):
    os.makedirs(SD, exist_ok=True)
    S = scenes(a.n, a.seed, a.n_band)
    lib = fixed_library()
    ids = version_ids(lib)
    print("fixed library", {f: (lib[f]["recovers"], lib[f]["source"]) for f in lib}, ids)
    C = []
    for name, th in S.items():
        for fam in RESTRICTED + (STAGED,):
            C.append(dict(scenario=name, theta_s=th, family=fam, kind="fixed", theta_p=lib[fam]["theta_p"],
                          search=dict(source=lib[fam]["source"]), surrogate=sur(th, fam, lib[fam]["theta_p"]),
                          rc20=dict(arm="fixed")))
    with Pool(a.workers) as p:
        for out in p.imap(search_one, list(S.items())):
            C += out
            print(out[0]["scenario"], {o["family"]: o["surrogate"]["verdict"] for o in out}, flush=True)
    for c in C:                                       # the versions this candidate belongs to
        c["rc20"]["versions"] = [ids[v] for v, (kind, fams) in VERSIONS.items()
                                 if kind == c["rc20"]["arm"] and c["family"] in fams]
    json.dump(S, open(f"{SD}/scenarios.json", "w"), indent=1)
    json.dump(dict(versions={v: dict(id=ids[v], kind=k, families=list(f)) for v, (k, f) in VERSIONS.items()},
                   fixed=lib, plan_bounds=case.FAM[PLAN], ranges=RANGES, band=BAND, n=a.n, n_band=a.n_band, seed=a.seed, pop=POP, iters=ITERS, seeds=SEEDS),
              open(f"{SD}/library.json", "w"), indent=1)
    json.dump(C, open(f"{SD}/candidates.json", "w"), indent=1)
    print(f"{len(S)} scenes, {len(C)} candidates -> {SD}")
    summary(C, "surrogate", lambda c: c["surrogate"]["verdict"] == "PASS")


def summary(C, what, ok):
    rec = {}
    for c in C:
        rec.setdefault(c["scenario"], {}).setdefault((c["rc20"]["arm"], c["family"]), ok(c))
    n = Counter()
    for s, d in rec.items():
        for v, (kind, fams) in VERSIONS.items():
            n[v] += any(d.get((kind, f)) for f in fams)
    print(what, "scenes recovered per version:", dict(n), "of", len(rec))


def cmd_label(a):
    C = json.load(open(f"{SD}/candidates.json"))
    L = json.load(open(f"{SD}/library.json"))
    R = {}
    for f in glob.glob(f"{CD}/results_shard*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            R[r["id"]] = r
    res, plan_kw = {}, {}
    for i, c in enumerate(C):
        r = R.get(f"{i:03d}_{c['scenario']}_{c['family']}_{c['kind']}")
        ok = None if r is None or "carla" not in r or r["carla"]["label"] not in ("PASS", "FAIL") else r["carla"]["label"] == "PASS"
        c["carla_ok"] = ok
        res[(c["scenario"], c["rc20"]["arm"], c["family"])] = ok
        if c["family"] == PLAN:
            plan_kw[c["scenario"]] = c["theta_p"]

    def any_rec(vals):
        return True if any(v is True for v in vals) else (None if any(v is None for v in vals) else False)
    out = []
    for name in sorted({c["scenario"] for c in C}):
        rec = {v: any_rec([res[(name, kind, f)] for f in fams]) for v, (kind, fams) in VERSIONS.items()}
        some = any_rec(list(rec.values()))
        row = dict(scenario=name, recovers=rec)
        for v in VERSIONS:
            row[v] = dict(version=L["versions"][v]["id"],
                          label="R_REF" if rec[v] is True else "UNRESOLVED" if rec[v] is None or some is None
                          else "U_LIBRARY" if some else "NONE_AT_BUDGET")
        kw = plan_kw[name]
        row["planner_plan"] = dict(t_slow=kw["t_slow"], a_slow=kw["a_slow"],
                                   staged=bool(kw["t_slow"] > 0.3 and kw["a_slow"] > 0.5))
        row["staged_needed"] = bool(rec["full"] is True and rec["restricted"] is False)
        out.append(row)
    json.dump(out, open(f"{OUT}/labels.json", "w"), indent=1)
    for v in VERSIONS:
        print(f"{v:17s}", dict(Counter(r[v]["label"] for r in out)))
    nd = [r for r in out if r["staged_needed"]]
    print("staged needed (full recovers, restricted does not):", len(nd), "of", len(out),
          "| the planner recovers", sum(r["recovers"]["planner"] is True for r in nd),
          "of them, with a slow-then-change plan in", sum(r["planner_plan"]["staged"] and r["recovers"]["planner"] is True for r in nd))
    summary(C, "CARLA", lambda c: c["carla_ok"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "label"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--n-band", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    cmd_search(a) if a.cmd == "search" else cmd_label(a)
