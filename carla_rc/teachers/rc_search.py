"""Policy search per variation on the surrogate: which options recover each variation, and which mitigates best when
none does (team approach, 2026-11-17; output layout as results_search/<rc>/search/<batch>/).

For every variation (a parameter draw from the RC's ranges, hidden parameters included: the search answers "which
policy recovers THIS variation", it is not a deployable planner) and every family (spec.modes):
    1. the family's fixed settings (options.SETTINGS)
    2. CEM over the family's parameters (options.PARAM_SPACE, normalised to [0, 1]): POP candidates per iteration
       around the elite mean, ITERS iterations; a family stops once it has a clear recovery (r >= R_CLEAR)
Every rollout is scored by `score`: a recovery (CARLA-style success: no contact, r >= 0.05, goal, envelope) beats
any failure; among failures (mitigation) no contact beats contact, then a lower impact speed, then a larger r.

Verdicts: a family's top recoveries are re-run in perturbed surrogate worlds (WORLDS: the ego's start, braking,
steering lag, grip); PASS = recovers in all of them, FAIL = in none, UNCERTAIN = some (the surrogate abstains and
CARLA labels it). The most robust recovery is the family's best.

Per variation the result is the recovery profile: per family the best candidate (recovers / mitigates) and its
margin and verdict, plus the boundary candidate (the recovering one with the smallest r) - the "edge" runs CARLA
must confirm. A variation is recoverable (some family PASS), uncertain (none PASS, some UNCERTAIN) or unrecoverable;
when not recoverable, the mitigation is the family whose best does least harm.

    python carla_rc/teachers/rc_search.py RC07-S5-opt --site A --n 200 --seed 71 --batch baseline --workers 9
-> <out>/<rc>/search/<batch>/{scenarios.json, candidates.json, profiles.json, runs.jsonl}
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import surrogate  # noqa: E402,F401  (offline stand-ins first)
from evidence import OUT, impact  # noqa: E402
from options import PARAM_SPACE, SETTINGS, ModeExpert, full_cfg  # noqa: E402

POP, ELITE, ITERS, SD0 = 10, 3, 3, 0.25
R_OK, R_CLEAR = 0.05, 0.3


def env_sid(sid, site):
    return sid if site == "A" else f"{sid}@{site}"


def rc_of(sid):
    return sid.split("-")[0].lower()                      # "RC07-S5-opt" -> "rc07"


def rollout(env, p, m, cfg):
    ex = ModeExpert(env, m, cfg)
    obs, _ = env.reset(options={"params": p})
    ex.reset()
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(ex.act(obs))
        done = term or trunc
    ticks = [r for r in env.log["tick"] if r["t"] >= -1e-6]
    near = min(ticks, key=lambda r: r["m"][0])["near"] if ticks else ""
    who, dv = impact(env, info)
    return dict(family=m, cfg={k: float(v) for k, v in cfg.items()}, success=bool(info["success"]),
                r=float(info["r_nb"]), fail=info["fail"] or ("" if info["success"] else "margin"), near=near,
                contact=who, impact_speed=dv, margins={k: float(v) for k, v in info["min_margins"].items()},
                t_det=ex.t_det, t_hb=ex.t_hb)


def score(o):
    if o["success"]:
        return 100.0 + min(o["r"], 2.0)
    if o["contact"] is None:
        return min(o["r"], 2.0)
    return -10.0 - (o["impact_speed"] if np.isfinite(o["impact_speed"]) else 50.0)


def to_unit(cfg, space):
    return np.array([(cfg[k] - lo) / (hi - lo) for k, (lo, hi) in space.items()])


def from_unit(u, space):
    return {k: float(lo + np.clip(x, 0.0, 1.0) * (hi - lo)) for x, (k, (lo, hi)) in zip(u, space.items())}


def search_family(env, p, m, rng):
    runs = [rollout(env, p, m, full_cfg(m, c)) for c in SETTINGS.get(m, [{}])]
    space = PARAM_SPACE.get(m)
    if m == "keep" or not space:
        return runs
    best = max(runs, key=score)
    mean, sd = to_unit({**full_cfg(m, {}), **best["cfg"]}, space), np.full(len(space), SD0)
    for _ in range(ITERS):
        if best["success"] and best["r"] >= R_CLEAR:
            break
        pop = [from_unit(mean + sd * rng.standard_normal(len(space)), space) for _ in range(POP)]
        new = [rollout(env, p, m, c) for c in pop]
        runs += new
        el = sorted(new + [best], key=score, reverse=True)[:ELITE]
        U = np.array([to_unit(o["cfg"], space) for o in el])
        mean, sd = U.mean(0), np.maximum(U.std(0), 0.05)
        prev, best = best, max(runs, key=score)
        if prev["success"] and best["r"] < prev["r"] + 0.02:     # recovers, and the margin stopped improving
            break
    return runs


# Perturbed surrogate worlds for the verdict: what CARLA may differ in (the ego's start, braking limit and tracking
# bias, steering lag, grip).
# A candidate is PASS when it recovers in the nominal world and all of these, FAIL when it fails in all of them, and
# UNCERTAIN otherwise (the surrogate abstains; CARLA labels it).
WORLDS = [dict(start=(0.6, 0.15)), dict(start=(-0.6, -0.15)), dict(dyn=(0.9, 1.0, 1.0, 1.0)), dict(dyn=(1.1, 1.0, 1.0, 1.0)),
          dict(dyn=(1.0, 1.0, 1.3, 1.0)), dict(dyn=(1.0, 1.0, 0.7, 1.0)), dict(dyn=(1.0, 1.0, 1.0, 0.95)),
          dict(bias=0.0), dict(bias=0.8)]
# Lane changes: CARLA's starts slower and overshoots ~0.3-0.5 m, a shape no lag / delay of the surrogate fits (RC07-S5
# baseline); a lateral PASS needs this much margin above R_OK in every world, else the surrogate abstains (UNCERTAIN).
PASS_MARGIN = {"right>away": 0.13, "right>return": 0.13, "right>stay": 0.13}
N_ROBUST = 3                       # a family's top recoveries checked in WORLDS; the most robust one is its best


def robust(env, p, o):
    """o with its outcomes in WORLDS and the three-way verdict."""
    W = []
    for w in WORLDS:
        env.start, env.dyn, env.bias = w.get("start", (0.0, 0.0)), w.get("dyn", (1.0, 1.0, 1.0, 1.0)), w.get("bias")
        W.append(rollout(env, p, o["family"], o["cfg"]))
    env.start, env.dyn, env.bias = (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), None
    ok = [o["success"]] + [w["success"] for w in W]
    r_worst = min([o["r"]] + [w["r"] for w in W])
    v = "FAIL" if not any(ok) else "UNCERTAIN"
    if all(ok) and r_worst >= R_OK + PASS_MARGIN.get(o["family"], 0.0):
        v = "PASS"
    return dict(o, verdict=v, n_pass=sum(ok), n_worlds=len(ok), r_worst=r_worst,
                worlds=[{k: w[k] for k in ("success", "r", "fail", "contact", "impact_speed")} for w in W])


def profile(env, p, runs, modes):
    """Per family: best (the most robust recovery, else the best mitigation), boundary (smallest-r recovery), each
    with its three-way verdict; counts."""
    out = {}
    for m in modes:
        R = [o for o in runs if o["family"] == m]
        ok = sorted([o for o in R if o["success"]], key=score, reverse=True)
        top, seen = [], set()
        for o in ok:
            key = tuple(sorted(o["cfg"].items()))
            if key not in seen:
                seen.add(key)
                top.append(o)
            if len(top) == N_ROBUST:
                break
        if top:
            checked = [robust(env, p, o) for o in top]
            b = max(checked, key=lambda o: (o["n_pass"], o["r_worst"]))
        else:
            b = robust(env, p, max(R, key=score))
        bd = min(ok, key=lambda o: o["r"]) if ok else None
        if bd is not None:
            bd = next((c for c in (checked if top else []) if c["cfg"] == bd["cfg"]), None) or robust(env, p, bd)
        out[m] = dict(recovers=b["verdict"] == "PASS", verdict=b["verdict"], best=b, boundary=bd,
                      n_evals=len(R), n_success=len(ok))
    return out


def verdict(prof, modes):
    """recoverable (some family PASS), uncertain (none PASS, some UNCERTAIN), or not; the mitigation otherwise."""
    rec = [m for m in modes if prof[m]["verdict"] == "PASS"]
    unc = [m for m in modes if prof[m]["verdict"] == "UNCERTAIN"]
    if rec:
        return dict(recoverable=True, status="recoverable", recovering=rec, uncertain=unc, mitigation=None)
    best = max(modes, key=lambda m: score(prof[m]["best"]))
    b = prof[best]["best"]
    return dict(recoverable=False, status="uncertain" if unc else "unrecoverable", recovering=[], uncertain=unc,
                mitigation=dict(family=best, contact=b["contact"], impact_speed=b["impact_speed"], r=b["r"],
                                verdict=b["verdict"]))


_ENV = {}


def work(job):
    sid, site, i, p, seed = job
    from surrogate import SurrogateEnv
    key = env_sid(sid, site)
    if key not in _ENV:
        _ENV[key] = SurrogateEnv(key, seed=0)
    env = _ENV[key]
    rng = np.random.default_rng([seed, i, 7])
    t0 = time.time()
    runs = []
    for m in env.spec.modes:
        runs += search_family(env, p, m, rng)
    prof = profile(env, p, runs, env.spec.modes)
    return dict(i=i, site=site, params=p, profile=prof, verdict=verdict(prof, env.spec.modes), runs=runs,
                secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--site", default="A")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", default="baseline")
    ap.add_argument("--set", nargs="*", default=[], help="range overrides k=v1,v2 or k=lo:hi")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    from scenarios import SPECS
    spec = SPECS[env_sid(args.sid, args.site)]
    over = {}
    for kv in args.set:
        k, v = kv.split("=")
        over[k] = tuple(float(x) for x in v.split(":")) if ":" in v else \
            [type(spec.ranges[k][0])(x) if not isinstance(spec.ranges[k][0], str) else x for x in v.split(",")]
    spec.overrides = over
    rng = np.random.default_rng([args.seed, 1])
    P = [spec.sample(rng) for _ in range(args.n)]
    d = os.path.join(args.out, rc_of(args.sid), "search", args.batch)
    os.makedirs(d, exist_ok=True)
    tag = f"{args.site}_s{args.seed}"
    jobs = [(args.sid, args.site, i, p, args.seed) for i, p in enumerate(P)]
    t0, rows = time.time(), []
    with mp.Pool(args.workers) as pool, open(os.path.join(d, f"runs_{tag}.jsonl"), "w") as fh:
        for r in pool.imap_unordered(work, jobs):
            rows.append(r)
            fh.write(json.dumps(dict(variation_id=f"{tag}_{r['i']:04d}", runs=r["runs"]), default=float) + "\n")
            v = r["verdict"]
            fams = " / ".join(f"{m} {r['profile'][m]['verdict']}" for m in spec.modes)
            print(f"[{time.time() - t0:6.0f}s] {r['i']:4d} {v['status']:<13} {fams}"
                  + ("" if v["recoverable"] else f"  -> mitigate: {v['mitigation']['family']} ({v['mitigation']['contact']}, {v['mitigation']['impact_speed']:.1f} m/s)"),
                  flush=True)
    rows.sort(key=lambda r: r["i"])
    scen = [dict(variation_id=f"{tag}_{r['i']:04d}", rc=rc_of(args.sid), sid=env_sid(args.sid, r["site"]), site=r["site"],
                 split=spec.split, scene_seed=args.seed, index=r["i"], scenario_params=r["params"],
                 hidden=list(getattr(spec, "hidden", ()))) for r in rows]
    prof = [dict(variation_id=s["variation_id"], verdict=r["verdict"],
                 families={m: {k: v for k, v in r["profile"][m].items()} for m in spec.modes}, secs=r["secs"])
            for s, r in zip(scen, rows)]
    cands = []
    for s, r in zip(scen, rows):
        for m in spec.modes:
            f = r["profile"][m]
            kinds = [("best", f["best"])] + ([("boundary", f["boundary"])] if f["boundary"] is not None and
                                             f["boundary"]["cfg"] != f["best"]["cfg"] else [])
            for kind, o in kinds:
                cands.append(dict(variation_id=s["variation_id"], sid=s["sid"], site=s["site"], family=m, kind=kind,
                                  theta_s=s["scenario_params"], theta_p=o["cfg"],
                                  surrogate=dict(verdict=o["verdict"], nominal="PASS" if o["success"] else "FAIL",
                                                 r=o["r"], r_worst=o["r_worst"], n_pass=o["n_pass"],
                                                 n_worlds=o["n_worlds"], fail=o["fail"], near=o["near"],
                                                 contact=o["contact"], impact_speed=o["impact_speed"],
                                                 margins=o["margins"])))
    for name, obj in (("scenarios", scen), ("profiles", prof), ("candidates", cands)):
        path = os.path.join(d, f"{name}_{tag}.json")
        json.dump(obj, open(path, "w"), indent=1, default=float)
    from collections import Counter
    print(f"done: {len(rows)} variations {dict(Counter(r['verdict']['status'] for r in rows))}, {len(cands)} "
          f"candidates {dict(Counter(c['surrogate']['verdict'] for c in cands))} -> {d}")


if __name__ == "__main__":
    main()
