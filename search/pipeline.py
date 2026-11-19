"""(vsearch) Combined policy-search pipeline: Victor's search (Case, iCEM, worst-case cost over the case's perturbed
scenes, success / mitigate / fail, best / boundary / ref candidates, every evaluation logged) with Kent's additions
(stratified random variations at scale + presets + boundary sweeps, three-way verdicts scored with FP and FN,
event-driven families, every candidate run in CARLA with per-tick evidence).

Stages (one batch = results_search/<rc>/search/<batch>/, the layout carla_verify.py reads):
    random     n stratified draws (case.stratified)                 -> {scenarios,profiles,candidates}_<site>_s<seed>.json
    presets    the case's named cells at every site                 -> ..._<site>_presets.json
    boundary   per family, sweeps from variations it recovers along the parameter that best separates its PASS from
               not-PASS in the batch so far (AUC), to the range's hard end, then bisection where PASS flips; every
               point is a full variation (all families searched), plus the sweep's starting policy kept fixed (ref)
                                                                    -> ..._<site>_b<round>.json
Per variation and family: the family's fixed settings, then iCEM (cem.icem, warm-started at the best setting) on the
robust cost = the worst of case.cost over case.perturbed(theta_s) (a contact in the nominal scene ends the evaluation
there: the worst case is at least that). The N_VERIFY lowest-cost policies and the boundary one (the smallest-margin
nominal recovery) get the three-way verdict: nominal + scene perturbations + ego-dynamics worlds; PASS = all
recover with margin, FAIL = none, UNCERTAIN = some. The family's best = PASS first, then the lowest robust cost.

    python -m search.pipeline rc07 --stage random --n 288 --seed 73 --batch vsearch --workers 9
    python -m search.pipeline rc07 --stage presets --batch vsearch --workers 9
    python -m search.pipeline rc07 --stage boundary --round 1 --k 6 --batch vsearch --workers 9
then carla_verify.py select / run / score / summarize --rc rc07 --batch vsearch (every candidate, per-tick evidence).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search import kent_bridge as kb  # noqa: E402
from search.cem import icem  # noqa: E402
from search.kent_cases import CASES  # noqa: E402

N_VERIFY = 2
ICEM = dict(pop=16, iters=4, elite_frac=0.25, init_std=0.2, decay=1.25, keep_frac=0.3, min_pop=8, sd_floor=0.03,
            alpha=0.7)
RANK = {"PASS": 2, "UNCERTAIN": 1, "FAIL": 0}
SIM_KEYS = ("success", "r", "fail", "contact", "impact_speed")


def _key(cfg):
    return tuple(sorted((k, round(v, 4)) for k, v in cfg.items()))


def robust_cost(case, site, th, fam, cfg):
    """(worst cost over the perturbed scenes, [nominal rollout, scene rollouts...]); a nominal contact ends it."""
    sims = []
    worst = -np.inf
    for i, t in enumerate(case.perturbed(th, site)):
        o = case.simulate(t, cfg, fam, site)
        o["w"] = "nominal" if i == 0 else f"scene{i}"
        sims.append(o)
        worst = max(worst, case.cost(o))
        if i == 0 and o["contact"] is not None:
            break
    return float(worst), sims


def search_family(case, site, th, fam, rng):
    """[eval] for one family: its settings, then iCEM warm-started at the best one."""
    evals = []

    def ev(cfg):
        cfg = kb.full_cfg(fam, cfg)
        c, sims = robust_cost(case, site, th, fam, cfg)
        evals.append(dict(family=fam, cfg=cfg, robust_cost=c, sims=sims))
        return c

    for cfg in case.settings(fam):
        ev(cfg)
    space = case.space(fam)
    if not space or fam == "keep":
        return evals
    best = min(evals, key=lambda e: e["robust_cost"])
    clear = best["robust_cost"] <= -2.0 * 0.3
    stop = lambda h: h[-1] <= -0.6 or (len(h) >= 2 and h[-2] - h[-1] < 0.05 and h[-1] < 100.0)
    icem(lambda u: ev(case.decode(fam, u)), len(space), rng, init_mu=case.encode(fam, best["cfg"]),
         stop=stop, **{**ICEM, "iters": 1 if clear else ICEM["iters"]})
    return evals


def verify(case, site, th, fam, e):
    """The candidate's three-way verdict: nominal + every scene perturbation + every ego-dynamics world."""
    cfg = e["cfg"]
    sims = list(e["sims"])
    have = {s["w"] for s in sims}
    for i, t in enumerate(case.perturbed(th, site)[1:], 1):
        if f"scene{i}" not in have:
            o = case.simulate(t, cfg, fam, site)
            o["w"] = f"scene{i}"
            sims.append(o)
    for j, w in enumerate(kb.WORLDS):
        o = case.simulate(th, cfg, fam, site, w)
        o["w"] = f"ego{j}"
        sims.append(o)
    nom = sims[0]
    ok = [s["success"] for s in sims]
    r_worst = min(s["r"] for s in sims)
    v = "FAIL" if not any(ok) else "UNCERTAIN"
    if all(ok) and r_worst >= kb.R_OK + case.pass_margin(fam):
        v = "PASS"
    out = {k: nom[k] for k in nom if k != "w"}
    out.update(cfg=cfg, verdict=v, n_pass=int(sum(ok)), n_worlds=len(ok), r_worst=float(r_worst),
               robust_cost=e["robust_cost"], cls=case.outcome_class(nom),
               worlds=[dict({k: s[k] for k in SIM_KEYS}, w=s["w"]) for s in sims[1:]])
    comp = {}
    for name, t in case.companions(th):                 # e.g. the mirror world: does it use only what is observable?
        o = case.simulate(t, cfg, fam, site)
        comp[name] = {k: o[k] for k in SIM_KEYS}
    if comp:
        out["companions"] = comp
    return out


def profile(case, site, th, fam, evals):
    E, seen = [], set()
    for e in sorted(evals, key=lambda e: e["robust_cost"]):
        k = _key(e["cfg"])
        if k not in seen:
            seen.add(k)
            E.append(e)
    checked = [verify(case, site, th, fam, e) for e in E[:N_VERIFY]]
    best = max(checked, key=lambda o: (RANK[o["verdict"]], -o["robust_cost"], o["r_worst"]))
    ok = [e for e in E if e["sims"][0]["success"]]
    bd = None
    if ok:
        b = min(ok, key=lambda e: e["sims"][0]["r"])
        bd = next((c for c in checked if _key(c["cfg"]) == _key(b["cfg"])), None) or verify(case, site, th, fam, b)
    # R4: the family's first fixed setting, the same policy in every scene
    k0 = _key(case.settings(fam)[0])
    e0 = next((e for e in E if _key(e["cfg"]) == k0), None)
    sh = None
    if e0 is not None:
        sh = next((c for c in checked + ([bd] if bd else []) if _key(c["cfg"]) == k0), None) or \
            verify(case, site, th, fam, e0)
    return dict(recovers=best["verdict"] == "PASS", verdict=best["verdict"], best=best, boundary=bd, shared=sh,
                n_evals=len(evals), n_success=len(ok), n_sims=sum(len(e["sims"]) for e in evals))


def verdict(case, prof):
    fams = list(prof)
    rec = [m for m in fams if prof[m]["verdict"] == "PASS"]
    unc = [m for m in fams if prof[m]["verdict"] == "UNCERTAIN"]
    obs = {}
    if any("companions" in prof[m]["best"] for m in fams):       # recovers in every companion world too
        obs = dict(observable_recovering=[m for m in fams if prof[m]["best"]["success"] and
                                          all(c["success"] for c in prof[m]["best"].get("companions", {}).values())])
    if rec:
        return dict(recoverable=True, status="recoverable", recovering=rec, uncertain=unc, mitigation=None, **obs)
    m = min(fams, key=lambda m: prof[m]["best"]["robust_cost"])
    b = prof[m]["best"]
    return dict(recoverable=False, status="uncertain" if unc else "unrecoverable", recovering=[], uncertain=unc,
                mitigation=dict(family=m, contact=b["contact"], impact_speed=b["impact_speed"], r=b["r"],
                                verdict=b["verdict"], cls=b["cls"]), **obs)


def run_variation(case, site, th, rng, families=None):
    runs, prof = [], {}
    for fam in families or case.families:
        ev = search_family(case, site, th, fam, rng)
        runs += ev
        prof[fam] = profile(case, site, th, fam, ev)
    return dict(profile=prof, verdict=verdict(case, prof), runs=runs)


# ---------------------------------------------------------------- jobs (one process each; the case is rebuilt there)
_CASE = {}


def _case(name):
    if name not in _CASE:
        _CASE[name] = CASES[name]()
    return _CASE[name]


def job_variation(job):
    name, vid, site, th, seed = job
    case = _case(name)
    t0 = time.time()
    r = run_variation(case, site, th, np.random.default_rng([seed, zlib.crc32(vid.encode()), 7]))
    return dict(r, variation_id=vid, site=site, params=th, secs=time.time() - t0)


def job_sweep(job):
    """One boundary sweep: the hard end, then bisection while PASS flips; every point a full variation, plus ref."""
    name, sw, seed, n_bis = job
    case = _case(name)
    fam, ax, site, th0, ref = sw["family"], sw["axis"], sw["site"], sw["theta_s"], sw["ref_cfg"]
    lo, hi = th0[ax], sw["hard"]
    pts = []

    def point(x, k):
        th = case.clip({**th0, ax: x}, site)
        vid = f"{sw['tag']}_{sw['n']:03d}_{k}"
        t0 = time.time()
        r = run_variation(case, site, th, np.random.default_rng([seed, zlib.crc32(vid.encode()), 7]))
        e = dict(family=fam, cfg=ref)
        e["robust_cost"], e["sims"] = robust_cost(case, site, th, fam, ref)
        r["ref"] = verify(case, site, th, fam, e)
        pts.append(dict(r, variation_id=vid, site=site, params=th, secs=time.time() - t0,
                        source=f"boundary:{fam}:{ax}:{sw['seed_vid']}"))
        return r["profile"][fam]["verdict"] == "PASS"

    if not point(hi, 0):
        for k in range(1, n_bis + 1):
            mid = 0.5 * (lo + hi)
            if point(mid, k):
                lo = mid
            else:
                hi = mid
    return pts


# ---------------------------------------------------------------- output
def write_stage(case, d, rows, tag_of, seed, how):
    """rows grouped by file tag -> scenarios / profiles / candidates / runs (carla_verify's format)."""
    by = {}
    for r in rows:
        by.setdefault(tag_of(r), []).append(r)
    n_c = Counter()
    for tag, R in by.items():
        R.sort(key=lambda r: r["variation_id"])
        scen, prof, cands = [], [], []
        for i, r in enumerate(R):
            sp = case.spec(r["site"])
            vid = r["variation_id"]
            scen.append(dict(variation_id=vid, rc=case.name, sid=case.site_sid(r["site"]), site=r["site"],
                             split=sp.split, scene_seed=seed, index=i, scenario_params=r["params"],
                             hidden=list(getattr(sp, "hidden", ())), source=r.get("source", how), history_id=vid))
            prof.append(dict(variation_id=vid, verdict=r["verdict"], families=r["profile"], secs=r["secs"]))
            ps = dict(optimizer="vsearch: the family's settings, then iCEM (search/cem.py) on the worst-case cost over "
                                "case.perturbed", rng=[seed, zlib.crc32(vid.encode()), 7],
                      verdict_worlds="nominal + case.SCENE_PERTURB + kent_bridge.WORLDS")
            ss = dict(seed=seed, index=i, how=r.get("source", how))
            items = []
            for m, f in r["profile"].items():
                items.append((m, "best", f["best"]))
                if f["boundary"] is not None and _key(f["boundary"]["cfg"]) != _key(f["best"]["cfg"]):
                    items.append((m, "boundary", f["boundary"]))
                if f.get("shared") is not None and _key(f["shared"]["cfg"]) not in \
                        {_key(o["cfg"]) for mm, k, o in items if mm == m}:
                    items.append((m, "shared", f["shared"]))
            if "ref" in r:
                fam = r["source"].split(":")[1]
                if _key(r["ref"]["cfg"]) not in {_key(o["cfg"]) for m, k, o in items if m == fam}:
                    items.append((fam, "ref", r["ref"]))
            for m, kind, o in items:
                cands.append(dict(variation_id=vid, sid=case.site_sid(r["site"]), site=r["site"], family=m, kind=kind,
                                  theta_s=r["params"], theta_p=o["cfg"], scene_seed=ss,
                                  policy_source=dict(ps, kind=kind),
                                  surrogate=dict(verdict=o["verdict"], nominal="PASS" if o["success"] else "FAIL",
                                                 r=o["r"], r_worst=o["r_worst"], n_pass=o["n_pass"],
                                                 n_worlds=o["n_worlds"], fail=o["fail"], near=o["near"],
                                                 contact=o["contact"], impact_speed=o["impact_speed"],
                                                 margins=o["margins"], cls=o["cls"], robust_cost=o["robust_cost"],
                                                 **({"companions": o["companions"]} if "companions" in o else {}))))
                n_c[o["verdict"]] += 1
            # R5: every candidate also in the other hidden worlds of this history (its own scene ids, same history)
            for wn, tw, (src, key) in case.hidden_worlds(r["params"]):
                wid = f"{vid}~{wn}"
                scen.append(dict(scen[-1] if scen[-1]["variation_id"] == vid else
                                 next(x for x in scen if x["variation_id"] == vid),
                                 variation_id=wid, scenario_params=tw, source=f"hidden_world:{wn}:{vid}",
                                 history_id=vid, hidden_world=wn))
                for m, kind, o in items:
                    sim = o.get("companions", {}).get(key) if src == "companion" else \
                        next((w for w in o.get("worlds", []) if w["w"] == key), None)
                    if sim is None:
                        continue
                    cands.append(dict(variation_id=wid, sid=case.site_sid(r["site"]), site=r["site"], family=m,
                                      kind=kind, theta_s=tw, theta_p=o["cfg"], scene_seed=ss,
                                      policy_source=dict(ps, kind=kind), history_id=vid, hidden_world=wn,
                                      surrogate=dict(verdict="WORLD", nominal="PASS" if sim["success"] else "FAIL",
                                                     r=sim["r"], fail=sim["fail"], contact=sim["contact"],
                                                     impact_speed=sim["impact_speed"], of_candidate=kind)))
                    n_c["WORLD"] += 1
        for name, obj in (("scenarios", scen), ("profiles", prof), ("candidates", cands)):
            json.dump(obj, open(os.path.join(d, f"{name}_{tag}.json"), "w"), indent=1, default=float)
        with open(os.path.join(d, f"runs_{tag}.jsonl"), "w") as fh:          # every evaluated policy (Victor's log)
            for r in R:
                ev = [dict(family=e["family"], cfg=e["cfg"], robust_cost=e["robust_cost"],
                           sims=[dict({k: s[k] for k in SIM_KEYS}, w=s["w"]) for s in e["sims"]]) for e in r["runs"]]
                fh.write(json.dumps(dict(variation_id=r["variation_id"], evals=ev), default=float) + "\n")
    print(f"wrote {sum(len(R) for R in by.values())} variations, candidates {dict(n_c)} -> {d}", flush=True)


def log_row(t0, r, case):
    v = r["verdict"]
    fams = " / ".join(f"{m} {r['profile'][m]['verdict'][:4]}" for m in r["profile"])
    print(f"[{time.time() - t0:6.0f}s] {r['variation_id']:<22} {v['status']:<13} {fams}  ({r['secs']:.0f}s)", flush=True)


def load_batch(d):
    S, P = {}, {}
    for f in os.listdir(d):
        if f.startswith("scenarios_"):
            S.update({s["variation_id"]: s for s in json.load(open(os.path.join(d, f)))})
        if f.startswith("profiles_"):
            P.update({p["variation_id"]: p for p in json.load(open(os.path.join(d, f)))})
    return S, P


def auc(x, y):
    """P(x of a positive > x of a negative)."""
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def plan_sweeps(case, d, k, rnd, rng):
    S, P = load_batch(d)
    vids = [v for v in P if S[v].get("source", "random").startswith(("random", "preset"))]
    sweeps = []
    for fam in case.families:
        if not case.space(fam):
            continue
        y = np.array([P[v]["families"][fam]["verdict"] == "PASS" for v in vids])
        best_ax, best_s, sgn = None, 0.0, 1
        for ax in case.boundary_axes():
            x = np.array([S[v]["scenario_params"][ax] for v in vids], float)
            a = auc(x, y)
            if abs(a - 0.5) > best_s:
                best_ax, best_s, sgn = ax, abs(a - 0.5), (1 if a < 0.5 else -1)   # PASS at low x -> hard end high
        if best_ax is None:
            continue
        seeds = [v for v, yy in zip(vids, y) if yy]
        for v in rng.permutation(seeds)[:k]:
            s = S[v]
            lo, hi = case.range_of(best_ax, s["site"], s["scenario_params"])
            sweeps.append(dict(family=fam, axis=best_ax, auc_dev=best_s, hard=hi if sgn > 0 else lo, site=s["site"],
                               theta_s=s["scenario_params"], seed_vid=v, ref_cfg=P[v]["families"][fam]["best"]["cfg"],
                               tag=f"{s['site']}_b{rnd}", n=len(sweeps)))
        print(f"boundary: {fam}: axis {best_ax} (|AUC-0.5| {best_s:.2f}), towards its {'high' if sgn > 0 else 'low'} "
              f"end, {min(k, len(seeds))} sweeps", flush=True)
    return sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--stage", choices=["random", "presets", "boundary"], required=True)
    ap.add_argument("--batch", default="vsearch")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--k", type=int, default=6, help="boundary: sweeps per family")
    ap.add_argument("--bisect", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (a smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", default=kb.OUT)
    args = ap.parse_args()
    case = _case(args.case)
    d = os.path.join(args.out, case.name, "search", args.batch)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng([args.seed, 1])
    t0 = time.time()
    part = os.path.join(d, f"partial_{args.stage}{args.round if args.stage == 'boundary' else ''}_s{args.seed}.jsonl")
    rows = [json.loads(l) for l in open(part)] if os.path.exists(part) else []      # checkpoint: resume
    done = {r["variation_id"] for r in rows} | {r.get("sweep") for r in rows}
    if rows:
        print(f"resuming: {len(rows)} variations already in {part}", flush=True)

    def keep(r, fh):
        rows.append(r)
        fh.write(json.dumps(r, default=float) + "\n")
        fh.flush()
        log_row(t0, r, case)

    if args.stage == "boundary":
        sweeps = plan_sweeps(case, d, args.k, args.round, rng)
        jobs = [(args.case, sw, args.seed, args.bisect) for sw in sweeps if f"{sw['tag']}_{sw['n']:03d}" not in done]
        jobs = jobs[:args.limit] if args.limit else jobs
        with mp.Pool(args.workers) as pool, open(part, "a") as fh:
            for pts in pool.imap_unordered(job_sweep, jobs):
                for r in pts:
                    keep(dict(r, sweep=r["variation_id"].rsplit("_", 1)[0]), fh)
        write_stage(case, d, rows, lambda r: r["variation_id"].rsplit("_", 2)[0], args.seed, "boundary")
        return
    if args.stage == "random":
        V = [(f"{site}_s{args.seed}_{i:04d}", site, th, "random") for i, (site, th) in enumerate(case.stratified(args.n, rng))]
        tag_of = lambda r: r["variation_id"].rsplit("_", 1)[0]
    else:
        V = [(f"{site}_p_{name}", site, th, f"preset:{name}") for site in sorted(getattr(case, "PRESET_SITES", None) or case.sites())
             for name, th in case.presets(site).items()]
        tag_of = lambda r: f"{r['site']}_presets"
    src = {v[0]: v[3] for v in V}
    jobs = [(args.case, vid, site, th, args.seed) for vid, site, th, _ in V if vid not in done]
    jobs = jobs[:args.limit] if args.limit else jobs
    print(f"{case.name} {args.stage}: {len(jobs)} variations to run, {args.workers} workers", flush=True)
    with mp.Pool(args.workers) as pool, open(part, "a") as fh:
        for r in pool.imap_unordered(job_variation, jobs):
            keep(dict(r, source=src[r["variation_id"]]), fh)
    write_stage(case, d, rows, tag_of, args.seed, args.stage)
    print(f"{dict(Counter(r['verdict']['status'] for r in rows))} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
