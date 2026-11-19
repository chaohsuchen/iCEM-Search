"""CARLA check of the surrogate's policy-search results (rc_search.py), with per-run evidence (evidence.py).

    select  pick the candidates CARLA must confirm from search/<batch>/candidates_*.json:
              recommended every variation's recommended policy (its most robust recovery, else its mitigation)
              uncertain   the surrogate abstained (its perturbed worlds disagree)
              edge        the surrogate's margin is within R_EDGE of the success threshold (either verdict)
              boundary    the smallest-margin recovery the search found (kind "boundary")
              mitigation  every family's best on a variation the surrogate calls unrecoverable
              handback    the hand-back families' bests where they disagree (one recovers, another not)
              sample      a random --frac of the rest (clear recoveries and clear failures); "complete" when
                          --frac is 1 (the default: every candidate gets a CARLA label and per-tick evidence)
            -> carla/<batch>/selected.json (run index NNN = position in this list)
            --complete: append every candidate not yet in an existing selected.json (why "complete"), numbered
            after it, so finished runs keep their folders and shards skip them
    run     execute selected candidates in CARLA (shard k of n, one server per shard): run folder
            carla/<batch>/runs/<NNN>_<variation>_<family>_<kind>/ + one line in results_shard<k>.jsonl
            (surrogate prediction and CARLA result side by side); resumes where a shard stopped
    rerun   candidates again with the ego's start shifted (--starts, default +-0.6 m / +-0.15 m/s; --ids or
            --uncertain): does CARLA's label hold? -> carla/<batch>/<tag>/ (runs/ + results_shard<k>.jsonl)
    search  rc_search's family search in CARLA on one variation (--variation, --family): a recovery the surrogate
            missed? -> carla/<batch>/carla_search/<variation>_<family>.jsonl
    score   safety score over the CARLA-checked candidates: false positives (surrogate PASS, CARLA FAIL), false
            negatives (surrogate FAIL, CARLA PASS); score = 1 - (FP + FN) / N; the disagreements, listed

    python carla_rc/teachers/carla_verify.py select --rc rc07 --batch baseline
    python carla_rc/teachers/carla_verify.py select --rc rc07 --batch baseline --complete
    python carla_rc/teachers/carla_verify.py run --rc rc07 --batch baseline --shard 0 --of 4 --port 2100
    python carla_rc/teachers/carla_verify.py score --rc rc07 --batch baseline
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("RC_RESULTS", os.path.join(os.path.dirname(os.path.dirname(HERE)), "results_search"))

R_EDGE = 0.1


def batch_dirs(args):
    root = os.path.join(args.out, args.rc)
    return os.path.join(root, "search", args.batch), os.path.join(root, "carla", args.batch)


def run_id(i, c):
    return f"{i:03d}_{c['variation_id']}_{c['family'].replace('>', '-')}_{c['kind']}"


def recommended(prof):
    """The family whose best a variation recommends: the most robust recovery (PASS in the most worlds, then the
    largest worst-case margin), or the mitigation when nothing recovers. Every variation gets this run in CARLA, so
    every scene has per-tick evidence."""
    v = prof["verdict"]
    if not v["recoverable"]:
        return v["mitigation"]["family"]
    F = prof["families"]
    return max(v["recovering"], key=lambda m: (F[m]["best"]["n_pass"], F[m]["best"]["r_worst"]))


def cmd_select(args):
    sd, cd = batch_dirs(args)
    C = [c for f in sorted(glob.glob(os.path.join(sd, "candidates_*.json"))) for c in json.load(open(f))]
    P = {p["variation_id"]: p for f in sorted(glob.glob(os.path.join(sd, "profiles_*.json"))) for p in json.load(open(f))}
    S = {s["variation_id"]: s for f in sorted(glob.glob(os.path.join(sd, "scenarios_*.json"))) for s in json.load(open(f))}
    for c in C:                       # the seeds, explicit: the scene draw and the search that found the policy
        s = S[c["variation_id"]]
        c.setdefault("scene_seed", dict(seed=s["scene_seed"], index=s["index"],
                                        how="spec.sample(np.random.default_rng([seed, 1])), draw number `index`"))
        c.setdefault("policy_source", dict(optimizer="CEM over the family's parameters (rc_search.py) after its fixed "
                                           "settings", rng=[s["scene_seed"], s["index"], 7], kind=c["kind"],
                                           verdict_worlds="rc_search.WORLDS (perturbed surrogate worlds)"))
    if args.keep:                     # only these histories (a searched scene and its hidden worlds)
        keep = set(open(args.keep).read().split())
        C = [c for c in C if c["variation_id"].split("~")[0] in keep]
    if args.complete:
        old = json.load(open(os.path.join(cd, "selected.json")))
        have = {(c["variation_id"], c["family"], c["kind"]) for c in old}
        new = [dict(c, why=["complete"]) for c in C if (c["variation_id"], c["family"], c["kind"]) not in have]
        new.sort(key=lambda c: (c["sid"], c["variation_id"], c["family"], c["kind"]))
        for i, c in enumerate(new, len(old)):
            c["run_id"] = run_id(i, c)
        json.dump(old + new, open(os.path.join(cd, "selected.json"), "w"), indent=1, default=float)
        print(f"{len(C)} candidates: {len(old)} already selected, {len(new)} appended")
        return
    rng = np.random.default_rng(0)
    sel, rest = [], []
    for c in C:
        if c.get("hidden_world"):            # R5: the candidate in another hidden world of its history (vsearch)
            sel.append(dict(c, why=["hidden_world"]))
            continue
        prof = P[c["variation_id"]]
        hb = [m for m in prof["families"] if ">" in m]
        split = len({prof["families"][m]["recovers"] for m in hb}) > 1
        why = ["recommended"] if c["kind"] == "best" and c["family"] == recommended(prof) else []
        if c["surrogate"]["verdict"] == "UNCERTAIN":
            why.append("uncertain")
        if abs(c["surrogate"]["r"] - 0.05) <= R_EDGE:
            why.append("edge")
        if c["kind"] == "boundary":
            why.append("boundary")
        if c["kind"] == "shared":
            why.append("shared")
        if not prof["verdict"]["recoverable"] and c["kind"] == "best":
            why.append("mitigation")
        if split and c["family"] in hb and c["kind"] == "best":
            why.append("handback")
        (sel if why else rest).append(dict(c, why=why))
    for c in rest:
        if args.frac >= 1.0 or rng.random() < args.frac:
            sel.append(dict(c, why=["sample" if args.frac < 1.0 else "complete"]))
    sel.sort(key=lambda c: (c["sid"], c["variation_id"], c["family"], c["kind"]))
    for i, c in enumerate(sel):
        c["run_id"] = run_id(i, c)
    os.makedirs(cd, exist_ok=True)
    json.dump(sel, open(os.path.join(cd, "selected.json"), "w"), indent=1, default=float)
    from collections import Counter
    print(f"{len(C)} candidates -> {len(sel)} selected:",
          dict(Counter(w for c in sel for w in c["why"])), dict(Counter(c["surrogate"]["verdict"] for c in sel)))


def make_env(sid, port):
    from env import TeacherEnv
    return TeacherEnv(sid, port=port, seed=0, timeout=180.0)


def restart(port, town):
    script = os.environ.get("RC_RESTART", os.path.expanduser("~/rc/restart_srv.sh"))
    if os.path.exists(script):
        subprocess.run(["bash", script, str(port), town], check=False)


def server_rss_gb(port):
    """Resident memory of the CARLA server on port (GB; 0 if not found). The servers grow over long batches until
    the kernel's OOM killer takes one, which aborts the client (an uncatchable TimeoutException)."""
    out = subprocess.run(["pgrep", "-f", f"Shipping.*carla-rpc-port={port} "], capture_output=True, text=True).stdout
    for pid in out.split():
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
        except OSError:
            pass
    return 0.0


RSS_MAX_GB = 7.0


def execute(todo, res_path, run_dir, args, start=lambda c: (0.0, 0.0)):
    """Run each candidate in CARLA (server args.port): run folder run_dir/<run_id>, one result line in res_path.
    start(c) = the ego's extra start offset (x [m], v [m/s]) for c; a server crash / hang restarts it, one retry."""
    from env import TeacherEnv  # noqa: F401  (CARLA / ScenarioRunner modules before anything stubs them)
    import evidence
    from evidence import impact
    from scenarios import SPECS
    env, cur, t0 = None, None, time.time()
    for k, c in enumerate(todo):
        out = None
        if server_rss_gb(args.port) > RSS_MAX_GB:               # before the OOM killer: exit, the shard loop restarts
            print(f"server {args.port} RSS {server_rss_gb(args.port):.1f} GB: exit for a fresh server", flush=True)
            os._exit(3)                                         # (a restart from here aborts on the stale client)
        for attempt in range(2):
            try:
                if env is None or cur != c["sid"]:
                    if env is not None:
                        env.close()
                    env, cur = make_env(c["sid"], args.port), c["sid"]
                env.start = start(c)
                info, rec = evidence.run(env, c["theta_s"], c["family"], c["theta_p"])
                who, dv = impact(env, info)
                summ = evidence.write(rec, os.path.join(run_dir, c["run_id"]), dict(
                    run_id=c["run_id"], rc=args.rc.upper(), variation_id=c["variation_id"], kind=c["kind"],
                    scenario_params=c["theta_s"], hidden=list(getattr(SPECS[c["sid"]], "hidden", ())),
                    policy_params=c["theta_p"], surrogate=c["surrogate"], simulator="CARLA 0.9.15",
                    history_id=c.get("history_id", c["variation_id"]), hidden_world=c.get("hidden_world"),
                    scene_seed=c.get("scene_seed"), policy_source=dict(c.get("policy_source", {}), selected_for=c["why"])))
                out = dict(label=summ["label"], reason=summ["reason"], r=float(info["r_nb"]), fail=info["fail"],
                           binding=min(info["min_margins"], key=info["min_margins"].get), contact=who,
                           impact_speed=dv, margins={k: float(v) for k, v in info["min_margins"].items()})
                break
            except AssertionError as e:                             # a bad site / spec: no restart (it would abort us)
                traceback.print_exc()
                env, cur = None, None
                out = dict(label="INVALID", reason=f"{type(e).__name__}: {e}"[:300])
                break
            except Exception as e:                                  # a server crash / hang: restart it, retry once
                traceback.print_exc()
                try:
                    env.close()
                except Exception:
                    pass
                env = None
                restart(args.port, SPECS[c["sid"]].town)
                out = dict(label="INVALID", reason=f"{type(e).__name__}: {e}"[:300])
        line = dict(id=c["run_id"], variation_id=c["variation_id"], sid=c["sid"], family=c["family"], kind=c["kind"],
                    why=c["why"], theta_s=c["theta_s"], theta_p=c["theta_p"], surrogate=c["surrogate"], carla=out,
                    history_id=c.get("history_id", c["variation_id"]), hidden_world=c.get("hidden_world"))
        if "of" in c:
            line.update(of=c["of"], start=list(start(c)))
        with open(res_path, "a") as fh:
            fh.write(json.dumps(line, default=float) + "\n")
        print(f"[{time.time() - t0:6.0f}s] {k + 1}/{len(todo)} {c['run_id']}: surrogate {c['surrogate']['verdict']} "
              f"({c['surrogate']['r']:+.2f}) carla {out['label']} ({out.get('r', float('nan')):+.2f}) {out.get('reason', '')[:60]}",
              flush=True)
    if env is not None:
        env.close()


def done_ids(res_path):
    return {json.loads(l)["id"] for l in open(res_path) if l.strip()} if os.path.exists(res_path) else set()


def cmd_run(args):
    sd, cd = batch_dirs(args)
    sel = json.load(open(os.path.join(cd, "selected.json")))
    mine = [c for i, c in enumerate(sel) if i % args.of == args.shard]
    res_path = os.path.join(cd, f"results_shard{args.shard}.jsonl")
    done = done_ids(res_path)
    todo = [c for c in mine if c["run_id"] not in done]
    if args.skip_town:                 # e.g. Town13: four servers loading it at once run out of memory
        from scenarios import SPECS
        skip = set(args.skip_town.split(","))
        todo = [c for c in todo if SPECS[c["sid"]].town not in skip]
    print(f"shard {args.shard}/{args.of}: {len(mine)} candidates, {len(todo)} to run", flush=True)
    execute(todo, res_path, os.path.join(cd, "runs"), args)


def cmd_rerun(args):
    """Selected candidates again, with the ego's start shifted (--starts dx:dv,...): does CARLA's label hold?
    --ids picks run_ids, --uncertain every candidate the surrogate abstained on. -> carla/<batch>/<tag>/:
    runs/<run_id>_x<dx>_v<dv>/ and results_shard<k>.jsonl (fields of / start: the original run and the offset)."""
    sd, cd = batch_dirs(args)
    sel = json.load(open(os.path.join(cd, "selected.json")))
    ids = set(args.ids.split(",")) if args.ids else None
    if args.near:                            # R6: repeats near the pass / fail boundary (CARLA's margin within --near)
        R = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(os.path.join(cd, "results_shard*.jsonl"))
             for l in open(f) if l.strip()}
        near = {i for i, r in R.items() if r["carla"].get("label") in ("PASS", "FAIL")
                and abs(r["carla"]["r"] - 0.05) <= args.near}
        ids = near if ids is None else ids & near
    keep = set(open(args.keep).read().split()) if args.keep else None
    base = [c for c in sel if (ids is None or c["run_id"] in ids)
            and (keep is None or c["variation_id"].split("~")[0] in keep)
            and (not args.uncertain or c["surrogate"]["verdict"] == "UNCERTAIN")]
    starts = [tuple(float(x) for x in s.split(":")) for s in args.starts.split(",")]
    jobs = [dict(c, of=c["run_id"], run_id=f"{c['run_id']}_x{dx:+.2f}_v{dv:+.2f}", dstart=(dx, dv))
            for c in base for dx, dv in starts]
    mine = [c for i, c in enumerate(jobs) if i % args.of == args.shard]
    td = os.path.join(cd, args.tag)
    os.makedirs(td, exist_ok=True)
    res_path = os.path.join(td, f"results_shard{args.shard}.jsonl")
    done = done_ids(res_path)
    todo = [c for c in mine if c["run_id"] not in done]
    print(f"shard {args.shard}/{args.of}: {len(mine)} reruns, {len(todo)} to run", flush=True)
    execute(todo, res_path, os.path.join(td, "runs"), args, start=lambda c: c["dstart"])


def cmd_search(args):
    """rc_search's family search, run in CARLA instead of the surrogate, on one variation (--variation) and family
    (--family; one server each): does a policy the surrogate missed recover it? -> carla/<batch>/carla_search/
    <variation>_<family>.jsonl, one line per rollout (rc_search.rollout's fields)."""
    from env import TeacherEnv  # noqa: F401  (before rc_search, which stubs the CARLA modules for the surrogate)
    import rc_search
    sd, cd = batch_dirs(args)
    site = args.variation.split("_")[0]
    scen = {v["variation_id"]: v for f in glob.glob(os.path.join(sd, f"scenarios_{site}_s*.json"))
            for v in json.load(open(f))}[args.variation]
    sid = rc_search.env_sid(scen["sid"], scen["site"])
    rc_search.ITERS, rc_search.R_CLEAR = args.iters, 1e9          # search the whole budget: the best margin, not the first
    env = make_env(sid, args.port)
    od = os.path.join(cd, "carla_search")
    os.makedirs(od, exist_ok=True)
    runs = rc_search.search_family(env, scen["scenario_params"], args.family, np.random.default_rng(args.seed))
    env.close()
    with open(os.path.join(od, f"{args.variation}_{args.family.replace('>', '-')}.jsonl"), "w") as fh:
        for o in runs:
            fh.write(json.dumps(o, default=float) + "\n")
    ok = [o for o in runs if o["success"]]
    print(f"{args.variation} {args.family}: {len(runs)} CARLA rollouts, {len(ok)} recover"
          + (f", best r {max(o['r'] for o in ok):+.2f}" if ok else
             f", best {max(runs, key=rc_search.score)['fail']} r {max(runs, key=rc_search.score)['r']:+.2f}"), flush=True)


def cmd_score(args):
    sd, cd = batch_dirs(args)
    A = [json.loads(l) for f in sorted(glob.glob(os.path.join(cd, "results_shard*.jsonl"))) for l in open(f) if l.strip()]
    A = [r for r in A if r["carla"]["label"] in ("PASS", "FAIL")]
    U = [r for r in A if r["surrogate"]["verdict"] == "UNCERTAIN"]
    R = [r for r in A if r["surrogate"]["verdict"] in ("PASS", "FAIL")]
    fp = [r for r in R if r["surrogate"]["verdict"] == "PASS" and r["carla"]["label"] == "FAIL"]
    fn = [r for r in R if r["surrogate"]["verdict"] == "FAIL" and r["carla"]["label"] == "PASS"]
    n = len(R)
    print(f"{len(A)} CARLA-checked candidates: {n} with a surrogate verdict -> FP {len(fp)}, FN {len(fn)}, safety score "
          f"{1 - (len(fp) + len(fn)) / max(n, 1):.3f}; {len(U)} UNCERTAIN (CARLA: "
          f"{sum(r['carla']['label'] == 'PASS' for r in U)} PASS, {sum(r['carla']['label'] == 'FAIL' for r in U)} FAIL)")
    print(f"  coverage (decided / (decided + UNCERTAIN)) {n / max(n + len(U), 1):.1%}; {len(A) - n - len(U)} runs "
          f"without a verdict (hidden worlds, R5); target: safety score 1.000")
    nom = [r for r in A if "nominal" in r["surrogate"]]
    if nom:
        bad = sum((r["surrogate"]["nominal"] == "PASS") != (r["carla"]["label"] == "PASS") for r in nom)
        print(f"  (nominal surrogate run alone, no abstention: {bad} of {len(nom)} disagree, score {1 - bad / len(nom):.3f})")
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0, 0])
    for r in R:
        for key in (f"family {r['family']}", f"why {'+'.join(r['why'])}", f"site {r['sid']}"):
            by[key][0] += 1
            by[key][1] += r in fp
            by[key][2] += r in fn
    for k in sorted(by):
        t, a, b = by[k]
        print(f"  {k:<34} n {t:4d}  FP {a:3d}  FN {b:3d}  score {1 - (a + b) / t:.3f}")
    for name, L in (("false positives (surrogate PASS, CARLA FAIL)", fp), ("false negatives (surrogate FAIL, CARLA PASS)", fn)):
        print(name)
        for r in L:
            print(f"  {r['id']}: surrogate r {r['surrogate']['r']:+.2f} {r['surrogate']['fail']}; CARLA r "
                  f"{r['carla']['r']:+.2f} {r['carla']['reason'][:70]}")
    inv = [json.loads(l) for f in sorted(glob.glob(os.path.join(cd, "results_shard*.jsonl"))) for l in open(f)
           if l.strip() and json.loads(l)["carla"]["label"] == "INVALID"]
    if inv:
        print(f"{len(inv)} INVALID runs (not scored)")


def cmd_summarize(args):
    """<rc>/summary/: recovery_profiles.json (per variation and family: the label - CARLA's where that candidate ran,
    else the surrogate's verdict - and the policy; the variation's recovering families or its mitigation) and
    carla_runs.json (one row per CARLA run, as the team index)."""
    sd, cd = batch_dirs(args)
    S = {s["variation_id"]: s for f in sorted(glob.glob(os.path.join(sd, "scenarios_*.json"))) for s in json.load(open(f))}
    P = {p["variation_id"]: p for f in sorted(glob.glob(os.path.join(sd, "profiles_*.json"))) for p in json.load(open(f))}
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(cd, "results_shard*.jsonl"))) for l in open(f) if l.strip()]
    ran = {(r["variation_id"], r["family"], r["kind"]): r for r in R}
    sp = os.path.join(cd, "stability", "stability.json")
    stab = json.load(open(sp)) if os.path.exists(sp) else {}     # rerun --uncertain: is CARLA's label stable?
    out = []
    for vid, p in P.items():
        fam = {}
        for m, f in p["families"].items():
            # the family's CARLA candidates (best, boundary): it recovers if any of them passed in CARLA; the reported
            # one is a passing candidate (stable label first, then the larger CARLA margin), else the best
            C = [ran[k] for k in ((vid, m, "best"), (vid, m, "boundary"))
                 if k in ran and ran[k]["carla"]["label"] in ("PASS", "FAIL")]
            ok = [r for r in C if r["carla"]["label"] == "PASS"]
            r = (max(ok, key=lambda r: (stab.get(r["id"], {}).get("stable", True), r["carla"]["r"])) if ok
                 else (C[0] if C else None))
            lab = r["carla"]["label"] if r else None
            src = (f["best"] if r is None or r["kind"] == "best" else f.get("boundary")) or f["best"]
            fam[m] = dict(label=lab or f["verdict"], source="carla" if lab else "surrogate", surrogate=f["verdict"],
                          policy=r["theta_p"] if r else f["best"]["cfg"], kind=r["kind"] if r else "best",
                          r_surrogate=src["r"], r_carla=r["carla"].get("r") if lab else None,
                          stable=stab[r["id"]]["stable"] if r and r["id"] in stab else None,
                          contact=(r["carla"].get("contact") if lab else f["best"]["contact"]),
                          impact_speed=(r["carla"].get("impact_speed") if lab else f["best"]["impact_speed"]),
                          run=r["id"] if r else None)
        rec = [m for m, f in fam.items() if f["label"] == "PASS"]
        unv = [m for m, f in fam.items() if f["label"] == "UNCERTAIN"]
        if rec:
            status, mit = "recoverable", None
        else:
            status = "unverified" if unv else "unrecoverable"
            key = lambda m: (fam[m]["contact"] is None, -(fam[m]["impact_speed"] or 0.0))
            m = max(fam, key=key)
            mit = dict(family=m, policy=fam[m]["policy"], contact=fam[m]["contact"], impact_speed=fam[m]["impact_speed"],
                       source=fam[m]["source"])
        s = S[vid]
        out.append(dict(variation_id=vid, site=s["site"], split=s["split"], scenario_params=s["scenario_params"],
                        status=status, recovering=rec, mitigation=mit, families=fam))
    d = os.path.join(args.out, args.rc, "summary")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"recovery_profiles_{args.batch}.json"), "w"), indent=1, default=float)
    idx = [dict(case=args.rc, batch=args.batch, run_id=r["id"], variation_id=r["variation_id"], family=r["family"],
                kind=r["kind"], selected_for=r["why"], theta_s=r["theta_s"], theta_p=r["theta_p"],
                history_id=r.get("history_id", r["variation_id"]), hidden_world=r.get("hidden_world"),
                surrogate=r["surrogate"], carla_label=r["carla"]["label"], carla_reason=r["carla"].get("reason"),
                contact=r["carla"].get("contact"), evidence_dir=f"results_search/{args.rc}/carla/{args.batch}/runs/{r['id']}")
           for r in R]
    json.dump(idx, open(os.path.join(d, f"carla_runs_{args.batch}.json"), "w"), indent=1, default=float)
    from collections import Counter
    print(f"{len(out)} variations: {dict(Counter(o['status'] for o in out))}; {len(idx)} CARLA runs")
    sets = Counter((o["split"], " + ".join(o["recovering"]) if o["recovering"] else f"none -> {o['mitigation']['family']}")
                   for o in out)
    for (sp, k), n in sorted(sets.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {sp:<8} {n:4d}  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "rerun", "search", "score", "summarize"])
    ap.add_argument("--rc", default="rc07")
    ap.add_argument("--batch", default="baseline")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--frac", type=float, default=1.0, help="select: share of the clear-cut rest sent to CARLA")
    ap.add_argument("--complete", action="store_true", help="select: append the candidates not yet selected")
    ap.add_argument("--ids", default="", help="rerun: comma-separated run_ids")
    ap.add_argument("--keep", default="", help="select / rerun: file of history ids (one per line) to keep")
    ap.add_argument("--skip-town", default="", help="run: comma-separated towns to leave out (e.g. Town13)")
    ap.add_argument("--uncertain", action="store_true", help="rerun: every candidate the surrogate abstained on")
    ap.add_argument("--near", type=float, default=0.0, help="rerun: candidates whose CARLA margin is within this of 0.05")
    ap.add_argument("--starts", default="0.6:0.15,-0.6:-0.15", help="rerun: ego start offsets dx:dv,...")
    ap.add_argument("--tag", default="stability", help="rerun: output folder under carla/<batch>/")
    ap.add_argument("--variation", default="", help="search: variation id")
    ap.add_argument("--family", default="", help="search: family")
    ap.add_argument("--iters", type=int, default=6, help="search: CEM iterations")
    ap.add_argument("--seed", type=int, default=0, help="search: CEM seed")
    args = ap.parse_args()
    {"select": cmd_select, "run": cmd_run, "rerun": cmd_rerun, "search": cmd_search, "score": cmd_score,
     "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
