"""(vsearch) RC19 batches: anchor scenes of a searched batch re-run under the benign cases A-D and the semantic control S
(carla_rc/teachers/rc19_opts.py). No new search: each anchor's candidates (best / boundary / shared of every family,
the searched scene only) are run in all five cases. The surrogate's prediction for A-D is the anchor's own (the
benign objects are outside the observation and never interact); for S it is the nominal surrogate rollout in the S
scene (verdict PASS / FAIL, no robust worlds: `nominal_only`).

Writes results_search/rc19/carla/on_<rc>/selected.json (carla_verify.py run / rerun read it) and
results_search/rc19/search/on_<rc>/anchors.json.

    python -m search.rc19_batch rc03 vsearch --n 12
    python -m search.rc19_batch rc10 vsearch_rear --n 12
    python -m search.rc19_batch rc07 vsearch --n 12
"""
import argparse
import glob
import json
import os

import numpy as np

from . import kent_bridge as kb
from .kent_cases import CASES

KINDS = ("best", "boundary", "shared")


def pick(P, n, rng):
    """n anchors from the random stage: half recoverable, the rest unrecoverable / uncertain, spread over sites."""
    rows = [p for p in P if "~" not in p["variation_id"] and "_s" in p["variation_id"]]
    rec = [p for p in rows if p["verdict"]["status"] == "recoverable"]
    oth = [p for p in rows if p["verdict"]["status"] != "recoverable"]
    rng.shuffle(rec), rng.shuffle(oth)
    k = n // 2
    return rec[:k] + oth[:n - len(rec[:k])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rc", choices=["rc03", "rc10", "rc07"])
    ap.add_argument("batch")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args()
    case = CASES[args.rc]()
    sd = os.path.join(kb.OUT, args.rc, "search", args.batch)
    P = [p for f in sorted(glob.glob(os.path.join(sd, "profiles_*.json"))) for p in json.load(open(f))]
    C = [c for f in sorted(glob.glob(os.path.join(sd, "candidates_*.json"))) for c in json.load(open(f))]
    anchors = pick(P, args.n, np.random.default_rng(args.seed))
    A = {p["variation_id"] for p in anchors}
    sel = []
    for c in C:
        if c["variation_id"] not in A or c.get("hidden_world") or c["kind"] not in KINDS:
            continue
        for cs in "ABCDS":
            th = dict(c["theta_s"], case=cs)
            sur = dict(c["surrogate"])
            if cs == "S":
                o = kb.sim("RC19-" + c["sid"], th, c["family"], kb.full_cfg(c["family"], c["theta_p"]))
                sur = dict(verdict="PASS" if o["success"] else "FAIL", nominal="PASS" if o["success"] else "FAIL",
                           r=o["r"], fail=o["fail"], contact=o["contact"], impact_speed=o["impact_speed"],
                           nominal_only=True)
            sel.append(dict(c, sid="RC19-" + c["sid"], theta_s=th, variation_id=f"{c['variation_id']}_{cs}",
                            history_id=f"{args.rc}_{c['variation_id']}", hidden_world=None, surrogate=sur,
                            rc19=dict(base_rc=args.rc, base_batch=args.batch, anchor=c["variation_id"], case=cs),
                            why=["rc19_" + cs]))
    sel.sort(key=lambda c: (c["rc19"]["anchor"], c["family"], c["kind"], c["rc19"]["case"]))
    for i, c in enumerate(sel):
        c["run_id"] = f"{i:03d}_{c['variation_id']}_{c['family'].replace('>', '-')}_{c['kind']}"
    b = f"on_{args.rc}"
    for d in ("carla", "search"):
        os.makedirs(os.path.join(kb.OUT, "rc19", d, b), exist_ok=True)
    json.dump(sel, open(os.path.join(kb.OUT, "rc19", "carla", b, "selected.json"), "w"), indent=1, default=float)
    json.dump([dict(variation_id=p["variation_id"], status=p["verdict"]["status"],
                    recovering=p["verdict"]["recovering"]) for p in anchors],
              open(os.path.join(kb.OUT, "rc19", "search", b, "anchors.json"), "w"), indent=1)
    from collections import Counter
    print(f"rc19 on {args.rc}/{args.batch}: {len(anchors)} anchors, {len(sel)} runs",
          dict(Counter(p["verdict"]["status"] for p in anchors)),
          "S verdicts", dict(Counter(c["surrogate"]["verdict"] for c in sel if c["rc19"]["case"] == "S")))


if __name__ == "__main__":
    main()
