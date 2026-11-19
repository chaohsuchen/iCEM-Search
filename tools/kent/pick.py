"""Pick the meaningful histories of a batch (searched scene + its hidden worlds) for CARLA per-tick runs.
pools (from the surrogate profile of the searched scene):
  defining  the RC's story holds (rc10: an observable recovery with the rear vehicle; rc11: no observable recovery;
            rc14: memory_left the only observable recovery; rc07: hand-back families split)
  contrast  the policy choice matters (some family PASS, some FAIL)
  hard      no family recovers (mitigation)
  easy      every family recovers
usage: pick.py <batch dir> <rc> <n> <stages: presets,s,b1> <out keep file>"""
import glob, json, os, sys
import numpy as np

d, rc, n, stages, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","), sys.argv[5]
rng = np.random.default_rng(0)
rows = []
for st in stages:
    pat = "s*" if st == "s" else st
    P = {p["variation_id"]: p for f in glob.glob(f"{d}/profiles_*_{pat}.json") for p in json.load(open(f))}
    S = {s["variation_id"]: s for f in glob.glob(f"{d}/scenarios_*_{pat}.json") for s in json.load(open(f))}
    for vid, p in P.items():
        if "~" in vid:
            continue
        v, fam = p["verdict"], p["families"]
        obs, verd = set(v.get("observable_recovering") or []), {f["verdict"] for f in fam.values()}
        th = S[vid]["scenario_params"]
        hb = [m for m in fam if ">" in m]
        if rc == "rc10":
            defining = bool(obs) and th.get("rear", 0) == 1
        elif rc == "rc11":
            defining = not obs
        elif rc == "rc14":
            defining = "memory_left" in obs and not obs & {"keep", "brake", "stop_end"}
        elif rc in ("rc03", "rc12"):                # braking fails, a lateral escape recovers
            rec = set(v["recovering"])
            defining = "brake" not in rec and bool(rec & {"left", "right", "left_brake", "right_brake"})
        elif rc == "rc16":                          # the body fits through the passage
            defining = "shift_pass" in set(v["recovering"])
        elif rc == "rc08":                          # the protected hold recovers, overtaking now does not
            rec = set(v["recovering"])
            defining = "stop_end" in rec and "overtake" not in rec
        else:
            defining = len({fam[m]["recovers"] for m in hb}) > 1
        pool = ("defining" if defining else "contrast" if {"PASS", "FAIL"} <= verd
                else "hard" if not v["recoverable"] else "easy" if verd == {"PASS"} else "contrast")
        rows.append((vid, st, pool))
if rc == "rc12":                                 # one row per scene: all its delays together
    import re
    G = {}
    for vid, st, pool in rows:
        g = re.sub(r"_d[0-9.]+$", "", vid) if "_p_" in vid else vid.rsplit("_", 1)[0] + "_" + str(int(vid.rsplit("_", 1)[1]) // 6)
        G.setdefault((g, st), []).append((vid, pool))
    rows = []
    for (g, st), m in G.items():
        recs = [pool != "hard" for _, pool in sorted(m)]
        pool = "defining" if recs and recs[0] and not recs[-1] else "hard" if not any(recs) else \
            "easy" if all(recs) else "contrast"
        rows.append(("|".join(vid for vid, _ in m), st, pool))
quota = dict(defining=0.4, contrast=0.35, hard=0.15, easy=0.1)
keep = []
for pool, q in quota.items():
    cand = [r for r in rows if r[2] == pool]
    by_st = {st: [r for r in cand if r[1] == st] for st in stages}   # spread over the stages
    for v in by_st.values():
        rng.shuffle(v)
    k = round(q * n)
    while k > 0 and any(by_st.values()):
        for st in stages:
            if k > 0 and by_st[st]:
                keep.append(by_st[st].pop()); k -= 1
short = n - len(keep)                                                # a pool ran short: fill from the rest
rest = [r for r in rows if r not in keep]
rng.shuffle(rest)
keep += rest[:max(short, 0)]
open(out, "w").write("\n".join(v for r in keep for v in r[0].split("|")) + "\n")
from collections import Counter
print(rc, os.path.basename(d), f"{len(keep)}/{len(rows)} histories", dict(Counter((r[2]) for r in keep)),
      dict(Counter(r[1] for r in keep)), "| all:", dict(Counter(r[2] for r in rows)))
