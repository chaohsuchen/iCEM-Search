"""Executed-behaviour diversity of the CARLA-PASS policies of the RC04 free-form pilot (nominal world w0): descriptor = speed / 5 and lateral offset at t = 1..7 s after the cue, taken from
ego_observable.csv; a set's size = number of greedy picks with pairwise RMS distance >= tau (tau = 0.5, the same as the selection in tools/run_free.py). Writes results_free/rc04/diversity.json."""
import csv, glob, json, os, sys
import numpy as np
R = "results_free/rc04"; TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
M = json.load(open(f"{R}/manifest.json")); out = {}


def desc(path):
    rows = [r for r in csv.DictReader(open(path)) if r["t_since_cue"] != ""]
    t = np.array([float(r["t_since_cue"]) for r in rows]); v = np.array([float(r["speed"]) for r in rows]); y = np.array([float(r["ego_t"]) for r in rows])
    idx = [int(np.argmin(np.abs(t - k))) for k in range(1, 8)]; return np.concatenate([v[idx] / 5.0, y[idx] / 1.0])


def cover(ds):
    sel = []
    for d in ds:
        if all(np.linalg.norm(d - s) / np.sqrt(len(d)) >= TAU for s in sel): sel.append(d)
    return len(sel)


for fam_group, batch in (("free", "free_pilot"), ("structured", "struct_matched")):
    cands = json.load(open(f"{R}/search/{batch}/candidates.json")); res = {}
    for f in glob.glob(f"{R}/carla/{batch}/results_shard*.jsonl"):
        for l in open(f): r = json.loads(l); res[r["id"][:3]] = r
    for i, c in enumerate(cands):
        r = res.get(f"{i:03d}")
        if not r or "carla" not in r or r["carla"]["label"] != "PASS" or c["world_id"] != "w0" or c["kind"] == "random": continue
        p = f"{R}/carla/{batch}/runs/{r['id']}/ego_observable.csv"
        if os.path.exists(p): out.setdefault(c["scene"], {}).setdefault(fam_group, []).append(desc(p))
print(f"| scene | CARLA-PASS policies free / structured | distinct behaviours (tau {TAU}) free / structured | distinct when pooled |"); print("|---|---|---|---|")
res_json = {}
for sc in M["scenes"]:
    d = out.get(sc, {}); f, s = d.get("free", []), d.get("structured", [])
    res_json[sc] = dict(n_free=len(f), n_struct=len(s), distinct_free=cover(f), distinct_struct=cover(s), distinct_pooled=cover(f + s))
    print(f"| {sc} | {len(f)} / {len(s)} | {cover(f)} / {cover(s)} | {cover(f + s)} |")
json.dump(res_json, open(f"{R}/diversity.json", "w"), indent=1)
