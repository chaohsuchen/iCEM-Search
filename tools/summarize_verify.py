"""Summarize surrogate-vs-CARLA verification: python tools/summarize_verify.py <dir with results_shard*.jsonl> [--list]"""
import glob, json, sys
import numpy as np
d = sys.argv[1]; R = [json.loads(l) for f in sorted(glob.glob(f"{d}/results_shard*.jsonl")) for l in open(f)]
R = sorted({r["id"]: r for r in R}.values(), key=lambda r: r["id"])
err = [r for r in R if "error" in r]; R = [r for r in R if "carla" in r]
if "--list" in sys.argv:
    f = lambda x: float("nan") if x is None else x
    for r in R:
        s, c = r["surrogate"], r["carla"]
        print(f"{r['id']:44s} sur {s['verdict']:4s} {f(s['min_front']):6.2f}/{f(s['min_rear']):6.2f} | carla {c['label']:7s} {f(c['min_front']):6.2f}/{f(c['min_rear']):6.2f} | {c['reason'][:70]}")
valid = [r for r in R if r["carla"]["label"] in ("PASS", "FAIL")]
sp = [r for r in valid if r["surrogate"]["verdict"] == "PASS"]; fs = [r for r in sp if r["carla"]["label"] != "PASS"]
sf = [r for r in valid if r["surrogate"]["verdict"] != "PASS"]
print(f"{d}: results {len(R)}  errors {len(err)}  valid {len(valid)}  INVALID/UNRESOLVED {len(R)-len(valid)}")
print(f"  surrogate PASS {len(sp)} -> CARLA not PASS {len(fs)}  (false-safe {len(fs)/max(1,len(sp)):.1%});  surrogate FAIL {len(sf)} -> CARLA PASS {sum(r['carla']['label']=='PASS' for r in sf)}")
for k in ("best", "boundary"):
    q = [r for r in sp if r["kind"].startswith(k)]; print(f"  {k}: surrogate PASS {len(q)}, false-safe {sum(r['carla']['label']!='PASS' for r in q)}")
for r in fs: print("  FALSE-SAFE:", r["id"], r["carla"]["reason"][:80])
for r in R:
    if r["carla"]["label"] not in ("PASS", "FAIL"): print("  not valid:", r["id"], r["carla"]["label"], r["carla"]["reason"][:90])
