"""Add scene_name, hidden_settings and the constraint block (rcx/constraints.py) to the summary.json of every existing run, from the recorded
ego_observable.csv + ground_truth.csv + summary.json (no CARLA needed). New runs get the same fields from rcx/common.py:run_engine.
Check: constraints_pass must agree with the CARLA label (PASS <=> all constraints passed) for every run; disagreements are listed and NOT written.
python tools/backfill_constraints.py [--dry-run] [--root results_search]"""
import argparse, csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcx import constraints

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_search"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()


def val(x):
    if x in ("True", "False"): return x == "True"
    try: return float(x)
    except ValueError: return x


def load_rows(d):
    ego = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(d, "ego_observable.csv")))}
    out = []
    for g in csv.DictReader(open(os.path.join(d, "ground_truth.csv"))):
        r = dict(ego[int(g["frame"])]); r.update(g); out.append({k: val(v) for k, v in r.items()})
    return out


n = bad = 0; skipped = {}
for f in sorted(glob.glob(os.path.join(a.root, "*", "carla", "*", "runs", "*", "summary.json"))):
    d = os.path.dirname(f); s = json.load(open(f)); case = s["rc_case"]
    ann = constraints.annotate(case, s, load_rows(d) if s["label"] not in ("INVALID", "UNRESOLVED") else [])
    n += 1
    ok = ann["constraints_pass"] is None or ann["constraints_pass"] == (s["label"] == "PASS")
    if not ok:
        bad += 1; failed = [c["name"] for c in ann["constraints"] if not c["passed"]]
        print(f"MISMATCH {d}: label {s['label']} constraints_pass {ann['constraints_pass']} failed={failed} | {s['reason'][:100]}"); continue
    if not a.dry_run: s.update(ann); json.dump(s, open(f, "w"), indent=2, default=float)
print(f"{n} runs, {bad} mismatches" + (" (dry run, nothing written)" if a.dry_run else ""))
