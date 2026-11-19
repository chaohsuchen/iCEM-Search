"""Flat index of every CARLA run: one row per executed policy, labelled by CARLA (surrogate prediction kept only as reference).
-> results_search/summary/carla_runs.json  (case, batch, run_id, scenario params, family, kind, policy params, surrogate verdict, CARLA label / reason / metrics, evidence folder)
Rows exist for all batches (baseline, presets_*, boundary*); the label is always the CARLA verdict, so runs made with an older surrogate stay valid data."""
import glob, json, os
R = "results_search"; rows = []
for f in sorted(glob.glob(f"{R}/*/carla/*/results_shard*.jsonl")):
    case, _, batch = f.split("/")[1:4]
    for l in open(f):
        r = json.loads(l)
        if "carla" not in r: continue
        d = f"{R}/{case}/carla/{batch}/runs/{r['id']}"; sm = json.load(open(f"{d}/summary.json")) if os.path.exists(f"{d}/summary.json") else {}
        m = sm.get("metrics", {})
        rows.append(dict(case=case, batch=batch, run_id=r["id"], scenario=r["scenario"], family=r["family"], kind=r["kind"], theta_s=r["theta_s"], theta_p=r["theta_p"],
                         surrogate=r["surrogate"], carla_label=r["carla"]["label"], carla_reason=r["carla"]["reason"], min_clear_by=m.get("min_clear_by"), contact=sm.get("contact", {}).get("actors"),
                         new_format=os.path.exists(f"{d}/ego_observable.csv"), evidence_dir=d if os.path.isdir(d) else None))
json.dump(rows, open(f"{R}/summary/carla_runs.json", "w"), indent=1)
import collections
c = collections.Counter((r["case"], r["batch"]) for r in rows)
for k, v in sorted(c.items()): print(k, v, "runs;", sum(r["new_format"] for r in rows if (r["case"], r["batch"]) == k), "in new format")
print(len(rows), "rows ->", f"{R}/summary/carla_runs.json")
