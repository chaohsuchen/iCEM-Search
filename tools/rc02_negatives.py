"""RC02 (too late to stop) = the rc01 scenes where no policy reaches PASS in CARLA. Reads summary/carla_runs.json, writes summary/rc02_unrecoverable.json:
one entry per rc01 scene (gap, delay_s, v_ego) with no PASS, listing each executed policy, its CARLA label/reason and whether it still hit the van (mitigation only:
impact-speed reduction is NOT recovery, 20RCs.md). No new CARLA runs are needed. python tools/rc02_negatives.py"""
import collections, json
rows = json.load(open("results_search/summary/carla_runs.json")); rows = rows if isinstance(rows, list) else rows["rows"]
g = collections.defaultdict(list)
for r in rows:
    if r["case"] != "rc01" or r["carla_label"] == "INVALID": continue
    th = r["theta_s"]; g[(th.get("gap", 20), th.get("delay_s", 0), th.get("v_ego", 8))].append(r)
out = []
for (gap, d, v), rs in sorted(g.items()):
    if any(r["carla_label"] == "PASS" for r in rs): continue
    out.append(dict(rc="RC02", source_rc="rc01", scene=dict(gap=gap, delay_s=d, v_ego=v), n_policies=len(rs), n_contact=sum(bool(r["contact"]) for r in rs),
                    runs=[dict(batch=r["batch"], run_id=r["run_id"], family=r["family"], kind=r["kind"], label=r["carla_label"], reason=r["carla_reason"], contact=bool(r["contact"]),
                               evidence_dir=r["evidence_dir"]) for r in rs]))
json.dump(out, open("results_search/summary/rc02_unrecoverable.json", "w"), indent=1)
print(len(out), "unrecoverable rc01 scenes ->", "results_search/summary/rc02_unrecoverable.json")
for e in out: print(e["scene"], e["n_policies"], "policies,", e["n_contact"], "with contact")
