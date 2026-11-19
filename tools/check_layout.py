"""Check that results_search/ follows the naming rules of results_search/README.md. Prints violations (exit code 1 if any) and a one-line status per batch."""
import glob, os, re, sys
R = "results_search"; bad = []
BATCH = re.compile(r"^(baseline|presets_[a-z0-9]+|boundary_r\d+)$"); RC = re.compile(r"^rc\d\d$"); RUN = re.compile(r"^\d{3}_[A-Za-z0-9_.@-]+_(best|boundary|ref)$")
import csv, json
SUMMARY_KEYS = {"schema", "run_id", "rc", "rc_case", "variation_id", "candidate_kind", "family", "scenario_params", "policy_params", "policy_source", "scene_seed", "label", "reason", "end_reason", "metrics", "contact", "files"}
EGO_COLS = {"frame", "sim_time", "t_since_cue", "ego_s", "ego_t", "speed", "a_req", "throttle", "brake", "steer", "contact"}
GT_COLS = {"frame", "sim_time", "contact_actor"}
DEC_KEYS = {"frame", "query_time", "observation", "observation_history_frames", "policy_output", "controller_state", "applied_command", "pending_commands"}


def check_run(d, tag):
    """Format freeze (schema rcx-run-1): every run folder must have these keys / columns."""
    try:
        s = json.load(open(f"{d}/summary.json")); m = SUMMARY_KEYS - set(s)
        if s.get("schema") != "rcx-run-1": bad.append(f"{tag}: schema != rcx-run-1")
        if m: bad.append(f"{tag}: summary.json missing {sorted(m)}")
        m = EGO_COLS - set(next(csv.reader(open(f"{d}/ego_observable.csv")))); m and bad.append(f"{tag}: ego_observable.csv missing {sorted(m)}")
        m = GT_COLS - set(next(csv.reader(open(f"{d}/ground_truth.csv")))); m and bad.append(f"{tag}: ground_truth.csv missing {sorted(m)}")
        m = DEC_KEYS - set(json.loads(open(f"{d}/decisions.jsonl").readline())); m and bad.append(f"{tag}: decisions.jsonl missing {sorted(m)}")
    except Exception as ex: bad.append(f"{tag}: unreadable evidence ({ex})")


NEW = {"summary.json", "ego_observable.csv", "ground_truth.csv", "decisions.jsonl", "run.mp4"}
for e in sorted(os.listdir(R)):
    if e in ("README.md", "logs", "summary") or RC.match(e): continue
    bad.append(f"unexpected top-level entry: {e}")
for rc in sorted(e for e in os.listdir(R) if RC.match(e)):
    for sub in os.listdir(f"{R}/{rc}"):
        if sub not in ("search", "carla", "scans"): bad.append(f"{rc}/{sub}: not one of search/ carla/ scans/")
    for kind, need in (("search", ["candidates.json", "scenarios.json"]), ("carla", [])):
        for b in sorted(glob.glob(f"{R}/{rc}/{kind}/*")):
            n = os.path.basename(b)
            if not BATCH.match(n): bad.append(f"{rc}/{kind}/{n}: batch name must match baseline | presets_<tag> | boundary_r<N>"); continue
            for f in need:
                if not os.path.exists(f"{b}/{f}"): bad.append(f"{rc}/{kind}/{n}: missing {f}")
            if kind == "carla":
                if not glob.glob(f"{b}/results_shard*.jsonl"): bad.append(f"{rc}/carla/{n}: no results_shard*.jsonl")
                runs = sorted(os.listdir(f"{b}/runs")) if os.path.isdir(f"{b}/runs") else []; legacy = 0
                for r in runs:
                    if not RUN.match(r): bad.append(f"{rc}/carla/{n}/runs/{r}: run folder name"); continue
                    have = set(os.listdir(f"{b}/runs/{r}"))
                    if not NEW <= have: legacy += 1
                    else: check_run(f"{b}/runs/{r}", f"{rc}/carla/{n}/runs/{r}")
                    if "telemetry.csv" in have and NEW <= have: bad.append(f"{rc}/carla/{n}/runs/{r}: redundant telemetry.csv")
                print(f"{rc}/carla/{n}: {len(runs)} runs, {legacy} legacy-format (pending re-run)")
for f in sorted(glob.glob(f"{R}/logs/*")):
    if not re.match(r"^cem_rc\d\d_(baseline|presets_[a-z0-9]+|boundary_r\d+)\.log$", os.path.basename(f)): bad.append(f"logs/{os.path.basename(f)}: name must be cem_<rc>_<batch>.log")
print("\n".join(["VIOLATIONS:"] + bad) if bad else "layout OK"); sys.exit(1 if bad else 0)
