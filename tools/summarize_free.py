"""Summary of the RC04 free-form pilot (docs/FREE_SEARCH_PROPOSAL.md): surrogate stage (always) and CARLA stage (when results_free/rc04/carla/<batch>/results_shard*.jsonl exist).
python tools/summarize_free.py   ->  prints markdown tables and writes results_free/rc04/summary.json"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = "results_free/rc04"; M = json.load(open(f"{R}/manifest.json")); out = {}


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def load_carla(batch):
    rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/carla/{batch}/results_shard*.jsonl")) for l in open(f)]
    return {r["id"]: r for r in rows if "carla" in r}


def pkey(c): return json.dumps(c["theta_p"], sort_keys=True)


# ---------------- surrogate stage ----------------
print("## Surrogate stage (iCEM, no CARLA)\n")
print("| scene | role | struct best (brake / lane_change / staged) | free supported / unique samples | of which warm / cold | diverse kept | min pair dist | free best robust cost | struct best robust cost | kept supported robust over worlds |")
print("|---|---|---|---|---|---|---|---|---|---|")
sur = {}
for sc, spec in M["scenes"].items():
    fr = json.load(open(f"{R}/search/free_pilot/{sc}.json")); pr = json.load(open(f"{R}/search/struct_matched/{sc}/profiles.json"))
    sv = " / ".join(next(p["verdict"] for p in pr if p["family"] == f) for f in ("brake", "lane_change", "staged"))
    cands = [c for c in fr["candidates"] if c["stratum"] == "icem_selected" and c["kind"] in ("best", "diverse", "boundary") and c["surrogate"]["verdict"] == "PASS"]
    org = [c["search"].get("origin", "?") for c in cands]
    rob = [c["surrogate"].get("robust_over_worlds") for c in cands]
    sup = [c for c in fr["candidates"] if c["stratum"] == "icem_selected" and c["kind"] in ("best", "diverse")]
    d = [c for c in fr["candidates"] if c["kind"] == "diverse" and c["stratum"] == "icem_selected"]
    best = next((c["search"]["robust_cost"] for c in fr["candidates"] if c["kind"] == "best"), None)
    sb = min(p["robust_cost"] for p in pr)
    sur[sc] = dict(role=spec["role"], struct=sv, free_supported=fr["pool"]["n_supported_nominal"], unique=fr["budget"]["unique_samples"], free_best_cost=best, struct_best_cost=sb, n_diverse=len(d),
                   evals_free=fr["budget"]["robust_evals"], evals_cold=fr["budget"]["robust_evals_cold"], evals_warm=fr["budget"]["robust_evals_warm"], kept_supported=len(cands), kept_robust=int(sum(bool(x) for x in rob)))
    print(f"| {sc} | {spec['role']} | {sv} | {fr['pool']['n_supported_nominal']} / {fr['budget']['unique_samples']} | warm {sum(o.startswith('warm') for o in org)} / cold {sum(o == 'cold' for o in org)} (kept) | {len(d)} | {min([c["search"]["behaviour_distance_to_selected"] for c in d], default=float("nan")):.2f} | {best:.1f} | {sb:.1f} | {sum(bool(x) for x in rob)} / {len(rob)} |")
out["surrogate"] = sur
print(f"\nBudget per scene: free {sur['near_obstacle']['evals_cold']} cold + {sur['near_obstacle']['evals_warm']} warm robust evaluations; structured families about 1350 (3 families x 2 seeds), each robust evaluation = 5 surrogate rollouts.\n")

# ---------------- CARLA stage ----------------
free_c = json.load(open(f"{R}/search/free_pilot/candidates.json")); st_c = json.load(open(f"{R}/search/struct_matched/candidates.json"))
fres, sres = load_carla("free_pilot"), load_carla("struct_matched")
if not fres and not sres: print("(no CARLA results yet)"); json.dump(out, open(f"{R}/summary.json", "w"), indent=1); sys.exit()
lab = lambda res, i, c: res.get(next((k for k in res if k.startswith(f"{i:03d}_")), None), {}).get("carla", {}).get("label")
for c_list, res in ((free_c, fres), (st_c, sres)):
    for i, c in enumerate(c_list): c["carla_label"] = lab(res, i, c)
print("## CARLA stage\n")
print(f"CARLA-run candidates: free {sum(c['carla_label'] is not None for c in free_c)} of {len(free_c)}, structured {sum(c['carla_label'] is not None for c in st_c)} of {len(st_c)}\n")
# surrogate transfer (nominal world only: stratum random_prior / icem_selected, world w0)
print("### Surrogate transfer (false-safe = surrogate PASS, CARLA not PASS; false-negative = surrogate FAIL, CARLA PASS; INVALID/missing excluded)\n")
print("| family / stratum | surrogate PASS: n | false-safe | 95% CI | surrogate FAIL: n | false-negative | 95% CI |"); print("|---|---|---|---|---|---|---|")
tr = {}
for name, cl in (("free random_prior", [c for c in free_c if c["stratum"] == "random_prior"]), ("free iCEM-selected (w0)", [c for c in free_c if c["stratum"] == "icem_selected"]),
                 ("free world replays", [c for c in free_c if c["stratum"] == "world_replay"]), ("structured iCEM-selected (w0)", [c for c in st_c if c["stratum"] == "icem_selected"]),
                 ("structured world replays", [c for c in st_c if c["stratum"] == "world_replay"])):
    cl = [c for c in cl if c["carla_label"] in ("PASS", "FAIL", "UNRESOLVED")]
    sp = [c for c in cl if c["surrogate"]["verdict"] == "PASS"]; sf = [c for c in cl if c["surrogate"]["verdict"] != "PASS"]
    fs = sum(c["carla_label"] != "PASS" for c in sp); fn = sum(c["carla_label"] == "PASS" for c in sf)
    tr[name] = dict(sur_pass=len(sp), false_safe=fs, ci_fs=wilson(fs, len(sp)), sur_fail=len(sf), false_neg=fn, ci_fn=wilson(fn, len(sf)))
    print(f"| {name} | {len(sp)} | {fs} ({fs / len(sp):.0%}) | {wilson(fs, len(sp))} | {len(sf)} | {fn} ({fn / max(1, len(sf)):.0%}) | {wilson(fn, len(sf))} |" if sp else f"| {name} | 0 | - | - | {len(sf)} | {fn} | {wilson(fn, len(sf))} |")
out["transfer"] = tr
# per scene coverage
print("\n### Coverage per scene (nominal world w0, CARLA labels)\n")
print("| scene | role | struct PASS (best+boundary) / run | free PASS (best, diverse, boundary) / run | free refuted / mitigation / random PASS | free supported in all worlds | struct supported in all worlds |"); print("|---|---|---|---|---|---|---|")
cov = {}
for sc, spec in M["scenes"].items():
    f0 = [c for c in free_c if c["scene"] == sc and c["world_id"] == "w0" and c["carla_label"]]; s0 = [c for c in st_c if c["scene"] == sc and c["world_id"] == "w0" and c["carla_label"]]
    fp = [c for c in f0 if c["kind"] in ("best", "diverse", "boundary") and c["carla_label"] == "PASS"]; fr_ = [c for c in f0 if c["kind"] in ("best", "diverse", "boundary")]
    sp = [c for c in s0 if c["carla_label"] == "PASS"]

    def robust(cl_all, base):
        rep = [c for c in cl_all if c["scene"] == sc and pkey(c) == pkey(base) and c["family"] == base["family"] and c["carla_label"]]
        return len(rep) > 0 and all(c["carla_label"] == "PASS" for c in rep) and len(rep) == sum(1 for c in cl_all if c["scene"] == sc and pkey(c) == pkey(base) and c["family"] == base["family"])
    frob = [c for c in fp if robust(free_c, c)]; srob = [c for c in sp if robust(st_c, c)]
    oth = lambda kind: [c["carla_label"] for c in f0 if c["kind"] == kind]
    cov[sc] = dict(struct_pass=len(sp), struct_run=len(s0), free_pass=len(fp), free_run=len(fr_), free_robust=len(frob), struct_robust=len(srob), refuted=oth("refuted"), mitigation=oth("mitigation"),
                   random_pass=sum(c["carla_label"] == "PASS" for c in f0 if c["kind"] == "random"), random_run=sum(1 for c in f0 if c["kind"] == "random"))
    print(f"| {sc} | {spec['role']} | {len(sp)} / {len(s0)} | {len(fp)} / {len(fr_)} | {oth('refuted')} / {oth('mitigation')} / {cov[sc]['random_pass']} of {cov[sc]['random_run']} | {len(frob)} | {len(srob)} |")
out["coverage"] = cov
json.dump(out, open(f"{R}/summary.json", "w"), indent=1, default=float)
