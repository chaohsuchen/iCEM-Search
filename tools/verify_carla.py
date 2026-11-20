"""Replay surrogate candidates in CARLA and record surrogate-vs-CARLA outcomes (false-safe tracking), for rc01 (incl. RC17 delays) and rc09.
python tools/verify_carla.py --case rc01 --port P --shard i --nshards n [--ids 1,2,3] [--out DIR]"""
import argparse, json, os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import carla
from rc01.config import PolicyConfig
from rc01.calibrate import calibrate, physics_settings
from rcx.common import run_engine
from search.cases import CASES

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
ap.add_argument("--batch", default="baseline", help="candidates from results_search/<rc>/search/<batch>/, runs to results_search/<rc>/carla/<batch>/"); ap.add_argument("--cands", default=None); ap.add_argument("--out", default=None); ap.add_argument("--ids", default="")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
cands_path = a.cands or f"results_search/{a.case}/search/{a.batch}/candidates.json"; out = a.out or f"results_search/{a.case}/carla/{a.batch}"; os.makedirs(out, exist_ok=True)
dt = 0.05
client = carla.Client(a.host, a.port); client.set_timeout(900)
TOWN = dict(rc01="Town01", rc09="Town04", rc13="Town03_Opt", rc05="Town04", rc03="Town04", rc04="Town04", rc06="Town04")[a.case]
world = client.load_world(TOWN)
s = world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = True, dt; physics_settings(s, dt)
world.apply_settings(s); world.set_weather(carla.WeatherParameters.ClearNoon)
for _ in range(10): world.tick()
if a.case == "rc01": from rcx.scenarios import RC01 as SCN
elif a.case == "rc09": from rcx.scenarios import RC09 as SCN
elif a.case == "rc13": from rcx.scenarios import RC13 as SCN
elif a.case == "rc05": from rcx.scenarios import RC05 as SCN
elif a.case == "rc03": from rcx.scenarios import RC03 as SCN
elif a.case in ("rc04", "rc06"): from rcx.scenarios import RC04 as SCN
cal_cache = {}
CASE = CASES[a.case]()


def calib_for(v):
    if v not in cal_cache:
        scn0 = SCN(world); fr = scn0.lane_frame(scn0.p["road"], scn0.p["lane"], scn0.p["s0"])
        cfg = types.SimpleNamespace(dt=dt, target_speed=v, lane_center=0.0, ego_bp=scn0.p["ego_bp"])
        cal_cache[v] = calibrate(world, cfg, os.path.join(out, f"calibration_{a.case}_v{v:g}_shard{a.shard}.json"), frame=fr, x0=6.0)
    return cal_cache[v]


def build(c):
    th, kw = dict(c["theta_s"]), c["theta_p"]
    v = th.get("v_ego", SCN.defaults["v_ego"])
    if a.case == "rc01": th.setdefault("clear", 1.5)
    if a.case in ("rc03", "rc04", "rc06"): kw = {"v_target": v, "esc_dir": -float(th.get("side", 1)), **kw}
    if c["family"] in ("brake", "two_stage", "surge_brake") and "v_target" not in kw: kw = {**kw, "v_target": v}
    return SCN(world, **th), CASE.make_policy(dt, c["family"], kw), PolicyConfig(target_speed=v), calib_for(v)


cands = json.load(open(cands_path))
res_path = os.path.join(out, f"results_shard{a.shard}.jsonl")
done = {r["id"] for r in map(json.loads, open(res_path)) if "carla" in r} if os.path.exists(res_path) else set()
want = {int(x) for x in a.ids.split(",")} if a.ids else None
for i, c in enumerate(cands):
    if i % a.nshards != a.shard or (want is not None and i not in want): continue
    cid = f"{i:03d}_{c['scenario']}_{c['family']}_{c['kind']}"
    if cid in done: continue
    t0 = time.time()
    try:
        scn, pol, pcfg, cal = build(c)
        extra = dict(rc_case=a.case, variation_id=c["scenario"], candidate_kind=c["kind"], family=c["family"], scenario_params=c["theta_s"], policy_params=c["theta_p"],
                     policy_source=c.get("search", {}), surrogate_prediction=c["surrogate"])
        extra.update({k: c[k] for k in ("policy_spec", "world_id", "scene", "role", "stratum") if k in c})      # free-form pilot (results_free): full policy description, hidden world id, scene role, candidate stratum
        sm = run_engine(world, scn, pol, pcfg, cal, cid, os.path.join(out, "runs", cid), meta_extra=extra); m = sm["metrics"]
        carla_r = dict(label=sm["label"], reason=sm["reason"], min_front=m.get("min_front_gap", m.get("min_gap", m.get("min_ped_dist", m.get("min_clearance_lead")))), min_rear=m.get("min_rear_gap", m.get("min_clearance_rear_left")),
                       stopping_gap=m.get("stopping_gap"), min_wall=m.get("min_wall_clearance"))
        r = dict(id=cid, scenario=c["scenario"], family=c["family"], kind=c["kind"], theta_s=c["theta_s"], theta_p=c["theta_p"], surrogate=c["surrogate"], carla=carla_r)
    except Exception as e:
        r = dict(id=cid, scenario=c["scenario"], family=c["family"], kind=c["kind"], error=repr(e))
    open(res_path, "a").write(json.dumps(r) + "\n")
    print(cid, r.get("carla", {}).get("label", r.get("error")), "| surrogate", c["surrogate"]["verdict"], f"({time.time() - t0:.0f}s)", flush=True)
