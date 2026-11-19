#!/usr/bin/env python
"""Matched sweeps for RC01 / RC17 / RC05 / RC09 / RC13.   python run_v2.py --port P --group rc05 --out results_v3
(Legacy driver; it produced the CARLA calibration files in results_v3/<rc>/calibration*.json that the surrogates load. Current batches are run by tools/verify_carla.py.)"""
import argparse, copy, json, os, time, types
import carla
from rc01.config import ScenarioConfig, PolicyConfig
from rc01.calibrate import calibrate

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
ap.add_argument("--group", required=True); ap.add_argument("--out", default="results_v3")
ap.add_argument("--only", default=""); ap.add_argument("--gap17", type=float, default=17.0)
ap.add_argument("--delays17", default="0.0,0.1,0.3")
a = ap.parse_args()
dt = 0.05
client = carla.Client(a.host, a.port); client.set_timeout(900)
only = [x for x in a.only.split(",") if x]
want = lambda rid: not only or any(o in rid for o in only)
root = os.path.join(a.out, a.group); os.makedirs(root, exist_ok=True)


def load(town):
    w = client.load_world(town)
    s = w.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = True, dt
    from rc01.calibrate import physics_settings; physics_settings(s, dt)
    w.apply_settings(s); w.set_weather(carla.WeatherParameters.ClearNoon)
    for _ in range(10): w.tick()
    return w


# ---------------------------------------------------------------- RC01 / RC17 (Town01 narrow one-way street, shared engine)
def street_group():
    from rcx.common import run_engine
    from rcx.scenarios import RC01
    from rcx.policies import SafeStopAdapter
    world = load("Town01")
    import rc01.calibrate as _cal
    calibs = {}

    def go(sweep, label, **kw):
        rid = f"{a.group}_{sweep}_{label}"
        if not want(rid): return
        v = kw.get("v_ego", 8.0)
        if v not in calibs:
            scn0 = RC01(world); fr = scn0.lane_frame(scn0.p["road"], scn0.p["lane"], scn0.p["s0"])
            cfg = types.SimpleNamespace(dt=dt, target_speed=v, lane_center=0.0, ego_bp=scn0.p["ego_bp"])
            calibs[v] = calibrate(world, cfg, os.path.join(root, f"calibration_v{v:g}.json"), frame=fr, x0=6.0)
            print("calibration v=%s" % v, {k: calibs[v][k] for k in ("brake_cap", "coast_decel", "max_decel", "cruise_throttle")}, flush=True)
        scn = RC01(world, **kw)
        pcfg = PolicyConfig(target_speed=v)
        s = run_engine(world, scn, SafeStopAdapter(dt, pcfg), pcfg, calibs[v], rid, os.path.join(root, sweep, rid))
        print(rid, s["label"], "|", s["reason"][:100], "| start v/gap", s.get("start_speed") and round(s["start_speed"], 2), s.get("start_gap") and round(s["start_gap"], 2),
              "| stop gap", s.get("stopping_gap"), "| clear", round(s["metrics"].get("clear_width_measured", 0), 2), flush=True)

    if a.group == "rc01":
        go("baseline", "g20_c1.5_ped3_v8", clear=1.5)
        for g in (16.0, 12.0): go("gap", f"g{g:g}", gap=g, clear=1.5)
        for c in (1.2, 1.9): go("clear", f"c{c:g}", clear=c)
        for n in (0, 6): go("ped", f"ped{n}", n_per_side=n, clear=1.5)
        for v in (6.0, 9.5): go("speed", f"v{v:g}", v_ego=v, clear=1.5)
    else:
        for d in [float(x) for x in a.delays17.split(",")]:
            go("delay", f"g{a.gap17:g}_delay{d:g}", gap=a.gap17, delay_s=d, clear=1.5)


# ---------------------------------------------------------------- RC05 / RC09 / RC13 (real maps)
def real_group():
    from rcx.common import run_engine
    from rcx.scenarios import RC05, RC09, RC13
    from rcx.policies import LaneChangeAbortPolicy, StopPolicy, ReachableSetPolicy
    spec = {"rc05": (RC05, 10.0, "Town04"), "rc09": (RC09, 8.0, "Town04"), "rc13": (RC13, 8.0, "Town03_Opt")}[a.group]
    cls, v, town = spec
    world = load(town)
    scn0 = cls(world); fr = scn0.lane_frame(scn0.p["road"], scn0.p["lane"], scn0.p["s0"])
    cfg = types.SimpleNamespace(dt=dt, target_speed=v, lane_center=0.0, ego_bp=scn0.p["ego_bp"])
    cal = calibrate(world, cfg, os.path.join(root, "calibration.json"), frame=fr, x0=6.0)
    print(a.group, "calibration", {k: cal[k] for k in ("brake_cap", "coast_decel", "max_decel", "cruise_throttle")}, flush=True)
    if a.group == "rc05":
        plan = [("baseline", "c2_g25_p0.5", {})] + [("closing", f"c{c}", {"closing": c}) for c in (0.5, 4.5)] + \
               [("rear_gap", f"g{g}", {"rear_gap": g}) for g in (12.0, 40.0)] + [("progress", f"p{p}", {"hazard_progress": p}) for p in (0.2, 0.8)]
        mk = lambda kw: LaneChangeAbortPolicy(dt, mode="abort")
    elif a.group == "rc09":
        plan = [("baseline", "moderate_f24_r18_c4_d1.4", {})] + [("front_gap", f"f{g}", {"front_gap": g}) for g in (18.0, 32.0)] + \
               [("rear_gap", f"r{g}", {"rear_gap": g}) for g in (12.0, 26.0)] + [("closing", f"c{c}", {"closing": c}) for c in (2.0, 6.0)] + \
               [("rear_delay", f"d{d}", {"rear_delay_s": d}) for d in (0.8, 2.0)] + [("reference_hard", "hard_baseline", {"_mode": "hard"})]
        mk = lambda kw: StopPolicy(dt, mode=kw.get("_mode", "moderate"))
    else:
        plan = [("pair", "vped1.8", {"_v": 1.8}), ("pair", "vped3.0", {"_v": 3.0})]
        mk = lambda kw: ReachableSetPolicy(dt, v_ped_max=kw["_v"], horizon=2.0)
    for sweep, label, kw in plan:
        rid = f"{a.group}_{sweep}_{label}"
        if not want(rid): continue
        skw = {k: x for k, x in kw.items() if not k.startswith("_")}
        scn = cls(world, **skw)
        s = run_engine(world, scn, mk(kw), PolicyConfig(), cal, rid, os.path.join(root, sweep, rid))
        print(rid, s["label"], "|", s["reason"][:140], flush=True)


if a.group in ("rc01", "rc17"): street_group()
else: real_group()
