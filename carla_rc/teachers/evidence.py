"""Per-run evidence in the team's run-folder format (results_search README, schema "rcx-run-1"):

    summary.json         RC / variation ids, scenario and policy parameters, seeds, label + reason, metrics, contact
    ego_observable.csv   per tick (0.05 s): ego state, the policy's observation (obs_*), what it perceived of every
                         actor (seen_<actor>_*: visible flag + relative state, from the delivered sight-line snapshot),
                         requested (a_req, r_req) and applied (throttle / brake / steer) controls
    decisions.jsonl      per decision (0.1 s): query time, observation, observation_history_frames, policy output,
                         controller state, applied command, pending commands (the command held over the step)
    ground_truth.csv     per tick: true state of every actor (background included), its clearance to the ego, the
                         margins and the corridor bounds (hidden information: analysis / labels only)

`run(env, p, family, cfg)` executes one candidate on a TeacherEnv (CARLA) or SurrogateEnv and returns (outcome,
recorder); `write(rec, out_dir, meta)` writes the folder. The same code runs on both simulators, so a surrogate run and
its CARLA check have identical files.
"""
import csv
import json
import math
import os

import numpy as np

from laws import EGO_L, EGO_W, obb, signed_distance
from options import ModeExpert, hist_start

R_OK, C0, T_H = 0.05, 0.5, 0.3      # the success margin and env.py's clearance reserve / look-ahead (R1 block)

ACTOR_F = ["s", "t", "yaw_rel", "vs", "vt", "length", "width"]
OUT = os.environ.get("RC_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                 "results_search"))


def impact(env, info):
    """(actor, closing speed [m/s]) at the contact that ended the episode; (None, 0) without contact."""
    if not info["fail"].startswith(("contact", "collision")):
        return None, 0.0
    r = env.log["tick"][-1]
    e = r["ego"]
    name = info["fail"].split(":", 1)[1] if ":" in info["fail"] else r["near"]
    b = r["bodies"].get(name)
    if b is None:
        return name, float("nan")
    ve = np.array([e[3] * math.cos(e[2]), e[3] * math.sin(e[2])])
    return name, float(np.linalg.norm(ve - np.array(b[4:6])))


class Recorder:
    def __init__(self, env, ex):
        self.env, self.ex, self.decisions, self.frame0 = env, ex, [], None

    def act(self, obs):
        env, ex = self.env, self.ex
        z = ex.act(obs)
        phase = "prefix" if env.t < -1e-6 else ("handback" if ex.t_hb is not None else
                                                 ("stage1" if ex.t_det is not None else "before_trigger"))
        hist = [s for s in env.snaps if s[0] >= hist_start(env)] or env.snaps[-1:]
        frames = [self.frame_at(t) for t, _ in hist[-10:]]
        if not env.log["tick"]:                        # the first decision: the frame the warm-up ended on
            self.frame0 = env.world.get_snapshot().frame if hasattr(env, "world") else 0
        frame = env.log["tick"][-1]["frame"] if env.log["tick"] else self.frame0
        self.decisions.append(dict(
            frame=frame, query_time=round(env.t, 4),
            t_since_cue=None if ex.t_det is None else round(env.t - ex.t_det, 4),
            observation={n: float(v) for n, v in zip(env.spec.obs_names, np.asarray(obs, float))},
            observation_history_frames=frames, observation_history_len=len(hist),
            policy_output=dict(phase=phase, family=ex.mode, z=[float(v) for v in z],
                               handback=getattr(ex.pol, "choice", None) and ex.pol.choice[0]),
            controller_state=self.controller_state()))
        return z

    def frame_at(self, t):
        ticks = self.env.log["tick"]
        if not ticks or t < ticks[0]["t"] - 1e-6:          # the reset snapshot: the frame the warm-up ended on
            return self.frame0
        return min(ticks, key=lambda r: abs(r["t"] - t))["frame"]

    def controller_state(self):
        env = self.env
        st = dict(gov_k=float(getattr(env, "gov_k", 1.0)), app=[float(v) for v in getattr(env, "app", (0.0, 0.0))])
        lc = getattr(env, "long", None)
        if lc is not None:
            st.update(v_ref=float(lc.v_ref), i=float(lc.i))
        x = getattr(env, "x", None)                   # the surrogate's ego state (point mass with lags)
        if isinstance(x, dict):
            st.update(a_state=float(x["a"]), r_state=float(x["r"]))
        return st


def run(env, p, family, cfg):
    ex = ModeExpert(env, family, cfg)
    rec = Recorder(env, ex)
    obs, _ = env.reset(options={"params": p})
    ex.reset()
    done, info = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(rec.act(obs))
        done = term or trunc
    rec.info = info
    # the step log carries the command each decision produced: attach it, and the ticks it was held over
    ticks = env.log["tick"]
    k = 0
    for d, s in zip(rec.decisions, env.log["step"]):
        held = []
        while k < len(ticks) and ticks[k]["t"] <= s["t"] + 1e-6:
            held.append(ticks[k])
            k += 1
        d["policy_output"].update(a_req=float(s["a_cmd"]), r_req=float(s["r_cmd"]), envelope_req=[float(v) for v in s["req"]])
        d["applied_command"] = (dict(frame=held[0]["frame"], throttle=held[0]["ctrl"][0], brake=held[0]["ctrl"][1],
                                     steer=held[0]["ctrl"][2]) if held else None)
        d["pending_commands"] = [dict(frame=h["frame"], a_req=float(s["a_cmd"]), r_req=float(s["r_cmd"]),
                                      throttle=h["ctrl"][0], brake=h["ctrl"][1], steer=h["ctrl"][2]) for h in held]
    return info, rec


def _rel(e, b):
    """Actor state relative to the ego, in the lane frame (s along, t to the right)."""
    return dict(s=b[0] - e[0], t=b[1] - e[1], yaw_rel=b[2] - e[2], vs=b[4] - e[3] * math.cos(e[2]),
                vt=b[5] - e[3] * math.sin(e[2]))


def write(rec, out, meta):
    """The run folder `out` from a finished Recorder; meta = ids, parameters, surrogate prediction (summary.json)."""
    env, info, ex = rec.env, rec.info, rec.ex
    os.makedirs(out, exist_ok=True)
    ticks = env.log["tick"]
    names = list(env.bodies)
    sizes = {n: (env.bodies[n].LENGTH, env.bodies[n].WIDTH) for n in names}
    seen_names = [n for n in names if not n.startswith("bg_")]
    snaps = env.snaps[1:]                                  # snaps[0] is the reset snapshot; then one per tick
    steps = env.log["step"]
    t_det = ex.t_det
    fail = info.get("fail", "")
    contact_actor = impact(env, info)[0] or ""
    obs_names = env.spec.obs_names
    ends = np.array([s["t"] for s in steps])               # a step is logged at its end; obs = at its start

    ego_rows, gt_rows = [], []
    for i, r in enumerate(ticks):
        e = r["ego"]
        si = int(np.searchsorted(ends, r["t"] - 1e-6))
        s = steps[min(si, len(steps) - 1)] if steps else None
        is_contact = int(bool(contact_actor) and i == len(ticks) - 1)
        phase = "prefix" if r["t"] < -1e-6 else ("handback" if ex.t_hb is not None and r["t"] >= ex.t_hb - 1e-6 else
                                                 ("stage1" if t_det is not None and r["t"] >= t_det - 1e-6 else
                                                  "before_trigger"))
        row = dict(frame=r["frame"], sim_time=r["t"], t_since_cue="" if t_det is None or r["t"] < t_det - 1e-6
                   else round(r["t"] - t_det, 4), phase=phase, ego_s=e[0], ego_t=e[1] - env.scene["home_y"],
                   ego_yaw_rel=e[2], speed=e[3], accel_long=e[4], yaw_rate=e[5],
                   accel_long_f=r["meas"][0], accel_lat_f=r["meas"][1],
                   a_req=r["cmd"][0], r_req=r["cmd"][1], throttle=r["ctrl"][0], brake=r["ctrl"][1], steer=r["ctrl"][2],
                   contact=is_contact)
        obs = s["obs"] if s is not None else None
        for n, v in zip(obs_names, obs if obs is not None else [None] * len(obs_names)):
            row[f"obs_{n}"] = "" if v is None else float(v)
        snap = snaps[i][1] if i < len(snaps) else {}
        for n in seen_names:
            b = snap.get(n)
            row[f"seen_{n}_visible"] = int(b is not None)
            if b is None:
                for f in ACTOR_F:
                    row[f"seen_{n}_{f}"] = ""
                continue
            vx, vy = b["v"] * math.cos(b["psi"]), b["v"] * math.sin(b["psi"])
            rel = _rel(e, (b["x"], b["y"], b["psi"], b["v"], vx, vy))
            row.update({f"seen_{n}_{k}": v for k, v in rel.items()})
            row[f"seen_{n}_length"], row[f"seen_{n}_width"] = b["length"], b["width"]
        ego_rows.append(row)
        lo, hi = env.scene["bounds"](e[0])
        g = dict(frame=r["frame"], sim_time=r["t"], contact_actor=contact_actor if is_contact else "",
                 m_clear=r["m"][0], m_road=r["m"][1], dyn=r["m"][2], near=r["near"], bound_lo=lo - env.scene["home_y"],
                 bound_hi=hi - env.scene["home_y"], ego_x=e[0], ego_y=e[1], ego_psi=e[2], ego_v=e[3],
                 ego_world_x=r["ego_world"][0], ego_world_y=r["ego_world"][1], ego_world_yaw=r["ego_world"][3])
        ep = obb(e[0], e[1], e[2], EGO_L, EGO_W)
        for n in names:
            b = r["bodies"][n]
            rel = _rel(e, b)
            g.update({f"actor_{n}_{k}": v for k, v in rel.items()})
            L, Wd = sizes[n]
            g[f"actor_{n}_clear"] = signed_distance(ep, obb(b[0], b[1], b[2], L, Wd)) if abs(rel["s"]) < 30 else ""
            g[f"actor_{n}_length"], g[f"actor_{n}_width"] = L, Wd
        gt_rows.append(g)

    for name, rows in (("ego_observable.csv", ego_rows), ("ground_truth.csv", gt_rows)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out, "decisions.jsonl"), "w") as fh:
        for d in rec.decisions:
            fh.write(json.dumps(d, default=float) + "\n")

    m = info.get("min_margins", {})
    ok = bool(info.get("success"))
    end = ("horizon" if not fail or fail in ("goal", "envelope") else fail)
    if ok:
        reason = (f"no contact, inside the corridor and the envelope; min margin r = {info['r_nb']:.2f} (clearance "
                  f"{m.get('clear', 0):.2f}, road {m.get('road', 0):.2f}, friction {m.get('dyn', 0):.2f}), lane-centred at the end")
    elif contact_actor:
        reason = f"contact with {contact_actor}"
    elif fail in ("left_corridor", "envelope", "goal"):
        reason = {"left_corridor": "left the permitted corridor", "envelope": "outside the vehicle envelope",
                  "goal": "terminal contract not met"}[fail]
    else:
        worst = min(m, key=m.get) if m else "?"
        reason = f"no contact, but min margin r = {info.get('r_nb', float('nan')):.2f} < 0.05 (binding: {worst})"
    ticks0 = [r for r in ticks if r["t"] >= -1e-6]
    who, dv = impact(env, info)
    term = info.get("term")
    env_s = info.get("envelope") or {}
    # R1 (PIPELINE_HANDOFF): every constraint the label was judged on, with its measured and required value
    constraints = [
        dict(name="clearance", measured=m.get("clear"), required=R_OK, unit="m", passed=bool(m.get("clear", -1) >= R_OK),
             definition=f"min over ticks and actors of min(footprint distance, footprint distance predicted {T_H} s "
                        f"ahead at the closing velocity) - {C0} m reserve, capped at 2 m"),
        dict(name="road", measured=m.get("road"), required=R_OK, unit="m", passed=bool(m.get("road", -1) >= R_OK),
             definition="min distance of the ego footprint inside the permitted corridor (scenario bounds)"),
        dict(name="friction", measured=m.get("dyn"), required=R_OK, unit="fraction of the budget",
             passed=bool(m.get("dyn", -1) >= R_OK),
             definition="min of 1 - commanded combined acceleration / usable friction budget"),
        dict(name="terminal", measured=term, required=R_OK, unit="m", passed=bool(term is not None and term >= R_OK),
             definition="at the horizon: lane-centred and straight (the terminal contract); not reached when the run "
                        "ended early"),
        dict(name="no_contact", measured=dict(actor=who, impact_speed=dv), required="no contact", unit="m/s",
             passed=bool(who is None), definition="no footprint overlap with any actor, and no CARLA collision event; "
                                            "impact_speed = closing speed at the contact"),
        dict(name="goal", measured=info.get("goal"), required=getattr(env.spec, "goal_text", "the scene's goal"),
             passed=bool(info.get("goal_ok", fail != "goal")),
             definition="the scenario's own end condition (spec.goal), e.g. merged into the target lane"),
        dict(name="envelope", measured=env_s.get("viol_ticks"), required=0, unit="ticks",
             passed=bool(not env_s or env_s.get("viol_ticks", 0) == 0),
             definition="measured acceleration inside the vehicle's 1 g contract"),
    ]
    summary = dict(
        run_id=meta["run_id"], rc=meta["rc"], label="PASS" if ok else "FAIL", reason=reason, end_reason=end,
        scenario=dict(sid=env.spec.sid, town=env.spec.town, site=getattr(env.spec, "site_key", None),
                      frame_site=env.spec.site, lane_width=env.W, horizon_s=env.spec.T, prefix_s=env.spec.prefix_T,
                      description=env.spec.description, **{"params": meta["scenario_params"]}),
        policy=dict(name="options." + ex.mode, family=ex.mode, params=meta["policy_params"],
                    t_trigger=t_det, t_handback=ex.t_hb,
                    handback=getattr(ex.pol, "choice", None) and ex.pol.choice[0]),
        metrics=dict(r=info.get("r_nb"), min_margins=m, start_speed=ticks0[0]["ego"][3] if ticks0 else None,
                     end_speed=ticks[-1]["ego"][3] if ticks else None,
                     progress_m=(ticks[-1]["ego"][0] - ticks0[0]["ego"][0]) if ticks0 else None,
                     terminal=info.get("term"), envelope=info.get("envelope"), goal=info.get("goal")),
        contact=dict(frames=[ticks[-1]["frame"]] if contact_actor else [], actors=[contact_actor] if contact_actor else [],
                     impact_speed=dv if contact_actor else None),
        constraints=constraints, pass_rule=f"PASS = every constraint passed (margins >= {R_OK}); margin r = min of "
                                           f"clearance, road, friction and terminal",
        scene=dict(name=getattr(env.spec, "scene_name", env.spec.sid.split("@")[0]), uid=meta["variation_id"],
                   site=getattr(env.spec, "site_key", None), split=getattr(env.spec, "split", None),
                   history_id=meta.get("history_id", meta["variation_id"]), hidden_world=meta.get("hidden_world"),
                   hidden_params=meta.get("hidden", [])),
        schema="rcx-run-1", simulator=meta.get("simulator"), scene_seed=meta.get("scene_seed"),
        files=["summary.json", "ego_observable.csv", "ground_truth.csv", "decisions.jsonl"],
        rc_case=meta["rc"].lower(), variation_id=meta["variation_id"], candidate_kind=meta.get("kind"),
        family=ex.mode, scenario_params=meta["scenario_params"], hidden_params=meta.get("hidden", []),
        policy_params=meta["policy_params"], policy_source=meta.get("policy_source", {}),
        surrogate_prediction=meta.get("surrogate"),
        observation=dict(source="line of sight (sight.py): an actor is delivered while visible from the driver's eye "
                                "(held 0.3 s after it disappears); background traffic outside the corridor is never "
                                "delivered", delay_s=0.0))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return summary
