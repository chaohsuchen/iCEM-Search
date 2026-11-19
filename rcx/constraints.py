"""Per-run constraint block: every check the CARLA judge applied, with measured value, required value and pass/fail.

`annotate(summary, rows)` returns the fields added to summary.json:
  scene_name       name of the scene family (separate from the RC number)
  hidden_settings  scene parameters that are not observable in the recorded history at the decision point (declared per case)
  constraints      list of {name, kind, measured, required, op, unit, passed}
  constraints_pass True when every listed constraint passed (None for INVALID / UNRESOLVED runs: not evaluated)

The block mirrors the rules in rcx/scenarios.py (RC03-RC13) and rc01/evaluate.py; tools/backfill_constraints.py checks on every
existing run that `constraints_pass` agrees with the CARLA label. The block is derived from the same recorded numbers as the label,
it is not an independent measurement. `rows` = list of per-tick dicts (ego_observable + ground_truth columns merged by frame).
"""
import numpy as np
from rc01.config import PolicyConfig

SCENE_NAME = dict(rc01="street_van", rc03="spun_car", rc04="stopped_car_truck_alongside", rc05="lane_change_abort",
                  rc06="broken_car_passer", rc09="tunnel_squeeze", rc13="sidewalk_pedestrian_dart")

# Declared per case from 20RCs.md ("Hidden at the trigger"), written with the implementation's parameter names.
HIDDEN = dict(rc01=[],
              rc03=["crash", "intrude", "psi_rest", "T_slide", "lead_brake", "lead_react"],
              rc04=["side_acc", "side_react", "fol_headway", "fol_react"],
              rc05=["rear_acc", "rear_vmax"],
              rc06=["side_acc", "side_react", "fol_headway", "fol_react"],
              rc09=["rear_delay_s", "rear_decel"],
              rc13=["dart_speed", "dart_dist"])


def _c(name, kind, measured, required, op, unit, passed=None):
    if passed is None:
        passed = {">=": measured >= required, "<=": measured <= required, "==": measured == required}[op]
    return dict(name=name, kind=kind, measured=(None if measured is None else float(measured) if not isinstance(measured, bool) else bool(measured)),
                required=(float(required) if not isinstance(required, bool) else bool(required)), op=op, unit=unit, passed=bool(passed))


def _envelope(m):
    mx, pc = m["max_filtered_accel"], PolicyConfig()
    return [_c("envelope_forward_accel", "limit", mx["forward"], pc.max_fwd_acc, "<=", "m/s2"),
            _c("envelope_braking", "limit", mx["brake"], pc.max_brake, "<=", "m/s2"),
            _c("envelope_lateral_accel", "limit", mx["lateral"], pc.max_lat_acc, "<=", "m/s2"),
            _c("envelope_planar_accel", "limit", mx["planar"], pc.max_planar_acc, "<=", "m/s2")]


def _cue_rows(rows):
    i = next((i for i, r in enumerate(rows) if r["t_since_cue"] != ""), None)
    return None if i is None else rows[i:]


def _lane_case(case, s, rows):
    p, m = s["scenario"], s["metrics"]; ev = _cue_rows(rows)
    L = __import__("rcx.rc03_laws" if case == "rc03" else "rcx.rc04_laws", fromlist=["x"])
    out = [_c("no_contact", "contact", len(s["contact"]["frames"]), 0, "==", "frames")]
    for k, x in m["min_clear_by"].items(): out.append(_c(f"clearance_{k}", "clearance", x, p["min_clear"], ">=", "m"))
    out.append(_c("ego_settled", "state", m["settled"], True, "==", "bool"))
    out.append(_c("stays_on_road", "clearance", float(np.abs([r["ego_t"] for r in ev]).max()), 1.5 * p["lane_w"], "<=", "m"))
    l = ev[-1]; tgt = "victim" if case == "rc03" else "obstacle"
    cl = L.still_closing(l[tgt + "_rear_s"] - l["ego_front_s"], l["speed"], abs(l["ego_t_c"] - l[tgt + "_t"]) - 2.08, 2.0)
    out.append(_c(f"not_closing_on_{tgt}", "state", cl, False, "==", "bool"))
    for k in (("esc_follow", "rear") if case == "rc03" else ("fol",)):
        if k + "_front_s" in l:
            cl = L.rear_closing(l["ego_rear_s"] - l[k + "_front_s"], l[k + "_v"] - l["speed"], abs(l["ego_t_c"] - l[k + "_t"]) - 1.08 - 0.95, 2.0)
            out.append(_c(f"not_closing_from_{k}", "state", cl, False, "==", "bool"))
    return out + _envelope(m)


def _rc05(s, rows):
    p, m = s["scenario"], s["metrics"]; ev = _cue_rows(rows); et = np.array([r["ego_t"] for r in ev], float)
    out = [_c("no_contact", "contact", len(s["contact"]["frames"]), 0, "==", "frames"),
           _c("road_edge_left", "clearance", min(r["ego_left"] for r in ev), -(p["lane_w"] + 1.75), ">=", "m"),
           _c("road_edge_right", "clearance", max(r["ego_right"] for r in ev), 1.75, "<=", "m"),
           _c("clearance_rear_left", "clearance", m["min_clearance_rear_left"], p["min_rear_clear"], ">=", "m"),
           _c("clearance_lead", "clearance", m["min_clearance_lead"], p["min_lead_clear"], ">=", "m"),
           _c("ego_settled_in_lane", "state", bool(np.all(np.abs(et[-20:]) < 0.5) or np.all(np.abs(et[-20:] + p["lane_w"]) < 0.5)), True, "==", "bool")]
    if m.get("stable_after_return") is not None: out.append(_c("stable_after_return", "state", m["stable_after_return"], True, "==", "bool"))
    return out + _envelope(m)


def _rc09(s, rows):
    p, m = s["scenario"], s["metrics"]
    return [_c("no_contact", "contact", len(s["contact"]["frames"]), 0, "==", "frames"),
            _c("clearance_front", "clearance", m["min_front_gap"], p["min_clear"], ">=", "m"),
            _c("clearance_rear", "clearance", m["min_rear_gap"], p["min_clear"], ">=", "m"),
            _c("ego_stopped", "state", m["ego_stopped"], True, "==", "bool")] + _envelope(m)


def _rc13(s, rows):
    m = s["metrics"]
    out = [_c("no_contact", "contact", len(s["contact"]["frames"]), 0, "==", "frames")]
    if m["dart"] and m["pedestrian_entered_lane"]: out.append(_c("clearance_pedestrian", "clearance", m["min_ped_dist"], 0.5, ">=", "m"))
    out.append(_c("no_unnecessary_slowing", "state", m["unnecessary_slowing"], False, "==", "bool"))
    return out + _envelope(m)


def _rc01(s, rows):
    p, m = s["scenario"], s["metrics"]; ev = _cue_rows(rows)
    gap_cue = next((i for i, r in enumerate(rows) if r.get("obs_generated", r.get("obs_delivered"))), 0)
    corr = int(sum(1 for r in rows[gap_cue:] if float(r.get("corridor_violation", 0) or 0) > 0))
    held = m.get("stopping_gap") is not None
    out = [_c("no_contact", "contact", len(s["contact"]["frames"]), 0, "==", "frames"),
           _c("in_corridor", "state", corr, 0, "==", "frames"),
           _c("stop_and_hold", "state", held, True, "==", "bool")]
    if held: out.append(_c("clearance_during_hold", "clearance", m["hold_min_gap"], p["min_clearance"], ">=", "m"))
    return out + _envelope(m)


def compute(case, s, rows):
    if s["label"] in ("INVALID", "UNRESOLVED"): return None
    if case in ("rc03", "rc04", "rc06"): return _lane_case(case, s, rows)
    return dict(rc05=_rc05, rc09=_rc09, rc13=_rc13, rc01=_rc01)[case](s, rows)


def annotate(case, summary, rows):
    cons = compute(case, summary, rows)
    return dict(scene_name=SCENE_NAME[case], hidden_settings=HIDDEN[case], constraints=cons,
                constraints_pass=None if cons is None else all(c["passed"] for c in cons))
