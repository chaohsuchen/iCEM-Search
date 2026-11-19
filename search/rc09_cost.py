"""RC09 cost + outcome class from a surrogate rollout. Lower is better. The surrogate is ~0.4 m optimistic on front clearance (validated vs CARLA),
so a conservative margin is added in the cost; the verdict used for labels always comes from CARLA."""
import numpy as np

FRONT_MARGIN, REAR_MARGIN = 0.4, 0.0


def outcome_class(r, min_clear=1.0):
    if r["verdict"] == "PASS": return "success"
    if (not r["contact"] and r["stopped"]) or (r["contact"] and r["rel_speed"] < 2.0): return "mitigate"   # clearance shortfall, or low-speed contact
    return "fail"


def cost(r, min_clear=1.0):
    rows = np.array(r["rows"]); a = rows[:, 6]; peak = float(max(0.0, -a.min())) if len(a) else 0.0
    if r["contact"]: return 1000.0 + 50.0 * r["rel_speed"] + 0.1 * peak
    c = 0.1 * peak + 0.3 * max(0.0, 3.0 - r["min_front_gap"]) + 0.3 * max(0.0, 3.0 - r["min_rear_gap"])      # soft preference for >= 3 m clearance
    c += 50.0 * max(0.0, min_clear + FRONT_MARGIN - r["min_front_gap"]) + 50.0 * max(0.0, min_clear + REAR_MARGIN - r["min_rear_gap"])
    if not r["stopped"]: c += 200.0 + 20.0 * float(rows[-1][1])
    return c
