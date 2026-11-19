"""Shared lateral/geometry core for lane-type cases (RC05 today; lane_sim and RC03/04/06/08/16 next): Tesla Model 3 steering limit, fitted lateral plant
constants, and convex-footprint geometry in lane coordinates (s forward, t lateral)."""
import math

MAX_STEER = math.radians(70.0)                 # wheel limit used by the CARLA controller (max_steer_rad)
LAT_K, LAT_TAU = math.radians(19.4), 0.05      # yaw rate [rad/s] = LAT_K * v * steer_filtered   (fit on 7 CARLA runs, rmse 0.7 deg/s)
EGO_HL, EGO_HW = 2.35, 1.08                    # ego half length / half width


def _rect(s, t, yaw, hl, hw):
    c, sn = math.cos(yaw), math.sin(yaw)
    return [(s + x * c - y * sn, t + x * sn + y * c) for x, y in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw))]


def _pt_seg(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay; u = ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby + 1e-12)
    u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
    return math.hypot(px - (ax + u * abx), py - (ay + u * aby))


def quad_box_dist(Q, box):
    """Distance between a convex quad Q (4 (s,t) points) and an axis-aligned box (smin, smax, tmin, tmax); 0 if they overlap. AABB culling above 4 m
    (returns the AABB gap, a lower bound) keeps the surrogate fast. Exact (SAT + vertex/edge distances) otherwise."""
    sm, sx, tm, tx = box
    qs = [p[0] for p in Q]; qt = [p[1] for p in Q]
    gs = max(sm - max(qs), min(qs) - sx, 0.0); gt = max(tm - max(qt), min(qt) - tx, 0.0)
    g = math.hypot(gs, gt)
    if g > 4.0: return g
    B = [(sm, tm), (sx, tm), (sx, tx), (sm, tx)]
    sep = g > 0.0
    if not sep:
        for i in (0, 1):
            ex, ey = Q[i + 1][0] - Q[i][0], Q[i + 1][1] - Q[i][1]; nx, ny = -ey, ex
            pq = [nx * p[0] + ny * p[1] for p in Q]; pb = [nx * p[0] + ny * p[1] for p in B]
            if max(pq) < min(pb) or max(pb) < min(pq): sep = True; break
    if not sep: return 0.0
    best = 1e9
    for p in Q:
        best = min(best, math.hypot(max(sm - p[0], 0.0, p[0] - sx), max(tm - p[1], 0.0, p[1] - tx)))
    for p in B:
        for i in range(4): best = min(best, _pt_seg(p[0], p[1], Q[i][0], Q[i][1], Q[(i + 1) % 4][0], Q[(i + 1) % 4][1]))
    return best


def quad_quad_dist(P, Q):
    """Distance between two convex quads (4 (s,t) points each); 0 if they overlap. AABB culling above 4 m returns the AABB gap (a lower bound)."""
    ps = [p[0] for p in P]; pt = [p[1] for p in P]; qs = [p[0] for p in Q]; qt = [p[1] for p in Q]
    g = math.hypot(max(min(qs) - max(ps), min(ps) - max(qs), 0.0), max(min(qt) - max(pt), min(pt) - max(qt), 0.0))
    if g > 4.0: return g
    sep = False
    for A in (P, Q):
        for i in (0, 1):
            ex, ey = A[i + 1][0] - A[i][0], A[i + 1][1] - A[i][1]; nx, ny = -ey, ex
            a = [nx * p[0] + ny * p[1] for p in P]; b = [nx * p[0] + ny * p[1] for p in Q]
            if max(a) < min(b) or max(b) < min(a): sep = True; break
        if sep: break
    if not sep: return 0.0
    best = 1e9
    for X, Y in ((P, Q), (Q, P)):
        for p in X:
            for i in range(4): best = min(best, _pt_seg(p[0], p[1], Y[i][0], Y[i][1], Y[(i + 1) % 4][0], Y[(i + 1) % 4][1]))
    return best
