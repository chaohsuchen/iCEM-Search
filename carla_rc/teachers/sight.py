"""What the ego can see: a 2-D line-of-sight model in world coordinates, for specs whose observability matters.

An actor is delivered (Spec.deliver -> the observation, the experts' view and the trigger) when, from the driver's eye,
a sight line to its centre or one of its corners passes no occluder and it is within the sensing range (RANGE, or the spec's sight_range, e.g. headlights at night). Occluders:
every other vehicle body (its footprint; heights are ignored, so a car hides what is behind it as a van would - on
the safe side) and the scene's static occluders (env.scene["occluders"]: polygons in scenario-frame coordinates,
e.g. the buildings at a junction corner or a bend's inside wall). Walkers and props never occlude.

Hills: the sight line also has to clear the road's real elevation profile between the eye and the target
(_over_crest; a queue beyond a crest stays hidden until the crest no longer covers it). The test runs in world
coordinates (the site frame's centreline), so on a bend an actor straight "behind" the lead in
the lane can be in view past its side - the effect RC03-S2's curved sites rely on. A track survives a short occlusion
(TRACK_HOLD) as a perception stack's tracker would. Sensing range shrinks with rain (RANGE).
"""
import math

import numpy as np

RANGE = {"dry": 150.0, "wet": 120.0, "heavy": 90.0}          # [m]
H_EYE, H_TARGET = 1.2, 1.0       # [m] driver's eye; the part of a car / person that has to show over a crest
TRACK_HOLD = 0.3                                              # [s] a seen actor stays delivered through a short gap
EYE = (0.4, -0.4)                                             # driver's eye from the ego centre (forward, right) [m]


def _world(env, x, y):
    """Scenario-frame (x, y) -> world (x, y) through the site's centreline (identity without a frame)."""
    f = getattr(env, "frame", None)
    if f is None:
        return np.asarray(x, float), np.asarray(y, float)
    px, py = np.interp(x, f.S, f.P[:, 0]), np.interp(x, f.S, f.P[:, 1])
    h = np.interp(x, f.S, f.H)
    d = np.asarray(y, float) - f.y_ego
    return px - np.sin(h) * d, py + np.cos(h) * d


def _heading(env, x):
    f = getattr(env, "frame", None)
    return 0.0 if f is None else float(np.interp(x, f.S, f.H))


def _boxes(env, names):
    """World oriented boxes (cx, cy, cos, sin, half_l, half_w) of the given bodies."""
    out = []
    for n in names:
        b = env.bodies[n]
        cx, cy = _world(env, b.position[0], b.position[1])
        h = _heading(env, b.position[0]) + b.heading
        out.append((float(cx), float(cy), math.cos(h), math.sin(h), b.LENGTH / 2, b.WIDTH / 2))
    return np.array(out, float).reshape(-1, 6)


def _corners(box):
    cx, cy, c, s, hl, hw = box
    return [(cx, cy)] + [(cx + c * a - s * b, cy + s * a + c * b) for a in (-hl, hl) for b in (-hw, hw)]


def _blocked_by_boxes(p, q, B):
    """Does segment p -> q cross any box of B? (slab test in each box's frame; vectorised over boxes)"""
    if not len(B):
        return False
    d = np.array(q) - np.array(p)
    rp = np.array(p)[None] - B[:, :2]
    c, s = B[:, 2], B[:, 3]
    o = np.stack([rp[:, 0] * c + rp[:, 1] * s, -rp[:, 0] * s + rp[:, 1] * c], 1)          # start in box frame
    v = np.stack([d[0] * c + d[1] * s, -d[0] * s + d[1] * c], 1)                         # direction in box frame
    half = B[:, 4:6]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / v, (half - o) / v
    lo = np.where(np.abs(v) < 1e-9, np.where(np.abs(o) <= half, -np.inf, np.inf), np.minimum(t1, t2)).max(1)
    hi = np.where(np.abs(v) < 1e-9, np.where(np.abs(o) <= half, np.inf, -np.inf), np.maximum(t1, t2)).min(1)
    return bool(np.any((lo <= hi) & (hi > 0.0) & (lo < 1.0)))


def _over_crest(env, xe, xt):
    """Vertical sight line over the site's real road profile (the frame's centreline elevation): does the line from
    the eye at xe to the target at xt stay above the road between them?"""
    f = getattr(env, "frame", None)
    if f is None or abs(xt - xe) < 20.0:
        return True
    xs = np.linspace(xe, xt, 24)[1:-1]
    z = np.interp(xs, f.S, f.Z)
    z0, z1 = float(np.interp(xe, f.S, f.Z)) + H_EYE, float(np.interp(xt, f.S, f.Z)) + H_TARGET
    line = z0 + (z1 - z0) * (xs - xe) / (xt - xe)
    return bool(np.all(z < line))


def _cross(a, b, c, d):
    o = lambda p, q, r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def _blocked_by_polys(p, q, polys):
    for P in polys:
        if any(_cross(p, q, P[i], P[(i + 1) % len(P)]) for i in range(len(P))):
            return True
    return False


def _edges(polys):
    """The occluder polygons' edges as one array [n, 2 (start, end), 2 (x, y)]."""
    E = [(P[i], P[(i + 1) % len(P)]) for P in polys for i in range(len(P))]
    return np.array(E, float).reshape(-1, 2, 2)


def _blocked_by_edges(p, q, E):
    """_blocked_by_polys on the edge array (same strict crossing test, vectorised: long screens have many edges)."""
    if not len(E):
        return False
    o = lambda P, Q, R: (Q[..., 0] - P[..., 0]) * (R[..., 1] - P[..., 1]) - (Q[..., 1] - P[..., 1]) * (R[..., 0] - P[..., 0])
    p, q, a, b = np.asarray(p, float), np.asarray(q, float), E[:, 0], E[:, 1]
    return bool(np.any((o(p, q, a) * o(p, q, b) < 0) & (o(a, b, p) * o(a, b, q) < 0)))


def visible_set(env):
    """Names of the bodies the ego sees now (cached per tick)."""
    m = env.mem.setdefault("sight", {})
    if m.get("t") == env.t:
        return m["vis"]
    if "polys" not in m:                                       # static occluders, once per episode, in world coordinates
        m["polys"] = [[tuple(float(v) for v in _world(env, x, y)) for x, y in P] for P in env.scene.get("occluders", [])]
        m["edges"] = _edges(m["polys"])
        m["last"] = {}
    e = env.ego
    ex, ey = _world(env, e.position[0] + EYE[0], e.position[1] + EYE[1])
    eye = (float(ex), float(ey))
    rng = getattr(env.spec, "sight_range", RANGE)[env.params.get("wx", "dry")]
    names = list(env.bodies)
    vehicles = [n for n in names if (env.bodies[n].bp or "vehicle.").startswith("vehicle.")]
    B = _boxes(env, vehicles)
    vis = set()
    for i, n in enumerate(names):
        b = env.bodies[n]
        box = _boxes(env, [n])[0]
        if math.hypot(box[0] - eye[0], box[1] - eye[1]) > rng or not _over_crest(env, e.position[0], b.position[0]):
            continue
        others = B[[j for j, k in enumerate(vehicles) if k != n]] if len(B) else B
        for pt in _corners(box):
            if not _blocked_by_boxes(eye, pt, others) and not _blocked_by_edges(eye, pt, m["edges"]):
                vis.add(n)
                break
    for n in vis:
        m["last"][n] = env.t
    held = {n for n, t in m["last"].items() if env.t - t <= TRACK_HOLD + 1e-6}
    m["t"], m["vis"] = env.t, vis | held
    return m["vis"]


class SightMixin:
    """Spec mixin: deliver = line of sight (plus `always`: actors known without sight, e.g. by the radar of a
    following car)."""
    always = ()

    def deliver(self, env, name):
        return name in self.always or name in visible_set(env)
