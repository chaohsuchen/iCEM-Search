"""Scenario frame on a real CARLA map.

x = arc length along the ego lane's centreline from the scenario origin (the ego centre at t0), y = lateral offset to
the right, offset so that scenario lane k (left to right) is centred at y = k * W, W = the site's real lane width.
The sites are straight to within ~1 m, so placing a pose as centreline(x) + right_normal(x) * (y - y_ego) is exact
up to the residual curvature. curve=True (a curved motorway site) follows the lane through its bend instead of
continuing straight where it turns, making x, y curvilinear (lane-relative) coordinates.
"""
import math

import numpy as np

import carla


def same_dir_lanes(wp):
    """Driving lanes in wp's direction, left to right."""
    sign = np.sign(wp.lane_id)
    left, w = [], wp
    while True:
        l = w.get_left_lane()
        if l is None or np.sign(l.lane_id) != sign or l.lane_type != carla.LaneType.Driving:
            break
        left.append(l)
        w = l
    right, w = [], wp
    while True:
        r = w.get_right_lane()
        if r is None or np.sign(r.lane_id) != sign or r.lane_type != carla.LaneType.Driving:
            break
        right.append(r)
        w = r
    return left[::-1] + [wp] + right


def _straightest(cands, p0, n):
    return min(cands, key=lambda c: abs((np.array([c.transform.location.x, c.transform.location.y]) - p0) @ n))


class LaneFrame:
    def __init__(self, cmap, start_xy, ego_lane_id, ego_idx, s0=0.0, ahead=700.0, behind=120.0, step=1.0, curve=False):
        wp = cmap.get_waypoint(carla.Location(x=start_xy[0], y=start_xy[1], z=0.5), project_to_road=True,
                               lane_type=carla.LaneType.Driving)
        lanes = same_dir_lanes(wp)
        match = [l for l in lanes if l.lane_id == ego_lane_id]
        assert match, f"lane {ego_lane_id} not found at {start_xy}: {[l.lane_id for l in lanes]}"
        wp = match[0]
        if s0 > 0:
            wp = wp.next(float(s0))[0]
        self.W = float(wp.lane_width)
        self.y_ego = ego_idx * self.W
        yaw0 = math.radians(wp.transform.rotation.yaw)
        f0 = np.array([math.cos(yaw0), math.sin(yaw0)])
        n0 = np.array([-f0[1], f0[0]])
        p0 = np.array([wp.transform.location.x, wp.transform.location.y])
        back, w = [], wp
        for _ in range(int(behind / step)):
            prev = w.previous(step)
            if not prev:
                break
            w = prev[0] if curve and len(prev) == 1 else _straightest(prev, p0, n0)
            back.append(w)
        fwd, w = [], wp
        for _ in range(int(ahead / step)):
            nxt = w.next(step)
            if not nxt:
                break
            w = nxt[0] if curve and len(nxt) == 1 else _straightest(nxt, p0, n0)
            if not curve and abs(math.remainder(math.radians(w.transform.rotation.yaw) - yaw0, 2 * math.pi)) > 0.12:
                break                                  # the lane turns (e.g. a T-junction): continue straight instead
            fwd.append(w)
        pts = back[::-1] + [wp] + fwd
        P = [[p.transform.location.x, p.transform.location.y] for p in pts]
        Z = [p.transform.location.z for p in pts]
        n_ext = int(ahead / step) - len(fwd)
        if n_ext > 0:                                  # straight extension along the last heading
            h = math.radians(pts[-1].transform.rotation.yaw)
            P += [[P[-1][0] + k * step * math.cos(h), P[-1][1] + k * step * math.sin(h)] for k in range(1, n_ext + 1)]
            Z += [Z[-1]] * n_ext
        self.P, self.Z = np.array(P), np.array(Z)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        S = np.concatenate([[0.0], np.cumsum(seg)])
        self.S = S - S[len(back)]                      # x = 0 at the origin waypoint
        d = np.gradient(self.P, axis=0)
        self.H = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.origin = wp
        self.wps = pts
        self.straight_from = self.S[len(pts) - 1] if n_ext > 0 else None

    def _at(self, x):
        return (np.interp(x, self.S, self.P[:, 0]), np.interp(x, self.S, self.P[:, 1]),
                np.interp(x, self.S, self.Z), np.interp(x, self.S, self.H))

    def to_world(self, x, y, psi=0.0, z_off=0.0):
        px, py, pz, h = self._at(x)
        r = np.array([-math.sin(h), math.cos(h)])            # right normal (CARLA is left-handed: +y is right)
        q = np.array([px, py]) + r * (y - self.y_ego)
        return carla.Transform(carla.Location(x=float(q[0]), y=float(q[1]), z=float(pz + z_off)),
                               carla.Rotation(yaw=math.degrees(h + psi)))

    def from_world(self, loc, yaw_deg):
        """(x, y, psi) of a world pose."""
        q = np.array([loc.x, loc.y])
        i = int(np.argmin(np.sum((self.P - q) ** 2, axis=1)))
        h = self.H[i]
        f = np.array([math.cos(h), math.sin(h)])
        r = np.array([-f[1], f[0]])
        d = q - self.P[i]
        x = self.S[i] + float(d @ f)
        y = self.y_ego + float(d @ r)
        psi = math.atan2(math.sin(math.radians(yaw_deg) - h), math.cos(math.radians(yaw_deg) - h))
        return x, y, psi

    def vec_from_world(self, v, x):
        """Rotate a world (x, y) vector into the frame at arc length x: (along, right)."""
        h = float(np.interp(x, self.S, self.H))
        return v[0] * math.cos(h) + v[1] * math.sin(h), -v[0] * math.sin(h) + v[1] * math.cos(h)
