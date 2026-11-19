"""Longitudinal ego plant fitted from the brake calibration (rc01/calibrate.py trials). Pure numpy, no CARLA.
decel(brake, v) = c0(brake) + c1(brake) * v  (least squares on the calibration v-series above the low-speed region);
non-locking brake commands only (the controller caps brake at brake_cap). Locking brakes are not modelled (never commanded).
Re-fit from a new calibration.json whenever the actuator changes (e.g. after the PWM / sub-step experiments)."""
import json
import numpy as np


class LongPlant:
    def __init__(self, calib, dt=0.05, v_lo=0.3, extra_delay_frames=0):
        self.dt, self.calib = dt, calib
        pts = []
        for t in calib["trials"]:
            if t["locks"]: continue
            v = np.array(t["v_series"], float)
            a = -np.diff(v) / dt
            k = 4                                                   # decel averaged over 4 ticks (the series alternates tick to tick)
            va = np.array([v[i:i + k + 1].mean() for i in range(len(v) - k)]); aa = (v[:-k] - v[k:]) / (k * dt)
            m = va > v_lo + 0.5
            c1, c0 = np.polyfit(va[m], aa[m], 1)
            snap = t["snap_from_speed"]
            pts.append((t["brake"], c0, c1, snap))
        pts.sort()
        self.b = np.array([p[0] for p in pts]); self.c0 = np.array([p[1] for p in pts]); self.c1 = np.array([p[2] for p in pts])
        snaps = [p[3] for p in pts if p[3] is not None]
        self.v_snap = float(np.median(snaps)) if snaps else 1.5      # any brake below this speed: one-tick stop (measured)
        self.delay_frames = max(1, int(round(calib.get("brake_delay_s", 0.05) / dt))) + extra_delay_frames
        self.cruise, self.v_cruise, self.k_thr = calib.get("cruise_throttle", 0.3), calib.get("cruise_speed", 8.0), 0.12
        self.thr_fit = calib.get("throttle_fit")      # high-speed throttle->accel map fitted on CARLA rc03 telemetry (the 8 m/s cruise model is far too weak above ~12 m/s)
        self.reset()

    def reset(self):
        self._q = []

    def _thr_acc(self, thr, v):
        """Measured throttle -> acceleration map (tables per speed, from CARLA telemetry); linear in throttle between table points, extended linearly beyond them, linear in v between speeds."""
        f = self.thr_fit; vs = f["v"]; out = []
        for tt, aa in zip(f["thr"], f["a"]):
            tt, aa = np.array(tt), np.array(aa)
            if thr <= tt[0]: sl = (aa[1] - aa[0]) / max(tt[1] - tt[0], 1e-6); out.append(aa[0] + sl * (thr - tt[0]))
            elif thr >= tt[-1]: sl = (aa[-1] - aa[-2]) / max(tt[-1] - tt[-2], 1e-6); out.append(aa[-1] + sl * (thr - tt[-1]))
            else: out.append(float(np.interp(thr, tt, aa)))
        return float(np.interp(v, vs, out))

    def decel(self, brake, v):
        return float(np.interp(brake, self.b, self.c0) + np.interp(brake, self.b, self.c1) * v)

    def step(self, v, throttle, brake, hold):
        """-> (v_next, a_long). Brake command goes through a pure delay of delay_frames ticks."""
        self._q.append(brake); b = self._q.pop(0) if len(self._q) > self.delay_frames else 0.0
        if hold: return 0.0, -v / self.dt
        if self.thr_fit and throttle > 0.0 and v >= self.thr_fit["v_min"]:
            a = self._thr_acc(throttle, v)
        elif throttle > 0.0 or b <= 0.0 and brake <= 0.0:
            ref = self.cruise * (0.1 + 0.9 * min(1.0, max(v, 0.0) / self.v_cruise))      # steady throttle vs speed (low-speed creep fit on CARLA logs: 0.077 at 1.2 m/s)
            if throttle >= ref: a = (throttle - ref) / self.k_thr
            else: a = -self.decel(0.0, v) * (1.0 - throttle / max(ref, 1e-6))          # partial coast: from coast decel (throttle 0) to 0 at the holding throttle
        else:
            a = -self.decel(b, v)
        if (b > 0.0 or hold) and v < self.v_snap and a < 0: return 0.0, -v / self.dt
        v2 = max(0.0, v + a * self.dt)
        return v2, (v2 - v) / self.dt

    @staticmethod
    def from_file(path, dt=0.05): return LongPlant(json.load(open(path)), dt)
