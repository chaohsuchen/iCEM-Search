"""Low-level controller: requested accel -> throttle / brake / steer, using measured brake calibration."""
import math, os
import numpy as np
from .config import G, PolicyConfig


# Low-speed brake taper (off by default so finished results stay reproducible): below LOW_V_TAPER m/s the brake command is scaled by v / LOW_V_TAPER
# (floor LOW_V_FLOOR) to avoid CARLA's one-tick stop snap (~1 m/s -> 0 in one tick, 15-24 m/s^2). Set RC_LOW_V_TAPER=2.0 to enable.
LOW_V_TAPER = float(os.environ.get("RC_LOW_V_TAPER", "0")); LOW_V_FLOOR = float(os.environ.get("RC_LOW_V_FLOOR", "0.25"))


class LowLevelController:
    def __init__(self, pcfg: PolicyConfig, calib: dict, lane_center: float, max_steer_rad: float, dt: float):
        self.p, self.dt, self.yc, self.max_steer = pcfg, dt, lane_center, max_steer_rad
        self.cruise = calib.get("cruise_throttle", 0.25)
        self.v_cruise = calib.get("cruise_speed", 8.0)
        self.k_thr = 0.12                                    # throttle per m/s^2
        pts = sorted(zip(calib["brake_decel"], calib["brake_cmd"]))
        self.dec_tab = np.array([0.0] + [p[0] for p in pts])
        self.brk_tab = np.array([0.0] + [p[1] for p in pts])
        # measured vehicle limit: decel at full brake, never exceed 1g or 95% of measured
        self.max_decel = min(self.p.max_brake, 0.95 * calib["max_decel"])
        self.brake_cap = calib["brake_cap"]                  # largest non-locking brake command (measured)
        self.delay_frames = int(math.ceil(calib.get("brake_delay_s", 0.0) / dt)) + 1
        self.coast = calib.get("coast_decel", 0.0)           # decel with zero throttle/brake (drag)
        self.ki_b = 0.01
        self.reset()

    def reset(self):
        self._ib, self._brk_frames, self._hist = 0.0, 0, []
        self._it, self._thr_prev = 0.0, 0.0

    def _steer(self, y, yaw_deg, v, t_des=None, t_rate=0.0):
        yaw = math.radians(((yaw_deg + 180) % 360) - 180)
        cte = y - (self.yc if t_des is None else t_des)      # +y is right in CARLA
        yaw_des = math.atan2(t_rate, max(v, 1.0))
        ang = -(yaw - yaw_des) - math.atan2(1.0 * cte, v + 2.0)
        return float(np.clip(ang / self.max_steer, -1, 1))

    def step(self, out, y, yaw_deg, v, a_long):
        steer = self._steer(y, yaw_deg, v, out.t_des, out.t_rate) if out.lane_keep else 0.0
        self._hist = (self._hist + [a_long])[-2:]
        a_f = float(np.mean(self._hist))
        a_req = float(np.clip(out.a_req, -self.max_decel, self.p.max_fwd_acc))
        thr = brk = 0.0
        hand = False
        if out.phase == "hold":
            brk, hand = 1.0, True
        elif a_req >= 0 or out.phase == "approach" or (self.coast > 0 and -a_req < self.coast):
            self._ib, self._brk_frames = 0.0, 0
            if a_req >= 0 or self.coast <= 0:
                ref = self.cruise * (0.3 + 0.7 * min(1.0, max(v, 0.0) / self.v_cruise))   # drag grows with speed
                if a_req >= 0 and out.phase in ("approach", "follow"):
                    self._it = float(np.clip(self._it + 0.4 * (a_req - a_f) * self.dt, -0.25, 0.25))   # accel tracking
                thr = float(np.clip(ref + self.k_thr * a_req + self._it, 0, 1))
                if a_f > 0.9 * self.p.max_fwd_acc:                                       # never exceed the forward cap
                    thr = min(thr, 0.8 * self._thr_prev)
            else:                                            # partial coast: throttle between cruise and 0
                thr = float(np.clip(self.cruise * (1.0 + a_req / self.coast), 0, 1))
            if out.phase == "approach" and a_req < -0.3 and self.coast <= 0:
                brk = float(np.interp(-a_req, self.dec_tab, self.brk_tab)); thr = 0.0
        else:
            d = -a_req
            self._brk_frames += 1
            if self._brk_frames > self.delay_frames:         # wait out brake build-up before feedback
                self._ib += self.ki_b * (d + a_f)
                self._ib = float(np.clip(self._ib, -0.15, 0.3))
            brk = float(np.clip(np.interp(d, self.dec_tab, self.brk_tab) + self._ib, 0, self.brake_cap))
            if LOW_V_TAPER > 0 and v < LOW_V_TAPER: brk *= max(LOW_V_FLOOR, v / LOW_V_TAPER)
        self._thr_prev = thr
        return dict(throttle=thr, brake=brk, steer=steer, hand_brake=hand, a_req_clipped=a_req)
