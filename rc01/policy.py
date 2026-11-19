"""Rule-based ego policy: approach -> brake -> hold. Pure logic, no CARLA imports."""
from dataclasses import dataclass
from .config import PolicyConfig


@dataclass
class Observation:
    frame: int
    sim_time: float
    ego_speed: float
    blocked_road: bool           # delivered blocked-road cue
    obstacle_gap: float          # bumper-to-bumper gap (idealized, from CARLA state)
    source: str = "carla_ground_truth"


@dataclass
class PolicyOutput:
    phase: str                   # approach | brake | hold
    a_req: float                 # requested longitudinal accel (m/s^2), negative = brake
    lane_keep: bool = True       # stay in original corridor (no lateral maneuver)
    t_des: float = None          # desired lateral position in the scenario frame (None = lane centre)
    t_rate: float = 0.0          # desired lateral velocity (m/s), heading feed-forward
    info: dict = None            # extra values to log


class SafeStopPolicy:
    def __init__(self, cfg: PolicyConfig, dt: float):
        self.cfg, self.dt = cfg, dt
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._i = 0.0
        self.cue_frame = None

    def step(self, obs: Observation) -> PolicyOutput:
        c = self.cfg
        if self.phase == "approach" and obs.blocked_road:
            self.phase, self.cue_frame = "brake", obs.frame
        if self.phase == "brake" and obs.ego_speed <= c.stop_speed:
            self.phase = "hold"
        if self.phase == "approach":
            e = c.target_speed - obs.ego_speed
            a = c.kp * e + c.ki * self._i
            a_cl = max(-1.0, min(a, c.approach_accel))      # comfortable acceleration limit
            if a_cl == a:                                   # anti-windup: integrate only when not saturated
                self._i = max(-3.0, min(3.0, self._i + e * self.dt))
            return PolicyOutput("approach", a_cl)
        if self.phase == "brake":
            return PolicyOutput("brake", -c.brake_decel)
        return PolicyOutput("hold", 0.0)
