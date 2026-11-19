"""Fixed settings for RC01-01 (safe stop on a narrow one-way street)."""
from dataclasses import dataclass, asdict

G = 9.81


@dataclass
class ScenarioConfig:
    scene_id: str = "rc01_oneway_street_v1"
    seed: int = 7
    dt: float = 0.05                 # synchronous fixed step
    max_substep: float = 0.01
    # road (CARLA frame: x along travel, +y to the right of travel)
    road_length: float = 400.0
    van_clear_width: float = 1.4     # clear passage beside the van inside the lane (the swept 'road width' factor)
    van_width: float = 1.99
    lane_width: float = 0.0          # driving lane = allowed corridor, y in [0, lane_width]; derived = van_clear_width + van_width
    parking_width: float = 2.2
    sidewalk_width: float = 2.5
    # actors
    ego_bp: str = "vehicle.tesla.model3"
    van_bp: str = "vehicle.mercedes.sprinter"
    van_x: float = 200.0
    parked_bps: tuple = ("vehicle.audi.a2", "vehicle.toyota.prius", "vehicle.seat.leon",
                         "vehicle.mini.cooper_s", "vehicle.citroen.c3", "vehicle.nissan.micra")
    parked_x_range: tuple = (110.0, 250.0)
    parked_spacing: float = 6.2
    n_walkers_per_side: int = 3
    walker_x_range: tuple = (120.0, 245.0)
    dressing: bool = True
    start_from_rest: bool = True
    cruise_before_cue_s: float = 2.5   # ego holds target speed at least this long before the cue
    # test conditions
    target_speed: float = 8.0
    target_gap: float = 20.0
    speed_tol: float = 0.2
    gap_tol: float = 0.5
    approach_time: float = 4.0       # seconds at speed before the cue
    # evaluation
    time_limit: float = 8.0
    hold_time: float = 2.0
    hold_speed: float = 0.1
    min_clearance: float = 2.0
    accel_filter_s: float = 0.2      # trailing-mean window for envelope checks
    post_contact_s: float = 1.0      # keep recording after contact (video only)
    video_w: int = 1280
    video_h: int = 720
    card_s: float = 3.0

    def __post_init__(self):
        if not self.lane_width: self.lane_width = round(self.van_clear_width + self.van_width, 3)

    @property
    def corridor(self):
        return (0.0, self.lane_width)

    @property
    def lane_center(self):
        return self.lane_width / 2

    def to_dict(self):
        return asdict(self)


@dataclass
class PolicyConfig:
    version: str = "rc01-safe-stop-v1.1"   # v1.1: approach accel limit + anti-windup (brake logic unchanged)
    target_speed: float = 8.0
    brake_decel: float = 3.0         # requested deceleration magnitude, m/s^2
    stop_speed: float = 0.05         # brake -> hold below this speed
    kp: float = 1.0                  # speed hold gains (approach)
    ki: float = 0.6
    max_fwd_acc: float = 0.5 * G
    approach_accel: float = 1.5      # comfortable accel limit while speeding up / re-accelerating (below the 0.5 g cap)
    max_brake: float = 1.0 * G
    max_lat_acc: float = 1.0 * G
    max_planar_acc: float = 1.0 * G

    def to_dict(self):
        return asdict(self)
