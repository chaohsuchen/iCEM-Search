# RC18 from Victor's RC03 runs: requested control beyond the available authority

Source: Victor's 111 RC03 CARLA runs (`results_search_victor/rc03/carla/{baseline,presets_22mps,boundary_r3}`, Town04,
dry, straight). A run is kept when, before any contact, the policy's requested acceleration was limited
(`a_req` ≠ `a_req_clipped`) or the steering was saturated (|steer| ≥ 0.99).

**Result:** 11 runs, all `brake` (6 PASS, 5 FAIL). None of the lane changes was limited and no run saturated the steering.
- The requests reached −6.3 to −8.3 m/s². The limiter applied −6.1 m/s², and CARLA measured −8.7 to −9.1 m/s².
- Limiting lasted 2.4–5 s (48–99 ticks).

**Files**
- `index.json`: one row per kept run: the run folder, label, reason, family, kind, scene, policy parameters, limited
  ticks, first limited `t_since_cue`, minimum requested / applied (`a_req_clipped`) / measured (`accel_long_f`)
  acceleration, saturated-steer ticks, peak |lateral acceleration|.
- `<batch>__<run>.csv`: per tick (0.05 s): `frame, sim_time, t_since_cue, phase, speed, a_req, a_req_clipped,
  accel_long_f, accel_lat_f, accel_planar_f, throttle, brake, steer, contact`, plus the flags `limited` (the limiter
  intervened this tick) and `steer_saturated`.

**Limits.** Victor's RC03 is dry and straight. RC18's story (a lane change at bend speed in wet or heavy rain) is not in
these runs, so this only covers braking authority. The bend and rain case needs RC03 scenes on the bend sites.
