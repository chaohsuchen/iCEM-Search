"""(vsearch) Bridge to Kent's CARLA environment (carla_rc/teachers): the scenes there run in CARLA through
carla_rc's TeacherEnv, so the surrogate of this RC group is the one calibrated against that environment
(surrogate.SurrogateEnv: same scene spec, actors, observations and margins; fitted ego dynamics). Policies are the
event-driven families of options.py (ModeExpert: an observable trigger, then observable decision points such as the
hand-back).

sim(sid, theta_s, family, cfg, world) -> one rollout (rc_search.rollout's dict); world = an ego-dynamics perturbation
(rc_search.WORLDS entry: start / dyn / bias) or None for the nominal surrogate.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the repository root
TEACHERS = os.path.join(ROOT, "carla_rc", "teachers")
OUT = os.environ.get("RC_RESULTS", os.path.join(ROOT, "results_search"))
sys.path.insert(0, TEACHERS)

import surrogate  # noqa: E402,F401  (offline stand-ins for the CARLA modules first)
import rc_search  # noqa: E402
from rc_search import PASS_MARGIN, R_OK, WORLDS, env_sid, from_unit, rollout, to_unit  # noqa: E402,F401
from options import PARAM_SPACE, SETTINGS, full_cfg  # noqa: E402,F401

_ENV = {}


def env_for(sid):
    from surrogate import SurrogateEnv
    if sid not in _ENV:
        _ENV[sid] = SurrogateEnv(sid, seed=0)
    return _ENV[sid]


def specs():
    from scenarios import SPECS
    return SPECS


def sim(sid, theta_s, family, cfg, world=None):
    env = env_for(sid)
    w = world or {}
    env.start, env.dyn, env.bias = w.get("start", (0.0, 0.0)), w.get("dyn", (1.0, 1.0, 1.0, 1.0)), w.get("bias")
    try:
        return rollout(env, theta_s, family, full_cfg(family, cfg))
    finally:
        env.start, env.dyn, env.bias = (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), None
