"""Toy check (no CARLA): the program space contains trivial programs AND a conditional one, and a plain iCEM on a toy objective finds the conditional one
without it being hard-coded. Run: python search/experimental/test_program.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from search.experimental.program import program_family, GuardedProgram
from search.families import decode_spec
from search.cem import icem

FEATS = {"rear_ttc": ("rear_ttc", 0.0, 10.0), "front_gap": ("front_gap", 0.0, 60.0)}
FAM = program_family(FEATS, K=3, lat_range=(-1, 1), a_range=(-7.2, 2.0))
dt = 0.05


def toy(kw, threat):
    """Ego changes lane left (lat -1). A rear car in the target lane appears at t=2 s (threat=True): TTC drops from 9 to 2 over 2 s. Reward: finish in left lane if no threat,
    return to lane 0 within 1.5 s of the TTC<4 event if threat. Cost = lane error at the end + penalty for being out of lane 0 while TTC<3."""
    p = GuardedProgram(dt, **kw); p.reset(); cost = 0.0; ttc = None
    for n in range(int(8 / dt)):
        t = n * dt; ttc = 9.0 if not threat or t < 2.0 else max(0.5, 9.0 - 3.5 * (t - 2.0))
        out = p.step(dict(v=8.0, cue=True, t=t, rear_ttc=ttc, front_gap=40.0)); y = out.t_des / 3.5
        if ttc < 3.0 and abs(y) > 0.3: cost += 1.0
    cost += 10 * abs(y - (0.0 if threat else -1.0))
    return cost


def triv_kw(): return decode_spec(FAM, np.full(len(FAM["names"]), 0.5))


def f(u):
    kw = decode_spec(FAM, u); return max(toy(kw, True), toy(kw, False))      # robust over threat / no threat: needs a CONDITIONAL program


u0 = np.full(len(FAM["names"]), 0.5); print("trivial-ish program cost:", round(f(u0), 1))
best = min((icem(f, len(FAM["names"]), np.random.default_rng(s), pop=96, iters=14) for s in range(4)), key=lambda r: r[1])
print("iCEM best robust cost:", round(best[1], 2), "(0 = solved both worlds; an open-loop program cannot get below ~10)")
kw = decode_spec(FAM, best[0]); print({k: round(v, 2) for k, v in kw.items() if isinstance(v, float)})
