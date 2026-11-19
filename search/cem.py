"""Black-box optimizers over a unit box: CEM, iCEM (decaying population + elite reuse + variance floor), MPPI. Minimise f(u) -> cost.
Every optimizer returns (best_u, best_cost, history) and logs every evaluated (u, cost) in `log` for the recovery-profile dataset."""
import numpy as np


def _clip(u): return np.clip(u, 0.0, 1.0)


def cem(f, dim, rng, pop=64, iters=10, elite_frac=0.125, init_std=0.3, alpha=0.7, log=None):
    mu, sd = np.full(dim, 0.5), np.full(dim, init_std); best = (None, np.inf); hist = []
    ne = max(2, int(pop * elite_frac))
    for it in range(iters):
        U = _clip(mu + sd * rng.standard_normal((pop, dim))); c = np.array([f(u) for u in U])
        if log is not None: log.extend(zip(U, c))
        idx = np.argsort(c)[:ne]; E = U[idx]
        mu = alpha * E.mean(0) + (1 - alpha) * mu; sd = alpha * E.std(0) + (1 - alpha) * sd + 1e-3
        if c[idx[0]] < best[1]: best = (U[idx[0]].copy(), float(c[idx[0]]))
        hist.append(best[1])
    return best[0], best[1], hist


def icem(f, dim, rng, pop=64, iters=10, elite_frac=0.125, init_std=0.3, decay=1.25, keep_frac=0.3, min_pop=16, sd_floor=0.02, alpha=0.7, log=None, init_mu=None):
    """iCEM-style: population shrinks by `decay` each iteration, the best elites of the previous iteration are re-evaluated in the next one
    (keep_frac), the candidate mean is also injected, and a variance floor stops early collapse. (Colored noise does not apply to a static parameter vector.)"""
    mu, sd = (np.full(dim, 0.5) if init_mu is None else np.array(init_mu, float)), np.full(dim, init_std); best = (None, np.inf); hist = []; keep = np.zeros((0, dim)); n = float(pop)      # init_mu: warm start from a neighbouring configuration
    for it in range(iters):
        npop = max(min_pop, int(round(n))); n /= decay
        U = _clip(mu + sd * rng.standard_normal((npop, dim)))
        U = np.vstack([U, keep, mu[None]]) if len(keep) else np.vstack([U, mu[None]])
        c = np.array([f(u) for u in U])
        if log is not None: log.extend(zip(U, c))
        ne = max(2, int(len(U) * elite_frac)); idx = np.argsort(c)[:ne]; E = U[idx]
        mu = alpha * E.mean(0) + (1 - alpha) * mu; sd = np.maximum(alpha * E.std(0) + (1 - alpha) * sd, sd_floor)
        keep = E[:max(1, int(keep_frac * ne))]
        if c[idx[0]] < best[1]: best = (U[idx[0]].copy(), float(c[idx[0]]))
        hist.append(best[1])
    return best[0], best[1], hist


def mppi(f, dim, rng, pop=64, iters=10, lam=None, init_std=0.3, sd_decay=0.8, log=None):
    mu, sd = np.full(dim, 0.5), init_std; best = (None, np.inf); hist = []
    for it in range(iters):
        U = _clip(mu + sd * rng.standard_normal((pop, dim))); c = np.array([f(u) for u in U])
        if log is not None: log.extend(zip(U, c))
        l = lam if lam is not None else max(np.std(c), 1e-6) * 0.5
        w = np.exp(-(c - c.min()) / l); w /= w.sum(); mu = _clip((w[:, None] * U).sum(0)); sd *= sd_decay
        if c.min() < best[1]: best = (U[c.argmin()].copy(), float(c.min()))
        hist.append(best[1])
    return best[0], best[1], hist
