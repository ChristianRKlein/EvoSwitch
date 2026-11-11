"""Two-population model of drug-sensitive and drug-resistant cancer cells.

This module implements Eq. (1) of the manuscript:

    ds/dt = g_S * s * (1 - s - r) - kappa * u * s - nu * s
    dr/dt = g_R * r * (1 - r - alpha * s) + nu * s

s and r are the sensitive and resistant cell populations, scaled to a common
carrying capacity, b = s + r is the total tumor burden, and u is the treatment
status (u = 1: drug given, u = 0: treatment interruption). All rates are per
model day.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

# Numerical settings used for all results in the manuscript. They can be
# changed with set_solver() (used only by the convergence analysis).
SOLVER = {"method": "LSODA", "rtol": 1e-10, "atol": 1e-13, "max_step": 2.0}


def set_solver(rtol: float, atol: float, max_step: float) -> None:
    """Change the solver tolerances globally (used for convergence checks)."""
    SOLVER.update(rtol=rtol, atol=atol, max_step=max_step)


@dataclass(frozen=True)
class Params:
    """Model parameters.

    Attributes
    ----------
    g_S, g_R : growth rates of sensitive and resistant cells (1/day)
    kappa    : drug-induced death rate of sensitive cells (1/day)
    alpha    : competition coefficient (effect of sensitive on resistant cells)
    nu       : rate of conversion from sensitive to resistant cells (1/day)
    b0       : initial tumor burden (fraction of the carrying capacity)
    f0       : pre-existing resistant fraction of the initial tumor
    """

    g_S: float
    g_R: float
    kappa: float
    alpha: float
    nu: float
    b0: float
    f0: float

    def with_(self, **changes) -> "Params":
        """Return a copy with some parameters changed."""
        return replace(self, **changes)


def rhs(t: float, x, p: Params, u: int):
    """Right-hand side of Eq. (1)."""
    s, r = x
    ds = p.g_S * s * (1.0 - s - r) - p.kappa * u * s - p.nu * s
    dr = p.g_R * r * (1.0 - r - p.alpha * s) + p.nu * s
    return [ds, dr]


def burden_rate(t: float, x, p: Params, u: int) -> float:
    """Time derivative of the total burden b = s + r."""
    ds, dr = rhs(t, x, p, u)
    return ds + dr


def initial_state(p: Params) -> np.ndarray:
    """Initial state (s0, r0) = (b0 (1 - f0), b0 f0)."""
    return np.array([p.b0 * (1.0 - p.f0), p.b0 * p.f0])


def integrate(x, t0: float, t1: float, p: Params, u: int, events=None, t_eval=None):
    """Integrate Eq. (1) from t0 to t1 with constant treatment status u."""
    return solve_ivp(rhs, (t0, t1), np.asarray(x, float), args=(p, u), events=events,
                     t_eval=t_eval, **SOLVER)


def flow(x, duration: float, p: Params, u: int) -> np.ndarray:
    """State after `duration` days with constant treatment status u."""
    if duration <= 0:
        return np.asarray(x, float)
    return integrate(x, 0.0, duration, p, u).y[:, -1]


def trajectory(p: Params, segments, dt: float = 0.5):
    """Simulate a treatment schedule for plotting.

    Parameters
    ----------
    segments : list of (duration, u) pairs, applied in order from t = 0.

    Returns
    -------
    t, s, r : arrays sampled every `dt` days (segment boundaries included).
    """
    x, t0 = initial_state(p), 0.0
    T, S, R = [0.0], [x[0]], [x[1]]
    for duration, u in segments:
        n = max(1, int(np.ceil(duration / dt)))
        t_eval = np.linspace(t0, t0 + duration, n + 1)
        sol = integrate(x, t0, t0 + duration, p, u, t_eval=t_eval)
        T += list(sol.t[1:]); S += list(sol.y[0, 1:]); R += list(sol.y[1, 1:])
        x, t0 = sol.y[:, -1], t0 + duration
    return np.array(T), np.array(S), np.array(R)
