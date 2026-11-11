"""Start-time sensitivity of the interruption effect (Eq. S4).

    dDeltaT/dt1 = -kappa * (s2 n'M1 e_S - s1 n'M1 M0 e_S) / (n' F1(x_T))

x1: state at the start of the interruption, x2: state at restart, x_T: state at
progression; M0, M1: Jacobians of the off- and on-treatment flows; n = (1, 1),
e_S = (1, 0). The Jacobians are evaluated by central differences.
"""
from __future__ import annotations

import numpy as np

from .endpoints import progression_with_interruption
from .model import Params, flow, initial_state, rhs


def flow_jacobian(x, duration: float, p: Params, u: int, eps: float = 1e-7) -> np.ndarray:
    """Jacobian of the flow map x -> flow(x, duration) by central differences."""
    J = np.zeros((2, 2))
    for j in range(2):
        step = eps * max(1e-3, abs(x[j]))
        e = np.zeros(2); e[j] = step
        J[:, j] = (flow(x + e, duration, p, u) - flow(x - e, duration, p, u)) / (2 * step)
    return J


def start_time_sensitivity(p: Params, start: float, length: float, b_crit: float) -> float:
    """Derivative of Delta T with respect to the start day (Eq. S4)."""
    T = progression_with_interruption(p, start, length, b_crit).progression_time
    x1 = flow(initial_state(p), start, p, 1)
    x2 = flow(x1, length, p, 0)
    xT = flow(x2, T - start - length, p, 1)
    M0 = flow_jacobian(x1, length, p, 0)
    M1 = flow_jacobian(x2, T - start - length, p, 1)
    n, e_S = np.ones(2), np.array([1.0, 0.0])
    numerator = p.kappa * (x2[0] * n @ M1 @ e_S - x1[0] * n @ M1 @ M0 @ e_S)
    return -numerator / (n @ np.array(rhs(0.0, xT, p, 1)))


def start_time_sensitivity_fd(p: Params, start: float, length: float, b_crit: float,
                              h: float = 0.01) -> float:
    """Central finite difference of the progression time with respect to the start day."""
    plus = progression_with_interruption(p, start + h, length, b_crit).progression_time
    minus = progression_with_interruption(p, start - h, length, b_crit).progression_time
    return (plus - minus) / (2 * h)
