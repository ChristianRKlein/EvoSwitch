"""Response, progression and the effect of a treatment interruption.

Progression (main analyses) is the first upward crossing of a fixed burden
threshold b_crit. The effect of an interruption is

    Delta T = T_interruption - T_continuous,

the time to progression gained (positive) or lost (negative) compared with
continuous therapy. Interruptions during which the burden already crosses
b_crit are flagged and excluded from the analyses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Params, flow, initial_state, integrate

T_MAX = 20000.0  # simulation horizon (model days)


def first_upward_crossing(x, t0: float, p: Params, u: int, threshold: float,
                          t_max: float = T_MAX):
    """First time >= t0 at which the burden rises through `threshold`.

    Returns (time, state); time is inf if no crossing occurs before t_max.
    """
    def crossing(t, y, p, u):
        return y[0] + y[1] - threshold
    crossing.terminal, crossing.direction = True, 1
    sol = integrate(x, t0, t_max, p, u, events=crossing)
    if sol.t_events[0].size:
        return sol.t_events[0][0], sol.y_events[0][0]
    return np.inf, sol.y[:, -1]


def response_day(p: Params, remaining: float = 0.70, t_max: float = 5000.0) -> float:
    """First day on which the burden has fallen to `remaining` x b0.

    remaining = 0.70 is a 30% reduction; remaining = 0.7**3 = 0.343 is the
    volume equivalent of a RECIST partial response (30% reduction in diameter).
    Returns inf if the criterion is never met.
    """
    def reached(t, y, p, u):
        return y[0] + y[1] - remaining * p.b0
    reached.terminal, reached.direction = True, -1
    sol = integrate(initial_state(p), 0.0, t_max, p, 1, events=reached)
    return sol.t_events[0][0] if sol.t_events[0].size else np.inf


def progression_continuous(p: Params, b_crit: float, t_max: float = T_MAX) -> float:
    """Time to progression under continuous therapy."""
    return first_upward_crossing(initial_state(p), 0.0, p, 1, b_crit, t_max)[0]


@dataclass
class InterruptionOutcome:
    progression_time: float
    crossed_during_interruption: bool


def progression_with_interruption(p: Params, start: float, length: float, b_crit: float,
                                  t_max: float = T_MAX) -> InterruptionOutcome:
    """Time to progression with one interruption on [start, start + length)."""
    x1 = flow(initial_state(p), start, p, 1)
    t_cross, _ = first_upward_crossing(x1, start, p, 0, b_crit, start + length)
    if np.isfinite(t_cross):
        return InterruptionOutcome(t_cross, True)
    x2 = flow(x1, length, p, 0)
    t_prog, _ = first_upward_crossing(x2, start + length, p, 1, b_crit, t_max)
    return InterruptionOutcome(t_prog, False)


def interruption_gain(p: Params, start: float, length: float, b_crit: float,
                      t_continuous: float | None = None) -> tuple[float, bool]:
    """Delta T for an interruption, and whether it crossed b_crit while off treatment."""
    if t_continuous is None:
        t_continuous = progression_continuous(p, b_crit)
    out = progression_with_interruption(p, start, length, b_crit)
    return out.progression_time - t_continuous, out.crossed_during_interruption


def gain_by_start_day(p: Params, start_days, length: float, b_crit: float):
    """Delta T for each start day (NaN where the interruption crosses b_crit)."""
    t_cont = progression_continuous(p, b_crit)
    gains = []
    for d in start_days:
        g, crossed = interruption_gain(p, d, length, b_crit, t_cont)
        gains.append(np.nan if crossed else g)
    return np.array(gains)
