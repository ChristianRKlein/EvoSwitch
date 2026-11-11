"""Nadir-relative progression criteria and restart rules (Supplementary Note 5).

Progression is the first time the burden reaches q times the lowest burden
reached so far (the running nadir). q = 1.20 is a marker-based criterion;
q = 1.2**3 = 1.728 is the volume equivalent of RECIST progression (a 20%
increase in diameter).

Assumption (checked in all settings of the manuscript): the burden declines
monotonically before the interruption starts and does not fall below its value
at the start of the interruption while treatment is withheld, so that the
burden at the start of the interruption is the nadir.
"""
from __future__ import annotations

import numpy as np

from .model import Params, burden_rate, flow, initial_state, integrate

RECIST_VOLUME_Q = 1.2 ** 3
T_MAX = 20000.0


def _rising_through(level):
    def event(t, y, p, u):
        return y[0] + y[1] - level
    event.terminal, event.direction = True, 1
    return event


def progression_on_treatment(x, t0: float, p: Params, running_nadir: float, q: float = 1.2,
                             t_max: float = T_MAX) -> float:
    """Nadir-relative progression time on continuous treatment from state x at t0.

    The burden first falls to its minimum (if it is still declining) and then
    progression occurs when it reaches q x min(running_nadir, minimum).
    """
    x = np.asarray(x, float)
    if burden_rate(t0, x, p, 1) >= 0:          # already rising: nadir is at t0
        t_min, x_min = t0, x
    else:
        def minimum(t, y, p, u):
            return burden_rate(t, y, p, u)
        minimum.terminal, minimum.direction = True, 1
        sol = integrate(x, t0, t_max, p, 1, events=minimum)
        if not sol.t_events[0].size:
            return np.inf
        t_min, x_min = sol.t_events[0][0], sol.y_events[0][0]
    level = q * min(running_nadir, x_min.sum())
    sol = integrate(x_min, t_min, t_max, p, 1, events=_rising_through(level))
    return sol.t_events[0][0] if sol.t_events[0].size else np.inf


def progression_continuous_nadir(p: Params, q: float = 1.2) -> float:
    """Nadir-relative progression time under continuous therapy."""
    return progression_on_treatment(initial_state(p), 0.0, p, p.b0, q)


def time_to_criterion_off_treatment(p: Params, start: float, q: float = 1.2) -> float:
    """tau_q: time an uninterrupted withdrawal from `start` needs to reach q x b(start)."""
    x1 = flow(initial_state(p), start, p, 1)
    sol = integrate(x1, start, start + 2000.0, p, 0, events=_rising_through(q * x1.sum()))
    return sol.t_events[0][0] - start if sol.t_events[0].size else np.inf


def tau_q_approximation(b1: float, g_S: float, q: float = 1.2) -> float:
    """Logistic approximation for a predominantly sensitive tumor."""
    return np.log(q * (1 - b1) / (1 - q * b1)) / g_S


def tau_q_lower_bound(b1: float, p: Params, q: float = 1.2) -> float:
    """Lower bound of Proposition S2 (Eq. S6)."""
    g_max, a = max(p.g_S, p.g_R), min(p.alpha, 1.0)
    return np.log(q * (1 - a * b1) / (1 - q * a * b1)) / g_max


def fixed_interruption_nadir(p: Params, start: float, length: float, q: float = 1.2):
    """Gain from a fixed-length interruption under the nadir-relative criterion.

    Returns (gain, crossed_during_interruption, time_to_criterion_if_crossed).
    """
    x1 = flow(initial_state(p), start, p, 1)
    nadir = x1.sum()
    t_cont = progression_continuous_nadir(p, q)
    sol = integrate(x1, start, start + length, p, 0, events=_rising_through(q * nadir))
    if sol.t_events[0].size:
        return sol.t_events[0][0] - t_cont, True, sol.t_events[0][0] - start
    x2 = sol.y[:, -1]
    return progression_on_treatment(x2, start + length, p, nadir, q) - t_cont, False, np.nan


def restart_after_fraction(p: Params, start: float, fraction: float = 0.8, q: float = 1.2):
    """Restart after `fraction` x tau_q. Returns (gain, tau_q, drug-free days)."""
    tau_q = time_to_criterion_off_treatment(p, start, q)
    x1 = flow(initial_state(p), start, p, 1)
    off = fraction * tau_q
    x2 = flow(x1, off, p, 0)
    gain = progression_on_treatment(x2, start + off, p, x1.sum(), q) - progression_continuous_nadir(p, q)
    return gain, tau_q, off


def restart_at_marker(p: Params, start: float, marker: float = 1.10, q: float = 1.2):
    """Restart once the burden reaches `marker` x nadir. Returns (gain, drug-free days)."""
    x1 = flow(initial_state(p), start, p, 1)
    nadir = x1.sum()
    sol = integrate(x1, start, start + 2000.0, p, 0, events=_rising_through(marker * nadir))
    t_restart, x_restart = sol.t_events[0][0], sol.y_events[0][0]
    gain = progression_on_treatment(x_restart, t_restart, p, nadir, q) - progression_continuous_nadir(p, q)
    return gain, t_restart - start
