"""Twelve extreme scenarios (Supplementary Note 5, Supplementary Table 15).

Each scenario changes one feature of the reference settings and re-evaluates
(i) the optimal start in the timing example, (ii) tumors A and B and (iii) the
upper critical fraction at 30% and 40% shrinkage under weak competition.
Output
  results/extreme_scenarios.csv
"""
import numpy as np
from scipy.optimize import brentq
from _common import parallel_map, write_csv

from evoswitch import MATCHED_BASE, TIMING_EXAMPLE, matched, response_day
from evoswitch.endpoints import progression_continuous, interruption_gain

SCENARIOS = [  # name, timing-example changes, response-matched changes, thresholds, interruption length
    ("Reference", {}, {}, 0.27, 0.70, 28),
    ("No fitness cost (g_R = g_S)", dict(g_R=0.0156), dict(g_R=0.015), 0.27, 0.70, 28),
    ("Resistant cells faster (g_R = 1.2 g_S)", dict(g_R=0.0187), dict(g_R=0.018), 0.27, 0.70, 28),
    ("Large fitness cost (g_R = 0.5 g_S)", dict(g_R=0.0078), dict(g_R=0.0075), 0.27, 0.70, 28),
    ("No new resistance (nu = 0)", dict(nu=0.0), dict(nu=0.0), 0.27, 0.70, 28),
    ("Frequent new resistance (nu = 1e-6)", dict(nu=1e-6), dict(nu=1e-6), 0.27, 0.70, 28),
    ("Slow growth (g_S, g_R x 0.5)", dict(g_S=0.0078, g_R=0.00455), dict(g_S=0.0075, g_R=0.006), 0.27, 0.70, 28),
    ("Fast growth (g_S, g_R, kappa x 2)", dict(g_S=0.0312, g_R=0.0182, kappa=0.044), dict(g_S=0.03, g_R=0.024),
     0.27, 0.70, 28),
    ("Tight threshold", {}, {}, 0.26, 0.63, 28),
    ("Loose threshold", {}, {}, 0.40, 0.90, 28),
    ("Short interruption (14 days)", {}, {}, 0.27, 0.70, 14),
    ("Long interruption (56 days)", {}, {}, 0.27, 0.70, 56),
]


def optimal_start(p, b_crit, tau):
    """One-day grid over the first 60 eligible days, then every 10 days for 400 days."""
    first = int(np.ceil(response_day(p)))
    t_cont = progression_continuous(p, b_crit)
    out = []
    for d in list(range(first, first + 61)) + list(range(first + 70, first + 400, 10)):
        if d + tau >= t_cont:
            break
        g, crossed = interruption_gain(p, d, tau, b_crit, t_cont)
        if not crossed:
            out.append((d, g))
    best = max(out, key=lambda x: x[1])
    return out[0][0], best[0] == out[0][0], out[0][1]


def upper_limit(base, b_crit, tau, shrinkage, alpha=0.1):
    def gain(f0):
        p = matched(base.with_(alpha=alpha, f0=f0), shrinkage)
        g, crossed = interruption_gain(p, 56, tau, b_crit)
        return np.nan if crossed else g
    grid = np.r_[0.0005, 0.001, 0.002, 0.005, np.linspace(0.01, 0.15, 15)]
    values = [gain(f) for f in grid]
    for i in range(len(grid) - 1):
        if values[i] > 0 >= values[i + 1]:
            return f"{100 * brentq(gain, grid[i], grid[i + 1], xtol=1e-9):.2f}"
    return ">15"


def evaluate(scenario):
    name, d_timing, d_matched, thr_t, thr_m, tau = scenario
    first, earliest, gain_first = optimal_start(TIMING_EXAMPLE.with_(**d_timing), thr_t, tau)
    base = MATCHED_BASE.with_(**d_matched)
    ab = []
    for f0, alpha in ((1e-6, 0.1), (0.03, 3.0)):
        g, crossed = interruption_gain(matched(base.with_(f0=f0, alpha=alpha), 0.40), 56, tau, thr_m)
        ab.append("crossed" if crossed else f"{g:.1f}")
    return [name, first, earliest, f"{gain_first:.1f}", *ab,
            upper_limit(base, thr_m, tau, 0.30), upper_limit(base, thr_m, tau, 0.40)]


if __name__ == "__main__":
    print("09 extreme scenarios")
    write_csv("extreme_scenarios.csv", ["scenario", "first_day", "first_day_optimal", "delta_T_first",
                                        "tumor_A", "tumor_B", "upper_limit_30", "upper_limit_40"],
              parallel_map(evaluate, SCENARIOS))
