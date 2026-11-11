"""Systematic scans of the optimal start day (Supplementary Note 3, Supplementary Table 9).

Scan 1: alpha x kappa x f0 around the timing example (140 combinations), each
        evaluated at every integer start day from the 30% criterion to day 500.
Grid 16: alpha in {2,4,6,8} x kappa in {0.018,...,0.030}, f0 = 1%.
Outputs
  results/timing_scan_1day.csv
  results/timing_grid16.csv
"""
import itertools

import numpy as np
from _common import parallel_map, write_csv

from evoswitch import TIMING_EXAMPLE, TIMING_THRESHOLD as BCRIT, INTERRUPTION_DAYS as TAU, response_day
from evoswitch.endpoints import progression_continuous, interruption_gain
from evoswitch.model import flow, initial_state

LAST_DAY = 500


def evaluate(args):
    alpha, kappa, f0 = args
    p = TIMING_EXAMPLE.with_(alpha=alpha, kappa=kappa, f0=f0)
    t_resp = response_day(p, 0.70, t_max=2000)
    if not np.isfinite(t_resp):
        return [alpha, kappa, f0, "no_response"] + [""] * 6
    t_cont = progression_continuous(p, BCRIT)
    days, gains = [], []
    for d in range(int(np.ceil(t_resp)), LAST_DAY + 1):
        if d >= t_cont:
            break
        g, crossed = interruption_gain(p, d, TAU, BCRIT, t_cont)
        if not crossed:
            days.append(d); gains.append(g)
    if len(days) < 4:
        return [alpha, kappa, f0, "too_few_days"] + [""] * 6
    gains = np.array(gains)
    i = int(np.argmax(gains))
    # first day on which a 50%, RECIST-equivalent (65.7%) or 80% reduction is reached
    deeper = [response_day(p, q, t_max=3000) for q in (0.5, 0.7 ** 3, 0.2)]
    return [alpha, kappa, f0, "ok", len(days), days[0], days[i], f"{gains[0]:.4f}",
            bool(np.all(np.diff(gains) < 0)),
            ";".join("" if not np.isfinite(t) else f"{t:.2f}" for t in deeper)]


def grid16(args):
    alpha, kappa = args
    p = TIMING_EXAMPLE.with_(alpha=alpha, kappa=kappa)
    t_resp = response_day(p)
    t_cont = progression_continuous(p, BCRIT)
    out = []
    for d in range(int(np.ceil(t_resp)), LAST_DAY + 1):
        if d >= t_cont:
            break
        g, crossed = interruption_gain(p, d, TAU, BCRIT, t_cont)
        if not crossed:
            out.append((d, g))
    best = max(out, key=lambda x: x[1])
    return [alpha, kappa, f"{t_resp:.2f}", out[0][0], f"{out[0][1]:.2f}", best[0] == out[0][0]]


if __name__ == "__main__":
    print("02 timing scans")
    combos = list(itertools.product([0.1, 0.5, 1, 2, 4, 6, 8], [0.016, 0.018, 0.022, 0.028, 0.034],
                                    [1e-4, 1e-3, 0.01, 0.05]))
    rows = parallel_map(evaluate, combos)
    write_csv("timing_scan_1day.csv",
              ["alpha", "kappa", "f0", "status", "n_start_days", "first_day", "best_day",
               "delta_T_first", "strictly_decreasing", "day_50_66_80_percent"], rows)
    ok = [r for r in rows if r[3] == "ok"]
    print(f"  evaluable: {len(ok)}; first day optimal: {sum(r[5] == r[6] for r in ok)}; "
          f"strictly decreasing: {sum(r[8] for r in ok)}; Delta T < 0 at first day: "
          f"{sum(float(r[7]) < 0 for r in ok)}")
    reach = [sum(1 for r in ok if r[9].split(';')[k] and float(r[9].split(';')[k]) <= LAST_DAY - 4)
             for k in range(3)]
    print(f"  reaching 50% / 65.7% / 80% reduction before day {LAST_DAY - 4}: {reach}")

    rows = parallel_map(grid16, list(itertools.product([2, 4, 6, 8], [0.018, 0.022, 0.026, 0.030])))
    write_csv("timing_grid16.csv", ["alpha", "kappa", "response_day", "first_day", "delta_T_first",
                                    "first_day_optimal"], rows)
