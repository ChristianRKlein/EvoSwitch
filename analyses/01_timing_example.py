"""Timing example (Fig. 2; Supplementary Tables 6 and 7; Supplementary Fig. 1).

Tumor with 1% pre-existing resistant cells and strong competition (alpha = 6).
Outputs
  results/timing_trajectory.csv          sensitive, resistant and total burden (Fig. 2a)
  results/timing_gain_by_start_day.csv   Delta T for each start day (Fig. 2b)
  results/timing_response_criteria.csv   first eligible day and Delta T per criterion (Suppl. Table 7)
  results/start_time_sensitivity.csv     Eq. (S4) vs finite differences (Suppl. Table 6, Suppl. Fig. 1)
  results/timing_rare_resistance.csv     optimal start with 0.0001% resistant cells (Suppl. Note 3)
"""
import numpy as np
from _common import write_csv

from evoswitch import (TIMING_EXAMPLE as P, TIMING_THRESHOLD as BCRIT, INTERRUPTION_DAYS as TAU,
                       response_day, trajectory, gain_by_start_day, flow, initial_state)
from evoswitch.sensitivity import start_time_sensitivity, start_time_sensitivity_fd

print("01 timing example")

t, s, r = trajectory(P, [(750, 1)], dt=1.0)
write_csv("timing_trajectory.csv", ["day", "sensitive", "resistant", "total"],
          [[f"{a:g}", f"{b:.8g}", f"{c:.8g}", f"{b + c:.8g}"] for a, b, c in zip(t, s, r)])

starts = np.arange(38, 701)
gains = gain_by_start_day(P, starts, TAU, BCRIT)
write_csv("timing_gain_by_start_day.csv", ["start_day", "delta_T_days"],
          [[d, f"{g:.6f}"] for d, g in zip(starts, gains)])

rows = []
for reduction in (0.30, 0.50, 1 - 0.7 ** 3, 0.80):
    first = int(np.ceil(response_day(P, 1 - reduction)))
    g = gain_by_start_day(P, [first], TAU, BCRIT)[0]
    x1 = flow(initial_state(P), first, P, 1)
    x2 = flow(x1, TAU, P, 0)
    rows.append([f"{100 * reduction:.1f}", first, f"{g:.2f}",
                 f"{100 * x1[1] / x1.sum():.2f}", f"{100 * x2[1] / x2.sum():.2f}"])
write_csv("timing_response_criteria.csv",
          ["reduction_percent", "first_eligible_day", "delta_T_days",
           "resistant_percent_at_start", "resistant_percent_at_restart"], rows)

rows = []
for d in (38, 60, 100, 150, 200, 300, 400, 500, 600):
    rows.append([d, f"{start_time_sensitivity(P, d, TAU, BCRIT):.7f}",
                 f"{start_time_sensitivity_fd(P, d, TAU, BCRIT):.7f}"])
write_csv("start_time_sensitivity.csv", ["start_day", "eq_S4", "finite_difference"], rows)

rare = P.with_(f0=1e-6)
first = int(np.ceil(response_day(rare)))
days = np.arange(first, 501)
g = gain_by_start_day(rare, days, TAU, BCRIT)
best = int(np.nanargmax(g))
write_csv("timing_rare_resistance.csv", ["f0", "first_day", "delta_T_first", "best_day", "delta_T_best"],
          [[1e-6, first, f"{g[0]:.2f}", int(days[best]), f"{g[best]:.2f}"]])

print(f"  30% criterion on day {response_day(P):.2f}; RECIST-equivalent on day "
      f"{response_day(P, 0.7 ** 3):.2f}; Delta T(38) = {gains[0]:.2f}, Delta T(126) = {gains[88]:.2f}")
