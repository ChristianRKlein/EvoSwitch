"""Two tumors with identical early response (Table 1, Fig. 3a,b; Suppl. Tables 3-5).

Tumor A: 0.0001% resistant cells, weak competition (alpha = 0.1).
Tumor B: 3% resistant cells, strong competition (alpha = 3).
Both are matched to the same shrinkage at day 56.
Outputs
  results/tumors_AB.csv                 full-precision summary (Table 1, Suppl. Table 3)
  results/tumors_AB_robustness.csv      interruption length x threshold (Suppl. Table 4)
  results/tumors_AB_deeper_response.csv 40-90% response (Suppl. Table 5)
  results/tumors_AB_trajectories.csv    burden curves for Fig. 3a,b
"""
import numpy as np
from _common import write_csv

from evoswitch import MATCHED_THRESHOLD as BCRIT, tumor_A, tumor_B, trajectory
from evoswitch.endpoints import progression_continuous, progression_with_interruption, interruption_gain
from evoswitch.model import flow, initial_state

print("04 tumors A and B")
rows = []
for name, p in (("A", tumor_A()), ("B", tumor_B())):
    x56 = flow(initial_state(p), 56, p, 1)
    tc = progression_continuous(p, BCRIT)
    th = progression_with_interruption(p, 56, 28, BCRIT)
    rows.append([name, p.f0, p.alpha, f"{p.kappa:.10f}", f"{100 * x56[1] / x56.sum():.5f}",
                 f"{tc:.2f}", f"{th.progression_time:.2f}", f"{th.progression_time - tc:.2f}"])
write_csv("tumors_AB.csv", ["tumor", "f0", "alpha", "kappa", "resistant_percent_day56",
                            "progression_continuous", "progression_interruption", "delta_T_days"], rows)

rows = []
for tau in (21, 28, 42, 56):
    row = [tau]
    for b_crit in (0.65, 0.70, 0.75):
        for p in (tumor_A(), tumor_B()):
            g, crossed = interruption_gain(p, 56, tau, b_crit)
            row.append("crossed" if crossed else f"{g:.2f}")
    rows.append(row)
write_csv("tumors_AB_robustness.csv", ["interruption_days", "A_0.65", "B_0.65", "A_0.70", "B_0.70",
                                       "A_0.75", "B_0.75"], rows)

rows = []
for shrink in (0.40, 1 - 0.7 ** 3, 0.80, 0.90):
    row = [f"{100 * shrink:.1f}"]
    for p in (tumor_A(shrink), tumor_B(shrink)):
        for tau in (28, 56):
            g, crossed = interruption_gain(p, 56, tau, BCRIT)
            row.append("crossed" if crossed else f"{g:.2f}")
    rows.append(row)
write_csv("tumors_AB_deeper_response.csv", ["reduction_percent", "A_28d", "A_56d", "B_28d", "B_56d"], rows)

rows = []
for name, p in (("A", tumor_A()), ("B", tumor_B())):
    for label, segs in (("continuous", [(1300, 1)]), ("interruption", [(56, 1), (28, 0), (1216, 1)])):
        t, s, r = trajectory(p, segs, dt=2.0)
        rows += [[name, label, f"{a:g}", f"{b + c:.8g}"] for a, b, c in zip(t, s, r)]
write_csv("tumors_AB_trajectories.csv", ["tumor", "schedule", "day", "total_burden"], rows)
