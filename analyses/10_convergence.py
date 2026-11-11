"""Numerical convergence (Supplementary Table 17).

Delta T for the key examples at three solver settings.
Output
  results/convergence.csv
"""
from _common import write_csv

from evoswitch import MATCHED_BASE, TIMING_EXAMPLE, set_solver, tumor_A, tumor_B
from evoswitch.endpoints import interruption_gain

print("10 convergence")
rows = []
for rtol, max_step in ((1e-8, 2.0), (1e-10, 2.0), (1e-12, 0.5)):
    set_solver(rtol=rtol, atol=1e-13, max_step=max_step)
    fixed = MATCHED_BASE.with_(alpha=0.1, kappa=0.018)
    values = [interruption_gain(TIMING_EXAMPLE, 38, 28, 0.27)[0],
              interruption_gain(tumor_A(), 56, 28, 0.70)[0],
              interruption_gain(tumor_B(), 56, 28, 0.70)[0],
              interruption_gain(fixed.with_(f0=1e-4), 56, 28, 0.70)[0],
              interruption_gain(fixed.with_(f0=0.10), 56, 28, 0.70)[0]]
    rows.append([rtol, max_step] + [f"{v:.5f}" for v in values])
set_solver(rtol=1e-10, atol=1e-13, max_step=2.0)
write_csv("convergence.csv", ["rtol", "max_step", "timing_example", "tumor_A", "tumor_B", "f0_0.01pct",
                              "f0_10pct"], rows)
