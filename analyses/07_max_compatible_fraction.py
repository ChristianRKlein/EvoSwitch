"""Resistant fraction compatible with a response (Supplementary Note 4, Suppl. Table 13).

A tumor cannot shrink below its resistant population. For a reduction A at
day 56, the pre-existing resistant fraction is therefore bounded from above.
This script computes the bound (Eq. S5 and numerically for three competition
strengths) and checks whether any upper critical fraction exists below it.
Outputs
  results/max_compatible_fraction.csv
  results/no_upper_limit_check.csv
"""
import numpy as np
from _common import parallel_map, write_csv

from evoswitch import (MATCHED_BASE, MATCHED_THRESHOLD as BCRIT, matched, max_compatible_fraction,
                       max_compatible_fraction_formula)
from evoswitch.endpoints import interruption_gain

REDUCTIONS = (0.30, 0.50, 1 - 0.7 ** 3, 0.80, 0.90, 0.99, 0.999)


def numerical(args):
    shrinkage, alpha = args
    return max_compatible_fraction(MATCHED_BASE.with_(alpha=alpha), shrinkage)


def min_gain_above(args):
    """Minimum Delta T for f0 between 15% and 97% of the maximum compatible fraction."""
    alpha, shrinkage = args
    f_max = max_compatible_fraction_formula(shrinkage, 56, MATCHED_BASE.g_R, MATCHED_BASE.b0)
    lo = 0.15 if f_max > 0.2 else 0.01
    gains = []
    for f0 in np.linspace(lo, 0.97 * f_max, 25):
        g, crossed = interruption_gain(matched(MATCHED_BASE.with_(f0=f0, alpha=alpha), shrinkage), 56, 28, BCRIT)
        gains.append(-np.inf if crossed else g)
    rare = interruption_gain(matched(MATCHED_BASE.with_(f0=1e-6, alpha=alpha), shrinkage), 56, 28, BCRIT)[0]
    return [alpha, shrinkage, f"{100 * lo:.1f}", f"{100 * 0.97 * f_max:.2f}", f"{min(gains):.3f}", f"{rare:.3f}"]


if __name__ == "__main__":
    print("07 maximum compatible fraction")
    combos = [(a, al) for a in REDUCTIONS for al in (0.1, 1.0, 6.0)]
    values = parallel_map(numerical, combos)
    rows = []
    for a in REDUCTIONS:
        num = [values[combos.index((a, al))] for al in (0.1, 1.0, 6.0)]
        formula = max_compatible_fraction_formula(a, 56, MATCHED_BASE.g_R, MATCHED_BASE.b0)
        rows.append([f"{100 * a:.1f}", f"{100 * formula:.4f}"] + [f"{100 * v:.4f}" for v in num])
    write_csv("max_compatible_fraction.csv", ["reduction_percent", "eq_S5_percent", "alpha_0.1_percent",
                                              "alpha_1_percent", "alpha_6_percent"], rows)

    checks = [(0.1, s) for s in (0.45, 0.475, 0.50, 0.60, 0.70, 0.90)]
    checks += [(0.2, s) for s in (0.375, 0.40, 0.425, 0.45, 0.475, 0.50)]
    checks += [(a, s) for a in (0.3, 0.5, 1.0) for s in (0.30, 0.40)]
    write_csv("no_upper_limit_check.csv", ["alpha", "shrinkage", "f0_from_percent", "f0_to_percent",
                                           "min_delta_T_days", "delta_T_at_f0_1e-6"],
              parallel_map(min_gain_above, checks))
