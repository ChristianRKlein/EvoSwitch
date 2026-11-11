"""Critical fractions of pre-existing resistance (Fig. 4; Table 2; Suppl. Tables 10-12; Suppl. Fig. 2).

Weak competition (alpha = 0.1) unless stated otherwise; 4-week interruption
from day 56; progression threshold 0.70.
Outputs
  results/fixed_drug_effect_table.csv    only f0 varies, kappa = 0.018 (Suppl. Table 10)
  results/fixed_drug_effect_curve.csv    the same on a fine grid (Suppl. Fig. 2)
  results/fixed_drug_effect_roots.csv    lower and upper critical fractions at fixed kappa
  results/limit_curves.csv               Delta T vs f0 at 30/40/50% shrinkage (Fig. 4a)
  results/upper_limits.csv               upper critical fraction vs shrinkage (Fig. 4b, Table 2, Suppl. Table 11)
  results/lower_limits.csv               lower critical fraction at matched response (Suppl. Table 11)
  results/lower_limit_scaling.csv        lower critical fraction vs nu and alpha (Suppl. Table 12)
"""
import numpy as np
from scipy.optimize import brentq
from _common import parallel_map, write_csv

from evoswitch import MATCHED_BASE, MATCHED_THRESHOLD as BCRIT, matched, max_compatible_fraction_formula
from evoswitch.endpoints import interruption_gain
from evoswitch.model import flow, initial_state

FIXED = MATCHED_BASE.with_(alpha=0.1, kappa=0.018)


def gain_fixed(f0, base=FIXED):
    return interruption_gain(base.with_(f0=f0), 56, 28, BCRIT)[0]


def gain_matched(f0, alpha, shrinkage):
    p = matched(MATCHED_BASE.with_(f0=f0, alpha=alpha), shrinkage)
    g, crossed = interruption_gain(p, 56, 28, BCRIT)
    return np.nan if crossed else g


def sign_changes(func, grid):
    values = [func(x) for x in grid]
    return [brentq(func, grid[i], grid[i + 1], xtol=1e-12 * grid[i])
            for i in range(len(grid) - 1) if values[i] > 0 >= values[i + 1] or values[i] <= 0 < values[i + 1]]


def upper_limit(args):
    alpha, shrinkage = args
    f_max = max_compatible_fraction_formula(shrinkage, 56, MATCHED_BASE.g_R, MATCHED_BASE.b0)
    grid = np.r_[0.001, 0.003, np.linspace(0.01, 0.97 * f_max, 30)]
    func = lambda f: gain_matched(f, alpha, shrinkage)
    values = [func(f) for f in grid]
    for i in range(len(grid) - 1):
        if values[i] > 0 >= values[i + 1]:
            return [alpha, shrinkage, f"{100 * brentq(func, grid[i], grid[i + 1], xtol=1e-10):.4f}",
                    f"{100 * f_max:.2f}"]
    return [alpha, shrinkage, "none", f"{100 * f_max:.2f}"]


def curve(args):
    shrinkage, f0 = args
    return gain_matched(f0, 0.1, shrinkage)


if __name__ == "__main__":
    print("06 resistance limits")
    def shrinkage(f0):
        p = FIXED.with_(f0=f0)
        return 1 - flow(initial_state(p), 56, p, 1).sum() / p.b0

    write_csv("fixed_drug_effect_table.csv", ["f0_percent", "shrinkage_percent", "delta_T_days"],
              [[f"{100 * f0:g}", f"{100 * shrinkage(f0):.2f}", f"{gain_fixed(f0):.2f}"]
               for f0 in (1e-4, 1e-3, 0.01, 0.03, 0.06, 0.08, 0.10)])
    write_csv("fixed_drug_effect_curve.csv", ["f0_percent", "shrinkage_percent", "delta_T_days"],
              [[f"{100 * f0:.4f}", f"{100 * shrinkage(f0):.4f}", f"{gain_fixed(f0):.4f}"]
               for f0 in np.linspace(1e-4, 0.10, 41)])

    lower = sign_changes(gain_fixed, np.logspace(-8, -4, 41))
    upper = brentq(gain_fixed, 0.01, 0.05, xtol=1e-12)
    write_csv("fixed_drug_effect_roots.csv", ["lower_percent", "upper_percent"],
              [[f"{100 * lower[0]:.8f}", f"{100 * upper:.8f}"]])

    fractions = np.logspace(-6, np.log10(0.15), 45)
    combos = [(a, f) for a in (0.3, 0.4, 0.5) for f in fractions]
    gains = parallel_map(curve, combos)
    write_csv("limit_curves.csv", ["shrinkage", "f0", "delta_T_days"],
              [[a, f"{f:.8g}", f"{g:.6f}"] for (a, f), g in zip(combos, gains)])

    combos = [(alpha, round(s, 3)) for alpha in (0.1, 0.2) for s in np.arange(0.30, 0.5001, 0.025)]
    combos += [(alpha, s) for alpha in (0.3, 0.5, 1.0) for s in (0.30, 0.40)]
    rows = parallel_map(upper_limit, combos)
    write_csv("upper_limits.csv", ["alpha", "shrinkage", "upper_limit_percent",
                                   "max_compatible_percent"], rows)

    rows = []
    for shrinkage in (0.30, 0.40, 0.50):
        roots = sign_changes(lambda f: gain_matched(f, 0.1, shrinkage), np.logspace(-8, -3, 26))
        rows.append([shrinkage, f"{100 * roots[0]:.6f}" if roots else "none"])
    write_csv("lower_limits.csv", ["shrinkage", "lower_limit_percent"], rows)

    rows = []
    for label, changes in (("nu x 0.1", dict(nu=2.5e-9)), ("reference", {}), ("nu x 10", dict(nu=2.5e-7)),
                           ("alpha = 0.05", dict(alpha=0.05)), ("alpha = 0.2", dict(alpha=0.2))):
        base = FIXED.with_(**changes)
        root = sign_changes(lambda f: gain_fixed(f, base), np.logspace(-9, -4, 51))[0]
        p = base.with_(f0=root)
        r56 = flow(initial_state(p), 56, p, 1)[1]
        rows.append([label, p.nu, p.alpha, f"{100 * root:.7f}", f"{p.alpha * p.g_R * r56 / p.nu:.3f}"])
    write_csv("lower_limit_scaling.csv", ["variant", "nu", "alpha", "lower_limit_percent",
                                          "alpha_gR_r56_over_nu"], rows)
