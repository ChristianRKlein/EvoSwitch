"""Optimal start and drug effect (Supplementary Note 3, Supplementary Table 8).

If the drug effect kappa is close to or below the growth rate of sensitive cells
g_S, therapy holds sensitive cells in check without depleting them and a later
interruption can be better. This script scans kappa/g_S in both parameter
settings and reports the worked example (kappa = 0.9 g_S).
Outputs
  results/drug_effect_scan.csv
  results/drug_effect_example.csv
"""
import itertools

import numpy as np
from _common import parallel_map, write_csv

from evoswitch import MATCHED_BASE, MATCHED_THRESHOLD, TIMING_EXAMPLE, TIMING_THRESHOLD, response_day
from evoswitch.endpoints import progression_continuous, interruption_gain, gain_by_start_day

TAU = 28.0
SETTINGS = {"timing": (TIMING_EXAMPLE, TIMING_THRESHOLD),
            "response_matched": (MATCHED_BASE, MATCHED_THRESHOLD)}


def evaluate(args):
    setting, ratio, alpha, f0 = args
    base, b_crit = SETTINGS[setting]
    p = base.with_(kappa=ratio * base.g_S, alpha=alpha, f0=f0)
    t_resp = response_day(p, 0.70, t_max=5000)
    t_cont = progression_continuous(p, b_crit)
    if not (np.isfinite(t_resp) and np.isfinite(t_cont)):
        return [setting, ratio, alpha, f0, "no_response", "", "", "", "", ""]
    first = int(np.ceil(t_resp))
    days = np.arange(first, int(min(t_cont - 30, first + 1500)), 10)
    gains = []
    for d in days:
        g, crossed = interruption_gain(p, d, TAU, b_crit, t_cont)
        gains.append(np.nan if crossed else g)
    gains = np.array(gains)
    i = int(np.nanargmax(gains))
    return [setting, ratio, alpha, f0, "ok", first, f"{gains[0]:.3f}", int(days[i]), f"{gains[i]:.3f}",
            bool(days[i] - first < 10)]


if __name__ == "__main__":
    print("03 drug-effect scan")
    combos = list(itertools.product(SETTINGS, [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.2],
                                    [0.5, 2.0, 6.0], [1e-4, 1e-3, 1e-2, 5e-2]))
    rows = parallel_map(evaluate, combos)
    write_csv("drug_effect_scan.csv", ["setting", "kappa_over_gS", "alpha", "f0", "status", "first_day",
                                       "delta_T_first", "best_day", "delta_T_best",
                                       "first_day_optimal_within_10_days"], rows)
    for ratio in [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.2]:
        ok = [r for r in rows if r[1] == ratio and r[4] == "ok"]
        print(f"  kappa/g_S = {ratio}: {len(ok)} with response, first day optimal in "
              f"{sum(r[9] for r in ok)}")

    # Worked example (kappa = 0.9 g_S, alpha = 6, f0 = 0.01%)
    p = MATCHED_BASE.with_(kappa=0.0135, alpha=6.0, f0=1e-4)
    days = [60, 200, 500]
    g = gain_by_start_day(p, days, TAU, MATCHED_THRESHOLD)
    write_csv("drug_effect_example.csv", ["response_day", "start_day", "delta_T_days",
                                          "alpha_gR_r0_over_nu"],
              [[f"{response_day(p):.2f}", d, f"{x:.2f}", f"{p.alpha * p.g_R * p.b0 * p.f0 / p.nu:.1f}"]
               for d, x in zip(days, g)])
