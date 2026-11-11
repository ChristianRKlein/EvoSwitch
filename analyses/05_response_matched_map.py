"""1,368 response-matched virtual tumors (Fig. 3c; Supplementary Note 2).

All tumors shrink by exactly 40% at day 56; they differ in the pre-existing
resistant fraction f0 (36 values, 1e-6 to 0.15) and the competition coefficient
alpha (38 values, 0.03 to 1.6). The same 4-week interruption from day 56 is
applied to each.
Output
  results/response_matched_map.csv
"""
import itertools

import numpy as np
from _common import parallel_map, write_csv

from evoswitch import MATCHED_BASE, MATCHED_THRESHOLD as BCRIT, matched
from evoswitch.endpoints import interruption_gain

ALPHAS = np.linspace(0.03, 1.6, 38)
FRACTIONS = np.logspace(-6, np.log10(0.15), 36)


def evaluate(args):
    f0, alpha = args
    p = matched(MATCHED_BASE.with_(f0=f0, alpha=alpha), 0.40)
    g, crossed = interruption_gain(p, 56, 28, BCRIT)
    return [f"{f0:.10g}", f"{alpha:.10g}", f"{p.kappa:.12f}", f"{g:.6f}", crossed]


if __name__ == "__main__":
    print("05 response-matched map")
    rows = parallel_map(evaluate, list(itertools.product(FRACTIONS, ALPHAS)))
    write_csv("response_matched_map.csv", ["f0", "alpha", "kappa", "delta_T_days", "crossed"], rows)
    g = np.array([float(r[3]) for r in rows])
    harmful = [r for r in rows if float(r[3]) <= 0]
    print(f"  beneficial: {(g > 0).sum()}, harmful: {(g <= 0).sum()}, range {g.min():.1f} to {g.max():.1f}, "
          f"max alpha among harmful: {max(float(r[1]) for r in harmful):.3f}, "
          f"crossed during interruption: {sum(r[4] for r in rows)}")
