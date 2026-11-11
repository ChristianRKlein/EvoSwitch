"""Nadir-relative progression and restart rules (Supplementary Note 5, Suppl. Table 14, Suppl. Fig. 3).

Criteria: q = 1.20 x nadir (marker-based) and q = 1.2**3 = 1.728 x nadir
(volume equivalent of RECIST progression, a 20% increase in diameter).
Restart rules: after 0.8 tau_q, or once the burden reaches 1.10 x nadir.
Outputs
  results/nadir_examples.csv     timing example, tumors A and B
  results/restart_rules_map.csv  both rules in the 1,368 response-matched tumors
"""
import itertools

import numpy as np
from _common import parallel_map, write_csv

from evoswitch import MATCHED_BASE, TIMING_EXAMPLE, matched, tumor_A, tumor_B
from evoswitch.model import flow, initial_state
from evoswitch.nadir import (RECIST_VOLUME_Q, fixed_interruption_nadir, restart_after_fraction,
                             restart_at_marker, tau_q_approximation, tau_q_lower_bound,
                             time_to_criterion_off_treatment)

ALPHAS = np.linspace(0.03, 1.6, 38)
FRACTIONS = np.logspace(-6, np.log10(0.15), 36)


def restart_map(args):
    f0, alpha = args
    p = matched(MATCHED_BASE.with_(f0=f0, alpha=alpha), 0.40)
    g08, tau_q, off08 = restart_after_fraction(p, 56)
    gm, offm = restart_at_marker(p, 56)
    b1 = flow(initial_state(p), 56, p, 1).sum()
    return [f"{f0:.10g}", f"{alpha:.10g}", f"{tau_q:.4f}", f"{tau_q_approximation(b1, p.g_S):.4f}",
            f"{tau_q_lower_bound(b1, p):.4f}", f"{off08:.4f}", f"{g08:.4f}", f"{offm:.4f}", f"{gm:.4f}"]


if __name__ == "__main__":
    print("08 nadir-relative criteria")
    cases = [("timing example, day 38", TIMING_EXAMPLE, 38), ("timing example, day 126", TIMING_EXAMPLE, 126),
             ("tumor A", tumor_A(), 56), ("tumor B", tumor_B(), 56)]
    rows = []
    for name, p, start in cases:
        b1 = flow(initial_state(p), start, p, 1).sum()
        _, crossed12, t_cross = fixed_interruption_nadir(p, start, 28, 1.2)
        g08, tau_q, off08 = restart_after_fraction(p, start)
        gm, offm = restart_at_marker(p, start)
        g173, crossed173, _ = fixed_interruption_nadir(p, start, 28, RECIST_VOLUME_Q)
        rows.append([name, f"{tau_q:.2f}", f"{tau_q_approximation(b1, p.g_S):.2f}",
                     f"{t_cross:.1f}" if crossed12 else "not crossed", f"{off08:.2f}", f"{g08:.2f}",
                     f"{offm:.2f}", f"{gm:.2f}",
                     f"{time_to_criterion_off_treatment(p, start, RECIST_VOLUME_Q):.1f}",
                     "crossed" if crossed173 else f"{g173:.2f}"])
    write_csv("nadir_examples.csv",
              ["case", "tau_q_1.20", "tau_q_approximation", "fixed_28d_crosses_1.20_after_days",
               "off_days_0.8tau_q", "delta_T_0.8tau_q", "off_days_marker_1.10", "delta_T_marker_1.10",
               "tau_q_recist_volume", "delta_T_fixed_28d_recist_volume"], rows)

    rows = parallel_map(restart_map, list(itertools.product(FRACTIONS, ALPHAS)))
    write_csv("restart_rules_map.csv", ["f0", "alpha", "tau_q", "tau_q_approximation", "tau_q_lower_bound",
                                        "off_days_0.8tau_q", "delta_T_0.8tau_q", "off_days_marker",
                                        "delta_T_marker"], rows)
    a = np.array([[float(x) for x in r[2:]] for r in rows])
    print(f"  tau_q {a[:, 0].min():.1f}-{a[:, 0].max():.1f} days; approximation error "
          f"max {100 * np.max(np.abs(a[:, 1] - a[:, 0]) / a[:, 0]):.1f}%; bound held: {np.all(a[:, 2] <= a[:, 0])}")
    print(f"  0.8 tau_q: off {a[:, 3].min():.1f}-{a[:, 3].max():.1f} d, gain {a[:, 4].min():.1f} to {a[:, 4].max():.1f}")
    print(f"  1.10 x nadir: off {a[:, 5].min():.1f}-{a[:, 5].max():.1f} d, gain {a[:, 6].min():.1f} to {a[:, 6].max():.1f}")
