"""Parameter settings used in the manuscript (Table 3).

Growth rates and competition strength are orders of magnitude taken from a
two-population model of abiraterone-treated prostate cancer (Zhang et al.,
eLife 2022); they are not a calibration to a specific cancer. All durations
are model days.
"""
from __future__ import annotations

from .matching import matched
from .model import Params

NU = 2.5e-8                 # conversion to resistance (1/day)
INTERRUPTION_DAYS = 28.0    # length of the interruption
RESPONSE_30 = 0.70          # burden remaining after a 30% reduction
RESPONSE_RECIST_PR = 0.7 ** 3  # volume equivalent of a RECIST partial response (65.7% reduction)

# Timing example (Fig. 2): strong competition, 1% pre-existing resistant cells.
TIMING_EXAMPLE = Params(g_S=0.0156, g_R=0.0091, kappa=0.022, alpha=6.0, nu=NU, b0=0.25, f0=0.01)
TIMING_THRESHOLD = 0.27

# Response-matched setting (Figs. 3 and 4): kappa is fitted to the response.
MATCHED_BASE = Params(g_S=0.015, g_R=0.012, kappa=0.0, alpha=0.0, nu=NU, b0=0.60, f0=0.0)
MATCHED_THRESHOLD = 0.70
MATCHED_START_DAY = 56.0


def tumor_A(shrinkage: float = 0.40) -> Params:
    """Tumor A: 0.0001% pre-existing resistant cells, weak competition."""
    return matched(MATCHED_BASE.with_(f0=1e-6, alpha=0.1), shrinkage)


def tumor_B(shrinkage: float = 0.40) -> Params:
    """Tumor B: 3% pre-existing resistant cells, strong competition."""
    return matched(MATCHED_BASE.with_(f0=0.03, alpha=3.0), shrinkage)
