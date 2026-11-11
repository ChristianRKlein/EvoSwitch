"""EvoSwitch: timing and patient selection for strategic interruptions of targeted cancer therapy.

Code accompanying Klein & Brossart, "Optimal Initiation and Clonal Competition in
Targeted Cancer Therapy Interruptions".
"""
from .model import Params, rhs, initial_state, flow, trajectory, set_solver, SOLVER
from .endpoints import (response_day, progression_continuous, progression_with_interruption,
                        interruption_gain, gain_by_start_day)
from .matching import (match_drug_effect, matched, max_compatible_fraction,
                       max_compatible_fraction_formula, burden_at)
from .parameters import (TIMING_EXAMPLE, TIMING_THRESHOLD, MATCHED_BASE, MATCHED_THRESHOLD,
                         MATCHED_START_DAY, INTERRUPTION_DAYS, RESPONSE_30, RESPONSE_RECIST_PR,
                         tumor_A, tumor_B)

__version__ = "1.0.0"
