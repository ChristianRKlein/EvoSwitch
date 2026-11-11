"""Fast checks that key numbers of the manuscript are reproduced (run with `pytest`)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoswitch import (MATCHED_BASE, TIMING_EXAMPLE, flow, initial_state, interruption_gain,
                       max_compatible_fraction_formula, response_day, tumor_A, tumor_B)
from evoswitch.nadir import RECIST_VOLUME_Q, fixed_interruption_nadir, restart_after_fraction, restart_at_marker
from evoswitch.sensitivity import start_time_sensitivity, start_time_sensitivity_fd


def test_timing_example():
    assert response_day(TIMING_EXAMPLE) == pytest.approx(37.22, abs=0.01)
    assert interruption_gain(TIMING_EXAMPLE, 38, 28, 0.27)[0] == pytest.approx(78.97, abs=0.01)
    assert response_day(TIMING_EXAMPLE, 0.7 ** 3) == pytest.approx(125.36, abs=0.01)
    assert interruption_gain(TIMING_EXAMPLE, 126, 28, 0.27)[0] == pytest.approx(46.11, abs=0.01)


def test_tumors_A_and_B():
    a, b = tumor_A(), tumor_B()
    assert a.kappa == pytest.approx(0.0171898197, abs=1e-9)
    assert b.kappa == pytest.approx(0.0173705230, abs=1e-9)
    assert interruption_gain(a, 56, 28, 0.70)[0] == pytest.approx(-3.62, abs=0.01)
    assert interruption_gain(b, 56, 28, 0.70)[0] == pytest.approx(70.08, abs=0.01)


def test_fixed_drug_effect():
    base = MATCHED_BASE.with_(alpha=0.1, kappa=0.018)
    assert interruption_gain(base.with_(f0=1e-4), 56, 28, 0.70)[0] == pytest.approx(1.82, abs=0.01)
    assert interruption_gain(base.with_(f0=0.10), 56, 28, 0.70)[0] == pytest.approx(-2.60, abs=0.01)


def test_start_time_sensitivity_matches_finite_difference():
    analytic = start_time_sensitivity(TIMING_EXAMPLE, 100, 28, 0.27)
    numeric = start_time_sensitivity_fd(TIMING_EXAMPLE, 100, 28, 0.27)
    assert analytic == pytest.approx(-0.3358, abs=1e-4)
    assert abs(analytic - numeric) < 1e-5


def test_slowly_depleting_drug_favors_later_start():
    """Counterexample of Supplementary Note 3 (kappa = 0.9 g_S)."""
    p = MATCHED_BASE.with_(kappa=0.0135, alpha=6.0, f0=1e-4)
    gains = [interruption_gain(p, d, 28, 0.70)[0] for d in (60, 200, 500)]
    assert gains == pytest.approx([123.91, 173.49, 273.63], abs=0.01)


def test_max_compatible_fraction():
    assert max_compatible_fraction_formula(0.99, 56, 0.012, 0.6) == pytest.approx(0.00512, abs=1e-5)


def test_nadir_relative_rules():
    b = tumor_B()
    assert restart_after_fraction(b, 56)[0] == pytest.approx(39.29, abs=0.01)
    assert restart_at_marker(b, 56)[0] == pytest.approx(24.87, abs=0.01)
    gain, crossed, _ = fixed_interruption_nadir(b, 56, 28, RECIST_VOLUME_Q)
    assert not crossed and gain == pytest.approx(65.3, abs=0.1)


def test_solutions_stay_non_negative_and_bounded():
    p = TIMING_EXAMPLE
    for u in (0, 1):
        x = flow(initial_state(p), 1000, p, u)
        assert np.all(x >= -1e-12) and x.sum() <= 1 + 1e-9
