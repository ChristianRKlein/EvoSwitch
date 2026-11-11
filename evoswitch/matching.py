"""Response matching and the resistant fraction compatible with a response.

Response matching: for given parameters, find the drug effect kappa such that
the burden at day `day` equals (1 - shrinkage) x b0. This allows tumors with
identical early response but different composition to be compared.

Maximum compatible fraction: the largest pre-existing resistant fraction f0
for which some drug effect yields a given shrinkage (Supplementary Note 4).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .model import Params, flow, initial_state

RESPONSE_DAY = 56.0


def burden_at(p: Params, day: float) -> float:
    """Total burden after `day` days of continuous therapy."""
    return flow(initial_state(p), day, p, 1).sum()


def match_drug_effect(p: Params, shrinkage: float, day: float = RESPONSE_DAY,
                      bracket=(5e-4, 50.0)) -> float:
    """Drug effect kappa that yields exactly `shrinkage` at `day` (Brent's method)."""
    def residual(kappa):
        return burden_at(p.with_(kappa=kappa), day) / p.b0 - (1.0 - shrinkage)
    return brentq(residual, *bracket, xtol=1e-14)


def matched(p: Params, shrinkage: float, day: float = RESPONSE_DAY) -> Params:
    """Copy of p with kappa matched to the given shrinkage."""
    return p.with_(kappa=match_drug_effect(p, shrinkage, day))


def max_compatible_fraction_formula(shrinkage: float, day: float, g_R: float, b0: float) -> float:
    """Eq. (S5): exact bound without competition (alpha = 0, kappa -> infinity).

    Resistant cells grow logistically, L' = g_R L (1 - L); a tumor cannot shrink
    below its resistant population, so L(day) <= (1 - shrinkage) b0.
    """
    r_end = (1.0 - shrinkage) * b0
    e = np.exp(-g_R * day)
    return r_end * e / (1.0 - r_end * (1.0 - e)) / b0


def max_compatible_fraction(p: Params, shrinkage: float, day: float = RESPONSE_DAY,
                            kappa_grid=None) -> float:
    """Numerical maximum of f0 compatible with `shrinkage` at `day`.

    For each f0, the burden at `day` is minimized over a logarithmic grid of
    drug effects; the largest f0 for which this minimum reaches the target is
    found by bisection on a logarithmic scale.
    """
    if kappa_grid is None:
        kappa_grid = np.geomspace(0.01, 20.0, 40)

    def min_burden(f0):
        return min(burden_at(p.with_(f0=f0, kappa=k), day) for k in kappa_grid) / p.b0

    lo, hi = 1e-7, 0.999
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if min_burden(mid) <= 1.0 - shrinkage:
            lo = mid
        else:
            hi = mid
    return lo
