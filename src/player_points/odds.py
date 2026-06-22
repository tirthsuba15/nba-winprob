"""
Over/under probabilities from a points projection — INFORMATIONAL ONLY.

We approximate the predicted points distribution as Normal with:
    mean  = the model's mean prediction
    sigma = (q90 - q10) / 2.563    (an 80% interval spans ~2.563 std devs:
            z_0.90 - z_0.10 = 1.2816 - (-1.2816) = 2.5631)

This is a deliberately simple, transparent approximation — points are actually
right-skewed and bounded at 0 — so treat the numbers as a rough guide, NOT
betting advice.
"""
from __future__ import annotations
import math

# z_0.90 - z_0.10 — the width of an 80% interval in standard deviations.
SIGMA_DIVISOR = 2.563


def normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sigma_from_interval(lo: float, hi: float) -> float:
    """Recover sigma from an 80% prediction interval. Floored to avoid /0."""
    return max((hi - lo) / SIGMA_DIVISOR, 1e-6)


def prob_over(mean: float, lo: float, hi: float, line: float) -> float:
    """P(points > line) under the Normal approximation. Returns a value in [0,1]."""
    sigma = sigma_from_interval(lo, hi)
    z = (line - mean) / sigma
    return 1.0 - normal_cdf(z)


def round_to_half(x: float) -> float:
    """Round to the nearest 0.5 (typical sportsbook line granularity)."""
    return round(x * 2) / 2
