"""Interval and quantile scores on (issue time × lead) matrices (m3_plan.md §2.4).

Conventions:

* A central interval at level ``L`` % has miscoverage ``α = 1 − L/100`` (α = 0.2 at 80%)
  and bounds ``l ≤ u``. It covers ``y`` when ``l ≤ y ≤ u``.
* **Interval score** (Gneiting & Raftery 2007): ``IS_α = (u − l) + (2/α)(l − y)⁺ +
  (2/α)(y − u)⁺``. Lower is better; it rewards narrow intervals and penalises misses in
  proportion to their size.
* **Pinball loss** of a quantile ``q`` at level ``τ``: ``(1{y < q} − τ)(q − y)``.
* **Weighted interval score** (Bracher et al. 2021) of a median ``m`` and K central
  intervals: ``(½|y − m| + Σ_k (α_k/2) IS_{α_k}) / (K + ½)``. It equals the mean pinball
  loss over the 2K + 1 quantiles, times two.

Scores are elementwise; :func:`summarise` pools them over a mask (all pairs, or per lead).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

LEVELS: tuple[int, ...] = (50, 80, 95)


def alpha(level: int) -> float:
    """Miscoverage of a central interval at ``level`` percent."""
    if not 0 < level < 100:
        raise ValueError(f"level must be in (0, 100), got {level}")
    return 1.0 - level / 100.0


def covered(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``l ≤ y ≤ u`` (False where any input is NaN)."""
    return (lower <= y) & (y <= upper)


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """Elementwise interval score at miscoverage ``a`` (NaN where any input is NaN)."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    below = np.maximum(lo - yy, 0.0)
    above = np.maximum(yy - hi, 0.0)
    return (hi - lo) + (2.0 / a) * (below + above)


def pinball(q: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise pinball (quantile) loss of ``q`` at level ``tau``."""
    qq = np.asarray(q, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return ((yy < qq).astype(np.float64) - tau) * (qq - yy)


def weighted_interval_score(
    median: np.ndarray,
    intervals: Mapping[int, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
) -> np.ndarray:
    """Elementwise WIS of ``median`` and central intervals keyed by level (percent)."""
    yy = np.asarray(y, dtype=np.float64)
    total = 0.5 * np.abs(yy - np.asarray(median, dtype=np.float64))
    for level, (lo, hi) in intervals.items():
        a = alpha(level)
        total = total + (a / 2.0) * interval_score(lo, hi, yy, a)
    return total / (len(intervals) + 0.5)


def rearrange(quantiles: np.ndarray) -> np.ndarray:
    """Sort quantile forecasts along axis 0 (levels ascending), so they never cross.

    Rearrangement (Chernozhukov et al. 2010) never increases the pinball loss summed over
    the levels. NaN sorts last, so callers must mask pairs with a missing level.
    """
    return np.sort(quantiles, axis=0)


def crossing_share(quantiles: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Share of pairs where some quantile lies below the one at the next lower level.

    ``quantiles`` has the levels on axis 0, in ascending order. Pairs with any NaN level
    (or outside ``mask``) are not counted.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    ok = np.isfinite(q).all(axis=0)
    if mask is not None:
        ok &= mask
    crossed = (np.diff(q, axis=0) < 0).any(axis=0)
    n = int(ok.sum())
    return float(crossed[ok].sum() / n) if n else float("nan")


SUMMARY_STATS: tuple[str, ...] = (
    "coverage",
    "below",
    "above",
    "width",
    "interval_score",
)


def summarise(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    level: int,
    axis: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Coverage, share below/above, mean width and mean interval score over ``mask``.

    ``axis=None`` pools every masked pair; ``axis=0`` gives one value per column (lead).
    Returns ``(values_by_stat, n)``; values are NaN where ``n == 0``. A masked pair whose
    bounds are NaN is an error: every scored pair must have an interval.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if np.any(mask & ~(np.isfinite(lo) & np.isfinite(hi) & np.isfinite(yy))):
        raise ValueError("a scored pair has a missing bound or actual")
    a = alpha(level)
    n = np.asarray(mask.sum(axis=axis))
    stats = {
        "coverage": covered(lo, hi, yy),
        "below": yy < lo,
        "above": yy > hi,
        "width": hi - lo,
        "interval_score": interval_score(lo, hi, yy, a),
    }
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, values in stats.items():
            total = np.where(mask, values, 0.0).sum(axis=axis)
            out[name] = np.asarray(total / n)
    return out, n


def mean_interval_scores(
    bounds: Mapping[int, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    mask: np.ndarray,
    levels: Sequence[int] = LEVELS,
) -> dict[int, float]:
    """Pooled mean interval score per level (a convenience for reports and tests)."""
    out = {}
    for level in levels:
        lo, hi = bounds[level]
        values, _ = summarise(lo, hi, y, mask, level)
        out[level] = float(values["interval_score"])
    return out
