import numpy as np
import pytest

from gridcast.intervals.scores import (
    alpha,
    covered,
    crossing_share,
    interval_score,
    pinball,
    rearrange,
    summarise,
    weighted_interval_score,
)


def test_alpha() -> None:
    assert alpha(80) == pytest.approx(0.2)
    assert alpha(95) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        alpha(100)


def test_interval_score_by_hand() -> None:
    lo, hi = np.array([10.0, 10.0, 10.0]), np.array([20.0, 20.0, 20.0])
    y = np.array([15.0, 5.0, 23.0])
    # inside: width only; below by 5: 10 + (2/0.2)·5; above by 3: 10 + 10·3
    np.testing.assert_allclose(interval_score(lo, hi, y, 0.2), [10.0, 60.0, 40.0])
    np.testing.assert_array_equal(covered(lo, hi, y), [True, False, False])
    # bounds are inclusive
    assert covered(np.array([1.0]), np.array([2.0]), np.array([2.0]))[0]


def test_pinball_by_hand() -> None:
    # y above q: tau·(y − q); y below q: (1 − tau)·(q − y)
    assert pinball(np.array([10.0]), np.array([14.0]), 0.9)[0] == pytest.approx(0.9 * 4)
    assert pinball(np.array([10.0]), np.array([6.0]), 0.9)[0] == pytest.approx(0.1 * 4)


def test_wis_equals_mean_pinball_times_two() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    med = rng.normal(size=500) * 0.1
    intervals = {}
    quantiles = {0.5: med}
    for level in (50, 80, 95):
        a = alpha(level)
        width = rng.uniform(0.5, 3.0, size=500)
        lo, hi = med - width * (1 - a), med + width
        intervals[level] = (lo, hi)
        quantiles[a / 2] = lo
        quantiles[1 - a / 2] = hi
    wis = weighted_interval_score(med, intervals, y)
    mean_pinball = np.mean([pinball(q, y, tau) for tau, q in quantiles.items()], axis=0)
    np.testing.assert_allclose(wis, 2 * mean_pinball)


def test_crossing_and_rearrangement() -> None:
    q = np.array([[1.0, 3.0, np.nan], [2.0, 2.0, 1.0], [3.0, 1.0, 2.0]])  # levels × pairs
    assert crossing_share(q) == pytest.approx(0.5)  # pair 3 has a NaN, pair 2 crosses
    assert crossing_share(rearrange(q[:, :2])) == 0.0


def test_summarise_pooled_and_per_lead() -> None:
    lo = np.array([[0.0, 0.0], [0.0, 0.0]])
    hi = np.array([[2.0, 4.0], [2.0, 4.0]])
    y = np.array([[1.0, 5.0], [3.0, np.nan]])
    mask = np.isfinite(y)
    pooled, n = summarise(lo, hi, y, mask, 80)
    assert n == 3
    assert pooled["coverage"] == pytest.approx(1 / 3)
    assert pooled["above"] == pytest.approx(2 / 3)
    assert pooled["width"] == pytest.approx(8 / 3)
    per, n_h = summarise(lo, hi, y, mask, 80, axis=0)
    np.testing.assert_array_equal(n_h, [2, 1])
    np.testing.assert_allclose(per["coverage"], [0.5, 0.0])
    np.testing.assert_allclose(per["interval_score"], [(2 + 2 + 10) / 2, 4 + 10])


def test_summarise_rejects_missing_bounds() -> None:
    with pytest.raises(ValueError, match="missing bound"):
        summarise(np.array([np.nan]), np.array([1.0]), np.array([0.5]), np.array([True]), 80)
