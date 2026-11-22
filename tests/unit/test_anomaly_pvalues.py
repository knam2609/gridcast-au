"""Conformal window p-values and tiers (m4_plan.md §2.2), against brute force."""

import numpy as np
import pytest

from gridcast.anomalies.pvalues import (
    NONE,
    NOTABLE,
    SEVERE,
    Queries,
    conformal_p,
    static_pvalues,
    tiers,
    window_pvalues,
)


def _brute(scores: np.ndarray, valid: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n, h = scores.shape
    count = np.full((n, h), -1)
    size = np.full((n, h), -1)
    for i in range(n):
        lo = max(0, i - window)
        for k in range(h):
            if not valid[i, k]:
                continue
            w = scores[lo:i, k][valid[lo:i, k]]
            count[i, k] = int((w >= scores[i, k]).sum())
            size[i, k] = len(w)
    return count, size


def _data(seed: int, n: int = 300, h: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = np.round(rng.normal(size=(n, h)), 1)  # many ties
    valid = rng.random((n, h)) > 0.15
    scores[~valid] = np.nan
    return scores, valid


@pytest.mark.parametrize("window", [1, 7, 50, 1000])
def test_window_counts_match_brute_force(window: int) -> None:
    scores, valid = _data(window)
    wp = window_pvalues(scores, valid, window)
    count, size = _brute(scores, valid, window)
    np.testing.assert_array_equal(wp.count, count)
    np.testing.assert_array_equal(wp.n_window, size)
    with np.errstate(invalid="ignore"):
        expected = np.where(valid, (1 + count) / (1 + size), np.nan)
    np.testing.assert_array_equal(wp.p, expected)


def test_extra_queries_use_the_same_windows() -> None:
    scores, valid = _data(5)
    rng = np.random.default_rng(9)
    rows = rng.integers(0, len(scores), 400)
    cols = rng.integers(0, scores.shape[1], 400)
    values = np.round(rng.normal(size=400), 1)
    wp = window_pvalues(scores, valid, 40, Queries(rows, cols, values))
    for r, c, v, got_c, got_n in zip(
        rows, cols, values, wp.extra_count, wp.extra_n_window, strict=True
    ):
        lo = max(0, r - 40)
        w = scores[lo:r, c][valid[lo:r, c]]
        assert got_c == (w >= v).sum()
        assert got_n == len(w)
    # Extra queries do not change the base p-values.
    np.testing.assert_array_equal(wp.p, window_pvalues(scores, valid, 40).p)


def test_the_current_score_is_not_in_its_own_window() -> None:
    scores = np.array([[1.0], [2.0], [3.0]])
    wp = window_pvalues(scores, np.ones_like(scores, dtype=bool), 10)
    np.testing.assert_array_equal(wp.n_window[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(wp.p[:, 0], [1.0, 1 / 2, 1 / 3])


def test_ties_count_against_the_alert() -> None:
    scores = np.array([[5.0]] * 4)
    wp = window_pvalues(scores, np.ones_like(scores, dtype=bool), 10)
    np.testing.assert_array_equal(wp.p[:, 0], [1.0, 1.0, 1.0, 1.0])


def test_p_values_are_calibrated_for_exchangeable_scores() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(60_000, 1))
    p = window_pvalues(scores, np.ones_like(scores, dtype=bool), 5_000).p[5_000:, 0]
    for alpha in (0.01, 0.001):
        assert alpha * 0.7 < np.mean(p <= alpha) < alpha * 1.3


def test_static_pvalues() -> None:
    ref = np.array([1.0, 2.0, 2.0, 3.0])
    p = static_pvalues(ref, np.array([0.5, 2.0, 2.5, 4.0, np.nan]))
    np.testing.assert_array_equal(p[:4], [5 / 5, 4 / 5, 2 / 5, 1 / 5])
    assert np.isnan(p[4])


def test_conformal_p_missing() -> None:
    p = conformal_p(np.array([0, -1]), np.array([9, -1]))
    assert p[0] == 0.1
    assert np.isnan(p[1])


def test_tiers_boundaries() -> None:
    p = np.array([np.nan, 0.5, 0.001, 0.0010001, 0.0001, 0.00005, 52 / 52_000])
    got = tiers(p, 0.001, 0.0001)
    np.testing.assert_array_equal(got, [NONE, NONE, NOTABLE, NONE, SEVERE, SEVERE, NOTABLE])


def test_queries_validate() -> None:
    with pytest.raises(ValueError, match="one length"):
        Queries(np.array([1]), np.array([0, 1]), np.array([0.0]))
    with pytest.raises(ValueError, match="finite"):
        Queries(np.array([1]), np.array([0]), np.array([np.nan]))
    scores, valid = _data(1, n=10)
    with pytest.raises(ValueError, match="outside"):
        window_pvalues(scores, valid, 5, Queries(np.array([10]), np.array([0]), np.array([0.0])))
