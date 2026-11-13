"""Rolling conformal intervals: the fast run against the slow reference, and its rules."""

from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.intervals.conformal import (
    CandidateGrid,
    ConformalResult,
    band_index,
    conformal_rank,
    known_lag,
    run_conformal,
    tod_labels,
)
from gridcast.intervals.scores import summarise
from tests.interval_fixtures import (
    MEL,
    make_point_problem,
    make_quantile_problem,
    reference_conformal,
    synthetic_point,
    synthetic_quantiles,
)

GRID = CandidateGrid((0.25, 0.5), ("lead", "lead_tod"), (0.0, 0.05))
BANDS = [(1, 4), (5, 12)]
LEVELS = (50, 80, 95)


def _retrains(times: pd.DatetimeIndex, days: tuple[int, ...] = (6, 8)) -> list[pd.Timestamp]:
    return [times[0] + pd.Timedelta(days=d) for d in days]


@cache
def _small() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    return synthetic_point(n_days=10, n_leads=12, seed=1)


def test_known_lag() -> None:
    h = np.arange(1, 13)
    np.testing.assert_array_equal(known_lag(h, 6), [1] * 6 + [2] * 6)
    np.testing.assert_array_equal(known_lag(h, 6, latency_steps=1), [1] * 5 + [2] * 6 + [3])


def test_conformal_rank() -> None:
    k, clip = conformal_rank(np.array([9, 9, 4, 0]), np.array([0.1, 0.0, 0.1, 0.1]))
    # ⌈10·0.9⌉ = 9 (no float round-up); ⌈10·1⌉ = 10 > 9 clipped; ⌈5·0.9⌉ = 5 > 4 clipped
    np.testing.assert_array_equal(k, [9, 9, 4, 1])
    np.testing.assert_array_equal(clip, [False, True, True, True])


def test_tod_labels_follow_local_time_across_dst() -> None:
    times = pd.date_range("2024-04-06 13:00", periods=96, freq="30min", tz="UTC")
    labels, n = tod_labels(times, 12, 6, MEL, [6, 18])
    assert n == 2
    hours = np.array(
        [[(t + pd.Timedelta(minutes=5 * k)).tz_convert(MEL).hour for k in range(12)] for t in times]
    )
    np.testing.assert_array_equal(labels, np.where((hours >= 6) & (hours < 18), 0, 1))


def test_band_index() -> None:
    np.testing.assert_array_equal(band_index([(1, 2), (3, 4)], 4), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        band_index([(1, 2), (4, 4)], 4)
    with pytest.raises(ValueError):
        band_index([(1, 2)], 4)


def test_matches_reference() -> None:
    times, y, point = _small()
    problem = make_point_problem(times, y, point)
    res = run_conformal(problem, GRID, LEVELS, _retrains(times), BANDS, 0.25)
    lower, upper, choices = reference_conformal(
        problem, GRID, LEVELS, _retrains(times), BANDS, 0.25
    )
    for level in LEVELS:
        np.testing.assert_array_equal(res.lower[level], lower[level].astype(np.float32))
        np.testing.assert_array_equal(res.upper[level], upper[level].astype(np.float32))
    got = sorted((s.retrain, s.level, s.band, s.chosen) for s in res.selections)
    assert got == sorted(choices)
    assert len({c for *_, c in got}) > 1  # the test exercises more than one choice


def test_output_covers_every_scored_pair() -> None:
    times, y, point = _small()
    problem = make_point_problem(times, y, point)
    res = run_conformal(problem, GRID, LEVELS, _retrains(times), BANDS, 0.25)
    i0 = len(times) - len(res.origin_times)
    assert res.origin_times[0] == _retrains(times)[0]
    scored = problem.valid[i0:]
    for level in LEVELS:
        assert np.isfinite(res.lower[level][scored]).all()
        assert (res.lower[level] <= res.upper[level])[scored].all()
        assert np.isnan(res.lower[level][np.isnan(point[i0:])]).all()


def test_ties_go_to_the_first_candidate() -> None:
    times, y, point = _small()
    # One time-of-day block: 'lead_tod' streams equal 'lead' streams exactly.
    problem = make_point_problem(times, y, point, block_starts=(0,))
    res = run_conformal(problem, GRID, LEVELS, _retrains(times), BANDS, 0.25)
    assert all("lead_tod" not in s.chosen for s in res.selections)


def _iid_result() -> tuple[ConformalResult, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    n, n_leads = 30 * 48, 12
    times = pd.date_range("2024-01-01 14:00", periods=n, freq="30min", tz="UTC")
    point = np.full((n, n_leads), 5000.0)
    y = point + rng.normal(0, 100, (n, n_leads))
    problem = make_point_problem(times, y, point)
    grid = CandidateGrid((1.0,), ("lead",), (0.0,))
    res = run_conformal(problem, grid, LEVELS, [times[0] + pd.Timedelta(days=16)], BANDS, 1.0)
    i0 = n - len(res.origin_times)
    return res, y[i0:], problem.valid[i0:]


def test_iid_coverage_is_nominal() -> None:
    res, y, mask = _iid_result()
    for level in LEVELS:
        stats, _ = summarise(res.lower[level], res.upper[level], y, mask, level)
        assert stats["coverage"] == pytest.approx(level / 100, abs=0.02)


def test_rejects_short_warm_up_and_off_grid_retrains() -> None:
    times, y, point = _small()
    problem = make_point_problem(times, y, point)
    with pytest.raises(ValueError, match="longest window"):
        run_conformal(problem, GRID, LEVELS, _retrains(times, (3,)), BANDS, 0.25)
    off = [times[0] + pd.Timedelta(days=6, minutes=10)]
    with pytest.raises(ValueError, match="issue time"):
        run_conformal(problem, GRID, LEVELS, off, BANDS, 0.25)
    gap = make_point_problem(times.delete(100), np.delete(y, 100, 0), np.delete(point, 100, 0))
    with pytest.raises(ValueError, match="regular grid"):
        run_conformal(gap, GRID, LEVELS, _retrains(times), BANDS, 0.25)


def test_cqr_matches_reference() -> None:
    times, y, point = _small()
    problem = make_quantile_problem(times, y, synthetic_quantiles(point))
    res = run_conformal(problem, GRID, LEVELS, _retrains(times), BANDS, 0.25)
    lower, upper, choices = reference_conformal(
        problem, GRID, LEVELS, _retrains(times), BANDS, 0.25
    )
    for level in LEVELS:
        np.testing.assert_array_equal(res.lower[level], lower[level].astype(np.float32))
        np.testing.assert_array_equal(res.upper[level], upper[level].astype(np.float32))
    assert sorted((s.retrain, s.level, s.band, s.chosen) for s in res.selections) == sorted(choices)
