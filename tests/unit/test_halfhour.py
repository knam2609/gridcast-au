"""Half-hour intervals (M3 step 5): periods, mapped bands, and the run against the reference."""

from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.config import load_config
from gridcast.intervals.conformal import CandidateGrid, run_conformal, tod_labels
from gridcast.intervals.halfhour import (
    PERIOD_STEPS,
    build_inputs,
    period_bands,
    period_mask,
    period_means,
)
from gridcast.intervals.quantile import quantile_key, quantile_levels
from tests.interval_fixtures import (
    MEL,
    make_point_problem,
    make_quantile_problem,
    period_quantiles,
    reference_conformal,
    synthetic_point,
    synthetic_quantiles,
)

GRID = CandidateGrid((0.25, 0.5), ("lead", "lead_tod"), (0.0, 0.05))
BANDS30 = [(1, 2), (3, 6)]
LEVELS = (50, 80, 95)


def _retrains(times: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return [times[0] + pd.Timedelta(days=d) for d in (6, 8)]


@cache
def _small() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Ten days of issue times with 36 five-minute leads (six periods)."""
    return synthetic_point(n_days=10, n_leads=36, seed=4)


def test_period_means_and_mask() -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 12)
    np.testing.assert_array_equal(period_means(values), [[2.5, 8.5], [14.5, 20.5]])
    values[1, 7] = np.nan
    assert np.isnan(period_means(values)[1, 1]) and np.isfinite(period_means(values)[1, 0])
    mask = np.ones((2, 12), dtype=bool)
    mask[0, 5] = False
    np.testing.assert_array_equal(period_mask(mask), [[False, True], [True, True]])
    with pytest.raises(ValueError, match="whole number"):
        period_means(np.zeros((2, 10)))
    with pytest.raises(ValueError, match="whole number"):
        period_mask(np.ones((2, 10), dtype=bool))


def test_m2_lead_bands_map_to_whole_periods() -> None:
    bands = load_config().ml.lead_bands
    assert period_bands(bands) == [(1, 1), (2, 4), (5, 12), (13, 48)]
    with pytest.raises(ValueError, match="whole periods"):
        period_bands([(1, 4), (5, 12)])


def test_period_tod_block_is_that_of_the_period_start() -> None:
    times, _, _ = _small()
    five = tod_labels(times, 36, 6, MEL, (0, 8, 16))[0]
    thirty = tod_labels(times, 6, 1, MEL, (0, 8, 16), step_minutes=30)[0]
    np.testing.assert_array_equal(thirty, five[:, ::PERIOD_STEPS])


@pytest.mark.parametrize("method", ["point", "cqr"])
def test_matches_reference_on_periods(method: str) -> None:
    times, y, fc = _small()
    y30 = period_means(y)
    if method == "point":
        problem = make_point_problem(times, y30, period_means(fc), step_minutes=30)
    else:
        q30 = period_quantiles(synthetic_quantiles(fc))
        problem = make_quantile_problem(times, y30, q30, step_minutes=30)
    res = run_conformal(problem, GRID, LEVELS, _retrains(times), BANDS30, 0.25)
    lower, upper, choices = reference_conformal(
        problem, GRID, LEVELS, _retrains(times), BANDS30, 0.25
    )
    for level in LEVELS:
        np.testing.assert_array_equal(res.lower[level], lower[level].astype(np.float32))
        np.testing.assert_array_equal(res.upper[level], upper[level].astype(np.float32))
    got = sorted((s.retrain, s.level, s.band, s.chosen) for s in res.selections)
    assert got == sorted(choices)
    assert len({c for *_, c in got}) > 1  # more than one choice is exercised


def test_periods_need_every_five_minute_pair() -> None:
    """A period whose 5-minute pairs are not all scored neither calibrates nor is scored."""
    times, y, fc = _small()
    keep = np.isfinite(y) & np.isfinite(fc)
    y30 = period_means(y)
    problem = make_point_problem(times, y30, period_means(fc), step_minutes=30)
    np.testing.assert_array_equal(problem.valid, period_mask(keep))
    assert (~problem.valid).mean() > (~keep).mean()  # missing 5-minute pairs remove periods


def test_build_inputs_period_means() -> None:
    cfg = load_config()
    rng = np.random.default_rng(0)
    n, h = 96, cfg.forecast.horizon_steps
    times = pd.date_range("2024-01-01 13:00", periods=n, freq="30min", tz="UTC")
    actual = rng.normal(5000, 100, (n, h))
    mask = rng.random((n, h)) > 0.01
    keys = [quantile_key(t) for t in quantile_levels(cfg.intervals.levels)]
    quantiles = {k: actual + i for i, k in enumerate(keys)}
    inp = build_inputs(cfg, "VIC1", actual, mask, times, {"lgbm_C": actual - 1}, quantiles)
    assert inp.actual.shape == (n, h // PERIOD_STEPS)
    np.testing.assert_allclose(inp.actual, period_means(actual))
    np.testing.assert_array_equal(inp.mask, period_mask(mask))
    np.testing.assert_allclose(inp.points["lgbm_C"], period_means(actual) - 1)
    lo, hi = inp.quantiles[80]
    np.testing.assert_allclose(lo, period_means(actual) + keys.index(quantile_key(0.1)))
    np.testing.assert_allclose(hi, period_means(actual) + keys.index(quantile_key(0.9)))
    assert set(inp.groups) == {"lead", "lead_tod"}
    one = inp.only(model="lgbm_C")
    assert set(one.points) == {"lgbm_C"} and not one.quantiles
