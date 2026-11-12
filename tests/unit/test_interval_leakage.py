"""Leakage property tests for the rolling conformal intervals (m3_plan.md §2.5, U4).

For random cutoffs ``c``, every interval issued at or before ``c`` must be bit-for-bit
identical after anything unknown at ``c`` is perturbed:

* actual demand whose interval ends after ``c`` (new values, or missing);
* forecasts issued after ``c`` (point forecasts, or the quantile forecasts of CQR).

This covers the calibration windows, ACI and the per-retrain selection at once, since all
three feed the output. The half-hour methods (``point30``, ``cqr30``: the same recipe on
30-minute period means, M3 step 5) are perturbed at 5-minute resolution, before the
period means are taken, so a period with any interval ending after ``c`` is unknown.

Controls show the tests have teeth: an off-by-one information rule (a lead's target used
one issue time before it has ended) is caught, and perturbing data known at ``c`` does
change the intervals.
"""

from collections.abc import Callable
from functools import cache

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gridcast.intervals import conformal
from gridcast.intervals.conformal import (
    CandidateGrid,
    ConformalProblem,
    ConformalResult,
    epoch_minutes,
)
from gridcast.intervals.halfhour import period_means
from tests.interval_fixtures import (
    make_point_problem,
    make_quantile_problem,
    period_quantiles,
    synthetic_point,
    synthetic_quantiles,
)

GRID = CandidateGrid((0.25, 0.5), ("lead", "lead_tod"), (0.0, 0.05))
BANDS = [(1, 4), (5, 12)]
BANDS30 = [(1, 2), (3, 6)]  # half-hour periods of 36 five-minute leads
LEVELS = (50, 80, 95)
METHODS = ("point", "cqr", "point30", "cqr30")
Data = tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]


@cache
def _data5() -> Data:
    return synthetic_point(n_days=10, n_leads=12, seed=3)


@cache
def _data30() -> Data:
    return synthetic_point(n_days=10, n_leads=36, seed=3)


def _data(method: str) -> Data:
    return _data30() if method.endswith("30") else _data5()


def _problem(
    method: str, times: pd.DatetimeIndex, y: np.ndarray, fc: np.ndarray
) -> ConformalProblem:
    if method == "point":
        return make_point_problem(times, y, fc)
    if method == "cqr":
        return make_quantile_problem(times, y, synthetic_quantiles(fc))
    y30 = period_means(y)
    if method == "point30":
        return make_point_problem(times, y30, period_means(fc), step_minutes=30)
    q30 = period_quantiles(synthetic_quantiles(fc))
    return make_quantile_problem(times, y30, q30, step_minutes=30)


def _run(method: str, times: pd.DatetimeIndex, y: np.ndarray, fc: np.ndarray) -> ConformalResult:
    retrains = [times[0] + pd.Timedelta(days=d) for d in (6, 8)]
    bands = BANDS30 if method.endswith("30") else BANDS
    return conformal.run_conformal(
        _problem(method, times, y, fc), GRID, LEVELS, retrains, bands, 0.25
    )


@cache
def _baseline(method: str) -> ConformalResult:
    return _run(method, *_data(method))


def _cutoff(res: ConformalResult, n: int, share: float) -> int:
    i0 = n - len(res.origin_times)
    return i0 + int(share * (n - 2 - i0))


def _ends_after(times: pd.DatetimeIndex, n_leads: int, c: int) -> np.ndarray:
    """Pairs whose target interval ends after issue time ``c``."""
    minutes = epoch_minutes(times)
    ends = minutes[:, None] + 5 * np.arange(1, n_leads + 1)[None, :]
    return np.asarray(ends > minutes[c])


def _changed_upto(a: ConformalResult, b: ConformalResult, c: int, n: int) -> bool:
    upto = c - (n - len(a.origin_times)) + 1
    return any(
        not np.array_equal(x[level][:upto], y[level][:upto], equal_nan=True)
        for level in LEVELS
        for x, y in ((a.lower, b.lower), (a.upper, b.upper))
    )


def _missing_share(method: str) -> float:
    """Share of 5-minute values made missing: about half the targets either way (a period
    is missing when any of its six values is: 1 − 0.9⁶ ≈ 47%)."""
    return 0.1 if method.endswith("30") else 0.5


def _perturb(
    values: np.ndarray, where: np.ndarray, mode: str, seed: int, missing: float = 0.5
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = values.copy()
    if mode == "missing":
        out[where & (rng.random(values.shape) < missing)] = np.nan
    else:
        out[where] += rng.normal(0, 500, int(where.sum()))
    return out


@settings(max_examples=24, deadline=None)
@given(
    method=st.sampled_from(METHODS),
    share=st.floats(0.0, 1.0),
    mode=st.sampled_from(["values", "missing"]),
    seed=st.integers(0, 2**31),
)
def test_future_actuals_leave_past_intervals_unchanged(
    method: str, share: float, mode: str, seed: int
) -> None:
    times, y, fc = _data(method)
    base = _baseline(method)
    c = _cutoff(base, len(times), share)
    future = _ends_after(times, y.shape[1], c)
    after = _run(method, times, _perturb(y, future, mode, seed, _missing_share(method)), fc)
    assert not _changed_upto(base, after, c, len(times))


@settings(max_examples=24, deadline=None)
@given(
    method=st.sampled_from(METHODS),
    share=st.floats(0.0, 1.0),
    mode=st.sampled_from(["values", "missing"]),
    seed=st.integers(0, 2**31),
)
def test_later_forecasts_leave_past_intervals_unchanged(
    method: str, share: float, mode: str, seed: int
) -> None:
    times, y, fc = _data(method)
    base = _baseline(method)
    c = _cutoff(base, len(times), share)
    later = np.broadcast_to((np.arange(len(times)) > c)[:, None], y.shape)
    after = _run(method, times, y, _perturb(fc, later, mode, seed, _missing_share(method)))
    assert not _changed_upto(base, after, c, len(times))


# ------------------------------------------------------------------------------ controls


def _leaky_lag(real: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    def lag(*args: object, **kwargs: object) -> np.ndarray:
        return np.maximum(real(*args, **kwargs) - 1, 1)

    return lag


@pytest.mark.parametrize("method", METHODS)
def test_control_off_by_one_information_rule_is_caught(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    times, y, fc = _data(method)
    base = _baseline(method)
    c = _cutoff(base, len(times), 0.5)
    future = _ends_after(times, y.shape[1], c)
    monkeypatch.setattr(conformal, "known_lag", _leaky_lag(conformal.known_lag))
    clean = _run(method, times, y, fc)
    leaky = _run(method, times, _perturb(y, future, "values", 0), fc)
    assert _changed_upto(clean, leaky, c, len(times))


@pytest.mark.parametrize("method", METHODS)
def test_control_known_actuals_do_change_intervals(method: str) -> None:
    times, y, fc = _data(method)
    base = _baseline(method)
    c = _cutoff(base, len(times), 0.5)
    known = ~_ends_after(times, y.shape[1], c)
    recent = known & (np.arange(len(times)) >= c - 48)[:, None]
    after = _run(method, times, _perturb(y, recent, "values", 0), fc)
    assert _changed_upto(base, after, c, len(times))
