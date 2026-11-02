import numpy as np
import pandas as pd
import pytest

from gridcast.forecasting.origins import ForecastSetup, origin_positions, target_positions
from gridcast.models.baselines import (
    DAY_STEPS,
    WEEK_STEPS,
    MedianProfile,
    Persistence,
    SeasonalNaive,
    assert_known,
    default_baselines,
)
from tests.forecast_fixtures import make_ctx

SETUP = ForecastSetup()


@pytest.fixture(scope="module")
def ctx():  # type: ignore[no-untyped-def]
    return make_ctx()


def _origins(ctx, setup: ForecastSetup = SETUP) -> np.ndarray:  # type: ignore[no-untyped-def]
    return origin_positions(ctx, setup, ctx.times[WEEK_STEPS + 10])


def test_origins_on_30_minute_grid_with_full_horizon(ctx) -> None:  # type: ignore[no-untyped-def]
    o = _origins(ctx)
    minutes = ctx.times[o].minute
    assert set(minutes) <= {0, 30}
    assert (o + SETUP.horizon_steps < len(ctx)).all()
    assert np.all(np.diff(o) == 6)


def test_persistence(ctx) -> None:  # type: ignore[no-untyped-def]
    o = _origins(ctx)[:5]
    pred = Persistence().predict(ctx, o, SETUP)
    assert pred.shape == (5, 288)
    assert np.array_equal(pred, np.repeat(ctx.y[o][:, None], 288, axis=1))


def test_daily_and_weekly_naive(ctx) -> None:  # type: ignore[no-untyped-def]
    o = _origins(ctx)[:5]
    t = target_positions(o, SETUP)
    daily = SeasonalNaive(DAY_STEPS, "daily_naive").predict(ctx, o, SETUP)
    weekly = SeasonalNaive(WEEK_STEPS, "weekly_naive").predict(ctx, o, SETUP)
    assert np.array_equal(daily, ctx.y[t - DAY_STEPS])
    assert np.array_equal(weekly, ctx.y[t - WEEK_STEPS])


def test_seasonal_naive_steps_back_an_extra_period_under_latency(ctx) -> None:  # type: ignore[no-untyped-def]
    setup = ForecastSetup(latency_steps=3)
    o = _origins(ctx, setup)[:2]
    pred = SeasonalNaive(DAY_STEPS, "daily_naive").predict(ctx, o, setup)
    t = target_positions(o, setup)
    # h <= 285: t - 288 <= o - 3 is known; h >= 286 must fall back to t - 576.
    assert np.array_equal(pred[:, :285], ctx.y[t[:, :285] - DAY_STEPS])
    assert np.array_equal(pred[:, 285:], ctx.y[t[:, 285:] - 2 * DAY_STEPS])


def test_median_profile_separates_day_types() -> None:
    # Value = 100 on working days, 1000 on weekend/holiday days, plus a slot pattern.
    base = make_ctx(days=40)
    y = np.where(base.day_type == 1, 1000.0, 100.0) + base.slot
    ctx = base.with_y(y)
    o = _origins(ctx)[:48]
    pred = MedianProfile(lookback_days=28).predict(ctx, o, SETUP)
    t = target_positions(o, SETUP)
    expected = np.where(ctx.day_type[t] == 1, 1000.0, 100.0) + ctx.slot[t]
    assert np.allclose(pred, expected)


def test_median_profile_window_excludes_origin_day() -> None:
    """Data from the origin's own local day must not enter its profile."""
    base = make_ctx(days=40)
    o = _origins(base)[:1]
    origin_day = base.local_day[o[0]]
    y = np.where(base.local_day >= origin_day, 1e6, 50.0)  # future-ish days are huge
    pred = MedianProfile(lookback_days=28).predict(base.with_y(y), o, SETUP)
    assert np.allclose(pred, 50.0)


def test_assert_known_detects_leakage() -> None:
    origins = np.array([100, 200])
    with pytest.raises(AssertionError, match="leakage"):
        assert_known(np.array([[100, 101], [150, 160]]), origins, SETUP)
    assert_known(np.array([[100, 99], [200, 1]]), origins, SETUP)


def test_missing_history_gives_nan_not_error() -> None:
    ctx = make_ctx(days=10)
    o = origin_positions(ctx, SETUP, ctx.times[0])[:3]  # no week of history yet
    weekly = SeasonalNaive(WEEK_STEPS, "weekly_naive").predict(ctx, o, SETUP)
    assert np.isnan(weekly).all()


def test_default_baselines() -> None:
    assert [m.name for m in default_baselines("demand")] == [
        "persistence",
        "daily_naive",
        "weekly_naive",
    ]
    assert "median_profile" in [m.name for m in default_baselines("price")]
    with pytest.raises(ValueError):
        default_baselines("wind")


def test_dst_day_uses_fixed_absolute_lag() -> None:
    """Across the autumn DST change, 'yesterday' is exactly 24 h earlier in UTC."""
    ctx = make_ctx(start="2022-03-20 00:05", days=20)
    o = origin_positions(ctx, SETUP, pd.Timestamp("2022-04-02 13:00", tz="UTC"))[:1]
    pred = SeasonalNaive(DAY_STEPS, "daily_naive").predict(ctx, o, SETUP)
    t = target_positions(o, SETUP)
    assert (ctx.times[t[0]] - ctx.times[t[0] - DAY_STEPS] == pd.Timedelta(hours=24)).all()
    assert np.array_equal(pred[0], ctx.y[t[0] - DAY_STEPS])
