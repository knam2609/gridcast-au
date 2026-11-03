"""Leakage property tests for the M2 feature builder (m2_plan.md §2.5).

For random issue times, the features of every pair (o, h) must be bit-for-bit identical
after anything unknown at *o* is perturbed:

* demand after the data cutoff ``o − latency``: every variant, including the oracles;
* forecast rows with ``available_at > o``, replaced by noise or deleted: C and C⁺.

Controls show the tests have teeth: perturbing *published* forecasts changes C, and
perturbing future ERA5 changes D (the documented perfect-foresight oracle).
"""

from functools import cache

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from gridcast.config import WEATHER_VARIABLES
from gridcast.features import catalogue
from gridcast.features.builder import FeatureBuilder, FeatureMatrix
from gridcast.forecasting.origins import ForecastSetup, SeriesContext, origin_positions
from tests.feature_fixtures import hourly_weather, weather_inputs
from tests.forecast_fixtures import make_ctx

ALL_VARIANTS = [v.id for v in catalogue.VARIANTS]
FORECAST_VARIANTS = [v.id for v in catalogue.VARIANTS if v.weather == "forecast"]


@cache
def _data() -> tuple[SeriesContext, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    ctx = make_ctx(days=12, values=lambda pos: rng.normal(5000, 500, len(pos)))
    fc, obs = hourly_weather(ctx.times, seed=3)
    return ctx, fc, obs


def _builder(
    ctx: SeriesContext, fc: pd.DataFrame, obs: pd.DataFrame, latency: int = 0
) -> FeatureBuilder:
    return FeatureBuilder(
        ctx, ForecastSetup(latency_steps=latency), weather_inputs(ctx.times, fc, obs)
    )


def _origin(ctx: SeriesContext, setup: ForecastSetup, rank: int) -> int:
    candidates = origin_positions(ctx, setup, ctx.times[7 * 288 + 1])
    return int(candidates[rank % len(candidates)])


def _assert_same(a: FeatureMatrix, b: FeatureMatrix) -> None:
    pd.testing.assert_frame_equal(a.X, b.X, check_exact=True, obj=f"variant {a.variant}")
    if a.weather_offset is not None:
        assert b.weather_offset is not None
        np.testing.assert_array_equal(a.weather_offset, b.weather_offset)


def _noise(frame: pd.DataFrame, rows: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for var in WEATHER_VARIABLES:
        out.loc[rows, var] = rng.normal(-500.0, 500.0, int(rows.sum()))
    return out


@settings(max_examples=40, deadline=None)
@given(
    rank=st.integers(min_value=0, max_value=10_000),
    latency=st.sampled_from([0, 1, 7]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_features_do_not_depend_on_future_demand(rank: int, latency: int, seed: int) -> None:
    ctx, fc, obs = _data()
    before = _builder(ctx, fc, obs, latency)
    origin = _origin(ctx, before.setup, rank)
    perturbed = ctx.y.copy()
    unknown = np.arange(len(ctx)) > origin - latency
    perturbed[unknown] = np.random.default_rng(seed).normal(-1e5, 1e5, unknown.sum())
    after = _builder(ctx.with_y(perturbed), fc, obs, latency)
    for variant in ALL_VARIANTS:  # D and D⁺ are oracles for weather only, not for demand
        _assert_same(
            before.build(variant, np.array([origin])), after.build(variant, np.array([origin]))
        )


@settings(max_examples=40, deadline=None)
@given(
    rank=st.integers(min_value=0, max_value=10_000),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    delete=st.booleans(),
)
def test_features_do_not_depend_on_unpublished_forecasts(
    rank: int, seed: int, delete: bool
) -> None:
    ctx, fc, obs = _data()
    before = _builder(ctx, fc, obs)
    origin = _origin(ctx, before.setup, rank)
    unpublished = (fc["available_at"] > ctx.times[origin]).to_numpy()
    assert unpublished.any()
    changed = fc[~unpublished] if delete else _noise(fc, unpublished, seed)
    after = _builder(ctx, changed, obs)
    for variant in FORECAST_VARIANTS:
        _assert_same(
            before.build(variant, np.array([origin])), after.build(variant, np.array([origin]))
        )


def test_unpublished_rows_include_the_later_bracketing_hour() -> None:
    """For some target, the earlier hour of day1 is published at o but the later one is not.

    Interpolating with the published hour alone would be a leak-free-looking mistake; the
    builder must fall back to day 2 instead, and the property test above perturbs exactly
    these later hours.
    """
    ctx, fc, obs = _data()
    b = _builder(ctx, fc, obs)
    origin = _origin(ctx, b.setup, 0)
    o_time = ctx.times[origin]
    day1 = fc[fc["offset_days"] == 1].set_index(["site_id", "valid_time"])["available_at"]
    t = pd.Series(ctx.times[origin + np.arange(1, 289)])
    lo = day1.reindex(pd.MultiIndex.from_arrays([["melbourne"] * len(t), t.dt.floor("1h")]))
    hi = day1.reindex(pd.MultiIndex.from_arrays([["melbourne"] * len(t), t.dt.ceil("1h")]))
    split = (lo.to_numpy() <= o_time) & (hi.to_numpy() > o_time)
    assert split.any()
    used = b.build("C", np.array([origin])).weather_offset
    assert used is not None
    assert (used[split] != 1).all()


def test_controls_show_the_property_tests_can_fail() -> None:
    ctx, fc, obs = _data()
    before = _builder(ctx, fc, obs)
    origin = _origin(ctx, before.setup, 0)
    o = np.array([origin])

    published = (fc["available_at"] <= ctx.times[origin]).to_numpy()
    after = _builder(ctx, _noise(fc, published, 1), obs)
    assert not before.build("C", o).X.equals(after.build("C", o).X)

    future = (obs["valid_time"] > ctx.times[origin]).to_numpy()
    after = _builder(ctx, fc, _noise(obs, future, 2))
    assert not before.build("D", o).X.equals(after.build("D", o).X)
    # The same perturbation leaves the forecast variants alone.
    _assert_same(before.build("C", o), after.build("C", o))
