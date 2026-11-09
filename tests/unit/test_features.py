from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.config import PROJECT_ROOT, load_config
from gridcast.features import catalogue
from gridcast.features.builder import FeatureBuilder
from gridcast.features.inputs import publication_delay
from gridcast.features.weather import NEVER, AlignedWeather, WeatherInputs, select_forecast, to_ns
from gridcast.forecasting.origins import ForecastSetup, SeriesContext, origin_positions
from tests.feature_fixtures import hourly_weather, weather_inputs
from tests.forecast_fixtures import make_ctx

CATALOGUE_DOC = PROJECT_ROOT / "docs" / "feature_catalogue.md"


@cache
def _ctx() -> SeriesContext:
    return make_ctx(days=12)  # y = position


@cache
def _builder(latency: int = 0) -> FeatureBuilder:
    ctx = _ctx()
    fc, obs = hourly_weather(ctx.times)
    return FeatureBuilder(
        ctx, ForecastSetup(latency_steps=latency), weather_inputs(ctx.times, fc, obs)
    )


def _origin(builder: FeatureBuilder, day: int = 8) -> int:
    ctx = builder.ctx
    return int(origin_positions(ctx, builder.setup, ctx.times[day * 288])[0])


@pytest.mark.parametrize("variant", [v.id for v in catalogue.VARIANTS])
def test_columns_follow_the_catalogue(variant: str) -> None:
    b = _builder()
    fm = b.build(variant, np.array([_origin(b)]))
    assert list(fm.X.columns) == catalogue.feature_names(variant)
    assert len(fm.X) == 288 and fm.X.index.equals(pd.RangeIndex(288))
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in fm.X.dtypes)
    assert (fm.weather_offset is not None) == (catalogue.variant(variant).weather == "forecast")


def test_mirror_variants_share_feature_names() -> None:
    assert catalogue.feature_names("C") == catalogue.feature_names("D")
    assert catalogue.feature_names("Cplus") == catalogue.feature_names("Dplus")
    assert set(catalogue.feature_names("A")) < set(catalogue.feature_names("C"))
    assert set(catalogue.feature_names("C")) < set(catalogue.feature_names("Cplus"))
    names = [f.name for g in catalogue.GROUPS for f in g.features]
    assert len(names) == len(set(names))


def test_demand_features_on_a_position_series() -> None:
    b = _builder()
    o = _origin(b)
    fm = b.build("A", np.array([o]))
    X, h = fm.X, np.arange(1, 289)
    t = o + h
    assert (X["lead_minutes"] == 5 * h).all()
    assert (X["dem_now"] == o).all()
    assert (X["dem_now_vs_yesterday"] == 288).all()
    assert (X["dem_change_1h"] == 12).all()
    assert X["dem_mean_1h"].iloc[0] == pytest.approx(o - 5.5)
    assert X["dem_std_1h"].iloc[0] == pytest.approx(np.arange(12).std())
    assert X["dem_mean_24h"].iloc[0] == pytest.approx(o - 143.5)
    np.testing.assert_array_equal(X["dem_same_time_yesterday"], t - 288)
    np.testing.assert_array_equal(X["dem_same_time_last_week"], t - 2016)
    assert (X["dem_yesterday_vs_last_week"] == 2016 - 288).all()
    np.testing.assert_array_equal(fm.actual, t)


def test_latency_moves_the_cutoff_and_the_seasonal_lag() -> None:
    b = _builder(latency=7)
    o = _origin(b)
    X = b.build("A", np.array([o])).X
    assert (X["dem_now"] == o - 7).all()
    t = o + np.arange(1, 289)
    # h + 7 > 288 for h >= 282: yesterday is unknown at the cutoff, so two days back.
    expected = np.where(np.arange(1, 289) + 7 > 288, t - 576, t - 288)
    np.testing.assert_array_equal(X["dem_same_time_yesterday"], expected)


def test_calendar_features_are_those_of_the_target_and_of_yesterday() -> None:
    b = _builder()
    o = _origin(b)
    X = b.build("A", np.array([o])).X
    t = o + np.arange(1, 289)
    frame = b.ctx.frame
    np.testing.assert_array_equal(X["cal_slot"], frame["slot_of_day"].to_numpy()[t])
    np.testing.assert_array_equal(X["cal_day_of_week"], frame["day_of_week"].to_numpy()[t])
    np.testing.assert_array_equal(X["cal_tod_cos"], frame["tod_cos"].to_numpy()[t])
    np.testing.assert_array_equal(X["cal_non_working"], b.ctx.day_type[t])
    np.testing.assert_array_equal(X["cal_yesterday_non_working"], b.ctx.day_type[t - 288])


def test_origins_without_a_week_of_history_are_rejected() -> None:
    b = _builder()
    with pytest.raises(IndexError, match="history"):
        b.build("A", np.array([6 * 288]))


def test_horizon_subsets_and_row_order() -> None:
    b = _builder()
    o = _origin(b)
    fm = b.build("C", np.array([o, o + 6]), horizons=np.array([3, 6, 288]))
    np.testing.assert_array_equal(fm.origins, [o, o, o, o + 6, o + 6, o + 6])
    np.testing.assert_array_equal(fm.horizons, [3, 6, 288, 3, 6, 288])
    full = b.build("C", np.array([o + 6])).X.iloc[[2, 5, 287]].reset_index(drop=True)
    pd.testing.assert_frame_equal(fm.X.iloc[3:].reset_index(drop=True), full)
    with pytest.raises(ValueError, match="horizons"):
        b.build("A", np.array([o]), horizons=np.array([0]))


# ------------------------------------------------------------------------ weather selection


def _manual_inputs(n: int, av1: np.ndarray, v1: np.ndarray, av2: np.ndarray) -> WeatherInputs:
    def aligned(av: np.ndarray, v: np.ndarray) -> AlignedWeather:
        return AlignedWeather(
            series={"temp_weighted": v, "temp_single": v.copy()},
            available_at={"temp_weighted": av, "temp_single": av.copy()},
        )

    times = pd.date_range("2025-01-01 00:05", periods=n, freq="5min", tz="UTC")
    return WeatherInputs(
        times_ns=to_ns(times),
        forecasts={1: aligned(av1, v1), 2: aligned(av2, np.full(n, 2.0))},
        observed=None,
    )


def test_freshest_available_offset_wins_with_fallback() -> None:
    n = 10
    inp = _manual_inputs(
        n,
        av1=np.where(np.arange(n) < 6, 0, NEVER),  # day1 published for targets 0..5 only
        v1=np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        av2=np.where(np.arange(n) < 9, 0, NEVER),  # day2 for targets 0..8
    )
    origin = np.zeros(n, dtype=np.int64)  # time of position 0 > 0 ns: everything with av=0 known
    values, used = select_forecast(inp, ["temp_weighted", "temp_single"], origin, np.arange(n))
    np.testing.assert_array_equal(used, [1, 2, 1, 1, 1, 1, 2, 2, 2, 0])
    np.testing.assert_array_equal(values["temp_weighted"][:9], [1, 2, 1, 1, 1, 1, 2, 2, 2])
    assert np.isnan(values["temp_weighted"][9])


def test_value_published_after_the_origin_is_never_used() -> None:
    n = 4
    times_ns = to_ns(pd.date_range("2025-01-01 00:05", periods=n, freq="5min", tz="UTC"))
    inp = _manual_inputs(
        n,
        av1=times_ns + 1,
        v1=np.ones(n),
        av2=np.full(n, times_ns[0]),  # day1: 1 ns too late
    )
    _, used = select_forecast(inp, ["temp_weighted"], np.arange(n), np.arange(n))
    np.testing.assert_array_equal(used, [2, 2, 2, 2])
    inp2 = _manual_inputs(n, av1=times_ns, v1=np.ones(n), av2=times_ns)  # exactly at o: usable
    _, used = select_forecast(inp2, ["temp_weighted"], np.arange(n), np.arange(n))
    np.testing.assert_array_equal(used, [1, 1, 1, 1])


def test_offset_choice_matches_the_run_timing_rule() -> None:
    ctx = _ctx()
    fc, obs = hourly_weather(ctx.times, missing_values=0.0, missing_rows=0.0)
    b = FeatureBuilder(ctx, ForecastSetup(), weather_inputs(ctx.times, fc, obs))
    o = _origin(b)
    used = b.build("C", np.array([o])).weather_offset
    assert used is not None
    # A 5-minute value at t uses hours floor(t) and ceil(t); the later one decides. day1 for
    # hour T comes from the run started floor_6h(T − 24 h), published 6 h later.
    t = pd.Series(ctx.times[o + np.arange(1, 289)])
    day1_at = (t.dt.ceil("1h") - pd.Timedelta(hours=24)).dt.floor("6h") + pd.Timedelta(hours=6)
    expected = np.where(day1_at <= ctx.times[o], 1, 2)  # day2 is always published by o
    np.testing.assert_array_equal(used, expected)
    assert (used[: 17 * 12] == 1).all()  # leads up to 17 h always have day1


def test_weather_features_on_an_archive_with_gaps() -> None:
    b = _builder()
    o = _origin(b)
    c = b.build("C", np.array([o]))
    assert c.weather_offset is not None
    x = c.X
    np.testing.assert_array_equal(x["wx_temp"].isna(), c.weather_offset == 0)
    np.testing.assert_array_equal(x["wx_temp_single_site"].isna(), c.weather_offset == 0)
    assert (c.weather_offset == 2).sum() > 0 and (c.weather_offset == 1).sum() > 100
    complete = x["wx_temp"].notna()
    np.testing.assert_allclose(
        x.loc[complete, "wx_cooling_degrees"], np.maximum(x.loc[complete, "wx_temp"] - 18, 0)
    )
    np.testing.assert_allclose(
        x.loc[complete, "wx_heating_degrees"], np.maximum(18 - x.loc[complete, "wx_temp"], 0)
    )
    np.testing.assert_allclose(x["wx_temp_sq"], x["wx_temp"] ** 2)
    assert not c.complete()[~complete].any()


def test_observed_weather_ignores_the_origin() -> None:
    b = _builder()
    o = _origin(b)
    early = b.build("Dplus", np.array([o]), horizons=np.array([288]))
    late = b.build("Dplus", np.array([o + 287]), horizons=np.array([1]))
    assert early.targets[0] == late.targets[0]
    weather = [c for c in early.X.columns if c.startswith("wx_")]
    pd.testing.assert_frame_equal(early.X[weather], late.X[weather])


def test_delay_override_recomputes_availability() -> None:
    ctx = _ctx()
    fc, obs = hourly_weather(ctx.times, missing_values=0.0, missing_rows=0.0)
    six = weather_inputs(ctx.times, fc, obs)
    eight = weather_inputs(ctx.times, fc, obs, publication_delay=pd.Timedelta(hours=8))
    a6, a8 = six.forecasts[1].available_at, eight.forecasts[1].available_at
    assert a6 is not None and a8 is not None
    known = a6["temp_weighted"] != NEVER
    diff = a8["temp_weighted"][known] - a6["temp_weighted"][known]
    assert (diff == pd.Timedelta(hours=2).value).all()
    b6 = FeatureBuilder(ctx, ForecastSetup(), six)
    b8 = FeatureBuilder(ctx, ForecastSetup(), eight)
    o = np.array([_origin(b6)])
    u6, u8 = b6.build("C", o).weather_offset, b8.build("C", o).weather_offset
    assert u6 is not None and u8 is not None
    assert (u8 == 1).sum() < (u6 == 1).sum()


def test_delay_override_cannot_be_shorter_than_configured() -> None:
    cfg = load_config()
    assert publication_delay(cfg, None) is None
    assert publication_delay(cfg, 8) == pd.Timedelta(hours=8)
    with pytest.raises(ValueError, match="leak"):
        publication_delay(cfg, cfg.weather.forecasts.publication_delay_hours - 0.5)


def test_weather_variant_without_weather_inputs_fails() -> None:
    b = FeatureBuilder(_ctx(), ForecastSetup())
    with pytest.raises(ValueError, match="weather"):
        b.build("C", np.array([_origin(b)]))


# ------------------------------------------------------------------------ catalogue document


def test_catalogue_document_is_up_to_date() -> None:
    expected = catalogue.render_catalogue(load_config())
    actual = CATALOGUE_DOC.read_text(encoding="utf-8") if CATALOGUE_DOC.exists() else ""
    assert actual == expected, "run `uv run gridcast feature-catalogue` to regenerate"


def test_catalogue_mentions_every_feature_once() -> None:
    text = catalogue.render_catalogue(load_config())
    for v in catalogue.VARIANTS:
        for name in catalogue.feature_names(v.id):
            assert text.count(f"| `{name}` |") == 1, name
    assert "{" not in text and "}" not in text  # every placeholder was filled in
