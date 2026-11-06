import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from gridcast.config import AppConfig, load_config
from gridcast.database.queries import load_weather_observations
from gridcast.features import catalogue
from gridcast.features.audit import render_audit, run_audit
from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.features.inputs import load_feature_builder
from gridcast.forecasting.origins import nem_midnight_utc, origin_positions
from gridcast.weather.align import site_weighted, to_five_minute

pytestmark = pytest.mark.integration


@pytest.fixture
def ready(loaded: Engine) -> Engine:
    """October 2021 market data and weather, plus the calendar the context needs."""
    days = build_calendar_days(
        "VIC", "Australia/Melbourne", dt.date(2021, 9, 1), dt.date(2021, 11, 30)
    )
    with loaded.begin() as conn:
        upsert_calendar_days(conn, days)
    return loaded


def _october_cfg() -> AppConfig:
    cfg = load_config()
    cfg.ml = cfg.ml.model_copy(
        update={
            "first_issue": dt.date(2021, 10, 9),
            "plus_first_issue": dt.date(2021, 10, 20),
            "plus_train_start": dt.date(2021, 10, 9),
        }
    )
    cfg.backtest.holdout_start = dt.date(2021, 10, 25)
    return cfg


def test_builder_reads_weather_from_the_database(ready: Engine) -> None:
    cfg = load_config()
    b = load_feature_builder(ready, cfg, "VIC1")
    origins = origin_positions(b.ctx, b.setup, nem_midnight_utc("2021-10-10"))[:4]
    for v in catalogue.VARIANTS:
        fm = b.build(v.id, origins)
        assert list(fm.X.columns) == catalogue.feature_names(v.id)
        assert fm.complete().all(), v.id  # the fake archive has no gaps
    c = b.build("C", origins)
    assert c.weather_offset is not None and set(np.unique(c.weather_offset)) == {1, 2}

    # D's weighted temperature equals ERA5 weighted and interpolated directly.
    first, last = b.ctx.times[0].floor("1h"), b.ctx.times[-1].ceil("1h")
    obs = load_weather_observations(ready, cfg.weather.observations.model, first, last)
    weighted = site_weighted(obs, cfg.weather.site_weights("VIC1"), ["temperature_2m"])
    d = b.build("D", origins)
    targets = pd.DatetimeIndex(b.ctx.times[d.targets])
    expected = to_five_minute(weighted, targets, {"temperature_2m": "instant"})["temperature_2m"]
    np.testing.assert_allclose(d.X["wx_temp"], expected)


def test_sensitivity_delay_uses_older_forecasts(ready: Engine) -> None:
    cfg = load_config()
    b6 = load_feature_builder(ready, cfg, "VIC1")
    b8 = load_feature_builder(ready, cfg, "VIC1", delay_hours=8, ctx=b6.ctx)
    origins = origin_positions(b6.ctx, b6.setup, nem_midnight_utc("2021-10-10"))[:48]
    u6, u8 = b6.build("C", origins).weather_offset, b8.build("C", origins).weather_offset
    assert u6 is not None and u8 is not None
    assert (u8 == 1).sum() < (u6 == 1).sum()
    with pytest.raises(ValueError, match="leak"):
        load_feature_builder(ready, cfg, "VIC1", delay_hours=5, ctx=b6.ctx)


def test_audit_runs_end_to_end(ready: Engine) -> None:
    res = run_audit(ready, _october_cfg(), "VIC1")
    c = res.variants["C"]
    assert c.pairs == c.origins * 288 and c.complete == c.pairs
    assert res.variants["Cplus"].origins < c.origins
    assert res.temperature_gap.best_shift == 0  # fake forecasts track the fake ERA5 closely
    text = render_audit(res)
    assert "No variant excludes any pair." in text and "(aligned)" in text


def test_audit_without_development_origins_fails_clearly(ready: Engine) -> None:
    with pytest.raises(LookupError, match="no development issue times"):
        run_audit(ready, load_config(), "VIC1")  # ml.first_issue is after October 2021
