import datetime as dt

import pytest

from gridcast.config import MarketLimits, load_config, load_market_limits


def test_base_config_loads() -> None:
    cfg = load_config()
    assert "VIC1" in cfg.regions
    assert cfg.region("VIC1").local_tz == "Australia/Melbourne"
    assert cfg.nem.interval == dt.timedelta(minutes=5)
    assert cfg.forecast.horizon_steps * cfg.nem.dispatch_interval_minutes == 24 * 60
    assert cfg.backtest.eval_start < cfg.backtest.holdout_start


def test_unknown_region_raises() -> None:
    with pytest.raises(KeyError, match="not configured"):
        load_config().region("XX1")


def test_market_cap_by_financial_year() -> None:
    limits = load_market_limits(load_config())
    assert limits.market_floor == -1000.0
    # Financial year runs 1 July -> 30 June.
    assert limits.cap_for(dt.date(2025, 6, 30)) == 17500.0
    assert limits.cap_for(dt.date(2025, 7, 1)) == 20300.0
    assert limits.cap_for(dt.date(1999, 1, 1)) is None


def test_market_limits_rejects_bad_fy_key() -> None:
    with pytest.raises(ValueError, match="Financial-year key"):
        MarketLimits.model_validate(
            {"market_floor": -1000, "market_price_cap": {"2024-26": {"value": 1, "source": "x"}}}
        )


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Settings built from the process environment only (ignoring the local .env file)."""
    from gridcast import config

    monkeypatch.delenv("GRIDCAST_DATA_ROOT", raising=False)
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(_env_file=None))
    return monkeypatch


def test_data_paths_default_to_project_data_dir(isolated_settings: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from gridcast.config import PROJECT_ROOT

    cfg = load_config()
    assert cfg.data_root() == PROJECT_ROOT / "data"
    assert cfg.data_path(cfg.paths.raw_dir) == PROJECT_ROOT / "data" / "raw"
    assert cfg.data_path(Path("/abs/elsewhere")) == Path("/abs/elsewhere")


def test_data_root_env_override(isolated_settings: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    isolated_settings.setenv("GRIDCAST_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    assert cfg.data_root() == tmp_path
    assert cfg.data_path(cfg.paths.backtest_dir) == tmp_path / "backtests"


def test_missing_external_data_root_fails_loudly(
    isolated_settings: pytest.MonkeyPatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    isolated_settings.setenv("GRIDCAST_DATA_ROOT", str(tmp_path / "unmounted-drive" / "data"))
    cfg = load_config()
    with pytest.raises(FileNotFoundError, match="mount the drive"):
        cfg.data_root()
    assert not (tmp_path / "unmounted-drive").exists()
