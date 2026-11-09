import numpy as np
import pandas as pd
import pytest

from gridcast.config import load_config
from gridcast.evaluation.segments import target_segments
from gridcast.forecasting.backtest import evaluate, monthly_folds, run_backtest
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.models.baselines import default_baselines
from tests.forecast_fixtures import make_ctx

SETUP = ForecastSetup()


@pytest.fixture(scope="module")
def periodic_ctx():  # type: ignore[no-untyped-def]
    # Purely daily-periodic series (period 288 steps): daily & weekly naive are exact.
    return make_ctx(
        start="2024-01-01 00:05",
        days=70,
        values=lambda p: 5000 + 1000 * np.sin(2 * np.pi * p / 288),
    )


def test_seasonal_naive_is_exact_on_periodic_series(periodic_ctx) -> None:  # type: ignore[no-untyped-def]
    start = periodic_ctx.times[7 * 288 + 1]
    out = run_backtest(periodic_ctx, default_baselines("demand"), SETUP, "demand", start)
    mask = out.common_mask()
    for name in ("daily_naive", "weekly_naive"):
        err = (out.predictions[name] - out.actual)[mask]
        assert np.max(np.abs(err)) < 1e-2, name  # float32 storage rounding only
    assert np.max(np.abs((out.predictions["persistence"] - out.actual)[mask])) > 100


def test_folds_are_chronological_nem_months(periodic_ctx) -> None:  # type: ignore[no-untyped-def]
    start = periodic_ctx.times[7 * 288 + 1]
    out = run_backtest(periodic_ctx, default_baselines("demand"), SETUP, "demand", start)
    assert out.folds == sorted(out.folds)
    assert out.folds[0] == "2024-01"
    labels = monthly_folds(out.origin_times)
    assert sum(len(idx) for _, idx in labels) == len(out.origins)


def test_evaluate_splits_segments_and_horizons(periodic_ctx) -> None:  # type: ignore[no-untyped-def]
    cfg = load_config()
    start = periodic_ctx.times[7 * 288 + 1]
    out = run_backtest(periodic_ctx, default_baselines("demand"), SETUP, "demand", start)
    holdout = nem_midnight_utc("2024-02-15")
    metrics = evaluate(out, periodic_ctx, cfg.evaluation, holdout)

    all_seg = metrics[(metrics["segment"] == "all") & (metrics["metric"] == "mae")]
    per_model = all_seg[all_seg["split"] == "all"].groupby("model")["horizon"].nunique()
    assert (per_model == 289).all()  # pooled (0) + 288 horizons
    # dev + holdout pair counts add up to the "all" split.
    pooled = all_seg[all_seg["horizon"] == 0].pivot_table(
        index="model", columns="split", values="n", aggfunc="first"
    )
    assert ((pooled["dev"] + pooled["holdout"]) == pooled["all"]).all()
    # Non-"all" segments are reported at the configured horizons only.
    peak = metrics[(metrics["segment"] == "peak") & (metrics["metric"] == "mae")]
    assert set(peak["horizon"]) == {0, *cfg.evaluation.horizons_reported}
    assert set(metrics["metric"]) == {"mae", "rmse", "bias", "mape"}


def test_models_scored_on_identical_pairs() -> None:
    ctx = make_ctx(days=20)
    y = ctx.y.copy()
    y[3000:3010] = np.nan  # a gap affects every model's eligible pairs equally
    ctx = ctx.with_y(y)
    out = run_backtest(ctx, default_baselines("price"), SETUP, "price", ctx.times[7 * 288 + 1])
    metrics = evaluate(out, ctx, load_config().evaluation, nem_midnight_utc("2030-01-01"))
    n = metrics[
        (metrics["segment"] == "all") & (metrics["horizon"] == 0) & (metrics["split"] == "all")
    ]
    assert (n["n"] == n["n"].iloc[0]).all()


def test_segments() -> None:
    cfg = load_config().evaluation
    frame = pd.DataFrame(
        {
            "rrp_aud_mwh": [50.0, 500.0, -20.0, 80.0],
            "apc_flag_bits": [0, 20, 0, 1],  # 20 = IRLF + MPC binding (not administered)
            "market_suspended_flag": [0, 0, 1, 0],
            "local_hour": [3, 18, 20, 12],
            "is_weekend": [False, True, False, False],
            "is_public_holiday": [False, False, True, False],
            "is_dst_change_day": [False, False, False, True],
        }
    )
    seg = target_segments(frame, cfg)
    assert seg["price_spike"].tolist() == [False, True, False, False]
    assert seg["negative_price"].tolist() == [False, False, True, False]
    assert seg["market_intervention"].tolist() == [False, False, True, True]
    assert seg["normal"].tolist() == [True, False, False, False]
    assert seg["peak"].tolist() == [False, True, True, False]
    assert seg["working_day"].tolist() == [True, False, False, True]
