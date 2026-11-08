"""M2 regressors (LightGBM, ridge), weather segments and ML provenance helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridcast.config import LightGBMConfig, RidgeConfig, load_config
from gridcast.evaluation.segments import describe_weather_segments, weather_segments
from gridcast.features import catalogue
from gridcast.forecasting.ml_backtest import ml_config_hash
from gridcast.models.ml import (
    BASE_FEATURE,
    LightGBMModel,
    RidgeDesign,
    RidgeModel,
    grid_choices,
    make_model,
)

LGB = LightGBMConfig(
    learning_rate=0.3, min_data_in_leaf=20, max_rounds=200, early_stopping_rounds=10, num_threads=2
)


def _features(variant: str, n: int, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Plausible feature rows for ``variant`` (calendar codes in range)."""
    rng = np.random.default_rng(seed)
    cols = catalogue.feature_names(variant)
    X = rng.normal(0.0, 1.0, (n, len(cols)))
    ranges = {
        "lead_minutes": (5, 1440, 5),
        "cal_slot": (0, 287, 1),
        "cal_day_of_week": (0, 6, 1),
        "cal_month": (1, 12, 1),
    }
    for name, (lo, hi, step) in ranges.items():
        X[:, cols.index(name)] = rng.integers(lo // step, hi // step + 1, n) * step
    for name in (
        "cal_non_working",
        "cal_public_holiday",
        "cal_dst_change_day",
        "cal_yesterday_non_working",
        "cal_yesterday_public_holiday",
    ):
        X[:, cols.index(name)] = rng.integers(0, 2, n)
    X[:, cols.index(BASE_FEATURE)] = rng.normal(5000, 500, n)
    if "wx_temp" in cols:
        X[:, cols.index("wx_temp")] = rng.normal(18, 6, n)
    return cols, X.astype(np.float32)


def _demand(cols: list[str], X: np.ndarray) -> np.ndarray:
    """Yesterday's value plus a learnable correction."""
    base = X[:, cols.index(BASE_FEATURE)].astype(np.float64)
    return (
        base
        + 300 * X[:, cols.index("dem_now_vs_yesterday")]
        + 50 * X[:, cols.index("cal_non_working")]
    )


def _split(n: int) -> tuple[np.ndarray, np.ndarray]:
    es_valid = np.zeros(n, dtype=bool)
    es_valid[int(0.8 * n) :] = True
    return ~es_valid, es_valid


@pytest.mark.parametrize("form", ["residual", "direct", "anchored", "latest"])
def test_target_forms_round_trip(form: str) -> None:
    cols, X = _features("A", 50)
    model = RidgeModel(cols, form, 1.0, RidgeConfig(), seed=0)
    actual = _demand(cols, X)
    y = model.target(X, actual)
    np.testing.assert_allclose(model.to_forecast(X, y), actual, rtol=1e-6)
    base = X[:, cols.index(BASE_FEATURE)].astype(np.float64)
    expected = {
        "residual": actual - base,
        "direct": actual,
        "anchored": actual - base - X[:, cols.index("dem_now_vs_yesterday")],
        "latest": actual - X[:, cols.index("dem_now")],
    }[form]
    np.testing.assert_allclose(y, expected, rtol=1e-6)


def test_lightgbm_scores_refits_and_saves(tmp_path: Path) -> None:
    cols, X = _features("A", 4000)
    actual = _demand(cols, X)
    es_train, es_valid = _split(len(X))
    model = LightGBMModel(cols, "residual", "l2", 15, LGB, seed=1)
    score = model.score(X, actual, es_train, es_valid)
    naive = np.mean(np.abs(actual - X[:, cols.index(BASE_FEATURE)])[es_valid])
    assert score.best_iteration is not None and score.mae < 0.5 * naive
    model.refit(X, actual, score.best_iteration)
    assert model.params()["n_trees"] == score.best_iteration
    pred = model.predict(X[es_valid])
    assert np.mean(np.abs(pred - actual[es_valid])) < 0.5 * naive
    path = model.save(tmp_path / "lgbm" / "2024-01-01")
    assert path.name == "2024-01-01.txt.gz" and path.stat().st_size > 0


def test_lightgbm_is_deterministic() -> None:
    cols, X = _features("A", 3000, seed=4)
    actual = _demand(cols, X)
    es_train, es_valid = _split(len(X))
    preds = []
    for _ in range(2):
        m = LightGBMModel(cols, "direct", "l1", 31, LGB, seed=7)
        s = m.score(X, actual, es_train, es_valid)
        m.refit(X, actual, s.best_iteration)
        preds.append(m.predict(X))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_models_reject_incomplete_rows() -> None:
    cols, X = _features("A", 200)
    actual = _demand(cols, X)
    X[3, 0] = np.nan
    es_train, es_valid = _split(len(X))
    with pytest.raises(ValueError, match="complete"):
        LightGBMModel(cols, "residual", "l2", 7, LGB, 0).score(X, actual, es_train, es_valid)


def test_ridge_design_columns_and_spline() -> None:
    cols_a, Xa = _features("A", 300)
    cols_c, Xc = _features("C", 300)
    design_a = RidgeDesign(cols_a, spline_knots=5).fit(Xa)
    design_c = RidgeDesign(cols_c, spline_knots=5).fit(Xc)
    Da, Dc = design_a.transform(Xa), design_c.transform(Xc)
    assert Da.shape == (300, design_a.n_columns)
    assert Dc.shape == (300, design_c.n_columns)
    assert design_a.spline is None and design_c.spline is not None
    # C adds 5 weather columns plus the temperature spline basis.
    assert design_c.n_columns - design_a.n_columns == 5 + design_c.spline.n_features_out_
    bad = Xa.copy()
    bad[0, cols_a.index("cal_month")] = 13
    with pytest.raises(ValueError, match="month"):
        design_a.transform(bad)


def test_ridge_learns_the_correction_and_subsamples() -> None:
    cols, X = _features("C", 3000)
    actual = _demand(cols, X)
    es_train, es_valid = _split(len(X))
    model = RidgeModel(cols, "residual", 1.0, RidgeConfig(max_rows=1000, spline_knots=4), seed=0)
    score = model.score(X, actual, es_train, es_valid)
    assert score.mae < 20 and score.best_iteration is None
    model.refit(X, actual)
    assert model.params()["rows_fitted"] == 1000


def test_grid_order_and_factory() -> None:
    cfg = load_config().ml
    lgbm = grid_choices("lightgbm", cfg)
    ridge = grid_choices("ridge", cfg)
    g = cfg.grid
    assert len(lgbm) == len(g.target_forms) * len(g.lightgbm_objective) * len(g.lightgbm_num_leaves)
    assert len(ridge) == len(g.target_forms) * len(g.ridge_alpha)
    assert lgbm[0]["target_form"] == "residual"  # the planned target form wins ties
    cols, _ = _features("A", 10)
    assert isinstance(make_model("lightgbm", cols, lgbm[0], cfg), LightGBMModel)
    assert isinstance(make_model("ridge", cols, ridge[0], cfg), RidgeModel)
    with pytest.raises(ValueError, match="family"):
        make_model("svm", cols, lgbm[0], cfg)


def test_weather_segments_use_local_days_and_daytime() -> None:
    cfg = load_config().evaluation
    day = np.repeat([1, 2, 3], 24)
    hour = np.tile(np.arange(24), 3)
    temp = np.full(72, 20.0)
    temp[5] = 36.0  # day 1 reaches 36 °C once
    cloud = np.full(72, 10.0)
    cloud[24 + 9 : 24 + 17] = 90.0  # day 2: cloudy all daytime
    cloud[48:56] = 100.0  # day 3: cloudy only at night
    seg = weather_segments(day, hour, temp, cloud, cfg)
    np.testing.assert_array_equal(seg["very_hot_day"], day == 1)
    np.testing.assert_array_equal(seg["very_cloudy_day"], day == 2)
    assert set(describe_weather_segments(cfg)) == set(seg)


def test_weather_segments_ignore_missing_values() -> None:
    cfg = load_config().evaluation
    day = np.repeat([1, 2], 24)
    hour = np.tile(np.arange(24), 2)
    temp = np.where(day == 1, np.nan, 40.0)
    cloud = np.full(48, np.nan)
    seg = weather_segments(day, hour, temp, cloud, cfg)
    np.testing.assert_array_equal(seg["very_hot_day"], day == 2)
    assert not seg["very_cloudy_day"].any()


def test_config_hash_tracks_model_settings_only() -> None:
    cfg = load_config()
    h = ml_config_hash(cfg)
    assert h == ml_config_hash(cfg)
    moved = cfg.model_copy(
        update={"ml": cfg.ml.model_copy(update={"model_dir": Path("elsewhere")})}
    )
    assert ml_config_hash(moved) == h
    changed = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"seed": 1})})
    assert ml_config_hash(changed) != h


def test_ml_config_matches_holdout_freeze() -> None:
    """The holdout config is frozen (m2_plan.md step 6): an ML/feature setting change fails."""
    cfg = load_config()
    assert cfg.backtest.frozen_ml_config_hash is not None
    assert ml_config_hash(cfg) == cfg.backtest.frozen_ml_config_hash


def test_evaluate_accepts_extra_segments_and_split_subset() -> None:
    from gridcast.forecasting.backtest import evaluate, run_backtest
    from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
    from gridcast.models.baselines import default_baselines
    from tests.forecast_fixtures import make_ctx

    cfg = load_config()
    ctx = make_ctx(start="2024-01-01 00:05", days=20)
    out = run_backtest(
        ctx, default_baselines("demand"), ForecastSetup(), "demand", ctx.times[7 * 288 + 1]
    )
    extra = {"first_half": np.arange(len(ctx)) < len(ctx) // 2}
    m = evaluate(out, ctx, cfg.evaluation, nem_midnight_utc("2030-01-01"), extra, splits=("dev",))
    assert set(m["split"]) == {"dev"}
    assert "first_half" in set(m["segment"])
    with pytest.raises(ValueError, match="defined twice"):
        evaluate(
            out, ctx, cfg.evaluation, nem_midnight_utc("2030-01-01"), {"peak": extra["first_half"]}
        )
