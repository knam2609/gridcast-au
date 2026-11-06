"""Walk-forward harness for the M2 ML models (m2_plan.md §2.4).

Covers the retrain schedule, the staggered training sample, the training-row and
early-stopping rules at a cutoff, per-retrain selection, shared rows across variants,
and an end-to-end leakage property: forecasts issued at or before a time *c* do not change
when every demand value after *c* is replaced by noise.
"""

from __future__ import annotations

import datetime as dt
import itertools
from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.config import LightGBMConfig, MlConfig, RidgeConfig, load_config
from gridcast.features.builder import FeatureBuilder
from gridcast.forecasting import walkforward as wf
from gridcast.forecasting.origins import (
    ForecastSetup,
    SeriesContext,
    nem_midnight_utc,
    origin_positions,
)
from gridcast.models.baselines import DAY_STEPS
from tests.feature_fixtures import hourly_weather, weather_inputs
from tests.forecast_fixtures import make_ctx

SETUP = ForecastSetup()


def small_ml_config() -> MlConfig:
    base = load_config().ml
    return base.model_copy(
        update={
            "early_stopping_weeks": 1,
            "lightgbm": LightGBMConfig(
                learning_rate=0.2,
                min_data_in_leaf=50,
                max_rounds=25,
                early_stopping_rounds=5,
                num_threads=2,
            ),
            "ridge": RidgeConfig(max_rows=20_000, spline_knots=4),
        }
    )


def _demand(pos: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(11)
    daily = 800 * np.sin(2 * np.pi * pos / DAY_STEPS)
    return 5000 + daily + rng.normal(0, 120, len(pos))


@cache
def _data() -> tuple[SeriesContext, pd.DataFrame, pd.DataFrame]:
    ctx = make_ctx(start="2024-03-01 00:05", days=60, values=_demand)
    fc, obs = hourly_weather(ctx.times, seed=5)
    return ctx, fc, obs


def _builder(ctx: SeriesContext | None = None) -> FeatureBuilder:
    base, fc, obs = _data()
    ctx = ctx if ctx is not None else base
    return FeatureBuilder(ctx, SETUP, weather_inputs(ctx.times, fc, obs))


def _nem_day(day: str) -> pd.Timestamp:
    return nem_midnight_utc(day)


# --------------------------------------------------------------------------- schedule


def test_quarterly_schedule_matches_the_plan() -> None:
    cfg = load_config()
    end = _nem_day(str(cfg.backtest.holdout_start))
    schedule = wf.retrain_schedule(cfg.ml.first_issue, end, cfg.ml.retrain_every_months)
    assert len(schedule) == 12
    assert schedule[0] == _nem_day("2022-11-01")
    assert schedule[-1] == _nem_day("2025-08-01")
    assert _nem_day(str(cfg.ml.plus_first_issue)) in schedule
    assert all(b > a for a, b in itertools.pairwise(schedule))


def test_add_months_keeps_the_day() -> None:
    assert wf.add_months(dt.date(2024, 11, 1), 3) == dt.date(2025, 2, 1)
    assert wf.add_months(dt.date(2024, 11, 30), 3) == dt.date(2025, 2, 28)


# --------------------------------------------------------------------------- sample


def test_stagger_covers_every_phase_for_every_issue_time_of_day() -> None:
    times = pd.date_range("2024-03-01", periods=3 * 48, freq="30min", tz="UTC")
    phase = wf.stagger_phase(times, SETUP, 3)
    table = pd.DataFrame({"tod": times.strftime("%H:%M"), "phase": phase})
    per_tod = table.groupby("tod")["phase"].apply(lambda p: sorted(p))
    assert (per_tod.apply(lambda p: p == [0, 1, 2])).all()


def test_training_sample_covers_every_lead_time() -> None:
    builder = _builder()
    origins = wf.train_origins(builder, 20 * DAY_STEPS)
    sample = wf.training_sample(builder, "A", origins, 3)
    per_origin = pd.Series(sample.horizons).groupby(sample.origins).size()
    assert (per_origin == 96).all()
    assert set(np.unique(sample.horizons)) == set(range(1, 289))
    # Every 5-minute slot of the day is a training target.
    slots = builder.ctx.slot[sample.targets]
    assert set(np.unique(slots)) == set(range(288))
    assert sample.X.dtype == np.float32


def test_fit_rows_respect_the_cutoff_and_order_the_early_stopping_split() -> None:
    builder = _builder()
    cutoff = 30 * DAY_STEPS
    es_steps = 7 * DAY_STEPS
    origins = wf.train_origins(builder, cutoff)
    sample = wf.training_sample(builder, "C", origins, 3)
    rows, es_train, es_valid = wf.fit_rows(sample, sample.usable, cutoff, es_steps)
    targets, issued = sample.targets[rows], sample.origins[rows]
    assert targets.max() <= cutoff
    assert sample.usable[rows].all()
    assert issued[es_valid].min() >= cutoff - es_steps
    assert targets[es_train].max() <= cutoff - es_steps  # fitted targets precede monitoring
    assert not (es_train & es_valid).any()
    # Pairs straddling the split are in neither early-stopping set but are refit rows.
    assert (~es_train & ~es_valid).any()


def test_variants_of_a_group_share_training_rows() -> None:
    builder = _builder()
    origins = wf.train_origins(builder, 25 * DAY_STEPS)
    a = wf.training_sample(builder, "A", origins, 3)
    c = wf.training_sample(builder, "C", origins, 3)
    shared = wf.shared_usable([a, c])
    assert a.usable.all() and not c.usable.all()  # synthetic forecasts have gaps
    np.testing.assert_array_equal(shared, c.usable)


# --------------------------------------------------------------------------- selection


def test_selection_picks_the_lowest_early_stopping_mae_and_refits_it() -> None:
    builder = _builder()
    cfg = small_ml_config()
    cutoff = 30 * DAY_STEPS
    sample = wf.training_sample(builder, "A", wf.train_origins(builder, cutoff), 3)
    rows, es_train, es_valid = wf.fit_rows(sample, sample.usable, cutoff, 7 * DAY_STEPS)
    candidates = (
        {"target_form": "residual", "alpha": 1e9},  # crushed to the intercept: poor
        {"target_form": "residual", "alpha": 1.0},
    )
    spec = wf.ModelSpec("ridge_A", "A", "ridge", candidates=candidates)
    model, record = wf.select_and_refit(
        spec, sample.columns, sample.X[rows], sample.actual[rows], es_train, es_valid, "x", cfg
    )
    maes = [c["mae"] for c in record.candidates]
    assert record.chosen == candidates[int(np.argmin(maes))] == candidates[1]
    assert record.rows == len(rows)
    assert model.params()["alpha"] == 1.0


def test_fixed_settings_skip_the_search() -> None:
    builder = _builder()
    cfg = small_ml_config()
    cutoff = 30 * DAY_STEPS
    sample = wf.training_sample(builder, "A", wf.train_origins(builder, cutoff), 3)
    rows, es_train, es_valid = wf.fit_rows(sample, sample.usable, cutoff, 7 * DAY_STEPS)
    setting = {"target_form": "direct", "objective": "l2", "num_leaves": 7}
    spec = wf.ModelSpec("lgbm_A_x", "A", "lightgbm", fixed={"2024-04-01": setting})
    _model, record = wf.select_and_refit(
        spec,
        sample.columns,
        sample.X[rows],
        sample.actual[rows],
        es_train,
        es_valid,
        "2024-04-01",
        cfg,
    )
    assert record.chosen == setting and len(record.candidates) == 1
    assert record.params["n_trees"] == record.candidates[0]["best_iteration"]
    with pytest.raises(KeyError, match="no fixed setting"):
        wf.select_and_refit(
            spec, sample.columns, sample.X[rows], sample.actual[rows], es_train, es_valid, "y", cfg
        )


def test_spec_needs_exactly_one_way_of_choosing() -> None:
    with pytest.raises(ValueError, match="either candidates or fixed"):
        wf.ModelSpec("m", "A", "ridge")


# --------------------------------------------------------------------------- end to end


def _specs() -> list[wf.ModelSpec]:
    lgbm = ({"target_form": "residual", "objective": "l2", "num_leaves": 7},)
    ridge = ({"target_form": "residual", "alpha": 1.0}, {"target_form": "direct", "alpha": 1.0})
    return [
        wf.ModelSpec("lgbm_A", "A", "lightgbm", candidates=lgbm),
        wf.ModelSpec("ridge_A", "A", "ridge", candidates=ridge),
        wf.ModelSpec("lgbm_C", "C", "lightgbm", candidates=lgbm),
    ]


def _run(builder: FeatureBuilder) -> tuple[wf.WalkForwardResult, list[pd.Timestamp]]:
    retrains = [_nem_day("2024-04-01"), _nem_day("2024-04-15")]
    eval_origins = origin_positions(builder.ctx, SETUP, retrains[0])
    eval_origins = eval_origins[builder.ctx.times[eval_origins] < _nem_day("2024-04-22")]
    return wf.run_walk_forward(
        builder, _specs(), eval_origins, retrains, small_ml_config()
    ), retrains


@cache
def _baseline_run() -> tuple[wf.WalkForwardResult, list[pd.Timestamp]]:
    return _run(_builder())


def test_walk_forward_shapes_folds_and_missing_weather() -> None:
    res, _retrains = _baseline_run()
    assert res.folds == ["2024-04-01", "2024-04-15"]
    for name, pred in res.predictions.items():
        assert pred.shape == (len(res.origins), SETUP.horizon_steps), name
    assert np.isfinite(res.predictions["lgbm_A"]).all()
    # C is NaN exactly where a pair has no published forecast weather.
    c = res.predictions["lgbm_C"]
    assert np.isnan(c).any() and np.isfinite(c).mean() > 0.5
    # One fit per retrain and lead band, each selected on its own rows.
    assert all(len(r) == 2 * len(small_ml_config().lead_bands) for r in res.retrains.values())
    first = res.retrains["lgbm_A"][:2]
    assert [r.band for r in first] == ["h001-024", "h025-288"]
    assert first[0].rows < first[1].rows
    assert res.retrains["ridge_A"][0].chosen["target_form"] in ("residual", "direct")


def test_forecasts_ignore_demand_after_their_issue_time() -> None:
    base, retrains = _baseline_run()
    ctx = _data()[0]
    times = ctx.times[base.origins]
    # c lies inside the first fold: forecasts issued up to c use the first model only.
    c_time = _nem_day("2024-04-08")
    c = int(ctx.times.searchsorted(c_time))
    y = ctx.y.copy()
    y[c + 1 :] = np.random.default_rng(3).normal(-1e4, 1e4, len(y) - c - 1)
    perturbed, _ = _run(_builder(ctx.with_y(y)))
    before = np.asarray(times <= c_time)
    for name in base.predictions:
        np.testing.assert_array_equal(
            base.predictions[name][before], perturbed.predictions[name][before], err_msg=name
        )
    # Control: the second retrain fits on the perturbed days, so later forecasts change.
    after = np.asarray(times >= retrains[1])
    assert not np.allclose(
        base.predictions["lgbm_A"][after], perturbed.predictions["lgbm_A"][after], equal_nan=True
    )


def test_banded_model_routes_each_horizon_to_its_band() -> None:
    class Const:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), self.value)

    model = wf.BandedModel({(1, 24): Const(1.0), (25, 288): Const(2.0)})  # type: ignore[dict-item]
    horizons = np.array([1, 24, 25, 288])
    np.testing.assert_array_equal(model.predict(np.zeros((4, 3)), horizons), [1, 1, 2, 2])
    with pytest.raises(ValueError, match="outside every lead band"):
        wf.BandedModel({(1, 24): Const(1.0)}).predict(np.zeros((1, 3)), np.array([30]))  # type: ignore[dict-item]


def test_lead_bands_must_partition_the_horizons() -> None:
    from gridcast.config import AppConfig

    raw = load_config().model_dump()
    raw["ml"]["lead_bands"] = [(1, 24), (26, 288)]
    with pytest.raises(ValueError, match="contiguous"):
        AppConfig.model_validate(raw)
    raw["ml"]["lead_bands"] = [(1, 200)]
    with pytest.raises(ValueError, match="end at"):
        AppConfig.model_validate(raw)
