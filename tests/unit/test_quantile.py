"""Quantile LightGBM walk-forward (M3 step 2, m3_plan.md §2.3).

Covers the quantile levels and keys, the quantile model (pinball early stopping, a refit
that reproduces it), the per-quantile selection, the walk-forward's exact reproduction of
the point model trained by M2's harness (and its refusal when the rows or tree counts
differ), and the leakage property: quantiles issued at or before a time *c* do not change
when every demand value after *c* is replaced by noise.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.features.builder import FeatureBuilder
from gridcast.forecasting import walkforward as wf
from gridcast.forecasting.origins import origin_positions
from gridcast.intervals.quantile import (
    QrResult,
    QuantileLightGBMModel,
    ReproductionError,
    crossed_pairs,
    interval_quantiles,
    point_fits,
    quantile_grid,
    quantile_key,
    quantile_levels,
    run_qr_walk_forward,
    select_quantile,
)
from gridcast.intervals.scores import pinball
from tests.unit.test_ml_models import LGB, _demand, _features, _split
from tests.unit.test_walkforward import SETUP, _builder, _data, _nem_day, small_ml_config

TAUS = quantile_levels([50, 80, 95])
RETRAINS = ("2024-04-01", "2024-04-15")
GRID = ({"target_form": "residual", "num_leaves": 7}, {"target_form": "latest", "num_leaves": 15})


def test_quantile_levels_and_keys() -> None:
    assert TAUS == (0.025, 0.1, 0.25, 0.75, 0.9, 0.975)
    assert [quantile_key(t) for t in TAUS] == ["q025", "q100", "q250", "q750", "q900", "q975"]
    assert interval_quantiles(80) == ("q100", "q900")
    assert interval_quantiles(95) == ("q025", "q975")
    with pytest.raises(ValueError):
        quantile_key(0.0125)
    with pytest.raises(ValueError):
        quantile_levels([100])


def test_quantile_grid_is_m2s_forms_times_leaves_in_order() -> None:
    cfg = small_ml_config()
    grid = quantile_grid(cfg)
    assert len(grid) == len(cfg.grid.target_forms) * len(cfg.grid.lightgbm_num_leaves) == 8
    assert grid[0] == {"target_form": "residual", "num_leaves": 31}
    assert grid[1] == {"target_form": "residual", "num_leaves": 127}
    assert {g["target_form"] for g in grid} == {"residual", "direct", "anchored", "latest"}


def test_crossed_pairs_ignores_missing_levels() -> None:
    q = np.array([[[1.0, 3.0, np.nan]], [[2.0, 2.0, 5.0]]])  # 2 levels, 1 issue time, 3 leads
    finite, crossed = crossed_pairs(q)
    np.testing.assert_array_equal(finite, [[True, True, False]])
    np.testing.assert_array_equal(crossed, [[False, True, False]])


# --------------------------------------------------------------------------- the model


def _noisy(cols: list[str], X: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _demand(cols, X) + rng.normal(0, 100, len(X))


def test_quantile_model_early_stops_on_pinball_loss() -> None:
    cols, X = _features("A", 5000, seed=2)
    actual = _noisy(cols, X)
    es_train, es_valid = _split(len(X))
    model = QuantileLightGBMModel(cols, "residual", 0.9, 15, LGB, seed=1)
    score = model.score(X, actual, es_train, es_valid)
    assert score.best_iteration is not None and model.n_trees == score.best_iteration
    # Refitting the early-stopping rows with the stopped tree count rebuilds the same booster,
    # so its pinball loss on the slice is the score LightGBM reported.
    again = QuantileLightGBMModel(cols, "residual", 0.9, 15, LGB, seed=1)
    again.refit(X[es_train], actual[es_train], score.best_iteration)
    q = again.predict(X[es_valid])
    loss = float(np.mean(pinball(q, actual[es_valid], 0.9)))
    assert loss == pytest.approx(score.mae, rel=1e-6)
    assert 0.8 < np.mean(actual[es_valid] <= q) < 0.97
    params = again.params()
    assert params["family"] == "lightgbm_quantile" and params["quantile"] == 0.9
    assert params["objective"] == "quantile" and params["n_trees"] == score.best_iteration


def test_quantile_model_orders_its_levels_and_is_deterministic() -> None:
    cols, X = _features("A", 4000, seed=3)
    actual = _noisy(cols, X, seed=1)
    es_train, es_valid = _split(len(X))
    preds = {}
    for tau in (0.1, 0.9, 0.9):
        m = QuantileLightGBMModel(cols, "direct", tau, 15, LGB, seed=5)
        s = m.score(X, actual, es_train, es_valid)
        m.refit(X, actual, s.best_iteration)
        p = m.predict(X)
        if tau in preds:
            np.testing.assert_array_equal(preds[tau], p)
        preds[tau] = p
    assert np.mean(preds[0.1] < preds[0.9]) > 0.95
    with pytest.raises(ValueError, match="quantile level"):
        QuantileLightGBMModel(cols, "direct", 1.0, 15, LGB, seed=5)


def test_selection_picks_the_lowest_pinball_and_refits_it() -> None:
    cols, X = _features("A", 4000, seed=6)
    actual = _noisy(cols, X, seed=2)
    es_train, es_valid = _split(len(X))
    cfg = small_ml_config()
    grid = [
        {"target_form": "direct", "num_leaves": 3},
        {"target_form": "residual", "num_leaves": 15},
    ]
    model, fit = select_quantile(cols, X, actual, es_train, es_valid, 0.9, grid, cfg)
    losses = [c["pinball"] for c in fit.candidates]
    assert [c["setting"] for c in fit.candidates] == grid
    assert fit.chosen == grid[int(np.argmin(losses))] and fit.es_pinball == min(losses)
    assert fit.chosen == {"target_form": "residual", "num_leaves": 15}  # the learnable form
    assert model.target_form == "residual" and model.num_leaves == 15 and model.tau == 0.9
    assert fit.n_trees == fit.candidates[1]["n_trees"] == model.params()["n_trees"]
    # Ties go to the first candidate in grid order.
    _m, tie = select_quantile(cols, X, actual, es_train, es_valid, 0.9, [grid[1], grid[1]], cfg)
    assert tie.candidates[0]["pinball"] == tie.candidates[1]["pinball"]
    assert tie.chosen == grid[1]


# --------------------------------------------------------------------------- walk-forward


def _point_run(builder: FeatureBuilder) -> tuple[wf.WalkForwardResult, np.ndarray]:
    """M2's harness with A and C searched together (C trains on the rows shared with A)."""
    retrains = [_nem_day(d) for d in RETRAINS]
    origins = origin_positions(builder.ctx, SETUP, retrains[0])
    origins = origins[builder.ctx.times[origins] < _nem_day("2024-04-22")]
    grid = (
        {"target_form": "residual", "objective": "l2", "num_leaves": 7},
        {"target_form": "latest", "objective": "l1", "num_leaves": 15},
    )
    specs = [
        wf.ModelSpec("lgbm_A", "A", "lightgbm", candidates=grid),
        wf.ModelSpec("lgbm_C", "C", "lightgbm", candidates=grid),
    ]
    return wf.run_walk_forward(builder, specs, origins, retrains, small_ml_config()), origins


@cache
def _point() -> tuple[wf.WalkForwardResult, np.ndarray]:
    return _point_run(_builder())


def _records() -> list[dict]:
    return [dataclasses.asdict(r) for r in _point()[0].retrains["lgbm_C"]]


def _qr(
    builder: FeatureBuilder, fits: dict | None = None, reference: np.ndarray | None = None
) -> QrResult:
    _res, origins = _point()
    return run_qr_walk_forward(
        builder,
        origins,
        [_nem_day(d) for d in RETRAINS],
        small_ml_config(),
        fits if fits is not None else point_fits(_records()),
        TAUS,
        GRID,
        group=("A", "C"),
        reference=reference,
    )


@cache
def _baseline_qr() -> QrResult:
    return _qr(_builder(), reference=_point()[0].predictions["lgbm_C"])


def test_walk_forward_reproduces_the_point_model_exactly() -> None:
    res = _baseline_qr()
    point = _point()[0].predictions["lgbm_C"]
    np.testing.assert_array_equal(res.point, point)
    assert res.folds == list(RETRAINS)
    assert res.quantiles.shape == (len(TAUS), *point.shape)
    # Quantiles exist exactly where the point model forecasts (C weather present).
    for q in res.quantiles:
        np.testing.assert_array_equal(np.isfinite(q), np.isfinite(point))
    bands = small_ml_config().lead_bands
    assert len(res.fits) == len(RETRAINS) * len(bands)
    for rec, stored in zip(res.fits, _records(), strict=True):
        assert [rec.retrain, rec.band, rec.rows] == [stored[k] for k in ("retrain", "band", "rows")]
        assert rec.point_chosen == stored["chosen"]
        assert rec.point_n_trees == stored["params"]["n_trees"]
        assert [q.tau for q in rec.quantiles] == list(TAUS)
        for q in rec.quantiles:
            assert [c["setting"] for c in q.candidates] == list(GRID)
            assert q.chosen in GRID and q.n_trees >= 1 and q.es_pinball > 0
            assert q.es_pinball == min(c["pinball"] for c in q.candidates)
        assert 0.0 <= rec.crossing_share <= 1.0
    # The fitted quantiles are ordered on average (crossings are rare, then sorted later).
    q = res.quantiles
    ok = np.isfinite(q).all(axis=0)
    assert np.mean(q[1][ok] < q[4][ok]) > 0.95  # q0.1 below q0.9


def test_walk_forward_refuses_other_rows_or_tree_counts() -> None:
    fits = point_fits(_records())
    key = next(iter(fits))
    rows = dict(fits)
    rows[key] = dataclasses.replace(fits[key], rows=fits[key].rows - 1)
    with pytest.raises(ReproductionError, match="training rows differ"):
        _qr(_builder(), fits=rows)
    trees = dict(fits)
    trees[key] = dataclasses.replace(fits[key], n_trees=fits[key].n_trees + 3)
    with pytest.raises(ReproductionError, match="point forecasts differ"):
        _qr(_builder(), fits=trees, reference=_point()[0].predictions["lgbm_C"])


def test_quantiles_ignore_demand_after_their_issue_time() -> None:
    base = _baseline_qr()
    ctx = _data()[0]
    times = pd.DatetimeIndex(ctx.times[base.origins])
    c_time = _nem_day("2024-04-08")  # inside the first fold
    c = int(ctx.times.searchsorted(c_time))
    y = ctx.y.copy()
    y[c + 1 :] = np.random.default_rng(3).normal(-1e4, 1e4, len(y) - c - 1)
    # Same stored fits (row counts do not depend on demand values); no forecast check.
    perturbed = _qr(_builder(ctx.with_y(y)))
    before = np.asarray(times <= c_time)
    np.testing.assert_array_equal(base.quantiles[:, before], perturbed.quantiles[:, before])
    # Control: the second retrain fits on the perturbed days, so later quantiles change.
    after = np.asarray(times >= _nem_day(RETRAINS[1]))
    assert not np.allclose(base.quantiles[:, after], perturbed.quantiles[:, after], equal_nan=True)
