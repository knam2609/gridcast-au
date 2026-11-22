"""M4 v2 (m4_plan.md §2.10): D4 with asinh(price / $100) and D6 as a robust distance.

Everything else is v1's, so on the same data v2 must reproduce v1's D1, D2, D3 and D5
scores and p-values exactly.
"""

from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from gridcast.anomalies.detectors import (
    CATEGORY,
    ISOLATION_FOREST,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    ROBUST_DISTANCE,
    ROLLING,
    Recipe,
    compute_scores,
    price_x,
)
from gridcast.anomalies.distance import fit_distance, fit_distance_models
from gridcast.anomalies.injection import injected_scores, injection_specs
from gridcast.anomalies.recipe import anomaly_config_hash, recipe_config_hash, recipe_for
from gridcast.anomalies.run import DetectorRun, run_detectors
from gridcast.anomalies.series import layout
from gridcast.config import RobustDistanceConfig, load_config
from tests.anomaly_fixtures import base_inputs, small_config, small_v2_recipe, split
from tests.unit.test_anomaly_injection import _modified, _starts

CFG = small_config()
SPLIT = split()


@cache
def _runs() -> tuple[DetectorRun, DetectorRun]:
    series, band = base_inputs()
    return run_detectors(series, band, CFG), run_detectors(
        series, band, CFG, recipe=small_v2_recipe()
    )


# ----------------------------------------------------------------------------- the recipe


def test_v2_recipe_from_the_config() -> None:
    cfg = load_config()
    r = recipe_for(cfg, 2)
    assert r.version == 2 and r.multivariate == ROBUST_DISTANCE
    assert r.price_level_scale_aud == cfg.anomalies_v2.price_level.scale_aud == 100
    assert r.price_volatility_scale_aud == cfg.anomalies.price_level.scale_aud == 10
    assert r.detectors == (*ROLLING, ROBUST_DISTANCE) == r.rolling
    assert recipe_for(cfg, 1).rolling == ROLLING
    assert CATEGORY[ROBUST_DISTANCE] == CATEGORY[ISOLATION_FOREST] == "multivariate"
    with pytest.raises(ValueError, match="needs its settings"):
        Recipe(2, 100.0, 10.0, ROBUST_DISTANCE)
    with pytest.raises(ValueError, match="no anomaly recipe version"):
        recipe_for(cfg, 3)


def test_v2_hash_covers_v2_settings_and_leaves_v1_alone() -> None:
    cfg = load_config()
    h1, h2 = anomaly_config_hash(cfg), recipe_config_hash(cfg, 2)
    assert recipe_config_hash(cfg, 1) == h1 != h2
    v2 = cfg.anomalies_v2
    moved = cfg.model_copy(
        update={
            "anomalies_v2": v2.model_copy(
                update={"price_level": v2.price_level.model_copy(update={"scale_aud": 99.0})}
            )
        }
    )
    assert recipe_config_hash(moved, 2) != h2
    assert anomaly_config_hash(moved) == h1  # v1 never reads anomalies_v2
    record = cfg.model_copy(
        update={"anomalies_v2": v2.model_copy(update={"frozen_commit": "abc1234"})}
    )
    assert recipe_config_hash(record, 2) == h2  # the freeze record is not hashed


# ----------------------------------------------------------------------------- the distance


def _training(seed: int = 0, n: int = 4000, k: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(k, k))
    return rng.standard_t(5, size=(n, k)) @ mix


def _model(x: np.ndarray):
    ends = pd.date_range("2024-01-01", periods=len(x), freq="5min", tz="UTC")
    return fit_distance(x, 24 * 12 + 1, ends, np.arange(len(x)))


def test_distance_is_the_mahalanobis_form_about_the_median() -> None:
    x = _training()
    mm = _model(x)
    centre = np.median(x, axis=0)
    np.testing.assert_array_equal(mm.centre, centre)
    lw = LedoitWolf(assume_centered=True).fit(x - centre)
    np.testing.assert_allclose(mm.precision, np.linalg.inv(lw.covariance_), rtol=1e-8)
    rows = _training(seed=1, n=50)
    d = rows - centre
    expected = np.sqrt(np.einsum("ij,jk,ik->i", d, mm.precision, d))
    np.testing.assert_allclose(mm.score(rows), expected, rtol=1e-12)


def test_a_row_scores_the_same_alone_or_in_any_batch() -> None:
    mm = _model(_training())
    rows = _training(seed=2, n=37)
    together = mm.score(rows)
    alone = np.array([mm.score(rows[i : i + 1])[0] for i in range(len(rows))])
    assert np.array_equal(together, alone)
    assert np.array_equal(mm.score(rows[5:20]), together[5:20])


def test_a_one_feature_extreme_keeps_growing() -> None:
    """v1's diagnosis: the forest capped this at the training extreme; the distance does not."""
    x = _training()
    mm = _model(x)
    base = np.median(x, axis=0)
    train_max = mm.score(x).max()
    scores = []
    for v in (5.0, 20.0, 80.0):
        probe = base.copy()
        probe[0] = base[0] + v * x[:, 0].std()
        scores.append(mm.score(probe[None, :])[0])
    assert scores[0] < scores[1] < scores[2]
    assert scores[2] > train_max


def test_month_models_see_only_the_past_and_need_enough_rows() -> None:
    series, _ = base_inputs()
    _, state = compute_scores(series, None, CFG, recipe=small_v2_recipe())
    lay = layout(series, CFG.profile.slot_minutes)
    assert state.distances and not state.models
    for m, mm in state.distances.items():
        assert mm.train_last_end <= mm.start
        assert mm.n_train >= 64
        assert m in set(np.unique(lay.nem_month))
    strict = RobustDistanceConfig(train_weeks=1, min_train_rows=10**9)
    assert fit_distance_models(state.features, series.ends, lay, strict) == {}


# ----------------------------------------------------------------------------- v2 against v1


def test_v2_keeps_d1_d2_d3_d5_exactly() -> None:
    v1, v2 = _runs()
    for d in ROLLING:
        if d == PRICE_LEVEL:
            continue
        assert np.array_equal(v1.scores[d], v2.scores[d], equal_nan=True), d
        assert np.array_equal(v1.pvalues[d], v2.pvalues[d], equal_nan=True), d
        assert np.array_equal(v1.n_window[d], v2.n_window[d]), d


def test_v2_d4_uses_asinh_over_100() -> None:
    v1, v2 = _runs()
    series, _ = base_inputs()
    x = price_x(series.price, 100.0)
    prof = v2.state.price_profile
    expected = np.abs(prof.z(x, CFG.profile.mad_consistency, CFG.price_level.floor))
    assert np.array_equal(v2.scores[PRICE_LEVEL], expected, equal_nan=True)
    assert not np.array_equal(v1.scores[PRICE_LEVEL], v2.scores[PRICE_LEVEL], equal_nan=True)
    # D5 still reads x at $10: its volatility state is v1's.
    assert np.array_equal(v1.state.v, v2.state.v, equal_nan=True)
    assert PRICE_VOLATILITY in v2.recipe.rolling


def test_v2_d6_is_calibrated_on_its_own_trailing_scores() -> None:
    _, v2 = _runs()
    s = v2.scores[ROBUST_DISTANCE]
    valid = np.isfinite(s)
    assert valid.any() and ISOLATION_FOREST not in v2.scores
    window = CFG.pvalues.window_weeks * 7 * 288
    for t in np.flatnonzero(valid)[:: max(1, int(valid.sum()) // 25)]:
        past = s[max(0, t - window) : t]
        past = past[np.isfinite(past)]
        assert v2.n_window[ROBUST_DISTANCE][t] == len(past)
        expected = (1 + np.sum(past >= s[t])) / (1 + len(past))
        assert v2.pvalues[ROBUST_DISTANCE][t] == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------------- injection


@pytest.mark.parametrize(
    ("label", "size"), [(PRICE_LEVEL, 15000.0), (PRICE_LEVEL, -1000.0), (ROBUST_DISTANCE, 1000.0)]
)
def test_v2_injected_scores_match_a_full_recompute(label: str, size: float) -> None:
    """As v1's harness test: D4 recomputed in full; D6 as recomputed features under the
    unmodified month models (a recompute would refit later months on earlier injections)."""
    _, run = _runs()
    recipe = run.recipe
    specs = {s.label: s for s in injection_specs(CFG, recipe)}
    spec = specs[label]
    starts = _starts(PRICE_LEVEL if label == PRICE_LEVEL else ISOLATION_FOREST)
    assert len(starts) >= 3
    series, _ = base_inputs()
    harness = injected_scores(spec, size, starts, series, run, CFG)
    modified, band = _modified(
        PRICE_LEVEL if label == PRICE_LEVEL else ISOLATION_FOREST, size, starts
    )
    scores, state = compute_scores(modified, band, CFG, recipe=recipe)
    rows = starts[:, None] + np.arange(spec.span)[None, :]
    if label == ROBUST_DISTANCE:
        months = run.state.lay.nem_month[rows]
        expected = np.full(rows.shape, np.nan)
        for m in np.unique(months):
            sel = months == m
            expected[sel] = run.state.distances[int(m)].score(state.features[rows][sel])
    else:
        expected = scores[label][rows]
    np.testing.assert_allclose(harness, expected, rtol=1e-12, atol=1e-12)
