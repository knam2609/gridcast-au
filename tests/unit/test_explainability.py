"""SHAP core and criterion E1's gate (m4_plan.md §2.6, step 3a).

The end-to-end tests run a small synthetic M2 walk-forward (one LightGBM C spec per target
form), store its boosters as M2 does, and check that the explainer reloads them, rebuilds
the feature rows and reproduces the stored float32 forecasts bit for bit. Every forecast
decomposes into anchor + base value + Σ contributions.
"""

from __future__ import annotations

import dataclasses
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.config import AppConfig, LightGBMConfig, load_config
from gridcast.explainability.families import families
from gridcast.explainability.global_run import (
    ANCHOR_TERMS,
    anchor_from_features,
    explain_fit,
    shap_col,
    whole_forecast,
    x_col,
)
from gridcast.explainability.pairs import ModelCache, explain_pairs
from gridcast.explainability.sample import (
    Period,
    eligible_pairs,
    feature_rows,
    origin_positions,
    sample_pairs,
)
from gridcast.explainability.stored import (
    StoredFit,
    fits_from_records,
    locate,
    parse_band,
)
from gridcast.explainability.treeshap import GateError, explain, gate, require
from gridcast.features import catalogue
from gridcast.features.builder import FeatureBuilder
from gridcast.forecasting import walkforward as wf
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.forecasting.origins import origin_positions as op
from gridcast.models.baselines import DAY_STEPS
from gridcast.models.ml import TARGET_FORMS, LightGBMModel
from tests.feature_fixtures import hourly_weather, weather_inputs
from tests.forecast_fixtures import make_ctx

SETUP = ForecastSetup()
RETRAINS = ("2024-04-01", "2024-04-15")
EVAL_END = "2024-04-22"


def _cfg() -> AppConfig:
    base = load_config()
    ml = base.ml.model_copy(
        update={
            "early_stopping_weeks": 1,
            "lightgbm": LightGBMConfig(
                learning_rate=0.2,
                min_data_in_leaf=50,
                max_rounds=25,
                early_stopping_rounds=5,
                num_threads=2,
            ),
        }
    )
    return base.model_copy(update={"ml": ml})


def _demand(pos: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(11)
    return 5000 + 800 * np.sin(2 * np.pi * pos / DAY_STEPS) + rng.normal(0, 120, len(pos))


@cache
def _builder() -> FeatureBuilder:
    ctx = make_ctx(start="2024-03-01 00:05", days=60, values=_demand)
    fc, obs = hourly_weather(ctx.times, seed=5)
    return FeatureBuilder(ctx, SETUP, weather_inputs(ctx.times, fc, obs))


def _spec(form: str) -> wf.ModelSpec:
    setting = {"target_form": form, "objective": "l1", "num_leaves": 7}
    return wf.ModelSpec(f"lgbm_C_{form}", "C", "lightgbm", candidates=(setting,))


@dataclasses.dataclass
class _Run:
    root: Path
    result: wf.WalkForwardResult
    times: pd.DatetimeIndex


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> _Run:
    root = tmp_path_factory.mktemp("data")
    builder = _builder()
    retrains = [nem_midnight_utc(d) for d in RETRAINS]
    origins = op(builder.ctx, SETUP, retrains[0])
    origins = origins[builder.ctx.times[origins] < nem_midnight_utc(EVAL_END)]
    result = wf.run_walk_forward(
        builder,
        [_spec(f) for f in TARGET_FORMS],
        origins,
        retrains,
        _cfg().ml,
        model_dir=root / "models",
        data_root=root,
    )
    return _Run(root, result, pd.DatetimeIndex(builder.ctx.times[origins]))


def _fits(run: _Run, form: str) -> list[StoredFit]:
    records = [dataclasses.asdict(r) for r in run.result.retrains[f"lgbm_C_{form}"]]
    return fits_from_records(records, run.root, _cfg().ml.lead_bands)


# --------------------------------------------------------------------------- families


def test_variant_c_has_the_six_planned_families() -> None:
    fam = families("C")
    assert fam.keys == (
        "lead",
        "calendar",
        "calendar_yesterday",
        "demand_now",
        "demand_seasonal",
        "weather",
    )
    assert len(fam.index) == len(fam.columns) == 26
    assert fam.of("wx_temp_sq") == "weather" and fam.of("dem_now") == "demand_now"
    c = np.random.default_rng(0).normal(size=(5, 26))
    np.testing.assert_allclose(fam.sums(c).sum(axis=1), c.sum(axis=1))
    with pytest.raises(ValueError):
        fam.sums(c[:, :-1])


# --------------------------------------------------------------------------- stored fits


def test_band_keys_round_trip() -> None:
    assert parse_band("h007-024") == (7, 24)
    for bad in ("h7-24", "x007-024", "h007_024"):
        with pytest.raises(ValueError):
            parse_band(bad)


def _record(retrain: str, band: str) -> dict[str, object]:
    return {
        "retrain": retrain,
        "band": band,
        "chosen": {"target_form": "latest", "objective": "l1", "num_leaves": 31},
        "params": {"n_trees": 10},
        "path": f"models/x/{retrain}_{band}.txt.gz",
    }


BANDS = [[1, 6], [7, 24], [25, 72], [73, 288]]
KEYS = ("h001-006", "h007-024", "h025-072", "h073-288")


def test_fits_chain_folds_and_require_every_band(tmp_path: Path) -> None:
    records = [_record(d, b) for d in ("2023-02-01", "2022-11-01") for b in KEYS]
    fits = fits_from_records(records, tmp_path, BANDS)
    assert [f.retrain for f in fits[:4]] == ["2022-11-01"] * 4
    assert fits[0].fold_start == nem_midnight_utc("2022-11-01")
    assert fits[0].fold_end == nem_midnight_utc("2023-02-01")
    assert fits[-1].fold_end is None
    assert fits[0].path == tmp_path / "models/x/2022-11-01_h001-006.txt.gz"
    with pytest.raises(ValueError, match="one fit per retrain"):
        fits_from_records(records[:-1], tmp_path, BANDS)
    with pytest.raises(ValueError, match="one fit per retrain"):
        fits_from_records([*records, records[0]], tmp_path, BANDS)


def test_locate_sends_each_pair_to_its_fold_and_band(tmp_path: Path) -> None:
    records = [_record(d, b) for d in ("2022-11-01", "2023-02-01") for b in KEYS]
    fits = fits_from_records(records, tmp_path, BANDS)
    t = pd.DatetimeIndex(
        [
            nem_midnight_utc("2022-12-01"),
            nem_midnight_utc("2023-01-31") + pd.Timedelta(hours=23, minutes=30),
            nem_midnight_utc("2023-02-01"),
            nem_midnight_utc("2022-10-31"),
        ]
    )
    where = locate(fits, t, np.array([1, 288, 7, 1]))
    assert [fits[k].key for k in where[:3]] == [
        "2022-11-01_h001-006",
        "2022-11-01_h073-288",
        "2023-02-01_h007-024",
    ]
    assert where[3] == -1  # before the first retrain


# --------------------------------------------------------------------------- sampling


def test_sample_is_seeded_within_fold_band_period_and_mask(run: _Run) -> None:
    fit = next(f for f in _fits(run, "latest") if f.band == (7, 24))
    mask = np.isfinite(run.result.predictions["lgbm_C_latest"])
    period = Period("t", run.times[0], nem_midnight_utc("2024-04-10"))
    pool = eligible_pairs(fit, run.times, mask, period)
    a = sample_pairs(fit, run.times, mask, period, 300, seed=7)
    b = sample_pairs(fit, run.times, mask, period, 300, seed=7)
    c = sample_pairs(fit, run.times, mask, period, 300, seed=8)
    assert len(pool) > 300 and len(a) == 300
    np.testing.assert_array_equal(a.rows, b.rows)
    np.testing.assert_array_equal(a.leads, b.leads)
    assert not np.array_equal(a.rows * 1000 + a.leads, c.rows * 1000 + c.leads)
    keys = a.rows * 1000 + a.leads
    assert len(np.unique(keys)) == len(keys) and np.all(np.diff(keys) > 0)
    issued = run.times[a.rows]
    assert (issued >= fit.fold_start).all() and (issued < period.end).all()
    assert ((a.leads >= 7) & (a.leads <= 24)).all()
    assert mask[a.rows, a.leads - 1].all()
    everything = sample_pairs(fit, run.times, mask, period, 10**7, seed=7)
    np.testing.assert_array_equal(everything.rows, pool.rows)


def test_feature_rows_equal_the_full_build() -> None:
    builder = _builder()
    origins = op(builder.ctx, SETUP, nem_midnight_utc("2024-04-02"))[:5]
    full = builder.build("C", origins).X.to_numpy(dtype=np.float32, na_value=np.nan)
    rng = np.random.default_rng(1)
    pick = rng.choice(len(full), size=200, replace=False)
    o = np.repeat(origins, SETUP.horizon_steps)[pick]
    h = np.tile(SETUP.horizons, len(origins))[pick]
    np.testing.assert_array_equal(feature_rows(builder, "C", o, h), full[pick])


# --------------------------------------------------------------------------- E1 end to end


@pytest.mark.parametrize("form", TARGET_FORMS)
def test_stored_boosters_reproduce_forecasts_and_decompose(run: _Run, form: str) -> None:
    cfg = _cfg()
    fits = _fits(run, form)
    stored = run.result.predictions[f"lgbm_C_{form}"]
    rows, cols = np.nonzero(np.isfinite(stored))
    take = np.random.default_rng(2).choice(len(rows), size=600, replace=False)
    rows, leads = rows[take], cols[take] + 1
    res = explain_pairs(
        _builder(), fits, run.times[rows], leads, stored[rows, leads - 1], cfg, ModelCache(cfg)
    )
    assert res.gate.passed, res.gate
    assert res.gate.n == 600 and res.gate.forecast_mismatches == 0
    assert set(res.gate_by_fit) == {fits[k].key for k in np.unique(res.fit_index)}
    ex = res.explanation
    assert ex is not None and ex.target_form == form
    np.testing.assert_allclose(
        ex.anchor + ex.base + ex.contributions.sum(axis=1), ex.forecast, rtol=0, atol=1e-6
    )
    if form == "direct":
        assert np.all(ex.anchor == 0)
    else:
        assert np.all(ex.anchor != 0)


def test_gate_catches_a_one_ulp_difference_and_withholds(run: _Run) -> None:
    cfg = _cfg()
    fits = _fits(run, "residual")
    stored = run.result.predictions["lgbm_C_residual"]
    rows, cols = np.nonzero(np.isfinite(stored))
    rows, leads = rows[:50], cols[:50] + 1
    values = stored[rows, leads - 1].copy()
    values[3] = np.nextafter(values[3], np.float32(np.inf))
    res = explain_pairs(_builder(), fits, run.times[rows], leads, values, cfg)
    assert not res.gate.passed and res.gate.forecast_mismatches == 1
    assert 0 < res.gate.max_forecast_diff_mw < 0.01
    with pytest.raises(GateError, match="withheld"):
        require(res.gate, "test")


def test_gate_checks_the_shap_sum_and_float32_storage(run: _Run) -> None:
    cfg = _cfg()
    fit = _fits(run, "anchored")[0]
    model = ModelCache(cfg).get(fit)
    builder = _builder()
    origins = origin_positions(builder, run.times[:20])
    X = feature_rows(builder, "C", origins, np.full(20, fit.band[0]))
    keep = np.isfinite(X).all(axis=1)
    ex = explain(model, X[keep])
    stored = ex.forecast.astype(np.float32)
    assert gate(ex, stored, 1e-6).passed
    ex.contributions[0, 0] += 1e-3  # a SHAP sum that no longer adds up
    broken = gate(ex, stored, 1e-6)
    assert broken.sum_failures == 1 and not broken.passed
    with pytest.raises(ValueError, match="float32"):
        gate(ex, ex.forecast + 1e-9, 1e-6)  # float64 values M2 could not have stored
    with pytest.raises(ValueError, match="float32 rows"):
        explain(model, X[keep].astype(np.float64))


def test_a_pair_without_a_fit_is_refused(run: _Run) -> None:
    cfg = _cfg()
    fits = _fits(run, "latest")
    early = pd.DatetimeIndex([nem_midnight_utc("2024-03-20")])
    with pytest.raises(LookupError, match="no stored fit"):
        explain_pairs(_builder(), fits, early, np.array([1]), np.array([5000.0], np.float32), cfg)


def test_folds_are_clipped_to_the_batch_period(tmp_path: Path) -> None:
    # holdout_main's first retrain also forecast development issue times (Aug-Sep 2025).
    records = [_record(d, b) for d in ("2025-08-01", "2025-11-01") for b in KEYS]
    period = (nem_midnight_utc("2025-10-01"), nem_midnight_utc("2026-10-01"))
    fits = fits_from_records(records, tmp_path, BANDS, period)
    assert fits[0].fold_start == period[0] and fits[0].fold_end == nem_midnight_utc("2025-11-01")
    assert fits[-1].fold_end == period[1]
    sep = pd.DatetimeIndex([nem_midnight_utc("2025-09-15")])
    assert locate(fits, sep, np.array([1]))[0] == -1


# --------------------------------------------------------------------------- global sample (step 5)


def _fit_pairs(run: _Run, form: str, n: int = 300) -> tuple[StoredFit, np.ndarray, np.ndarray]:
    """``n`` stored pairs of the first fit of ``form`` (rows of the run, leads)."""
    fit = _fits(run, form)[0]
    stored = run.result.predictions[f"lgbm_C_{form}"]
    rows, cols = np.nonzero(np.isfinite(stored))
    leads = cols + 1
    inside = fit.covers(run.times[rows], leads)
    rows, leads = rows[inside], leads[inside]
    take = np.sort(np.random.default_rng(4).choice(len(rows), size=n, replace=False))
    return fit, rows[take], leads[take]


@pytest.mark.parametrize("form", TARGET_FORMS)
def test_explain_fit_stores_pairs_that_decompose_the_whole_forecast(run: _Run, form: str) -> None:
    cfg = _cfg()
    fit, rows, leads = _fit_pairs(run, form)
    stored = run.result.predictions[f"lgbm_C_{form}"][rows, leads - 1]
    actual = np.linspace(4000, 6000, len(rows))
    model = ModelCache(cfg).get(fit)
    s = explain_fit(fit, model, _builder(), "C", run.times[rows], leads, stored, actual, 1e-6, 999)
    assert s.gate.passed and s.gate.n == len(rows) and s.eligible == 999
    f = s.frame
    columns = list(model.columns)
    assert len(f) == len(rows) and np.array_equal(f["lead"], leads)
    assert (f["target_end"] - f["issue_time"] == pd.to_timedelta(leads * 5, unit="min")).all()
    np.testing.assert_array_equal(f["stored"], stored.astype(np.float64))
    np.testing.assert_array_equal(f["actual"], actual)
    np.testing.assert_array_equal(f["forecast"].astype(np.float32), stored)
    np.testing.assert_array_equal(anchor_from_features(f, form), f["anchor"])
    assert all(f[x_col(c)].dtype == np.float32 for c in columns)
    assert s.gain.shape == (len(columns),) and s.gain.sum() > 0
    raw = f[[shap_col(c) for c in columns]].to_numpy()
    np.testing.assert_allclose(f["base"] + raw.sum(axis=1), f["raw"], rtol=0, atol=1e-6)

    whole = whole_forecast(f, form, columns)
    np.testing.assert_allclose(
        whole.constant + whole.contributions.sum(axis=1), f["forecast"], rtol=0, atol=1e-6
    )
    terms = ANCHOR_TERMS[form]
    others = [j for j, c in enumerate(columns) if c not in terms]
    np.testing.assert_array_equal(whole.contributions[:, others], raw[:, others])
    for term in terms:  # the anchor's share has mean zero over the fit's sample
        j = columns.index(term)
        np.testing.assert_allclose((whole.contributions[:, j] - raw[:, j]).mean(), 0, atol=1e-6)
    if form == "direct":
        np.testing.assert_array_equal(whole.constant, f["base"])
    else:
        np.testing.assert_allclose(whole.constant, f["base"] + f["anchor"].mean())


def test_explain_fit_refuses_a_pair_outside_its_fold_or_band(run: _Run) -> None:
    cfg = _cfg()
    fit, rows, leads = _fit_pairs(run, "latest", n=20)
    model = ModelCache(cfg).get(fit)
    stored = run.result.predictions["lgbm_C_latest"][rows, leads - 1]
    out_of_band = leads.copy()
    out_of_band[0] = fit.band[1] + 1 if fit.band[1] < SETUP.horizon_steps else fit.band[0] - 1
    with pytest.raises(ValueError, match="outside its fold or band"):
        explain_fit(
            fit,
            model,
            _builder(),
            "C",
            run.times[rows],
            out_of_band,
            stored,
            np.zeros(20),
            1e-6,
            20,
        )


def test_anchor_terms_are_m2_baselines() -> None:
    columns = catalogue.feature_names("C")
    X = np.random.default_rng(3).normal(1000, 300, size=(50, len(columns))).astype(np.float32)
    frame = pd.DataFrame({x_col(c): X[:, j] for j, c in enumerate(columns)})
    for form in TARGET_FORMS:
        model = LightGBMModel(columns, form, "l1", 7, _cfg().ml.lightgbm, 0)
        np.testing.assert_array_equal(anchor_from_features(frame, form), model.baseline(X))


def test_a_replaced_global_sample_keeps_its_run_history(tmp_path: Path) -> None:
    """explain-dev/-holdout reruns are allowed with replace; the manifest lists the runs
    they superseded, oldest first (m4_plan.md §1, "Clarified in step 6")."""
    import json

    from gridcast.explainability.global_run import MANIFEST, run_history

    assert run_history(tmp_path) == []
    first = {"commit": "aaa", "created_at": "t1", "gate": {"n": 1}, "seconds": 2.0, "fits": []}
    (tmp_path / MANIFEST).write_text(json.dumps(first))
    assert run_history(tmp_path) == [
        {"commit": "aaa", "created_at": "t1", "gate": {"n": 1}, "seconds": 2.0}
    ]
    second = {**first, "commit": "bbb", "created_at": "t2", "previous_runs": run_history(tmp_path)}
    (tmp_path / MANIFEST).write_text(json.dumps(second))
    assert [r["commit"] for r in run_history(tmp_path)] == ["aaa", "bbb"]


def test_the_global_run_knows_only_the_two_splits() -> None:
    from gridcast.explainability.global_run import run_global

    with pytest.raises(ValueError, match="unknown split"):
        run_global(None, load_config(), "VIC1", "smoke")  # type: ignore[arg-type]
