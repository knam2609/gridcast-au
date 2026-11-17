"""Detector pieces D1–D6 and the lines (m4_plan.md §2.3)."""

import numpy as np
import pandas as pd
import pytest

from gridcast.anomalies.detectors import (
    DETECTORS,
    ISOLATION_FOREST,
    RESIDUAL_DAYAHEAD,
    RESIDUAL_SHORT,
    band_at_targets,
    curtailment,
    log_ratio,
    price_x,
    residual_score,
    volatility,
    volatility_baseline,
)
from gridcast.anomalies.isolation import month_start_utc
from gridcast.anomalies.run import LINE_NOTABLE, LINE_SEVERE, alert_tiers, detect, run_detectors
from gridcast.anomalies.series import QuantileBand
from gridcast.timeutils import INTERVAL, NEM_TZ
from tests.anomaly_fixtures import base_inputs, small_config, split


def test_price_x_is_signed_and_compressive() -> None:
    x = price_x(np.array([-1000.0, 0.0, 10.0, 15000.0]), 10.0)
    assert x[1] == 0 and x[0] < 0
    np.testing.assert_allclose(x[0], -price_x(np.array([1000.0]), 10.0)[0])
    np.testing.assert_allclose(x[2], np.arcsinh(1.0))
    assert x[3] < 10


def test_volatility_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    v = volatility(x, 12)
    for t in (12, 50, 199):
        dx = np.diff(x[t - 12 : t + 1])
        np.testing.assert_allclose(v[t], np.sqrt(np.mean(dx**2)))
    assert np.isnan(v[:12]).all()


def test_volatility_baseline_window_and_lag() -> None:
    v = np.arange(100.0)
    b = volatility_baseline(v, 10, 12)
    # b[t] = median of v[t-12-9 .. t-12]
    np.testing.assert_allclose(b[30], np.median(v[9:19]))
    assert np.isnan(b[: 12 + 9]).all()
    v[50] = np.nan
    b = volatility_baseline(v, 10, 12)
    assert np.isnan(b[62]) and np.isnan(b[71]) and np.isfinite(b[72])


def test_log_ratio_is_two_sided() -> None:
    r = log_ratio(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.01)
    assert r[0] < 0 < r[1]
    np.testing.assert_allclose(abs(r[0]), r[1])


def test_curtailment_against_uigf() -> None:
    share = curtailment(
        np.array([50.0, 80.0, 0.0, 120.0, np.nan]),
        np.array([99.0, 100.0, 200.0, 100.0, 300.0]),
        100.0,
    )
    np.testing.assert_allclose(share[:4], [0.0, 0.2, 1.0, -0.2])
    assert np.isnan(share[4])


def test_residual_score() -> None:
    s = residual_score(
        np.array([100.0, 150.0, 260.0, 10.0]),
        np.array([100.0, 100.0, 100.0, 0.0]),
        np.array([200.0, 200.0, 200.0, 20.0]),
        50.0,
    )
    np.testing.assert_allclose(s, [0.0, -0.5, 0.6, -0.2])


def _band(issues: pd.DatetimeIndex, n_leads: int) -> QuantileBand:
    m = len(issues)
    # lower = issue row, upper = lead, so the picked (issue, lead) can be read back.
    lower = np.repeat(np.arange(m, dtype=float)[:, None], n_leads, axis=1)
    upper = np.repeat(np.arange(1, n_leads + 1, dtype=float)[None, :], m, axis=0)
    return QuantileBand(issues, lower, upper, np.ones((m, n_leads), dtype=bool))


@pytest.mark.parametrize("leads", [(1, 6), (283, 288)])
def test_band_at_targets_picks_the_one_issue_time_in_range(leads: tuple[int, int]) -> None:
    issues = pd.date_range("2024-01-01 00:00", periods=200, freq="30min", tz="UTC")
    ends = pd.date_range("2024-01-02 00:05", periods=400, freq="5min", tz="UTC")
    lower, upper, ok = band_at_targets(_band(issues, 288), ends, leads)
    assert ok.all()
    for k, t in enumerate(ends):
        o = issues[int(lower[k])]
        lead = int(upper[k])
        assert t == o + lead * INTERVAL
        assert leads[0] <= lead <= leads[1]
        # The latest such issue time: one 30 min later would be outside the leads.
        assert t - (o + pd.Timedelta(minutes=30)) < leads[0] * INTERVAL


def test_band_at_targets_missing_issue_or_mask() -> None:
    issues = pd.date_range("2024-01-01 00:00", periods=10, freq="30min", tz="UTC")
    band = _band(issues, 6)
    band.valid[2, :] = False
    ends = pd.date_range("2024-01-01 00:05", periods=80, freq="5min", tz="UTC")
    _, _, ok = band_at_targets(band, ends, (1, 6))
    assert ok[:12].all() and not ok[12:18].any() and ok[18:60].all() and not ok[60:].any()


def test_scores_and_validity_on_synthetic_data() -> None:
    series, band = base_inputs()
    run = run_detectors(series, band, small_config())
    for d in DETECTORS:
        s, p = run.scores[d], run.pvalues[d]
        assert np.isfinite(s).sum() > len(series) // 2, d
        np.testing.assert_array_equal(np.isfinite(s), np.isfinite(p), err_msg=d)
        assert np.nanmin(p) > 0 and np.nanmax(p) <= 1
    # D2/D3 only where M2's mask allows.
    for d in (RESIDUAL_SHORT, RESIDUAL_DAYAHEAD):
        ok = run.state.bands[d][2]
        np.testing.assert_array_equal(np.isfinite(run.scores[d]), ok)


def test_isolation_forest_models_saw_only_the_past() -> None:
    series, band = base_inputs()
    run = run_detectors(series, band, small_config())
    models = run.state.models
    assert len(models) == 2
    for month, mm in models.items():
        assert mm.start == month_start_utc(month)
        assert mm.train_last_end <= mm.start
        assert mm.train_first_end > mm.start - pd.Timedelta(weeks=1)
        scored = np.flatnonzero(np.isfinite(run.scores[ISOLATION_FOREST]))
        in_month = scored[run.state.lay.nem_month[scored] == month]
        assert (series.ends[in_month] > mm.start).all()
    # NEM month by interval start: the interval ending 00:00 on the 1st is the old month.
    first_of_month = pd.Timestamp("2024-04-01", tz=NEM_TZ).tz_convert("UTC")
    k = series.ends.get_loc(first_of_month)
    assert run.state.lay.nem_month[k] + 1 == run.state.lay.nem_month[k + 1]


def test_price_lines_and_scored_periods() -> None:
    series, band = base_inputs()
    cfg = small_config()
    price = series.price.copy()
    sp = split()
    masks = sp.masks(series.ends)
    first_scored = int(np.flatnonzero(masks[LINE_NOTABLE])[0])
    price[first_scored - 1] = 20_000  # warm-up: no alert
    price[first_scored + 5] = 300  # exactly at the line
    price[first_scored + 6] = 5_000
    s2 = series.with_values(price=price)
    det = detect(s2, band, sp, cfg)
    t = alert_tiers(det.run, price, sp, cfg)
    assert t[LINE_NOTABLE][first_scored - 1] == 0 and t[LINE_SEVERE][first_scored - 1] == 0
    assert t[LINE_NOTABLE][first_scored + 5] == 1 and t[LINE_SEVERE][first_scored + 5] == 0
    assert t[LINE_NOTABLE][first_scored + 6] == 1 and t[LINE_SEVERE][first_scored + 6] == 2
    for d in DETECTORS:
        assert not t[d][~masks[d]].any()
    residual_first = int(np.flatnonzero(masks[RESIDUAL_SHORT])[0])
    assert residual_first > first_scored


def test_check_band_actuals() -> None:
    from gridcast.anomalies.inputs import check_band_actuals
    from gridcast.anomalies.series import epoch_minutes

    series, band = base_inputs()
    minutes = epoch_minutes(series.ends)
    idx = (
        epoch_minutes(band.issue_times)[:, None]
        + 5 * np.arange(1, band.n_leads + 1)[None, :]
        - minutes[0]
    ) // 5
    inside = (idx >= 0) & (idx < len(series))
    actual = np.full(idx.shape, np.nan)
    actual[inside] = series.demand[idx[inside]]
    n = check_band_actuals(band, actual, series)
    assert n == int((band.valid & inside).sum()) > 0
    actual[np.argwhere(band.valid & inside)[0][0], np.argwhere(band.valid & inside)[0][1]] += 1
    with pytest.raises(ValueError, match="differ from market demand"):
        check_band_actuals(band, actual, series)
