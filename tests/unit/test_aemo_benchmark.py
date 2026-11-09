import numpy as np
import pandas as pd
import pytest

from gridcast.evaluation.aemo_benchmark import (
    aligned,
    lead_runs,
    mae_difference,
    strict_runs,
    thirty_minute_means,
)

T0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")


def _t(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def test_thirty_minute_means_blocks_of_six_and_nan_propagates() -> None:
    wide = np.arange(2 * 288, dtype=float).reshape(2, 288)
    wide[1, 7] = np.nan
    out = thirty_minute_means(wide)
    assert out.shape == (2, 48)
    assert out[0, 0] == pytest.approx(np.mean(np.arange(6)))  # h = 1..6
    assert out[0, 1] == pytest.approx(np.mean(np.arange(6, 12)))
    assert np.isnan(out[1, 1]) and np.isfinite(out[1, 0])
    with pytest.raises(ValueError):
        thirty_minute_means(np.zeros((1, 12)))


def test_strict_runs_latest_published_by_issue_time() -> None:
    runs = pd.DataFrame(
        {
            "run_time": [_t(0), _t(30), _t(60), _t(90)],
            # The 60-min run is published late (after the 90-min run is due).
            "available_at": [_t(2), _t(32), _t(100), _t(92)],
        }
    )
    issue = pd.DatetimeIndex([_t(0), _t(30), _t(60), _t(90), _t(120)])
    got = strict_runs(runs, issue)
    assert pd.isna(got[0])  # nothing published at or before 00:00
    assert list(got[1:]) == [_t(0), _t(30), _t(30), _t(90)]


def test_aligned_looks_up_run_and_target_and_leaves_gaps() -> None:
    fc = pd.DataFrame(
        {
            "run_time": [_t(0), _t(0), _t(30)],
            "target_end": [_t(30), _t(60), _t(60)],
            "total_demand_mw": [1.0, 2.0, 3.0],
        }
    )
    issue = pd.DatetimeIndex([_t(0), _t(30)])
    matched = aligned(fc, issue, issue, 30, 2)
    np.testing.assert_array_equal(matched, [[1.0, 2.0], [3.0, np.nan]])
    # Strict: the issue time 30 uses the run from 0, whose period ending 60 is "lead 1".
    strict = aligned(fc, issue, pd.DatetimeIndex([pd.NaT, _t(0)]), 30, 2)
    np.testing.assert_array_equal(strict, [[np.nan, np.nan], [2.0, np.nan]])


def test_aligned_rejects_duplicate_keys() -> None:
    fc = pd.DataFrame(
        {"run_time": [_t(0)] * 2, "target_end": [_t(30)] * 2, "total_demand_mw": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        aligned(fc, pd.DatetimeIndex([_t(0)]), pd.DatetimeIndex([_t(0)]), 30, 1)


def test_mae_difference_point_estimate_and_interval() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, (40, 3))
    fa = actual + 1.0  # MAE exactly 1
    fb = actual + 3.0  # MAE exactly 3
    mask = np.ones_like(actual, dtype=bool)
    mask[0, 0] = False
    days = np.repeat(np.arange(10), 4).astype(str)
    d = mae_difference("a", "b", fa, fb, actual, mask, days, seed=1, draws=200)
    assert (d.mae_a, d.mae_b, d.diff) == pytest.approx((1.0, 3.0, -2.0))
    assert d.lo == pytest.approx(-2.0) and d.hi == pytest.approx(-2.0)
    assert d.n_pairs == 119 and d.n_days == 10


def test_lead_runs_groups_contiguous_leads() -> None:
    assert lead_runs([True, True, False, True], [1, 2, 3, 4]) == [(1, 2), (4, 4)]
    assert lead_runs([False, False], [1, 2]) == []


def test_window_development_and_holdout() -> None:
    from gridcast.config import load_config
    from gridcast.evaluation.aemo_benchmark import window

    cfg = load_config()
    dev_first, dev_end = window(cfg)
    hold_first, hold_end = window(cfg, "holdout")
    assert dev_first == pd.Timestamp("2024-09-30 14:00", tz="UTC")  # 1 Oct 2024 NEM time
    assert dev_end == hold_first == pd.Timestamp("2025-09-30 14:00", tz="UTC")
    assert hold_end == pd.Timestamp("2026-09-30 14:00", tz="UTC")  # exclusive
    with pytest.raises(ValueError, match="period"):
        window(cfg, "both")
