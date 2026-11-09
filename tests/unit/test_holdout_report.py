"""Holdout report computations that need no database: the bias-by-month diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluation.holdout_report import _bias_section, monthly_bias
from gridcast.evaluation.m2_report import Pairs


def _pairs() -> Pairs:
    # Issue times straddle a NEM-time month boundary: 31 Jan 23:30 and 1 Feb 00:00 NEM
    # are 31 Jan 13:30 and 14:00 UTC, so the months must be taken in NEM time, not UTC.
    times = pd.DatetimeIndex(
        ["2026-01-31 13:00", "2026-01-31 13:30", "2026-01-31 14:00", "2026-01-31 14:30"],
        tz="UTC",
    )
    actual = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, np.nan]])
    c = actual + np.array([[-10.0, -10.0], [-30.0, -30.0], [20.0, 0.0], [0.0, 0.0]])
    naive = actual + 5.0
    naive[1, 0] = np.nan  # not a common pair: excluded for every model
    predictions = {"lgbm_C": c.astype(np.float32), "daily_naive": naive.astype(np.float32)}
    mask = np.isfinite(actual) & np.isfinite(c) & np.isfinite(naive)
    return Pairs(times, np.arange(4), predictions, actual, mask)


def test_monthly_bias_groups_by_nem_month_on_common_pairs() -> None:
    out = monthly_bias(_pairs(), ["lgbm_C", "daily_naive"])
    assert list(out.index) == ["2026-01", "2026-02"]
    jan, feb = out.loc["2026-01"], out.loc["2026-02"]
    # January: pairs (0,0), (0,1), (1,1); (1,0) is not common.
    assert jan["n"] == 3
    assert jan["actual"] == pytest.approx((100 + 200 + 400) / 3)
    assert jan["bias_lgbm_C"] == pytest.approx((-10 - 10 - 30) / 3)
    assert jan["mae_lgbm_C"] == pytest.approx((10 + 10 + 30) / 3)
    assert jan["bias_daily_naive"] == pytest.approx(5.0)
    # February: (2,0), (2,1), (3,0); (3,1) has no actual.
    assert feb["n"] == 3
    assert feb["bias_lgbm_C"] == pytest.approx(20 / 3)
    assert feb["mae_lgbm_C"] == pytest.approx(20 / 3)


def _months(values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            k: {"n": 10.0, "actual": a, "bias_lgbm_C": b, "mae_lgbm_C": abs(b) + 1}
            for k, (a, b) in values.items()
        }
    ).T


def test_bias_section_names_the_worst_and_highest_months() -> None:
    dev = _months({"2025-06": (5000.0, -50.0), "2025-07": (5100.0, -80.0)})
    hold = _months({"2026-06": (5300.0, -200.0), "2026-07": (5200.0, -100.0)})
    text = "\n".join(_bias_section(dev, hold, ["lgbm_C"]))
    assert "(diagnostic, added after scoring)" in text
    assert "| Jun 2025 | development |" in text and "| Jul 2026 | holdout |" in text
    assert "most negative monthly bias in the holdout is Jun 2026 (-200.0 MW)" in text
    assert "its most negative was Jul 2025 (-80.0 MW)" in text
    assert "holdout month is Jun 2026 (5,300.0 MW)" in text
    assert "development month, Jul 2025 (5,100.0 MW)" in text
    assert "higher than every development month" in text

    lower = _months({"2026-06": (4000.0, -10.0)})
    text = "\n".join(_bias_section(dev, lower, ["lgbm_C"]))
    assert "higher than every development month" not in text
    assert "| Jun 2025 |" not in text  # only as many development months as holdout months
