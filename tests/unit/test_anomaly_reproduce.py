"""The reproduction check that carries v1's code freeze (m4_plan.md §1): it must see any
difference, and only real ones."""

import numpy as np
import pandas as pd

from gridcast.anomalies.detectors import DETECTORS, ISOLATION_FOREST, ROLLING, v1_recipe
from gridcast.anomalies.reproduce import (
    EVENT_COLUMNS,
    events_check,
    frame_check,
    record_check,
)
from gridcast.config import load_config


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_end": pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC"),
            "d__score": [1.0, np.nan, 3.0, 4.0],
            "d__tier": np.array([0, 1, 0, 2], dtype=np.int8),
        }
    )


def test_identical_frames_pass_with_nan_equal_to_nan() -> None:
    assert frame_check("f", _frame(), _frame()).ok


def test_one_changed_value_is_named() -> None:
    got = _frame()
    got.loc[2, "d__score"] = 3.0000000000000004
    check = frame_check("f", got, _frame())
    assert not check.ok
    assert check.differences == ("d__score: 1 rows differ (first at row 2)",)


def test_a_nan_that_became_a_value_differs() -> None:
    got = _frame()
    got.loc[1, "d__score"] = 0.0
    assert not frame_check("f", got, _frame()).ok


def test_dtype_columns_and_length_differences() -> None:
    got = _frame().astype({"d__tier": np.int16})
    assert "dtype" in frame_check("f", got, _frame()).differences[0]
    reordered = _frame()[["d__tier", "d__score", "interval_end"]]
    assert "columns" in frame_check("f", reordered, _frame()).differences[0]
    assert "rows" in frame_check("f", _frame().head(3), _frame()).differences[0]


def _events() -> pd.DataFrame:
    end = pd.Timestamp("2024-01-01 00:05", tz="UTC")
    row = {
        "first_interval_end": end,
        "last_interval_end": end,
        "peak_interval_end": end,
        "tier": "notable",
        "detectors": ["price_level"],
        "min_p": 1.9e-5,
        "categories": ["unusual price"],
        "lines_only": False,
        "evidence": {"a": {"value": 1.5, "nested": [1, None]}},
        "summary": "Notable event.",
    }
    lines = {**row, "detectors": ["price_line_300"], "min_p": None, "lines_only": True}
    return pd.DataFrame([row, lines], columns=list(EVENT_COLUMNS))


def test_events_identical_and_changed() -> None:
    assert events_check(_events(), _events()).ok
    got = _events()
    got["summary"] = ["Notable event.", "Notable event, changed."]
    got["evidence"] = [{"a": {"value": 1.5, "nested": [1, 2]}}, got["evidence"][1]]
    check = events_check(got, _events())
    assert [d.split(":")[0] for d in check.differences] == ["evidence", "summary"]
    assert not events_check(_events().head(1), _events()).ok


def test_run_record_ignores_timings_only() -> None:
    stored = {"split": "dev", "counts": {"events": 3}, "seconds": {"detect": 40.0}}
    same = {**stored, "seconds": {"detect": 55.0}}
    assert record_check(same, stored).ok
    changed = {**stored, "counts": {"events": 4}}
    assert not record_check(changed, stored).ok


def test_v1_recipe_is_the_frozen_recipe() -> None:
    cfg = load_config().anomalies
    r = v1_recipe(cfg)
    assert r.version == 1 and r.multivariate == ISOLATION_FOREST
    assert r.detectors == DETECTORS and r.rolling == ROLLING
    assert r.price_level_scale_aud == r.price_volatility_scale_aud == cfg.price_level.scale_aud
