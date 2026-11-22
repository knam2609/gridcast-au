"""The holdout run's development-part check (m4_plan.md §1, "Clarified in step 6").

On synthetic data: a "development run" on the series truncated at a boundary is stored (a
parquet round trip, as the real batch), then one continuous run over the whole series is
checked against it. Causality makes them equal; a one-value change in the stored batch, a
changed event or a changed month model must be caught.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from gridcast.anomalies.detectors import Recipe
from gridcast.anomalies.dev import forest_frame, interval_frame
from gridcast.anomalies.holdout import DEV_EVENT_COLUMNS, development_part
from gridcast.anomalies.report import StoredRun
from gridcast.anomalies.run import Detection, Split, detect
from gridcast.anomalies.series import MarketSeries, QuantileBand
from gridcast.config import AppConfig, load_config
from gridcast.forecasting.origins import nem_midnight_utc
from tests.anomaly_fixtures import START, base_inputs, small_config, small_v2_recipe, split

BOUNDARY = START + dt.timedelta(days=30)


def _cfg() -> AppConfig:
    cfg = load_config()
    cfg.anomalies = small_config()
    return cfg


def _dev() -> Split:
    return replace(split(), end_day=BOUNDARY)


def _holdout() -> Split:
    return Split("holdout", BOUNDARY, BOUNDARY, START + dt.timedelta(days=45))


def _stored(det: Detection, series: MarketSeries, tmp_path: Path) -> StoredRun:
    """The development run as the database and the parquet files return it."""
    interval_frame(det, _dev()).to_parquet(tmp_path / "intervals.parquet", index=False)
    forest_frame(det).to_parquet(tmp_path / "forests.parquet", index=False)
    ends = series.ends
    events = pd.DataFrame(
        [
            {
                "first_interval_end": ends[e.first],
                "last_interval_end": ends[e.last],
                "peak_interval_end": ends[e.peak],
                "tier": e.tier,
                "detectors": list(e.detectors),
                "min_p": e.min_p,
                "categories": list(e.categories),
                "lines_only": e.lines_only,
            }
            for e in det.events
        ],
        columns=list(DEV_EVENT_COLUMNS),
    )
    return StoredRun(
        run_id=1,
        batch_id="b",
        git_sha=None,
        config={},
        directory=tmp_path,
        intervals=pd.read_parquet(tmp_path / "intervals.parquet"),
        injections=pd.DataFrame(),
        forests=pd.read_parquet(tmp_path / "forests.parquet"),
        cuts=pd.DataFrame(),
        events=events,
    )


def _runs(
    recipe: Recipe | None, tmp_path: Path
) -> tuple[Detection, MarketSeries, QuantileBand, StoredRun]:
    series, band = base_inputs()
    a = small_config()
    last = nem_midnight_utc(str(BOUNDARY))
    n_dev = int((series.ends <= last).sum())
    short = series.head(n_dev)
    dev_det = detect(short, band.issued_until(last), _dev(), a, recipe=recipe)
    assert dev_det.events, "the synthetic development part must have events"
    stored = _stored(dev_det, short, tmp_path)
    full = detect(series, band, _holdout(), a, recipe=recipe)
    return full, series, band, stored


@pytest.mark.parametrize("version", [1, 2])
def test_the_continuous_run_reproduces_the_development_part(version: int, tmp_path: Path) -> None:
    recipe = None if version == 1 else small_v2_recipe()
    full, series, _, stored = _runs(recipe, tmp_path)
    checks = development_part(full, series, _dev(), stored, _cfg())
    assert [c.name for c in checks] == [
        "development intervals",
        "development month models",
        "development events",
    ]
    assert all(c.ok for c in checks), [c.differences for c in checks]
    assert checks[0].items == len(stored.intervals) < len(series)
    # The continuous run's holdout part has events of its own; none is compared.
    assert any(
        series.ends[e.first] > stored.intervals["interval_end"].iloc[-1] for e in full.events
    )


def test_a_one_value_change_in_the_stored_batch_is_caught(tmp_path: Path) -> None:
    full, series, _, stored = _runs(None, tmp_path)
    col = "demand_level__p"
    row = int(stored.intervals[col].notna().to_numpy().argmax())
    stored.intervals.loc[row, col] = stored.intervals.loc[row, col] * (1 + 1e-12)
    checks = development_part(full, series, _dev(), stored, _cfg())
    bad = [c for c in checks if not c.ok]
    assert [c.name for c in bad] == ["development intervals"]
    assert col in bad[0].differences[0]


def test_a_changed_event_or_month_model_is_caught(tmp_path: Path) -> None:
    full, series, _, stored = _runs(None, tmp_path)
    stored.events.loc[0, "tier"] = (
        "severe" if stored.events.loc[0, "tier"] != "severe" else "notable"
    )
    stored.forests.loc[0, "n_train"] = stored.forests.loc[0, "n_train"] + 1
    checks = {c.name: c for c in development_part(full, series, _dev(), stored, _cfg())}
    assert not checks["development events"].ok
    assert "tier" in checks["development events"].differences[0]
    assert not checks["development month models"].ok
    assert checks["development intervals"].ok


def test_a_different_grid_is_refused(tmp_path: Path) -> None:
    full, series, _, stored = _runs(None, tmp_path)
    stored.intervals["interval_end"] = stored.intervals["interval_end"] + pd.Timedelta("5min")
    (check,) = development_part(full, series, _dev(), stored, _cfg())
    assert not check.ok and check.name == "development grid"
