"""M2 step 4 end to end on the test database: stored ML run → drop/add test → report."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import Engine

from gridcast.evaluation.m2_report import LeakageOutcome, build_report
from gridcast.evaluation.m2_results import load_evaluation_set
from gridcast.evaluation.report import load_latest
from gridcast.features import catalogue
from gridcast.forecasting.ablation import (
    fits_frame,
    flag_table,
    read_ablation,
    reproduction_ok,
    run_weather_ablation,
    write_ablation,
)
from gridcast.forecasting.ml_backtest import run_ml_backtest, store_sets
from tests.integration.test_ml_backtest import _cfg, ready  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration


def test_ablation_reproduces_the_stored_run(ready: Engine, tmp_path: Path) -> None:  # noqa: F811
    cfg = _cfg(tmp_path)
    cfg.ml = cfg.ml.model_copy(update={"lead_bands": [(1, 24), (25, 288)]})  # as in run 1
    cfg.evaluation = cfg.evaluation.model_copy(
        update={"weather_ablation_retrains": [dt.date(2021, 10, 20)]}
    )
    cfg.paths.reports_dir = tmp_path / "reports"  # never the repository's reports/
    sets = run_ml_backtest(
        ready, cfg, "VIC1", plus=False, delay=True, model_root=cfg.data_path(cfg.ml.model_dir)
    )
    batches = store_sets(ready, cfg, "VIC1", sets)

    # M1's report ignores the ML batches (they also hold baseline rows).
    with pytest.raises(LookupError):
        load_latest(ready, "VIC1", "demand")
    assert load_evaluation_set(ready, "VIC1", "main").batch_id == str(batches["main"])
    with pytest.raises(LookupError, match="plus"):
        load_evaluation_set(ready, "VIC1", "plus")

    result = run_weather_ablation(ready, cfg, "VIC1")
    weather = [c for c in catalogue.feature_names("C") if c not in catalogue.feature_names("A")]
    assert result["weather_features"] == weather
    assert len(result["fits"]) == len(cfg.ml.lead_bands) * (2 + 2 * len(weather))
    # The all-features fit is the stored run's own early-stopping fit, bit for bit.
    assert reproduction_ok(result)
    assert len(result["reproduced"]) == len(cfg.ml.lead_bands)

    path = write_ablation(cfg, result)
    assert path.is_file()
    again = read_ablation(cfg, str(batches["main"]))
    assert again is not None and again["fits"] == result["fits"]
    flags = flag_table(fits_frame(again), again["flag_share"])
    assert list(flags["feature"]) == weather

    # The report: S1–S4 computed from the stored metrics, every section present.
    text = build_report(ready, cfg, "VIC1", leakage=LeakageOutcome(5, 0, 0))
    for heading in (
        "## Provenance",
        "## Success criteria (pre-registered)",
        "## Short lead times and the band boundary",
        "## Weather value: the “investigate before believing” check",
        "## Publication-delay sensitivity",
        "## Model selection at each retrain",
        "## Feature importance (LightGBM C)",
        "## Failure cases",
    ):
        assert heading in text
    assert "## C⁺ and D⁺" not in text  # no plus batch stored
    assert "## Development runs" not in text  # a single development run
    for key in ("S1", "S2", "S3", "S4"):
        assert f"| {key} |" in text
    assert "matches the current config" in text
    assert "reproduced the stored run's early-stopping MAE exactly" in text
    assert "5 passed, 0 failed" in text
    figures = tmp_path / "reports" / "figures"
    assert (figures / "m2_demand_mae_by_lead.png").stat().st_size > 0
    assert (figures / "m2_demand_short_leads.png").stat().st_size > 0
    failing = build_report(ready, cfg, "VIC1", leakage=LeakageOutcome(4, 1, 1))
    assert "| S4 |" in failing and "4 passed, 1 failed" in failing

    # The test refuses a retrain the stored run does not have.
    cfg.evaluation = cfg.evaluation.model_copy(
        update={"weather_ablation_retrains": [dt.date(2021, 10, 1)]}
    )
    with pytest.raises(ValueError, match="not retrains"):
        run_weather_ablation(ready, cfg, "VIC1")

    # A second development run (more lead bands, the latest target form) on the same pairs:
    # the report compares the two and applies the pre-registered adoption rule.
    second = _cfg(tmp_path)
    second.paths.reports_dir = tmp_path / "reports"
    second.ml = second.ml.model_copy(
        update={
            "lead_bands": [(1, 24), (25, 72), (73, 288)],
            "grid": second.ml.grid.model_copy(
                update={"target_forms": [*second.ml.grid.target_forms, "latest"]}
            ),
        }
    )
    sets = run_ml_backtest(
        ready, second, "VIC1", plus=False, delay=True, model_root=second.data_path(Path("models2"))
    )
    store_sets(ready, second, "VIC1", sets)
    text = build_report(ready, second, "VIC1", leakage=LeakageOutcome(5, 0, 0))
    assert "## Development runs" in text
    assert "lead bands 2 → 3; target form latest added" in text
    assert "Every other ML and setup setting is identical" in text
    assert "**Outcome:**" in text and "for the holdout.**" in text
    assert "| 6 h 5 min |" in text  # the new boundary's first step is in the tables
