"""CQR batch helpers (M3 step 3): per-level runs merge into the single run, metrics, hash."""

from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.config import load_config
from gridcast.intervals.batch import interval_config_hash
from gridcast.intervals.conformal import CandidateGrid, ConformalResult, run_conformal
from gridcast.intervals.cqr_batch import cqr_metrics, merge_levels
from tests.interval_fixtures import make_quantile_problem, synthetic_point, synthetic_quantiles

GRID = CandidateGrid((0.25, 0.5), ("lead", "lead_tod"), (0.0, 0.05))
BANDS = [(1, 4), (5, 12)]
LEVELS = (50, 80, 95)


@cache
def _inputs() -> tuple[pd.DatetimeIndex, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    times, y, point = synthetic_point(n_days=10, n_leads=12, seed=3)
    return times, y, synthetic_quantiles(point)


def _run(levels: tuple[int, ...]) -> ConformalResult:
    times, y, q = _inputs()
    problem = make_quantile_problem(times, y, {lv: q[lv] for lv in levels})
    retrains = [times[0] + pd.Timedelta(days=d) for d in (6, 8)]
    return run_conformal(problem, GRID, list(levels), retrains, BANDS, 0.25)


def test_per_level_runs_merge_into_the_single_run() -> None:
    whole = _run(LEVELS)
    merged = merge_levels([_run((lv,)) for lv in (80, 50, 95)], LEVELS)
    assert merged.origin_times.equals(whole.origin_times)
    assert list(merged.lower) == list(LEVELS)
    for level in LEVELS:
        np.testing.assert_array_equal(merged.lower[level], whole.lower[level])
        np.testing.assert_array_equal(merged.upper[level], whole.upper[level])
    assert merged.selections == whole.selections
    assert merged.crossed == whole.crossed
    assert [c.key for c in merged.candidates] == [c.key for c in whole.candidates]
    pd.testing.assert_frame_equal(merged.daily, whole.daily)
    pd.testing.assert_frame_equal(merged.clipped, whole.clipped)


def test_merge_refuses_a_level_twice() -> None:
    part = _run((80,))
    with pytest.raises(ValueError, match="exactly one part"):
        merge_levels([part, part], (80,))


def _result(lower: dict[int, list[float]], upper: dict[int, list[float]]) -> ConformalResult:
    times = pd.date_range("2024-01-01", periods=1, freq="30min", tz="UTC")
    return ConformalResult(
        times,
        {lv: np.array([v], dtype=np.float32) for lv, v in lower.items()},
        {lv: np.array([v], dtype=np.float32) for lv, v in upper.items()},
        [],
        [],
        pd.DataFrame(),
        pd.DataFrame(),
    )


def test_cqr_metrics_reliability_and_nesting() -> None:
    # Two leads; lead 2's 95% lower bound sits above its 80% lower bound (not nested).
    res = _result(
        {50: [90.0, 95.0], 80: [80.0, 90.0], 95: [70.0, 92.0]},
        {50: [110.0, 105.0], 80: [120.0, 110.0], 95: [130.0, 120.0]},
    )
    y = np.array([[100.0, 100.0]])
    mask = np.ones_like(y, dtype=bool)
    m = cqr_metrics(res, y, mask, np.array([[100.0, 100.0]])).set_index(["metric", "horizon"])
    v = m["value"]
    assert v[("non_nested", 0)] == pytest.approx(0.5)
    assert v[("non_nested", 1)] == 0.0 and v[("non_nested", 2)] == 1.0
    assert v[("share_le_q100", 0)] == 0.0 and v[("share_le_q900", 0)] == 1.0
    # pinball at τ = 0.1 for bound 80 below y = 100: τ (y − q) = 0.1 · 20
    assert v[("pinball_q100", 1)] == pytest.approx(2.0)
    assert v[("coverage_95", 0)] == 1.0


def test_interval_hash_ignores_the_qr_batch_pointer() -> None:
    cfg = load_config()
    base = interval_config_hash(cfg)
    cfg.intervals = cfg.intervals.model_copy(update={"qr_batch": "another"})
    assert interval_config_hash(cfg) == base
    cfg.intervals = cfg.intervals.model_copy(update={"selection_weeks": 9})
    assert interval_config_hash(cfg) != base


def test_step1_interval_hash_is_unchanged() -> None:
    """Step 1's batch (72eb5209…) and step 2's run recorded this hash of base.yaml."""
    assert interval_config_hash(load_config()) == "d34273400899af69"
