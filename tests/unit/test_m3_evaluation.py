"""M3 step 4 scoring helpers: rearrangement, aggregation, bootstrap, U1–U4, weather groups."""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.evaluation.m2_report import LeakageOutcome
from gridcast.intervals.evaluation import (
    Aggregator,
    bootstrap_ratio,
    forecast_day_max,
    issue_day_codes,
    nesting_failures,
    pair_stats,
    per_lead,
    pooled,
    success_criteria,
    tau_of,
    week_blocks,
)
from gridcast.intervals.scores import (
    interval_score,
    pinball,
    rearrange_bounds,
    weighted_interval_score,
)

LEVELS = (50, 80, 95)
OK = LeakageOutcome(5, 0, 0)


def _bounds(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[dict, dict]:
    centre = rng.normal(0, 10, shape)
    half = {50: 3.0, 80: 6.0, 95: 9.0}
    lower = {lv: centre - half[lv] + rng.normal(0, 2, shape) for lv in LEVELS}
    upper = {lv: centre + half[lv] + rng.normal(0, 2, shape) for lv in LEVELS}
    return lower, upper


def test_tau_of_each_bound() -> None:
    assert [tau_of(lv, s) for lv in LEVELS for s in ("lo", "hi")] == [
        0.25, 0.75, 0.1, 0.9, 0.025, 0.975
    ]  # fmt: skip


def test_rearrangement_nests_every_level_and_keeps_the_values() -> None:
    rng = np.random.default_rng(0)
    lower, upper = _bounds(rng, (50, 7))
    assert any(f.any() for f in nesting_failures(lower, upper).values())  # the noise crosses
    lo, hi = rearrange_bounds(lower, upper)
    assert not any(f.any() for f in nesting_failures(lo, hi).values())
    before = np.sort(np.stack([*lower.values(), *upper.values()]), axis=0)
    after = np.sort(np.stack([*lo.values(), *hi.values()]), axis=0)
    np.testing.assert_array_equal(before, after)
    assert np.all(lo[95] <= lo[80]) and np.all(lo[50] <= hi[50]) and np.all(hi[80] <= hi[95])


def test_rearrangement_blanks_a_pair_with_a_missing_bound() -> None:
    lower = {50: np.array([[4.0, 1.0]]), 80: np.array([[np.nan, 0.0]])}
    upper = {50: np.array([[5.0, 2.0]]), 80: np.array([[6.0, 3.0]])}
    lo, hi = rearrange_bounds(lower, upper)
    assert np.isnan(lo[50][0, 0]) and np.isnan(hi[80][0, 0])
    assert (lo[80][0, 1], lo[50][0, 1], hi[50][0, 1], hi[80][0, 1]) == (0.0, 1.0, 2.0, 3.0)


def test_pair_stats_match_the_score_functions() -> None:
    rng = np.random.default_rng(1)
    lower, upper = _bounds(rng, (20, 5))
    y = rng.normal(0, 10, (20, 5))
    point = y + rng.normal(0, 3, (20, 5))
    stats = dict(pair_stats(lower, upper, y, point))
    for lv in LEVELS:
        lo, hi = lower[lv], upper[lv]
        np.testing.assert_array_equal(stats[f"cov_{lv}"], (lo <= y) & (y <= hi))
        np.testing.assert_allclose(stats[f"is_{lv}"], interval_score(lo, hi, y, 1 - lv / 100))
        np.testing.assert_allclose(stats[f"width_{lv}"], hi - lo)
        assert np.all(stats[f"cov_{lv}"] + stats[f"below_{lv}"] + stats[f"above_{lv}"] >= 1)
    np.testing.assert_allclose(stats["pin_q100"], pinball(lower[80], y, 0.1))
    np.testing.assert_array_equal(stats["le_q975"], y <= upper[95])
    bounds = {lv: (lower[lv], upper[lv]) for lv in LEVELS}
    np.testing.assert_allclose(stats["wis"], weighted_interval_score(point, bounds, y))
    any_fail = np.zeros(y.shape, dtype=bool)
    for name in ("lo_50_80", "hi_50_80", "lo_80_95", "hi_80_95"):
        any_fail |= stats[f"nest_{name}"].astype(bool)
    np.testing.assert_array_equal(stats["non_nested"], any_fail)


def test_aggregator_sums_match_brute_force() -> None:
    rng = np.random.default_rng(2)
    n, h = 12, 4
    day_code = np.repeat(np.arange(4), 3)
    mask = rng.random((n, h)) > 0.2
    values = rng.normal(0, 1, (n, h))
    hour = rng.integers(0, 3, (n, h))
    group = rng.random((n, h)) > 0.5
    agg = Aggregator(mask, day_code, {"hour": (hour, 3)}, {"g": group})
    sums = agg.add(values)
    for d in range(4):
        rows = day_code == d
        np.testing.assert_allclose(sums.by_lead[d], np.where(mask, values, 0)[rows].sum(axis=0))
        for k in range(3):
            sel = mask[rows] & (hour[rows] == k)
            assert sums.by_label["hour"][d, k] == pytest.approx(values[rows][sel].sum())
        sel = mask[rows] & group[rows]
        assert sums.by_group["g"][d] == pytest.approx(values[rows][sel].sum())
    assert agg.counts.by_lead.sum() == mask.sum()
    assert pooled(sums, agg.counts) == pytest.approx(values[mask].mean())
    expected = np.array([values[:, j][mask[:, j]].mean() for j in range(h)])
    np.testing.assert_allclose(per_lead(sums, agg.counts), expected)
    assert agg.add(values, conditional=False).by_label == {}


def test_aggregator_refuses_unsorted_days() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        Aggregator(np.ones((3, 2), bool), np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="sorted"):
        issue_day_codes(np.array([738000, 737999]))


def test_bootstrap_pools_and_resamples_whole_blocks() -> None:
    sums = np.array([1.0, 2.0, 3.0, 0.0, 10.0])
    counts = np.array([2.0, 2.0, 2.0, 0.0, 4.0])
    e = bootstrap_ratio(sums, counts, seed=7)
    assert e.value == pytest.approx(16 / 10) and e.n_pairs == 10 and e.n_blocks == 4
    assert e.lo <= e.value <= e.hi
    assert bootstrap_ratio(sums, counts, seed=7) == e  # seeded
    weekly = bootstrap_ratio(sums, counts, seed=7, blocks=np.array([0, 0, 1, 1, 1]))
    assert weekly.value == pytest.approx(e.value) and weekly.n_blocks == 2
    same = bootstrap_ratio(sums - sums, counts, seed=7)
    assert (same.value, same.lo, same.hi) == (0.0, 0.0, 0.0)
    empty = bootstrap_ratio(np.zeros(3), np.zeros(3), seed=7)
    assert empty.n_pairs == 0 and np.isnan(empty.value)


def test_week_blocks_count_from_the_first_day() -> None:
    days = np.array([100, 101, 106, 107, 113, 114])
    np.testing.assert_array_equal(week_blocks(days), [0, 0, 0, 1, 1, 2])


def _criteria(cov: np.ndarray, is80: np.ndarray, pooled_cov: float = 80.0) -> list[bool]:
    base = {"a": np.full(len(is80), 100.0), "b": np.full(len(is80), 120.0)}
    out = success_criteria(cov, pooled_cov, is80, float(is80.mean()), base, OK)
    return [c.passed for c in out]


def test_criteria_pass_and_fail_as_registered() -> None:
    h = 288
    cov, is80 = np.full(h, 80.0), np.full(h, 90.0)
    assert _criteria(cov, is80) == [True, True, True, True]
    assert _criteria(cov, is80, pooled_cov=77.9)[0] is False
    assert _criteria(cov, is80, pooled_cov=82.0)[0] is True
    low = cov.copy()
    low[200] = 74.9
    assert _criteria(low, is80)[1] is False
    short = is80.copy()
    short[:23] = 150.0  # above the envelope before 2 h only: U3's per-lead part ignores them
    assert _criteria(cov, short)[2] is True
    long = is80.copy()
    long[23] = 100.0  # equal to the envelope at exactly 2 h
    assert _criteria(cov, long)[2] is False
    failed = success_criteria(
        cov, 80.0, is80, 90.0, {"a": np.full(h, 100.0)}, LeakageOutcome(3, 1, 1)
    )
    assert failed[3].passed is False and "3 passed, 1 failed" in failed[3].measured


def test_criteria_pooled_part_uses_the_mean_envelope() -> None:
    h = 288
    envelope = np.full(h, 100.0)
    is80 = np.full(h, 99.0)
    out = success_criteria(np.full(h, 80.0), 80.0, is80, 100.5, {"a": envelope}, OK)
    assert out[2].passed is False and "pooled 100.5 vs 100.0 MW" in out[2].measured


# --------------------------------------------------------------------------- weather groups

NS = 1_000_000_000


def _day_grid(n_days: int, per_day: int) -> np.ndarray:
    return np.repeat(np.arange(n_days), per_day)


def test_day_max_uses_the_freshest_forecast_known_at_the_issue_time() -> None:
    local_day = _day_grid(2, 4)  # positions 0-3 day 0, 4-7 day 1
    fresh = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    fresh_at = np.full(8, 10 * NS)
    stale = np.array([0, 0, 0, 0, 50, 50, 50, 50], dtype=float)
    stale_at = np.zeros(8, dtype=np.int64)
    targets = np.array([[1, 2, 5], [1, 2, 5]])
    origins = np.array([5 * NS, 10 * NS])
    out = forecast_day_max(origins, targets, local_day, [(fresh, fresh_at), (stale, stale_at)])
    # Before the fresh run is published only the stale values exist (day 0: 0, day 1: 50).
    np.testing.assert_array_equal(out[0], [0, 0, 50])
    np.testing.assert_array_equal(out[1], [4, 4, 8])


def test_day_max_ignores_forecasts_published_after_the_issue_time() -> None:
    rng = np.random.default_rng(4)
    local_day = _day_grid(3, 6)
    values = [rng.normal(20, 5, 18), rng.normal(20, 5, 18)]
    available = [rng.integers(0, 20, 18) * NS, rng.integers(0, 20, 18) * NS]
    targets = np.array([[2, 8, 13], [5, 6, 11], [9, 12, 15]])
    origins = np.array([7, 9, 12]) * NS
    base = forecast_day_max(origins, targets, local_day, list(zip(values, available, strict=True)))
    for i, o in enumerate(origins):
        changed = [np.where(a > o, v + 1e3, v) for v, a in zip(values, available, strict=True)]
        again = forecast_day_max(
            origins[i : i + 1], targets[i : i + 1], local_day,
            list(zip(changed, available, strict=True)),
        )  # fmt: skip
        np.testing.assert_array_equal(again, base[i : i + 1])
    # Control: changing a value published before the issue time does change the label.
    known = [np.where(a <= origins[0], v + 1e3, v) for v, a in zip(values, available, strict=True)]
    moved = forecast_day_max(origins, targets, local_day, list(zip(known, available, strict=True)))
    assert not np.array_equal(moved[0], base[0])


def test_day_max_spans_three_days_and_is_nan_without_a_known_value() -> None:
    local_day = np.array([0, 0, 1, 1, 2, 2])  # a short middle day, as when clocks go forward
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    available = np.array([0, 0, 0, 0, 99, 99]) * NS
    out = forecast_day_max(np.array([1 * NS]), np.array([[1, 2, 3, 4]]), local_day,
                           [(values, available)])  # fmt: skip
    assert out[0, :3].tolist() == [2.0, 4.0, 4.0] and np.isnan(out[0, 3])
