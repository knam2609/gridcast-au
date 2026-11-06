"""M2 report computations: S1–S4, lead-time facts, day and year errors, selection tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluation.m2_report import (
    LeakageOutcome,
    Pairs,
    band_jump,
    chosen_counts,
    daily_errors,
    first_better,
    group_means,
    parse_pytest_summary,
    selection_table,
    success_criteria,
    yearly_errors,
)

H = 288
BASELINES = ["persistence", "daily_naive"]
OK = LeakageOutcome(passed=12, failed=0, returncode=0)


def _table(c_long: float = 250.0) -> pd.DataFrame:
    """MAE by horizon: persistence grows, daily naive flat, C below both after 10 min."""
    h = np.arange(1, H + 1)
    persistence = 40 + 10 * h
    return pd.DataFrame(
        {
            "persistence": persistence.astype(float),
            "daily_naive": np.full(H, 470.0),
            "lgbm_A": np.where(h <= 24, 50 + 3 * h, 330.0),
            "lgbm_C": np.where(h <= 24, 55 + 3 * h, c_long),
            "lgbm_D": np.where(h <= 24, 55 + 3 * h, c_long - 5),
        },
        index=pd.Index(h, name="horizon"),
    )


def _pooled(table: pd.DataFrame) -> dict[str, float]:
    return {m: float(table[m].mean()) for m in table.columns}


def test_criteria_pass_on_a_clear_win() -> None:
    table = _table()
    result = success_criteria(table, _pooled(table), BASELINES, OK)
    assert [c.key for c in result] == ["S1", "S2", "S3", "S4"]
    assert all(c.passed for c in result)
    # S1 ignores leads under 2 h, where C loses to persistence at 5 minutes.
    assert table.loc[1, "lgbm_C"] > table.loc[1, "persistence"]
    assert "265 of 265 lead times" in result[0].measured


def test_s1_fails_if_one_lead_time_is_lost() -> None:
    table = _table()
    table.loc[200, "lgbm_C"] = 470.0  # ties the best baseline: not strictly lower
    s1 = success_criteria(table, _pooled(table), BASELINES, OK)[0]
    assert not s1.passed
    assert "264 of 265" in s1.measured and "16 h 40 min" in s1.measured


def test_s3_and_s4_compare_pooled_mae_and_need_the_leakage_tests() -> None:
    table = _table()
    pooled = _pooled(table)
    worse = {**pooled, "lgbm_C": pooled["lgbm_A"] + 1}
    assert not success_criteria(table, worse, BASELINES, OK)[2].passed
    better_than_oracle = {**pooled, "lgbm_D": pooled["lgbm_C"] + 0.1}
    assert not success_criteria(table, better_than_oracle, BASELINES, OK)[3].passed
    equal = {**pooled, "lgbm_D": pooled["lgbm_C"]}
    assert success_criteria(table, equal, BASELINES, OK)[3].passed  # ≤ allows a tie
    failing = LeakageOutcome(passed=11, failed=1, returncode=1)
    s4 = success_criteria(table, pooled, BASELINES, failing)[3]
    assert not s4.passed and "1 failed" in s4.measured


def test_s2_uses_the_mean_of_the_per_lead_envelope() -> None:
    table = _table(c_long=460.0)
    pooled = _pooled(table)
    envelope = table[BASELINES].min(axis=1).mean()
    s2 = success_criteria(table, pooled, BASELINES, OK)[1]
    assert s2.passed == (pooled["lgbm_C"] < envelope)
    assert f"{envelope:,.1f}" in s2.measured


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("....\n12 passed in 3.1s\n", (12, 0)),
        ("..F.\n1 failed, 11 passed in 3.1s\n", (11, 1)),
        ("E\n1 error in 0.2s\n", (0, 1)),
        ("", (0, 0)),
    ],
)
def test_pytest_summary_parsing(output: str, expected: tuple[int, int]) -> None:
    assert parse_pytest_summary(output) == expected
    passed, failed = expected
    assert LeakageOutcome(passed, failed, 0 if failed == 0 else 1).ok == (
        passed > 0 and failed == 0
    )


def test_lead_time_facts() -> None:
    table = _table()
    groups = group_means(table, ((1, 12), (25, 288)))
    assert list(groups.index) == ["5 min–1 h", "2 h 5 min–24 h"]
    assert groups.loc["2 h 5 min–24 h", "lgbm_C"] == pytest.approx(250.0)
    assert first_better(table, "lgbm_C", "persistence") == 3  # 61 vs 60 at 10 min
    assert first_better(table, "daily_naive", "lgbm_C") is None
    jump, typical = band_jump(table, "lgbm_C", 24)
    assert jump == pytest.approx(250.0 - (55 + 3 * 24))
    assert typical == pytest.approx(0.0)  # flat beyond the boundary dominates the median


def _pairs() -> Pairs:
    """Two issue times, 4 horizons, on a grid whose local day changes every 3 steps."""
    times = pd.date_range("2024-01-01 00:00", periods=2, freq="30min", tz="UTC")
    actual = np.array([[100.0, 100, 100, 100], [100, 100, 100, np.nan]])
    pred = {
        "m1": np.array([[110.0, 90, 100, 130], [100, 100, 80, 100]], dtype=np.float32),
        "m2": np.array([[100.0, 100, 100, 100], [100, 100, 100, 100]], dtype=np.float32),
    }
    mask = np.isfinite(actual)
    return Pairs(times, np.array([0, 1]), pred, actual, mask)


def test_daily_errors_group_by_the_target_day() -> None:
    pairs = _pairs()
    local_day = np.array([7, 7, 7, 8, 8, 8])  # positions 0…5; targets are 1…5
    out = daily_errors(pairs, local_day, ["m1", "m2"], 1, 4)
    # Day 7: targets 1, 2 of origin 0 and target 2 of origin 1 → errors 10, −10, 0.
    # Day 8: origin 0 targets 3, 4 (0, 30); origin 1 targets 3, 4 (0, −20); target 5 is
    # missing.
    assert out.loc[7, "n"] == 3 and out.loc[8, "n"] == 4
    assert out.loc[7, "m1"] == pytest.approx(20 / 3)
    assert out.loc[8, "m1"] == pytest.approx(50 / 4)
    assert out.loc[8, "bias"] == pytest.approx((0 + 30 + 0 - 20) / 4)
    assert out.loc[8, "m2"] == 0


def test_yearly_errors_label_partial_years() -> None:
    times = pd.DatetimeIndex(["2022-11-30 14:00", "2023-06-30 14:00"], tz="UTC")
    actual = np.array([[10.0], [10.0]])
    pairs = Pairs(times, np.array([0, 1]), {"m": np.array([[12.0], [16.0]])}, actual,
                  np.ones((2, 1), dtype=bool))  # fmt: skip
    out = yearly_errors(pairs, ["m"])
    assert list(out.index) == ["2022 (Dec)", "2023 (Jul)"]
    assert out["m"].tolist() == [2.0, 6.0]


def _record(retrain: str, band: str, chosen_form: str) -> dict[str, object]:
    settings = [
        {"target_form": f, "objective": "l1", "num_leaves": 31}
        for f in ("residual", "direct", "anchored")
    ]
    maes = {"residual": 260.0, "direct": 250.0, "anchored": 255.0}
    maes[chosen_form] = 240.0
    chosen = next(s for s in settings if s["target_form"] == chosen_form)
    return {
        "retrain": retrain,
        "band": band,
        "chosen": chosen,
        "params": {"n_trees": 321},
        "candidates": [{"setting": s, "mae": maes[str(s["target_form"])]} for s in settings],
    }


def test_selection_table_and_counts() -> None:
    records = [_record("2024-02-01", "h001-024", "direct"),
               _record("2024-02-01", "h025-288", "anchored")]  # fmt: skip
    table = selection_table(records)
    assert "| 2024-02-01 | 5 min–2 h | direct, L1, 31 leaves | 321 |" in table
    assert "**240.0**" in table
    counts = chosen_counts({"lgbm_C": records, "ridge_C": [_record("2024-02-01", "h001-024",
                                                                    "residual")]})  # fmt: skip
    assert counts.loc[("LightGBM", "2 h 5 min–24 h"), "anchored"] == 1
    assert counts.loc[("Ridge", "5 min–2 h"), "residual"] == 1
    assert counts.to_numpy().sum() == 3
