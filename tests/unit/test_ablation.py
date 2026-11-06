"""M2 weather drop/add test helpers (m2_plan.md §2.5) and report labels."""

from __future__ import annotations

import pandas as pd
import pytest

from gridcast.evaluation.report import model_label
from gridcast.forecasting.ablation import (
    feature_sets,
    flag_table,
    pooled_mae,
    reproduction_ok,
)

COLUMNS = ["lead", "dem_now", "wx_a", "cal", "wx_b"]
WEATHER = ["wx_a", "wx_b"]


def test_feature_sets_cover_full_none_drop_and_add_in_column_order() -> None:
    sets = feature_sets(COLUMNS, WEATHER)
    assert [(k, f) for k, f, _ in sets] == [
        ("full", None), ("none", None),
        ("drop", "wx_a"), ("drop", "wx_b"),
        ("add", "wx_a"), ("add", "wx_b"),
    ]  # fmt: skip
    cols = {(k, f): c for k, f, c in sets}
    assert cols[("full", None)] == COLUMNS
    assert cols[("none", None)] == ["lead", "dem_now", "cal"]
    assert cols[("drop", "wx_a")] == ["lead", "dem_now", "cal", "wx_b"]
    assert cols[("add", "wx_b")] == ["lead", "dem_now", "cal", "wx_b"]
    with pytest.raises(ValueError, match="not in the columns"):
        feature_sets(COLUMNS, ["wx_missing"])


def _fits(maes: dict[tuple[str, str | None], tuple[float, float]]) -> pd.DataFrame:
    """Two bands: (short-band MAE, long-band MAE) per (kind, feature); 1 vs 3 rows weight."""
    rows = []
    for (kind, feature), (short, long) in maes.items():
        for band, mae, n in (("h001-024", short, 100), ("h025-288", long, 300)):
            rows.append(
                {"retrain": "2025-02-01", "band": band, "kind": kind, "feature": feature,
                 "es_mae": mae, "es_valid_rows": n}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_pooled_mae_weights_bands_by_monitored_rows() -> None:
    fits = _fits({("full", None): (40.0, 200.0), ("none", None): (40.0, 280.0)})
    pooled = pooled_mae(fits).set_index(["kind"])["es_mae"]
    assert pooled["full"] == pytest.approx((40 * 100 + 200 * 300) / 400)
    assert pooled["none"] == pytest.approx((40 * 100 + 280 * 300) / 400)


def test_flags_mark_features_above_the_share() -> None:
    # Pooled: none 200, full 160; wx_a alone 165 (17.5% gain); wx_b alone 196 (2%).
    # Without wx_a: 190 (keeping it gains 15.8%); without wx_b: 161 (0.6%).
    fits = _fits(
        {
            ("full", None): (160.0, 160.0),
            ("none", None): (200.0, 200.0),
            ("drop", "wx_a"): (190.0, 190.0),
            ("drop", "wx_b"): (161.0, 161.0),
            ("add", "wx_a"): (165.0, 165.0),
            ("add", "wx_b"): (196.0, 196.0),
        }
    )
    table = flag_table(fits, share=0.15).set_index("feature")
    assert table.loc["wx_a", "add_gain"] == pytest.approx(1 - 165 / 200)
    assert table.loc["wx_a", "keep_gain"] == pytest.approx(1 - 160 / 190)
    assert table.loc["wx_b", "group_gain"] == pytest.approx(0.2)
    assert table["flagged"].to_dict() == {"wx_a": True, "wx_b": False}
    assert not flag_table(fits, share=0.2)["flagged"].any()


def test_reproduction_needs_every_full_fit_to_match() -> None:
    ok = {"reproduced": [{"es_mae": 251.25, "stored_es_mae": 251.25}]}
    off = {"reproduced": [*ok["reproduced"], {"es_mae": 251.3, "stored_es_mae": 251.25}]}
    assert reproduction_ok(ok)
    assert not reproduction_ok(off)
    assert not reproduction_ok({"reproduced": []})


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("persistence", "Persistence"),
        ("lgbm_C", "LightGBM C"),
        ("ridge_Dplus", "Ridge D⁺"),
        ("lgbm_C_delay8h", "LightGBM C (8 h delay)"),
        ("something", "something"),
    ],
)
def test_model_labels(name: str, label: str) -> None:
    assert model_label(name) == label
