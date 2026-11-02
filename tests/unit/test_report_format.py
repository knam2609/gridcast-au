import pytest

from gridcast.evaluation.report import horizon_label, horizon_ranges, money


@pytest.mark.parametrize(
    ("h", "label"),
    [(1, "5 min"), (6, "30 min"), (12, "1 h"), (20, "1 h 40 min"), (288, "24 h")],
)
def test_horizon_label(h: int, label: str) -> None:
    assert horizon_label(h) == label


def test_horizon_ranges_split_non_contiguous_runs() -> None:
    assert horizon_ranges([1, 2, 3, 287]) == "5 min–15 min, 23 h 55 min"
    assert horizon_ranges([12]) == "1 h"


def test_money() -> None:
    assert money(-1000) == "−$1,000"
    assert money(60.614, 2) == "$60.61"
