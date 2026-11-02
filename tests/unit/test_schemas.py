import pandas as pd
import pytest

from gridcast.validation.schemas import validate_market_intervals
from tests.integration.factories import clean_month


def _frame() -> pd.DataFrame:
    return clean_month().head(50).copy()


def test_clean_frame_validates() -> None:
    assert len(validate_market_intervals(_frame(), ["VIC1"])) == 50


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda df: df.assign(region_id="NSW1"), "isin"),
        (lambda df: df.assign(rrp_aud_mwh=-2000.0), "in_range"),
        (
            lambda df: df.assign(
                interval_end=df["interval_end"] + pd.Timedelta(minutes=1),
                interval_start=df["interval_start"] + pd.Timedelta(minutes=1),
            ),
            "5-minute grid",
        ),
        (
            lambda df: df.assign(interval_start=df["interval_start"] - pd.Timedelta(minutes=5)),
            "equal 5 minutes",
        ),
        (lambda df: pd.concat([df, df.iloc[[3]]]), "duplicate"),
        (lambda df: df.assign(total_demand_mw=float("nan"), rrp_aud_mwh=float("nan")), "neither"),
        (lambda df: df.assign(extra=1), "extra"),
        (
            lambda df: df.assign(
                interval_end=df["interval_end"].dt.tz_convert("Australia/Melbourne")
            ),
            "UTC",
        ),
    ],
)
def test_invalid_frames_rejected(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(Exception, match=message):
        validate_market_intervals(mutate(_frame()), ["VIC1"])
