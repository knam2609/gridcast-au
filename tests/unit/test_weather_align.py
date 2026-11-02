import numpy as np
import pandas as pd
import pytest

from gridcast.weather.align import site_weighted, to_five_minute

T0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
KINDS = {"temperature_2m": "instant", "shortwave_radiation": "preceding_hour_mean"}


def _hourly(temps: list[float | None], radiation: list[float] | None = None) -> pd.DataFrame:
    n = len(temps)
    return pd.DataFrame(
        {
            "valid_time": pd.date_range(T0, periods=n, freq="1h"),
            "temperature_2m": pd.array(temps, dtype="float64"),
            "shortwave_radiation": radiation if radiation is not None else [0.0] * n,
        }
    )


def _targets(n_steps: int, start: pd.Timestamp = T0) -> pd.DatetimeIndex:
    return pd.date_range(start + pd.Timedelta(minutes=5), periods=n_steps, freq="5min")


def test_instant_values_are_interpolated_linearly() -> None:
    out = to_five_minute(_hourly([10.0, 22.0]), _targets(12), KINDS).set_index("interval_end")
    temp = out["temperature_2m"]
    assert temp[T0 + pd.Timedelta(minutes=5)] == pytest.approx(11.0)
    assert temp[T0 + pd.Timedelta(minutes=30)] == pytest.approx(16.0)
    assert temp[T0 + pd.Timedelta(hours=1)] == pytest.approx(22.0)  # on the hour: exact value


def test_missing_bracketing_hour_gives_missing_value() -> None:
    # 01:00 is missing: 00:00 is exact, but 00:30 and 01:00 both need the 01:00 value.
    hourly = _hourly([10.0, None, 30.0])
    targets = pd.DatetimeIndex([T0, T0 + pd.Timedelta(minutes=30), T0 + pd.Timedelta(hours=1)])
    out = to_five_minute(hourly, targets, KINDS)["temperature_2m"].tolist()
    assert out[0] == 10.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_missing_row_is_treated_like_missing_value() -> None:
    hourly = _hourly([10.0, 20.0, 30.0]).drop(index=1)
    out = to_five_minute(hourly, _targets(24), KINDS)["temperature_2m"]
    assert out.isna().sum() == 23  # every target from 00:05 to 01:55 needs 01:00
    assert out.iloc[-1] == 30.0  # 02:00


def test_preceding_hour_mean_uses_the_hour_ending_at_or_after_t() -> None:
    hourly = _hourly([0.0, 0.0, 0.0], radiation=[100.0, 200.0, 300.0])
    out = to_five_minute(hourly, _targets(13), KINDS).set_index("interval_end")
    rad = out["shortwave_radiation"]
    # Intervals ending in (00:00, 01:00] take the value labelled 01:00 …
    assert (rad[T0 + pd.Timedelta(minutes=5) : T0 + pd.Timedelta(hours=1)] == 200.0).all()
    # … and 01:05 starts the next hour.
    assert rad[T0 + pd.Timedelta(minutes=65)] == 300.0


def test_available_at_is_the_later_bracketing_hour() -> None:
    hourly = _hourly([10.0, 20.0, 30.0])
    hourly["available_at"] = hourly["valid_time"] + pd.Timedelta(hours=30)
    out = to_five_minute(hourly, _targets(24), KINDS).set_index("interval_end")
    assert out.loc[T0 + pd.Timedelta(minutes=5), "available_at"] == T0 + pd.Timedelta(hours=31)
    assert out.loc[T0 + pd.Timedelta(hours=1), "available_at"] == T0 + pd.Timedelta(hours=31)
    assert out.loc[T0 + pd.Timedelta(minutes=65), "available_at"] == T0 + pd.Timedelta(hours=32)


def test_available_at_is_missing_when_a_bracketing_row_is_missing() -> None:
    hourly = _hourly([10.0, 20.0, 30.0])
    hourly["available_at"] = hourly["valid_time"] + pd.Timedelta(hours=30)
    out = to_five_minute(hourly.drop(index=2), _targets(24), KINDS)
    assert out["available_at"].iloc[:12].notna().all()
    assert out["available_at"].iloc[12:].isna().all()


def test_groups_are_aligned_separately() -> None:
    a = _hourly([10.0, 20.0]).assign(offset_days=1)
    b = _hourly([0.0, 0.0]).assign(offset_days=2)
    out = to_five_minute(pd.concat([b, a]), _targets(12), KINDS, by="offset_days")
    assert out["offset_days"].tolist() == [1] * 12 + [2] * 12
    assert out.loc[out["offset_days"] == 1, "temperature_2m"].iloc[-1] == 20.0
    assert (out.loc[out["offset_days"] == 2, "temperature_2m"] == 0.0).all()


def test_rejects_non_utc_and_off_hour_input() -> None:
    with pytest.raises(ValueError, match="UTC"):
        to_five_minute(_hourly([1.0, 2.0]), _targets(3).tz_convert("Australia/Melbourne"), KINDS)
    off = _hourly([1.0, 2.0])
    off["valid_time"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="whole hours"):
        to_five_minute(off, _targets(3), KINDS)
    with pytest.raises(ValueError, match="unknown weather variable kind"):
        to_five_minute(_hourly([1.0, 2.0]), _targets(3), {"temperature_2m": "max"})


def _sites(values: dict[str, list[float | None]]) -> pd.DataFrame:
    frames = []
    for site, temps in values.items():
        f = _hourly(temps).assign(site_id=site)
        f["available_at"] = f["valid_time"] + pd.Timedelta(hours=30)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_site_weighted_mean() -> None:
    hourly = _sites({"a": [10.0, 20.0], "b": [30.0, 40.0]})
    out = site_weighted(hourly, {"a": 0.75, "b": 0.25}, ["temperature_2m"])
    assert out["temperature_2m"].tolist() == pytest.approx([15.0, 25.0])
    assert out["available_at"].tolist() == list(out["valid_time"] + pd.Timedelta(hours=30))


def test_site_weighted_requires_every_site() -> None:
    hourly = _sites({"a": [10.0, 20.0], "b": [30.0, None]})
    out = site_weighted(hourly, {"a": 0.75, "b": 0.25}, ["temperature_2m"])
    assert out["temperature_2m"].iloc[0] == pytest.approx(15.0)
    assert np.isnan(out["temperature_2m"].iloc[1])  # never renormalised over present sites
    only_a = hourly[hourly["site_id"] == "a"]
    assert (
        site_weighted(only_a, {"a": 0.75, "b": 0.25}, ["temperature_2m"])["temperature_2m"]
        .isna()
        .all()
    )


def test_site_weighted_by_offset_and_validation() -> None:
    hourly = pd.concat(
        [
            _sites({"a": [10.0], "b": [20.0]}).assign(offset_days=1),
            _sites({"a": [0.0], "b": [40.0]}).assign(offset_days=2),
        ]
    )
    out = site_weighted(hourly, {"a": 0.5, "b": 0.5}, ["temperature_2m"], by=["offset_days"])
    assert out.set_index("offset_days")["temperature_2m"].to_dict() == {1: 15.0, 2: 20.0}
    with pytest.raises(ValueError, match="sum to 1"):
        site_weighted(hourly, {"a": 0.5, "b": 0.6}, ["temperature_2m"])
    with pytest.raises(ValueError, match="duplicate"):
        site_weighted(hourly, {"a": 0.5, "b": 0.5}, ["temperature_2m"])  # offsets not in `by`
