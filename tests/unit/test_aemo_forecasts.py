import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from gridcast.ingestion.aemo_forecasts import (
    P5MIN,
    PREDISPATCH,
    extract_forecasts,
    predispatch_run_time,
)
from gridcast.ingestion.mms_csv import MmsFormatError
from tests.mms_fixtures import (
    P5MIN_COLS,
    PREDISPATCH_COLS,
    mms_lines,
    p5min_rows,
    predispatch_rows,
    write_zip,
)

DAY = dt.date(2024, 10, 1)


def _pd_zip(
    tmp_path: Path, rows: list[dict[str, str]], version: str = "8", cols=PREDISPATCH_COLS
) -> Path:
    lines = mms_lines("PREDISPATCH", "REGION_SOLUTION", version, cols, rows)
    return write_zip(tmp_path / f"pd_{version}.zip", "PD.CSV", lines)


def _p5_zip(tmp_path: Path, rows: list[dict[str, str]], version: str = "9") -> Path:
    lines = mms_lines("P5MIN", "REGIONSOLUTION", version, P5MIN_COLS, rows)
    return write_zip(tmp_path / f"p5_{version}.zip", "P5.CSV", lines)


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_predispatch_run_time_from_seqno() -> None:
    got = predispatch_run_time(pd.Series(["2024100101", "2024100148"], dtype="string"))
    # Trading day 1 Oct starts 04:00 NEM time = 18:00 UTC on 30 Sep; run 48 is 23.5 h later.
    assert list(got) == [_utc("2024-09-30 18:00"), _utc("2024-10-01 17:30")]


@pytest.mark.parametrize("bad", ["2024100100", "2024100149", "20241001", "2024-10-0101"])
def test_predispatch_run_time_rejects_malformed(bad: str) -> None:
    with pytest.raises(MmsFormatError):
        predispatch_run_time(pd.Series([bad], dtype="string"))


def test_predispatch_extract_keeps_region_and_derives_times(tmp_path: Path) -> None:
    path = _pd_zip(tmp_path, predispatch_rows(DAY, [1, 2], periods=3))
    ext = extract_forecasts([path], PREDISPATCH, ["VIC1"])
    f = ext.frame
    assert len(f) == 6 and set(f.region_id) == {"VIC1"}
    assert set(f.source) == {"predispatch"} and set(f.resolution_minutes) == {30}
    run1 = f[f.run_time == _utc("2024-09-30 18:00")]
    assert list(run1.target_end) == [
        _utc(t) for t in ("2024-09-30 18:30", "2024-09-30 19:00", "2024-09-30 19:30")
    ]
    assert (run1.available_at == _utc("2024-09-30 18:02:14")).all()
    assert run1.total_demand_mw.tolist() == [5011.0, 5021.0, 5031.0]
    assert ext.versions == {"8"}


def test_predispatch_intervention_rows_dropped(tmp_path: Path) -> None:
    rows = predispatch_rows(DAY, [1], periods=2, regions=("VIC1",))
    rows.append({**rows[0], "INTERVENTION": "1", "TOTALDEMAND": "1.0"})
    ext = extract_forecasts([_pd_zip(tmp_path, rows)], PREDISPATCH, ["VIC1"])
    assert ext.intervention_rows_dropped == 1 and len(ext.frame) == 2


def test_predispatch_version_9_with_appended_columns_is_mapped_by_name(tmp_path: Path) -> None:
    cols = [*PREDISPATCH_COLS[:3], "BDU_INITIAL_ENERGY_STORAGE", *PREDISPATCH_COLS[3:]]
    rows = [{**r, "BDU_INITIAL_ENERGY_STORAGE": "123"} for r in predispatch_rows(DAY, [1])]
    ext = extract_forecasts([_pd_zip(tmp_path, rows, "9", cols)], PREDISPATCH, ["VIC1"])
    assert ext.versions == {"9"} and len(ext.frame) == 4


def test_unsupported_version_fails(tmp_path: Path) -> None:
    with pytest.raises(MmsFormatError, match="unsupported version"):
        extract_forecasts(
            [_pd_zip(tmp_path, predispatch_rows(DAY, [1]), "10")], PREDISPATCH, ["VIC1"]
        )
    with pytest.raises(MmsFormatError, match="unsupported version"):
        extract_forecasts(
            [_p5_zip(tmp_path, p5min_rows(dt.datetime(2024, 10, 1, 0, 5), 1), "11")],
            P5MIN,
            ["VIC1"],
        )


def test_duplicate_key_fails(tmp_path: Path) -> None:
    rows = predispatch_rows(DAY, [1], regions=("VIC1",))
    rows.append({**rows[1], "TOTALDEMAND": "5000"})
    with pytest.raises(MmsFormatError, match="share a key"):
        extract_forecasts([_pd_zip(tmp_path, rows)], PREDISPATCH, ["VIC1"])


def test_predispatch_first_period_must_end_30_min_after_run(tmp_path: Path) -> None:
    rows = predispatch_rows(DAY, [1], periods=3, regions=("VIC1",))[1:]  # first period missing
    with pytest.raises(MmsFormatError, match="off the run grid"):
        extract_forecasts([_pd_zip(tmp_path, rows)], PREDISPATCH, ["VIC1"])


def test_p5min_extract_intervals_zero_to_55_min(tmp_path: Path) -> None:
    path = _p5_zip(tmp_path, p5min_rows(dt.datetime(2024, 10, 1, 0, 5), 2))
    ext = extract_forecasts([path], P5MIN, ["VIC1"])
    f = ext.frame
    assert len(f) == 24 and set(f.resolution_minutes) == {5}
    first = f[f.run_time == _utc("2024-09-30 14:05")]
    offsets = (first.target_end - first.run_time).dt.total_seconds() / 60
    assert offsets.tolist() == [5.0 * i for i in range(12)]
    assert (first.available_at == _utc("2024-09-30 14:00:30")).all()


def test_p5min_interval_beyond_55_min_fails(tmp_path: Path) -> None:
    rows = p5min_rows(dt.datetime(2024, 10, 1, 0, 5), 1, regions=("VIC1",))
    rows.append({**rows[-1], "INTERVAL_DATETIME": "2024/10/01 01:05:00"})
    with pytest.raises(MmsFormatError, match="off the run grid"):
        extract_forecasts([_p5_zip(tmp_path, rows)], P5MIN, ["VIC1"])


def test_blank_demand_fails(tmp_path: Path) -> None:
    rows = predispatch_rows(DAY, [1], regions=("VIC1",))
    rows[0]["TOTALDEMAND"] = ""
    with pytest.raises(MmsFormatError, match="blank demand"):
        extract_forecasts([_pd_zip(tmp_path, rows)], PREDISPATCH, ["VIC1"])
