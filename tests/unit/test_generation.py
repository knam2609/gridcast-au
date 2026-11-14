"""Supply totals from DISPATCHREGIONSUM (M4 step 1b): version rules and the long layout."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.database.load import GENERATION_COLUMNS
from gridcast.ingestion.generation import extract_generation
from gridcast.ingestion.mms_csv import MmsFormatError
from tests.mms_fixtures import (
    GENERATION_V5_COLS,
    GENERATION_V9_COLS,
    generation_rows,
    mms_lines,
    write_zip,
)

START = dt.datetime(2024, 6, 1, 0, 5)  # NEM time


def _zip(
    tmp_path: Path,
    rows: list[dict[str, str]],
    version: str = "9",
    cols: list[str] | None = None,
    name: str = "g.zip",
) -> Path:
    columns = cols or (GENERATION_V5_COLS if version == "5" else GENERATION_V9_COLS)
    lines = mms_lines("DISPATCH", "REGIONSUM", version, columns, rows)
    return write_zip(tmp_path / name, "PUBLIC_DVD_DISPATCHREGIONSUM.CSV", lines)


def _kind(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    return frame[frame["kind"] == kind].reset_index(drop=True)


def test_version_9_gives_four_kinds_per_interval(tmp_path: Path) -> None:
    extract = extract_generation([_zip(tmp_path, generation_rows(START, 3))], ["VIC1"])
    frame = extract.frame
    assert list(frame.columns) == list(GENERATION_COLUMNS)
    assert frame.groupby("kind").size().to_dict() == {
        "battery": 3, "dispatchable": 3, "solar": 3, "wind": 3,
    }  # fmt: skip
    wind = _kind(frame, "wind")
    assert wind["cleared_mw"].tolist() == [800.0, 801.0, 802.0]
    assert wind["uigf_mw"].tolist() == [900.0, 901.0, 902.0]
    assert wind["available_mw"].tolist() == [890.0, 891.0, 892.0]
    assert wind["load_mw"].isna().all()
    assert _kind(frame, "dispatchable")["cleared_mw"].tolist() == [4000.0, 4001.0, 4002.0]
    battery = _kind(frame, "battery")
    assert battery["cleared_mw"].tolist() == [0.0, 10.0, 20.0]
    assert battery["load_mw"].tolist() == [0.0, 0.0, 0.0]
    assert battery[["uigf_mw", "available_mw"]].isna().all().all()
    # NEM time (UTC+10) converted to UTC interval ends.
    assert wind["interval_end"].iloc[0] == pd.Timestamp("2024-05-31 14:05", tz="UTC")
    assert extract.versions == {"9"}


def test_version_5_has_no_availability_and_no_battery(tmp_path: Path) -> None:
    frame = extract_generation(
        [_zip(tmp_path, generation_rows(START, 2), version="5")], ["VIC1"]
    ).frame
    assert set(frame["kind"]) == {"wind", "solar", "dispatchable"}
    assert _kind(frame, "solar")["uigf_mw"].tolist() == [90.0, 91.0]
    assert frame["available_mw"].isna().all()


def test_later_version_must_carry_its_columns(tmp_path: Path) -> None:
    path = _zip(tmp_path, generation_rows(START, 2), version="8", cols=GENERATION_V5_COLS)
    with pytest.raises(MmsFormatError, match="v8 lacks"):
        extract_generation([path], ["VIC1"])


def test_unknown_version_fails(tmp_path: Path) -> None:
    with pytest.raises(MmsFormatError, match="unsupported"):
        extract_generation([_zip(tmp_path, generation_rows(START, 2), version="7")], ["VIC1"])


def test_blank_availability_fails_in_later_versions(tmp_path: Path) -> None:
    rows = generation_rows(START, 2)
    rows[1]["SS_WIND_AVAILABILITY"] = ""  # VIC1, first interval
    with pytest.raises(MmsFormatError, match="blank SS_WIND_AVAILABILITY"):
        extract_generation([_zip(tmp_path, rows)], ["VIC1"])


def test_blank_battery_rows_are_skipped(tmp_path: Path) -> None:
    rows = generation_rows(START, 2, battery=False) + generation_rows(
        START + dt.timedelta(minutes=10), 1
    )
    frame = extract_generation([_zip(tmp_path, rows)], ["VIC1"]).frame
    assert _kind(frame, "battery")["interval_end"].tolist() == [
        pd.Timestamp("2024-05-31 14:15", tz="UTC")
    ]
    assert len(_kind(frame, "wind")) == 3


def test_battery_generation_without_load_fails(tmp_path: Path) -> None:
    rows = generation_rows(START, 2)
    rows[1]["BDU_CLEAREDMW_LOAD"] = ""
    with pytest.raises(MmsFormatError, match="only one of BDU"):
        extract_generation([_zip(tmp_path, rows)], ["VIC1"])


def test_blank_wind_fails(tmp_path: Path) -> None:
    rows = generation_rows(START, 2, battery=False)
    rows[3]["SS_WIND_CLEAREDMW"] = ""
    with pytest.raises(MmsFormatError, match="blank values"):
        extract_generation([_zip(tmp_path, rows, version="5")], ["VIC1"])


def test_intervention_rows_dropped_and_duplicates_keep_latest(tmp_path: Path) -> None:
    rows = generation_rows(START, 1, regions=("VIC1",))
    intervention = {**rows[0], "INTERVENTION": "1", "SS_WIND_CLEAREDMW": "1"}
    rerun = {
        **rows[0],
        "RUNNO": "2",
        "LASTCHANGED": "2024/06/01 00:04:59",
        "SS_WIND_CLEAREDMW": "555",
    }
    extract = extract_generation([_zip(tmp_path, [*rows, intervention, rerun])], ["VIC1"])
    assert extract.intervention_rows_dropped == 1
    assert extract.duplicates_dropped == 1
    assert _kind(extract.frame, "wind")["cleared_mw"].tolist() == [555.0]


def test_mixed_version_months_concatenate(tmp_path: Path) -> None:
    v5 = _zip(tmp_path, generation_rows(START, 1), version="5", name="a.zip")
    v9 = _zip(tmp_path, generation_rows(START + dt.timedelta(minutes=5), 1), name="b.zip")
    extract = extract_generation([v5, v9], ["VIC1"])
    wind = _kind(extract.frame, "wind")
    assert extract.versions == {"5", "9"}
    assert np.isnan(wind["available_mw"].iloc[0])
    assert wind["available_mw"].iloc[1] == 890.0
