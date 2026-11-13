"""PDPASA POE forecasts (M3 step 5): parsing and the loader's structural checks."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from gridcast.ingestion.aemo_poe import extract_poe
from gridcast.ingestion.mms_csv import MmsFormatError
from tests.mms_fixtures import PDPASA_COLS, mms_lines, pdpasa_rows, write_zip

FIRST = dt.datetime(2024, 10, 1, 0, 30)  # RUN_DATETIME of the first run (NEM time)


def _zip(tmp_path: Path, rows: list[dict[str, str]], version: str = "7", cols=PDPASA_COLS) -> Path:
    lines = mms_lines("PDPASA", "REGIONSOLUTION", version, cols, rows)
    return write_zip(tmp_path / f"pdpasa_{version}.zip", "PDPASA.CSV", lines)


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_extract_keeps_lor_runs_of_the_region_and_derives_times(tmp_path: Path) -> None:
    ext = extract_poe([_zip(tmp_path, pdpasa_rows(FIRST, 2, periods=3))], ["VIC1"])
    f = ext.frame
    assert len(f) == 6 and set(f.region_id) == {"VIC1"}
    assert ext.other_run_type_rows == 6 and ext.rows_parsed == 12 and ext.versions == {"7"}
    # RUN_DATETIME 00:30 NEM (14:30 UTC, 30 Sep) ends the first period; run time is 30 min before.
    run1 = f[f.run_time == _utc("2024-09-30 14:00")]
    assert list(run1.target_end) == [
        _utc(t) for t in ("2024-09-30 14:30", "2024-09-30 15:00", "2024-09-30 15:30")
    ]
    assert (run1.available_at == _utc("2024-09-30 14:05")).all()
    assert run1.demand50_mw.tolist() == [5000.0, 5010.0, 5020.0]
    assert run1.demand10_mw.tolist() == [5050.0, 5070.0, 5090.0]
    assert run1.demand90_mw.tolist() == [4950.0, 4950.0, 4950.0]


def test_columns_are_mapped_by_name(tmp_path: Path) -> None:
    cols = [*PDPASA_COLS[:3], "NEWCOLUMN", *PDPASA_COLS[3:]]
    rows = [{**r, "NEWCOLUMN": "1"} for r in pdpasa_rows(FIRST, 1)]
    ext = extract_poe([_zip(tmp_path, rows, cols=cols)], ["VIC1"])
    assert len(ext.frame) == 4


def test_unknown_version_fails(tmp_path: Path) -> None:
    with pytest.raises(MmsFormatError, match="unsupported version"):
        extract_poe([_zip(tmp_path, pdpasa_rows(FIRST, 1), version="8")], ["VIC1"])


def test_out_of_order_band_fails(tmp_path: Path) -> None:
    rows = pdpasa_rows(FIRST, 1, regions=("VIC1",), run_types=("LOR",))
    rows[1] = {**rows[1], "DEMAND90": str(float(rows[1]["DEMAND50"]) + 1)}
    with pytest.raises(MmsFormatError, match="POE10 < POE50 < POE90"):
        extract_poe([_zip(tmp_path, rows)], ["VIC1"])


def test_first_period_must_end_at_run_datetime(tmp_path: Path) -> None:
    rows = pdpasa_rows(FIRST, 1, regions=("VIC1",), run_types=("LOR",))
    later = (FIRST + dt.timedelta(minutes=30)).strftime("%Y/%m/%d %H:%M:%S")
    rows = [{**r, "RUN_DATETIME": later} if i == 0 else r for i, r in enumerate(rows)]
    with pytest.raises(MmsFormatError, match="off the run grid"):
        extract_poe([_zip(tmp_path, rows)], ["VIC1"])


def test_several_publication_times_in_a_run_fail(tmp_path: Path) -> None:
    rows = pdpasa_rows(FIRST, 1, regions=("VIC1",), run_types=("LOR",))
    rows[2] = {**rows[2], "LASTCHANGED": "2024/10/01 00:09:00"}
    with pytest.raises(MmsFormatError, match="several LASTCHANGED"):
        extract_poe([_zip(tmp_path, rows)], ["VIC1"])


def test_duplicate_key_fails(tmp_path: Path) -> None:
    rows = pdpasa_rows(FIRST, 1, regions=("VIC1",), run_types=("LOR",))
    with pytest.raises(MmsFormatError, match="share a key"):
        extract_poe([_zip(tmp_path, [*rows, rows[0]])], ["VIC1"])
