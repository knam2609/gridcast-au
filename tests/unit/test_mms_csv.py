import datetime as dt
from pathlib import Path

import pytest

from gridcast.ingestion.mms_csv import MmsFormatError, ReportKey, read_mms_report
from tests.mms_fixtures import REGIONSUM_COLS, mms_lines, regionsum_rows, write_zip

KEY = ReportKey("DISPATCH", "REGIONSUM")
START = dt.datetime(2021, 10, 1, 0, 5)


def _zip(tmp_path: Path, lines: list[str], name: str = "f.zip") -> Path:
    return write_zip(tmp_path / name, "PUBLIC_DVD_DISPATCHREGIONSUM.CSV", lines)


def test_reads_by_column_name_and_filters_region(tmp_path: Path) -> None:
    # Shuffle column order: parsing must rely on the I-row header, not positions.
    cols = list(reversed(REGIONSUM_COLS))
    path = _zip(tmp_path, mms_lines("DISPATCH", "REGIONSUM", "5", cols, regionsum_rows(START, 3)))
    frame, stats = read_mms_report(path, KEY, ["SETTLEMENTDATE", "TOTALDEMAND"], {"VIC1"})
    assert list(frame.columns) == ["SETTLEMENTDATE", "TOTALDEMAND"]
    assert len(frame) == 3  # VIC1 only
    assert frame["TOTALDEMAND"].tolist() == ["5000.00", "5010.00", "5020.00"]
    assert stats.data_rows_seen == 6
    assert stats.data_rows_kept == 3
    assert stats.versions == {"5"}


def test_ignores_other_report_sections(tmp_path: Path) -> None:
    other = mms_lines("DISPATCH", "PRICE", "4", ["SETTLEMENTDATE", "REGIONID"], [])[1:-1]
    lines = mms_lines("DISPATCH", "REGIONSUM", "5", REGIONSUM_COLS, regionsum_rows(START, 2))
    lines = [lines[0], *other, *lines[1:-1]]
    lines.append(f'C,"END OF REPORT",{len(lines) + 1}')
    frame, _ = read_mms_report(_zip(tmp_path, lines), KEY, ["TOTALDEMAND"], {"VIC1"})
    assert len(frame) == 2


def test_missing_footer_is_treated_as_truncation(tmp_path: Path) -> None:
    lines = mms_lines(
        "DISPATCH", "REGIONSUM", "5", REGIONSUM_COLS, regionsum_rows(START, 2), footer=False
    )
    with pytest.raises(MmsFormatError, match="truncated"):
        read_mms_report(_zip(tmp_path, lines), KEY, ["TOTALDEMAND"])


def test_footer_row_count_mismatch_raises(tmp_path: Path) -> None:
    lines = mms_lines(
        "DISPATCH", "REGIONSUM", "5", REGIONSUM_COLS, regionsum_rows(START, 2), footer_override=99
    )
    with pytest.raises(MmsFormatError, match="footer says 99"):
        read_mms_report(_zip(tmp_path, lines), KEY, ["TOTALDEMAND"])


def test_missing_required_column_raises(tmp_path: Path) -> None:
    cols = [c for c in REGIONSUM_COLS if c != "TOTALDEMAND"]
    path = _zip(tmp_path, mms_lines("DISPATCH", "REGIONSUM", "5", cols, regionsum_rows(START, 1)))
    with pytest.raises(MmsFormatError, match="TOTALDEMAND"):
        read_mms_report(path, KEY, ["TOTALDEMAND"])


def test_absent_section_raises(tmp_path: Path) -> None:
    path = _zip(tmp_path, mms_lines("DISPATCH", "PRICE", "4", ["SETTLEMENTDATE"], []))
    with pytest.raises(MmsFormatError, match=r"no DISPATCH\.REGIONSUM section"):
        read_mms_report(path, KEY, ["SETTLEMENTDATE"])
