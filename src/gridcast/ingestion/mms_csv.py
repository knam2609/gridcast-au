"""Streaming parser for AEMO's MMS "CSV" report format.

Every row starts with a record type:

* ``C`` — comment. The first row is a file header; the last is ``C,"END OF REPORT",<n>``
  where ``<n>`` is the total number of rows in the file (used as an integrity check).
* ``I`` — column header: ``I,<package>,<report>,<version>,<col1>,<col2>,…``
* ``D`` — data row laid out like the preceding ``I`` row.

Columns are always mapped *by name* from the ``I`` row because table versions (and hence
column positions) change over time.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class MmsFormatError(ValueError):
    """The file does not follow the MMS CSV report format (or is truncated)."""


@dataclass(frozen=True, slots=True)
class ReportKey:
    """Identifies one report section within an MMS file, e.g. ``("DISPATCH", "PRICE")``."""

    package: str
    report: str


@dataclass(slots=True)
class ParseStats:
    total_rows: int = 0
    data_rows_seen: int = 0
    data_rows_kept: int = 0
    versions: set[str] = field(default_factory=set)
    members: int = 0


def _zip_members(zf: zipfile.ZipFile, label: str) -> Iterator[tuple[str, Iterator[list[str]]]]:
    """CSV members of an open zip, then those of every zip nested in it (in name order).

    NEMWeb's daily report archives are zips of zips: one inner zip (holding one CSV) per
    five-minute report.
    """
    names = zf.namelist()
    csvs = [n for n in names if n.upper().endswith(".CSV")]
    nested = sorted(n for n in names if n.upper().endswith(".ZIP"))
    if not csvs and not nested:
        raise MmsFormatError(f"{label}: zip contains no .CSV member")
    for name in csvs:
        with zf.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="latin-1", newline="")
            yield name, csv.reader(text)
    for name in nested:
        with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
            yield from _zip_members(inner, name)


def _iter_members(path: Path) -> Iterator[tuple[str, Iterator[list[str]]]]:
    """Yield ``(member_name, csv_rows)`` for a plain CSV or a zip of CSVs (or of such zips)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            yield from _zip_members(zf, path.name)
    else:
        with path.open(encoding="latin-1", newline="") as fh:
            yield path.name, csv.reader(fh)


def read_mms_report(
    path: Path,
    key: ReportKey,
    columns: Collection[str],
    region_filter: Collection[str] | None = None,
    region_column: str = "REGIONID",
) -> tuple[pd.DataFrame, ParseStats]:
    """Read one report section from an MMS file, keeping only ``columns`` (as strings).

    Rows are filtered by ``region_filter`` while streaming to keep memory small.
    Raises :class:`MmsFormatError` if required columns are missing, the section is absent,
    or the ``END OF REPORT`` footer is missing / disagrees with the number of rows read
    (a truncated download).
    """
    stats = ParseStats()
    wanted = list(columns)
    regions = set(region_filter) if region_filter is not None else None
    out: list[list[str]] = []

    idx: list[int] = []
    region_idx: int | None = None
    section_found = False

    for member, rows in _iter_members(path):
        stats.members += 1
        member_rows = 0
        footer: int | None = None
        in_section = False
        for row in rows:
            if not row:
                continue
            member_rows += 1
            kind = row[0]
            if kind == "I":
                in_section = len(row) > 4 and row[1] == key.package and row[2] == key.report
                if in_section:
                    section_found = True
                    stats.versions.add(row[3])
                    header = row[4:]
                    missing = [c for c in wanted if c not in header]
                    if missing:
                        raise MmsFormatError(
                            f"{member}: {key.package}.{key.report} v{row[3]} lacks {missing}"
                        )
                    # +4 offsets the record-type/package/report/version prefix.
                    idx = [header.index(c) + 4 for c in wanted]
                    region_idx = header.index(region_column) + 4 if regions is not None else None
            elif kind == "D":
                if not in_section:
                    continue
                stats.data_rows_seen += 1
                if (
                    region_idx is not None
                    and regions is not None
                    and row[region_idx] not in regions
                ):
                    continue
                out.append([row[i] for i in idx])
            elif kind == "C" and len(row) >= 3 and row[1] == "END OF REPORT":
                try:
                    footer = int(row[2])
                except ValueError as exc:
                    raise MmsFormatError(f"{member}: unparseable footer {row!r}") from exc
        stats.total_rows += member_rows
        if footer is None:
            raise MmsFormatError(f"{member}: missing END OF REPORT footer (truncated file?)")
        if footer != member_rows:
            raise MmsFormatError(f"{member}: footer says {footer} rows but {member_rows} were read")

    if not section_found:
        raise MmsFormatError(f"{path.name}: no {key.package}.{key.report} section found")

    stats.data_rows_kept = len(out)
    frame = pd.DataFrame(out, columns=wanted, dtype="string")
    return frame, stats
