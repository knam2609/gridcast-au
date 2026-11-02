"""Builders for synthetic MMS-format files used across tests."""

from __future__ import annotations

import datetime as dt
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

REGIONSUM_COLS = [
    "SETTLEMENTDATE", "RUNNO", "REGIONID", "DISPATCHINTERVAL", "INTERVENTION", "TOTALDEMAND",
    "AVAILABLEGENERATION", "NETINTERCHANGE", "LASTCHANGED",
]  # fmt: skip
PRICE_COLS = [
    "SETTLEMENTDATE", "RUNNO", "REGIONID", "DISPATCHINTERVAL", "INTERVENTION", "RRP", "EEP",
    "APCFLAG", "MARKETSUSPENDEDFLAG", "LASTCHANGED", "PRICE_STATUS",
]  # fmt: skip


def nem(ts: dt.datetime) -> str:
    return ts.strftime("%Y/%m/%d %H:%M:%S")


def mms_lines(
    package: str,
    report: str,
    version: str,
    columns: Sequence[str],
    rows: Iterable[dict[str, str]],
    *,
    footer: bool = True,
    footer_override: int | None = None,
) -> list[str]:
    lines = [f"C,SETP.WORLD,DVD_{package}{report},AEMO,PUBLIC,2021/11/08,00:40:06,1,,1"]
    lines.append(",".join(["I", package, report, version, *columns]))
    for row in rows:
        values = [row.get(c, "") for c in columns]
        lines.append(",".join(["D", package, report, version, *[f'"{v}"' for v in values]]))
    if footer:
        n = footer_override if footer_override is not None else len(lines) + 1
        lines.append(f'C,"END OF REPORT",{n}')
    return lines


def write_zip(path: Path, member: str, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, "\r\n".join(lines) + "\r\n")
    return path


def regionsum_rows(
    start_end: dt.datetime, n: int, regions: Sequence[str] = ("NSW1", "VIC1"), base: float = 4000
) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        end = start_end + dt.timedelta(minutes=5 * i)
        for j, region in enumerate(regions):
            rows.append(
                {
                    "SETTLEMENTDATE": nem(end),
                    "RUNNO": "1",
                    "REGIONID": region,
                    "DISPATCHINTERVAL": "1",
                    "INTERVENTION": "0",
                    "TOTALDEMAND": f"{base + 10 * i + 1000 * j:.2f}",
                    "AVAILABLEGENERATION": "9000",
                    "NETINTERCHANGE": "-50.5",
                    "LASTCHANGED": nem(end - dt.timedelta(seconds=5)),
                }
            )
    return rows


def price_rows(
    start_end: dt.datetime, n: int, regions: Sequence[str] = ("NSW1", "VIC1")
) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        end = start_end + dt.timedelta(minutes=5 * i)
        for region in regions:
            rows.append(
                {
                    "SETTLEMENTDATE": nem(end),
                    "RUNNO": "1",
                    "REGIONID": region,
                    "DISPATCHINTERVAL": "1",
                    "INTERVENTION": "0",
                    "RRP": f"{50 + i:.5f}",
                    "EEP": "0",
                    "APCFLAG": "0",
                    "MARKETSUSPENDEDFLAG": "0",
                    "LASTCHANGED": nem(end - dt.timedelta(seconds=5)),
                    "PRICE_STATUS": "FIRM",
                }
            )
    return rows
