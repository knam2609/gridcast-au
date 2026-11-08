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


# AEMO forecast archives (M2 benchmark). Real files carry ~120 columns; mapping is by name.
PREDISPATCH_COLS = [
    "PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID", "INTERVENTION", "TOTALDEMAND",
    "AVAILABLEGENERATION", "LASTCHANGED", "DATETIME",
]  # fmt: skip
P5MIN_COLS = [
    "RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID", "RRP", "TOTALDEMAND", "LASTCHANGED",
    "INTERVENTION",
]  # fmt: skip


def predispatch_rows(
    trading_day: dt.date,
    runs: Sequence[int],
    periods: int = 4,
    regions: Sequence[str] = ("NSW1", "VIC1"),
) -> list[dict[str, str]]:
    """Runs ``PP`` of a trading day: run time 04:00 NEM + (PP-1) x 30 min, periods 30 min apart."""
    rows = []
    for pp in runs:
        run_time = dt.datetime.combine(trading_day, dt.time(4)) + dt.timedelta(
            minutes=30 * (pp - 1)
        )
        for k in range(1, periods + 1):
            for j, region in enumerate(regions):
                rows.append(
                    {
                        "PREDISPATCHSEQNO": f"{trading_day:%Y%m%d}{pp:02d}",
                        "RUNNO": "1",
                        "REGIONID": region,
                        "PERIODID": f"{k:02d}",
                        "INTERVENTION": "0",
                        "TOTALDEMAND": f"{4000 + 10 * k + 1000 * j + pp:.2f}",
                        "AVAILABLEGENERATION": "9000",
                        "LASTCHANGED": nem(run_time + dt.timedelta(seconds=134)),
                        "DATETIME": nem(run_time + dt.timedelta(minutes=30 * k)),
                    }
                )
    return rows


def p5min_rows(
    first_run: dt.datetime, n_runs: int, regions: Sequence[str] = ("NSW1", "VIC1")
) -> list[dict[str, str]]:
    """5-minute runs every 5 min from ``first_run`` (NEM time), intervals 0..11."""
    rows = []
    for r in range(n_runs):
        run_time = first_run + dt.timedelta(minutes=5 * r)
        for i in range(12):
            for j, region in enumerate(regions):
                rows.append(
                    {
                        "RUN_DATETIME": nem(run_time),
                        "INTERVAL_DATETIME": nem(run_time + dt.timedelta(minutes=5 * i)),
                        "REGIONID": region,
                        "RRP": "50",
                        "TOTALDEMAND": f"{4000 + i + 1000 * j + r:.2f}",
                        "LASTCHANGED": nem(run_time - dt.timedelta(seconds=270)),
                        "INTERVENTION": "0",
                    }
                )
    return rows
