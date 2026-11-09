"""`ingest_daily`: match check against loaded MMSDM rows, then load days MMSDM lacks."""

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, DailyReportsConfig, load_config
from gridcast.database.load import (
    ensure_regions,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_market_intervals,
)
from gridcast.ingestion.daily import DailyCheckFailed, ingest_daily, month_days
from gridcast.ingestion.nemweb import NemwebClient
from gridcast.ingestion.pipeline import transform_month
from tests.mms_fixtures import (
    PRICE_COLS,
    REGIONSUM_COLS,
    mms_lines,
    price_rows,
    regionsum_rows,
    write_daily_archive,
    write_zip,
)

pytestmark = pytest.mark.integration

LISTING_URL = "https://nemweb.test/Reports/Archive/DispatchIS_Reports/"
CHECK = dt.date(2022, 2, 1)
LOAD = dt.date(2022, 3, 1)
START = dt.datetime(2022, 2, 1, 0, 5)
N_CHECK = 28 * 288


def _rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    n = N_CHECK + 288
    return regionsum_rows(START, n), price_rows(START, n)


def _setup(db: Engine, tmp_path: Path) -> tuple[AppConfig, NemwebClient]:
    """MMSDM-loaded February 2022 plus a fake daily archive for February and 1 March."""
    regionsum, price = _rows()
    # "MMSDM": February from monthly-style files, loaded under an mmsdm ingestion run.
    month = tmp_path / "mmsdm"
    rs = write_zip(
        month / "rs.zip",
        "rs.CSV",
        mms_lines("DISPATCH", "REGIONSUM", "9", REGIONSUM_COLS, regionsum[: 2 * N_CHECK]),
    )
    pr = write_zip(
        month / "pr.zip",
        "pr.CSV",
        mms_lines("DISPATCH", "PRICE", "5", PRICE_COLS, price[: 2 * N_CHECK]),
    )
    frame, _, _ = transform_month([rs], [pr], ["VIC1"])
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", CHECK)
        upsert_market_intervals(conn, frame, run)
        finish_ingestion_run(conn, run, status="success")

    # Daily archives: rows are per (interval, region), two regions per interval.
    blobs: dict[str, bytes] = {}
    for i, day in enumerate([*month_days(CHECK), LOAD]):
        sl = slice(2 * 288 * i, 2 * 288 * (i + 1))
        name = f"PUBLIC_DISPATCHIS_{day:%Y%m%d}.zip"
        path = write_daily_archive(tmp_path / "daily" / name, regionsum[sl], price[sl])
        blobs[name] = path.read_bytes()
    listing = "".join(f'<A HREF="/Reports/ARCHIVE/DispatchIS_Reports/{n}">{n}</A>' for n in blobs)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == LISTING_URL:
            return httpx.Response(200, text=listing)
        name = url.rsplit("/", 1)[-1]
        return httpx.Response(200, content=blobs[name]) if name in blobs else httpx.Response(404)

    cfg.sources.nemweb_daily = DailyReportsConfig(listing_url=LISTING_URL)
    cfg.paths.interim_dir = tmp_path / "interim"
    src = cfg.sources.mmsdm.model_copy(update={"request_delay_seconds": 0.0})
    client = NemwebClient(src, tmp_path / "raw", transport=httpx.MockTransport(handler))
    return cfg, client


def _count(db: Engine, sql: str) -> int:
    with db.connect() as conn:
        return int(conn.execute(text(sql)).scalar_one())


def test_match_check_then_load(db: Engine, tmp_path: Path) -> None:
    cfg, client = _setup(db, tmp_path)
    result = ingest_daily(db, cfg, [LOAD], CHECK, ["VIC1"], client=client)
    check = result.check
    assert check.passed
    assert (check.days, check.only_daily, check.only_loaded) == (28, 0, 0)
    assert check.loaded_sources == ["mmsdm"]
    assert all(c.differing == 0 and c.compared == N_CHECK for c in check.columns)
    assert result.loaded == {"2022-03-01": 288}
    assert _count(db, "SELECT count(*) FROM market_intervals WHERE region_id = 'VIC1'") == (
        N_CHECK + 288
    )
    assert (
        _count(
            db,
            """SELECT count(*) FROM market_intervals m JOIN ingestion_runs r
           ON r.run_id = m.ingestion_run_id
           WHERE r.source = 'nemweb_daily' AND r.status = 'success'""",
        )
        == 288
    )
    saved = json.loads((tmp_path / "interim" / "nemweb_daily" / "check_2022-02.json").read_text())
    assert saved["check"]["passed"] and saved["loaded"] == {"2022-03-01": 288}
    assert len(saved["files"]) == 28

    # Idempotent: a second run reloads its own rows and changes nothing.
    again = ingest_daily(db, cfg, [LOAD], CHECK, ["VIC1"], client=client)
    assert again.check.passed and again.loaded == {"2022-03-01": 288}


def test_demand_mismatch_loads_nothing(db: Engine, tmp_path: Path) -> None:
    cfg, client = _setup(db, tmp_path)
    with db.begin() as conn:
        conn.execute(
            text(
                """UPDATE market_intervals SET total_demand_mw = total_demand_mw + 0.01
                   WHERE region_id = 'VIC1' AND interval_end = '2022-02-14 03:00+00'"""
            )
        )
    with pytest.raises(DailyCheckFailed) as err:
        ingest_daily(db, cfg, [LOAD], CHECK, ["VIC1"], client=client)
    demand = next(c for c in err.value.check.columns if c.column == "total_demand_mw")
    assert demand.differing == 1
    assert _count(db, "SELECT count(*) FROM ingestion_runs WHERE source = 'nemweb_daily'") == 0
    saved = json.loads((tmp_path / "interim" / "nemweb_daily" / "check_2022-02.json").read_text())
    assert not saved["check"]["passed"] and saved["loaded"] == {}


def test_price_difference_is_reported_but_does_not_block(db: Engine, tmp_path: Path) -> None:
    cfg, client = _setup(db, tmp_path)
    with db.begin() as conn:
        conn.execute(text("UPDATE market_intervals SET rrp_aud_mwh = rrp_aud_mwh + 1"))
    result = ingest_daily(db, cfg, [LOAD], CHECK, ["VIC1"], client=client)
    price = next(c for c in result.check.columns if c.column == "rrp_aud_mwh")
    assert result.check.passed and price.differing == N_CHECK


def test_refuses_to_overwrite_mmsdm_or_to_load_inside_the_check_month(
    db: Engine, tmp_path: Path
) -> None:
    cfg, client = _setup(db, tmp_path)
    with pytest.raises(ValueError, match="outside the check month"):
        ingest_daily(db, cfg, [dt.date(2022, 2, 10)], CHECK, ["VIC1"], client=client)
    # 1 March already loaded from MMSDM: MMSDM stays authoritative.
    regionsum, price = _rows()
    rs = write_zip(
        tmp_path / "m2" / "rs.zip",
        "rs.CSV",
        mms_lines("DISPATCH", "REGIONSUM", "9", REGIONSUM_COLS, regionsum[2 * N_CHECK :]),
    )
    pr = write_zip(
        tmp_path / "m2" / "pr.zip",
        "pr.CSV",
        mms_lines("DISPATCH", "PRICE", "5", PRICE_COLS, price[2 * N_CHECK :]),
    )
    frame, _, _ = transform_month([rs], [pr], ["VIC1"])
    with db.begin() as conn:
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", LOAD)
        upsert_market_intervals(conn, frame, run)
    with pytest.raises(ValueError, match="already loaded from MMSDM"):
        ingest_daily(db, cfg, [LOAD], CHECK, ["VIC1"], client=client)
