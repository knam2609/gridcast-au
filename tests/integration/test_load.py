import datetime as dt
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, text

from gridcast.config import load_config
from gridcast.database.load import (
    ensure_regions,
    start_ingestion_run,
    upsert_market_intervals,
)
from gridcast.ingestion.nemweb import NemwebClient, month_dir_url
from gridcast.ingestion.pipeline import ingest_range
from tests.integration.factories import clean_month
from tests.mms_fixtures import (
    PRICE_COLS,
    REGIONSUM_COLS,
    mms_lines,
    price_rows,
    regionsum_rows,
)

pytestmark = pytest.mark.integration


def _count(engine: Engine, sql: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(sql)).scalar_one())


def test_upsert_is_idempotent_and_applies_revisions(db: Engine) -> None:
    cfg = load_config()
    frame = clean_month()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "test", "market_intervals", dt.date(2021, 10, 1))
        assert upsert_market_intervals(conn, frame, run) == len(frame)
    with db.begin() as conn:
        assert upsert_market_intervals(conn, frame, run) == 0  # unchanged rows are skipped

    revised = frame.copy()
    revised.loc[10, "rrp_aud_mwh"] = 999.0
    with db.begin() as conn:
        assert upsert_market_intervals(conn, revised, run) == 1
    assert _count(db, "SELECT count(*) FROM market_intervals") == len(frame)
    assert _count(db, "SELECT count(*) FROM market_intervals WHERE rrp_aud_mwh = 999") == 1


def test_null_values_round_trip(db: Engine) -> None:
    cfg = load_config()
    frame = clean_month().head(5)
    frame.loc[0, "rrp_aud_mwh"] = float("nan")
    frame.loc[1, "price_status"] = None
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "test", "market_intervals", dt.date(2021, 10, 1))
        upsert_market_intervals(conn, frame, run)
    assert _count(db, "SELECT count(*) FROM market_intervals WHERE rrp_aud_mwh IS NULL") == 1
    assert _count(db, "SELECT count(*) FROM market_intervals WHERE price_status IS NULL") == 1


def test_db_rejects_impossible_price(db: Engine) -> None:
    cfg = load_config()
    frame = clean_month().head(1)
    frame.loc[0, "rrp_aud_mwh"] = -5000.0
    with pytest.raises(Exception, match="rrp_sanity"), db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "test", "market_intervals", dt.date(2021, 10, 1))
        upsert_market_intervals(conn, frame, run)


def _fake_nemweb(base_url: str) -> httpx.MockTransport:
    """Serve October 2021 (first 2 days) and 404 for November 2021."""
    start = dt.datetime(2021, 10, 1, 0, 5)
    files = {
        "PUBLIC_DVD_DISPATCHREGIONSUM_202110010000.zip": mms_lines(
            "DISPATCH", "REGIONSUM", "5", REGIONSUM_COLS, regionsum_rows(start, 576)
        ),
        "PUBLIC_DVD_DISPATCHPRICE_202110010000.zip": mms_lines(
            "DISPATCH", "PRICE", "4", PRICE_COLS, price_rows(start, 576)
        ),
    }
    import io
    import zipfile

    blobs: dict[str, bytes] = {}
    for name, lines in files.items():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name.replace(".zip", ".CSV"), "\r\n".join(lines) + "\r\n")
        blobs[name] = buf.getvalue()
    oct_dir = month_dir_url(base_url, 2021, 10)
    listing = "".join(f'<A HREF="{httpx.URL(oct_dir).path}{n}">{n}</A>' for n in blobs)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == oct_dir:
            return httpx.Response(200, text=listing)
        for name, blob in blobs.items():
            if url.endswith(name):
                return httpx.Response(200, content=blob)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_ingest_range_end_to_end_with_failure_recorded(db: Engine, tmp_path: Path) -> None:
    cfg = load_config()
    cfg.paths.interim_dir = tmp_path / "interim"
    src = cfg.sources.mmsdm.model_copy(
        update={"base_url": "https://nemweb.test/MMSDM", "request_delay_seconds": 0.0}
    )
    client = NemwebClient(src, tmp_path / "raw", transport=_fake_nemweb(src.base_url))
    results = ingest_range(
        db, cfg, dt.date(2021, 10, 1), dt.date(2021, 11, 1), ["VIC1"], client=client
    )
    assert [r.status for r in results] == ["success", "failed"]
    assert results[0].rows_loaded == 576
    assert "404" in (results[1].error or "")
    assert _count(db, "SELECT count(*) FROM market_intervals") == 576
    assert _count(db, "SELECT count(*) FROM ingestion_runs WHERE status = 'failed'") == 1
    assert (tmp_path / "interim" / "market_intervals" / "202110.parquet").exists()

    # Second run: everything from cache, nothing changes.
    again = ingest_range(
        db, cfg, dt.date(2021, 10, 1), dt.date(2021, 10, 1), ["VIC1"], client=client
    )
    assert again[0].rows_loaded == 0
    assert _count(db, "SELECT count(*) FROM market_intervals") == 576
