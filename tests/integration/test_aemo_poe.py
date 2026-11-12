"""PDPASA POE forecasts: ingestion end to end, idempotent (M3 step 5)."""

import datetime as dt
import io
import zipfile
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, text

from gridcast.config import load_config
from gridcast.ingestion.aemo_poe import ingest_poe_range
from gridcast.ingestion.nemweb import NemwebClient, month_dir_url
from tests.mms_fixtures import PDPASA_COLS, mms_lines, pdpasa_rows

pytestmark = pytest.mark.integration

OCT24 = dt.date(2024, 10, 1)
BASE = "https://nemweb.test/MMSDM"


def _fake_nemweb() -> httpx.MockTransport:
    """October 2024 with one day of PDPASA runs (LOR and OUTAGE_LRC) in DATA/."""
    lines = mms_lines(
        "PDPASA", "REGIONSOLUTION", "7", PDPASA_COLS,
        pdpasa_rows(dt.datetime(2024, 10, 1, 0, 30), 48, periods=4),
    )  # fmt: skip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("X.CSV", "\r\n".join(lines) + "\r\n")
    page = month_dir_url(BASE, 2024, 10, "DATA")
    name = "PUBLIC_ARCHIVE%23PDPASA_REGIONSOLUTION%23FILE01%23202410010000.zip"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == page:
            return httpx.Response(200, text=f'<A HREF="{httpx.URL(page).path}{name}">x</A>')
        if url == page + name:
            return httpx.Response(200, content=buf.getvalue())
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _ingest(db: Engine, tmp_path: Path) -> list[int]:
    cfg = load_config()
    cfg.paths.interim_dir = tmp_path / "interim"
    src = cfg.sources.mmsdm.model_copy(update={"base_url": BASE, "request_delay_seconds": 0.0})
    client = NemwebClient(src, tmp_path / "raw", transport=_fake_nemweb())
    results = ingest_poe_range(db, cfg, OCT24, OCT24, ["VIC1"], client=client)
    assert [r.status for r in results] == ["success"]
    return [r.rows_loaded for r in results]


def test_ingest_poe_end_to_end_and_idempotent(db: Engine, tmp_path: Path) -> None:
    assert _ingest(db, tmp_path) == [48 * 4]
    assert _ingest(db, tmp_path) == [0]  # nothing changed, nothing rewritten
    with db.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT count(*) AS n, count(DISTINCT run_time) AS runs,
                       min(target_end - run_time) AS first_lead,
                       max(available_at - run_time) AS published,
                       bool_and(demand10_mw - demand50_mw = demand50_mw - demand90_mw) AS sym
                FROM aemo_demand_poe WHERE region_id = 'VIC1'
                """
            )
        ).one()
    assert (row.n, row.runs) == (192, 48)
    assert row.first_lead == dt.timedelta(minutes=30)
    assert row.published == dt.timedelta(minutes=5)
    assert row.sym
