"""AEMO forecast archives: ingestion end to end and the AEMO data-quality checks."""

import datetime as dt
import io
import zipfile
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, text

from gridcast.config import load_config, load_market_limits
from gridcast.database.load import ensure_regions, start_ingestion_run, upsert_market_intervals
from gridcast.ingestion.aemo_forecasts import ingest_aemo_range
from gridcast.ingestion.nemweb import NemwebClient, month_dir_url
from gridcast.validation.checks import AEMO_CHECKS, DQResult, run_checks
from tests.integration.factories import clean_month
from tests.mms_fixtures import (
    P5MIN_COLS,
    PREDISPATCH_COLS,
    mms_lines,
    p5min_rows,
    predispatch_rows,
)

pytestmark = pytest.mark.integration

OCT24 = dt.date(2024, 10, 1)
BASE = "https://nemweb.test/MMSDM"


def _zip(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("X.CSV", "\r\n".join(lines) + "\r\n")
    return buf.getvalue()


def _fake_nemweb() -> httpx.MockTransport:
    """October 2024 with one trading day of pre-dispatch and one day of 5-minute runs."""
    pd_lines = mms_lines(
        "PREDISPATCH", "REGION_SOLUTION", "8", PREDISPATCH_COLS,
        predispatch_rows(OCT24, range(1, 49), periods=4),
    )  # fmt: skip
    p5_lines = mms_lines(
        "P5MIN", "REGIONSOLUTION", "9", P5MIN_COLS,
        p5min_rows(dt.datetime(2024, 10, 1, 0, 5), 288),
    )  # fmt: skip
    files = {
        "PREDISP_ALL_DATA": ("PREDISPATCHREGIONSUM", _zip(pd_lines)),
        "P5MIN_ALL_DATA": ("P5MIN_REGIONSOLUTION", _zip(p5_lines)),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for folder, (table, blob) in files.items():
            page = month_dir_url(BASE, 2024, 10, folder)
            name = f"PUBLIC_ARCHIVE%23{table}%23ALL%23FILE01%23202410010000.zip"
            if url == page:
                return httpx.Response(200, text=f'<A HREF="{httpx.URL(page).path}{name}">x</A>')
            if url == page + name:
                return httpx.Response(200, content=blob)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _ingest(db: Engine, tmp_path: Path) -> dict[str, list[int]]:
    cfg = load_config()
    cfg.paths.interim_dir = tmp_path / "interim"
    src = cfg.sources.mmsdm.model_copy(update={"base_url": BASE, "request_delay_seconds": 0.0})
    client = NemwebClient(src, tmp_path / "raw", transport=_fake_nemweb())
    results = ingest_aemo_range(db, cfg, OCT24, OCT24, ["VIC1"], client=client)
    assert {k: [r.status for r in v] for k, v in results.items()} == {
        "predispatch": ["success"],
        "p5min": ["success"],
    }
    return {k: [r.rows_loaded for r in v] for k, v in results.items()}


def _checks(db: Engine) -> dict[str, DQResult]:
    cfg = load_config()
    results = run_checks(
        db, cfg, load_market_limits(cfg), ["VIC1"], OCT24, OCT24, checks=AEMO_CHECKS, persist=False
    )
    return {r.check_name: r for r in results}


def _scalar(db: Engine, sql: str) -> object:
    with db.connect() as conn:
        return conn.execute(text(sql)).scalar_one()


def test_ingest_aemo_end_to_end_and_idempotent(db: Engine, tmp_path: Path) -> None:
    loaded = _ingest(db, tmp_path)
    assert loaded == {"predispatch": [48 * 4], "p5min": [288 * 12]}
    assert _scalar(db, "SELECT count(DISTINCT region_id) FROM aemo_demand_forecasts") == 1
    first_pd = _scalar(
        db, "SELECT min(run_time) FROM aemo_demand_forecasts WHERE source = 'predispatch'"
    )
    assert first_pd == dt.datetime(2024, 9, 30, 18, 0, tzinfo=dt.UTC)  # 04:00 NEM, 1 Oct
    assert (tmp_path / "interim" / "aemo_p5min" / "202410.parquet").exists()
    # Second run: files from cache, nothing changes.
    assert _ingest(db, tmp_path) == {"predispatch": [0], "p5min": [0]}


def test_aemo_checks(db: Engine, tmp_path: Path) -> None:
    _ingest(db, tmp_path)
    got = _checks(db)
    assert set(got) == {c.name for c in AEMO_CHECKS}
    # One trading day / one day of runs out of 31 are present.
    assert got["aemo_missing_runs"].status == "warn"
    assert got["aemo_missing_runs"].n_violations == 30 * 48 + 30 * 288
    for name in ("aemo_run_timing", "aemo_out_of_range", "aemo_p5min_alignment"):
        assert got[name].status == "pass", name

    # Interval 0 vs dispatch demand: the fixture's values differ everywhere -> fail.
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", OCT24)
        upsert_market_intervals(conn, clean_month(2024, 10), run)
    assert _checks(db)["aemo_p5min_alignment"].status == "fail"
    with db.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE aemo_demand_forecasts f SET total_demand_mw = m.total_demand_mw
                FROM market_intervals m
                WHERE f.source = 'p5min' AND f.target_end = f.run_time
                  AND m.region_id = f.region_id AND m.interval_end = f.target_end
                """
            )
        )
    assert _checks(db)["aemo_p5min_alignment"].status == "pass"

    # A run published late, and a value out of range.
    with db.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE aemo_demand_forecasts SET available_at = run_time + interval '1 hour'
                WHERE source = 'predispatch' AND run_time = '2024-09-30 18:00+00'
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE aemo_demand_forecasts SET total_demand_mw = -5
                WHERE source = 'p5min' AND run_time = '2024-09-30 14:05+00'
                  AND target_end = '2024-09-30 14:10+00'
                """
            )
        )
    got = _checks(db)
    assert got["aemo_run_timing"].status == "warn" and got["aemo_run_timing"].n_violations == 1
    assert got["aemo_out_of_range"].status == "fail" and got["aemo_out_of_range"].n_violations == 1
