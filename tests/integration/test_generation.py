"""Supply totals (M4 step 1b): ingestion, the 0007 schema, data-quality checks, daily days."""

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from gridcast.config import AppConfig, DailyReportsConfig, load_config, load_market_limits
from gridcast.database.load import ensure_regions
from gridcast.ingestion.daily import month_days
from gridcast.ingestion.generation import (
    GenerationCheckFailed,
    ingest_generation_daily,
    ingest_generation_range,
)
from gridcast.ingestion.nemweb import NemwebClient, month_dir_url
from gridcast.validation.checks import GENERATION_CHECKS, DQResult, run_checks
from tests.mms_fixtures import (
    GENERATION_V5_COLS,
    GENERATION_V9_COLS,
    generation_rows,
    mms_lines,
    write_daily_archive,
    write_zip,
)

pytestmark = pytest.mark.integration

BASE = "https://nemweb.test/MMSDM"
LISTING_URL = "https://nemweb.test/Reports/Archive/DispatchIS_Reports/"
FEB = dt.date(2022, 2, 1)
MAR1 = dt.date(2022, 3, 1)
START = dt.datetime(2022, 2, 1, 0, 5)  # NEM time
N_FEB = 28 * 288


def _client(
    tmp_path: Path, version: str, daily: dict[str, bytes] | None = None
) -> tuple[AppConfig, NemwebClient]:
    """A fake NEMWeb serving February 2022 (both regions) and, optionally, daily archives."""
    cols = GENERATION_V5_COLS if version == "5" else GENERATION_V9_COLS
    lines = mms_lines("DISPATCH", "REGIONSUM", version, cols, generation_rows(START, N_FEB))
    blob = write_zip(tmp_path / "src" / "m.zip", "M.CSV", lines).read_bytes()
    page = month_dir_url(BASE, 2022, 2, "DATA")
    name = "PUBLIC_DVD_DISPATCHREGIONSUM_202202010000.zip"
    listing = "".join(f'<A HREF="/Reports/x/{n}">{n}</A>' for n in daily or {})

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == page:
            return httpx.Response(200, text=f'<A HREF="{httpx.URL(page).path}{name}">x</A>')
        if url == page + name:
            return httpx.Response(200, content=blob)
        if url == LISTING_URL:
            return httpx.Response(200, text=listing)
        file = url.rsplit("/", 1)[-1]
        if daily and file in daily:
            return httpx.Response(200, content=daily[file])
        return httpx.Response(404)

    cfg = load_config()
    cfg.paths.interim_dir = tmp_path / "interim"
    cfg.sources.nemweb_daily = DailyReportsConfig(listing_url=LISTING_URL)
    src = cfg.sources.mmsdm.model_copy(update={"base_url": BASE, "request_delay_seconds": 0.0})
    return cfg, NemwebClient(src, tmp_path / "raw", transport=httpx.MockTransport(handler))


def _ingest(db: Engine, tmp_path: Path, version: str = "5") -> int:
    cfg, client = _client(tmp_path, version)
    results = ingest_generation_range(db, cfg, FEB, FEB, ["VIC1"], client=client)
    assert [r.status for r in results] == ["success"], results
    return results[0].rows_loaded


def _scalar(db: Engine, sql: str) -> object:
    with db.connect() as conn:
        return conn.execute(text(sql)).scalar_one()


def _sql(db: Engine, sql: str) -> None:
    with db.begin() as conn:
        conn.execute(text(sql))


def _dq(db: Engine) -> dict[str, DQResult]:
    cfg = load_config()
    results = run_checks(
        db, cfg, load_market_limits(cfg), ["VIC1"], FEB, FEB, checks=GENERATION_CHECKS
    )
    return {r.check_name: r for r in results}


def test_ingest_end_to_end_and_idempotent(db: Engine, tmp_path: Path) -> None:
    assert _ingest(db, tmp_path) == 3 * N_FEB  # version 5: wind, solar, dispatchable
    assert _ingest(db, tmp_path) == 0  # nothing changed, nothing rewritten
    with db.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT count(*) FILTER (WHERE kind = 'wind') AS wind,
                       count(*) FILTER (WHERE kind = 'battery') AS battery,
                       count(available_mw) AS available,
                       min(interval_end) AS first_end,
                       max(cleared_mw) FILTER (WHERE kind = 'dispatchable') AS max_disp
                FROM generation_intervals WHERE region_id = 'VIC1'
                """
            )
        ).one()
    assert (row.wind, row.battery, row.available) == (N_FEB, 0, 0)
    assert row.first_end == dt.datetime(2022, 1, 31, 14, 5, tzinfo=dt.UTC)
    assert row.max_disp == 4000 + N_FEB - 1
    runs = "SELECT count(*) FROM ingestion_runs WHERE dataset = 'generation_intervals'"
    assert _scalar(db, runs) == 2


def test_clean_month_passes_and_findings_are_flagged(db: Engine, tmp_path: Path) -> None:
    _ingest(db, tmp_path, version="9")
    assert {n: r.status for n, r in _dq(db).items()} == {
        "generation_missing": "pass",
        "generation_negative": "pass",
        "generation_above_uigf": "pass",
    }
    # Two missing wind intervals in a row, one battery interval after its first row.
    _sql(
        db,
        """DELETE FROM generation_intervals WHERE (kind = 'wind' AND interval_end IN
           ('2022-02-10 00:05+00', '2022-02-10 00:10+00'))
           OR (kind = 'battery' AND interval_end = '2022-02-11 00:05+00')""",
    )
    _sql(db, "UPDATE generation_intervals SET load_mw = -1 WHERE kind = 'battery' "
             "AND interval_end = '2022-02-12 00:05+00'")  # fmt: skip
    _sql(db, "UPDATE generation_intervals SET cleared_mw = uigf_mw + 5 WHERE kind = 'solar' "
             "AND interval_end = '2022-02-13 00:05+00'")  # fmt: skip
    r = _dq(db)
    assert r["generation_missing"].status == "fail"
    assert r["generation_missing"].n_violations == 3
    assert {e["kind"]: e["n_missing"] for e in r["generation_missing"].details["examples"]} == {
        "battery": 1,
        "wind": 2,
    }
    assert (r["generation_negative"].status, r["generation_negative"].n_violations) == ("fail", 1)
    above = r["generation_above_uigf"]
    assert (above.status, above.n_violations) == ("warn", 1)
    assert above.details["examples"][0]["above_uigf"] == 1


def test_schema_rejects_inconsistent_rows(db: Engine, tmp_path: Path) -> None:
    _ingest(db, tmp_path)
    bad = [
        # wind without UIGF; dispatchable with a load; off the 5-minute grid; unknown kind
        "('VIC1', '2022-03-01 00:05+00', 'wind', 1, NULL, NULL, NULL)",
        "('VIC1', '2022-03-01 00:05+00', 'dispatchable', 1, NULL, NULL, 2)",
        "('VIC1', '2022-03-01 00:06+00', 'solar', 1, 1, NULL, NULL)",
        "('VIC1', '2022-03-01 00:05+00', 'hydro', 1, NULL, NULL, NULL)",
    ]
    for values in bad:
        with pytest.raises(IntegrityError):
            _sql(
                db,
                "INSERT INTO generation_intervals (region_id, interval_end, kind, cleared_mw, "
                f"uigf_mw, available_mw, load_mw) VALUES {values}",
            )


def _anomaly_run(db: Engine) -> int:
    with db.begin() as conn:
        ensure_regions(conn, load_config())
        return int(
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs (batch_id, model_name, model_kind, target, region_id)
                    VALUES (gen_random_uuid(), 'anomaly_detectors', 'anomaly', 'market', 'VIC1')
                    RETURNING run_id
                    """
                )
            ).scalar_one()
        )


def _insert_event(db: Engine, run: int, **overrides: str) -> None:
    values = {
        "first": "'2024-02-13 05:00+00'",
        "peak": "'2024-02-13 05:10+00'",
        "last": "'2024-02-13 05:30+00'",
        "tier": "'severe'",
        "detectors": "ARRAY['price_level', 'price_line_5000']",
        "min_p": "0.00002",
        "categories": "ARRAY['unusual price', 'high price']",
    } | overrides
    _sql(
        db,
        f"""
        INSERT INTO anomalies (run_id, region_id, split, first_interval_end, last_interval_end,
                               peak_interval_end, tier, detectors, min_p, categories, summary)
        VALUES ({run}, 'VIC1', 'dev', {values["first"]}, {values["last"]}, {values["peak"]},
                {values["tier"]}, {values["detectors"]}, {values["min_p"]},
                {values["categories"]}, 'Price at 16:10 was $16,600/MWh.')
        """,
    )


def test_anomalies_table_and_anomaly_runs(db: Engine) -> None:
    run = _anomaly_run(db)
    _insert_event(db, run)
    _insert_event(
        db,
        run,
        first="'2024-02-14 05:00+00'",
        peak="'2024-02-14 05:00+00'",
        last="'2024-02-14 05:00+00'",
        tier="'notable'",
        detectors="ARRAY['price_line_300']",
        min_p="NULL",
        categories="ARRAY['high price']",
    )
    assert _scalar(db, "SELECT array_agg(lines_only ORDER BY event_id)::text FROM anomalies") == (
        "{f,t}"
    )
    for bad in (
        {"peak": "'2024-02-13 04:55+00'"},  # peak before the start
        {"detectors": "ARRAY['price_spike']"},  # unknown detector
        {"categories": "ARRAY[]::text[]"},  # no category
        {"min_p": "0"},  # p-values are positive
        {"tier": "'minor'"},
    ):
        with pytest.raises(IntegrityError):
            _insert_event(db, run, first="'2024-03-01 00:00+00'", **bad)
    _sql(db, f"DELETE FROM model_runs WHERE run_id = {run}")  # events go with their run
    assert _scalar(db, "SELECT count(*) FROM anomalies") == 0


def _daily_archives(tmp_path: Path, rows: list[dict[str, str]]) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    for i, day in enumerate([*month_days(FEB), MAR1]):
        sl = slice(2 * 288 * i, 2 * 288 * (i + 1))
        name = f"PUBLIC_DISPATCHIS_{day:%Y%m%d}.zip"
        path = write_daily_archive(
            tmp_path / "daily" / name, rows[sl], [], regionsum_cols=GENERATION_V9_COLS
        )
        blobs[name] = path.read_bytes()
    return blobs


def test_daily_days_load_after_an_exact_match(db: Engine, tmp_path: Path) -> None:
    rows = generation_rows(START, N_FEB + 288)
    cfg, client = _client(tmp_path, "9", _daily_archives(tmp_path, rows))
    assert ingest_generation_range(db, cfg, FEB, FEB, ["VIC1"], client=client)[0].status == (
        "success"
    )
    result = ingest_generation_daily(db, cfg, [MAR1], FEB, ["VIC1"], client=client)
    check = result.check
    assert check.passed
    assert (check.compared, check.only_daily, check.only_loaded) == (4 * N_FEB, 0, 0)
    assert result.loaded == {"2022-03-01": 4 * 288}
    assert (
        _scalar(
            db,
            """SELECT count(*) FROM generation_intervals g JOIN ingestion_runs r
           ON r.run_id = g.ingestion_run_id WHERE r.source = 'nemweb_daily'""",
        )
        == 4 * 288
    )
    saved = json.loads(
        (tmp_path / "interim" / "nemweb_daily" / "generation_check_2022-02.json").read_text()
    )
    assert saved["check"]["passed"] and saved["loaded"] == {"2022-03-01": 1152}


def test_daily_mismatch_loads_nothing(db: Engine, tmp_path: Path) -> None:
    rows = generation_rows(START, N_FEB + 288)
    altered = [dict(r) for r in rows]
    altered[5 * 288 * 2 + 1]["SS_SOLAR_UIGF"] = "123.4"  # VIC1, 6 Feb, first interval
    cfg, client = _client(tmp_path, "9", _daily_archives(tmp_path, altered))
    ingest_generation_range(db, cfg, FEB, FEB, ["VIC1"], client=client)
    with pytest.raises(GenerationCheckFailed) as err:
        ingest_generation_daily(db, cfg, [MAR1], FEB, ["VIC1"], client=client)
    assert err.value.check.differing == {
        "cleared_mw": 0, "uigf_mw": 1, "available_mw": 0, "load_mw": 0,
    }  # fmt: skip
    assert _scalar(db, "SELECT count(*) FROM generation_intervals WHERE interval_end > "
                       "'2022-02-28 14:00+00'") == 0  # fmt: skip
