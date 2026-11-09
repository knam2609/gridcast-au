import datetime as dt

import pytest
from sqlalchemy import Engine, text

from gridcast.config import load_config, load_market_limits
from gridcast.database.load import ensure_regions, start_ingestion_run, upsert_market_intervals
from gridcast.validation.checks import MARKET_CHECKS, DQResult, run_checks
from tests.integration.factories import clean_month

pytestmark = pytest.mark.integration

OCT = dt.date(2021, 10, 1)


@pytest.fixture
def loaded(db: Engine) -> Engine:
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", OCT)
        upsert_market_intervals(conn, clean_month(), run)
        conn.execute(
            text(
                "UPDATE ingestion_runs SET status='success', duplicates_dropped=0 WHERE run_id=:r"
            ),
            {"r": run},
        )
    return db


def _run(engine: Engine, stale_after_days: int = 100_000) -> dict[str, DQResult]:
    cfg = load_config()
    cfg.sources.mmsdm.stale_after_days = stale_after_days
    results = run_checks(
        engine, cfg, load_market_limits(cfg), ["VIC1"], OCT, OCT, checks=MARKET_CHECKS
    )
    return {r.check_name: r for r in results}


def _sql(engine: Engine, sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql))


def test_clean_data_passes_every_check(loaded: Engine) -> None:
    results = _run(loaded)
    assert set(results) == {c.name for c in MARKET_CHECKS}
    assert {name: r.status for name, r in results.items() if r.status != "pass"} == {}


def test_results_are_persisted(loaded: Engine) -> None:
    _run(loaded)
    with loaded.connect() as conn:
        n = conn.execute(text("SELECT count(DISTINCT check_name) FROM dq_results")).scalar_one()
    assert n == len(MARKET_CHECKS)


def test_missing_intervals_reported_as_one_gap(loaded: Engine) -> None:
    _sql(
        loaded,
        "DELETE FROM market_intervals WHERE interval_end IN "
        "('2021-10-05 00:00+00','2021-10-05 00:05+00','2021-10-05 00:10+00')",
    )
    r = _run(loaded)
    assert r["missing_intervals"].status == "fail"
    assert r["missing_intervals"].n_violations == 3
    assert r["missing_intervals"].details["n_result_rows"] == 1  # one contiguous gap
    assert r["rows_per_month"].status == "fail"
    assert r["rows_per_month"].details["examples"][0]["shortfall"] == 3


def test_negative_demand_flagged(loaded: Engine) -> None:
    _sql(
        loaded,
        "UPDATE market_intervals SET total_demand_mw = -10 WHERE interval_end = "
        "'2021-10-02 00:00+00'",
    )
    assert _run(loaded)["demand_out_of_range"].n_violations == 1


def test_price_above_financial_year_cap_flagged(loaded: Engine) -> None:
    # FY2021-22 cap is 15,100 $/MWh: 15,200 is impossible in Oct 2021.
    _sql(
        loaded,
        "UPDATE market_intervals SET rrp_aud_mwh = 15200 WHERE interval_end = "
        "'2021-10-02 00:00+00'",
    )
    r = _run(loaded)["price_out_of_bounds"]
    assert r.status == "fail"
    assert r.details["examples"][0]["reason"] == "above_cap"


def test_null_price_flagged(loaded: Engine) -> None:
    _sql(
        loaded,
        "UPDATE market_intervals SET rrp_aud_mwh = NULL WHERE interval_end = '2021-10-02 00:00+00'",
    )
    assert _run(loaded)["null_values"].n_violations == 1


def test_implausible_timestamps_flagged(loaded: Engine) -> None:
    _sql(
        loaded,
        "INSERT INTO market_intervals (region_id, interval_start, interval_end, total_demand_mw) "
        "VALUES ('VIC1', '2021-10-03 00:02+00', '2021-10-03 00:07+00', 4000), "
        "       ('VIC1', '2030-01-01 00:00+00', '2030-01-01 00:05+00', 4000)",
    )
    r = _run(loaded)["implausible_timestamps"]
    assert r.status == "fail"
    assert sorted(e["reason"] for e in r.details["examples"]) == ["in_future", "off_grid"]


def test_stale_feed(loaded: Engine) -> None:
    assert _run(loaded, stale_after_days=30)["stale_feed"].status == "fail"
    assert _run(loaded, stale_after_days=100_000)["stale_feed"].status == "pass"


def test_market_flags_grouped_into_episodes(loaded: Engine) -> None:
    _sql(
        loaded,
        "UPDATE market_intervals SET apc_flag_bits = 1 WHERE interval_end BETWEEN "
        "'2021-10-10 00:00+00' AND '2021-10-10 00:25+00'",
    )
    r = _run(loaded)
    assert r["market_flags"].status == "warn"
    assert r["market_flags"].n_violations == 6
    assert r["market_flags"].details["n_result_rows"] == 1
    assert r["price_adjustments"].details["examples"][0]["bit_value"] == 1


def test_price_adjustment_bits_decoded(loaded: Engine) -> None:
    # 20 = 16 (IRLF scaled) + 4 (market price cap/floor binding): not administered pricing.
    _sql(
        loaded,
        "UPDATE market_intervals SET apc_flag_bits = 20 WHERE interval_end = '2021-10-02 00:00+00'",
    )
    r = _run(loaded)
    assert r["market_flags"].status == "pass"
    assert sorted(e["bit_value"] for e in r["price_adjustments"].details["examples"]) == [4, 16]


def test_not_firm_prices_warn(loaded: Engine) -> None:
    _sql(
        loaded,
        "UPDATE market_intervals SET price_status = 'NOT FIRM' WHERE interval_end = "
        "'2021-10-02 00:00+00'",
    )
    assert _run(loaded)["price_not_firm"].status == "warn"


def test_failed_ingestion_and_source_duplicates(loaded: Engine) -> None:
    _sql(loaded, "UPDATE ingestion_runs SET duplicates_dropped = 5")
    _sql(
        loaded,
        "INSERT INTO ingestion_runs (source, dataset, period_month, status, error) "
        "VALUES ('mmsdm', 'market_intervals', '2021-10-01', 'failed', 'boom')",
    )
    r = _run(loaded)
    assert r["ingestion_failures"].status == "fail"
    # Duplicates are read from the latest *successful* load of each month.
    assert r["source_duplicates"].status == "warn"
    assert r["source_duplicates"].n_violations == 5
