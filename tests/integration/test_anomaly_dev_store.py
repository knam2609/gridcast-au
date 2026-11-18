"""Storage of an anomaly run (M4 step 4): files, the run row, its events, and the run-once lock."""

import datetime as dt
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.anomalies.dev import DevOutputs, existing_runs, interval_frame, store_dev
from gridcast.anomalies.evidence_inputs import EvidenceRun
from gridcast.anomalies.freeze import FreezeState
from gridcast.anomalies.injection import run_injections
from gridcast.anomalies.run import detect
from gridcast.config import load_config
from gridcast.database.load import ensure_regions
from gridcast.explainability.treeshap import empty_gate
from gridcast.timeutils import INTERVAL, NEM_TZ
from tests.anomaly_fixtures import base_inputs, small_config, split

pytestmark = pytest.mark.integration


def _outputs() -> DevOutputs:
    series, band = base_inputs()
    cfg = small_config()
    # A bounded split: the run row needs an end, and the table accepts only dev/holdout.
    last_day = (series.ends[-1] - INTERVAL).tz_convert(NEM_TZ).date()
    sp = replace(split(), end_day=last_day + dt.timedelta(days=1))
    det = detect(series, band, sp, cfg)
    inj = run_injections(series, det.run, sp, cfg, n_starts=20)
    evidence = [{"event": {"index": i}} for i in range(len(det.events))]
    summaries = [f"Event {i}." for i in range(len(det.events))]
    return DevOutputs(
        split=sp,
        series=series,
        detection=det,
        injections=inj,
        cuts=[],
        forest_leaks=[],
        evidence=EvidenceRun(None, empty_gate(1e-6), evidence, summaries),  # type: ignore[arg-type]
        band_issue_times=len(band.issue_times),
        band_valid_pairs=int(band.valid.sum()),
        checked_actuals=0,
        seconds={"detect": 1.0},
    )


STATE = FreezeState("feedc0de", "feedc0de", "abc1234", "abc1234ff", False, ())


def test_store_writes_files_run_and_events(db: Engine, tmp_path: Path) -> None:
    cfg = load_config()
    cfg.anomalies = cfg.anomalies.model_copy(update={"output_dir": tmp_path})
    with db.begin() as conn:
        ensure_regions(conn, cfg)
    out = _outputs()
    assert out.detection.events, "the synthetic run must have events to store"

    run_id, batch_id, directory = store_dev(db, cfg, "VIC1", out, STATE)

    frame = pd.read_parquet(directory / "intervals.parquet")
    assert len(frame) == len(out.series)
    expected = interval_frame(out.detection, out.split)
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    assert len(pd.read_parquet(directory / "injections.parquet")) == len(out.injections)
    forests = pd.read_parquet(directory / "forests.parquet")
    assert len(forests) == len(out.detection.run.state.models)
    assert (directory / "cuts.parquet").exists()

    with db.connect() as conn:
        run = conn.execute(
            text("SELECT model_kind, target, config, git_sha FROM model_runs WHERE run_id = :r"),
            {"r": run_id},
        ).one()
        events = conn.execute(
            text(
                "SELECT first_interval_end, peak_interval_end, detectors, min_p, summary, "
                "evidence FROM anomalies WHERE run_id = :r ORDER BY first_interval_end"
            ),
            {"r": run_id},
        ).all()
    assert (run.model_kind, run.target, run.git_sha) == ("anomaly", "market", "abc1234ff")
    assert run.config["anomaly_config_hash"] == "feedc0de" and run.config["split"] == "dev"
    assert len(events) == len(out.detection.events)
    ends = out.series.ends
    for row, e, i in zip(events, out.detection.events, range(len(events)), strict=True):
        assert pd.Timestamp(row.first_interval_end) == ends[e.first]
        assert pd.Timestamp(row.peak_interval_end) == ends[e.peak]
        assert tuple(row.detectors) == e.detectors
        assert row.min_p == e.min_p or (row.min_p is None and e.min_p is None)
        assert row.summary == f"Event {i}." and row.evidence == {"event": {"index": i}}

    assert existing_runs(db, "VIC1", "dev", "feedc0de") == [(run_id, batch_id)]
    assert existing_runs(db, "VIC1", "holdout", "feedc0de") == []
    assert existing_runs(db, "VIC1", "dev", "other") == []


def test_a_failed_insert_leaves_no_files(db: Engine, tmp_path: Path) -> None:
    cfg = load_config()
    cfg.anomalies = cfg.anomalies.model_copy(update={"output_dir": tmp_path})
    # No regions row: the run insert violates its foreign key.
    with pytest.raises(Exception, match=r"foreign key|violates"):
        store_dev(db, cfg, "VIC1", _outputs(), STATE)
    assert list(tmp_path.iterdir()) == []
    with db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM model_runs")).scalar_one() == 0


def test_the_development_band_matches_run_2() -> None:
    """On the stored data (main data root): the development band is run 85's QR band with
    run 2's mask, and the holdout QR batch equals it on their overlap (development issue
    times only are compared)."""
    from gridcast.anomalies.inputs import check_qr_overlap, read_band, split_band

    cfg = load_config()
    root = cfg.data_path(cfg.intervals.output_dir)
    if not (root / cfg.anomalies.residual.dev_qr_batch).exists():
        pytest.skip("the stored QR batches are not available")
    band, actual = split_band(cfg, "dev")
    ref, ref_actual = read_band(
        cfg, cfg.anomalies.residual.dev_qr_batch, cfg.intervals.source_batch
    )
    assert band.issue_times.equals(ref.issue_times)
    assert np.array_equal(band.valid, ref.valid)
    assert np.array_equal(band.lower, ref.lower, equal_nan=True)
    assert np.array_equal(actual, ref_actual, equal_nan=True)
    assert check_qr_overlap(cfg) == 2928
